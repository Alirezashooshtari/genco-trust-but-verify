"""Ablation of the unresolved-case warning: does GENCO add information beyond the outage topology?

Every variant uses the warning's own training cases (from the fine-tuning set), its threshold rule (95 % recall on the
validation cases) and the same test sets (the 870 frozen-test cases and the 120 storms at the actual load). Feature sets:

    A    outage topology and load factor (no power flow): lines out, buses touched, outage spread in hops, edge
         betweenness of the lines out, lines at 345 kV and above, load factor; logistic regression and boosted trees
    B    GENCO's 9 scalar outputs (voltages, residuals, flagged lines and buses)
    C    GENCO's embeddings (mean and max over buses of the input of its bus read-out layer)
    BC   B and C together: the reported warning
    ABC  everything

Needs the feature matrices published with the fine-tuned model (data/ in the Hugging Face repository).

    python scripts/warning_ablation.py --features-dir <folder with the two .npz files>

Writes results/analysis/warning_ablation.{csv,json} and results/analysis/warning_scores.csv (the score of every test case).
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SETS = {"train": "head training (T1)", "val": "head validation (T1)", "F1": "F1", "K1": "K1 storms at the actual load"}
TOPOLOGY_FEATURES = [
    "lines out",
    "buses touched",
    "outage spread, max hops",
    "outage spread, median hops",
    "betweenness of lines out, sum",
    "betweenness of lines out, max",
    "lines out at 345 kV and above",
    "load factor",
]


class Topology:
    """Topology features of an outage set on the Texas grid."""

    def __init__(self, layout):
        self.f, self.t = np.array(layout["f"]), np.array(layout["t"])
        kv = np.array(layout["kv"], float)
        nb = int(max(self.f.max(), self.t.max()) + 1)
        graph = nx.Graph()
        graph.add_edges_from(zip(self.f.tolist(), self.t.tolist()))
        eb = nx.edge_betweenness_centrality(graph, k=400, seed=0, normalized=True)
        self.betweenness = np.array(
            [eb.get((int(x), int(y)), eb.get((int(y), int(x)), 0.0)) for x, y in zip(self.f, self.t)]
        )
        self.adjacency = csr_matrix((np.ones(len(self.f)), (self.f, self.t)), shape=(nb, nb))
        self.line_kv = np.maximum(kv[self.f], kv[self.t])

    def features(self, removed, load_factor):
        r = np.asarray(removed, int)
        if len(r) == 0:
            return [0, 0, 0.0, 0.0, 0.0, 0.0, 0, float(load_factor)]
        buses = np.unique(np.r_[self.f[r], self.t[r]])
        hops = dijkstra(self.adjacency, directed=False, unweighted=True, indices=buses)[:, buses]
        median_hops = float(np.median(hops[np.triu_indices(len(buses), 1)])) if len(buses) > 1 else 0.0
        return [
            len(r),
            len(buses),
            float(hops.max()),
            median_hops,
            float(self.betweenness[r].sum()),
            float(self.betweenness[r].max()),
            int((self.line_kv[r] >= 345).sum()),
            float(load_factor),
        ]


class Cases:
    """Removed lines, load factor and operating point of every case id: 'T1:<cid>' (fine-tuning set), 'F1:<sid>' (frozen test), 'K1:<sid>@<lam>' (storms)."""

    def __init__(self):
        self.t1 = pd.read_csv(RES / "finetune/train_cases.csv").set_index("cid")
        self.f1 = pd.read_csv(RES / "frozen_test/cases.csv").set_index("sid")
        self.k1 = pd.read_csv(RES / "storms/cases.csv").set_index("sid")

    def outage(self, case_id):
        kind, rest = case_id.split(":")
        if kind == "T1":
            r = self.t1.loc[int(rest)]
            return json.loads(r.removed_lines), r.lam
        if kind == "F1":
            return json.loads(self.f1.loc[int(rest)].removed_lines), 1.0
        sid, lam = rest.split("@")
        return json.loads(self.k1.loc[int(sid)].removed_lines), float(lam)

    def cluster(self, case_id):
        kind, rest = case_id.split(":")
        if kind == "T1":
            return int(self.t1.loc[int(rest)].base)
        if kind == "F1":
            return int(self.f1.loc[int(rest)].base)
        return int(rest.split("@")[0])


def cluster_bootstrap(stat, clusters, n_boot, seed=0):
    """Percentile interval of stat(idx) over a bootstrap of whole clusters (operating points or storms)."""
    rng = np.random.default_rng(seed)
    keys = np.unique(clusters)
    members = {k: np.nonzero(clusters == k)[0] for k in keys}
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.choice(keys, len(keys))])
        v = stat(idx)
        if v is not None:
            values.append(v)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def auc_or_none(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else None


def classifier(kind):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    topology = Topology(json.loads((RES / "grid_layout.json").read_text()))
    cases = Cases()
    topo_cache, rows, scores = {}, [], {}

    for model in ("released", "fine_tuned"):
        folder = Path(args.features_dir)
        path = folder / f"features_{model}.npz"
        if not path.exists():  # the name used in the Hugging Face repository
            path = folder / f"features_{model.replace('_', '')}.npz"
        Z = np.load(path)
        data = {k: (Z[f"{v}__X"], Z[f"{v}__y"].astype(int), Z[f"{v}__ids"].astype(str)) for k, v in SETS.items()}
        for k, (_, _, ids) in data.items():
            if k not in topo_cache:
                topo_cache[k] = np.array([topology.features(*cases.outage(i)) for i in ids])
        feature_sets = {
            "A topology": lambda k: topo_cache[k],
            "B GENCO outputs": lambda k: data[k][0][:, :9],
            "C GENCO embeddings": lambda k: data[k][0][:, 9:],
            "BC outputs + embeddings (reported)": lambda k: data[k][0],
            "ABC all": lambda k: np.c_[topo_cache[k], data[k][0]],
        }
        variants = [(name, fn, "logistic") for name, fn in feature_sets.items()]
        if model == "released":
            variants.append(("A topology", feature_sets["A topology"], "boosted trees"))
        else:
            variants = [v for v in variants if not v[0].startswith("A topology")]

        for name, fn, kind in variants:
            clf = classifier(kind).fit(fn("train"), data["train"][1])
            p_val = clf.predict_proba(fn("val"))[:, 1]
            threshold = float(np.quantile(p_val[data["val"][1] == 1], 0.05))
            for test in ("F1", "K1"):
                X, y, ids = fn(test), data[test][1], data[test][2]
                p = clf.predict_proba(X)[:, 1]
                flag = p >= threshold
                clusters = np.array([cases.cluster(i) for i in ids])
                scores[(model, name, kind, test)] = (p, y, clusters, ids)
                rows.append(
                    {
                        "genco_model": "none (no GENCO)" if name.startswith("A topology") else model,
                        "features": name,
                        "classifier": kind,
                        "test": SETS[test],
                        "cases": int(len(y)),
                        "unresolved": int(y.sum()),
                        "auroc": float(roc_auc_score(y, p)),
                        "auroc_ci95": cluster_bootstrap(lambda i: auc_or_none(y[i], p[i]), clusters, args.bootstrap),
                        "recall": float(flag[y == 1].mean()),
                        "false_alarm_rate": float(flag[y == 0].mean()),
                        "missed": int((~flag[y == 1]).sum()),
                        "n_features": int(X.shape[1]),
                    }
                )

    def paired_difference(key1, key0):
        p1, y, clusters, _ = scores[key1]
        p0 = scores[key0][0]

        def stat(i):
            a, b = auc_or_none(y[i], p1[i]), auc_or_none(y[i], p0[i])
            return None if a is None else a - b

        return {
            "difference": float(roc_auc_score(y, p1) - roc_auc_score(y, p0)),
            "ci95": cluster_bootstrap(stat, clusters, args.bootstrap),
        }

    pairs = {}
    for test in ("F1", "K1"):
        baseline = ("released", "A topology", "boosted trees", test)
        for model in ("released", "fine_tuned"):
            reported = (model, "BC outputs + embeddings (reported)", "logistic", test)
            pairs[f"{test}: {model} GENCO outputs + embeddings vs topology (boosted trees)"] = paired_difference(
                reported, baseline
            )
            pairs[f"{test}: {model} embeddings vs GENCO outputs alone"] = paired_difference(
                (model, "C GENCO embeddings", "logistic", test), (model, "B GENCO outputs", "logistic", test)
            )
            pairs[f"{test}: {model} all vs outputs + embeddings (topology added)"] = paired_difference(
                (model, "ABC all", "logistic", test), reported
            )

    table = pd.DataFrame(rows)
    table.to_csv(RES / "analysis/warning_ablation.csv", index=False)
    (RES / "analysis/warning_ablation.json").write_text(
        json.dumps({"topology_features": TOPOLOGY_FEATURES, "rows": rows, "paired_auroc_differences": pairs}, indent=1)
    )
    per_case = [
        pd.DataFrame(
            {
                "genco_model": m,
                "features": n,
                "classifier": k,
                "test": SETS[t],
                "case": ids,
                "base": cl,
                "label_unresolved": y,
                "score": p,
            }
        )
        for (m, n, k, t), (p, y, cl, ids) in scores.items()
        if n.startswith(("A topology", "BC"))
    ]
    pd.concat(per_case).to_csv(RES / "analysis/warning_scores.csv", index=False, float_format="%.6g")

    for k, v in pairs.items():
        print(f"{k:78s} {v['difference']:+.3f} ({v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f})")
    print(
        table[["genco_model", "features", "classifier", "test", "auroc", "recall", "false_alarm_rate"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

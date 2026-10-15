"""Routing between the released and the fine-tuned GENCO by outage spread (a post-hoc analysis).

Whether an outage is clustered is known from the outage itself, before any model runs. The rule sends an outage to the
fine-tuned model when it removes at least MIN_LINES lines and its spread (the largest hop distance between buses touched
by the removed lines) is at most a threshold; everything else goes to the released model. MIN_LINES is the smallest
outage in the fine-tuning set. The threshold is the one that classifies the fine-tuning training cases best (clustered
or random). The rule is then applied unchanged to the frozen test and the held-out footprint cases of every fine-tuning
run, and its routing rate on random N-2, N-10 and N-20 outages is estimated by drawing outages the way the released test
sets do (lines drawn uniformly at random). The first run did not store the lines of its footprint cases, so for it only
the frozen test is routed.

    python scripts/routing_analysis.py      # writes results/analysis/routing.json and routing_cases.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CLUSTERED = ("nearest", "corridor")
MIN_LINES = 5
RUNS = ("seed0", "seed1", "as_run")  # seed0 is the published model


def build_grid():
    layout = json.loads((RES / "grid_layout.json").read_text())
    f, t = np.array(layout["f"]), np.array(layout["t"])
    nb = int(max(f.max(), t.max()) + 1)
    return f, t, csr_matrix((np.ones(len(f)), (f, t)), shape=(nb, nb))


F, T, ADJ = build_grid()


def spread(removed):
    lines = np.asarray(removed, int)
    if len(lines) < 2:
        return 0.0
    buses = np.unique(np.r_[F[lines], T[lines]])
    hops = dijkstra(ADJ, directed=False, unweighted=True, indices=buses)[:, buses]
    return float(hops.max())


def paired_mean_ci(diff, clusters, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = np.unique(clusters)
    members = {k: np.nonzero(clusters == k)[0] for k in keys}
    boots = [diff[np.concatenate([members[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def route_run(run, route):
    folder = RES / f"evaluate_finetuned_{run}"
    removed = pd.read_csv(RES / "frozen_test/cases.csv").set_index("sid").removed_lines.to_dict()
    footprint = folder / "footprint_cases.csv"
    sets = ["F1"]
    if footprint.exists():
        removed |= pd.read_csv(footprint).set_index("sid").removed_lines.to_dict()
        sets.append("footprint")
    ev = pd.read_csv(folder / "cases.csv")
    ev = ev[ev.set.isin(sets) & ev.ref_status.str.startswith("verified")].copy()
    lines = [json.loads(removed[s]) for s in ev.sid]
    ev["spread"] = [spread(r) for r in lines]
    ev["route"] = [route(r) for r in lines]
    ev["routed_missed"] = np.where(ev.route == "fine_tuned", ev.fine_tuned_missed, ev.released_missed)
    ev["routed_worst_err"] = np.where(ev.route == "fine_tuned", ev.fine_tuned_worst_err, ev.released_worst_err)

    out = {}
    groups = [
        ("random", (ev.set == "F1") & ~ev.family.str.startswith(CLUSTERED)),
        ("clustered", (ev.set == "F1") & ev.family.str.startswith(CLUSTERED)),
        ("all", ev.set == "F1"),
    ]
    if "footprint" in sets:
        groups.append(("footprint", ev.set == "footprint"))
    for name, mask in groups:
        g = ev[mask]
        row = {"cases": int(len(g)), "routed_to_fine_tuned": int((g.route == "fine_tuned").sum())}
        for metric in ("missed", "worst_err"):
            for model in ("released", "fine_tuned", "routed"):
                row[f"{model}_{metric}_mean"] = float(g[f"{model}_{metric}"].mean())
            d = (g[f"routed_{metric}"] - g[f"released_{metric}"]).to_numpy()
            row[f"routed_minus_released_{metric}"] = {
                "mean": float(d.mean()),
                "ci95": paired_mean_ci(d, g.base.to_numpy()),
            }
        out[name] = row
    return out, ev.assign(run=run)


def main():
    train = pd.read_csv(RES / "finetune_seed0/train_cases.csv")
    train = train[(train.split == "train") & (train.k > 2)]
    train["spread"] = [spread(json.loads(r)) for r in train.removed_lines]
    clustered = train.family.str.startswith(CLUSTERED).to_numpy()
    candidates = np.arange(1, 31)
    accuracy = [float(((train.spread.to_numpy() <= c) == clustered).mean()) for c in candidates]
    threshold = float(candidates[int(np.argmax(accuracy))])

    def route(removed):
        return "fine_tuned" if len(removed) >= MIN_LINES and spread(removed) <= threshold else "released"

    out = {"min_lines": MIN_LINES, "threshold_hops": threshold, "training_accuracy": max(accuracy), "runs": {}}
    per_case = []
    for run in RUNS:
        out["runs"][run], cases = route_run(run, route)
        per_case.append(cases)

    rng = np.random.default_rng(0)
    out["random_outages_routed_to_fine_tuned"] = {}
    for k in (2, 10, 20):
        draws = [route(rng.choice(len(F), k, replace=False)) for _ in range(2000)]
        out["random_outages_routed_to_fine_tuned"][f"N-{k}"] = float(np.mean(np.array(draws) == "fine_tuned"))

    (RES / "analysis/routing.json").write_text(json.dumps(out, indent=1) + "\n")
    columns = ["run", "set", "sid", "base", "family", "spread", "route"]
    columns += [f"{m}_{x}" for x in ("missed", "worst_err") for m in ("released", "fine_tuned", "routed")]
    pd.concat(per_case)[columns].to_csv(RES / "analysis/routing_cases.csv", index=False, float_format="%.6g")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

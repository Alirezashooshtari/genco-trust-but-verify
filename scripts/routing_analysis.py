"""Routing between the released and the fine-tuned GENCO by outage spread (a post-hoc analysis).

Whether an outage is clustered is known from the outage itself, before any model runs. The rule sends an outage to the
fine-tuned model when it removes at least MIN_LINES lines and its spread (the largest hop distance between buses touched
by the removed lines) is at most a threshold; everything else goes to the released model. MIN_LINES is the smallest
outage in the fine-tuning set. The threshold is the one that classifies the fine-tuning training cases best (clustered
or random). The rule is then applied unchanged to the frozen test, and its routing rate on random N-2, N-10 and N-20
outages is estimated by drawing outages the way the released test sets do (lines drawn uniformly at random).

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


def main():
    train = pd.read_csv(RES / "finetune/train_cases.csv")
    train = train[(train.split == "train") & (train.k > 2)]
    train["spread"] = [spread(json.loads(r)) for r in train.removed_lines]
    clustered = train.family.str.startswith(CLUSTERED).to_numpy()
    candidates = np.arange(1, 31)
    accuracy = [float(((train.spread.to_numpy() <= c) == clustered).mean()) for c in candidates]
    threshold = float(candidates[int(np.argmax(accuracy))])

    def route(removed):
        return "fine_tuned" if len(removed) >= MIN_LINES and spread(removed) <= threshold else "released"

    cases = pd.read_csv(RES / "frozen_test/cases.csv").set_index("sid")
    ev = pd.read_csv(RES / "evaluate_finetuned/cases.csv")
    ev = ev[(ev.set == "F1") & ev.ref_status.str.startswith("verified")].copy()
    ev["spread"] = [spread(json.loads(cases.loc[s, "removed_lines"])) for s in ev.sid]
    ev["route"] = [route(json.loads(cases.loc[s, "removed_lines"])) for s in ev.sid]
    ev["routed_missed"] = np.where(ev.route == "fine_tuned", ev.fine_tuned_missed, ev.released_missed)
    ev["routed_worst_err"] = np.where(ev.route == "fine_tuned", ev.fine_tuned_worst_err, ev.released_worst_err)

    out = {"min_lines": MIN_LINES, "threshold_hops": threshold, "training_accuracy": max(accuracy), "frozen_test": {}}
    for name, mask in [
        ("random", ~ev.family.str.startswith(CLUSTERED)),
        ("clustered", ev.family.str.startswith(CLUSTERED)),
        ("all", ev.family.notna()),
    ]:
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
        out["frozen_test"][name] = row

    rng = np.random.default_rng(0)
    out["random_outages_routed_to_fine_tuned"] = {}
    for k in (2, 10, 20):
        draws = [route(rng.choice(len(F), k, replace=False)) for _ in range(2000)]
        out["random_outages_routed_to_fine_tuned"][f"N-{k}"] = float(np.mean(np.array(draws) == "fine_tuned"))

    (RES / "analysis/routing.json").write_text(json.dumps(out, indent=1) + "\n")
    ev[
        [
            "sid",
            "base",
            "family",
            "spread",
            "route",
            "released_missed",
            "fine_tuned_missed",
            "routed_missed",
            "released_worst_err",
            "fine_tuned_worst_err",
            "routed_worst_err",
        ]
    ].to_csv(RES / "analysis/routing_cases.csv", index=False, float_format="%.6g")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

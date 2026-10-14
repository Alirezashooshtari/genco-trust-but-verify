"""Recomputes every number quoted in the write-up from the files in results/.

python scripts/headline_numbers.py        # prints the table and writes results/headline_numbers.csv
"""

import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
rows = []
WORKFLOW_NAMES = {"classical": "classical multi-start", "genco_verify": "GENCO + verification"}
GROUP_NAMES = {
    "clustered (F1)": "clustered outages",
    "footprint (held out)": "held-out construction",
    "uniform (F1)": "random outages",
}
TEST_NAMES = {
    "F1": "frozen test",
    "K1": "storms",
    "footprint": "held-out construction",
    "K1 storms at the actual load": "storms at the actual load",
    "K1 points within 5 % of the fold": "storms within 5 % of the fold",
}


def claim(section, text, value, source):
    rows.append({"section": section, "claim": text, "value": value, "source": source})


def pct(a, b):
    return f"{a} of {b} ({100 * a / b:.0f} %)"


# ------------------------------------------------------------------ where GENCO fails
k1 = pd.read_csv(RES / "storms/sweep.csv")
at = k1[k1.at_actual_load & ~k1.rescued]
solved, unresolved = at[at.solvable_at_1], at[~at.solvable_at_1]
src = "storms/sweep.csv"
claim("fail", "clustered N-20 storms", len(at), src)
claim("fail", "storms with no converging start at the actual load", len(unresolved), src)
claim(
    "fail",
    "GENCO lowest voltage, median: solved / unresolved storms (p.u.)",
    f"{solved.genco_min_vm.median():.3f} / {unresolved.genco_min_vm.median():.3f}",
    src,
)
claim(
    "fail",
    "Newton from GENCO's state converged on an unresolved storm",
    int(unresolved.klu_from_genco_converged.sum()),
    src,
)

storms = pd.read_csv(RES / "clustered_outages/storm_cases.csv")
med = storms.groupby(["T", "mode"])["GENCO worst Vm error"].median()
for k in (10, 20):
    claim(
        "fail",
        f"N-{k}: median worst-bus error, random / clustered outages (p.u.)",
        f"{med[k, 'uniform']:.4f} / {med[k, 'clustered']:.4f}",
        "clustered_outages/storm_cases.csv",
    )
claim(
    "fail",
    "median worst-bus error, clustered / random outages: N-10, N-20",
    f"x{med[10, 'clustered'] / med[10, 'uniform']:.1f}, x{med[20, 'clustered'] / med[20, 'uniform']:.1f}",
    "clustered_outages/storm_cases.csv",
)

growth = pd.read_csv(RES / "load_response/load_growth_all_buses.csv")
claim(
    "why",
    "GENCO response to uniform load growth relative to AC (median slope)",
    f"{growth.slope_all_pq.median():.2f}",
    "load_response/load_growth_all_buses.csv",
)

# ------------------------------------------------------------------ the frozen test
wf = pd.read_csv(RES / "frozen_test/summary_by_workflow.csv", index_col=0)
alone = pd.read_csv(RES / "frozen_test/genco_alone_by_family.csv").set_index("family")
src = "frozen_test/summary_by_workflow.csv"
for w in ("classical", "genco_verify"):
    r = wf.loc[w]
    claim(
        "frozen",
        f"{WORKFLOW_NAMES[w]}: cases / verified / unresolved / false alarms",
        f"{r.cases:.0f} / {r.verified:.0f} / {r.unresolved:.0f} / {r.false_alarms:.0f}",
        src,
    )
    claim("frozen", f"{WORKFLOW_NAMES[w]}: median end-to-end ms per case", f"{r.ms_median:.1f}", src)
claim(
    "frozen",
    "GENCO + verification: verified cases whose decisions differ from the classical solution (two AC solutions)",
    f"{wf.loc['genco_verify', 'verified_with_a_missed_decision']:.0f}",
    src,
)
claim(
    "frozen",
    "GENCO alone: verified cases with at least one wrong decision",
    pct(int(alone.cases_with_a_miss.sum()), int(alone.cases.sum())),
    "frozen_test/genco_alone_by_family.csv",
)
for fams, label in [
    (["uniform-N10", "uniform-N20"], "random"),
    (["nearest-N10", "nearest-N20", "corridor-N20"], "clustered"),
]:
    a = alone.loc[fams]
    claim(
        "frozen",
        f"GENCO alone with the overload margin, {label} families: overloads missed",
        pct(int(a.missed_with_margin.sum()), int(a.ref_overloads.sum())),
        "frozen_test/genco_alone_by_family.csv",
    )
    claim(
        "frozen",
        f"GENCO alone, {label} families: voltage violations missed",
        pct(int(a.missed_vviol.sum()), int(a.ref_vviol.sum())),
        "frozen_test/genco_alone_by_family.csv",
    )
f1 = json.loads((RES / "frozen_test/summary.json").read_text())
claim(
    "frozen",
    "final set: cases / untouched operating points",
    f"{f1['final_set']['cases']} / {f1['final_set']['base_points']}",
    "frozen_test/summary.json",
)
timing = json.loads((RES / "frozen_test/timing.json").read_text())
claim(
    "frozen",
    "mean Newton iterations: classical / from GENCO's state",
    f"{timing['workflows']['classical']['iterations_mean']:.2f} / {timing['workflows']['genco_verify']['iterations_mean']:.2f}",
    "frozen_test/timing.json",
)
claim(
    "frozen",
    "GENCO inference, compiled float16, batch 16 (ms per case)",
    f"{timing['throughput_ms_per_case']['compiled_float16_batch16']:.1f}",
    "frozen_test/timing.json",
)
claim(
    "frozen",
    "GENCO path, median ms (graph + transfer + inference)",
    f"{timing['genco_path_components_ms_median']['genco_total_ms']:.1f}",
    "frozen_test/timing.json",
)
b1 = json.loads((RES / "branch_check/summary.json").read_text())
claim(
    "frozen",
    "branch check: verified cases with two distinct AC solutions among the starts",
    f"{b1['by_workflow']['genco_verify']['multiple_solutions']} of {b1['by_workflow']['genco_verify']['verified']}",
    "branch_check/summary.json",
)

c1 = pd.read_csv(RES / "overload_margin/scenario_level_results.csv")
c1 = c1[(c1.k == 2) & (c1.violation == "overloads") & (c1.rule == "scenario-level") & (c1.alpha == 0.01)].set_index(
    "method"
)
g, dc = c1.loc["GENCO"], c1.loc["DC (active power)"]
claim(
    "overload",
    "conformal overload margin (fraction of rating)",
    f"{g.margin:.4f}",
    "overload_margin/scenario_level_results.csv",
)
claim(
    "overload",
    "N-2 test: overloaded scenarios with nothing missed (95 % CI)",
    f"{g['of these, nothing missed (%)']:.1f} % ({g['95 % CI low (%)']:.1f} to {g['95 % CI high (%)']:.1f})",
    "overload_margin/scenario_level_results.csv",
)
claim(
    "overload",
    "false alarms per scenario, GENCO / DC at the same guarantee",
    f"{g['false alarms per scenario']:.2f} / {dc['false alarms per scenario']:.2f}",
    "overload_margin/scenario_level_results.csv",
)

# ------------------------------------------------------------------ fine-tuning
t1 = json.loads((RES / "finetune/summary.json").read_text())
claim(
    "fix",
    "training cases (regression targets) / minutes on one T4",
    f"{t1['cases']['regression_targets_train']} / {t1['training_minutes']:.0f}",
    "finetune/summary.json",
)
ep = pd.read_csv(RES / "evaluate_finetuned/endpoints_cases.csv").set_index(["group", "metric"])
for grp in ("clustered (F1)", "footprint (held out)", "uniform (F1)"):
    r = ep.loc[(grp, "missed")]
    claim(
        "fix",
        f"{GROUP_NAMES[grp]}: decisions missed per case, released -> fine-tuned (paired 95 % CI of the difference)",
        f"{r.released_mean:.2f} -> {r.fine_tuned_mean:.2f} ({r.diff_mean:+.2f}, {100 * r.diff_mean / r.released_mean:+.0f} %; "
        f"{r.diff_ci_lo:+.2f} to {r.diff_ci_hi:+.2f})",
        "evaluate_finetuned/endpoints_cases.csv",
    )
fg = pd.read_csv(RES / "evaluate_finetuned/endpoint_forgetting.csv").set_index("level")
claim(
    "fix",
    "ordinary N-2 scenarios: median worst-bus error, released -> fine-tuned (p.u.)",
    f"{fg.loc['N-2', 'released_median_err']:.4f} -> {fg.loc['N-2', 'fine_tuned_median_err']:.4f}",
    "evaluate_finetuned/endpoint_forgetting.csv",
)
nf = json.loads((RES / "evaluate_finetuned/endpoint_near_fold.json").read_text())
beyond = sum(nf[k]["beyond_fold_points"] for k in ("unresolved_at_actual_load", "solved_at_actual_load"))
conv = sum(nf[k]["fine_tuned_newton_converged_beyond"] for k in ("unresolved_at_actual_load", "solved_at_actual_load"))
claim(
    "fix",
    "load points beyond a fold where Newton from the fine-tuned state converged",
    f"{conv} of {beyond}",
    "evaluate_finetuned/endpoint_near_fold.json",
)

# ------------------------------------------------------------------ the warning
head = pd.read_csv(RES / "evaluate_finetuned/endpoint_head.csv").set_index(["model", "test"])
for test in ("F1", "footprint", "K1 storms at the actual load", "K1 points within 5 % of the fold"):
    r = head.loc[("released", test)]
    lo, hi = json.loads(r.auroc_ci)
    claim(
        "warning",
        f"released GENCO, {TEST_NAMES[test]}: AUROC (95 % CI); recall / false alarms at the frozen threshold",
        f"{r.auroc:.3f} ({lo:.3f} to {hi:.3f}); {r.recall_at_threshold:.2f} / {r.false_alarm_rate:.2f}",
        "evaluate_finetuned/endpoint_head.csv",
    )
scores = pd.read_csv(RES / "analysis/warning_scores.csv")
topo = scores[(scores.test == "F1") & (scores.features == "A topology") & (scores.classifier == "boosted trees")]
claim(
    "warning",
    "outage-topology baseline without GENCO (boosted trees), frozen-test AUROC",
    f"{roc_auc_score(topo.label_unresolved, topo.score):.3f}",
    "analysis/warning_scores.csv",
)
abl = json.loads((RES / "analysis/warning_ablation.json").read_text())["paired_auroc_differences"]
for test in ("F1", "K1"):
    d = abl[f"{test}: released GENCO outputs + embeddings vs topology (boosted trees)"]
    claim(
        "warning",
        f"{TEST_NAMES[test]}: AUROC gain of the GENCO warning over the topology baseline (paired 95 % CI)",
        f"{d['difference']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})",
        "analysis/warning_ablation.json",
    )
rows_abl = pd.DataFrame(json.loads((RES / "analysis/warning_ablation.json").read_text())["rows"])
fa = rows_abl[
    (rows_abl.test == "F1")
    & (
        ((rows_abl.features == "A topology") & (rows_abl.classifier == "boosted trees"))
        | (rows_abl.features == "BC outputs + embeddings (reported)") & (rows_abl.genco_model == "released")
    )
]
claim(
    "warning",
    "false-alarm rate at the 95 %-recall rule: topology baseline / GENCO warning",
    " / ".join(f"{v:.2f}" for v in fa.sort_values("features").false_alarm_rate),
    "analysis/warning_ablation.json",
)

routing = json.loads((RES / "analysis/routing.json").read_text())
claim(
    "routing",
    "rule: at least N lines out and spread at most H hops -> fine-tuned model; accuracy on training cases",
    f"N = {routing['min_lines']}, H = {routing['threshold_hops']:.0f}; {routing['training_accuracy']:.3f}",
    "analysis/routing.json",
)
for group in ("clustered", "random"):
    r = routing["frozen_test"][group]
    d = r["routed_minus_released_missed"]
    claim(
        "routing",
        f"frozen test, {group} outages: routed to fine-tuned; decisions missed per case, released -> routed (95 % CI)",
        f"{r['routed_to_fine_tuned']} of {r['cases']}; {r['released_missed_mean']:.2f} -> {r['routed_missed_mean']:.2f} "
        f"({d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f})",
        "analysis/routing.json",
    )
claim(
    "routing",
    "random N-2 / N-10 / N-20 outages routed to the fine-tuned model",
    " / ".join(f"{100 * routing['random_outages_routed_to_fine_tuned'][k]:.0f} %" for k in ("N-2", "N-10", "N-20")),
    "analysis/routing.json",
)
v1 = json.loads((RES / "independent_check/summary.json").read_text())["released_cases"]
claim(
    "check",
    "PYPOWER from raw parameters vs released solutions: largest |dVm| (p.u.); decisions agreeing",
    f"{v1['pypower_vs_released_max_dVm']:.1e}; {v1['overload_sets_agree']} of {v1['cases']}",
    "independent_check/summary.json",
)

if __name__ == "__main__":
    out = RES / "headline_numbers.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "claim", "value", "source"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    width = max(len(r["claim"]) for r in rows)
    for r in rows:
        print(f"{r['section']:9s} {r['claim']:{width}s}  {r['value']}")
    print(f"\n{len(rows)} numbers -> {out.relative_to(ROOT)}")

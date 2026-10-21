"""Recomputes every number quoted in the write-up from the files in results/.

python scripts/headline_numbers.py        # prints the table and writes results/headline_numbers.csv
"""

import csv
import json
from pathlib import Path

import numpy as np
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
# seed0 is the published checkpoint; seed1 and the first run (as_run) measure the spread between training runs
RUNS = {"seed0": "seed 0", "seed1": "seed 1", "as_run": "first run"}
PUBLISHED = "seed0"
t1 = json.loads((RES / f"finetune_{PUBLISHED}/summary.json").read_text())
claim(
    "fix",
    "training cases (regression targets) / minutes on one T4",
    f"{t1['cases']['regression_targets_train']} / {t1['training_minutes']:.0f}",
    f"finetune_{PUBLISHED}/summary.json",
)
endpoints = {
    run: pd.read_csv(RES / f"evaluate_finetuned_{run}/endpoints_cases.csv").set_index(["group", "metric"])
    for run in RUNS
}
ep = endpoints[PUBLISHED]
for grp in ("clustered (F1)", "footprint (held out)", "uniform (F1)"):
    r = ep.loc[(grp, "missed")]
    claim(
        "fix",
        f"{GROUP_NAMES[grp]}: decisions missed per case, released -> fine-tuned (paired 95 % CI of the difference)",
        f"{r.released_mean:.2f} -> {r.fine_tuned_mean:.2f} ({r.diff_mean:+.2f}, {100 * r.diff_mean / r.released_mean:+.0f} %; "
        f"{r.diff_ci_lo:+.2f} to {r.diff_ci_hi:+.2f})",
        f"evaluate_finetuned_{PUBLISHED}/endpoints_cases.csv",
    )
    claim(
        "runs",
        f"{GROUP_NAMES[grp]}: change in decisions missed per case (95 % CI), " + " / ".join(RUNS.values()),
        " / ".join(
            f"{e.loc[(grp, 'missed')].diff_mean:+.2f} ({e.loc[(grp, 'missed')].diff_ci_lo:+.2f} to "
            f"{e.loc[(grp, 'missed')].diff_ci_hi:+.2f})"
            for e in endpoints.values()
        ),
        "evaluate_finetuned_*/endpoints_cases.csv",
    )
forgetting = {
    run: pd.read_csv(RES / f"evaluate_finetuned_{run}/endpoint_forgetting.csv").set_index("level") for run in RUNS
}
fg = forgetting[PUBLISHED]
claim(
    "fix",
    "ordinary N-2 scenarios: median worst-bus error, released -> fine-tuned (p.u.)",
    f"{fg.loc['N-2', 'released_median_err']:.4f} -> {fg.loc['N-2', 'fine_tuned_median_err']:.4f}",
    f"evaluate_finetuned_{PUBLISHED}/endpoint_forgetting.csv",
)
claim(
    "runs",
    "ordinary N-2 scenarios: fine-tuned median worst-bus error (p.u.), " + " / ".join(RUNS.values()),
    " / ".join(f"{f.loc['N-2', 'fine_tuned_median_err']:.4f}" for f in forgetting.values()),
    "evaluate_finetuned_*/endpoint_forgetting.csv",
)
nf = json.loads((RES / f"evaluate_finetuned_{PUBLISHED}/endpoint_near_fold.json").read_text())
beyond = sum(nf[k]["beyond_fold_points"] for k in ("unresolved_at_actual_load", "solved_at_actual_load"))
conv = sum(nf[k]["fine_tuned_newton_converged_beyond"] for k in ("unresolved_at_actual_load", "solved_at_actual_load"))
claim(
    "fix",
    "load points beyond a fold where Newton from the fine-tuned state converged",
    f"{conv} of {beyond}",
    f"evaluate_finetuned_{PUBLISHED}/endpoint_near_fold.json",
)
heads = {
    run: pd.read_csv(RES / f"evaluate_finetuned_{run}/endpoint_head.csv").set_index(["model", "test"]) for run in RUNS
}
claim(
    "runs",
    "warning on the fine-tuned model, frozen-test AUROC, " + " / ".join(RUNS.values()),
    " / ".join(f"{h.loc[('fine_tuned', 'F1'), 'auroc']:.3f}" for h in heads.values()),
    "evaluate_finetuned_*/endpoint_head.csv",
)

# ------------------------------------------------------------------ fine-tuning with replay (notebooks/finetune_replay.ipynb)
rp = json.loads((RES / "finetune_replay_seed0/summary.json").read_text())
claim(
    "replay",
    "replay run: checkpoints within 10 % of the released ordinary error, of those evaluated; selected model",
    f"{rp['replay']['checkpoints_keeping_ordinary_accuracy']} of {rp['replay']['checkpoints_evaluated']}; "
    + ("the released model" if rp["replay"]["selected_is_released"] else f"step {rp['selected_step']}"),
    "finetune_replay_seed0/summary.json",
)
logs = {
    name: pd.read_csv(RES / f"{run}/training_log.csv")
    for name, run in (("published", "finetune_seed0"), ("replay", "finetune_replay_seed0"))
}
u0 = logs["published"].loc[logs["published"].step == 0, "worst_bus_err_median_uniform/base"].item()
inc = {n: (L[L.step > 0]["worst_bus_err_median_uniform/base"] - u0).mean() for n, L in logs.items()}
claim(
    "replay",
    "validation, random and base cases: mean rise of the median worst-bus error over the released model (p.u.), published / replay",
    f"{inc['published']:.4f} / {inc['replay']:.4f} ({100 * (inc['replay'] / inc['published'] - 1):+.0f} %)",
    "finetune_seed0, finetune_replay_seed0/training_log.csv",
)
clu = {n: L[L.step > 0].worst_bus_err_median_clustered for n, L in logs.items()}
claim(
    "replay",
    "validation, clustered cases: median worst-bus error (p.u.), released / range over checkpoints, published / replay",
    f"{logs['published'].loc[logs['published'].step == 0, 'worst_bus_err_median_clustered'].item():.4f} / "
    f"{clu['published'].min():.4f} to {clu['published'].max():.4f} / {clu['replay'].min():.4f} to {clu['replay'].max():.4f}",
    "finetune_seed0, finetune_replay_seed0/training_log.csv",
)
er = (
    pd.read_csv(RES / "evaluate_finetuned_replay_seed0/endpoints_cases.csv")
    .set_index(["group", "metric"])
    .loc[("clustered (F1)", "missed")]
)
claim(
    "replay",
    "replay evaluation, clustered outages: change in decisions missed per case (95 % CI)",
    f"{er.diff_mean:+.2f} ({er.diff_ci_lo:+.2f} to {er.diff_ci_hi:+.2f})",
    "evaluate_finetuned_replay_seed0/endpoints_cases.csv",
)

# ------------------------------------------------------------------ the warning
head = heads[PUBLISHED]
for test in ("F1", "footprint", "K1 storms at the actual load", "K1 points within 5 % of the fold"):
    r = head.loc[("released", test)]
    lo, hi = json.loads(r.auroc_ci)
    claim(
        "warning",
        f"released GENCO, {TEST_NAMES[test]}: AUROC (95 % CI); recall / false alarms at the frozen threshold",
        f"{r.auroc:.3f} ({lo:.3f} to {hi:.3f}); {r.recall_at_threshold:.2f} / {r.false_alarm_rate:.2f}",
        f"evaluate_finetuned_{PUBLISHED}/endpoint_head.csv",
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
for group, label in (
    ("clustered", "frozen test, clustered"),
    ("random", "frozen test, random"),
    ("footprint", "held-out construction"),
):
    r = routing["runs"][PUBLISHED][group]
    d = r["routed_minus_released_missed"]
    claim(
        "routing",
        f"{label} outages: routed to fine-tuned; decisions missed per case, released -> routed (95 % CI)",
        f"{r['routed_to_fine_tuned']} of {r['cases']}; {r['released_missed_mean']:.2f} -> {r['routed_missed_mean']:.2f} "
        f"({d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f})",
        "analysis/routing.json",
    )
claim(
    "runs",
    "routing, clustered outages: decisions missed per case, routed, " + " / ".join(RUNS.values()),
    " / ".join(f"{routing['runs'][run]['clustered']['routed_missed_mean']:.2f}" for run in RUNS),
    "analysis/routing.json",
)
claim(
    "routing",
    "random N-2 / N-10 / N-20 outages routed to the fine-tuned model",
    " / ".join(f"{100 * routing['random_outages_routed_to_fine_tuned'][k]:.0f} %" for k in ("N-2", "N-10", "N-20")),
    "analysis/routing.json",
)
# ------------------------------------------------------------------ triage: the queue ordered by the warning
pcs = pd.read_csv(RES / "evaluate_finetuned_seed0/unresolved_warning/per_case_scores.csv")
f1 = pcs[(pcs.model == "released") & (pcs.test == "F1")]


def opened_for(score, label, share):
    """Cases an engineer opens, in score order, before finding `share` of the unresolved ones."""
    found = label.to_numpy()[np.argsort(-score.to_numpy(), kind="stable")].cumsum()
    return int(np.argmax(found >= share * label.sum())) + 1


triage = {
    "the warning": f1.score,
    "power-mismatch p99 alone": f1.genco_resP_p99_MW,
    "lowest predicted voltage alone": -f1.genco_lowest_voltage,
}
claim(
    "triage",
    "frozen test, queue ordered by: " + " / ".join(triage) + ": share opened to find 95 % of the unresolved cases",
    " / ".join(f"{100 * opened_for(v, f1.label_unresolved, 0.95) / len(f1):.0f} %" for v in triage.values()),
    "evaluate_finetuned_seed0/unresolved_warning/per_case_scores.csv",
)
claim(
    "triage",
    "frozen test: unresolved cases / cases (the share an ideal ordering opens)",
    f"{int(f1.label_unresolved.sum())} / {len(f1)} ({100 * f1.label_unresolved.mean():.0f} %)",
    "evaluate_finetuned_seed0/unresolved_warning/per_case_scores.csv",
)
nfw = json.loads((RES / "near_fold_warning/summary.json").read_text())
nfa = pd.read_csv(RES / "near_fold_warning/auroc.csv").set_index("model")
claim(
    "near_fold",
    "warning trained near the fold, storms held out: AUROC (95 % CI); published warning; rule met",
    f"{nfw['primary_auroc']:.2f} ({nfw['primary_ci95'][0]:.2f} to {nfw['primary_ci95'][1]:.2f}); "
    f"{nfa.loc['published warning (unchanged)', 'auroc']:.2f}; {'yes' if nfw['rule_met'] else 'no (rule: at least 0.70)'}",
    "near_fold_warning/summary.json",
)
claim(
    "near_fold",
    "trained near the fold: AUROC, embeddings only / GENCO outputs only / load factor only",
    " / ".join(
        f"{nfa.loc[f'trained near the fold: {k}', 'auroc']:.2f}"
        for k in ("embeddings only", "outputs only", "load factor only")
    ),
    "near_fold_warning/auroc.csv",
)

# ------------------------------------------------------------------ a second grid (IEEE 118, its released GENCO)
g2 = json.loads((RES / "second_grid/summary.json").read_text())
g2c = pd.read_csv(RES / "second_grid/cases.csv")
g2r = g2c[g2c.resolved]
src = "second_grid/summary.json"
claim(
    "grid2",
    "operating points from the authors' test split / contingencies / resolved by the classical solver / unresolved",
    f"{g2['selection']['base_points']} / {g2['cases']} / {g2['total']['resolved']:.0f} / {g2['total']['unresolved']:.0f}",
    src,
)
claim(
    "grid2",
    "model check: GENCO's RMSE of Vm at PQ buses, ours / the authors' (p.u.)",
    f"{g2['model_check']['genco_rmse_vm_pq']:.5f} / {g2['model_check']['authors_rmse_vm_pq']:.5f}",
    src,
)
claim(
    "grid2",
    "verified workflow: AC decisions / false alarms / unresolved cases passed / decisions differing from the classical solution",
    f"{g2['total']['verified_decisions']:.0f} / {g2['total']['false_alarms']:.0f} / {g2['total']['missed_alarms']:.0f} / "
    f"{g2['total']['verified_with_a_different_decision']:.0f}",
    src,
)
miss = (g2r.genco_missed_overloads > 0) | (g2r.genco_missed_vviol > 0)
claim(
    "grid2",
    "GENCO alone: resolved cases with a missed decision",
    pct(int(miss.sum()), len(g2r)),
    "second_grid/cases.csv",
)
groups = {
    "one line": ["uniform-N1"],
    "random, 3 and 6 lines": ["uniform-N3", "uniform-N6"],
    "clustered, 3 and 6 lines": ["nearest-N3", "nearest-N6", "corridor-N6"],
}
claim(
    "grid2",
    "median worst-bus error of GENCO alone (p.u.): " + " / ".join(groups),
    " / ".join(f"{g2r[g2r.family.isin(f)].genco_worst_err.median():.4f}" for f in groups.values()),
    "second_grid/cases.csv",
)
lo, hi = g2["ci95"]["clustered_over_random_err_ratio"]
claim(
    "grid2",
    "median worst-bus error, clustered / random outages (95 % CI)",
    f"x{g2['clustered_over_random_err_ratio']:.1f} (x{lo:.1f} to x{hi:.1f})",
    src,
)
fam_err = g2r.groupby("family").genco_worst_err.median()
claim(
    "grid2",
    "median worst-bus error, clustered / random: N-3, N-6, six-line corridor against six random lines",
    f"x{fam_err['nearest-N3'] / fam_err['uniform-N3']:.1f}, x{fam_err['nearest-N6'] / fam_err['uniform-N6']:.1f}, "
    f"x{fam_err['corridor-N6'] / fam_err['uniform-N6']:.1f}",
    "second_grid/cases.csv",
)
u2 = g2c[~g2c.resolved]
claim(
    "grid2",
    "unresolved cases where GENCO predicts at least one violation; median overloads it predicts",
    f"{g2['unresolved']['cases'] - g2['unresolved']['genco_predicts_no_violation']} of {g2['unresolved']['cases']}; "
    f"{u2.genco_n_over.median():.0f}",
    src,
)
lg = g2["load_growth"]
claim(
    "grid2",
    "GENCO response to uniform load growth relative to AC (median slope, interquartile range)",
    f"{lg['slope_median']:.2f} ({lg['slope_iqr'][0]:.2f} to {lg['slope_iqr'][1]:.2f}), {lg['operating_points']} operating points",
    src,
)
claim(
    "grid2",
    "mean Newton iterations: classical / from GENCO's state",
    f"{g2r.classical_iterations.mean():.2f} / {g2r[g2r.verified_start == 'genco'].verified_iterations.mean():.2f}",
    "second_grid/cases.csv",
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

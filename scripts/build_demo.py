"""Builds the case explorer (demo/index.html) from the frozen-test results.

    python scripts/build_demo.py

Reads results/ and demo/app.html, writes demo/index.html: one self-contained page (also the Hugging Face Space).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FAMILIES = {
    "uniform-N10": "Random N-10",
    "uniform-N20": "Random N-20",
    "nearest-N10": "Clustered N-10",
    "nearest-N20": "Clustered N-20",
    "corridor-N20": "Corridor N-20",
}
SHOWCASE_NOTES = {
    "normal": "An ordinary random outage. GENCO's answer and the AC solution agree on every decision.",
    "near_limit": "GENCO flags two overloads that the AC solution does not have: a false alarm the check clears.",
    "misleading_corrected": "A clustered outage. GENCO misses 4 overloads and all 15 low-voltage buses; the check finds them.",
    "unresolved": "A corridor cut. No start converges, so the case is an alarm. GENCO alone shows no low voltage at all.",
}


def r4(values):
    return [round(float(v), 4) for v in values]


def cases_table():
    cases = pd.read_csv(RES / "frozen_test/cases.csv").set_index("sid")
    wf = pd.read_csv(RES / "frozen_test/workflows.csv")
    wf = wf[wf.set == "final"]
    classical = wf[wf.workflow == "classical"].set_index("sid")
    genco = wf[wf.workflow == "genco_verify"].set_index("sid")
    alone = pd.read_csv(RES / "frozen_test/genco_alone.csv").set_index("sid")
    warn = pd.read_csv(RES / "evaluate_finetuned_seed0/unresolved_warning/per_case_scores.csv")
    warn = warn[(warn.model == "released") & (warn.test == "F1")]
    warn = warn.assign(sid=warn.case.str.split(":").str[1].astype(int)).set_index("sid")

    rows = []
    for sid, c in classical.iterrows():
        meta, g, w = cases.loc[sid], genco.loc[sid], warn.loc[sid]
        row = {
            "id": int(sid),
            "base": int(meta.base),
            "family": meta.family,
            "centre": int(meta.centre),
            "removed": json.loads(meta.removed_lines),
            "status": c.status,
            "n_over": None if c.status == "unresolved" else int(c.n_over),
            "n_low_v": None if c.status == "unresolved" else int(c.n_low_v),
            "min_vm": None if c.status == "unresolved" else round(float(c.min_vm_all), 4),
            "start": g.start_used if isinstance(g.start_used, str) else None,
            "iterations": None if c.status == "unresolved" else int(g.iterations),
            "ms_genco": round(float(g.end_to_end_ms), 1),
            "ms_classical": round(float(c.end_to_end_ms), 1),
            "different_solution": bool(g.different_solution) if "different_solution" in g else False,
            "warning": round(float(w.score), 4),
            "genco_min_vm": round(float(w.genco_lowest_voltage), 4),
            "residual_mw": round(float(w.genco_resP_p99_MW), 2),
        }
        if sid in alone.index:
            a = alone.loc[sid]
            row["alone"] = {
                "over_ref": int(a.ref_overloads),
                "over_missed": int(a.missed_overloads),
                "v_ref": int(a.ref_vviol),
                "v_missed": int(a.missed_vviol),
                "false_flags": int(a.false_overloads),
            }
        rows.append(row)
    return rows


def showcase():
    demo = json.loads((RES / "frozen_test/demo_cases.json").read_text())
    by_name = {c["name"]: c for c in demo["cases"]}
    out = {}
    for name, note in SHOWCASE_NOTES.items():
        c = by_name[name]
        rated = np.array(c["rated_line_idx"])
        g, v = c["genco"], c.get("verified") or {}
        entry = {
            "note": note,
            "vm_genco": r4(g["vm"]),
            "vm_pre": r4(c["buses"]["pre_contingency_vm"]),
            "load_genco": dict(zip(rated.tolist(), r4(g["loading"]))),
            "over_genco": g["overloaded_lines"],
            "low_genco": g["low_voltage_buses"],
            "high_genco": g["high_voltage_buses"],
            "history": c["workflow_genco_verify"]["history"],
        }
        if v:
            entry.update(
                {
                    "vm_ac": r4(v["vm"]),
                    "load_ac": dict(zip(rated.tolist(), r4(v["loading"]))),
                    "over_ac": v["overloaded_lines"],
                    "low_ac": v["low_voltage_buses"],
                    "high_ac": v["high_voltage_buses"],
                }
            )
        if name == "misleading_corrected" and "reactive_limits" in by_name:
            entry["q_limits"] = by_name["reactive_limits"].get("with_q_limits")
        out[int(c["sid"])] = entry
    return out


def overview():
    summary = pd.read_csv(RES / "frozen_test/summary_by_workflow.csv", index_col=0)
    head = pd.read_csv(RES / "evaluate_finetuned_seed0/endpoint_head.csv").set_index(["model", "test"])
    return {
        "verified": int(summary.loc["classical", "verified"]),
        "unresolved": int(summary.loc["classical", "unresolved"]),
        "false_alarms": int(summary.loc["genco_verify", "false_alarms"]),
        "threshold": round(float(head.loc[("released", "F1"), "threshold"]), 4),
        "auroc": round(float(head.loc[("released", "F1"), "auroc"]), 3),
    }


def main():
    layout = json.loads((RES / "grid_layout.json").read_text())
    data = {
        "grid": {"x": layout["x"], "y": layout["y"], "kv": layout["kv"], "f": layout["f"], "t": layout["t"]},
        "families": FAMILIES,
        "cases": cases_table(),
        "showcase": showcase(),
        "overview": overview(),
    }
    blob = json.dumps(data, separators=(",", ":"))
    template = (ROOT / "demo/app.html").read_text()
    page = template.replace("/*DATA*/null", blob)
    (ROOT / "demo/index.html").write_text(page)
    print(f"wrote demo/index.html ({len(page) / 1e6:.2f} MB, {len(data['cases'])} cases)")


if __name__ == "__main__":
    main()

"""Tests of the verification workflow.

Most tests use a three-bus grid (`synthetic_case`) and need no data. The last one replays three cases of the final
evaluation and runs only when the released GENCO data are on disk.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from genco_verify import (
    DEFAULT_POLICY,
    STATUSES,
    StartAttempt,
    Verifier,
    case_from_tables,
    newton_solve,
    synthetic_case,
)


def test_ordinary_case_is_verified_safe():
    v = Verifier(synthetic_case(load_mw=80, rate_mva=200))
    r = v.run(policy_name="classical")
    assert r.status == "verified_safe" and r.converged and not r.violations
    assert r.residuals["mismatch_end_pu"] < DEFAULT_POLICY["tol_pu"]
    assert r.start_used == "base" and r.unresolved_reason is None


def test_overload_is_a_verified_violation():
    v = Verifier(synthetic_case(load_mw=80, rate_mva=40))
    r = v.run(policy_name="classical")
    assert r.status == "verified_violation" and r.violations and r.limits["n_over"] >= 1


def test_impossible_load_stays_unresolved():
    v = Verifier(synthetic_case(load_mw=3000, load_mvar=1500))
    r = v.run(policy_name="classical")
    assert r.status == "unresolved" and not r.converged and r.state is None
    assert "no start converged" in r.unresolved_reason
    assert all(not h["converged"] for h in r.history) and len(r.history) == 3
    r2 = v.run(policy_name="classical_recovery")
    assert r2.status == "unresolved" and len(r2.history) == 7  # the longer cap and the homotopy were tried and failed


def test_genco_state_is_used_first_and_fallbacks_follow():
    d = synthetic_case(load_mw=80, rate_mva=200)
    v = Verifier(d)
    truth = v.run(policy_name="classical")
    vm = np.array(truth.state["vm"])
    va = np.array(truth.state["va_deg"])
    r = v.run(genco_vm=vm + np.array([0, 0, 0.01]), genco_va_deg=va + 0.5, policy_name="genco_verify")
    assert r.status == "verified_safe" and r.start_used == "genco" and r.genco is not None
    assert abs(np.array(r.state["vm"]) - vm).max() < 1e-7
    # a nonsensical GENCO state: the fallbacks take over, the outcome is still verified and GENCO's attempt is in the history
    r = v.run(
        genco_vm=np.array([1.0, 1.0, 0.05]), genco_va_deg=np.array([0.0, 100.0, -170.0]), policy_name="genco_verify"
    )
    assert r.converged and r.history[0]["start"] == "genco"
    assert r.status in STATUSES


def test_unresolved_never_becomes_safe():
    v = Verifier(synthetic_case(load_mw=3000, load_mvar=1500))
    r = v.run(genco_vm=np.ones(3), genco_va_deg=np.zeros(3), policy_name="genco_verify")
    assert r.status == "unresolved" and r.violations is False and r.limits == {} and r.state is None


def test_reactive_limit_switching_changes_the_outcome():
    """Without the limits the PV bus supplies more reactive power than its 5 Mvar limit; with the limits enforced it is held
    at the limit as a PQ bus, its voltage drops below the set-point, and the complementarity condition holds."""
    d = synthetic_case(load_mw=80, load_mvar=120, rate_mva=200)
    v = Verifier(d)
    r = v.run(policy_name="classical")
    V = np.array(r.state["vm"]) * np.exp(1j * np.radians(np.array(r.state["va_deg"])))
    q_pv = (V * np.conj(v.Y @ V)).imag[1] * 100 + d["Qd"][1]
    assert r.status == "verified_safe" and q_pv > 5.0  # the PV bus supplies reactive power that the limit would forbid
    vq = Verifier(d, {"enforce_q_limits": True})
    rq = vq.run(policy_name="classical")
    assert rq.converged and vq.last_q_limits["held"] == {1: "max"}
    Vq = np.array(rq.state["vm"]) * np.exp(1j * np.radians(np.array(rq.state["va_deg"])))
    q_pv_q = (Vq * np.conj(vq.Y @ Vq)).imag[1] * 100 + d["Qd"][1]
    assert abs(q_pv_q - 5.0) < 1e-4 and np.abs(Vq[1]) < d["Vm"][1]
    assert rq.history[0]["note"].startswith("1 buses held")


def test_switched_bus_is_checked_against_the_voltage_limits():
    """A PV bus held at its reactive limit is a PQ bus in the solved state; its voltage must be checked like one. Here bus 1
    switches to Qmax and ends at 0.889 p.u., below the 0.9 limit: it must appear among the low-voltage buses."""
    d = synthetic_case(load_mw=80, load_mvar=300, rate_mva=2000)
    r = Verifier(d, {"enforce_q_limits": True}).run(policy_name="classical")
    assert r.converged and r.limits["q_limits"]["held"] == {1: "max"} and r.limits["buses_at_q_limit"] == [1]
    assert r.state["vm"][1] < 0.9 and 1 in r.limits["low_voltage_buses"] and 2 in r.limits["low_voltage_buses"]
    assert r.status == "verified_violation" and r.limits["voltage_checked_buses"] == 2
    assert (
        r.residuals["mismatch_end_pu"] < DEFAULT_POLICY["tol_pu"]
    )  # the residual is measured with the bus at its limit
    assert r.history[0]["q_limits"]["held"] == {1: "max"}


def test_requesting_limits_without_limit_data_is_an_error():
    d = synthetic_case()
    d.pop("qmin_pu")
    d.pop("qmax_pu")
    try:
        Verifier(d, {"enforce_q_limits": True})
    except ValueError as e:
        assert "qmin_pu" in str(e)
    else:
        raise AssertionError("a case without reactive limits was accepted with enforce_q_limits set")
    Verifier(d).run(policy_name="classical")  # without the option the same case is fine


def test_recovery_homotopy_enforces_the_limits():
    """Every fallback of the recovery policy must respect the same constraints: the load-stepping homotopy ends with bus 1 held
    at its 5 Mvar limit (the unconstrained solution would supply about 35 Mvar)."""
    d = synthetic_case(load_mw=80, load_mvar=120, rate_mva=2000)
    v = Verifier(d, {"enforce_q_limits": True})
    r = v.run(policy_name="classical_recovery", order=("homotopy",))
    assert r.converged and r.start_used == "homotopy" and r.limits["q_limits"]["held"] == {1: "max"}
    V = np.array(r.state["vm"]) * np.exp(1j * np.radians(np.array(r.state["va_deg"])))
    q_pv = (V * np.conj(v.Y @ V)).imag[1] * 100 + d["Qd"][1]
    assert abs(q_pv - 5.0) < 1e-4 and "held at a reactive limit" in r.history[0]["note"]
    r0 = Verifier(d).run(policy_name="classical_recovery", order=("homotopy",))
    V0 = np.array(r0.state["vm"]) * np.exp(1j * np.radians(np.array(r0.state["va_deg"])))
    assert (V0 * np.conj(v.Y @ V0)).imag[1] * 100 + d["Qd"][1] > 30  # without the option the limit is exceeded


def test_branch_check_reports_two_solutions_of_the_same_equations():
    """At 400 MW the three-bus grid has a high-voltage solution (reached from the pre-contingency, DC and flat starts) and a
    low-voltage one (reached from a low-voltage 'GENCO' state). The frozen policy stops at the first converged start and
    only flags the low voltage; the branch check runs every start, reports both solutions and whether their decisions differ,
    and takes the reference by the declared rule."""
    d = synthetic_case(load_mw=400, load_mvar=80, rate_mva=2000)
    low_vm, low_va = np.array([1.02, 1.0, 0.2]), np.array([0.0, -5.0, -45.0])
    frozen = Verifier(d).run(genco_vm=low_vm, genco_va_deg=low_va, policy_name="genco_verify")
    assert (
        frozen.start_used == "genco"
        and frozen.limits["min_vm_all_buses"] < 0.3
        and frozen.flags["low_voltage_branch_suspect"]
    )
    assert not frozen.flags["branch_check"] and "solutions" not in frozen.flags and len(frozen.history) == 1
    r = Verifier(d, {"branch_check": True}).run(genco_vm=low_vm, genco_va_deg=low_va, policy_name="genco_verify")
    assert r.flags["branch_check"] and r.flags["n_starts_converged"] == 4 and r.flags["n_solutions_found"] == 2
    assert r.flags["multiple_solutions"] and r.flags["decisions_differ"]
    assert (
        r.start_used == "genco" and r.status == "verified_violation"
    )  # branch_choice "first": the start order decides
    sols = {s_["start"]: s_ for s_ in r.flags["solutions"]}
    assert (
        sols["genco"]["violation"]
        and not sols["base"]["violation"]
        and sols["base"]["reached_by"] == ["base", "dc", "flat"]
    )
    assert sols["base"]["max_dvm_vs_chosen_pu"] > 0.5 and sols["base"]["min_vm_all_buses"] > 0.9
    r2 = Verifier(d, {"branch_check": True, "branch_choice": "highest_voltage"}).run(
        genco_vm=low_vm, genco_va_deg=low_va, policy_name="genco_verify"
    )
    assert (
        r2.start_used == "base"
        and r2.status == "verified_safe"
        and r2.flags["multiple_solutions"]
        and r2.flags["decisions_differ"]
    )
    assert not r2.flags["low_voltage_branch_suspect"]
    r3 = Verifier(d, {"branch_check": True}).run(policy_name="classical")  # the classical starts agree with each other
    assert r3.flags["n_solutions_found"] == 1 and not r3.flags["multiple_solutions"] and len(r3.history) == 3


def att(name):
    return StartAttempt(start=name, available=True, converged=True, iterations=1)


def test_branch_comparison_uses_complex_voltages_and_every_start():
    """The comparison must (a) tell apart states with equal magnitudes and different angles, (b) treat a global rotation of the
    angle reference as the same state, and (c) report a decision that differs inside a group (a limit crossed by less than the
    grouping tolerance), per start and in decisions_differ."""
    d = synthetic_case(load_mw=80, rate_mva=200)
    v = Verifier(d, {"branch_check": True})
    V, ok, _ = newton_solve(v.Y, v.S, v.starts["base"], v.ref, v.pv, v.pq, 1e-8, 30)
    assert ok
    d["min_vm_pu"] = np.array([0.9, 0.9, float(np.abs(V[2])) - 2e-5])  # bus 2 sits 2e-5 p.u. above its limit: safe
    v = Verifier(d, {"branch_check": True})
    A = V.copy()
    B = A.copy()
    B[2] *= np.exp(1j * np.radians(10.0))  # same magnitudes, another angle at bus 2
    C = A * np.exp(1j * np.radians(5.0))  # the same state with a rotated angle reference
    D = A.copy()
    D[2] *= (np.abs(A[2]) - 5e-5) / np.abs(A[2])  # within the grouping tolerance, but across the limit
    sols = [(att("a"), A), (att("b"), B), (att("c"), C), (att("d"), D)]
    out = v._compare_solutions(sols, sols[0], v.classify(A))
    assert out["n_starts_converged"] == 4 and out["n_solutions_found"] == 2 and out["multiple_solutions"]
    by = {st["start"]: st for st in out["starts"]}
    assert (
        by["b"]["max_dvm_vs_chosen_pu"] < 1e-9
        and by["b"]["max_dv_vs_chosen_pu"] > 0.1
        and by["b"]["solution"] != by["a"]["solution"]
    )
    assert by["c"]["max_dv_vs_chosen_pu"] < 1e-9 and by["c"]["solution"] == by["a"]["solution"]
    assert (
        by["d"]["solution"] == by["a"]["solution"]
        and not by["d"]["same_decisions_as_chosen"]
        and by["d"]["n_low_v"] == 1
    )
    assert out["decisions_differ"] and not out["solutions"][by["a"]["solution"]]["decisions_agree_within_group"]
    assert sorted(out["solutions"][by["a"]["solution"]]["reached_by"]) == ["a", "c", "d"]


def test_within_group_disagreement_is_detected_even_when_both_differ_from_the_chosen_state():
    """Two states of one group that both differ from the chosen state, but in different ways, must still be reported as
    disagreeing with each other (the flag compares decision sets, not their relation to the chosen state)."""
    d = synthetic_case(load_mw=80, rate_mva=200)
    v0 = Verifier(d, {"branch_check": True})
    V, ok, _ = newton_solve(v0.Y, v0.S, v0.starts["base"], v0.ref, v0.pv, v0.pq, 1e-8, 30)
    assert ok
    vm2 = float(np.abs(V[2]))
    d["min_vm_pu"] = np.array([0.9, 0.9, vm2 - 2e-5])
    d["max_vm_pu"] = np.array([1.1, 1.1, vm2 + 2e-5])  # a 4e-5 p.u. band at bus 2
    v = Verifier(d, {"branch_check": True})
    far = V * np.exp(1j * np.radians(np.array([0.0, 20.0, 40.0])))  # another solution group: the chosen state
    low = V.copy()
    low[2] *= (vm2 - 3e-5) / vm2  # below the band
    high = V.copy()
    high[2] *= (vm2 + 3e-5) / vm2  # above the band, same group as low (6e-5 apart)
    sols = [(att("chosen"), far), (att("low"), low), (att("high"), high)]
    out = v._compare_solutions(sols, sols[0], v.classify(far))
    by = {st["start"]: st for st in out["starts"]}
    assert by["low"]["solution"] == by["high"]["solution"] != by["chosen"]["solution"]
    assert not by["low"]["same_decisions_as_chosen"] and not by["high"]["same_decisions_as_chosen"]
    assert by["low"]["n_low_v"] == 1 and by["high"]["n_high_v"] == 1
    assert out["solutions"][by["low"]["solution"]]["decisions_agree_within_group"] is False


def test_limit_loop_rejects_an_inconsistent_state():
    """A bus held at its maximum while its voltage sits above the set-point should be released; if the release budget forbids
    it, the state is not a solution of the limited problem and must not be accepted. The final consistency check catches
    it independently of the loop's counter, and the start is reported as not converged."""
    d = synthetic_case(load_mw=80, load_mvar=0, rate_mva=200)
    v = Verifier(d, {"enforce_q_limits": True})
    S_eff, pv, pq = v.effective_injections({1: "max"}, v.S)
    V, ok, _ = newton_solve(v.Y, S_eff, v.starts["base"], v.ref, pv, pq, 1e-8, 30)
    assert ok and np.abs(V[1]) > d["Vm"][1] + 1e-6  # held at Qmax it does not need, the voltage rises
    bad = v.q_limit_consistency(V, {1: "max"})
    assert list(bad) == [1] and "above the set-point" in bad[1]
    # the loop with no release allowed: it would have accepted the held state before; now it reports that switching did not settle
    v0 = Verifier(d, {"enforce_q_limits": True, "q_limit_max_releases": 0})
    V0, ok0, _, note0, held0 = v0.solve_with_q_limits(v0.starts["base"], 30, held0={1: "max"})
    assert not ok0 and "did not settle" in note0 and held0 == {1: "max"}
    # with the normal release budget the same start releases the bus and ends consistent
    V1, ok1, _, note1, held1 = v.solve_with_q_limits(v.starts["base"], 30, held0={1: "max"})
    assert ok1 and held1 == {} and v.q_limit_consistency(V1, held1) == {}

    class Stuck(Verifier):  # every start begins with the bus held
        def solve_with_q_limits(self, V0, cap, S=None, held0=None):
            return Verifier.solve_with_q_limits(self, V0, cap, S=S, held0={1: "max"})

    r = Stuck(d, {"enforce_q_limits": True, "q_limit_max_releases": 0}).run(policy_name="classical")
    assert (
        r.status == "unresolved"
        and all(not h["converged"] for h in r.history)
        and "did not settle" in r.history[0]["note"]
    )
    assert "no start converged" in r.unresolved_reason
    r2 = Stuck(d, {"enforce_q_limits": True}).run(policy_name="classical")
    assert r2.status == "verified_safe" and r2.limits["q_limits"]["held"] == {}


def test_branch_check_keeps_unresolved_unresolved():
    r = Verifier(synthetic_case(load_mw=3000, load_mvar=1500), {"branch_check": True}).run(
        genco_vm=np.ones(3), genco_va_deg=np.zeros(3), policy_name="genco_verify"
    )
    assert r.status == "unresolved" and r.flags == {"branch_check": True} and len(r.history) == 4


def _load_fixtures():
    path = Path(__file__).resolve().parents[1] / "results" / "frozen_test" / "regression_fixtures.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_fixtures_from_the_final_evaluation():
    """Replays three cases of the final evaluation when the released data are on disk (skipped otherwise)."""
    fx = _load_fixtures()
    ds = pytest.importorskip("pyarrow.dataset")
    if fx is None or not fx.get("cases"):
        pytest.skip("no fixtures")
    cases = [c for c in fx["cases"] if all(Path(p).exists() for p in c["data_paths"].values())]
    if not cases:
        pytest.skip("released data not on disk")
    for c in cases:
        base = c["data_paths"]
        bus = (
            ds.dataset(base["bus"], format="parquet", partitioning="hive")
            .to_table(filter=ds.field("scenario") == c["base_scenario"])
            .to_pandas()
        )
        br = (
            ds.dataset(base["branch"], format="parquet", partitioning="hive")
            .to_table(filter=ds.field("scenario") == c["base_scenario"])
            .to_pandas()
        )
        d = case_from_tables(bus, br, removed_idx=c["removed_lines"])
        r = Verifier(d, fx["policy"]).run(policy_name=c["policy"])
        assert r.status == c["expected_status"], f"{c['name']}: {r.status} != {c['expected_status']}"
        if c.get("expected_start_used"):
            assert r.start_used == c["expected_start_used"]

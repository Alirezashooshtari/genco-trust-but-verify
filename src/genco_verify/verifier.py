"""The verification workflow: GENCO's predicted state is a starting point, Newton-Raphson decides.

Every case ends in one of three outcomes:

    verified_safe        Newton-Raphson converged from some start and the AC state violates no limit
    verified_violation   it converged and the AC state violates a line rating or a voltage limit
    unresolved           no start converged within the iteration cap; an alarm, never a pass

"Unresolved" is a statement about the solver and its starts, not a proof that the power-flow equations have no
solution.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .powerflow import SN_MVA, build_ybus, max_mismatch, newton_solve

STATUSES = ("verified_safe", "verified_violation", "unresolved")

# The policy frozen before the final evaluation (notebook `frozen_test` stores this dictionary with its results).
DEFAULT_POLICY = {
    "tol_pu": 1e-8,  # convergence threshold on the largest power mismatch
    "max_iter": 30,  # Newton-Raphson iterations per start
    "order": ("genco", "base", "dc", "flat"),
    "line_limit": 1.0,  # overloaded when max(|Sf|, |St|) / rate_a >= line_limit
    "voltage_buses": "pq",  # buses whose voltage limits are checked: "pq" or "nonslack"
    "low_voltage_flag": 0.5,  # flag converged states below this voltage as possible low-voltage solutions
    "recovery": {"max_iter": 100, "homotopy_steps": (0.25, 0.5, 0.75, 1.0)},
    "screen_margin": 0.0454,  # conformal overload margin for GENCO's own (unverified) screen
    "enforce_q_limits": False,  # enforce generator reactive limits (needs qmin_pu / qmax_pu in the case)
    "q_limit_rounds": 15,
    "q_limit_max_releases": 2,
    "branch_check": False,  # run every start and compare the solutions they reach
    "branch_tol_pu": 1e-4,  # states closer than this (complex voltage, p.u.) are the same solution
    "branch_choice": "first",  # reference among several solutions: "first" or "highest_voltage"
}

POLICY_ORDERS = {
    "genco_verify": ("genco", "base", "dc", "flat"),
    "classical": ("base", "dc", "flat"),
    "classical_first": ("base", "dc", "flat", "genco"),
    "classical_recovery": ("base", "dc", "flat", "base_long", "dc_long", "flat_long", "homotopy"),
}


@dataclass
class StartAttempt:
    start: str
    available: bool
    converged: bool = False
    iterations: int = 0
    ms: float = 0.0
    mismatch_start_pu: float = float("nan")
    mismatch_end_pu: float = float("nan")
    min_vm: float = float("nan")
    note: str = ""
    q_limits: Optional[dict] = None


@dataclass
class VerifyResult:
    status: str
    violations: bool
    converged: bool
    start_used: Optional[str]
    iterations: int
    unresolved_reason: Optional[str]
    genco: Optional[dict]  # GENCO's own prediction and the limit checks on it
    state: Optional[dict]  # verified state (vm, va_deg); None when unresolved
    residuals: dict
    limits: dict  # limit checks on the verified state
    history: List[dict]  # every start tried, in order
    timing_ms: dict
    flags: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Verifier:
    """Verification of one contingency case.

    `d` is a case dictionary as built by `case_from_tables`: bus arrays (Pd, Qd, Pg, Qg in MW/Mvar; Vm, Va of the
    pre-contingency solution; PQ, PV, REF masks; min_vm_pu, max_vm_pu; GS, BS) and branch arrays (f, t, Yff, Yft,
    Ytf, Ytt, rate, on, line). Branches with on == False are out of service.
    """

    def __init__(self, d: dict, policy: Optional[dict] = None):
        t0 = time.perf_counter()
        self.policy = dict(DEFAULT_POLICY, **(policy or {}))
        self.d = d
        if self.policy["enforce_q_limits"] and (d.get("qmin_pu") is None or d.get("qmax_pu") is None):
            raise ValueError(
                "enforce_q_limits needs per-bus reactive limits (qmin_pu / qmax_pu); "
                "build the case with case_from_tables(..., gen=generator_table)"
            )
        if self.policy["branch_choice"] not in ("first", "highest_voltage"):
            raise ValueError("branch_choice must be 'first' or 'highest_voltage'")

        on = d["on"]
        self.nb = d["nb"]
        self.Y = build_ybus(
            self.nb, d["f"][on], d["t"][on], d["Yff"][on], d["Yft"][on], d["Ytf"][on], d["Ytt"][on], d["GS"], d["BS"]
        )
        self.S = (d["Pg"] - d["Pd"] + 1j * (d["Qg"] - d["Qd"])) / SN_MVA
        self.ref, self.pv, self.pq = (np.nonzero(d[k])[0] for k in ("REF", "PV", "PQ"))
        self.nonref = np.nonzero(~d["REF"])[0]
        self.rated = on & (d["rate"] > 0)
        self.line_idx = d["line"][self.rated] if "line" in d else np.nonzero(self.rated)[0]

        vm_set = np.where(d["PQ"], 1.0, d["Vm"])
        va_ref = np.where(d["REF"], d["Va"], 0.0)
        va_dc = d.get("Va_dc")
        if va_dc is None or not np.all(np.isfinite(va_dc)):
            va_dc = self.dc_angles()
        self.starts = {
            "base": d["Vm"] * np.exp(1j * np.radians(d["Va"])),
            "flat": vm_set * np.exp(1j * np.radians(va_ref)),
            "dc": vm_set * np.exp(1j * np.radians(va_dc)),
        }
        self.prepare_ms = 1000 * (time.perf_counter() - t0)

    # ------------------------------------------------------------------ states and limit checks
    def dc_angles(self):
        """DC power-flow angles (degrees) of the post-contingency topology."""
        d, on = self.d, self.d["on"]
        f, t, b = d["f"][on], d["t"][on], d["Yft"][on].imag
        B = sp.csr_matrix(
            (np.r_[b, b, -b, -b], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(self.nb, self.nb)
        ).tocsc()
        keep = np.setdiff1d(np.arange(self.nb), self.ref)
        theta = np.zeros(self.nb)
        try:
            theta[keep] = splu(B[keep][:, keep]).solve(self.S.real[keep])
        except (RuntimeError, ValueError):
            pass
        return np.degrees(theta) + float(d["Va"][self.ref[0]])

    def state(self, vm, va_deg):
        """Complex voltages from magnitudes and angles, with PV/slack magnitudes and the slack angle fixed."""
        d = self.d
        vm = np.where(d["PQ"], vm, d["Vm"])
        va = np.where(d["REF"], d["Va"], va_deg)
        return vm * np.exp(1j * np.radians(va))

    def aligned(self, V):
        """The state rotated so that the slack angle equals the case's reference angle."""
        V = np.asarray(V, complex)
        shift = np.radians(float(self.d["Va"][self.ref[0]])) - np.angle(V[self.ref[0]])
        return V * np.exp(1j * shift)

    def loading(self, V):
        """Loading of every rated in-service line, as a fraction of its rating."""
        d = self.d
        Vf, Vt = V[d["f"]], V[d["t"]]
        Sf = Vf * np.conj(d["Yff"] * Vf + d["Yft"] * Vt) * SN_MVA
        St = Vt * np.conj(d["Ytt"] * Vt + d["Ytf"] * Vf) * SN_MVA
        return np.maximum(np.abs(Sf), np.abs(St))[self.rated] / d["rate"][self.rated]

    def mismatch(self, V, held=None, S=None):
        """Largest power mismatch of a state (p.u.); buses in `held` count as PQ buses at their reactive limit."""
        S_eff, _, pq_eff = self.effective_injections(held or {}, self.S if S is None else S)
        return max_mismatch(self.Y, S_eff, V, self.nonref, pq_eff)

    def effective_injections(self, held: dict, S_base):
        """Injections and bus types with the buses in `held` (bus -> "max" | "min") switched to PQ at their limit."""
        switched = np.array(sorted(held), dtype=int)
        pv = np.array([b for b in self.pv if b not in held], dtype=int)
        pq = np.array(sorted(set(self.pq.tolist()) | set(held)), dtype=int)
        S = np.asarray(S_base, complex).copy()
        if len(switched):
            d = self.d
            q_lim = np.array([d["qmax_pu"][b] if held[b] == "max" else d["qmin_pu"][b] for b in switched])
            S[switched] = S_base.real[switched] + 1j * (q_lim - d["Qd"][switched] / SN_MVA)
        return S, pv, pq

    def classify(self, V, switched=()) -> dict:
        """Limit checks on a state. Buses in `switched` are held at a reactive limit, so their voltage is checked too."""
        d, p = self.d, self.policy
        load = self.loading(V)
        over = load >= p["line_limit"]
        vm = np.abs(V)
        checked = d["PQ"].copy() if p["voltage_buses"] == "pq" else ~d["REF"]
        if len(switched):
            checked[np.asarray(list(switched), dtype=int)] = True
        low = checked & (vm < d["min_vm_pu"])
        high = checked & (vm > d["max_vm_pu"])
        vm_checked = np.where(checked, vm, np.inf)
        return {
            "overloaded_lines": self.line_idx[over].astype(int).tolist(),
            "n_over": int(over.sum()),
            "max_loading": float(load.max()) if len(load) else 0.0,
            "max_loading_line": int(self.line_idx[int(np.argmax(load))]) if len(load) else None,
            "low_voltage_buses": np.nonzero(low)[0].tolist(),
            "high_voltage_buses": np.nonzero(high)[0].tolist(),
            "n_low_v": int(low.sum()),
            "n_high_v": int(high.sum()),
            "min_vm": float(vm_checked.min()),
            "min_vm_bus": int(np.argmin(vm_checked)),
            "min_vm_all_buses": float(vm.min()),
            "voltage_checked_buses": int(checked.sum()),
            "buses_at_q_limit": sorted(int(b) for b in switched),
            "violation": bool(over.any() or low.any() or high.any()),
        }

    # ------------------------------------------------------------------ the workflow
    def run(
        self, genco_vm=None, genco_va_deg=None, policy_name="genco_verify", order: Optional[Sequence[str]] = None
    ) -> VerifyResult:
        """Verify the case.

        policy_name selects the start order: "genco_verify" (GENCO first, then the classical starts), "classical"
        (no GENCO), "classical_first" (GENCO as the last fallback) or "classical_recovery" (classical starts, a longer
        iteration cap and a load-stepping homotopy). `order` overrides it.
        """
        t_run = time.perf_counter()
        p = self.policy
        timing = {"prepare_ms": self.prepare_ms}

        genco, V_genco = None, None
        if genco_vm is not None:
            t0 = time.perf_counter()
            V_genco = self.state(np.asarray(genco_vm, float), np.asarray(genco_va_deg, float))
            genco = self._describe_genco(V_genco)
            timing["genco_convert_classify_ms"] = 1000 * (time.perf_counter() - t0)

        history: List[StartAttempt] = []
        solutions = []
        for name in order or POLICY_ORDERS[policy_name]:
            attempt, V, held = self._try_start(name, V_genco)
            history.append(attempt)
            if not attempt.available:
                continue
            timing[f"newton_{name}_ms"] = attempt.ms
            if attempt.converged:
                solutions.append((attempt, V))
                if not p["branch_check"]:
                    break

        t0 = time.perf_counter()
        result = self._result(solutions, history, genco, timing)
        timing["classify_ms"] = 1000 * (time.perf_counter() - t0)
        timing["total_ms"] = 1000 * (time.perf_counter() - t_run)
        return result

    def _describe_genco(self, V):
        p = self.policy
        res_p = np.abs((V * np.conj(self.Y @ V) - self.S).real[self.nonref]) * SN_MVA
        load = self.loading(V)
        return {
            "vm": np.abs(V).tolist(),
            "va_deg": np.degrees(np.angle(V)).tolist(),
            "mismatch_pu": self.mismatch(V),
            "resP_p99_MW": float(np.percentile(res_p, 99)) if len(res_p) else 0.0,
            "resP_max_MW": float(res_p.max()) if len(res_p) else 0.0,
            "flagged_lines_margin": self.line_idx[load >= p["line_limit"] - p["screen_margin"]].astype(int).tolist(),
            **self.classify(V),
        }

    def _try_start(self, name, V_genco):
        p = self.policy
        attempt = StartAttempt(start=name, available=True)
        held = {}
        V = None
        long_cap = name.endswith("_long")
        start = name[:-5] if long_cap else name

        if name == "homotopy":
            t0 = time.perf_counter()
            V, ok, it, attempt.note, held = self._homotopy()
            attempt.ms = 1000 * (time.perf_counter() - t0)
        else:
            V0 = V_genco if start == "genco" else self.starts.get(start)
            if V0 is None:
                attempt.available = False
                attempt.note = "no state for this start"
                return attempt, None, held
            cap = p["recovery"]["max_iter"] if long_cap else p["max_iter"]
            attempt.mismatch_start_pu = self.mismatch(V0)
            t0 = time.perf_counter()
            if p["enforce_q_limits"]:
                V, ok, it, attempt.note, held = self.solve_with_q_limits(V0, cap)
            else:
                V, ok, it = newton_solve(self.Y, self.S, V0, self.ref, self.pv, self.pq, p["tol_pu"], cap)
            attempt.ms = 1000 * (time.perf_counter() - t0)
            if not (ok and np.all(np.isfinite(V))) and not attempt.note:
                attempt.note = f"did not converge in {cap} iterations"

        attempt.iterations = int(it)
        attempt.converged = bool(ok) and bool(np.all(np.isfinite(V)))
        if attempt.converged:
            attempt.mismatch_end_pu = self.mismatch(V, held)
            attempt.min_vm = float(np.abs(V).min())
            if attempt.mismatch_end_pu > 10 * p["tol_pu"]:
                attempt.converged = False
                attempt.note = f"solver reported convergence but the mismatch is {attempt.mismatch_end_pu:.1e} p.u."
        if p["enforce_q_limits"]:
            attempt.q_limits = (
                {"held": {int(b): v for b, v in held.items()}, "rounds": getattr(self, "_last_rounds", None)}
                if attempt.converged
                else None
            )
        return attempt, V, held

    def _result(self, solutions, history, genco, timing) -> VerifyResult:
        p = self.policy
        if not solutions:
            tried = ", ".join(
                f"{h.start} ({h.iterations} it.)" if h.available else f"{h.start} (unavailable)" for h in history
            )
            return VerifyResult(
                status="unresolved",
                violations=False,
                converged=False,
                start_used=None,
                iterations=0,
                unresolved_reason=f"no start converged to {p['tol_pu']:.0e} p.u. within the iteration cap: {tried}",
                genco=genco,
                state=None,
                residuals={},
                limits={},
                history=[asdict(h) for h in history],
                timing_ms=timing,
                flags={"branch_check": bool(p["branch_check"])},
            )

        if len(solutions) > 1 and p["branch_choice"] == "highest_voltage":
            chosen = max(solutions, key=lambda s: float(np.abs(s[1]).min()))
        else:
            chosen = solutions[0]
        attempt, V = chosen
        held = (attempt.q_limits or {}).get("held", {}) if p["enforce_q_limits"] else {}
        limits = self.classify(V, switched=held.keys())
        if p["enforce_q_limits"]:
            limits["q_limits"] = attempt.q_limits
        flags = {
            "low_voltage_branch_suspect": bool(limits["min_vm_all_buses"] < p["low_voltage_flag"]),
            "branch_check": bool(p["branch_check"]),
        }
        if p["branch_check"]:
            flags.update(self._compare_solutions(solutions, chosen, limits))
        return VerifyResult(
            status="verified_violation" if limits["violation"] else "verified_safe",
            violations=limits["violation"],
            converged=True,
            start_used=attempt.start,
            iterations=attempt.iterations,
            unresolved_reason=None,
            genco=genco,
            state={"vm": np.abs(V).tolist(), "va_deg": np.degrees(np.angle(V)).tolist()},
            residuals={"mismatch_start_pu": attempt.mismatch_start_pu, "mismatch_end_pu": attempt.mismatch_end_pu},
            limits=limits,
            history=[asdict(h) for h in history],
            timing_ms=timing,
            flags=flags,
        )

    # ------------------------------------------------------------------ several solutions
    def _compare_solutions(self, solutions, chosen, limits_chosen) -> dict:
        """Group the converged states into distinct solutions and compare their limit decisions."""
        p = self.policy
        V_chosen = self.aligned(chosen[1])
        decisions_chosen = _decisions(limits_chosen)
        starts, groups, distinct = [], [], []  # groups: (aligned V, index into distinct, decisions)
        for attempt, V in solutions:
            Va = self.aligned(V)
            held = (attempt.q_limits or {}).get("held", {}) if p["enforce_q_limits"] else {}
            lim = self.classify(V, switched=held.keys())
            dec = _decisions(lim)
            entry = {
                "start": attempt.start,
                "iterations": attempt.iterations,
                "min_vm_all_buses": lim["min_vm_all_buses"],
                "n_over": lim["n_over"],
                "n_low_v": lim["n_low_v"],
                "n_high_v": lim["n_high_v"],
                "violation": lim["violation"],
                "max_dv_vs_chosen_pu": float(np.abs(Va - V_chosen).max()),
                "max_dvm_vs_chosen_pu": float(np.abs(np.abs(Va) - np.abs(V_chosen)).max()),
                "same_decisions_as_chosen": dec == decisions_chosen,
            }
            gi = next((i for i, (Vg, _, _) in enumerate(groups) if np.abs(Vg - Va).max() <= p["branch_tol_pu"]), None)
            if gi is None:
                groups.append((Va, len(distinct), dec))
                distinct.append(dict(entry, reached_by=[attempt.start], decisions_agree_within_group=True))
                gi = len(distinct) - 1
            else:
                group = distinct[groups[gi][1]]
                group["reached_by"].append(attempt.start)
                if dec != groups[gi][2]:
                    group["decisions_agree_within_group"] = False
            starts.append(dict(entry, solution=gi))
        return {
            "n_starts_converged": len(solutions),
            "n_solutions_found": len(distinct),
            "multiple_solutions": len(distinct) > 1,
            "decisions_differ": any(not s["same_decisions_as_chosen"] for s in starts),
            "branch_choice": p["branch_choice"],
            "starts": starts,
            "solutions": distinct,
        }

    # ------------------------------------------------------------------ generator reactive limits
    def q_limit_consistency(self, V, held: dict, S=None, eps_q=1e-6, eps_v=1e-6):
        """Buses whose reactive-limit conditions fail on state V, with the reason for each."""
        d = self.d
        qmin, qmax = d["qmin_pu"], d["qmax_pu"]
        limited = d["PV"] & (qmax >= qmin)
        q_gen = (V * np.conj(self.Y @ V)).imag + d["Qd"] / SN_MVA
        vm, v_set = np.abs(V), d["Vm"]
        bad = {}
        for b in map(int, np.nonzero(limited)[0]):
            if b in held:
                if held[b] == "max" and vm[b] > v_set[b] + eps_v:
                    bad[b] = "held at the maximum with the voltage above the set-point"
                elif held[b] == "min" and vm[b] < v_set[b] - eps_v:
                    bad[b] = "held at the minimum with the voltage below the set-point"
            elif q_gen[b] > qmax[b] + eps_q:
                bad[b] = "reactive generation above the maximum"
            elif q_gen[b] < qmin[b] - eps_q:
                bad[b] = "reactive generation below the minimum"
        return bad

    def solve_with_q_limits(self, V0, cap, S=None, held0=None):
        """Newton-Raphson with PV-PQ switching at the generator reactive limits.

        A state is accepted only when no further switching is needed and `q_limit_consistency` finds no violated
        condition. Returns (V, converged, total iterations, note, held).
        """
        d, p = self.d, self.policy
        qmin, qmax = d.get("qmin_pu"), d.get("qmax_pu")
        if qmin is None or qmax is None:
            raise ValueError("the case carries no per-bus reactive limits (qmin_pu / qmax_pu)")
        S_base = self.S if S is None else np.asarray(S, complex)
        limited = d["PV"] & (qmax >= qmin)
        v_set, q_load = d["Vm"], d["Qd"] / SN_MVA
        V = np.asarray(V0, complex).copy()
        held = {int(b): side for b, side in (held0 or {}).items() if limited[int(b)]}
        releases, total = {}, 0
        self._last_rounds = None

        for rnd in range(1, p["q_limit_rounds"] + 1):
            switched = np.array(sorted(held), dtype=int)
            S_eff, pv, pq = self.effective_injections(held, S_base)
            V[pv] = v_set[pv] * np.exp(1j * np.angle(V[pv]))
            V, ok, it = newton_solve(self.Y, S_eff, V, self.ref, pv, pq, p["tol_pu"], cap)
            total += int(it)
            if not (ok and np.all(np.isfinite(V))):
                return (
                    V,
                    False,
                    total,
                    f"round {rnd}: no convergence with {len(held)} buses held at a limit",
                    dict(held),
                )

            q_gen = (V * np.conj(self.Y @ V)).imag + q_load
            vm = np.abs(V)
            to_max = [int(b) for b in pv if limited[b] and q_gen[b] > qmax[b] + 1e-6]
            to_min = [int(b) for b in pv if limited[b] and q_gen[b] < qmin[b] - 1e-6]
            back = [
                int(b)
                for b in switched
                if releases.get(int(b), 0) < p["q_limit_max_releases"]
                and (
                    (held[int(b)] == "max" and vm[b] > v_set[b] + 1e-6)
                    or (held[int(b)] == "min" and vm[b] < v_set[b] - 1e-6)
                )
            ]

            if not (to_max or to_min or back):
                bad = self.q_limit_consistency(V, held, S_base)
                if bad:
                    why = "; ".join(f"bus {b}: {r}" for b, r in list(bad.items())[:5])
                    return (
                        V,
                        False,
                        total,
                        (
                            f"switching did not settle: {len(bad)} buses inconsistent after {rnd} "
                            f"round(s) (a limit released {p['q_limit_max_releases']} times keeps "
                            f"cycling): {why}"
                        ),
                        dict(held),
                    )
                self.last_q_limits = {"held": dict(held), "rounds": rnd}
                self._last_rounds = rnd
                return V, True, total, f"{len(held)} buses held at a reactive limit after {rnd} round(s)", dict(held)

            held.update({b: "max" for b in to_max})
            held.update({b: "min" for b in to_min})
            for b in back:
                del held[b]
                releases[b] = releases.get(b, 0) + 1
        return (
            V,
            False,
            total,
            f"switching did not settle in {p['q_limit_rounds']} rounds ({len(held)} buses held)",
            dict(held),
        )

    def _homotopy(self):
        """Load stepping from the pre-contingency state up to the full case."""
        p = self.policy
        V, total, held = self.starts["base"], 0, {}
        for lam in p["recovery"]["homotopy_steps"]:
            S = lam * self.S.real + 1j * np.where(self.d["PQ"], lam * self.S.imag, self.S.imag)
            if p["enforce_q_limits"]:
                V_next, ok, it, _, held = self.solve_with_q_limits(V, p["max_iter"], S=S, held0=held)
            else:
                V_next, ok, it = newton_solve(self.Y, S, V, self.ref, self.pv, self.pq, p["tol_pu"], p["max_iter"])
            total += int(it)
            if not (ok and np.all(np.isfinite(V_next))):
                return V, False, total, f"homotopy failed at load factor {lam}", {}
            V = V_next
        note = "homotopy reached the full case"
        if p["enforce_q_limits"]:
            note += f" ({len(held)} buses held at a reactive limit)"
        return V, True, total, note, held


def _decisions(limits):
    return (set(limits["overloaded_lines"]), set(limits["low_voltage_buses"]), set(limits["high_voltage_buses"]))

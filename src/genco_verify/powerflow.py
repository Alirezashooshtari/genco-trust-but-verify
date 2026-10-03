"""AC power-flow primitives: admittance matrix, power mismatch and Newton-Raphson."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

try:
    from lightsim2grid.algorithm import NR_KLU as _KLU
except ImportError:  # pragma: no cover - older lightsim2grid releases
    try:
        from lightsim2grid.solver import KLUSolver as _KLU
    except ImportError:
        _KLU = None

HAVE_KLU = _KLU is not None
SN_MVA = 100.0


def build_ybus(nb, f, t, Yff, Yft, Ytf, Ytt, GS, BS):
    """Bus admittance matrix from per-branch admittances and bus shunts (p.u.)."""
    rows = np.r_[f, f, t, t, np.arange(nb)]
    cols = np.r_[f, t, f, t, np.arange(nb)]
    vals = np.r_[Yff, Yft, Ytf, Ytt, GS + 1j * BS]
    return sp.csr_matrix((vals, (rows, cols)), shape=(nb, nb))


def max_mismatch(Y, S, V, nonref, pq):
    """Largest power mismatch: P at every non-slack bus, Q at the PQ buses (p.u.)."""
    m = V * np.conj(Y @ V) - S
    p = np.abs(m.real[nonref]).max() if len(nonref) else 0.0
    q = np.abs(m.imag[pq]).max() if len(pq) else 0.0
    return float(max(p, q))


def _newton_scipy(Y, S, V0, pv, pq, tol, max_it):
    """Polar Newton-Raphson with a sparse LU factorisation. Returns (V, converged, iterations)."""
    V = np.asarray(V0, complex).copy()
    pvpq = np.r_[pv, pq]
    n_ang = len(pvpq)
    Yc = Y.tocsc()
    for it in range(max_it + 1):
        m = V * np.conj(Yc @ V) - S
        F = np.r_[m.real[pvpq], m.imag[pq]]
        if len(F) == 0 or np.max(np.abs(F)) < tol:
            return V, True, it
        if it == max_it:
            break
        ib = Yc @ V
        dV = sp.diags(V)
        dS_dVa = 1j * dV @ np.conj(sp.diags(ib) - Yc @ dV)
        dS_dVm = dV @ np.conj(Yc @ sp.diags(V / np.abs(V))) + np.conj(sp.diags(ib)) @ sp.diags(V / np.abs(V))
        J = sp.vstack(
            [
                sp.hstack([dS_dVa[pvpq][:, pvpq].real, dS_dVm[pvpq][:, pq].real]),
                sp.hstack([dS_dVa[pq][:, pvpq].imag, dS_dVm[pq][:, pq].imag]),
            ],
            format="csc",
        )
        try:
            dx = splu(J).solve(-F)
        except (RuntimeError, ValueError):
            return V, False, it
        va, vm = np.angle(V), np.abs(V)
        va[pvpq] += dx[:n_ang]
        vm[pq] += dx[n_ang:]
        V = vm * np.exp(1j * va)
        if not np.all(np.isfinite(V)):
            return V, False, it
    return V, False, max_it


def newton_solve(Y, S, V0, ref, pv, pq, tol, max_it):
    """Newton-Raphson from V0, using lightsim2grid's KLU solver when installed.

    Returns (V, converged, iterations).
    """
    if not HAVE_KLU:
        return _newton_scipy(Y, S, V0, pv, pq, tol, max_it)
    weights = np.zeros(Y.shape[0])
    weights[ref] = 1.0 / len(ref)
    solver = _KLU()
    ok = solver.solve(
        Y.tocsc(),
        np.asarray(V0, complex),
        np.asarray(S, complex),
        np.asarray(ref, np.int32),
        weights,
        np.asarray(pv, np.int32),
        np.asarray(pq, np.int32),
        int(max_it),
        float(tol),
    )
    V = np.array(solver.get_V(), dtype=complex, copy=True)
    return V, bool(ok), int(solver.get_nb_iter())

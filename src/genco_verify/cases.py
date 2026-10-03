"""Building case dictionaries: from the released GENCO tables, or a three-bus test grid."""

from __future__ import annotations

import numpy as np

from .powerflow import SN_MVA

_BUS_COLUMNS = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va", "min_vm_pu", "max_vm_pu", "GS", "BS"]


def case_from_tables(bus, branch, removed_idx=(), gen=None):
    """A case dictionary from one scenario of the released bus and branch tables (pandas DataFrames).

    `removed_idx` lists branch indices to take out of service (the contingency). With the generator table, per-bus
    reactive limits are added so that `enforce_q_limits` can be used.
    """
    bus = bus.sort_values("bus").reset_index(drop=True)
    branch = branch.sort_values("idx").reset_index(drop=True)
    nb = len(bus)
    d = {c: bus[c].to_numpy(np.float64) for c in _BUS_COLUMNS}
    for c in ("PQ", "PV", "REF"):
        d[c] = bus[c].to_numpy(np.float64) > 0.5
    if "Va_dc" in bus:
        d["Va_dc"] = bus["Va_dc"].to_numpy(np.float64)
    d["nb"] = nb
    d["f"] = branch["from_bus"].to_numpy().astype(np.int64)
    d["t"] = branch["to_bus"].to_numpy().astype(np.int64)
    for y in ("Yff", "Yft", "Ytf", "Ytt"):
        d[y] = branch[y + "_r"].to_numpy(np.float64) + 1j * branch[y + "_i"].to_numpy(np.float64)
    d["rate"] = branch["rate_a"].to_numpy(np.float64)
    d["line"] = branch["idx"].to_numpy().astype(np.int64)
    d["on"] = branch["br_status"].to_numpy(np.float64) > 0.5
    if len(removed_idx):
        d["on"] &= ~np.isin(d["line"], np.asarray(removed_idx, dtype=np.int64))
        d.pop("Va_dc", None)  # stale for the new topology; the Verifier recomputes it
    if gen is not None:
        g = gen[gen["in_service"].to_numpy(np.float64) > 0.5]
        buses = g["bus"].to_numpy().astype(int)
        d["qmin_pu"], d["qmax_pu"] = np.zeros(nb), np.zeros(nb)
        np.add.at(d["qmin_pu"], buses, g["min_q_mvar"].to_numpy(np.float64) / SN_MVA)
        np.add.at(d["qmax_pu"], buses, g["max_q_mvar"].to_numpy(np.float64) / SN_MVA)
    return d


def synthetic_case(load_mw=80.0, load_mvar=20.0, rate_mva=120.0, x_pu=0.1):
    """Three buses: slack (1.02 p.u.), PV (1.0 p.u., 40 MW, at most 5 Mvar) and a PQ load; a triangle of lines."""
    nb = 3
    ys = 1 / (0.01 + 1j * x_pu)
    return {
        "nb": nb,
        "f": np.array([0, 1, 0]),
        "t": np.array([1, 2, 2]),
        "Yff": np.full(3, ys),
        "Yft": np.full(3, -ys),
        "Ytf": np.full(3, -ys),
        "Ytt": np.full(3, ys),
        "on": np.ones(3, bool),
        "rate": np.full(3, rate_mva),
        "line": np.arange(3),
        "GS": np.zeros(nb),
        "BS": np.zeros(nb),
        "Pg": np.array([0.0, 40.0, 0.0]),
        "Pd": np.array([0.0, 0.0, load_mw]),
        "Qg": np.zeros(nb),
        "Qd": np.array([0.0, 0.0, load_mvar]),
        "Vm": np.array([1.02, 1.0, 1.0]),
        "Va": np.zeros(nb),
        "PQ": np.array([False, False, True]),
        "PV": np.array([False, True, False]),
        "REF": np.array([True, False, False]),
        "min_vm_pu": np.full(nb, 0.9),
        "max_vm_pu": np.full(nb, 1.1),
        "qmin_pu": np.array([-10.0, -0.5, 0.0]),
        "qmax_pu": np.array([10.0, 0.05, 0.0]),
    }

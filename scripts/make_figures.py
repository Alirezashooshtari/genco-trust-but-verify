"""Draws the five figures of the write-up from the files in results/.

python scripts/make_figures.py            # all figures -> figures/*.png and figures/*.svg
python scripts/make_figures.py 2 5        # selected figures
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

# ---------------------------------------------------------------------------------------------------------- theme
INK, INK2, MUTED = "#1f1f1d", "#57564f", "#8d8c86"
GRID, AXIS = "#ebeae5", "#bdbcb4"
GENCO, GENCO_PALE, GENCO_DARK, TUNED = "#2a78d6", "#b7d3f6", "#184f95", "#4a3aa7"
AC, NEUTRAL = "#1f1f1d", "#a9a8a1"
SAFE, VIOLATION, UNRESOLVED = "#0ca30c", "#fab219", "#d03b3b"

_FONTS = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("TeX Gyre Heros", "Helvetica", "Arial", "Liberation Sans") if f in _FONTS), "DejaVu Sans")
plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.9,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "custom",
        "mathtext.cal": FONT,
        "mathtext.sf": FONT,
        "mathtext.rm": FONT,
        "mathtext.it": FONT,
        "mathtext.bf": FONT,
    }
)
WIDTH = 10.0


def headline(fig, title, subtitle, y=0.975):
    fig.text(0.035, y, title, fontsize=17, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.035, y - 0.075, subtitle, fontsize=11.5, color=INK2, va="top", ha="left", linespacing=1.35)


def panel_title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=12)


def source(fig, text):
    fig.text(0.035, 0.014, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("wrote", f"figures/{name}.png")


# ---------------------------------------------------------------------------------------------------------- 1
def figure1():
    """The verified workflow."""
    fig = plt.figure(figsize=(WIDTH, 4.0))
    headline(
        fig,
        "GENCO proposes, the solver decides",
        "Every contingency ends in one of three outcomes. A case the solver cannot resolve is an alarm, never a pass.",
    )
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.76])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def box(x, y, w, h, title, lines, accent=None):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec=AXIS, lw=1.1)
        )
        if accent:
            ax.plot([x + 1.8, x + w - 1.8], [y + h, y + h], color=accent, lw=5, solid_capstyle="butt")
        ax.text(x + 2.2, y + h - 5.4, title, fontsize=13, fontweight="bold", color=INK, va="center")
        for i, line in enumerate(lines):
            ax.text(x + 2.2, y + h - 11.0 - 4.8 * i, line, fontsize=10.5, color=INK2, va="center")

    def arrow(x0, y0, x1, y1, color=INK2):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4, shrinkA=0, shrinkB=0
            )
        )

    box(1, 17, 17, 27, "Contingency", ["operating point", "with 2 to 20", "lines out"])
    box(22, 17, 22, 27, "GENCO", ["one forward pass:", "every voltage and", "angle in ~70 ms"], accent=GENCO)
    box(
        48,
        17,
        24,
        27,
        "Newton-Raphson",
        ["from GENCO's state,", "then 3 classical starts;", "mismatch below $10^{-8}$ p.u."],
        accent=INK,
    )
    ax.add_patch(
        FancyBboxPatch((22, 1.5), 22, 11, boxstyle="round,pad=0,rounding_size=1.4", fc="#eef4fc", ec=GENCO_PALE, lw=1.1)
    )
    ax.text(24.2, 8.9, "Warning head", fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.text(24.2, 4.8, "“likely unresolved” score", fontsize=9.5, color=INK2, va="center")
    arrow(18, 30.5, 22, 30.5)
    arrow(44, 30.5, 48, 30.5)
    arrow(33, 17, 33, 12.5)

    outcomes = [
        (39, "Verified safe", "AC solution, no limit violated", SAFE),
        (26.5, "Verified violation", "AC solution, a limit violated", VIOLATION),
        (14, "Unresolved", "no start converges: alarm", UNRESOLVED),
    ]
    for y, title, sub, color in outcomes:
        ax.add_patch(
            FancyBboxPatch((76, y - 5), 23, 10, boxstyle="round,pad=0,rounding_size=1.4", fc="white", ec=AXIS, lw=1.1)
        )
        ax.plot([76.15, 76.15], [y - 3.6, y + 3.6], color=color, lw=6, solid_capstyle="butt")
        ax.text(78.4, y + 1.8, title, fontsize=11.5, fontweight="bold", color=INK, va="center")
        ax.text(78.4, y - 2.2, sub, fontsize=9.5, color=INK2, va="center")
        arrow(72, 30.5, 76, y, color=MUTED)
    ax.text(
        48,
        7.5,
        "GENCO alone has no third outcome:\nit returns a plausible grid for every case.",
        fontsize=10,
        color=MUTED,
        va="center",
        style="italic",
        linespacing=1.4,
    )
    save(fig, "fig1_workflow")


# ---------------------------------------------------------------------------------------------------------- 2
def figure2():
    """Storms beyond the collapse point: GENCO still answers."""
    S = pd.read_csv(RES / "storms/sweep.csv")
    Z = np.load(RES / "storms/showcase.npz")
    storm = 910000057
    fold, _, weak, centre = Z[f"{storm}_meta"]
    weak, n_out = int(weak), len(Z[f"{storm}_removed"])
    lam = Z[f"{storm}_lams"]
    v_genco = Z[f"{storm}_vm_genco"][:, weak].astype(float)
    v_ac = Z[f"{storm}_vm_true"][:, weak].astype(float)

    fig = plt.figure(figsize=(WIDTH, 5.1))
    headline(
        fig,
        "GENCO never says “no solution”",
        "120 clustered N-20 outages (“storms”) on the Texas 2,000-bus grid. In 33 of them no Newton start converges at the\n"
        "actual load: the tracked AC solution folds (voltage collapse) below it. GENCO still returns an ordinary-looking grid.",
    )
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.975, top=0.69, bottom=0.17, wspace=0.28, width_ratios=[1.25, 1])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    ok = np.isfinite(v_ac)
    a.axvspan(fold, lam.max() + 0.01, color=UNRESOLVED, alpha=0.07, lw=0)
    a.plot(lam[ok], v_ac[ok], color=AC)
    a.plot(lam, v_genco, color=GENCO)
    a.text(lam[0] + 0.004, v_genco[0] + 0.014, "GENCO", color=GENCO, fontsize=10.5, fontweight="bold", va="bottom")
    a.text(lam[0] + 0.004, v_ac[0] - 0.03, "AC solution", color=AC, fontsize=10.5, fontweight="bold", va="top")
    a.plot([lam[ok][-1]], [v_ac[ok][-1]], "o", ms=8, color=AC, mec="white", mew=2, zorder=5)
    i1 = int(np.argmin(np.abs(lam - 1.0)))
    a.plot([1.0], [v_genco[i1]], "o", ms=8, color=GENCO, mec="white", mew=2, zorder=5)
    a.axvline(1.0, color=MUTED, lw=1, ls=(0, (2, 3)))
    a.text(
        fold - 0.006,
        v_ac[ok][-1],
        f"fold at {fold:.3f}× load:\nthe AC solution ends",
        ha="right",
        va="center",
        fontsize=9.5,
        color=INK2,
    )
    a.text(
        1.008,
        0.80,
        f"actual load: GENCO\nshows {v_genco[i1]:.2f} p.u.;\nno start converges",
        ha="left",
        va="center",
        fontsize=9.5,
        color=INK2,
    )
    a.text(fold + 0.006, 0.985, "no AC solution found", color=UNRESOLVED, fontsize=9.5, va="top")
    a.set_xlim(lam.min() - 0.005, lam.max() + 0.005)
    a.set_ylim(0.5, 1.0)
    a.set_xlabel("Load factor (1 = actual load)")
    a.set_ylabel("Voltage at the weakest bus (p.u.)")
    panel_title(a, "a", f"One storm: {n_out} lines out around bus {int(centre)}")

    A = S[S.at_actual_load & ~S.rescued]
    solved, unresolved = A[A.solvable_at_1], A[~A.solvable_at_1]
    groups = [
        ("AC solution", solved.true_min_vm, AC, "solved storms"),
        ("GENCO", solved.genco_min_vm, GENCO, "solved storms"),
        ("GENCO", unresolved.genco_min_vm, UNRESOLVED, "unresolved"),
    ]
    rng = np.random.default_rng(3)
    for i, (_, vals, color, _) in enumerate(groups):
        v = vals.to_numpy(float)
        b.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=22, color=color, alpha=0.55, lw=0, zorder=2)
        m = float(np.median(v))
        b.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=2.2, zorder=3)
        b.text(i + 0.34, m, f"{m:.3f}", va="center", fontsize=10, color=INK)
    b.axhline(0.9, color=MUTED, lw=1, ls=(0, (2, 3)))
    b.text(2.95, 0.903, "limit", fontsize=9, color=MUTED, ha="right", va="bottom")
    b.set_xticks(range(3))
    b.set_xticklabels([f"{who}\n{grp}" for who, _, _, grp in groups], fontsize=9.5)
    b.set_xlim(-0.5, 2.97)
    b.set_ylim(0.6, 1.0)
    b.set_ylabel("Lowest voltage in the grid (p.u.)")
    panel_title(b, "b", f"At the actual load: {len(solved)} solved, {len(unresolved)} unresolved")
    source(
        fig,
        "Notebook `storms` (lightsim2grid Newton-Raphson); folds confirmed by continuation in notebook `fold_check`. Bars: medians.",
    )
    save(fig, "fig2_never_says_no")


# ---------------------------------------------------------------------------------------------------------- 3
def figure3():
    """Where GENCO goes wrong (clustered outages) and why (damped response to system-wide stress)."""
    C = pd.read_csv(RES / "clustered_outages/storm_cases.csv")
    T = pd.read_csv(RES / "load_response/load_growth_targets.csv")
    B = pd.read_csv(RES / "load_response/load_growth_all_buses.csv")

    fig = plt.figure(figsize=(WIDTH, 5.1))
    headline(
        fig,
        "Where GENCO goes wrong, and why",
        "Its errors grow when outages cluster in one area, the pattern of storms and wildfires. Probing it with small,\n"
        "realistic changes shows why: GENCO largely follows local physics, but under-reacts when the whole system is stressed.",
    )
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.975, top=0.69, bottom=0.2, wspace=0.3, width_ratios=[1.15, 1])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    groups = [(10, "uniform"), (10, "clustered"), (20, "uniform"), (20, "clustered")]
    xs = np.array([0, 1, 2.5, 3.5])
    rng = np.random.default_rng(1)
    medians = {}
    for x, (k, mode) in zip(xs, groups):
        e = C[(C["T"] == k) & (C["mode"] == mode)]["GENCO worst Vm error"].to_numpy(float)
        color = GENCO if mode == "clustered" else NEUTRAL
        a.scatter(x + rng.uniform(-0.2, 0.2, len(e)), e, s=20, color=color, alpha=0.7, lw=0, zorder=2)
        medians[(k, mode)] = m = float(np.median(e))
        a.plot([x - 0.3, x + 0.3], [m, m], color=INK, lw=2.2, zorder=3)
    for k, xc in [(10, 0.5), (20, 3.0)]:
        r = medians[(k, "clustered")] / medians[(k, "uniform")]
        a.text(xc, 0.62, f"×{r:.0f} median error", ha="center", fontsize=10, color=INK, fontweight="bold")
    a.set_yscale("log")
    a.set_ylim(4e-4, 1.0)
    a.set_xticks(xs)
    a.set_xticklabels(["random", "clustered"] * 2)
    for xc, lab in [(0.5, "N-10"), (3.0, "N-20")]:
        a.text(
            xc,
            -0.13,
            lab,
            transform=a.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
    a.set_xlim(-0.55, 4.05)
    a.set_ylabel("Worst bus-voltage error per case (p.u.)")
    panel_title(a, "a", "Random against clustered outages")

    for problem, color, label in [
        (False, "#86b6ef", "ordinary buses"),
        (True, GENCO_DARK, "buses GENCO often gets wrong"),
    ]:
        g = T[(T.group == "wrong pattern") == problem]
        b.scatter(-g.true * 0.01, -g.genco * 0.01, s=10, color=color, alpha=0.4, lw=0, label=label)
    hi = float(np.nanmax(-T.true * 0.01)) * 1.05
    slope = float(B.slope_all_pq.median())
    b.plot([0, hi], [0, hi], color=MUTED, lw=1.2, ls=(0, (2, 3)))
    b.plot([0, hi], [0, slope * hi], color=INK, lw=1.8)
    b.text(
        hi * 0.70,
        hi * 0.70 + hi * 0.02,
        "equal response",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
        rotation=45,
        rotation_mode="anchor",
        transform_rotates_text=True,
    )
    b.annotate(
        f"GENCO responds {slope:.2f}×\nas strongly as AC",
        xy=(hi * 0.66, slope * hi * 0.66),
        xytext=(hi * 0.04, hi * 0.60),
        fontsize=10.5,
        color=INK,
        fontweight="bold",
        va="center",
        arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkA=4, shrinkB=2),
    )
    b.set_xlim(0, hi)
    b.set_ylim(0, hi)
    b.grid(axis="x", visible=True)
    ticks = [0, 0.002, 0.004, 0.006]
    b.set_xticks(ticks)
    b.set_yticks(ticks)
    b.set_xticklabels([f"{t * 1000:.0f}" for t in ticks])
    b.set_yticklabels([f"{t * 1000:.0f}" for t in ticks])
    b.set_xlabel("AC voltage drop ($10^{-3}$ p.u.)")
    b.set_ylabel("GENCO's voltage drop ($10^{-3}$ p.u.)")
    b.legend(loc="upper left", markerscale=2.2)
    panel_title(b, "b", "Every load grows by 1 %")
    source(
        fig,
        "a: notebook `clustered_outages`, 42 to 60 cases per group; bars are medians.   b: notebook `load_response`, 129 scenarios, "
        "linearised from ±2 % changes; line: median slope.",
    )
    save(fig, "fig3_where_and_why")


# ---------------------------------------------------------------------------------------------------------- 4
def figure4():
    """The frozen final evaluation."""
    W = pd.read_csv(RES / "frozen_test/workflows.csv")
    W = W[W.set == "final"]
    alone = pd.read_csv(RES / "frozen_test/genco_alone_by_family.csv").set_index("family")
    families = [
        ("uniform-N10", "random N-10"),
        ("uniform-N20", "random N-20"),
        ("nearest-N10", "clustered N-10"),
        ("nearest-N20", "clustered N-20"),
        ("corridor-N20", "corridor N-20"),
    ]
    classical = W[W.workflow == "classical"]
    counts = classical.groupby(["family", "status"]).size().unstack(fill_value=0)

    fig = plt.figure(figsize=(WIDTH, 5.6))
    headline(
        fig,
        "A frozen test on 870 contingencies that no model or threshold had seen",
        "Protocol fixed in advance; 174 untouched operating points, five outage families. Verification returns AC decisions\n"
        "or an alarm. GENCO alone would have been wrong in 260 of the 748 cases that have a solution.",
    )
    gs = fig.add_gridspec(1, 2, left=0.135, right=0.975, top=0.69, bottom=0.2, wspace=0.32, width_ratios=[1.45, 1])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    y = np.arange(len(families))[::-1]
    for yi, (fam, _) in zip(y, families):
        left = 0
        for status, color in [("verified_safe", SAFE), ("verified_violation", VIOLATION), ("unresolved", UNRESOLVED)]:
            n = int(counts.loc[fam].get(status, 0))
            if n:
                a.barh(yi, n, height=0.56, left=left, color=color, edgecolor="white", linewidth=2)
                if n >= 14:
                    a.text(
                        left + n / 2,
                        yi,
                        f"{n}",
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        color=INK if status == "verified_violation" else "white",
                    )
            left += n
        a.text(
            186,
            yi,
            f"{int(alone.loc[fam, 'cases_with_a_miss'])}",
            ha="center",
            va="center",
            fontsize=11,
            color=GENCO,
            fontweight="bold",
        )
    a.text(
        186,
        len(families) - 0.42,
        "GENCO\nalone\nwrong",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK2,
        linespacing=1.1,
    )
    a.set_yticks(y)
    a.set_yticklabels([lab for _, lab in families])
    a.set_xlim(0, 198)
    a.set_xticks([0, 50, 100, 150, 174])
    a.set_xlabel("Cases (one per operating point)")
    a.grid(axis="y", visible=False)
    a.grid(axis="x", visible=True)
    handles = [
        Line2D([], [], color=c, lw=8, label=lab)
        for c, lab in [(SAFE, "verified safe"), (VIOLATION, "verified violation"), (UNRESOLVED, "unresolved (alarm)")]
    ]
    a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=10, handlelength=1.0)
    panel_title(a, "a", "Outcome of every case")

    g = W[W.workflow == "genco_verify"]
    c_ms = float(classical.end_to_end_ms.median())
    parts = [
        ("graph", float((g.graph_ms + g.transfer_ms).median()), GENCO_PALE),
        ("GENCO\ninference", float(g.inference_ms.median()), GENCO),
    ]
    parts.append(("Newton", float(g.end_to_end_ms.median()) - sum(p[1] for p in parts), INK2))
    b.barh(1, c_ms, height=0.5, color=INK2)
    b.text(c_ms + 2, 1, f"{c_ms:.0f} ms", va="center", fontsize=10.5, color=INK)
    left = 0
    for label, ms, color in parts:
        b.barh(0, ms, height=0.5, left=left, color=color, edgecolor="white", linewidth=2)
        b.text(left + ms / 2, -0.33, label, ha="center", va="top", fontsize=9, color=INK2, linespacing=1.1)
        left += ms
    b.text(left + 2, 0, f"{left:.0f} ms", va="center", fontsize=10.5, color=INK)
    b.set_yticks([1, 0])
    b.set_yticklabels(["classical\nmulti-start", "GENCO +\nverification"])
    b.set_ylim(-1.05, 1.5)
    b.set_xlim(0, 110)
    b.set_xlabel("Median time per case, end to end (ms)")
    b.grid(axis="y", visible=False)
    b.grid(axis="x", visible=True)
    panel_title(b, "b", "Cost: no speed-up on this grid")
    source(
        fig,
        "Notebook `frozen_test`. One T4 GPU (float32, batch 1); lightsim2grid KLU on one CPU core. Both workflows verify the same "
        "748 cases and flag the same 122.",
    )
    save(fig, "fig4_frozen_test")


# ---------------------------------------------------------------------------------------------------------- 5
def figure5():
    """Fine-tuning and the unresolved-case warning."""
    runs = ("seed0", "seed1", "as_run")  # seed0 is the published checkpoint, drawn as bars; the others as markers
    E = {r: pd.read_csv(RES / f"evaluate_finetuned_{r}/endpoints_cases.csv") for r in runs}
    F = {r: pd.read_csv(RES / f"evaluate_finetuned_{r}/endpoint_forgetting.csv") for r in runs}
    SC = pd.read_csv(RES / "analysis/warning_scores.csv")
    P = pd.read_csv(RES / "evaluate_finetuned_seed0/unresolved_warning/per_case_scores.csv")

    fig = plt.figure(figsize=(WIDTH, 5.3))
    headline(
        fig,
        "Teaching GENCO where it fails, and to warn when the solver will",
        "Fine-tuned on 6,737 solver-labelled contingencies, GENCO misses fewer decisions on cases it never saw, in all three\n"
        "training runs, at a price on ordinary ones. A small head on its embeddings flags cases the solver will not resolve.",
    )
    gs = fig.add_gridspec(
        1, 3, left=0.07, right=0.975, top=0.68, bottom=0.17, wspace=0.42, width_ratios=[1.15, 0.85, 1.0]
    )
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))

    m = E["seed0"][E["seed0"].metric == "missed"].set_index("group")
    groups = [
        ("clustered (F1)", "clustered"),
        ("footprint (held out)", "held-out\nconstruction"),
        ("uniform (F1)", "random"),
    ]
    x = np.arange(len(groups))
    a.bar(x - 0.19, [m.loc[g, "released_mean"] for g, _ in groups], 0.36, color=GENCO, label="released")
    a.bar(x + 0.19, [m.loc[g, "fine_tuned_mean"] for g, _ in groups], 0.36, color=TUNED, label="fine-tuned")
    for xi, (g, _) in zip(x, groups):
        r = m.loc[g]
        top = max(r.released_mean, r.fine_tuned_mean)
        a.text(
            xi, top + 0.22, f"{r.diff_mean:+.2f}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold"
        )
    for r in runs[1:]:
        other = E[r][E[r].metric == "missed"].set_index("group")
        a.scatter(x + 0.19, [other.loc[g, "fine_tuned_mean"] for g, _ in groups], s=14, color=INK, zorder=3, lw=0)
    a.scatter([], [], s=14, color=INK, lw=0, label="other runs")
    a.set_xticks(x)
    a.set_xticklabels([lab for _, lab in groups], fontsize=9.5)
    a.set_ylim(0, 7.4)
    a.set_ylabel("Decisions missed per case")
    a.legend(loc="upper right")
    panel_title(a, "a", "Unseen hard cases")

    f0 = F["seed0"]
    xs = np.arange(len(f0))
    b.bar(xs - 0.19, f0.released_median_err * 1000, 0.36, color=GENCO)
    b.bar(xs + 0.19, f0.fine_tuned_median_err * 1000, 0.36, color=TUNED)
    for r in runs[1:]:
        b.scatter(xs + 0.19, F[r].fine_tuned_median_err * 1000, s=14, color=INK, zorder=3, lw=0)
    b.set_xticks(xs)
    b.set_xticklabels(f0.level)
    b.set_ylim(0, 7.4)
    b.set_ylabel("Median worst-bus error ($10^{-3}$ p.u.)")
    panel_title(b, "b", "The price")

    f1 = SC[SC.test == "F1"]
    curves = [
        (f1[(f1.genco_model == "released") & f1.features.str.startswith("BC")], "score", GENCO, "-", "GENCO warning"),
        (
            f1[(f1.features == "A topology") & (f1.classifier == "boosted trees")],
            "score",
            NEUTRAL,
            "-",
            "outage topology only",
        ),
        (P[(P.model == "released") & (P.test == "F1")], "genco_resP_p99_MW", MUTED, (0, (2, 2)), "GENCO residual"),
    ]
    for d, col, color, ls, label in curves:
        fpr, tpr, _ = roc_curve(d.label_unresolved, d[col])
        c.plot(
            fpr,
            tpr,
            color=color,
            ls=ls,
            lw=2.0 if ls == "-" else 1.6,
            label=f"{label}   {roc_auc_score(d.label_unresolved, d[col]):.2f}",
        )
    c.plot([0, 1], [0, 1], color=GRID, lw=1)
    c.set_xlim(0, 1)
    c.set_ylim(0, 1.02)
    c.grid(axis="x", visible=True)
    c.set_xlabel("False-alarm rate")
    c.set_ylabel("Unresolved cases caught")
    c.legend(loc="lower right", fontsize=9, title="AUROC", title_fontsize=9, alignment="right")
    panel_title(c, "c", "Warning, frozen test")
    source(
        fig,
        "Bars: the published run (seed 0); dots: seed 1 and the first run. a: 401 clustered, 153 held-out, 347 random cases. "
        "b: released test scenarios. c: 870 frozen-test cases.",
    )
    save(fig, "fig5_fine_tuning_and_warning")


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5}

if __name__ == "__main__":
    for i in [int(a) for a in sys.argv[1:]] or FIGURES:
        FIGURES[i]()

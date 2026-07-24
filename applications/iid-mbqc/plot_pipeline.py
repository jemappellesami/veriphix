#!/usr/bin/env python3
"""Parameter plots for the i.i.d. pipeline. Pure theory -- no experiment data required.

These characterise the *protocol analysis*, not any particular device: given a measured
test-failure rate, what does the pipeline cost or deliver? They can be regenerated without
re-running any simulation, and they are what the heatmaps are read against.

Figures (PDF, into ``--outdir``):
  A  performance-driven: computation rounds d vs measured q_hat, one curve per eps_target.
  B  cost-driven: achieved eps_total vs measured q_hat, one curve per budget d.
  C  Clopper-Pearson vs Hoeffding: the confidence bound itself, and what it costs in d.
  D  the value of test rounds: d vs s, i.e. how much computation the estimator's slack buys
     back as the benchmarking stage grows.

Two walls appear on the q_hat axis and the distinction matters:
  * the hard wall ``q < alpha/k`` (0.25 at k=2, c=0) -- where majority voting stops helping
    even with a perfectly known noise rate;
  * the effective wall for a given s -- where the *upper bound* q_U crosses alpha/k. A tile
    whose raw rate clears the hard wall can still be uncertifiable because s was too small.
The gap between them is the price of finite statistics, and it closes as s grows.

Usage:
    python applications/iid-mbqc/plot_pipeline.py
    python applications/iid-mbqc/plot_pipeline.py --rounds 10000 --eps-target 1e-9
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iid_pipeline import (
    K_TESTS,
    admissible_q_threshold,
    clopper_pearson_upper,
    computation_error_upper,
    cost_driven,
    hoeffding_upper,
    optimal_allocation,
    required_rounds,
)

_USETEX = shutil.which("latex") is not None
matplotlib.rcParams.update(
    {
        "text.usetex": _USETEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
)
if _USETEX:
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}\usepackage{amsmath}"

# ── Edit here instead of the command line ───────────────────────────────────────
EPS_TARGETS = [1e-2, 1e-4, 1e-6, 1e-9]  # figure A
BUDGETS = [21, 51, 101, 501]  # figure B (odd)
SHOT_COUNTS = [1_000, 10_000, 100_000]  # figures C, D
Q_HAT_MARKS = [0.01, 0.05, 0.10, 0.15]  # figure D
Q_POINTS = 60
OUTDIR = Path(__file__).resolve().parent / "figures"
# ─────────────────────────────────────────────────────────────────────────────


def _eps_latex(eps: float) -> str:
    return rf"10^{{{round(math.log10(eps))}}}"


def _q_grid(threshold: float) -> np.ndarray:
    return np.linspace(1e-4, threshold * 0.995, Q_POINTS)


def _y_from_q(q_hat: float, rounds: int) -> int:
    """The integer count a rate implies -- Clopper-Pearson is a function of Y, not of q."""
    return round(q_hat * rounds)


def _effective_wall(rounds: int, eps_bench: float, threshold: float) -> float:
    """Largest q_hat whose Clopper-Pearson bound still clears the hard wall."""
    lo, hi = 0.0, threshold
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if clopper_pearson_upper(_y_from_q(mid, rounds), rounds, eps_bench) < threshold:
            lo = mid
        else:
            hi = mid
    return lo


def _draw_walls(ax, threshold: float, effective: float | None, rounds: int | None) -> None:
    ax.axvline(threshold, color="red", lw=1.4, zorder=4)
    ax.annotate(
        rf"$q=\alpha/k={threshold:.3f}$",
        xy=(threshold, 0.5),
        xycoords=("data", "axes fraction"),
        rotation=90,
        ha="right",
        va="center",
        fontsize=8,
        color="red",
    )
    if effective is not None and effective < threshold * 0.999:
        ax.axvline(effective, color="red", lw=1.2, ls="--", zorder=4)
        ax.annotate(
            rf"effective wall, $s={rounds}$",
            xy=(effective, 0.5),
            xycoords=("data", "axes fraction"),
            rotation=90,
            ha="right",
            va="center",
            fontsize=7,
            color="red",
        )


def figure_performance(rounds: int, k: int, c: float, outdir: Path) -> Path:
    threshold = admissible_q_threshold(k=k, c=c)
    q_grid = _q_grid(threshold)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")

    for i, eps in enumerate(EPS_TARGETS):
        ys = []
        for q_hat in q_grid:
            res = optimal_allocation(_y_from_q(q_hat, rounds), rounds, eps, k=k, c=c)
            ys.append(res.d if res.d is not None else np.nan)
        ax.plot(
            q_grid,
            ys,
            color=cmap(0.12 + 0.72 * i / max(len(EPS_TARGETS) - 1, 1)),
            lw=1.8,
            label=rf"$\epsilon_{{\mathrm{{target}}}}={_eps_latex(eps)}$",
        )

    _draw_walls(ax, threshold, _effective_wall(rounds, min(EPS_TARGETS) * 1e-2, threshold), rounds)
    ax.set_yscale("log")
    ax.set_xlim(0, threshold * 1.05)
    ax.set_xlabel(r"measured test-failure rate $\hat q$")
    ax.set_ylabel(r"computation rounds $d$")
    ax.set_title(rf"Performance-driven: rounds needed at fixed target ($s={rounds}$, $k={k}$, $c={c}$)")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    out = outdir / f"A_performance_s{rounds}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


def figure_cost(rounds: int, eps_bench: float, k: int, c: float, outdir: Path) -> Path:
    threshold = admissible_q_threshold(k=k, c=c)
    q_grid = _q_grid(threshold)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("plasma")

    for i, d in enumerate(BUDGETS):
        ys = [
            cost_driven(_y_from_q(q, rounds), rounds, d, eps_bench, k=k, c=c).epsilon_total
            for q in q_grid
        ]
        ax.plot(
            q_grid,
            ys,
            color=cmap(0.08 + 0.66 * i / max(len(BUDGETS) - 1, 1)),
            lw=1.8,
            label=rf"$d={d}$",
        )

    ax.axhline(eps_bench, color="grey", ls=":", lw=1.2, zorder=3)
    ax.annotate(
        rf"$\epsilon_{{\mathrm{{bench}}}}={_eps_latex(eps_bench)}$ floor",
        xy=(0.02, eps_bench),
        xycoords=("axes fraction", "data"),
        fontsize=7,
        color="grey",
        va="bottom",
    )
    _draw_walls(ax, threshold, _effective_wall(rounds, eps_bench, threshold), rounds)
    ax.set_yscale("log")
    ax.set_xlim(0, threshold * 1.05)
    ax.set_xlabel(r"measured test-failure rate $\hat q$")
    ax.set_ylabel(r"achieved $\epsilon_{\mathrm{total}}$")
    ax.set_title(rf"Cost-driven: correctness at fixed budget ($s={rounds}$, $k={k}$, $c={c}$)")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    out = outdir / f"B_cost_s{rounds}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


def figure_bounds(eps_bench: float, eps_target: float, k: int, c: float, outdir: Path) -> Path:
    """Why Clopper-Pearson: the bound itself (left) and what the difference costs (right)."""
    threshold = admissible_q_threshold(k=k, c=c)
    q_grid = _q_grid(threshold)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.3))
    cmap = plt.get_cmap("cividis")

    for i, s in enumerate(SHOT_COUNTS):
        color = cmap(0.1 + 0.7 * i / max(len(SHOT_COUNTS) - 1, 1))
        cp = [clopper_pearson_upper(_y_from_q(q, s), s, eps_bench) for q in q_grid]
        ho = [hoeffding_upper(_y_from_q(q, s), s, eps_bench) for q in q_grid]
        ax_l.plot(q_grid, cp, color=color, lw=1.8, label=rf"CP, $s={s}$")
        ax_l.plot(q_grid, ho, color=color, lw=1.4, ls="--", label=rf"Hoeffding, $s={s}$")

        def d_of(bound_fn, s=s):
            out = []
            for q in q_grid:
                p_u = computation_error_upper(bound_fn(_y_from_q(q, s), s, eps_bench), k=k, c=c)
                d = required_rounds(p_u, eps_target - eps_bench) if p_u < 0.5 else None
                out.append(d if d is not None else np.nan)
            return out

        ax_r.plot(q_grid, d_of(clopper_pearson_upper), color=color, lw=1.8, label=rf"CP, $s={s}$")
        ax_r.plot(q_grid, d_of(hoeffding_upper), color=color, lw=1.4, ls="--", label=rf"Hoeffding, $s={s}$")

    ax_l.plot(q_grid, q_grid, color="black", lw=0.9, ls=":", label=r"$\hat q$ (no correction)")
    ax_l.axhline(threshold, color="red", lw=1.2)
    ax_l.set_xlabel(r"measured $\hat q$")
    ax_l.set_ylabel(r"upper bound $q_U$")
    ax_l.set_title(rf"Confidence bound ($\epsilon_{{\mathrm{{bench}}}}={_eps_latex(eps_bench)}$)")
    ax_l.grid(True, alpha=0.25, lw=0.5)
    ax_l.legend(fontsize=7, framealpha=0.9)

    ax_r.set_yscale("log")
    ax_r.set_xlabel(r"measured $\hat q$")
    ax_r.set_ylabel(r"computation rounds $d$")
    ax_r.set_title(rf"Cost of the looser bound ($\epsilon_{{\mathrm{{target}}}}={_eps_latex(eps_target)}$)")
    ax_r.grid(True, which="both", alpha=0.25, lw=0.5)
    ax_r.legend(fontsize=7, framealpha=0.9)

    fig.suptitle("Hoeffding's slack is additive and ignores the observation; CP adapts to it", fontsize=10)
    plt.tight_layout()
    out = outdir / "C_bounds_cp_vs_hoeffding.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


def figure_shots(eps_target: float, k: int, c: float, outdir: Path) -> Path:
    """What test rounds buy: d as the benchmarking stage grows, against the s->inf floor."""
    s_grid = np.unique(np.round(np.logspace(2, 6, 34)).astype(int))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")

    for i, q_hat in enumerate(Q_HAT_MARKS):
        color = cmap(0.1 + 0.75 * i / max(len(Q_HAT_MARKS) - 1, 1))
        ys = []
        for s in s_grid:
            res = optimal_allocation(_y_from_q(q_hat, int(s)), int(s), eps_target, k=k, c=c)
            ys.append(res.d if res.d is not None else np.nan)
        ax.plot(s_grid, ys, color=color, lw=1.8, marker="o", ms=2.5, label=rf"$\hat q={q_hat}$")

        # s -> infinity: the noise rate is known exactly, only the vote costs anything.
        p_exact = computation_error_upper(q_hat, k=k, c=c)
        floor = required_rounds(p_exact, eps_target) if p_exact < 0.5 else None
        if floor is not None:
            ax.axhline(floor, color=color, lw=0.9, ls=":", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"test rounds $s$ (benchmarking stage)")
    ax.set_ylabel(r"computation rounds $d$")
    ax.set_title(
        rf"What test rounds buy back ($\epsilon_{{\mathrm{{target}}}}={_eps_latex(eps_target)}$, $k={k}$, $c={c}$)"
    )
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.text(0.5, 0.005, r"dotted: the $s\to\infty$ floor, where $q$ is known exactly", ha="center", fontsize=7)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    out = outdir / "D_value_of_test_rounds.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=10_000, help="s for figures A and B")
    parser.add_argument("--eps-target", type=float, default=1e-6)
    parser.add_argument("--eps-bench", type=float, default=1e-9)
    parser.add_argument("--k", type=int, default=K_TESTS)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    threshold = admissible_q_threshold(k=args.k, c=args.c)
    print(f"k={args.k} c={args.c} -> hard wall q < {threshold:.6f}")
    print(f"effective wall at s={args.rounds}: q_hat < {_effective_wall(args.rounds, args.eps_bench, threshold):.6f}")

    figure_performance(args.rounds, args.k, args.c, args.outdir)
    figure_cost(args.rounds, args.eps_bench, args.k, args.c, args.outdir)
    figure_bounds(args.eps_bench, args.eps_target, args.k, args.c, args.outdir)
    figure_shots(args.eps_target, args.k, args.c, args.outdir)


if __name__ == "__main__":
    main()

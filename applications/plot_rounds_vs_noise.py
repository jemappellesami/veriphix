#!/usr/bin/env python3
"""Round cost and security as a function of the measured circuit-level failure rate.

The heatmap scripts answer "how often does an honest noisy device fail a test round at this
circuit dimension?" and stop there. This script answers the two questions a user of the
machine actually asks once that number is known:

  performance-driven   fixed epsilon -> how many rounds N = d+s must I run?
  cost-driven          fixed budget N -> how small an epsilon can I claim?

Both are read off ``veriphix.util_rounds``: the measured ``p_failed_round`` enters the round
optimiser as ``rho_min``, the tolerated fraction of failed test rounds, and nothing else about
the device does. The protocol side (bqp error c, detection rate) is fixed per figure.

Two structural facts drive the shape of every curve here:

  * There is a hard wall at ``p_max = alpha * detection_rate`` (0.25 for c=0, 0.2222 for
    c=0.1). Above it no design tolerates the noise at any budget, so N diverges and the
    cost-driven epsilon hits 1. This is the same threshold the heatmap scripts draw as the
    red ``rho = 0.25`` iso-contour, so the figures can be read side by side.
  * Below the wall, ``N ~ K ln(1/eps) / (alpha - p/detection_rate)^2``. The log(1/eps) factor
    is the cheap one; the noise cost is the inverse-square blow-up as p approaches the wall.
    K is fitted from the computed curve and drawn as a dashed overlay, not hardcoded.

Figures (PDF, into ``--outdir``):
  A  N vs p, one curve per epsilon target.
  B  epsilon vs p, one curve per round budget N.
  C  per-tile (width, depth) map of N or epsilon, if ``--csv`` is given -- the benchmark
     heatmap re-read through the round optimiser. Tiles past the wall are hatched.

The experiment CSVs are only read, never written. Column names vary across benchmark
families (``p_depol``/``p_ent``, ``width``/``n``, ``depth``/``t``); all variants are accepted.

Usage:
    python applications/plot_rounds_vs_noise.py
    python applications/plot_rounds_vs_noise.py \
        --csv applications/benchmark-stim-msi-bro/benchmark_bro_results_p1.0e-04_s1000.csv \
        --bqp-error 0.1
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriphix.util_rounds import (
    alpha_from_c,
    min_epsilon_under_budget,
    optimize_with_robustness_constraint_over_lambda,
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
_HERE = Path(__file__).resolve().parent

BQP_ERROR = 0.1
DETECTION_RATE = 0.5  # FK12-analogue bipartite traps: detection rate 1/2

EPSILON_TARGETS = [1e-2, 1e-4, 1e-6, 1e-9]  # figure A: one curve each
ROUND_BUDGETS = [500, 1000, 2000, 5000]  # figure B: one curve each

# p grid, as a fraction of the wall p_max = alpha * detection_rate. Stops short of 1: the
# round count diverges there, so the last decade is drawn by the asymptote, not by samples.
# The low end sits below the cleanest measured tiles so the tile map interpolates rather
# than clamping.
P_FRAC_LO, P_FRAC_HI, P_POINTS = 0.002, 0.96, 28

# Bisection floor for the cost-driven search. Low enough that a generous budget at low noise
# is not reported as a flat line pinned to the floor.
EPSILON_FLOOR = 1e-30

# Optimiser grids. 200 x 11 lands within ~3% of 2000 x 21 at a twentieth of the cost, which
# is invisible on a log axis; raise for a final figure.
N_GRID_DELTA = 200
N_GRID_LAMBDA = 11

TILE_METRIC = "rounds"  # figure C: "rounds" (fixed epsilon) or "epsilon" (fixed budget)
TILE_EPSILON = 1e-6
TILE_BUDGET = 2000

OUTDIR = _HERE / "rounds_vs_noise"
# ─────────────────────────────────────────────────────────────────────────────

_WIDTH_COLS = ("width", "n")
_DEPTH_COLS = ("depth", "t")
_NOISE_COLS = ("p_depol", "p_ent")


def _pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _eps_latex(eps: float) -> str:
    return rf"10^{{{round(math.log10(eps))}}}"


def _solve(fn, **kwargs):
    """Design, or None past the wall -- the optimisers raise rather than return infeasible."""
    try:
        return fn(**kwargs)
    except (ValueError, RuntimeError):
        return None


def _p_grid(p_max: float) -> np.ndarray:
    return np.linspace(P_FRAC_LO * p_max, P_FRAC_HI * p_max, P_POINTS)


def rounds_vs_p(c: float, detection_rate: float, epsilon: float, p_values: np.ndarray) -> np.ndarray:
    """Minimum N = d+s tolerating w/s >= p at fixed security target; NaN where infeasible."""
    out = np.full(p_values.shape, np.nan)
    for i, p in enumerate(p_values):
        res = _solve(
            optimize_with_robustness_constraint_over_lambda,
            c=c,
            detection_rate=detection_rate,
            epsilon_target=epsilon,
            rho_min=float(p),
            n_grid_delta=N_GRID_DELTA,
            n_grid_lambda=N_GRID_LAMBDA,
        )
        if res is not None:
            out[i] = res.d + res.s
    return out


def epsilon_vs_p(c: float, detection_rate: float, budget: int, p_values: np.ndarray) -> np.ndarray:
    """Smallest reachable epsilon at fixed round budget; NaN where no design fits."""
    out = np.full(p_values.shape, np.nan)
    for i, p in enumerate(p_values):
        res = _solve(
            min_epsilon_under_budget,
            c=c,
            detection_rate=detection_rate,
            budget=budget,
            rho_min=float(p),
            n_grid_delta=N_GRID_DELTA,
            n_grid_lambda=N_GRID_LAMBDA,
            epsilon_lo=EPSILON_FLOOR,
        )
        if res is not None:
            out[i] = res.epsilon_target
    return out


def _fit_k(p_values: np.ndarray, n_values: np.ndarray, alpha: float, detection_rate: float, epsilon: float) -> float:
    """K in N ~ K ln(1/eps) / (alpha - p/detection_rate)^2, fitted to the computed curve.

    Median rather than least squares: K drifts slowly (~16 to ~20 over the range) because of
    the integer round counts and the floor in w, and the median keeps the overlay centred
    instead of letting the diverging tail dominate.
    """
    ok = ~np.isnan(n_values)
    if not ok.any():
        return float("nan")
    gap = alpha - p_values[ok] / detection_rate
    return float(np.median(n_values[ok] * gap**2 / math.log(1.0 / epsilon)))


def _draw_wall(ax, p_max: float) -> None:
    ax.axvline(p_max, color="red", lw=1.4, ls="-", zorder=4)
    ax.annotate(
        rf"$p_{{\max}}=\alpha\,\lambda_{{\mathrm{{det}}}}={p_max:.4f}$",
        xy=(p_max, 0.5),
        xycoords=("data", "axes fraction"),
        rotation=90,
        ha="right",
        va="center",
        fontsize=8,
        color="red",
    )


def _draw_measured(ax, measured: np.ndarray) -> list:
    """Where the benchmarked tiles actually sit on the p axis.

    A per-tile rug saturates into a solid bar at the thousand-tile scale of the CSVs, so the
    distribution is summarised instead: the p10-p90 span as a band, the median as a line.
    """
    if measured.size == 0:
        return []
    lo, med, hi = (float(np.quantile(measured, q)) for q in (0.10, 0.50, 0.90))
    band = ax.axvspan(lo, hi, color="tab:blue", alpha=0.12, zorder=0, label=r"measured tiles (p10--p90)")
    line = ax.axvline(med, color="tab:blue", lw=1.2, ls=":", zorder=4, label=rf"median tile $p={med:.3f}$")
    return [band, line]


def plot_rounds(
    c: float, detection_rate: float, p_values: np.ndarray, curves: dict, measured: np.ndarray, outdir: Path
) -> Path:
    alpha = alpha_from_c(c)
    p_max = alpha * detection_rate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")

    for i, (epsilon, n_values) in enumerate(curves.items()):
        color = cmap(0.12 + 0.72 * i / max(len(curves) - 1, 1))
        ax.plot(p_values, n_values, color=color, lw=1.8, marker="o", ms=2.5, label=rf"$\epsilon={_eps_latex(epsilon)}$")
        k = _fit_k(p_values, n_values, alpha, detection_rate, epsilon)
        if not math.isnan(k):
            dense = np.linspace(p_values[0], 0.995 * p_max, 400)
            ax.plot(dense, k * math.log(1.0 / epsilon) / (alpha - dense / detection_rate) ** 2, color=color, lw=0.9, ls="--")
            print(f"  epsilon={epsilon:.0e}: fitted K={k:.1f}")

    _draw_wall(ax, p_max)
    _draw_measured(ax, measured)

    ax.set_yscale("log")
    ax.set_xlim(0.0, p_max * 1.04)
    ax.set_xlabel(r"honest failure rate per test round $p_{\mathrm{failed\ round}}$")
    ax.set_ylabel(r"total rounds $N=d+s$")
    ax.set_title(
        rf"Performance-driven: rounds needed at fixed security "
        rf"($c={c}$, $\lambda_{{\mathrm{{det}}}}={detection_rate}$)"
    )
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.text(0.5, 0.005, r"dashed: $N\simeq K\ln(1/\epsilon)/(\alpha-p/\lambda_{\mathrm{det}})^2$", ha="center", fontsize=7)
    plt.tight_layout(rect=(0, 0.02, 1, 1))

    outpath = outdir / f"rounds_vs_p_bqp{c}_det{detection_rate}.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved -> {outpath}")
    return outpath


def plot_epsilon(
    c: float, detection_rate: float, p_values: np.ndarray, curves: dict, measured: np.ndarray, outdir: Path
) -> Path:
    alpha = alpha_from_c(c)
    p_max = alpha * detection_rate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("plasma")

    for i, (budget, eps_values) in enumerate(curves.items()):
        color = cmap(0.08 + 0.66 * i / max(len(curves) - 1, 1))
        ax.plot(p_values, eps_values, color=color, lw=1.8, marker="o", ms=2.5, label=rf"$N={budget}$")

    _draw_wall(ax, p_max)
    _draw_measured(ax, measured)

    ax.set_yscale("log")
    ax.set_xlim(0.0, p_max * 1.04)
    ax.set_xlabel(r"honest failure rate per test round $p_{\mathrm{failed\ round}}$")
    ax.set_ylabel(r"best reachable security $\epsilon$")
    ax.set_title(
        rf"Cost-driven: security reachable at fixed budget "
        rf"($c={c}$, $\lambda_{{\mathrm{{det}}}}={detection_rate}$)"
    )
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.text(
        0.5,
        0.005,
        r"gaps: no design with $d+s\leq N$ tolerates $w/s\geq p$",
        ha="center",
        fontsize=7,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 1))

    outpath = outdir / f"epsilon_vs_p_bqp{c}_det{detection_rate}.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved -> {outpath}")
    return outpath


def plot_tiles(
    df: pd.DataFrame,
    c: float,
    detection_rate: float,
    p_values: np.ndarray,
    curve: np.ndarray,
    metric: str,
    setting: str,
    csv_name: str,
    outdir: Path,
) -> Path | None:
    """Re-colour the (width, depth) grid by the round optimiser's verdict for each tile.

    The tile value is interpolated in log space off the 1-D curve rather than re-solved per
    tile: the curve is the same function of p, and a per-tile solve would multiply the run
    time by the number of tiles for a difference well below the optimiser's own grid error.
    """
    w_col, d_col = _pick_col(df, _WIDTH_COLS), _pick_col(df, _DEPTH_COLS)
    if w_col is None or d_col is None or "p_failed_round" not in df.columns:
        print(f"   skipping tile map: {csv_name} has no (width, depth, p_failed_round) columns")
        return None

    ok = ~np.isnan(curve)
    if ok.sum() < 2:
        print("   skipping tile map: curve has too few feasible points")
        return None

    grid_p = (
        df.pivot_table(index=w_col, columns=d_col, values="p_failed_round", aggfunc="mean")
        .sort_index(ascending=False)
        .sort_index(axis=1)
    )
    widths = list(grid_p.index)
    depths = list(grid_p.columns)
    p_tile = grid_p.to_numpy(dtype=float)

    # np.interp needs an increasing x; the curve is monotone in p on both metrics. Its default
    # end-clamping is what we want below the grid (the curve is nearly flat there, and those
    # are the cleanest tiles), but above the last feasible p the tile is not certifiable at
    # all and must be masked rather than clamped.
    values = 10.0 ** np.interp(p_tile, p_values[ok], np.log10(curve[ok]))
    infeasible = ~np.isnan(p_tile) & (p_tile > p_values[ok][-1])
    values[infeasible | np.isnan(p_tile)] = np.nan

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(values, cmap=cmap, aspect="auto", norm=matplotlib.colors.LogNorm())

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if infeasible[i, j]:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="xxx", edgecolor="grey", lw=0.0
                    )
                )

    d_ticks = [i for i, d in enumerate(depths) if d % 5 == 0]
    w_ticks = [i for i, w in enumerate(widths) if w % 5 == 0]
    ax.set_xticks(d_ticks or range(len(depths)))
    ax.set_xticklabels([str(depths[i]) for i in (d_ticks or range(len(depths)))])
    ax.set_yticks(w_ticks or range(len(widths)))
    ax.set_yticklabels([str(widths[i]) for i in (w_ticks or range(len(widths)))])
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("width (logical wires)")

    if metric == "rounds":
        ax.set_title(rf"Rounds $N=d+s$ needed per tile at $\epsilon={setting}$")
        label = r"total rounds $N=d+s$"
    else:
        ax.set_title(rf"Best reachable $\epsilon$ per tile at $N={setting}$")
        label = r"security $\epsilon$"
    fig.colorbar(im, ax=ax, label=label)
    fig.text(0.5, 0.005, "hatched / grey: past the wall, not certifiable at any budget", ha="center", fontsize=7)
    plt.tight_layout(rect=(0, 0.02, 1, 1))

    outpath = outdir / f"tiles_{metric}_{Path(csv_name).stem}_bqp{c}.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved -> {outpath}")
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="*", default=[], help="benchmark result CSVs, for the rug and the tile map")
    parser.add_argument("--bqp-error", type=float, default=BQP_ERROR)
    parser.add_argument("--detection-rate", type=float, default=DETECTION_RATE)
    parser.add_argument("--tile-metric", choices=["rounds", "epsilon", "none"], default=TILE_METRIC)
    parser.add_argument("--tile-epsilon", type=float, default=TILE_EPSILON)
    parser.add_argument("--tile-budget", type=int, default=TILE_BUDGET)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    c, detection_rate = args.bqp_error, args.detection_rate
    alpha = alpha_from_c(c)
    p_max = alpha * detection_rate
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"c={c}  alpha={alpha:.6f}  detection_rate={detection_rate}  wall p_max={p_max:.6f}")

    frames = {Path(p).name: pd.read_csv(Path(p), comment="#") for p in args.csv}
    measured = np.concatenate(
        [f["p_failed_round"].to_numpy(dtype=float) for f in frames.values() if "p_failed_round" in f.columns]
    ) if frames else np.array([])
    if measured.size:
        print(f"measured p_failed_round over {measured.size} tiles: "
              f"min={measured.min():.4f} median={np.median(measured):.4f} max={measured.max():.4f} "
              f"({100 * np.mean(measured >= p_max):.0f}% past the wall)")

    p_values = _p_grid(p_max)

    print("Figure A: rounds vs p")
    round_curves = {eps: rounds_vs_p(c, detection_rate, eps, p_values) for eps in EPSILON_TARGETS}
    plot_rounds(c, detection_rate, p_values, round_curves, measured, args.outdir)

    print("Figure B: epsilon vs p")
    eps_curves = {n: epsilon_vs_p(c, detection_rate, n, p_values) for n in ROUND_BUDGETS}
    plot_epsilon(c, detection_rate, p_values, eps_curves, measured, args.outdir)

    if args.tile_metric != "none" and frames:
        print("Figure C: per-tile map")
        if args.tile_metric == "rounds":
            curve = round_curves.get(args.tile_epsilon)
            if curve is None:
                curve = rounds_vs_p(c, detection_rate, args.tile_epsilon, p_values)
            setting = _eps_latex(args.tile_epsilon)
        else:
            curve = eps_curves.get(args.tile_budget)
            if curve is None:
                curve = epsilon_vs_p(c, detection_rate, args.tile_budget, p_values)
            setting = str(args.tile_budget)
        for name, frame in frames.items():
            plot_tiles(
                frame, c, detection_rate, p_values, curve, args.tile_metric, setting, name, args.outdir
            )


if __name__ == "__main__":
    main()

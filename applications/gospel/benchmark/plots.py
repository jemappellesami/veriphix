#!/usr/bin/env python3
"""gospel_plots-v2-select: heatmaps for a selected subset of (p_ent, bqp_error) results.

Edit FILTER_BQP and FILTER_P_ENT below to choose which combinations to plot.

Frontier lines show the boundary between safe/unsafe tiles for every combination
of round budget N and security parameter epsilon.  Color family encodes epsilon
(Blues / Greens / Purples), shade encodes N (light→dark as N increases).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, generic_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from veriphix.util_rounds import maximize_robustness_under_budget

matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amssymb}\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,
})

# ── Selection ─────────────────────────────────────────────────────────────────
FILTER_BQP   = [0.1]
FILTER_P_ENT = [6e-4, 2e-3]
# ─────────────────────────────────────────────────────────────────────────────

# ── Frontier sweep ────────────────────────────────────────────────────────────
# Per-p_ent sweep config: maps nominal p_ent → (N_values, epsilon_values).
# Keys are matched to actual data p_ent values in log-space (same as FILTER_P_ENT).
# None is the fallback used for any p_ent not explicitly listed.
FRONTIER_SWEEP: dict[float | None, tuple[list[int], list[float]]] = {
    6e-4: ([1800, 1850, 1900, 1950], [1e-7, 1e-8]),
    2e-3: ([4000, 5000, 6000],       [1e-7, 1e-8]),
    None: ([2000, 2500, 3000],       [1e-3, 1e-4]),  # default
}
FRONTIER_DETECT = 0.5   # detection_rate (FK12 / RandomTraps, hardcoded)

# Color families per epsilon value (all contrast with YlOrRd heatmap).
_CMAP_NAMES = ["Blues", "Greens", "Purples", "Oranges", "Greys"]
# ─────────────────────────────────────────────────────────────────────────────

OUTDIR_DISCRETE   = Path("applications/gospel/benchmark/plots-FK")
OUTDIR_CONTINUOUS = Path("applications/gospel/benchmark/plots-FK")
METRIC       = "p_failed_round"
METRIC_LABEL = "Average test round failure rate"
INTERP_FACTOR = 10


# ── Data helpers ──────────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[METRIC] = df["nr_failed_test_rounds"] / df["test_rounds"]
    return (
        df.groupby(["p_ent", "width", "depth", "bqp_error"], as_index=False)
        [[METRIC]].mean()
    )


def _build_grid(sub: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    depths = sorted(sub["depth"].unique())
    widths = sorted(sub["width"].unique(), reverse=True)
    grid = (
        sub.pivot(index="width", columns="depth", values=METRIC)
        .reindex(index=widths, columns=depths)
        .to_numpy(dtype=float)
    )
    return grid, depths, widths


def _color_range(grid: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    if np.isclose(lo, hi):
        lo = max(0.0, lo - 1e-3)
        hi = min(1.0, hi + 1e-3)
    return lo, hi


def _nearest_in_log(targets: list[float], available: list[float]) -> set[float]:
    log_avail = np.log(available)
    result = set()
    for t in targets:
        idx = int(np.argmin(np.abs(log_avail - np.log(t))))
        result.add(available[idx])
    return result


# ── Frontier helpers ──────────────────────────────────────────────────────────

def _compute_max_rho(bqp_error: float, N: int, eps: float) -> float | None:
    """Max tolerable noise w/s for given (bqp_error, N, eps)."""
    try:
        res = maximize_robustness_under_budget(
            c=bqp_error,
            detection_rate=FRONTIER_DETECT,
            epsilon_target=eps,
            budget=N,
            n_grid=300,
        )
        return res.w_over_s
    except (ValueError, RuntimeError):
        return None


def _frontier_label(N: int, eps: float) -> str:
    exp = int(round(-np.log10(eps)))
    return f"$N\\!=\\!{N},\\,\\varepsilon\\!=\\!10^{{-{exp}}}$"


def _lookup_sweep(p_ent: float) -> tuple[list[int], list[float]]:
    """Return (N_values, epsilon_values) for this p_ent from FRONTIER_SWEEP.

    Explicit keys are matched in log-space; None is the fallback.
    """
    explicit = {k: v for k, v in FRONTIER_SWEEP.items() if k is not None}
    default: tuple[list[int], list[float]] = FRONTIER_SWEEP.get(None, ([], []))  # type: ignore[assignment]
    if not explicit:
        return default
    keys = list(explicit.keys())
    idx = int(np.argmin(np.abs(np.log(keys) - np.log(p_ent))))
    nearest = keys[idx]
    if abs(np.log(nearest) - np.log(p_ent)) < 0.5 * np.log(10):
        return explicit[nearest]
    return default


def _frontier_configs(
    bqp_error: float,
    N_values: list[int],
    epsilon_values: list[float],
) -> list[tuple[float, tuple, str]]:
    """Return list of (max_rho, rgba_color, label) sorted by max_rho ascending."""
    eps_cmaps = {
        eps: plt.get_cmap(_CMAP_NAMES[i % len(_CMAP_NAMES)])
        for i, eps in enumerate(epsilon_values)
    }
    n_shades = np.linspace(0.38, 0.90, len(N_values)).tolist()
    configs = []
    for i_n, N in enumerate(N_values):
        for eps in epsilon_values:
            max_rho = _compute_max_rho(bqp_error, N, eps)
            if max_rho is None:
                continue
            color = eps_cmaps[eps](n_shades[i_n])
            label = _frontier_label(N, eps)
            configs.append((max_rho, color, label))
    configs.sort(key=lambda x: x[0])   # draw most restrictive first (permissive on top)
    return configs


def _place_label(ax, cs, label: str, color, x_frac: float, x_range: tuple) -> None:
    """Force a label on a contour at the midpoint of its longest path.

    Used as fallback when clabel declines to label a line (too short).
    """
    all_pts: list = []
    for path in cs.get_paths():
        if len(path.vertices):
            all_pts.append(path.vertices)
    if not all_pts:
        return
    pts = np.concatenate(all_pts, axis=0)
    x_target = x_range[0] + x_frac * (x_range[1] - x_range[0])
    x_target = float(np.clip(x_target, pts[:, 0].min(), pts[:, 0].max()))
    idx = int(np.argmin(np.abs(pts[:, 0] - x_target)))
    x0, y0 = pts[idx]
    ax.text(
        x0, y0, label,
        fontsize=8, color=color,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        zorder=10,
    )


def _title(p_ent: float, bqp_error: float) -> str:
    exp = int(round(np.log10(p_ent)))
    mantissa = p_ent / 10**exp
    if abs(mantissa - 1.0) < 0.05:
        p_str = rf"10^{{{exp}}}"
    else:
        p_str = rf"{mantissa:.1f}\times10^{{{exp}}}"
    return rf"Noise impact on failure rate per circuit dimension  ($p_{{\mathrm{{entangl}}}}={p_str}$)"


# ── Plot functions ────────────────────────────────────────────────────────────

def _draw_discrete_frontiers(
    ax,
    grid: np.ndarray,
    frontiers: list[tuple[float, tuple, str]],
    lw: float = 1.5,
) -> None:
    """Draw frontier edges for all max_rho levels without overlap.

    Frontiers are processed most-restrictive-first. Each boundary edge is
    drawn exactly once, colored by the most restrictive frontier that claims it.
    This avoids duplicate segments when coarse grid resolution causes two
    thresholds to share the same tile boundary.
    """
    n_w, n_d = grid.shape
    sorted_frontiers = sorted(frontiers, key=lambda x: x[0])

    edge_to_fidx: dict[tuple, int] = {}
    for f_idx, (max_rho, _color, _label) in enumerate(sorted_frontiers):
        safe = ~np.isnan(grid) & (grid <= max_rho)
        for i in range(n_w):
            for j in range(n_d):
                if not safe[i, j]:
                    continue
                candidates = []
                if i == 0 or not safe[i - 1, j]:
                    candidates.append(((j - 0.5, i - 0.5), (j + 0.5, i - 0.5)))
                if i == n_w - 1 or not safe[i + 1, j]:
                    candidates.append(((j - 0.5, i + 0.5), (j + 0.5, i + 0.5)))
                if j == 0 or not safe[i, j - 1]:
                    candidates.append(((j - 0.5, i - 0.5), (j - 0.5, i + 0.5)))
                if j == n_d - 1 or not safe[i, j + 1]:
                    candidates.append(((j + 0.5, i - 0.5), (j + 0.5, i + 0.5)))
                for edge in candidates:
                    if edge not in edge_to_fidx:
                        edge_to_fidx[edge] = f_idx

    for edge, f_idx in edge_to_fidx.items():
        (x1, y1), (x2, y2) = edge
        _, color, _ = sorted_frontiers[f_idx]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt", zorder=5)


def plot_discrete(
    grid: np.ndarray,
    depths: list,
    widths: list,
    p_ent: float,
    bqp_error: float,
    outdir: Path,
    frontiers: list[tuple[float, tuple, str]] | None = None,
    suffix: str = "",
) -> Path:
    n_w, n_d = len(widths), len(depths)
    vmin, vmax = _color_range(grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n_d))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(n_w))
    ax.set_yticklabels([str(w) for w in widths])
    ax.set_xlabel("Depth")
    ax.set_ylabel("Width (nqubits)")
    ax.set_title(_title(p_ent, bqp_error))

    for i in range(n_w):
        for j in range(n_d):
            v = grid[i, j]
            text = "NA" if np.isnan(v) else (f"{v:.1e}" if 0 < v < 0.001 else f"{v:.3f}")
            ax.text(j, i, text, ha="center", va="center", fontsize=10)

    if frontiers:
        _draw_discrete_frontiers(ax, grid, frontiers)
        legend_handles = [
            mlines.Line2D([0], [0], color=color, lw=1.5, label=label)
            for _max_rho, color, label in sorted(frontiers, key=lambda x: x[0])
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=7, framealpha=0.85)

    fig.colorbar(im, ax=ax, label=METRIC_LABEL)
    plt.tight_layout()

    outpath = outdir / f"discrete_p{p_ent:.0e}_bqp{bqp_error}{suffix}.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def plot_continuous(
    grid: np.ndarray,
    depths: list,
    widths: list,
    p_ent: float,
    bqp_error: float,
    outdir: Path,
    frontiers: list[tuple[float, tuple, str]] | None = None,
) -> Path:
    vmin, vmax = _color_range(grid)
    n_rows, n_cols = len(widths), len(depths)

    filled = grid.copy()
    if np.isnan(filled).any():
        filled = generic_filter(
            filled,
            lambda x: np.nanmean(x) if np.isnan(x[len(x) // 2]) else x[len(x) // 2],
            size=3, mode="nearest",
        )

    smooth = np.clip(zoom(filled, INTERP_FACTOR, order=3), vmin, vmax)
    n_rows_s, n_cols_s = smooth.shape

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(smooth, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto",
                   extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])

    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="white", linewidth=0.4, alpha=0.5)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="white", linewidth=0.4, alpha=0.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(w) for w in widths])
    ax.set_xlabel("Depth")
    ax.set_ylabel("Width (nqubits)")
    ax.set_title(_title(p_ent, bqp_error))

    # ── frontier lines ────────────────────────────────────────────────────────
    x_cont = np.linspace(-0.5, n_cols - 0.5, n_cols_s)
    y_cont = np.linspace(-0.5, n_rows - 0.5, n_rows_s)

    if frontiers:
        x_range = (-0.5, n_cols - 0.5)
        for max_rho, color, label in frontiers:
            if not (np.nanmin(smooth) < max_rho < np.nanmax(smooth)):
                continue
            cs = ax.contour(
                x_cont, y_cont, smooth,
                levels=[max_rho], colors=[color], linewidths=1.5, zorder=5,
            )
            ax.clabel(cs, fmt={max_rho: label}, fontsize=10, inline=True)
            if not cs.labelTexts:
                _place_label(ax, cs, label, color, 0.5, x_range)

    fig.colorbar(im, ax=ax, label=METRIC_LABEL)
    if frontiers:
        max_rho_sweep = max(rho for rho, _, _ in frontiers)
        ax.text(
            0.98, 0.98,
            rf"$p_{{\mathrm{{noise}}}}^* = {max_rho_sweep:.3f}$",
            transform=ax.transAxes,
            va="top", ha="right", fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85, linewidth=0.7),
            zorder=10,
        )
    plt.tight_layout()

    outpath = outdir / f"heatmap_p{p_ent:.0e}_bqp{bqp_error}.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="applications/gospel/benchmark/results/results-FK.csv")
    args = parser.parse_args()

    OUTDIR_DISCRETE.mkdir(parents=True, exist_ok=True)
    OUTDIR_CONTINUOUS.mkdir(parents=True, exist_ok=True)

    df = _aggregate(pd.read_csv(args.csv))

    all_bqp   = sorted(df["bqp_error"].unique())
    all_p_ent = sorted(df["p_ent"].unique())
    sel_bqp   = _nearest_in_log(FILTER_BQP, all_bqp)
    sel_p_ent = _nearest_in_log(FILTER_P_ENT, all_p_ent)

    for bqp_error in all_bqp:
        if bqp_error not in sel_bqp:
            continue

        for p_ent in all_p_ent:
            if p_ent not in sel_p_ent:
                continue
            sub = df[(df["p_ent"] == p_ent) & (df["bqp_error"] == bqp_error)]
            if sub.empty:
                continue

            N_values, epsilon_values = _lookup_sweep(p_ent)
            print(f"Computing frontiers for p_ent={p_ent:.0e} bqp={bqp_error} "
                  f"(N={N_values}, eps={epsilon_values}) ...")
            frontiers = _frontier_configs(bqp_error, N_values, epsilon_values)
            for rho, _, lbl in frontiers:
                print(f"  max_rho={rho:.5f}  {lbl}")

            grid, depths, widths = _build_grid(sub)

            path = plot_discrete(grid, depths, widths, p_ent, bqp_error,
                                 OUTDIR_DISCRETE, frontiers=frontiers)
            print(f"discrete    → {path}")

            path = plot_discrete(grid, depths, widths, p_ent, bqp_error,
                                 OUTDIR_DISCRETE, suffix="_nofrontier")
            print(f"discrete NF → {path}")

            path = plot_continuous(grid, depths, widths, p_ent, bqp_error,
                                   OUTDIR_CONTINUOUS, frontiers=frontiers)
            print(f"continuous  → {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gospel_plots-v2: heatmaps of failed-test-round fraction per (noise model, BQP error).

For every (p_ent, bqp_error) pair two plots are produced:
  - discrete heatmap  → applications/gospel/plots/
  - interpolated heatmap → applications/gospel/plots-continuous/

Usage
-----
    python applications/gospel/gospel_plots-v2.py
    python applications/gospel/gospel_plots-v2.py \\
        --csv applications/gospel/gospel_results_cluster.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amssymb}\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

OUTDIR_DISCRETE = Path("applications/gospel/plots")
OUTDIR_CONTINUOUS = Path("applications/gospel/plots-continuous")
METRIC = "p_failed_round"
METRIC_LABEL = r"$\mathbb{E}[X/t]$"
INTERP_FACTOR = 10  # upscaling factor for the continuous version


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[METRIC] = df["nr_failed_test_rounds"] / df["test_rounds"]
    return (
        df.groupby(["p_ent", "width", "depth", "bqp_error"], as_index=False)
        [[METRIC]]
        .mean()
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
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    if np.isclose(lo, hi):
        lo = max(0.0, lo - 1e-3)
        hi = min(1.0, hi + 1e-3)
    return lo, hi


def _title(p_ent: float, bqp_error: str) -> str:
    return f"Failed-test-round fraction  (p_ent={p_ent:.0e}, bqp={bqp_error})"


def plot_discrete(
    grid: np.ndarray,
    depths: list,
    widths: list,
    p_ent: float,
    bqp_error: str,
    outdir: Path,
) -> Path:
    vmin, vmax = _color_range(grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([str(w) for w in widths])
    ax.set_xlabel("Depth")
    ax.set_ylabel("Width (nqubits)")
    ax.set_title(_title(p_ent, bqp_error))

    for i in range(len(widths)):
        for j in range(len(depths)):
            v = grid[i, j]
            text = "NA" if np.isnan(v) else (f"{v:.1e}" if 0 < v < 0.001 else f"{v:.3f}")
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label=METRIC_LABEL)
    plt.tight_layout()

    outpath = outdir / f"heatmap_p{p_ent:.0e}_bqp{bqp_error}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def plot_continuous(
    grid: np.ndarray,
    depths: list,
    widths: list,
    p_ent: float,
    bqp_error: str,
    outdir: Path,
) -> Path:
    vmin, vmax = _color_range(grid)

    # Replace NaNs with nearest-neighbour fill before zooming so scipy doesn't
    # propagate NaN across the interpolated region.
    filled = grid.copy()
    nan_mask = np.isnan(filled)
    if nan_mask.any():
        from scipy.ndimage import generic_filter
        filled = generic_filter(filled, lambda x: np.nanmean(x) if np.isnan(x[len(x)//2]) else x[len(x)//2], size=3, mode="nearest")

    smooth = zoom(filled, INTERP_FACTOR, order=3)
    smooth = np.clip(smooth, vmin, vmax)

    n_rows, n_cols = len(widths), len(depths)
    n_rows_s, n_cols_s = smooth.shape

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(smooth, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto",
                   extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])

    # Overlay original cell borders
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

    fig.colorbar(im, ax=ax, label=METRIC_LABEL)
    plt.tight_layout()

    outpath = outdir / f"heatmap_p{p_ent:.0e}_bqp{bqp_error}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="applications/gospel/gospel_results_cluster.csv")
    args = parser.parse_args()

    OUTDIR_DISCRETE.mkdir(parents=True, exist_ok=True)
    OUTDIR_CONTINUOUS.mkdir(parents=True, exist_ok=True)

    df = _aggregate(pd.read_csv(args.csv))

    for bqp_error in sorted(df["bqp_error"].unique()):
        for p_ent in sorted(df["p_ent"].unique()):
            sub = df[(df["p_ent"] == p_ent) & (df["bqp_error"] == bqp_error)]
            if sub.empty:
                continue

            grid, depths, widths = _build_grid(sub)

            path = plot_discrete(grid, depths, widths, p_ent, bqp_error, OUTDIR_DISCRETE)
            print(f"discrete    → {path}")

            path = plot_continuous(grid, depths, widths, p_ent, bqp_error, OUTDIR_CONTINUOUS)
            print(f"continuous  → {path}")


if __name__ == "__main__":
    main()

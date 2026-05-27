#!/usr/bin/env python3
"""Plot heatmaps from gospel simulation CSV data.

The CSV has per-circuit rows; this script aggregates them into
(p_ent, width, depth, bqp_error) cells before plotting.

Aggregated metrics:
    p_failed_round   mean of nr_failed_test_rounds / test_rounds
    p_false_reject   fraction of runs where traps_passed == False

Usage
-----
    python applications/gospel/gospel_plots.py
    python applications/gospel/gospel_plots.py \\
        --csv applications/gospel/gospel_results_cluster.csv \\
        --metric p_failed_round \\
        --bqp-error 1e-1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTDIR = Path("applications/gospel/plots")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_failed_round"] = df["nr_failed_test_rounds"] / df["test_rounds"]
    df["p_false_reject"] = (~df["traps_passed"].astype(bool)).astype(float)
    return (
        df.groupby(["p_ent", "width", "depth", "bqp_error"], as_index=False)
        [["p_failed_round", "p_false_reject"]]
        .mean()
    )


def metric_label(metric: str) -> str:
    return {
        "p_failed_round": r"$\mathbb{E}[X/t]$",
        "p_false_reject": r"$\widehat{\Pr}[X>w]$",
    }.get(metric, metric)


def metric_title(metric: str) -> str:
    return {
        "p_failed_round": "Honest failed-test-round fraction",
        "p_false_reject": "Honest false-reject probability",
    }.get(metric, metric)


def plot_one_heatmap(
    df: pd.DataFrame,
    p_ent: float,
    bqp_error: str,
    metric: str,
    outdir: Path,
    vmax: float | None,
    annotate: bool,
) -> Path:
    sub = df[(df["p_ent"] == p_ent) & (df["bqp_error"] == bqp_error)].copy()
    if sub.empty:
        raise ValueError(f"No rows for p_ent={p_ent}, bqp_error={bqp_error}")

    depths = sorted(sub["depth"].unique())
    widths = sorted(sub["width"].unique(), reverse=True)

    grid = (
        sub.pivot(index="width", columns="depth", values=metric)
        .reindex(index=widths, columns=depths)
        .to_numpy()
    )

    if vmax is None:
        local_min = np.nanmin(grid)
        local_max = np.nanmax(grid)
        if np.isclose(local_min, local_max):
            local_min = max(0.0, local_min - 1e-3)
            local_max = min(1.0, local_max + 1e-3)
        vmin_use, vmax_use = local_min, local_max
    else:
        vmin_use, vmax_use = 0.0, vmax

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, vmin=vmin_use, vmax=vmax_use, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([str(w) for w in widths])
    ax.set_xlabel("Depth")
    ax.set_ylabel("Width (nqubits)")
    ax.set_title(f"{metric_title(metric)}  (p_ent={p_ent:.0e}, bqp={bqp_error})")

    if annotate:
        for i in range(len(widths)):
            for j in range(len(depths)):
                v = grid[i, j]
                text = "NA" if np.isnan(v) else (f"{v:.1e}" if 0 < v < 0.001 else f"{v:.3f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, label=metric_label(metric))
    plt.tight_layout()

    outpath = outdir / f"heatmap_{metric}_p{p_ent:.0e}_bqp{bqp_error}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="applications/gospel/gospel_results_cluster.csv")
    parser.add_argument("--metric", default="p_failed_round",
                        choices=["p_failed_round", "p_false_reject"])
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--p-ent", type=float, default=None,
                        help="Plot only this p_ent value (default: all)")
    parser.add_argument("--bqp-error", type=str, default=None,
                        help="Plot only this bqp_error value (default: all)")
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv)
    df = _aggregate(df_raw)

    p_ents     = [args.p_ent]     if args.p_ent     else sorted(df["p_ent"].unique())
    bqp_errors = [args.bqp_error] if args.bqp_error else sorted(df["bqp_error"].unique())

    for bqp in bqp_errors:
        for p_ent in p_ents:
            try:
                outpath = plot_one_heatmap(
                    df=df,
                    p_ent=p_ent,
                    bqp_error=bqp,
                    metric=args.metric,
                    outdir=outdir,
                    vmax=args.vmax,
                    annotate=not args.no_annotate,
                )
                print(f"Saved → {outpath}")
            except ValueError as e:
                print(f"Skipped: {e}")


if __name__ == "__main__":
    main()

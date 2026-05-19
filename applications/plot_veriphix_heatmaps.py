#!/usr/bin/env python3
"""
Plot heatmaps from Veriphix benchmarking CSV data.

Expected CSV columns:
    p_ent,width,depth,p_failed_round,p_false_reject

Example:
    python3 plot_veriphix_heatmaps.py \
        --csv applications/heatmaps/veriphix_benchmark_results.csv \
        --metric p_failed_round \
        --outdir applications/heatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metric_label(metric: str) -> str:
    labels = {
        "p_failed_round": r"$\mathbb{E}[X/t]$",
        "p_false_reject": r"$\widehat{\Pr}[X>w]$",
    }
    return labels.get(metric, metric)


def metric_title(metric: str) -> str:
    titles = {
        "p_failed_round": "Honest failed-test-round fraction",
        "p_false_reject": "Honest false-reject probability",
    }
    return titles.get(metric, metric)


def format_p_ent(p: float) -> str:
    return f"{p:.0e}"


def plot_one_heatmap(
    df: pd.DataFrame,
    p_ent: float,
    metric: str,
    outdir: Path,
    vmax: float | None,
    annotate: bool,
) -> Path:
    sub = df[df["p_ent"] == p_ent].copy()

    if sub.empty:
        raise ValueError(f"No rows found for p_ent={p_ent}")

    # Invert both axes
    depths = sorted(sub["depth"].unique())
    widths = sorted(sub["width"].unique(), reverse=True)

    grid = (
        sub.pivot(index="width", columns="depth", values=metric)
        .reindex(index=widths, columns=depths)
        .to_numpy()
    )

    if vmax is None:
        local_vmin = np.nanmin(grid)
        local_vmax = np.nanmax(grid)

        if np.isclose(local_vmin, local_vmax):
            local_vmin = max(0.0, local_vmin - 1e-3)
            local_vmax = min(1.0, local_vmax + 1e-3)

        vmin_use = local_vmin
        vmax_use = local_vmax
    else:
        vmin_use = 0.0
        vmax_use = vmax

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        grid,
        vmin=vmin_use,
        vmax=vmax_use,
        cmap="YlOrRd",
        aspect="auto",
    )

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([str(w) for w in widths])

    ax.set_xlabel("Depth")
    ax.set_ylabel("Width (nqubits)")
    ax.set_title(f"{metric_title(metric)}  (p_ent = {format_p_ent(p_ent)})")

    if annotate:
        for i, width in enumerate(widths):
            for j, depth in enumerate(depths):
                value = grid[i, j]
                if np.isnan(value):
                    text = "NA"
                elif value < 0.001 and value > 0:
                    text = f"{value:.1e}"
                else:
                    text = f"{value:.3f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, label=metric_label(metric))
    plt.tight_layout()

    outpath = outdir / f"heatmap_{metric}_p{format_p_ent(p_ent)}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument(
        "--metric",
        default="p_failed_round",
        choices=["p_failed_round", "p_false_reject"],
        help="Metric to plot.",
    )
    parser.add_argument(
        "--outdir",
        default="applications/heatmaps",
        help="Directory where heatmaps will be saved.",
    )
    parser.add_argument(
        "--p-ent",
        type=float,
        default=None,
        help="Plot only this p_ent value. By default, plot all p_ent values.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Colorbar maximum. Default is 1.0.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable cell annotations.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"p_ent", "width", "depth", "p_failed_round", "p_false_reject"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if args.p_ent is None:
        p_ents = sorted(df["p_ent"].unique())
    else:
        p_ents = [args.p_ent]

    for p_ent in p_ents:
        outpath = plot_one_heatmap(
            df=df,
            p_ent=p_ent,
            metric=args.metric,
            outdir=outdir,
            vmax=args.vmax,
            annotate=not args.no_annotate,
        )
        print(f"Saved → {outpath}")


if __name__ == "__main__":
    main()

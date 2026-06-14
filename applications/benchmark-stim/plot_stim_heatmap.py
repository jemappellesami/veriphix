#!/usr/bin/env python3
"""Discrete heatmap of p_failed_round for the Stim benchmark, with feasibility
frontiers overlaid (same style as applications/gospel/benchmark/plots.py).

Each frontier line is the boundary max_rho = w/s for a given (round budget N,
soundness target epsilon, bqp_error c, detection_rate). Tiles with
p_failed_round <= max_rho are "feasible": a verifier with that (N, epsilon)
design accepts the honest run with the target soundness.

Usage:
    python applications/benchmark-stim/plot_stim_heatmap.py \
        --csv applications/benchmark-stim/benchmark_stim_results.csv \
        --p-ent 1e-3 \
        --N 1500,3000,6000 --epsilon 1e-7,1e-8 --bqp-error 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from veriphix.util_rounds import maximize_robustness_under_budget

METRIC = "p_failed_round"
METRIC_LABEL = "Honest failed-test-round fraction"
_CMAP_NAMES = ["Blues", "Greens", "Purples", "Oranges", "Greys"]


def _color_range(grid: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    if np.isclose(lo, hi):
        lo = max(0.0, lo - 1e-3)
        hi = min(1.0, hi + 1e-3)
    return lo, hi


def _frontier_label(N: int, eps: float) -> str:
    exp = int(round(-np.log10(eps)))
    return f"$N={N},\\ \\varepsilon=10^{{-{exp}}}$"


def _frontier_configs(
    bqp_error: float, detection_rate: float, N_values: list[int], epsilon_values: list[float]
) -> list[tuple[float, tuple, str]]:
    eps_cmaps = {eps: plt.get_cmap(_CMAP_NAMES[i % len(_CMAP_NAMES)]) for i, eps in enumerate(epsilon_values)}
    shades = np.linspace(0.38, 0.90, len(N_values)).tolist()
    configs = []
    for i_n, N in enumerate(N_values):
        for eps in epsilon_values:
            try:
                res = maximize_robustness_under_budget(
                    c=bqp_error, detection_rate=detection_rate, epsilon_target=eps, budget=N, n_grid=300
                )
            except (ValueError, RuntimeError):
                continue
            configs.append((res.w_over_s, eps_cmaps[eps](shades[i_n]), _frontier_label(N, eps)))
    configs.sort(key=lambda x: x[0])
    return configs


def _draw_frontiers(ax, grid: np.ndarray, frontiers: list[tuple[float, tuple, str]], lw: float = 1.8) -> None:
    n_w, n_d = grid.shape
    edge_to_fidx: dict[tuple, int] = {}
    for f_idx, (max_rho, _color, _label) in enumerate(frontiers):
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
        _, color, _ = frontiers[f_idx]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt", zorder=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="applications/benchmark-stim/benchmark_stim_results_1e5_s100.csv")
    parser.add_argument("--p-ent", type=float, default=None, help="Select a single p_ent (default: all).")
    parser.add_argument("--N", default="1500,3000,6000", help="Comma-separated round budgets.")
    parser.add_argument("--epsilon", default="1e-7,1e-8", help="Comma-separated soundness targets.")
    parser.add_argument("--bqp-error", type=float, default=0.1, help="BQP completeness/soundness gap c.")
    parser.add_argument("--detection-rate", type=float, default=0.5)
    parser.add_argument("--outdir", default="applications/benchmark-stim/heatmaps")
    args = parser.parse_args()

    N_values = [int(x) for x in args.N.split(",") if x.strip()]
    epsilon_values = [float(x) for x in args.epsilon.split(",") if x.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    p_ents = sorted(df["p_ent"].unique()) if args.p_ent is None else [args.p_ent]

    for p_ent in p_ents:
        sub = df[df["p_ent"] == p_ent]
        if sub.empty:
            continue

        depths = sorted(sub["depth"].unique())
        widths = sorted(sub["width"].unique(), reverse=True)
        grid = (
            sub.pivot(index="width", columns="depth", values=METRIC)
            .reindex(index=widths, columns=depths)
            .to_numpy(dtype=float)
        )

        frontiers = _frontier_configs(args.bqp_error, args.detection_rate, N_values, epsilon_values)
        print(f"p_ent={p_ent:.0e}: frontiers (max_rho, label) = "
              f"{[(round(r, 4), lbl) for r, _, lbl in frontiers]}")

        vmin, vmax = _color_range(grid)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_yticks(range(len(widths)))
        ax.set_yticklabels([str(w) for w in widths])
        ax.set_xlabel("Depth")
        ax.set_ylabel("Width (nqubits)")
        ax.set_title(f"{METRIC_LABEL}  (p_ent = {p_ent:.0e}, c = {args.bqp_error})")

        # for i, _w in enumerate(widths):
        #     for j, _d in enumerate(depths):
        #         v = grid[i, j]
        #         text = "NA" if np.isnan(v) else (f"{v:.1e}" if 0 < v < 0.001 else f"{v:.3f}")
        #         ax.text(j, i, text, ha="center", va="center", fontsize=9)

        if frontiers:
            _draw_frontiers(ax, grid, frontiers)
            handles = [mlines.Line2D([0], [0], color=c, lw=1.8, label=lbl) for _r, c, lbl in frontiers]
            ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.85)

        fig.colorbar(im, ax=ax, label=METRIC_LABEL)
        plt.tight_layout()

        outpath = outdir / f"discrete_p{p_ent:.0e}_bqp{args.bqp_error}.pdf"
        fig.savefig(outpath)
        plt.close(fig)
        print(f"Saved -> {outpath}")


if __name__ == "__main__":
    main()

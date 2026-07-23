#!/usr/bin/env python3
"""Discrete heatmap of the honest-failure metric for the Broadbent-compiled Clifford+MSI
benchmark, with feasibility frontiers overlaid -- the (width, depth) analogue of
``applications/benchmark-stim/plot_stim_heatmap.py`` and
``applications/benchmark-stim-msi/plot_msi_heatmap.py``.

Each frontier line is the boundary ``max_rho = w/s`` for a given (round budget N, soundness
target epsilon, bqp_error c, detection_rate). Tiles with ``p_failed_round <= max_rho`` are
"feasible". For the FK12-analogue bipartite traps the detection rate is 1/2, the default.

Reads CSVs with columns ``p_depol,width,depth,p_failed_round,p_false_reject`` (one figure per
``(csv, p_depol)``). The shot count is read from the filename (``_s<N>``). Frontiers are drawn
only for ``--metric p_failed_round``; ``p_false_reject`` is drawn as a plain heatmap. LaTeX is
used only if a ``latex`` binary is on PATH, else mathtext -- runs on a bare laptop.

Usage:
    python applications/benchmark-stim-msi-bro/plot_bro_heatmap.py \
        --csv applications/benchmark-stim-msi-bro/benchmark_bro_results_p1.0e-03_s100.csv \
        --metric p_failed_round
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from veriphix.util_rounds import maximize_robustness_under_budget

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
        "legend.fontsize": 7,
    }
)
if _USETEX:
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}\usepackage{amsmath}"

# ── Edit here instead of the command line ───────────────────────────────────────
_HERE = Path(__file__).resolve().parent
CSV_FILES = sorted(_HERE.glob("benchmark_bro_results_p*_[sr]*.csv"))
AUTOFRONTIER = True
AUTO_QUANTILES = [0.3, 0.5, 0.7]
AUTO_N_GRID = list(range(500, 8001, 100))
AUTO_EPS_GRID = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

BQP_ERROR = 0.1
DETECTION_RATE = 0.5  # FK12-analogue bipartite traps: detection rate 1/2
OUTDIR = _HERE / "heatmaps"
ANNOTATE = False

# Iso-contours of the metric itself, drawn on top of the heatmap. These are *not* the
# feasibility frontiers: a frontier is a threshold derived from a verifier budget
# (N, epsilon) and traces the staircase edge of the accepted tile set, whereas a contour is
# a plain level set of the measured surface, interpolated between tile centres. 0.25 is
# drawn in red as the reference level.
CONTOUR_LEVELS = [0.1, 0.2, 0.25]
CONTOUR_COLORS = {0.25: "red"}
CONTOUR_STYLES = {0.1: "--", 0.2: "-.", 0.25: "-"}
CONTOUR_DEFAULT_COLOR = "black"

# Feasibility frontiers (the coloured per-(N, epsilon) staircases) are off by default: the
# iso-contours above are what the plots are read for. The machinery below is kept and can
# be switched back on with --frontiers. Leaving it off also skips the
# maximize_robustness_under_budget search, which is the slow part of this script.
DRAW_FRONTIERS = False
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_LABEL = {
    "p_failed_round": "Average test round failure rate",
    "p_false_reject": "Honest false-reject probability",
}
_CMAP_NAMES = ["Blues", "Greens", "Purples", "Oranges", "Greys"]
# ``_r<N>`` is the flat-round count (current runs); ``_s<N>`` the legacy
# shots-per-instance count from before the shots x test_rounds collapse.
_COUNT_RE = re.compile(r"_([sr])(\d+)")
_COUNT_WORD = {"r": "rounds", "s": "shots"}


def _count_from_name(csv_path: Path) -> tuple[str, str]:
    """``(kind, value)`` from a results filename: ``_r<N>`` rounds or legacy ``_s<N>`` shots.

    The letter is carried through to the figure title and output filename so a plot of
    legacy shots data is never mislabelled as (or overwritten by) a flat-rounds plot.
    """
    m = _COUNT_RE.search(csv_path.stem)
    return (m.group(1), m.group(2)) if m else ("r", "NA")


def _p_latex(p: float) -> str:
    exp = int(np.floor(np.log10(p)))
    mantissa = p / 10**exp
    if abs(mantissa - 1.0) < 0.05:
        return rf"10^{{{exp}}}"
    return rf"{mantissa:g}\times10^{{{exp}}}"


def _p_tag(p: float) -> str:
    return f"{p:.1e}"


def _color_range(grid: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    if np.isclose(lo, hi):
        lo = max(0.0, lo - 1e-3)
        hi = min(1.0, hi + 1e-3)
    return lo, hi


def _frontier_label(n_rounds: int, eps: float) -> str:
    exp = int(round(-np.log10(eps)))
    return rf"$N={n_rounds},\ \epsilon=10^{{-{exp}}}$"


def _frontier_configs(
    bqp_error: float, detection_rate: float, n_values: list[int], epsilon_values: list[float]
) -> list[tuple[float, tuple, str]]:
    eps_cmaps = {eps: plt.get_cmap(_CMAP_NAMES[i % len(_CMAP_NAMES)]) for i, eps in enumerate(epsilon_values)}
    shades = np.linspace(0.38, 0.90, len(n_values)).tolist()
    configs = []
    for i_n, n_rounds in enumerate(n_values):
        for eps in epsilon_values:
            try:
                res = maximize_robustness_under_budget(
                    c=bqp_error, detection_rate=detection_rate, epsilon_target=eps, budget=n_rounds, n_grid=300
                )
            except (ValueError, RuntimeError):
                continue
            configs.append((res.w_over_s, eps_cmaps[eps](shades[i_n]), _frontier_label(n_rounds, eps)))
    configs.sort(key=lambda x: x[0])
    return configs


def _candidate_max_rhos(bqp_error: float, detection_rate: float) -> list[tuple[float, int, float]]:
    cand: list[tuple[float, int, float]] = []
    for n_rounds in AUTO_N_GRID:
        for eps in AUTO_EPS_GRID:
            try:
                res = maximize_robustness_under_budget(
                    c=bqp_error, detection_rate=detection_rate, epsilon_target=eps, budget=n_rounds, n_grid=300
                )
            except (ValueError, RuntimeError):
                continue
            cand.append((res.w_over_s, n_rounds, eps))
    return cand


def _auto_frontier_configs(
    grid: np.ndarray, candidates: list[tuple[float, int, float]]
) -> list[tuple[float, tuple, str]]:
    vals = grid[~np.isnan(grid)]
    if vals.size == 0 or not candidates:
        return []
    targets = [float(np.quantile(vals, q)) for q in AUTO_QUANTILES]
    chosen: list[tuple[float, int, float]] = []
    seen: set[tuple[int, float]] = set()
    for target in targets:
        rho, n_rounds, eps = min(candidates, key=lambda c: abs(c[0] - target))
        if (n_rounds, eps) in seen:
            continue
        seen.add((n_rounds, eps))
        chosen.append((rho, n_rounds, eps))

    eps_list = sorted({eps for _r, _N, eps in chosen}, reverse=True)
    eps_cmaps = {eps: plt.get_cmap(_CMAP_NAMES[i % len(_CMAP_NAMES)]) for i, eps in enumerate(eps_list)}
    n_list = sorted({n_rounds for _r, n_rounds, _eps in chosen})
    shade = {n_rounds: s for n_rounds, s in zip(n_list, np.linspace(0.45, 0.90, max(len(n_list), 1)))}
    configs = [(rho, eps_cmaps[eps](shade[n_rounds]), _frontier_label(n_rounds, eps)) for rho, n_rounds, eps in chosen]
    configs.sort(key=lambda x: x[0])
    return configs


def _draw_frontiers(ax, grid: np.ndarray, frontiers: list[tuple[float, tuple, str]], lw: float = 1.8) -> None:
    n_rows, n_cols = grid.shape
    edge_to_fidx: dict[tuple, int] = {}
    for f_idx, (max_rho, _color, _label) in enumerate(frontiers):
        safe = ~np.isnan(grid) & (grid <= max_rho)
        for i in range(n_rows):
            for j in range(n_cols):
                if not safe[i, j]:
                    continue
                candidates = []
                if i == 0 or not safe[i - 1, j]:
                    candidates.append(((j - 0.5, i - 0.5), (j + 0.5, i - 0.5)))
                if i == n_rows - 1 or not safe[i + 1, j]:
                    candidates.append(((j - 0.5, i + 0.5), (j + 0.5, i + 0.5)))
                if j == 0 or not safe[i, j - 1]:
                    candidates.append(((j - 0.5, i - 0.5), (j - 0.5, i + 0.5)))
                if j == n_cols - 1 or not safe[i, j + 1]:
                    candidates.append(((j + 0.5, i - 0.5), (j + 0.5, i + 0.5)))
                for edge in candidates:
                    if edge not in edge_to_fidx:
                        edge_to_fidx[edge] = f_idx

    for edge, f_idx in edge_to_fidx.items():
        (x1, y1), (x2, y2) = edge
        _, color, _ = frontiers[f_idx]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt", zorder=5)


def _draw_contours(ax, grid: np.ndarray, levels: list[float]) -> list[tuple[float, str]]:
    """Overlay iso-lines of the metric; returns the (level, colour) pairs actually drawn.

    Levels outside the observed range are skipped (``contour`` would silently draw nothing)
    and reported, so a missing line is never mistaken for a line at the edge of the grid.
    Contour coordinates are array indices, which is exactly what ``imshow`` uses for the
    tile centres, so the two overlay without any extent juggling.
    """
    vals = grid[~np.isnan(grid)]
    if not levels or vals.size == 0 or grid.shape[0] < 2 or grid.shape[1] < 2:
        return []
    lo, hi = float(vals.min()), float(vals.max())
    drawn = sorted({lv for lv in levels if lo <= lv <= hi})
    skipped = sorted(set(levels) - set(drawn))
    if skipped:
        print(f"   no contour for {skipped}: outside the data range [{lo:.4f}, {hi:.4f}]")
    if not drawn:
        return []
    colors = [CONTOUR_COLORS.get(lv, CONTOUR_DEFAULT_COLOR) for lv in drawn]
    cs = ax.contour(
        grid,
        levels=drawn,
        colors=colors,
        linestyles=[CONTOUR_STYLES.get(lv, "-") for lv in drawn],
        linewidths=[2.0 if lv in CONTOUR_COLORS else 1.3 for lv in drawn],
        zorder=6,
    )
    ax.clabel(cs, fmt={lv: f"{lv:g}" for lv in drawn}, fontsize=7, inline=True)
    return list(zip(drawn, colors, strict=True))


def _build_grid(sub: pd.DataFrame, metric: str) -> tuple[np.ndarray, list, list]:
    """Grid with depth on the x-axis and width (logical wires) on the y-axis."""
    depths = sorted(sub["depth"].unique())
    widths = sorted(sub["width"].unique(), reverse=True)
    grid = (
        sub.pivot(index="width", columns="depth", values=metric)
        .reindex(index=widths, columns=depths)
        .to_numpy(dtype=float)
    )
    return grid, depths, widths


def _plot_one(
    grid: np.ndarray,
    depths: list,
    widths: list,
    p_depol: float,
    count: tuple[str, str],
    metric: str,
    bqp_error: float,
    frontiers: list[tuple[float, tuple, str]],
    outdir: Path,
    annotate: bool,
    contour_levels: list[float],
) -> Path:
    vals = grid[~np.isnan(grid)]
    if frontiers:
        accepted = [(round(r, 4), float(np.mean(vals <= r)) if vals.size else 0.0, lbl) for r, _c, lbl in frontiers]
        print(
            f"p_depol={_p_tag(p_depol)} {_COUNT_WORD[count[0]]}={count[1]}: "
            f"frontiers (max_rho, accepted_frac, label) = "
            f"{[(r, round(f, 2), lbl) for r, f, lbl in accepted]}"
        )

    vmin, vmax = _color_range(grid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")

    d_ticks = [i for i, d in enumerate(depths) if d % 5 == 0]
    w_ticks = [i for i, w in enumerate(widths) if w % 5 == 0]
    ax.set_xticks(d_ticks or range(len(depths)))
    ax.set_xticklabels([str(depths[i]) for i in (d_ticks or range(len(depths)))])
    ax.set_yticks(w_ticks or range(len(widths)))
    ax.set_yticklabels([str(widths[i]) for i in (w_ticks or range(len(widths)))])
    ax.set_xlabel(r"depth (layers)")
    ax.set_ylabel(r"width (logical wires)")
    ax.set_title(
        rf"Noise impact on failure rate per circuit dimension "
        rf"($p_{{\mathrm{{depol}}}}={_p_latex(p_depol)}$, "
        rf"$N_{{\mathrm{{{_COUNT_WORD[count[0]]}}}}}={count[1]}$)"
    )

    if annotate:
        for i in range(len(widths)):
            for j in range(len(depths)):
                v = grid[i, j]
                text = "NA" if np.isnan(v) else (f"{v:.1e}" if 0 < v < 0.001 else f"{v:.3f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=6)

    handles = []
    if frontiers:
        _draw_frontiers(ax, grid, frontiers)
        handles += [mlines.Line2D([0], [0], color=c, lw=1.8, label=lbl) for _r, c, lbl in frontiers]
    handles += [
        mlines.Line2D(
            [0], [0],
            color=color,
            lw=2.0 if lv in CONTOUR_COLORS else 1.3,
            linestyle=CONTOUR_STYLES.get(lv, "-"),
            label=rf"$\rho={lv:g}$",
        )
        for lv, color in _draw_contours(ax, grid, contour_levels)
    ]
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.85)

    fig.colorbar(im, ax=ax, label=_METRIC_LABEL.get(metric, metric))
    plt.tight_layout()

    outpath = outdir / f"discrete_{metric}_p{_p_tag(p_depol)}_{count[0]}{count[1]}_bqp{bqp_error}.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved -> {outpath}")
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Comma-separated CSVs (default: all in this folder).")
    parser.add_argument("--metric", default="p_failed_round", choices=["p_failed_round", "p_false_reject"])
    parser.add_argument("--p-depol", type=float, default=None, help="Select a single p_depol (default: all).")
    parser.add_argument("--N", default=None, help="Comma-separated round budgets (manual frontiers).")
    parser.add_argument("--epsilon", default=None, help="Comma-separated soundness targets (manual frontiers).")
    parser.add_argument("--bqp-error", type=float, default=BQP_ERROR)
    parser.add_argument("--detection-rate", type=float, default=DETECTION_RATE)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--annotate", action="store_true", default=ANNOTATE)
    parser.add_argument("--contours", default=None,
                        help=f"Comma-separated iso-contour levels (default: {CONTOUR_LEVELS}); empty string disables.")
    parser.add_argument("--frontiers", dest="frontiers", action="store_true", default=DRAW_FRONTIERS,
                        help="Also draw the coloured per-(N, epsilon) feasibility staircases.")
    parser.add_argument("--no-frontiers", dest="frontiers", action="store_false")
    parser.add_argument("--autofrontier", dest="autofrontier", action="store_true", default=AUTOFRONTIER)
    parser.add_argument("--no-autofrontier", dest="autofrontier", action="store_false")
    args = parser.parse_args()

    manual = args.N is not None or args.epsilon is not None
    contour_levels = (
        CONTOUR_LEVELS if args.contours is None
        else [float(x) for x in args.contours.split(",") if x.strip()]
    )
    csv_paths = CSV_FILES if args.csv is None else [Path(x.strip()) for x in args.csv.split(",") if x.strip()]
    outdir = Path(args.outdir) if args.outdir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    want_frontiers = args.frontiers and args.metric == "p_failed_round"
    candidates = (
        _candidate_max_rhos(args.bqp_error, args.detection_rate)
        if (want_frontiers and args.autofrontier and not manual) else []
    )

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"!! skipping missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"!! skipping empty CSV: {csv_path}")
            continue
        count = _count_from_name(csv_path)
        p_depols = sorted(df["p_depol"].unique()) if args.p_depol is None else [args.p_depol]
        for p_depol in p_depols:
            sub = df[df["p_depol"] == p_depol]
            if sub.empty:
                continue
            grid, depths, widths = _build_grid(sub, args.metric)

            if not want_frontiers:
                frontiers: list[tuple[float, tuple, str]] = []
            elif manual:
                n_vals = [int(x) for x in args.N.split(",")] if args.N else [1200, 2000]
                eps_vals = [float(x) for x in args.epsilon.split(",")] if args.epsilon else [1e-5, 1e-7]
                frontiers = _frontier_configs(args.bqp_error, args.detection_rate, n_vals, eps_vals)
            else:
                frontiers = _auto_frontier_configs(grid, candidates)

            _plot_one(grid, depths, widths, p_depol, count, args.metric, args.bqp_error,
                      frontiers, outdir, args.annotate, contour_levels)


if __name__ == "__main__":
    main()

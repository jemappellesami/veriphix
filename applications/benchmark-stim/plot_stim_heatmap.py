#!/usr/bin/env python3
"""Discrete heatmap of p_failed_round for the Stim benchmark, with feasibility
frontiers overlaid (LaTeX/PDF styling, matched to applications/gospel/benchmark/plots.py).

Each frontier line is the boundary max_rho = w/s for a given (round budget N,
soundness target epsilon, bqp_error c, detection_rate). Tiles with
p_failed_round <= max_rho are "feasible": a verifier with that (N, epsilon)
design accepts the honest run with the target soundness.

Multiple CSVs may be passed at once (comma-separated); one figure is produced per
(csv, p_ent). The number of shots is read from each CSV's filename (``_s<N>``) and
included in the output filename, alongside the *exact* (non-rounded) p_ent value.

Usage:
    python applications/benchmark-stim/plot_stim_heatmap.py \
        --csv applications/benchmark-stim/benchmark_stim_results_p1.5e-05_s1000.csv,\
applications/benchmark-stim/benchmark_stim_results_p2.0e-05_s1000.csv \
        --N 1200,2000 --epsilon 1e-5,1e-7 --bqp-error 0.1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# ── Edit here instead of the command line ───────────────────────────────────────
# CSVs to plot (one figure per (csv, p_ent)). Used when --csv is not passed.
_HERE = Path(__file__).resolve().parent
CSV_FILES = [
    _HERE / "benchmark_stim_results_p1.0e-05_s1000.csv",
    _HERE / "benchmark_stim_results_p1.5e-05_s1000.csv",
    _HERE / "benchmark_stim_results_p2.0e-05_s1000.csv",
    _HERE / "benchmark_stim_results_p3.0e-05_s1000.csv",
    _HERE / "benchmark_stim_results_p5.0e-05_s1000.csv",
]
# Auto-frontier: when True (and no --N/--epsilon given), the script searches for the
# (N, epsilon) pairs whose frontier accepts a target fraction of tiles (AUTO_QUANTILES),
# so the lines land in the middle of *each* noise level's data automatically.
AUTOFRONTIER = True
AUTO_QUANTILES = [0.3, 0.5, 0.7]          # fraction of tiles each auto frontier should accept
AUTO_N_GRID = list(range(500, 8001, 100))  # candidate round budgets to search
AUTO_EPS_GRID = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  # candidate soundness targets

# When AUTOFRONTIER is False: gospel-style per-noise-level (N_values, epsilon_values).
# Keys are matched to data p_ent in log-space; None is the fallback.
FRONTIER_SWEEP: dict[float | None, tuple[list[int], list[float]]] = {
    1.5e-5: ([1200, 2000], [1e-5, 1e-7]),
    2.0e-5: ([1500, 2500], [1e-5, 1e-7]),
    3.0e-5: ([2000, 3000], [1e-5, 1e-7]),
    5.0e-5: ([2500, 4000], [1e-5, 1e-7]),
    None:   ([1200, 2000], [1e-5, 1e-7]),
}

BQP_ERROR = 0.1
DETECTION_RATE = 0.5
OUTDIR = _HERE / "heatmaps"
ANNOTATE = False  # write the numeric value inside each cell
# ─────────────────────────────────────────────────────────────────────────────

METRIC = "p_failed_round"
METRIC_LABEL = "Average test round failure rate"
_CMAP_NAMES = ["Blues", "Greens", "Purples", "Oranges", "Greys"]
_SHOTS_RE = re.compile(r"_s(\d+)")


def _shots_from_name(csv_path: Path) -> str:
    """Extract the shot count from a results filename (``..._s<N>.csv``)."""
    m = _SHOTS_RE.search(csv_path.stem)
    return m.group(1) if m else "NA"


def _p_ent_latex(p_ent: float) -> str:
    """LaTeX for the *exact* p_ent value (mantissa kept, not rounded to one digit)."""
    exp = int(np.floor(np.log10(p_ent)))
    mantissa = p_ent / 10**exp
    if abs(mantissa - 1.0) < 0.05:
        return rf"10^{{{exp}}}"
    return rf"{mantissa:g}\times10^{{{exp}}}"


def _p_ent_tag(p_ent: float) -> str:
    """Filename-safe tag preserving the real value, e.g. 1.5e-05 (not 2e-05)."""
    return f"{p_ent:.1e}"


def _color_range(grid: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    if np.isclose(lo, hi):
        lo = max(0.0, lo - 1e-3)
        hi = min(1.0, hi + 1e-3)
    return lo, hi


def _frontier_label(N: int, eps: float) -> str:
    exp = int(round(-np.log10(eps)))
    return rf"$N\!=\!{N},\,\varepsilon\!=\!10^{{-{exp}}}$"


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


def _nearest_in_log(target: float, available: list[float]) -> float:
    log_avail = np.log(list(available))
    return list(available)[int(np.argmin(np.abs(log_avail - np.log(target))))]


def _lookup_sweep(p_ent: float) -> tuple[list[int], list[float]]:
    """Gospel-style per-noise-level (N_values, epsilon_values); None is the fallback."""
    explicit = {k: v for k, v in FRONTIER_SWEEP.items() if k is not None}
    default = FRONTIER_SWEEP.get(None, ([], []))
    if not explicit:
        return default
    nearest = _nearest_in_log(p_ent, list(explicit.keys()))
    if abs(np.log(nearest) - np.log(p_ent)) < 0.5 * np.log(10):
        return explicit[nearest]
    return default


def _candidate_max_rhos(bqp_error: float, detection_rate: float) -> list[tuple[float, int, float]]:
    """All (max_rho, N, eps) over the auto search grid. Independent of p_ent, so build once."""
    cand: list[tuple[float, int, float]] = []
    for N in AUTO_N_GRID:
        for eps in AUTO_EPS_GRID:
            try:
                res = maximize_robustness_under_budget(
                    c=bqp_error, detection_rate=detection_rate, epsilon_target=eps, budget=N, n_grid=300
                )
            except (ValueError, RuntimeError):
                continue
            cand.append((res.w_over_s, N, eps))
    return cand


def _auto_frontier_configs(
    grid: np.ndarray, candidates: list[tuple[float, int, float]]
) -> list[tuple[float, tuple, str]]:
    """Pick (N, eps) whose frontier accepts AUTO_QUANTILES fractions of the tiles.

    The accepted fraction for threshold ``max_rho`` is ``mean(p_failed_round <= max_rho)``,
    so targeting the q-th quantile of the tile distribution yields a frontier that accepts
    ~q of the tiles — i.e. lands in the middle of *this* noise level's data.
    """
    vals = grid[~np.isnan(grid)]
    if vals.size == 0 or not candidates:
        return []
    targets = [float(np.quantile(vals, q)) for q in AUTO_QUANTILES]
    chosen: list[tuple[float, int, float]] = []
    seen: set[tuple[int, float]] = set()
    for t in targets:
        rho, N, eps = min(candidates, key=lambda c: abs(c[0] - t))
        if (N, eps) in seen:
            continue
        seen.add((N, eps))
        chosen.append((rho, N, eps))

    eps_list = sorted({eps for _r, _N, eps in chosen}, reverse=True)
    eps_cmaps = {eps: plt.get_cmap(_CMAP_NAMES[i % len(_CMAP_NAMES)]) for i, eps in enumerate(eps_list)}
    N_list = sorted({N for _r, N, _eps in chosen})
    shade = {N: s for N, s in zip(N_list, np.linspace(0.45, 0.90, max(len(N_list), 1)))}
    configs = [(rho, eps_cmaps[eps](shade[N]), _frontier_label(N, eps)) for rho, N, eps in chosen]
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


def _build_grid(sub: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    depths = sorted(sub["depth"].unique())
    widths = sorted(sub["width"].unique(), reverse=True)
    grid = (
        sub.pivot(index="width", columns="depth", values=METRIC)
        .reindex(index=widths, columns=depths)
        .to_numpy(dtype=float)
    )
    return grid, depths, widths


def _plot_one(
    grid: np.ndarray,
    depths: list,
    widths: list,
    p_ent: float,
    shots: str,
    bqp_error: float,
    frontiers: list[tuple[float, tuple, str]],
    outdir: Path,
    annotate: bool,
) -> Path:
    vals = grid[~np.isnan(grid)]
    accepted = [(round(r, 4), float(np.mean(vals <= r)) if vals.size else 0.0, lbl) for r, _c, lbl in frontiers]
    print(f"p_ent={_p_ent_tag(p_ent)} shots={shots}: frontiers (max_rho, accepted_frac, label) = "
          f"{[(r, round(f, 2), lbl) for r, f, lbl in accepted]}")

    vmin, vmax = _color_range(grid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")

    # Sparse ticks so a large grid stays readable: label only multiples of 5.
    d_ticks = [i for i, d in enumerate(depths) if d % 5 == 0]
    w_ticks = [i for i, w in enumerate(widths) if w % 5 == 0]
    ax.set_xticks(d_ticks)
    ax.set_xticklabels([str(depths[i]) for i in d_ticks])
    ax.set_yticks(w_ticks)
    ax.set_yticklabels([str(widths[i]) for i in w_ticks])
    ax.set_xlabel("Depth")
    ax.set_ylabel("Width (nqubits)")
    ax.set_title(
        rf"Noise impact on failure rate per circuit dimension "
        rf"($p_{{\mathrm{{entangl}}}}={_p_ent_latex(p_ent)}$, $N_{{\mathrm{{shots}}}}={shots}$)"
    )

    if annotate:
        for i, _w in enumerate(widths):
            for j, _d in enumerate(depths):
                v = grid[i, j]
                text = "NA" if np.isnan(v) else (f"{v:.1e}" if 0 < v < 0.001 else f"{v:.3f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=6)

    if frontiers:
        _draw_frontiers(ax, grid, frontiers)
        handles = [mlines.Line2D([0], [0], color=c, lw=1.8, label=lbl) for _r, c, lbl in frontiers]
        ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.85)

    fig.colorbar(im, ax=ax, label=METRIC_LABEL)
    plt.tight_layout()

    outpath = outdir / f"discrete_p{_p_ent_tag(p_ent)}_s{shots}_bqp{bqp_error}.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved -> {outpath}")
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Comma-separated list of CSVs (default: CSV_FILES at top of file).")
    parser.add_argument("--p-ent", type=float, default=None, help="Select a single p_ent (default: all in each CSV).")
    parser.add_argument("--N", default=None, help="Comma-separated round budgets (default: N_VALUES).")
    parser.add_argument("--epsilon", default=None, help="Comma-separated soundness targets (default: EPSILON_VALUES).")
    parser.add_argument("--bqp-error", type=float, default=BQP_ERROR, help="BQP completeness/soundness gap c.")
    parser.add_argument("--detection-rate", type=float, default=DETECTION_RATE)
    parser.add_argument("--outdir", default=None, help="Output dir (default: OUTDIR).")
    parser.add_argument("--annotate", action="store_true", default=ANNOTATE, help="Write the value inside each cell.")
    parser.add_argument("--autofrontier", dest="autofrontier", action="store_true", default=AUTOFRONTIER,
                        help="Auto-pick (N, epsilon) so frontiers land mid-data (per noise level).")
    parser.add_argument("--no-autofrontier", dest="autofrontier", action="store_false",
                        help="Use the per-noise-level FRONTIER_SWEEP table instead.")
    args = parser.parse_args()

    # Explicit --N/--epsilon override everything (manual global pair).
    manual = args.N is not None or args.epsilon is not None
    csv_paths = CSV_FILES if args.csv is None else [Path(x.strip()) for x in args.csv.split(",") if x.strip()]

    outdir = Path(args.outdir) if args.outdir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    # Precompute the auto candidate table once (independent of p_ent) if needed.
    candidates = (
        _candidate_max_rhos(args.bqp_error, args.detection_rate)
        if (args.autofrontier and not manual) else []
    )

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"!! skipping missing CSV: {csv_path}")
            continue
        shots = _shots_from_name(csv_path)
        df = pd.read_csv(csv_path)
        p_ents = sorted(df["p_ent"].unique()) if args.p_ent is None else [args.p_ent]
        for p_ent in p_ents:
            sub = df[df["p_ent"] == p_ent]
            if sub.empty:
                continue
            grid, depths, widths = _build_grid(sub)

            if manual:
                N_vals = N_VALUES if args.N is None else [int(x) for x in args.N.split(",") if x.strip()]
                eps_vals = EPSILON_VALUES if args.epsilon is None else [float(x) for x in args.epsilon.split(",") if x.strip()]
                frontiers = _frontier_configs(args.bqp_error, args.detection_rate, N_vals, eps_vals)
            elif args.autofrontier:
                frontiers = _auto_frontier_configs(grid, candidates)
            else:  # gospel-style per-noise-level table
                N_vals, eps_vals = _lookup_sweep(p_ent)
                frontiers = _frontier_configs(args.bqp_error, args.detection_rate, N_vals, eps_vals)

            _plot_one(grid, depths, widths, p_ent, shots, args.bqp_error,
                      frontiers, outdir, args.annotate)


if __name__ == "__main__":
    main()

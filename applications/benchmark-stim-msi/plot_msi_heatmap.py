#!/usr/bin/env python3
"""Discrete heatmap of the honest-failure metric for the Clifford+MSI benchmark, with
feasibility frontiers overlaid -- the (n, t) analogue of
``applications/benchmark-stim/plot_stim_heatmap.py``.

Each frontier line is the boundary ``max_rho = w/s`` for a given (round budget N,
soundness target epsilon, bqp_error c, detection_rate). Tiles with
``p_failed_round <= max_rho`` are "feasible": a verifier with that (N, epsilon) design
accepts the honest run with the target soundness. For RandomTraps the detection rate is
1/2 (``veriphix.protocols.RandomTraps``), so that is the default ``--detection-rate``.

Reads CSVs with columns ``p_depol,n,t,p_failed_round,p_false_reject`` (one figure per
``(csv, p_depol)``). The shot count is read from the filename (``_s<N>``). Frontiers are
drawn only for ``--metric p_failed_round`` (the metric the ``w/s`` boundary compares
against); ``p_false_reject`` is drawn as a plain heatmap.

LaTeX is used only if a ``latex`` binary is on PATH; otherwise matplotlib's mathtext is
used, so this runs on a laptop with no TeX install.

Usage:
    python applications/benchmark-stim-msi/plot_msi_heatmap.py \
        --csv applications/benchmark-stim-msi/benchmark_msi_results_p1.0e-03_s100.csv \
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

# Use real LaTeX only when available; fall back to mathtext otherwise (labels below are
# written to render under both). Keeps the plot working on a bare laptop.
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
CSV_FILES = sorted(_HERE.glob("benchmark_msi_results_p*_s*.csv"))
# Auto-frontier: pick (N, epsilon) pairs whose frontier accepts AUTO_QUANTILES fractions
# of the tiles, so the lines land in the middle of each noise level's data automatically.
AUTOFRONTIER = True
AUTO_QUANTILES = [0.3, 0.5, 0.7]
AUTO_N_GRID = list(range(500, 8001, 100))
AUTO_EPS_GRID = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

BQP_ERROR = 0.1
DETECTION_RATE = 0.5  # RandomTraps: each harmful deviation caught w.p. >= 1/2
OUTDIR = _HERE / "heatmaps"
ANNOTATE = False
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_LABEL = {
    "p_failed_round": "Average test round failure rate",
    "p_false_reject": "Honest false-reject probability",
}
_CMAP_NAMES = ["Blues", "Greens", "Purples", "Oranges", "Greys"]
_SHOTS_RE = re.compile(r"_s(\d+)")


def _shots_from_name(csv_path: Path) -> str:
    m = _SHOTS_RE.search(csv_path.stem)
    return m.group(1) if m else "NA"


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


def _build_grid(sub: pd.DataFrame, metric: str) -> tuple[np.ndarray, list, list]:
    """Grid with t on the x-axis (injection layers) and n on the y-axis (data qubits)."""
    ts = sorted(sub["t"].unique())
    ns = sorted(sub["n"].unique(), reverse=True)
    grid = (
        sub.pivot(index="n", columns="t", values=metric)
        .reindex(index=ns, columns=ts)
        .to_numpy(dtype=float)
    )
    return grid, ts, ns


def _plot_one(
    grid: np.ndarray,
    ts: list,
    ns: list,
    p_depol: float,
    shots: str,
    metric: str,
    bqp_error: float,
    frontiers: list[tuple[float, tuple, str]],
    outdir: Path,
    annotate: bool,
) -> Path:
    vals = grid[~np.isnan(grid)]
    if frontiers:
        accepted = [(round(r, 4), float(np.mean(vals <= r)) if vals.size else 0.0, lbl) for r, _c, lbl in frontiers]
        print(
            f"p_depol={_p_tag(p_depol)} shots={shots}: frontiers (max_rho, accepted_frac, label) = "
            f"{[(r, round(f, 2), lbl) for r, f, lbl in accepted]}"
        )

    vmin, vmax = _color_range(grid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")

    t_ticks = [i for i, t in enumerate(ts) if t % 5 == 0]
    n_ticks = [i for i, n in enumerate(ns) if n % 5 == 0]
    ax.set_xticks(t_ticks or range(len(ts)))
    ax.set_xticklabels([str(ts[i]) for i in (t_ticks or range(len(ts)))])
    ax.set_yticks(n_ticks or range(len(ns)))
    ax.set_yticklabels([str(ns[i]) for i in (n_ticks or range(len(ns)))])
    ax.set_xlabel(r"$t$ (injection layers)")
    ax.set_ylabel(r"$n$ (data qubits)")
    ax.set_title(
        rf"Noise impact on failure rate per circuit dimension "
        rf"($p_{{\mathrm{{depol}}}}={_p_latex(p_depol)}$, $N_{{\mathrm{{shots}}}}={shots}$)"
    )

    if annotate:
        for i in range(len(ns)):
            for j in range(len(ts)):
                v = grid[i, j]
                text = "NA" if np.isnan(v) else (f"{v:.1e}" if 0 < v < 0.001 else f"{v:.3f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=6)

    if frontiers:
        _draw_frontiers(ax, grid, frontiers)
        handles = [mlines.Line2D([0], [0], color=c, lw=1.8, label=lbl) for _r, c, lbl in frontiers]
        ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.85)

    fig.colorbar(im, ax=ax, label=_METRIC_LABEL.get(metric, metric))
    plt.tight_layout()

    outpath = outdir / f"discrete_{metric}_p{_p_tag(p_depol)}_s{shots}_bqp{bqp_error}.pdf"
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
    parser.add_argument("--autofrontier", dest="autofrontier", action="store_true", default=AUTOFRONTIER)
    parser.add_argument("--no-autofrontier", dest="autofrontier", action="store_false")
    args = parser.parse_args()

    manual = args.N is not None or args.epsilon is not None
    csv_paths = CSV_FILES if args.csv is None else [Path(x.strip()) for x in args.csv.split(",") if x.strip()]
    outdir = Path(args.outdir) if args.outdir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    # Frontiers only make sense for the failure-rate metric (compared against w/s).
    want_frontiers = args.metric == "p_failed_round"
    candidates = (
        _candidate_max_rhos(args.bqp_error, args.detection_rate)
        if (want_frontiers and args.autofrontier and not manual) else []
    )

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"!! skipping missing CSV: {csv_path}")
            continue
        shots = _shots_from_name(csv_path)
        df = pd.read_csv(csv_path)
        p_depols = sorted(df["p_depol"].unique()) if args.p_depol is None else [args.p_depol]
        for p_depol in p_depols:
            sub = df[df["p_depol"] == p_depol]
            if sub.empty:
                continue
            grid, ts, ns = _build_grid(sub, args.metric)

            if not want_frontiers:
                frontiers: list[tuple[float, tuple, str]] = []
            elif manual:
                n_vals = [int(x) for x in args.N.split(",")] if args.N else [1200, 2000]
                eps_vals = [float(x) for x in args.epsilon.split(",")] if args.epsilon else [1e-5, 1e-7]
                frontiers = _frontier_configs(args.bqp_error, args.detection_rate, n_vals, eps_vals)
            else:
                frontiers = _auto_frontier_configs(grid, candidates)

            _plot_one(grid, ts, ns, p_depol, shots, args.metric, args.bqp_error, frontiers, outdir, args.annotate)


if __name__ == "__main__":
    main()

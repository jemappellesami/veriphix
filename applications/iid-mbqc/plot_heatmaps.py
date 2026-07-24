#!/usr/bin/env python3
"""Width-depth heatmaps: measured noise, then the same grid read through the pipeline.

Three maps per (CSV, noise level), the second and third derived from the first by
``iid_pipeline.py``:

  1. ``q_U``          Clopper-Pearson upper bound on the test-failure rate. Not the raw
                      rate: the raw rate is a point estimate, and the guarantee needs the
                      bound. The gap between them is what the benchmarking stage's size buys.
  2. ``d``            computation rounds needed at a fixed correctness target.
  3. ``eps_total``    correctness achieved at a fixed computation-round budget.

Tiles past the wall (``q_U >= alpha/k``, where majority voting cannot help) are hatched in
2 and 3 rather than coloured, so the certifiable region is the readable feature.

Input schema: ``experiment.py`` output is used directly. Older sweeps are also accepted --
``width``/``depth`` may be named ``n``/``t``, the noise column may be ``p_ent`` or
``p_depol``, and when ``n_fail``/``n_rounds`` are absent the count is reconstructed from
``p_failed_round`` with ``s`` read from the ``_r<N>``/``_s<N>`` filename suffix. That
reconstruction is a rounding, so prefer files that carry the integers.

Usage:
    python applications/iid-mbqc/plot_heatmaps.py
    python applications/iid-mbqc/plot_heatmaps.py --csv results/mbqc_iid.csv --eps-target 1e-9
    python applications/iid-mbqc/plot_heatmaps.py \
        --csv ../benchmark-stim/benchmark_stim_results_p1.0e-03_r10000.csv
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iid_pipeline import (
    K_TESTS,
    admissible_q_threshold,
    clopper_pearson_upper,
    computation_error_upper,
    majority_error,
    optimal_allocation,
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

_HERE = Path(__file__).resolve().parent
DEFAULT_CSV = _HERE / "results" / "mbqc_iid.csv"
OUTDIR = _HERE / "figures"

_WIDTH_COLS = ("width", "n")
_DEPTH_COLS = ("depth", "t")
_NOISE_COLS = ("p_ent", "p_depol")
_COUNT_RE = re.compile(r"_[sr](\d+)")


def _pick(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _eps_latex(eps: float) -> str:
    return rf"10^{{{round(math.log10(eps))}}}"


def _counts(df: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(Y, s)`` per row, from the integer columns when present."""
    if "n_fail" in df.columns and "n_rounds" in df.columns:
        return df["n_fail"].to_numpy(dtype=int), df["n_rounds"].to_numpy(dtype=int)
    m = _COUNT_RE.search(path.stem)
    if m is None or "p_failed_round" not in df.columns:
        raise ValueError(f"{path.name}: no n_fail/n_rounds and no _r<N>/_s<N> suffix to fall back on")
    s = int(m.group(1))
    print(f"   {path.name}: no integer counts; reconstructing Y = round(q_hat * {s})")
    return np.round(df["p_failed_round"].to_numpy(dtype=float) * s).astype(int), np.full(len(df), s)


def _grid(sub: pd.DataFrame, w_col: str, d_col: str, value: str) -> tuple[np.ndarray, list, list]:
    depths = sorted(sub[d_col].unique())
    widths = sorted(sub[w_col].unique(), reverse=True)
    grid = (
        sub.pivot_table(index=w_col, columns=d_col, values=value, aggfunc="mean")
        .reindex(index=widths, columns=depths)
        .to_numpy(dtype=float)
    )
    return grid, depths, widths


def _draw(
    grid: np.ndarray,
    infeasible: np.ndarray | None,
    depths: list,
    widths: list,
    title: str,
    cbar_label: str,
    outpath: Path,
    log_scale: bool,
    footnote: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("lightgrey")
    norm = matplotlib.colors.LogNorm() if log_scale else None
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto", norm=norm)

    if infeasible is not None:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if infeasible[i, j]:
                    ax.add_patch(
                        matplotlib.patches.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="xxx", edgecolor="grey", lw=0.0
                        )
                    )

    step_d = max(1, len(depths) // 10)
    step_w = max(1, len(widths) // 10)
    ax.set_xticks(range(0, len(depths), step_d))
    ax.set_xticklabels([str(depths[i]) for i in range(0, len(depths), step_d)])
    ax.set_yticks(range(0, len(widths), step_w))
    ax.set_yticklabels([str(widths[i]) for i in range(0, len(widths), step_w)])
    ax.set_xlabel("depth (brickwork layers)")
    ax.set_ylabel("width (logical wires)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=7)
    plt.tight_layout(rect=(0, 0.02, 1, 1) if footnote else None)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved -> {outpath}")
    return outpath


def plot_csv(
    path: Path, eps_target: float, eps_bench: float, budget: int, k: int, c: float, outdir: Path
) -> None:
    df = pd.read_csv(path, comment="#")
    w_col, d_col = _pick(df, _WIDTH_COLS), _pick(df, _DEPTH_COLS)
    if w_col is None or d_col is None:
        print(f"!! {path.name}: no width/depth columns, skipping")
        return
    noise_col = _pick(df, _NOISE_COLS)

    failures, rounds = _counts(df, path)
    threshold = admissible_q_threshold(k=k, c=c)

    df = df.assign(
        _q_upper=[clopper_pearson_upper(int(y), int(s), eps_bench) for y, s in zip(failures, rounds, strict=True)],
        _y=failures,
        _s=rounds,
    )
    df["_p_upper"] = [computation_error_upper(q, k=k, c=c) for q in df["_q_upper"]]

    # d at the fixed target, and eps at the fixed budget; NaN past the wall.
    d_vals, eps_vals = [], []
    for y, s, p_u in zip(df["_y"], df["_s"], df["_p_upper"], strict=True):
        if p_u >= 0.5:
            d_vals.append(np.nan)
            eps_vals.append(np.nan)
            continue
        res = optimal_allocation(int(y), int(s), eps_target, k=k, c=c)
        d_vals.append(res.d if res.d is not None else np.nan)
        eps_vals.append(min(1.0, eps_bench + majority_error(budget, p_u)))
    df["_d"] = d_vals
    df["_eps"] = eps_vals

    groups = df.groupby(noise_col) if noise_col else [(None, df)]
    for noise, sub in groups:
        tag = f"_p{noise:.1e}" if noise is not None else ""
        label = rf" ($p={noise:.1e}$)" if noise is not None else ""
        s_note = f"s={int(sub['_s'].iloc[0])}" if sub["_s"].nunique() == 1 else "mixed s"

        q_grid, depths, widths = _grid(sub, w_col, d_col, "_q_upper")
        p_grid, _, _ = _grid(sub, w_col, d_col, "_p_upper")
        infeasible = ~np.isnan(p_grid) & (p_grid >= 0.5)
        n_bad = int(infeasible.sum())
        n_tot = int((~np.isnan(p_grid)).sum())
        print(f"   {tag or 'all'}: {n_tot} tiles, {n_bad} past the wall ({100 * n_bad / max(n_tot, 1):.0f}%)")

        _draw(
            q_grid,
            infeasible,
            depths,
            widths,
            rf"Noise bound $q_U$ per circuit dimension{label}, {s_note}",
            r"Clopper--Pearson $q_U$",
            outdir / f"heat_qU_{path.stem}{tag}.pdf",
            log_scale=False,
            footnote=rf"hatched: $q_U \geq \alpha/k = {threshold:.3f}$, majority voting cannot help",
        )

        d_grid, _, _ = _grid(sub, w_col, d_col, "_d")
        _draw(
            d_grid,
            infeasible,
            depths,
            widths,
            rf"Computation rounds $d$ at $\epsilon_{{\mathrm{{target}}}}={_eps_latex(eps_target)}${label}",
            r"computation rounds $d$",
            outdir / f"heat_d_{path.stem}{tag}.pdf",
            log_scale=True,
            footnote=rf"hatched: not certifiable at any $d$; {s_note}",
        )

        eps_grid, _, _ = _grid(sub, w_col, d_col, "_eps")
        _draw(
            eps_grid,
            infeasible,
            depths,
            widths,
            rf"Achieved $\epsilon_{{\mathrm{{total}}}}$ at $d={budget}${label}",
            r"$\epsilon_{\mathrm{total}}$",
            outdir / f"heat_eps_{path.stem}{tag}.pdf",
            log_scale=True,
            footnote=rf"floor is $\epsilon_{{\mathrm{{bench}}}}={_eps_latex(eps_bench)}$; {s_note}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="*", type=Path, default=[DEFAULT_CSV])
    parser.add_argument("--eps-target", type=float, default=1e-6)
    parser.add_argument("--eps-bench", type=float, default=1e-9)
    parser.add_argument("--budget", type=int, default=101, help="odd d for the fixed-budget map")
    parser.add_argument("--k", type=int, default=K_TESTS)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"k={args.k} c={args.c} -> wall q_U < {admissible_q_threshold(k=args.k, c=args.c):.6f}")
    for path in args.csv:
        if not path.exists():
            print(f"!! missing: {path}")
            continue
        print(f"{path.name}:")
        plot_csv(path, args.eps_target, args.eps_bench, args.budget, args.k, args.c, args.outdir)


if __name__ == "__main__":
    main()

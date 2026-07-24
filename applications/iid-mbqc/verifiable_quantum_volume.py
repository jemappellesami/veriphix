#!/usr/bin/env python3
"""Verifiable quantum volume: the largest circuits a device can run at a target precision
within a fixed number of computation rounds.

The third application on top of the measured noise landscape, after the cost-driven and
performance-driven readings. Those two ask what a *given* tile costs or delivers; this one
turns the question around and asks how much of the grid is reachable at all::

    (d, eps_correctness)  -->  q_crit  -->  { tiles : q_U <= q_crit }  -->  volume

``q_crit`` is the largest noise bound a budget of ``d`` majority-voted rounds can still
absorb while meeting ``eps_correctness``. Since the majority error is increasing in q_U, it
is found by bisection. Every tile whose Clopper-Pearson bound sits below it is certifiable;
the boundary of that set is the frontier, and its extent is the volume.

Reported volume metrics (a (width, depth) grid admits several honest readings):
  * ``max_square``  largest n with tile (n, n) certified -- the closest analogue of the
    usual quantum volume, reported alongside ``2^n``;
  * ``max_area``    largest certified width*depth -- volume in the literal sense;
  * ``max_width`` / ``max_depth`` -- the extent along each axis separately;
  * ``n_pass`` / ``frac_pass`` -- how much of the measured grid is certifiable.

The inverse direction is what the example below uses: pick a fraction of the grid you want
to certify, read off the ``q_crit`` that achieves it, then ask which ``(d, eps)`` pairs
tolerate that noise. That is the practical planning question -- "I want to cover a third of
my device's operating range; what does that cost me in rounds?"

Consumes an existing results CSV; never re-runs the experiment.

Usage:
    python applications/iid-mbqc/verifiable_quantum_volume.py
    python applications/iid-mbqc/verifiable_quantum_volume.py --d 101 --eps-target 1e-6
    python applications/iid-mbqc/verifiable_quantum_volume.py --fractions 0.15,0.30,0.50
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
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
    required_rounds,
)
from plot_heatmaps import _DEPTH_COLS, _NOISE_COLS, _WIDTH_COLS, _counts, _grid, _pick

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

# The worked example: certify this share of the measured grid, then price it.
DEFAULT_FRACTIONS = [0.15, 0.30]
# (d, eps) pairs are reported over these targets for each frontier.
EPS_MENU = [1e-2, 1e-4, 1e-6, 1e-9]
# Figure B sweeps the round budget over this range.
D_SWEEP = [11, 21, 51, 101, 201, 501, 1001, 5001, 10001]
FRONTIER_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:orange"]


def max_tolerated_noise(
    d: int, eps_target: float, eps_bench: float, k: int = K_TESTS, c: float = 0.0
) -> float | None:
    """Largest ``q_U`` a budget of ``d`` rounds absorbs while meeting ``eps_target``.

    ``eps_bench`` is spent on the confidence bound, so the vote must fit in the remainder;
    if the target does not even cover ``eps_bench`` the question is unanswerable and this
    returns None. The majority error is monotone in q_U, so a plain bisection suffices.
    """
    if d < 1 or d % 2 == 0:
        raise ValueError("Need an odd number of computation rounds.")
    if not 0.0 < eps_bench < eps_target < 1.0:
        return None

    eps_vote = eps_target - eps_bench
    hard_wall = admissible_q_threshold(k=k, c=c)

    def ok(q_u: float) -> bool:
        p_u = computation_error_upper(q_u, k=k, c=c)
        return p_u < 0.5 and majority_error(d, p_u) <= eps_vote

    if not ok(1e-12):
        return None  # even a noiseless device cannot meet this target with this d
    lo, hi = 1e-12, hard_wall
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def rounds_for_noise(
    q_crit: float, eps_target: float, eps_bench: float, k: int, c: float
) -> tuple[int | None, str]:
    """Inverse of the above: minimum odd d that tolerates ``q_crit`` at ``eps_target``.

    Returns ``(d, reason)``; ``d`` is None when the target is unreachable, and ``reason``
    separates the two ways that happens, which have completely different fixes:

    * ``eps<=eps_bench`` -- the target is at or below the confidence floor. No number of
      computation rounds helps; the benchmarking stage has to be re-run with a smaller
      ``eps_bench`` (which raises every q_U, so ``q_crit`` shifts too).
    * ``past wall`` -- the noise itself is too high for majority voting at any budget.
    """
    if eps_target <= eps_bench:
        return None, "eps<=eps_bench"
    p_u = computation_error_upper(q_crit, k=k, c=c)
    if p_u >= 0.5:
        return None, "past wall"
    return required_rounds(p_u, eps_target - eps_bench), ""


def volume_metrics(sub: pd.DataFrame, w_col: str, d_col: str, passing: np.ndarray) -> dict:
    """Extent of the certified region, in the several senses a 2-D grid allows."""
    ok = sub[passing]
    if ok.empty:
        return {
            "n_pass": 0,
            "frac_pass": 0.0,
            "max_width": None,
            "max_depth": None,
            "max_area": None,
            "max_square": None,
        }
    widths, depths = ok[w_col].to_numpy(), ok[d_col].to_numpy()
    squares = [int(w) for w, dd in zip(widths, depths, strict=True) if w == dd]
    return {
        "n_pass": len(ok),
        "frac_pass": float(len(ok) / len(sub)),
        "max_width": int(widths.max()),
        "max_depth": int(depths.max()),
        "max_area": int((widths * depths).max()),
        "max_square": max(squares) if squares else None,
    }


def _frontier_edges(safe: np.ndarray) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Boundary segments of the certified tile set, in imshow index coordinates."""
    edges = []
    rows, cols = safe.shape
    for i in range(rows):
        for j in range(cols):
            if not safe[i, j]:
                continue
            if i == 0 or not safe[i - 1, j]:
                edges.append(((j - 0.5, i - 0.5), (j + 0.5, i - 0.5)))
            if i == rows - 1 or not safe[i + 1, j]:
                edges.append(((j - 0.5, i + 0.5), (j + 0.5, i + 0.5)))
            if j == 0 or not safe[i, j - 1]:
                edges.append(((j - 0.5, i - 0.5), (j - 0.5, i + 0.5)))
            if j == cols - 1 or not safe[i, j + 1]:
                edges.append(((j + 0.5, i - 0.5), (j + 0.5, i + 0.5)))
    return edges


def figure_frontiers(
    sub: pd.DataFrame,
    w_col: str,
    d_col: str,
    frontiers: list[dict],
    noise: float | None,
    s_note: str,
    stem: str,
    outdir: Path,
) -> Path:
    """The q_U heatmap with one certification frontier per target fraction."""
    grid, depths, widths = _grid(sub, w_col, d_col, "_q_upper")
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto")

    handles = []
    for front, color in zip(frontiers, FRONTIER_COLORS, strict=False):
        safe = ~np.isnan(grid) & (grid <= front["q_crit"])
        for (x1, y1), (x2, y2) in _frontier_edges(safe):
            ax.plot([x1, x2], [y1, y2], color=color, lw=2.0, solid_capstyle="butt", zorder=5)
        pct = f"{100 * front['target_frac']:.0f}" + (r"\%" if _USETEX else "%")
        # Frontiers from the forward direction carry the (d, eps) that produced them; the
        # inverse ones were defined by the fraction, so the fraction is the whole label.
        origin = front.get("origin")
        q_txt = rf"$q_{{\mathrm{{crit}}}}={front['q_crit']:.4f}$" if _USETEX else f"q_crit={front['q_crit']:.4f}"
        # Inverse frontiers are named by the fraction that defined them; forward ones by the
        # (d, eps) that produced them, with the achieved fraction appended.
        label = f"{origin}: {q_txt} ({pct})" if origin else f"{pct}: {q_txt}"
        handles.append(mlines.Line2D([0], [0], color=color, lw=2.0, label=label))

    step_d = max(1, len(depths) // 10)
    step_w = max(1, len(widths) // 10)
    ax.set_xticks(range(0, len(depths), step_d))
    ax.set_xticklabels([str(depths[i]) for i in range(0, len(depths), step_d)])
    ax.set_yticks(range(0, len(widths), step_w))
    ax.set_yticklabels([str(widths[i]) for i in range(0, len(widths), step_w)])
    ax.set_xlabel("depth (brickwork layers)")
    ax.set_ylabel("width (logical wires)")
    label = rf" ($p={noise:.1e}$)" if noise is not None else ""
    ax.set_title(rf"Verifiable quantum volume: certification frontiers{label}, {s_note}")
    # Upper right: the noisiest corner, and never inside the certified region.
    ax.legend(handles=handles, loc="upper right", framealpha=0.92)
    fig.colorbar(im, ax=ax, label=r"Clopper--Pearson $q_U$")
    fig.text(
        0.5,
        0.005,
        "a frontier encloses the tiles certifiable at that budget; see the table for the (d, eps) pairs",
        ha="center",
        fontsize=7,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    out = outdir / f"E_vqv_frontiers_{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


def figure_volume_vs_budget(
    sub: pd.DataFrame,
    w_col: str,
    d_col: str,
    eps_bench: float,
    k: int,
    c: float,
    noise: float | None,
    stem: str,
    outdir: Path,
) -> Path:
    """How the certified region grows with the round budget, at several precision targets."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.3))
    cmap = plt.get_cmap("viridis")
    q_upper = sub["_q_upper"].to_numpy()
    widths, depths_arr = sub[w_col].to_numpy(), sub[d_col].to_numpy()

    for i, eps in enumerate(EPS_MENU):
        color = cmap(0.12 + 0.72 * i / max(len(EPS_MENU) - 1, 1))
        fracs, areas = [], []
        for d in D_SWEEP:
            q_crit = max_tolerated_noise(d, eps, eps_bench, k=k, c=c)
            if q_crit is None:
                fracs.append(np.nan)
                areas.append(np.nan)
                continue
            passing = q_upper <= q_crit
            fracs.append(100.0 * passing.mean())
            areas.append((widths * depths_arr)[passing].max() if passing.any() else np.nan)
        label = rf"$\epsilon={{10^{{{round(math.log10(eps))}}}}}$"
        ax_l.plot(D_SWEEP, fracs, color=color, lw=1.8, marker="o", ms=3, label=label)
        ax_r.plot(D_SWEEP, areas, color=color, lw=1.8, marker="o", ms=3, label=label)

    for ax, ylabel, title in (
        (ax_l, r"certified tiles (\%)" if _USETEX else "certified tiles (%)", "Share of the measured grid"),
        (ax_r, r"max certified width$\times$depth", "Largest certified circuit"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(r"computation rounds $d$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.legend(framealpha=0.9)
    ax_r.set_yscale("log")

    label = rf" ($p={noise:.1e}$)" if noise is not None else ""
    fig.suptitle(f"Verifiable quantum volume vs round budget{label}", fontsize=11)
    plt.tight_layout()
    out = outdir / f"F_vqv_vs_budget_{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


def analyse(
    path: Path,
    fractions: list[float],
    eps_bench: float,
    k: float,
    c: float,
    fixed_d: int | None,
    fixed_eps: float | None,
    outdir: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    w_col, d_col = _pick(df, _WIDTH_COLS), _pick(df, _DEPTH_COLS)
    if w_col is None or d_col is None:
        raise ValueError(f"{path.name}: no width/depth columns")
    noise_col = _pick(df, _NOISE_COLS)

    failures, rounds = _counts(df, path)
    df = df.assign(
        _q_upper=[clopper_pearson_upper(int(y), int(s), eps_bench) for y, s in zip(failures, rounds, strict=True)],
        _s=rounds,
    )

    records = []
    groups = df.groupby(noise_col) if noise_col else [(None, df)]
    for noise, group in groups:
        # Reset so the boolean masks below index the same way as the pivoted grid.
        sub = group.reset_index(drop=True)
        stem = f"{path.stem}" + (f"_p{noise:.1e}" if noise is not None else "")
        s_note = f"s={int(sub['_s'].iloc[0])}" if sub["_s"].nunique() == 1 else "mixed s"
        q_upper = sub["_q_upper"].to_numpy()
        hard_wall = admissible_q_threshold(k=k, c=c)

        header = f"{path.name}" + (f"  p={noise:.1e}" if noise is not None else "")
        print(f"\n{header}   {len(sub)} tiles, {s_note}, wall q_U < {hard_wall:.4f}")
        print(f"  q_U over the grid: min={q_upper.min():.4f} median={np.median(q_upper):.4f} max={q_upper.max():.4f}")
        if sub["_s"].nunique() > 1:
            # q_U depends on s, so an under-measured tile fails certification for having a
            # wide interval rather than for being noisy. A frontier over mixed s is not a
            # clean statement about the device.
            counts = sorted(int(v) for v in sub["_s"].unique())
            print(
                f"  !! mixed s across tiles ({counts}): q_U is not comparable between them, "
                f"so the frontier mixes noise with measurement effort. Re-run at a uniform "
                f"--rounds, or split the CSV by n_rounds."
            )

        frontiers = []
        for frac in fractions:
            q_crit = float(np.quantile(q_upper, frac))
            passing = q_upper <= q_crit
            metrics = volume_metrics(sub, w_col, d_col, passing)
            front = {"target_frac": frac, "q_crit": q_crit, **metrics}

            print(f"\n  --- certify {100 * frac:.0f}% of the grid -> q_crit = {q_crit:.4f} ---")
            if q_crit >= hard_wall:
                print("      past the hard wall: no (d, eps) tolerates this noise at any budget.")
            square = f"{metrics['max_square']} (2^{metrics['max_square']})" if metrics["max_square"] else "-"
            print(
                f"      certified: {metrics['n_pass']}/{len(sub)} tiles "
                f"({100 * metrics['frac_pass']:.0f}%), max width={metrics['max_width']}, "
                f"max depth={metrics['max_depth']}, max area={metrics['max_area']}, max square={square}"
            )
            print(f"      {'eps_target':>12} {'min d':>12}   (eps_bench = {eps_bench:.0e})")
            for eps in EPS_MENU:
                d_min, reason = rounds_for_noise(q_crit, eps, eps_bench, k, c)
                print(f"      {eps:>12.0e} {d_min if d_min else reason or 'infeasible':>12}")
                front[f"d_at_eps_{eps:.0e}"] = d_min
            records.append({"csv": path.name, "noise": noise, **front})
            frontiers.append(front)

        if fixed_d is not None and fixed_eps is not None:
            q_crit = max_tolerated_noise(fixed_d, fixed_eps, eps_bench, k=k, c=c)
            print(f"\n  --- forward direction: d={fixed_d}, eps={fixed_eps:.0e} ---")
            if q_crit is None:
                print("      no tolerable noise: target unreachable at this budget.")
            else:
                passing = q_upper <= q_crit
                metrics = volume_metrics(sub, w_col, d_col, passing)
                square = f"{metrics['max_square']} (2^{metrics['max_square']})" if metrics["max_square"] else "-"
                print(
                    f"      q_crit = {q_crit:.4f} -> {metrics['n_pass']}/{len(sub)} tiles "
                    f"({100 * metrics['frac_pass']:.0f}%), max area={metrics['max_area']}, max square={square}"
                )
                origin = (
                    rf"$d={fixed_d}$, $\epsilon={fixed_eps:.0e}$"
                    if _USETEX
                    else f"d={fixed_d}, eps={fixed_eps:.0e}"
                )
                frontiers.append(
                    {"target_frac": metrics["frac_pass"], "q_crit": q_crit, "origin": origin, **metrics}
                )
                records.append({"csv": path.name, "noise": noise, "target_frac": metrics["frac_pass"],
                                "q_crit": q_crit, "fixed_d": fixed_d, "fixed_eps": fixed_eps, **metrics})

        figure_frontiers(sub, w_col, d_col, frontiers, noise, s_note, stem, outdir)
        figure_volume_vs_budget(sub, w_col, d_col, eps_bench, k, c, noise, stem, outdir)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="*", type=Path, default=[DEFAULT_CSV])
    parser.add_argument("--fractions", default=",".join(str(f) for f in DEFAULT_FRACTIONS))
    # Sits below the whole EPS_MENU: eps_bench is a floor on eps_total, and a target at or
    # under it is unreachable at any d. It also sets q_U, hence q_crit and every frontier.
    parser.add_argument("--eps-bench", type=float, default=1e-12)
    parser.add_argument("--d", type=int, default=None, help="forward direction: fixed round budget")
    parser.add_argument("--eps-target", type=float, default=None, help="forward direction: fixed target")
    parser.add_argument("--k", type=int, default=K_TESTS)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in args.csv:
        if not path.exists():
            print(f"!! missing: {path}")
            continue
        frames.append(
            analyse(path, fractions, args.eps_bench, args.k, args.c, args.d, args.eps_target, args.outdir)
        )

    if frames:
        out = args.outdir / "vqv_summary.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate i.i.d. circuit-noise benchmarking plots.

The script implements two complementary views of majority-vote amplification:

1. Performance-driven benchmark:
   fixed target correctness error -> required number of computation rounds.
2. Cost-driven benchmark:
   fixed number of computation rounds -> achieved correctness error.

It also accepts benchmark data indexed by circuit dimension. With raw test
counts, it computes a one-sided upper confidence bound on the test-failure
probability and produces dimension-dependent line plots or width/depth
heatmaps.

Default interpretation (``--noise-semantics test_failure``):
    p_err = q = average test/trap failure probability.
    A harmful round is detected with probability at least 1/k, so
        r <= k q,
    and the probability that one computation round is wrong is bounded by
        p_comp <= c + (1-c) min(1, k q).

For k=2 and c=0, majority amplification requires q < 1/4.

Example:
    python iid_benchmark_plots.py \
        --input-csv example_benchmark_data.csv \
        --output-dir figures \
        --k 2 --c 0 \
        --eps-target 1e-5 --eps-bench 1e-6 \
        --fixed-rounds 501

CSV input formats
-----------------
A. One-dimensional circuit family:
    circuit_dimension,test_rounds,failed_tests

B. Width/depth grid (heatmaps):
    circuit_width,circuit_depth,test_rounds,failed_tests

C. Instead of raw counts, either format may contain a column named ``p_err``.
   In that case ``p_err`` is treated as an already certified upper bound,
   interpreted according to ``--noise-semantics``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binom


SEMANTICS_LABELS = {
    "test_failure": "Average test-failure probability q",
    "harmful_round": "Harmful-round probability r",
    "computation_error": "Per-round computation error p",
}


def odd_ceil(value: float) -> int:
    """Return the smallest positive odd integer not smaller than value."""
    n = max(1, int(math.ceil(value)))
    return n if n % 2 == 1 else n + 1


def hoeffding_upper_bound(failures: int, rounds: int, epsilon: float) -> float:
    """One-sided Hoeffding upper confidence bound for a Bernoulli rate."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0, 1)")
    q_hat = failures / rounds
    radius = math.sqrt(math.log(1.0 / epsilon) / (2.0 * rounds))
    return min(1.0, q_hat + radius)


def clopper_pearson_upper_bound(
    failures: int, rounds: int, epsilon: float
) -> float:
    """Exact one-sided Clopper-Pearson upper confidence bound."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not 0 <= failures <= rounds:
        raise ValueError("failures must satisfy 0 <= failures <= rounds")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0, 1)")
    if failures == rounds:
        return 1.0
    return float(beta.ppf(1.0 - epsilon, failures + 1, rounds - failures))


def benchmark_upper_bound(
    failures: int,
    rounds: int,
    epsilon: float,
    method: str,
) -> float:
    if method == "hoeffding":
        return hoeffding_upper_bound(failures, rounds, epsilon)
    if method == "clopper-pearson":
        return clopper_pearson_upper_bound(failures, rounds, epsilon)
    raise ValueError(f"Unknown confidence method: {method}")


def effective_computation_error(
    p_err: float,
    *,
    k: int,
    c: float,
    noise_semantics: str,
) -> float:
    """Convert a benchmarked noise rate into a computation-round error bound.

    Parameters
    ----------
    p_err:
        Certified upper bound whose meaning is selected by ``noise_semantics``.
    k:
        Number of test types. For the user's current analog project, k=2.
    c:
        Intrinsic error probability of the ideal computation round.
    noise_semantics:
        - test_failure: p_err=q and r<=kq;
        - harmful_round: p_err=r;
        - computation_error: p_err already equals the total computation error.
    """
    if not 0.0 <= p_err <= 1.0:
        raise ValueError("p_err must lie in [0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 <= c < 0.5:
        raise ValueError("c must lie in [0, 1/2)")

    if noise_semantics == "test_failure":
        harmful = min(1.0, k * p_err)
        return min(1.0, c + (1.0 - c) * harmful)
    if noise_semantics == "harmful_round":
        harmful = min(1.0, p_err)
        return min(1.0, c + (1.0 - c) * harmful)
    if noise_semantics == "computation_error":
        return p_err
    raise ValueError(f"Unknown noise semantics: {noise_semantics}")


def admissible_noise_threshold(*, k: int, c: float, noise_semantics: str) -> float:
    """Largest benchmarked rate for which the conservative p_comp bound is < 1/2."""
    if noise_semantics == "test_failure":
        alpha = (1.0 - 2.0 * c) / (2.0 - 2.0 * c)
        return alpha / k
    if noise_semantics == "harmful_round":
        return (1.0 - 2.0 * c) / (2.0 - 2.0 * c)
    if noise_semantics == "computation_error":
        return 0.5
    raise ValueError(f"Unknown noise semantics: {noise_semantics}")


def exact_majority_error(rounds: int, p_comp: float) -> float:
    """Exact probability that an odd-round majority vote is wrong."""
    if rounds <= 0 or rounds % 2 == 0:
        raise ValueError("rounds must be a positive odd integer")
    if not 0.0 <= p_comp <= 1.0:
        raise ValueError("p_comp must lie in [0, 1]")
    return float(binom.sf(rounds // 2, rounds, p_comp))


def hoeffding_majority_bound(rounds: int, p_comp: float) -> float:
    """Hoeffding upper bound, valid in the amplifiable regime p_comp < 1/2."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if p_comp >= 0.5:
        return 1.0
    gap = 0.5 - p_comp
    return math.exp(-2.0 * rounds * gap * gap)


def required_rounds_hoeffding(p_comp: float, epsilon_vote: float) -> float:
    """Sufficient odd number of rounds from Hoeffding; inf if not amplifiable."""
    if not 0 < epsilon_vote < 1:
        raise ValueError("epsilon_vote must lie in (0, 1)")
    if p_comp >= 0.5:
        return math.inf
    d = math.log(1.0 / epsilon_vote) / (2.0 * (0.5 - p_comp) ** 2)
    return float(odd_ceil(d))


def required_rounds_exact(
    p_comp: float,
    epsilon_vote: float,
    max_rounds: int = 2_000_001,
) -> float:
    """Smallest odd d with exact binomial-tail error <= epsilon_vote.

    Hoeffding supplies a guaranteed upper bracket, then a binary search over odd
    integers locates the exact minimum.
    """
    if p_comp >= 0.5:
        return math.inf
    if exact_majority_error(1, p_comp) <= epsilon_vote:
        return 1.0

    d_hoeffding = int(required_rounds_hoeffding(p_comp, epsilon_vote))
    hi_d = min(d_hoeffding, max_rounds if max_rounds % 2 else max_rounds - 1)
    if exact_majority_error(hi_d, p_comp) > epsilon_vote:
        return math.inf

    lo_index = 0  # d = 2*index + 1
    hi_index = (hi_d - 1) // 2
    while lo_index < hi_index:
        mid = (lo_index + hi_index) // 2
        d = 2 * mid + 1
        if exact_majority_error(d, p_comp) <= epsilon_vote:
            hi_index = mid
        else:
            lo_index = mid + 1
    return float(2 * lo_index + 1)


def total_correctness_error(
    rounds: int,
    p_comp: float,
    epsilon_bench: float,
    *,
    method: str,
) -> float:
    """Union-bound total: benchmark confidence failure + majority-vote failure."""
    if method == "exact":
        vote = exact_majority_error(rounds, p_comp)
    elif method == "hoeffding":
        vote = hoeffding_majority_bound(rounds, p_comp)
    else:
        raise ValueError("method must be 'exact' or 'hoeffding'")
    return min(1.0, epsilon_bench + vote)


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def save_performance_plot(
    output_path: Path,
    *,
    p_values: np.ndarray,
    k: int,
    c: float,
    noise_semantics: str,
    epsilon_target: float,
    epsilon_bench: float,
    max_rounds: int,
) -> pd.DataFrame:
    epsilon_vote = epsilon_target - epsilon_bench
    if epsilon_vote <= 0:
        raise ValueError("eps-target must be strictly larger than eps-bench")

    p_comp_values = np.array(
        [
            effective_computation_error(
                p, k=k, c=c, noise_semantics=noise_semantics
            )
            for p in p_values
        ]
    )
    d_exact = np.array(
        [required_rounds_exact(p, epsilon_vote, max_rounds) for p in p_comp_values]
    )
    d_hoeffding = np.array(
        [required_rounds_hoeffding(p, epsilon_vote) for p in p_comp_values]
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(p_values, d_exact, label="Exact binomial minimum")
    plt.plot(p_values, d_hoeffding, linestyle="--", label="Hoeffding sufficient bound")
    plt.xlabel(SEMANTICS_LABELS[noise_semantics])
    plt.ylabel("Required computation rounds d")
    plt.title(
        "Performance-driven benchmark\n"
        f"target total correctness error = {epsilon_target:.1e}"
    )
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return pd.DataFrame(
        {
            "p_err": p_values,
            "p_comp_upper": p_comp_values,
            "required_rounds_exact": d_exact,
            "required_rounds_hoeffding": d_hoeffding,
        }
    )


def save_cost_plot(
    output_path: Path,
    *,
    p_values: np.ndarray,
    k: int,
    c: float,
    noise_semantics: str,
    fixed_rounds: int,
    epsilon_bench: float,
) -> pd.DataFrame:
    if fixed_rounds % 2 == 0:
        raise ValueError("fixed-rounds must be odd")

    p_comp_values = np.array(
        [
            effective_computation_error(
                p, k=k, c=c, noise_semantics=noise_semantics
            )
            for p in p_values
        ]
    )
    eps_exact = np.array(
        [
            total_correctness_error(
                fixed_rounds, p, epsilon_bench, method="exact"
            )
            for p in p_comp_values
        ]
    )
    eps_hoeffding = np.array(
        [
            total_correctness_error(
                fixed_rounds, p, epsilon_bench, method="hoeffding"
            )
            for p in p_comp_values
        ]
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(p_values, eps_exact, label="Exact binomial tail + benchmark epsilon")
    plt.plot(
        p_values,
        eps_hoeffding,
        linestyle="--",
        label="Hoeffding bound + benchmark epsilon",
    )
    plt.xlabel(SEMANTICS_LABELS[noise_semantics])
    plt.ylabel("Total correctness-error bound")
    plt.title(f"Cost-driven benchmark\nfixed computation rounds d = {fixed_rounds}")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return pd.DataFrame(
        {
            "p_err": p_values,
            "p_comp_upper": p_comp_values,
            "epsilon_total_exact": eps_exact,
            "epsilon_total_hoeffding": eps_hoeffding,
        }
    )


def enrich_benchmark_data(
    frame: pd.DataFrame,
    *,
    confidence_method: str,
    epsilon_bench: float,
    epsilon_target: float,
    fixed_rounds: int,
    k: int,
    c: float,
    noise_semantics: str,
    max_rounds: int,
) -> pd.DataFrame:
    out = frame.copy()

    if "p_err" in out.columns:
        out["p_err_upper"] = out["p_err"].astype(float)
        out["p_err_empirical"] = out["p_err"].astype(float)
    elif {"test_rounds", "failed_tests"}.issubset(out.columns):
        out["test_rounds"] = out["test_rounds"].astype(int)
        out["failed_tests"] = out["failed_tests"].astype(int)
        out["p_err_empirical"] = out["failed_tests"] / out["test_rounds"]
        out["p_err_upper"] = [
            benchmark_upper_bound(int(y), int(s), epsilon_bench, confidence_method)
            for y, s in zip(out["failed_tests"], out["test_rounds"])
        ]
    else:
        raise ValueError(
            "CSV must contain either p_err or both test_rounds and failed_tests"
        )

    out["p_comp_upper"] = [
        effective_computation_error(
            float(p), k=k, c=c, noise_semantics=noise_semantics
        )
        for p in out["p_err_upper"]
    ]

    epsilon_vote = epsilon_target - epsilon_bench
    if epsilon_vote <= 0:
        raise ValueError("eps-target must be strictly larger than eps-bench")

    out["required_rounds_exact"] = [
        required_rounds_exact(float(p), epsilon_vote, max_rounds)
        for p in out["p_comp_upper"]
    ]
    out["required_rounds_hoeffding"] = [
        required_rounds_hoeffding(float(p), epsilon_vote)
        for p in out["p_comp_upper"]
    ]
    out["epsilon_total_exact_fixed_d"] = [
        total_correctness_error(
            fixed_rounds, float(p), epsilon_bench, method="exact"
        )
        for p in out["p_comp_upper"]
    ]
    out["epsilon_total_hoeffding_fixed_d"] = [
        total_correctness_error(
            fixed_rounds, float(p), epsilon_bench, method="hoeffding"
        )
        for p in out["p_comp_upper"]
    ]
    return out


def save_dimension_line_plots(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    fixed_rounds: int,
    epsilon_target: float,
) -> None:
    x_col = "circuit_dimension"
    ordered = frame.sort_values(x_col)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(ordered[x_col], ordered["p_err_empirical"], marker="o", label="Empirical")
    plt.plot(ordered[x_col], ordered["p_err_upper"], marker="o", label="Upper confidence bound")
    plt.xlabel("Circuit dimension")
    plt.ylabel("Benchmarked noise rate")
    plt.title("Circuit-level noise versus circuit dimension")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dimension_noise.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        ordered[x_col],
        ordered["required_rounds_exact"],
        marker="o",
        label="Exact binomial minimum",
    )
    plt.plot(
        ordered[x_col],
        ordered["required_rounds_hoeffding"],
        marker="o",
        linestyle="--",
        label="Hoeffding sufficient bound",
    )
    plt.xlabel("Circuit dimension")
    plt.ylabel("Required computation rounds d")
    plt.title(f"Performance by circuit dimension, target epsilon = {epsilon_target:.1e}")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dimension_required_rounds.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        ordered[x_col],
        ordered["epsilon_total_exact_fixed_d"],
        marker="o",
        label="Exact binomial tail",
    )
    plt.plot(
        ordered[x_col],
        ordered["epsilon_total_hoeffding_fixed_d"],
        marker="o",
        linestyle="--",
        label="Hoeffding bound",
    )
    plt.xlabel("Circuit dimension")
    plt.ylabel("Total correctness-error bound")
    plt.title(f"Cost by circuit dimension, fixed d = {fixed_rounds}")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dimension_achieved_epsilon.png", dpi=220)
    plt.close()


def save_heatmap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
    log_values: bool = False,
) -> None:
    pivot = frame.pivot(
        index="circuit_width", columns="circuit_depth", values=value_col
    ).sort_index().sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)

    display_values = np.log10(values) if log_values else values
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(display_values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Circuit width")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        f"log10({colorbar_label})" if log_values else colorbar_label
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate i.i.d. noise performance- and cost-driven benchmarks."
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--k", type=int, default=2, help="Number of test types.")
    parser.add_argument(
        "--c",
        type=float,
        default=0.0,
        help="Intrinsic ideal computation error per round.",
    )
    parser.add_argument(
        "--noise-semantics",
        choices=tuple(SEMANTICS_LABELS),
        default="test_failure",
    )
    parser.add_argument(
        "--eps-target",
        type=float,
        default=1e-5,
        help="Target total correctness error for the performance plot.",
    )
    parser.add_argument(
        "--eps-bench",
        type=float,
        default=1e-6,
        help="Failure probability of the benchmark confidence statement.",
    )
    parser.add_argument(
        "--fixed-rounds",
        type=int,
        default=501,
        help="Odd computation-round budget for the cost plot.",
    )
    parser.add_argument(
        "--confidence-method",
        choices=("clopper-pearson", "hoeffding"),
        default="clopper-pearson",
    )
    parser.add_argument("--points", type=int, default=140)
    parser.add_argument(
        "--p-max",
        type=float,
        default=None,
        help="Maximum benchmarked noise on theoretical plots. Default: 96%% of threshold.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2_000_001,
        help="Search cap for the exact required-round calculation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fixed_rounds <= 0 or args.fixed_rounds % 2 == 0:
        raise SystemExit("--fixed-rounds must be a positive odd integer")
    if not 0 < args.eps_bench < args.eps_target < 1:
        raise SystemExit("Require 0 < eps-bench < eps-target < 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    threshold = admissible_noise_threshold(
        k=args.k, c=args.c, noise_semantics=args.noise_semantics
    )
    p_max = args.p_max if args.p_max is not None else 0.96 * threshold
    if not 0 < p_max < threshold:
        raise SystemExit(
            f"p-max must lie in (0, {threshold:.6g}) for the chosen model"
        )
    p_values = np.linspace(0.0, p_max, args.points)

    performance = save_performance_plot(
        args.output_dir / "performance_fixed_epsilon.png",
        p_values=p_values,
        k=args.k,
        c=args.c,
        noise_semantics=args.noise_semantics,
        epsilon_target=args.eps_target,
        epsilon_bench=args.eps_bench,
        max_rounds=args.max_rounds,
    )
    performance.to_csv(args.output_dir / "performance_fixed_epsilon.csv", index=False)

    cost = save_cost_plot(
        args.output_dir / "cost_fixed_rounds.png",
        p_values=p_values,
        k=args.k,
        c=args.c,
        noise_semantics=args.noise_semantics,
        fixed_rounds=args.fixed_rounds,
        epsilon_bench=args.eps_bench,
    )
    cost.to_csv(args.output_dir / "cost_fixed_rounds.csv", index=False)

    if args.input_csv is not None:
        frame = pd.read_csv(args.input_csv)
        enriched = enrich_benchmark_data(
            frame,
            confidence_method=args.confidence_method,
            epsilon_bench=args.eps_bench,
            epsilon_target=args.eps_target,
            fixed_rounds=args.fixed_rounds,
            k=args.k,
            c=args.c,
            noise_semantics=args.noise_semantics,
            max_rounds=args.max_rounds,
        )
        enriched.to_csv(args.output_dir / "benchmark_summary.csv", index=False)

        if "circuit_dimension" in enriched.columns:
            save_dimension_line_plots(
                enriched,
                args.output_dir,
                fixed_rounds=args.fixed_rounds,
                epsilon_target=args.eps_target,
            )
        elif {"circuit_width", "circuit_depth"}.issubset(enriched.columns):
            save_heatmap(
                enriched,
                value_col="p_err_upper",
                output_path=args.output_dir / "heatmap_noise_upper.png",
                title="Upper confidence bound on circuit-level noise",
                colorbar_label="p_err upper bound",
            )
            save_heatmap(
                enriched,
                value_col="required_rounds_exact",
                output_path=args.output_dir / "heatmap_required_rounds.png",
                title=(
                    "Performance-driven benchmark: required rounds\n"
                    f"target total epsilon = {args.eps_target:.1e}"
                ),
                colorbar_label="required rounds",
                log_values=True,
            )
            save_heatmap(
                enriched,
                value_col="epsilon_total_exact_fixed_d",
                output_path=args.output_dir / "heatmap_achieved_epsilon.png",
                title=(
                    "Cost-driven benchmark: achieved correctness error\n"
                    f"fixed d = {args.fixed_rounds}"
                ),
                colorbar_label="total correctness error",
                log_values=True,
            )
        else:
            raise SystemExit(
                "CSV needs circuit_dimension or both circuit_width and circuit_depth"
            )

    print(f"Wrote plots and tables to: {args.output_dir.resolve()}")
    print(f"Amplifiable-noise threshold in the selected semantics: {threshold:.8f}")


if __name__ == "__main__":
    main()

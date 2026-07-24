#!/usr/bin/env python3
"""The i.i.d. benchmarking pipeline: test-failure counts -> correctness guarantee.

This is the statistical half of the benchmark. It contains no simulation and no plotting,
so it can be unit-tested and re-used on cluster data without importing graphix or stim.

The pipeline
------------
The device is modelled not as malicious but as an unknown noise process that is identical
and independent across rounds. Under that assumption a test round is a Bernoulli sample and
the whole chain is four steps::

    (Y, s)  --Clopper-Pearson-->  q_U  --r<=kq-->  p_U  --majority vote-->  eps_vote
                                                                             |
                                              eps_total = eps_bench + eps_vote

where

    Y         failed test rounds observed
    s         test rounds run
    q         probability that a randomly chosen test round fails
    q_U       upper confidence bound on q, valid except with probability eps_bench
    r         probability that a round carries a harmful error;  r <= k q  (k test types)
    c         intrinsic error probability of the ideal computation
    p_U       per-round computation error bound, c + (1-c) min(1, k q_U)
    d         computation rounds, majority-voted

MBQC only. In the MBQC setting the FK12 trap colouring is native to the graph and gives
exactly two test-run types, so ``k = 2``; the same constant appears in the older
adversarial code as ``detection_rate = 1/2``. A harmful error may be caught by only one of
the two runs, hence the worst-case factor k.

Why the reported rate is q, not r or p: the experiment draws one of the two test runs
uniformly per round and reports failures over those rounds (see ``experiment.py``), so the
measured quantity is the average test-failure probability. The factor k is therefore
required and is not already baked into the data.

The wall
--------
Majority voting only helps when p_U < 1/2, which in terms of the measured rate is

    q_U < alpha / k,        alpha = (1 - 2c) / (2 - 2c).

For k=2, c=0 that is q_U < 1/4 -- the same 0.25 the adversarial analysis produced, since
alpha * detection_rate = alpha / k. Note the bound is on q_U, not on q_hat: a tile whose
raw rate clears the wall can still fail once estimator uncertainty is accounted for, and
how much room it loses is set by s.

Blindness is what makes this a statement about a *class*: because the server cannot tell a
test round from a computation round, nor which computation is running, q estimated on traps
applies to every MBQC computation on the same graph. That is the part XEB cannot do -- it
needs the ideal outcome, hence classical simulability, and reports an ensemble average.

Usage:
    python applications/iid-mbqc/iid_pipeline.py --failures 70 --rounds 1000
    python applications/iid-mbqc/iid_pipeline.py --failures 70 --rounds 1000 --d 200
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

from scipy.stats import beta, binom

# Two test-run types in the MBQC/FK12 bipartite trap colouring.
K_TESTS = 2


# ── stage 1: test observations -> noise confidence bound ────────────────────────────


def clopper_pearson_upper(failures: int, rounds: int, epsilon_bench: float) -> float:
    """One-sided exact upper confidence bound on q, with Pr[q > q_U] <= epsilon_bench.

    Defined as the largest q under which the observation would still not be a fluke::

        q_U = max { q : Pr[Bin(s, q) <= Y] >= epsilon_bench }

    The binomial CDF and the Beta CDF are the same identity read two ways
    (``Pr[Bin(s,q) <= Y] = Pr[Beta(Y+1, s-Y) >= q]``), which turns that maximisation into
    the quantile below -- no search, no inequality, no slack beyond the discreteness of the
    binomial itself.

    Preferred over Hoeffding because the slack adapts to what was observed. Hoeffding's
    penalty is additive and independent of Y, i.e. it charges the worst-case variance
    (q=1/2) even for a tile that failed twice in a thousand rounds -- exactly the clean
    tiles the benchmark most wants to certify.
    """
    if not 0 <= failures <= rounds:
        raise ValueError("Need 0 <= failures <= rounds.")
    if rounds <= 0:
        raise ValueError("Need rounds >= 1.")
    if not 0.0 < epsilon_bench < 1.0:
        raise ValueError("Need 0 < epsilon_bench < 1.")
    if failures >= rounds:
        return 1.0
    return float(beta.ppf(1.0 - epsilon_bench, failures + 1, rounds - failures))


def hoeffding_upper(failures: int, rounds: int, epsilon_bench: float) -> float:
    """Hoeffding upper bound on q. Kept for comparison and for the closed-form scaling law.

    ``q_hat + sqrt(log(1/eps) / 2s)`` -- note the slack does not depend on ``failures``.
    """
    if rounds <= 0:
        raise ValueError("Need rounds >= 1.")
    if not 0.0 < epsilon_bench < 1.0:
        raise ValueError("Need 0 < epsilon_bench < 1.")
    q_hat = failures / rounds
    return min(1.0, q_hat + math.sqrt(math.log(1.0 / epsilon_bench) / (2.0 * rounds)))


# ── stage 2: noise bound -> per-round computation error ─────────────────────────────


def computation_error_upper(q_upper: float, k: int = K_TESTS, c: float = 0.0) -> float:
    """``p_U = c + (1-c) min(1, k q_U)``: worst-case trap coverage, then intrinsic error."""
    if not 0.0 <= c < 1.0:
        raise ValueError("Need 0 <= c < 1.")
    if k < 1:
        raise ValueError("Need k >= 1.")
    return c + (1.0 - c) * min(1.0, k * q_upper)


def admissible_q_threshold(k: int = K_TESTS, c: float = 0.0) -> float:
    """The wall: majority voting helps only while ``q_U < alpha/k``."""
    alpha = (1.0 - 2.0 * c) / (2.0 - 2.0 * c)
    return alpha / k


# ── stage 3: per-round error -> majority-vote error ─────────────────────────────────


def majority_error(rounds: int, p_comp: float) -> float:
    """Exact probability that a majority vote over ``rounds`` odd rounds is wrong."""
    if rounds < 1 or rounds % 2 == 0:
        raise ValueError("Need an odd number of computation rounds.")
    if not 0.0 <= p_comp <= 1.0:
        raise ValueError("Need 0 <= p_comp <= 1.")
    return float(binom.sf((rounds - 1) // 2, rounds, p_comp))


def majority_error_hoeffding(rounds: int, p_comp: float) -> float:
    """``exp(-2 d (1/2 - p)^2)``. Looser than the exact tail; used for the scaling law."""
    if p_comp >= 0.5:
        return 1.0
    return math.exp(-2.0 * rounds * (0.5 - p_comp) ** 2)


def required_rounds(p_comp: float, epsilon_vote: float, d_max: int = 10**9) -> int | None:
    """Smallest odd ``d`` with exact majority error <= ``epsilon_vote``; None past the wall.

    The exact tail is monotone in d for p < 1/2, so this doubles to bracket and then
    bisects on ``d = 2m+1``. Typically a factor ~2 below the Hoeffding value.
    """
    if p_comp >= 0.5:
        return None
    if not 0.0 < epsilon_vote < 1.0:
        raise ValueError("Need 0 < epsilon_vote < 1.")

    def ok(m: int) -> bool:
        return majority_error(2 * m + 1, p_comp) <= epsilon_vote

    hi = 0
    while not ok(hi):
        hi = max(1, hi * 2)
        if 2 * hi + 1 > d_max:
            return None
    lo = hi // 2
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return 2 * lo + 1


def required_rounds_hoeffding(p_comp: float, epsilon_vote: float) -> float:
    """``log(1/eps) / (2 (1/2 - p)^2)`` -- the closed form, for the text and the asymptote."""
    if p_comp >= 0.5:
        return math.inf
    return math.log(1.0 / epsilon_vote) / (2.0 * (0.5 - p_comp) ** 2)


# ── the two driving modes ───────────────────────────────────────────────────────────


@dataclass
class IIDResult:
    """Every intermediate of one tile, so a CSV row can carry the whole derivation."""

    failures: int
    rounds: int
    q_hat: float
    epsilon_bench: float
    q_upper: float
    q_upper_hoeffding: float
    k: int
    c: float
    p_upper: float
    q_threshold: float
    feasible: bool
    d: int | None = None
    epsilon_vote: float | None = None
    epsilon_total: float | None = None
    d_hoeffding: float | None = None


def _base(failures: int, rounds: int, epsilon_bench: float, k: int, c: float) -> IIDResult:
    q_u = clopper_pearson_upper(failures, rounds, epsilon_bench)
    p_u = computation_error_upper(q_u, k=k, c=c)
    return IIDResult(
        failures=failures,
        rounds=rounds,
        q_hat=failures / rounds,
        epsilon_bench=epsilon_bench,
        q_upper=q_u,
        q_upper_hoeffding=hoeffding_upper(failures, rounds, epsilon_bench),
        k=k,
        c=c,
        p_upper=p_u,
        q_threshold=admissible_q_threshold(k=k, c=c),
        feasible=p_u < 0.5,
    )


def performance_driven(
    failures: int,
    rounds: int,
    epsilon_target: float,
    epsilon_bench: float,
    k: int = K_TESTS,
    c: float = 0.0,
) -> IIDResult:
    """Fixed correctness target -> how many computation rounds does it cost?

    The target is split by a union bound: ``eps_bench`` covers the confidence bound being
    wrong, the remainder is left for the vote.
    """
    if not 0.0 < epsilon_bench < epsilon_target < 1.0:
        raise ValueError("Need 0 < epsilon_bench < epsilon_target < 1.")
    res = _base(failures, rounds, epsilon_bench, k, c)
    res.epsilon_vote = epsilon_target - epsilon_bench
    if res.feasible:
        res.d = required_rounds(res.p_upper, res.epsilon_vote)
        res.d_hoeffding = required_rounds_hoeffding(res.p_upper, res.epsilon_vote)
        if res.d is not None:
            res.epsilon_total = epsilon_bench + majority_error(res.d, res.p_upper)
    return res


def cost_driven(
    failures: int,
    rounds: int,
    d: int,
    epsilon_bench: float,
    k: int = K_TESTS,
    c: float = 0.0,
) -> IIDResult:
    """Fixed computation-round budget -> what correctness does it achieve?"""
    if d < 1 or d % 2 == 0:
        raise ValueError("Need an odd number of computation rounds.")
    res = _base(failures, rounds, epsilon_bench, k, c)
    res.d = d
    vote = majority_error(d, res.p_upper) if res.feasible else 1.0
    res.epsilon_vote = vote
    res.epsilon_total = min(1.0, epsilon_bench + vote)
    return res


def optimal_allocation(
    failures: int,
    rounds: int,
    epsilon_target: float,
    k: int = K_TESTS,
    c: float = 0.0,
    n_grid: int = 60,
) -> IIDResult:
    """Pick the eps_bench / eps_vote split that minimises d, instead of guessing a half.

    Spending more of the budget on the benchmark tightens q_U (which enters d through the
    squared gap ``(1/2 - p_U)^2``) while costing the vote only a logarithm, so the optimum
    is usually far from an even split -- especially at small s, where q_U is the binding
    constraint.
    """
    best: IIDResult | None = None
    lo, hi = math.log10(epsilon_target) - 6.0, math.log10(epsilon_target * 0.999)
    for i in range(n_grid):
        eb = 10.0 ** (lo + (hi - lo) * i / (n_grid - 1))
        if not 0.0 < eb < epsilon_target:
            continue
        res = performance_driven(failures, rounds, epsilon_target, eb, k=k, c=c)
        if res.d is None:
            continue
        if best is None or best.d is None or res.d < best.d:
            best = res
    if best is None:
        return performance_driven(failures, rounds, epsilon_target, epsilon_target * 1e-3, k=k, c=c)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failures", type=int, required=True, help="Y, failed test rounds")
    parser.add_argument("--rounds", type=int, required=True, help="s, test rounds run")
    parser.add_argument("--eps-target", type=float, default=1e-6)
    parser.add_argument("--eps-bench", type=float, default=None, help="default: optimise the split")
    parser.add_argument("--d", type=int, default=None, help="cost-driven mode: fixed computation rounds")
    parser.add_argument("--k", type=int, default=K_TESTS)
    parser.add_argument("--c", type=float, default=0.0)
    args = parser.parse_args()

    if args.d is not None:
        res = cost_driven(
            args.failures, args.rounds, args.d, args.eps_bench or args.eps_target * 1e-3, k=args.k, c=args.c
        )
        mode = f"cost-driven (d={args.d} fixed)"
    elif args.eps_bench is None:
        res = optimal_allocation(args.failures, args.rounds, args.eps_target, k=args.k, c=args.c)
        mode = "performance-driven (allocation optimised)"
    else:
        res = performance_driven(
            args.failures, args.rounds, args.eps_target, args.eps_bench, k=args.k, c=args.c
        )
        mode = "performance-driven"

    print(mode)
    print("-" * len(mode))
    for key, value in asdict(res).items():
        if value is None:
            print(f"{key:>20} = -")
        elif isinstance(value, float):
            print(f"{key:>20} = {value:.6g}")
        else:
            print(f"{key:>20} = {value}")
    if not res.feasible:
        print(f"\n!! p_U = {res.p_upper:.4f} >= 1/2: past the wall, majority voting cannot help.")
        print(f"   need q_U < {res.q_threshold:.4f} (measured q_hat = {res.q_hat:.4f}).")


if __name__ == "__main__":
    main()

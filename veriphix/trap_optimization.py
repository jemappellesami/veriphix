"""Linear-programming optimisation of trap distributions.

This module implements the optimisation underlying the :class:`OptimizedTraps`
protocol — *Problem 1* of Kapourniotis et al., "Unifying Quantum Verification
and Error Detection" (arXiv:2206.00631, the "Optimisation of the Distribution
of Tests"):

    Given
      * a set of errors  ℰ  to be detected,
      * a set of feasible tests  ℋ,
      * a detection relation  R : ℋ × ℰ → {0,1}  (does test H detect error E?),
    find a distribution  p : ℋ → [0,1]  maximising the detection rate  ε  s.t.
      * Σ_H p(H) ≤ mass            (p is a (sub-)probability distribution)
      * ∀E ∈ ℰ : Σ_{H: R(H,E)=1} p(H) ≥ ε   (every error detected w.p. ≥ ε)

This is a maximin linear program: ``ε`` is the *worst-case* detection rate over
all errors in ℰ, and we push it as high as the test pool allows.  Its optimum is
the largest detection rate achievable with the available tests against the given
errors — for the canonical case (standard traps = independent sets of the graph,
errors = single-qubit deviations) it equals ``1/χ_f(G)``, the inverse fractional
chromatic number, achieved by a fractional graph colouring.

The **dual** solution is the optimal *attack*: a distribution of deviations that
achieves the minimal detection rate against the chosen tests (Remark following
Problem 1 in the paper).  :func:`solve_trap_distribution` returns it alongside
the primal so callers can visualise "which errors are hardest to catch".

Detection relation
------------------
A test round (canvas) fails iff *any* of its atomic traps flags, and an atomic
trap with stabiliser ``S`` flags on deviation ``E`` iff ``S`` anticommutes with
``E``.  So a test ``H`` detects ``E`` iff ``E`` anticommutes with at least one of
``H``'s atomic-trap stabilisers — an OR over the traps, *not* the product.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from veriphix.verifying import TestRun, build_stabilizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import stim


# ── feasible-test pools (ℋ) ─────────────────────────────────────────────────


def single_qubit_trap_pool(graph: nx.Graph) -> list[TestRun]:
    """Return one single-qubit-trap canvas per node (FK12-style atomic traps)."""
    n = len(graph)
    return [
        TestRun(graph=graph, nqubits=n, traps=frozenset({frozenset({node})}))
        for node in graph.nodes
    ]


def independent_set_pool(
    graph: nx.Graph,
    restrict_to: set[int] | None = None,
    max_sets: int | None = None,
) -> list[TestRun]:
    """Return one canvas per maximal independent set of the graph.

    Each maximal independent set ``I`` becomes a canvas whose atomic traps are the
    single-qubit traps ``{v}`` for ``v ∈ I``; these can be tested simultaneously
    because no two are adjacent.  Distributing over independent sets is exactly a
    *fractional graph colouring*, so the optimal distribution reaches the
    ``1/χ_f(G)`` detection rate.

    Maximal independent sets of ``G`` are the maximal cliques of the complement
    graph.  Restricting to *maximal* sets is without loss of optimality: any
    independent set is a subset of a maximal one, and supersets only increase
    each error's coverage.

    Parameters
    ----------
    graph : nx.Graph
        The resource graph.  Canvases (and their stabilisers) are always built
        over the *full* graph.
    restrict_to : set[int] | None
        If given, enumerate independent sets of the subgraph induced by these
        nodes only.  Use this when errors are localised to a known region (e.g. a
        learned noise heatmap): a canvas detects ``Z_v`` iff ``v`` is in its set,
        so only the noisy nodes need covering, and the number of independent sets
        shrinks from exponential-in-``|V|`` to exponential-in-``|region|``.
    max_sets : int | None
        Optional cap on the number of independent sets enumerated (the count can
        in principle be exponential).  When the cap is hit a warning is emitted
        and the (possibly suboptimal) truncated pool is returned.
    """
    n = len(graph)
    target = graph.subgraph(restrict_to) if restrict_to is not None else graph
    pool: list[TestRun] = []
    for i, clique in enumerate(nx.find_cliques(nx.complement(target))):
        if max_sets is not None and i >= max_sets:
            warnings.warn(
                f"independent_set_pool truncated to {max_sets} sets; "
                "the resulting distribution may be suboptimal.",
                stacklevel=2,
            )
            break
        pool.append(
            TestRun(graph=graph, nqubits=n, traps=frozenset({frozenset({v}) for v in clique}))
        )
    return pool


# ── detection relation R ─────────────────────────────────────────────────────


def multi_basis_independent_set_pool(
    graph: nx.Graph,
    restrict_to: set[int] | None = None,
    bases: tuple[str, ...] = ("X", "Y"),
    max_sets: int | None = None,
) -> list[TestRun]:
    """Independent-set canvases, each replicated in several measurement bases.

    Extends :func:`independent_set_pool` with *basis diversity*: every maximal
    independent set is offered as a separate test in each of ``bases``.  An
    independent set ``I`` tested in basis ``B`` places effective Pauli ``B`` on
    every node of ``I`` simultaneously (no inter-trap cross-talk, since the nodes
    are non-adjacent), letting the optimiser rotate each node's trap basis.

    Only ``X`` and ``Y`` are physical measurement bases in blind MBQC: blindness
    relies on a random ``+θ`` padding of the measurement angle, undone by a
    pre-``Z(θ)`` rotation, which works *only* because ``Z(θ)`` commutes through
    ``CZ``.  A ``Z``-basis (out-of-plane) measurement cannot be blinded, so the
    default is ``("X", "Y")``.  (Dummies — ``Z``-eigenstate *preparations* — are
    allowed, but they do not give a ``Z`` *measurement*.)

    Consequence: each X/Y trap detects exactly one of ``{X_v, Y_v}`` at a node
    (the only Pauli anticommuting both is ``Z``, which is unmeasurable), so the
    worst-case detection rate for noise containing *both* ``X`` and ``Y`` on a
    qubit is capped at ``1/2``.  The rate reaches ``1.0`` only for single-axis
    (biased / coherent) harmful noise — ``X``-only or ``Y``-only — by fixing the
    complementary basis.
    """
    n = len(graph)
    target = graph.subgraph(restrict_to) if restrict_to is not None else graph
    pool: list[TestRun] = []
    for i, clique in enumerate(nx.find_cliques(nx.complement(target))):
        if max_sets is not None and i >= max_sets:
            warnings.warn(
                f"multi_basis_independent_set_pool truncated to {max_sets} sets.",
                stacklevel=2,
            )
            break
        traps = frozenset({frozenset(clique)})
        for basis in bases:
            pool.append(TestRun(graph=graph, nqubits=n, traps=traps, meas_basis=basis))
    return pool


def mixed_basis_independent_set_pool(
    graph: nx.Graph,
    restrict_to: set[int] | None = None,
    bases: tuple[str, ...] = ("X", "Y"),
    max_sets: int | None = None,
    max_basis_combos: int | None = None,
) -> list[TestRun]:
    """Independent-set canvases with **per-node** measurement bases (mixed-basis traps).

    For each maximal independent set ``I`` this enumerates basis assignments
    ``b : I → bases``, so a single canvas can measure *different nodes in different
    bases simultaneously* (each node is its own atomic trap; the nodes are
    non-adjacent, so the dummies they place on shared neighbours are all ``Z`` and
    never conflict).

    This is strictly more general than :func:`multi_basis_independent_set_pool`,
    which forces every node in a set to share one basis — those per-set-uniform
    canvases are exactly the *constant* assignments here, a subset of this pool.
    Allowing mixed bases lets the optimiser, e.g., test one node in ``X`` and its
    independent neighbour in ``Y`` in the *same* round, which a uniform canvas
    cannot express.

    Only ``X`` and ``Y`` are physical measurement bases in blind MBQC (a ``Z``,
    out-of-plane, measurement cannot be blinded); dummies — ``Z``-eigenstate
    *preparations* — still arise automatically from the ``CZ`` conjugation.

    Parameters
    ----------
    bases : tuple[str, ...]
        The per-node basis alphabet (default ``("X", "Y")``).
    max_basis_combos : int | None
        Cap on the number of basis assignments per independent set.  An ``I`` of
        size ``k`` has ``|bases|**k`` assignments; pass this to truncate large sets.
    """
    n = len(graph)
    target = graph.subgraph(restrict_to) if restrict_to is not None else graph
    pool: list[TestRun] = []
    for i, clique in enumerate(nx.find_cliques(nx.complement(target))):
        if max_sets is not None and i >= max_sets:
            warnings.warn(
                f"mixed_basis_independent_set_pool truncated to {max_sets} sets.",
                stacklevel=2,
            )
            break
        clique = sorted(clique)
        traps = frozenset({frozenset({v}) for v in clique})  # per-node atomic traps (OR detection)
        for k, assignment in enumerate(product(bases, repeat=len(clique))):
            if max_basis_combos is not None and k >= max_basis_combos:
                break
            node_bases = dict(zip(clique, assignment))
            pool.append(TestRun(graph=graph, nqubits=n, traps=traps, meas_basis=node_bases))
    return pool


def build_detection_matrix(
    graph: nx.Graph,
    test_runs: Sequence[TestRun],
    errors: Sequence[stim.PauliString],
) -> np.ndarray:
    """Return the ``{0,1}`` detection matrix ``R`` of shape ``(|ℋ|, |ℰ|)``.

    ``R[i, j] = 1`` iff test ``test_runs[i]`` detects error ``errors[j]`` — i.e.
    the error anticommutes with at least one of the test's atomic-trap
    stabilisers.  Atomic-trap stabilisers are built in the test's own
    measurement basis (so X/Y/Z-basis traps are handled correctly) and memoised
    across tests.
    """
    n = len(graph)
    # Cache atomic-trap stabilisers by their per-node basis assignment.  The key is a
    # frozenset of (node, basis) pairs, which fully determines the stabiliser on a fixed
    # graph and handles both uniform-basis and mixed-basis traps.
    stab_cache: dict[frozenset[tuple[int, str]], stim.PauliString] = {}

    def atomic_stab(bases_key: frozenset[tuple[int, str]]) -> stim.PauliString:
        if bases_key not in stab_cache:
            trap = frozenset(node for node, _ in bases_key)
            stab_cache[bases_key] = build_stabilizer(graph, n, frozenset({trap}), dict(bases_key))
        return stab_cache[bases_key]

    matrix = np.zeros((len(test_runs), len(errors)), dtype=np.int8)
    for i, test_run in enumerate(test_runs):
        stabs = [
            atomic_stab(frozenset((v, test_run.basis_at(v)) for v in trap))
            for trap in test_run.traps
        ]
        for j, error in enumerate(errors):
            matrix[i, j] = int(any(not s.commutes(error) for s in stabs))
    return matrix


# ── the linear program ───────────────────────────────────────────────────────


@dataclass
class TrapLPResult:
    """Result of :func:`solve_trap_distribution`.

    Attributes
    ----------
    distribution : np.ndarray
        Optimal probability ``p(H)`` for each test, aligned with the ``test_runs``
        order passed to :func:`build_detection_matrix`.
    detection_rate : float
        The optimised worst-case detection rate ``ε``.
    adversary : np.ndarray
        Optimal attack: a distribution over the errors (the normalised dual
        marginals).  ``adversary[j]`` is the weight the worst-case adversary puts
        on ``errors[j]``.
    """

    distribution: np.ndarray
    detection_rate: float
    adversary: np.ndarray


def solve_trap_distribution(detection_matrix: np.ndarray, mass: float = 1.0) -> TrapLPResult:
    """Solve Problem 1 for the given detection matrix.

    Parameters
    ----------
    detection_matrix : np.ndarray
        The ``(|ℋ|, |ℰ|)`` matrix from :func:`build_detection_matrix`.
    mass : float
        Total probability mass assigned to tests (``Σ p = mass``).  ``1.0`` gives
        a proper distribution over tests; values ``< 1`` model leaving mass for
        non-test (e.g. computation) rounds.

    Returns
    -------
    TrapLPResult
    """
    n_tests, n_errors = detection_matrix.shape
    if n_errors == 0:
        # No errors to detect: any distribution is optimal; ε is unbounded → 1.
        uniform = np.full(n_tests, mass / n_tests) if n_tests else np.empty(0)
        return TrapLPResult(distribution=uniform, detection_rate=1.0, adversary=np.empty(0))

    # Variables x = [p_0 … p_{n_tests-1}, ε].  Maximise ε ⟺ minimise −ε.
    c = np.zeros(n_tests + 1)
    c[-1] = -1.0

    # Per-error coverage:  ε − Σ_i R[i,j] p_i ≤ 0   (one row per error j).
    a_ub = np.zeros((n_errors, n_tests + 1))
    a_ub[:, :n_tests] = -detection_matrix.T
    a_ub[:, -1] = 1.0
    b_ub = np.zeros(n_errors)

    # Normalisation: Σ p_i = mass.
    a_eq = np.zeros((1, n_tests + 1))
    a_eq[0, :n_tests] = 1.0
    b_eq = np.array([mass])

    bounds = [(0.0, None)] * n_tests + [(0.0, 1.0)]

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"trap-distribution LP failed: {res.message}")

    distribution = np.clip(res.x[:n_tests], 0.0, None)
    detection_rate = float(res.x[-1])

    # Dual marginals on the coverage constraints give the optimal attack.
    marginals = np.asarray(res.ineqlin.marginals, dtype=float)
    adversary = np.clip(-marginals, 0.0, None)  # ≤-constraint multipliers are ≤ 0
    total = adversary.sum()
    adversary = adversary / total if total > 0 else np.full(n_errors, 1.0 / n_errors)

    if detection_rate <= 1e-9:
        warnings.warn(
            "optimal detection rate is ~0 — some error in ℰ is undetectable by the "
            "given test pool (an all-zero column in the detection matrix).",
            stacklevel=2,
        )

    return TrapLPResult(distribution=distribution, detection_rate=detection_rate, adversary=adversary)

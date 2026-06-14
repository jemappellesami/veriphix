"""Turn a learned trap-failure heatmap into an error set ℰ for OptimizedTraps.

Everything here is *client-side*: the only input is the observed per-node
trap-failure rate from a previous verification experiment (e.g.
``applications/noise_learning``).  No ground-truth noise is used — the point is
that a client can build its error model purely from what it measured.

For per-node dephasing noise, the trap at node ``v`` flags iff a ``Z`` error
occurred at ``v`` that round, so the observed failure rate at ``v`` is an
estimate of that node's flip probability.  Nodes whose rate exceeds a threshold
become the support of the learned error set ``ℰ = {Z_v : v noisy}``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import stim

if TYPE_CHECKING:
    import networkx as nx


def load_learned_heatmap(
    results_dir: Path,
    pos_to_node: dict[tuple[int, int], int],
) -> dict[int, float]:
    """Aggregate observed trap-failure rates from per-circuit CSVs into ``node -> rate``.

    Reads the ``circuit_*.csv`` files written by the noise-learning simulation
    (columns ``node, col, row, failure_count, total_tests``) and averages the
    failure rate per node across all circuits.
    """
    failure_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    total_tests: defaultdict[tuple[int, int], int] = defaultdict(int)
    csv_files = sorted(results_dir.glob("circuit_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no circuit_*.csv files in {results_dir}")
    for csv_path in csv_files:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                pos = (int(row["col"]), int(row["row"]))
                failure_counts[pos] += int(row["failure_count"])
                total_tests[pos] += int(row["total_tests"])

    rates: dict[int, float] = {}
    for pos, total in total_tests.items():
        if total and pos in pos_to_node:
            rates[pos_to_node[pos]] = failure_counts[pos] / total
    return rates


def noisy_nodes(rates: dict[int, float], threshold: float) -> set[int]:
    """Return the nodes whose learned failure rate exceeds ``threshold``.

    TODO: replace the hard ``threshold`` with a principled finite-sample test.
    The per-node rate is an estimate from ``circuits × test_rounds`` samples, so
    a fixed cutoff is arbitrary (it only worked on the clean, bimodal simulated
    heatmap, where any value in ~[0.02, 0.2] selects the same nodes). Instead,
    take the per-node sample count ``N`` and the device baseline failure rate,
    and include ``v`` iff its rate is *statistically* above baseline, e.g.
        rate_v - z * sqrt(rate_v * (1 - rate_v) / N) > baseline
    This turns the knob into a confidence level ``delta`` with a real guarantee:
    "w.p. >= 1 - delta, every node with true flip prob > p* is in the error set."
    Bias toward inclusion: missing genuinely noisy nodes drops them from the
    error set (false security), which is worse than over-including quiet nodes
    (merely wasteful). A bimodal heatmap can also be cut at the histogram gap
    (Otsu's method) instead.
    """
    return {node for node, rate in rates.items() if rate > threshold}


def z_errors(graph: nx.Graph, nodes_subset: set[int]) -> list[stim.PauliString]:
    """Build single-qubit ``Z`` deviations on ``nodes_subset``.

    The Pauli strings are indexed in ``list(graph.nodes)`` order, matching the
    convention of :func:`veriphix.verifying.build_stabilizer`.
    """
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    errors: list[stim.PauliString] = []
    for v in sorted(nodes_subset):
        s = ["I"] * n
        s[index[v]] = "Z"
        errors.append(stim.PauliString("".join(s)))
    return errors

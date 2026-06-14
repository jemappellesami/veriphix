"""Graph-region builder for :class:`GaussianRegionNoiseModel`.

A *region* is a centre node together with all graph nodes within ``depth`` hops
of it (``depth=0`` → the centre alone, ``depth=1`` → centre + neighbours,
``depth=2`` → + neighbours-of-neighbours, …).  Each node in the region gets a
flip probability that decays as a Gaussian in the *graph hop distance* from the
centre::

    prob(node) = peak * exp(-hops**2 / (2 * sigma**2))

The model knows nothing about this — these helpers turn region specifications
into the flat ``node -> probability`` mapping the model consumes.  Multiple
regions are combined by summing their contributions per node and clipping to
``[0, 1]`` (sum-then-clip), so overlapping noisy regions reinforce each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class RegionSpec:
    """A Gaussian noisy region centred on a graph node.

    Parameters
    ----------
    center : int
        Centre node of the region.
    depth : int
        Maximum graph hop distance included in the region.
    sigma : float
        Standard deviation of the Gaussian, in hops.
    peak : float
        Flip probability at the centre (``hops = 0``).
    """

    center: int
    depth: int = 1
    sigma: float = 1.0
    peak: float = 0.5


def gaussian_region(graph: nx.Graph, spec: RegionSpec) -> dict[int, float]:
    """Return ``node -> probability`` for a single region.

    Nodes beyond ``spec.depth`` hops from the centre are omitted (contribute 0).
    """
    if spec.center not in graph:
        raise ValueError(f"center node {spec.center} not in graph")
    hops = nx.single_source_shortest_path_length(graph, spec.center, cutoff=spec.depth)
    two_sigma_sq = 2.0 * spec.sigma * spec.sigma
    return {
        node: spec.peak * math.exp(-(d * d) / two_sigma_sq)
        for node, d in hops.items()
    }


def build_node_probs(graph: nx.Graph, specs: Iterable[RegionSpec]) -> dict[int, float]:
    """Combine several regions into one ``node -> probability`` mapping.

    Contributions from overlapping regions are summed, then clipped to ``1.0``.
    """
    combined: dict[int, float] = {}
    for spec in specs:
        for node, prob in gaussian_region(graph, spec).items():
            combined[node] = combined.get(node, 0.0) + prob
    return {node: min(1.0, prob) for node, prob in combined.items()}

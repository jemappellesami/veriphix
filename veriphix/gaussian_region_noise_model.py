"""Stochastic per-node flip noise model.

This model knows nothing about geometry: it is given a flat mapping
``node -> probability`` and, independently for every test round and every
listed node, draws a Bernoulli trial.  On success it applies a full dephasing
flip (``DephasingNoise(prob=1)``) to that node just before its measurement.

The geometry — turning a *region* of the graph and a probability distribution
into the ``node -> probability`` mapping — is deliberately left to the caller
(see ``applications/noise_learning/regions.py``).  This keeps the noise model
dumb and reusable: the same mapping can describe a single noisy qubit, a qubit
and its neighbours, or several overlapping noisy regions.

Contrast with :class:`veriphix.malicious_noise_model.MaliciousNoiseModel`,
which freezes a single global on/off flag per round.  Here every node is drawn
independently and the draw happens live in :meth:`command`, so the noise is
genuinely stochastic and uncontrolled from round to round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.command import CommandKind
from graphix.noise_models.noise_model import ApplyNoise, NoiseModel
from graphix.rng import ensure_rng

# override introduced in Python 3.12
from typing_extensions import override

from veriphix.malicious_noise_model import DephasingNoise

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from graphix.command import BaseM
    from graphix.measurements import Outcome
    from graphix.noise_models.noise_model import CommandOrNoise
    from numpy.random import Generator


class GaussianRegionNoiseModel(NoiseModel):
    """Per-node stochastic dephasing flip noise.

    For each measured node listed in ``node_probs``, an independent Bernoulli
    trial is drawn at measurement time; on success a full dephasing flip is
    applied to that node immediately before its measurement.

    Parameters
    ----------
    node_probs : Mapping[int, float]
        Mapping ``node -> flip probability``.  Nodes absent from the mapping
        are never flipped.  Probabilities must lie in ``[0, 1]``.
    rng : Generator | None
        Random generator used for the Bernoulli draws.
    """

    def __init__(self, node_probs: Mapping[int, float], rng: Generator | None = None) -> None:
        for node, prob in node_probs.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"probability for node {node} out of range [0, 1]: {prob}")
        self.node_probs: dict[int, float] = {int(node): float(prob) for node, prob in node_probs.items()}
        self.rng = ensure_rng(rng)

    @override
    def input_nodes(
        self, nodes: Iterable[int], rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return []

    @override
    def command(
        self, cmd: CommandOrNoise, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to the command ``cmd``.

        Draws a fresh Bernoulli trial for the measured node (so the flip is
        independent every round).  The dephasing noise must *precede* the
        measurement — applying it afterwards has no effect on the classical
        outcome.
        """
        if cmd.kind == CommandKind.M:
            prob = self.node_probs.get(int(cmd.node))
            if prob is not None and self.rng.uniform() < prob:
                return [ApplyNoise(DephasingNoise(prob=1), [int(cmd.node)]), cmd]
        return [cmd]

    @override
    def confuse_result(
        self, cmd: BaseM, result: Outcome, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> Outcome:
        """Return the measurement result unchanged."""
        return result

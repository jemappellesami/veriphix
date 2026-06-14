"""Uncorrelated depolarising noise model.

Applies an **independent single-qubit** depolarising channel to *each* endpoint of
every ``CZ`` (``E``) command, rather than a single two-qubit channel on the edge.
This "uncorrelated" choice is what makes each graph edge carry the single-qubit
depolarising eigenvalue ``1 - 4 p / 3`` that ACES estimates; it differs from
graphix's built-in :class:`graphix.noise_models.depolarising.DepolarisingNoiseModel`,
which puts a :class:`TwoQubitDepolarisingNoise` on the edge instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.command import CommandKind
from graphix.noise_models.depolarising import DepolarisingNoise
from graphix.noise_models.noise_model import ApplyNoise, NoiseModel
from graphix.rng import ensure_rng

# override introduced in Python 3.12
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.command import BaseM
    from graphix.measurements import Outcome
    from graphix.noise_models.noise_model import CommandOrNoise
    from numpy.random import Generator


class UncorrelatedDepolarisingNoiseModel(NoiseModel):
    """Depolarising noise model with independent single-qubit channels.

    Parameters mirror graphix's :class:`DepolarisingNoiseModel`: each error
    probability gates a one-qubit depolarising channel attached to the relevant
    command. The entanglement error is applied to *both* nodes of a ``CZ`` edge
    as two independent single-qubit channels.
    """

    def __init__(
        self,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
        rng: Generator | None = None,
    ) -> None:
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        self.rng = ensure_rng(rng)

    @override
    def input_nodes(
        self, nodes: Iterable[int], rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return [ApplyNoise(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[node]) for node in nodes]

    @override
    def command(
        self, cmd: CommandOrNoise, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to the command ``cmd``."""
        if cmd.kind == CommandKind.N:
            return [cmd, ApplyNoise(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[cmd.node])]
        if cmd.kind == CommandKind.E:
            u, v = cmd.nodes
            return [
                cmd,
                ApplyNoise(noise=DepolarisingNoise(self.entanglement_error_prob), nodes=[u]),
                ApplyNoise(noise=DepolarisingNoise(self.entanglement_error_prob), nodes=[v]),
            ]
        if cmd.kind == CommandKind.M:
            # noise must precede the measurement to affect the classical outcome
            return [ApplyNoise(noise=DepolarisingNoise(self.measure_channel_prob), nodes=[cmd.node]), cmd]
        if cmd.kind == CommandKind.X:
            return [cmd, ApplyNoise(noise=DepolarisingNoise(self.x_error_prob), nodes=[cmd.node])]
        if cmd.kind == CommandKind.Z:
            return [cmd, ApplyNoise(noise=DepolarisingNoise(self.z_error_prob), nodes=[cmd.node])]
        # C / S / T / ApplyNoise commands are passed through unchanged.
        return [cmd]

    @override
    def confuse_result(
        self, cmd: BaseM, result: Outcome, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> Outcome:
        """Flip the measurement outcome with probability ``measure_error_prob``."""
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        return result

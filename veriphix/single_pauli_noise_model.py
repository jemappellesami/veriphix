"""Single-Pauli noise model.

A diagnostic model that attaches a deterministic single-qubit Pauli error (``X`` or
``Z``) of strength ``prob`` to qubit ``1`` right after the ``CZ`` on edge ``(0, 1)`` —
e.g. to probe error propagation on a 3-qubit line graph. Unlike a depolarising
channel, a pure ``Z`` error commutes with ``CZ``, which is exactly the case the
companion paper discusses as not being separable by gate reordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from graphix.channels import KrausChannel, KrausData
from graphix.command import CommandKind
from graphix.noise_models.noise_model import ApplyNoise, Noise, NoiseModel
from graphix.ops import Ops
from graphix.rng import ensure_rng

# override introduced in Python 3.12
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.command import BaseM
    from graphix.measurements import Outcome
    from graphix.noise_models.noise_model import CommandOrNoise
    from numpy.random import Generator


@dataclass
class SinglePauliNoise(Noise):
    """One-qubit single-Pauli (``X`` or ``Z``) error with probability ``prob``."""

    prob: float
    error_type: Literal["X", "Z"] = "X"

    @property
    @override
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 1

    @override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        operator = Ops.Z if self.error_type == "Z" else Ops.X
        return KrausChannel([KrausData(self.prob, operator)])


class SinglePauliNoiseModel(NoiseModel):
    """Deterministic single-Pauli error on qubit ``1`` of the ``(0, 1)`` ``CZ`` edge.

    Parameters
    ----------
    prob : float
        Strength of the single-Pauli error.
    error_type : {"X", "Z"}
        Which Pauli error to apply.
    rng : Generator | None
        Unused for the (deterministic) error itself; kept for interface parity.
    """

    def __init__(
        self,
        prob: float,
        error_type: Literal["X", "Z"] = "X",
        rng: Generator | None = None,
    ) -> None:
        self.rng = ensure_rng(rng)
        self.prob = prob
        self.error_type = error_type

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
        """Return the noise to apply to the command ``cmd``."""
        if cmd.kind == CommandKind.E and 0 in cmd.nodes and 1 in cmd.nodes:
            return [cmd, ApplyNoise(noise=SinglePauliNoise(self.prob, self.error_type), nodes=[1])]
        return [cmd]

    @override
    def confuse_result(
        self, cmd: BaseM, result: Outcome, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> Outcome:
        """Return the measurement result unchanged."""
        return result

"""Stim transpilation for Pauli MBQC patterns (vendored for the ACES experiment).

Ported from the legacy ``gospel.stim_pauli_preprocessing`` to the current graphix
API. Only the efficient sampling path used by the ACES experiment is kept:
``pattern_to_stim_circuit`` plus its Clifford-mapping helpers.

Differences from the original (see ``PORT_NOTES.md`` for the full table):

* ``graphix.states.BasicState`` (an enum with ``try_from_statevector``) was removed
  from graphix; a small local :class:`BasicState` is reconstructed here on top of the
  surviving ``graphix.states.BasicStates`` constants.
* The noise-apply command is now ``CommandKind.ApplyNoise`` (was ``CommandKind.A``),
  and ``DepolarisingNoise`` lives in ``graphix.noise_models.depolarising``.
* ``M`` commands no longer expose ``.plane``/``.angle``; the Pauli basis is read via
  ``cmd.measurement.try_to_pauli()``.
* The ``SinglePauliNoise`` match arm (gospel-only) was dropped, and the
  ``StimBackend`` / ``preprocess_pauli`` helpers (which relied on removed graphix
  internals and are unused by ACES) were not vendored.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, assert_never

import numpy as np
import stim
from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.fundamentals import Axis, Sign
from graphix.noise_models.depolarising import DepolarisingNoise, TwoQubitDepolarisingNoise
from graphix.sim.statevec import Statevec
from graphix.states import BasicStates

from veriphix.single_pauli_noise_model import SinglePauliNoise

if TYPE_CHECKING:
    import numpy.typing as npt
    from graphix import Pattern
    from graphix.measurements import PauliMeasurement
    from graphix.noise_models.noise_model import CommandOrNoise, NoiseModel


class BasicState(enum.Enum):
    """The six single-qubit stabiliser basis states.

    Reconstruction of the removed ``graphix.states.BasicState`` enum, backed by the
    surviving ``graphix.states.BasicStates`` planar-state constants.
    """

    ZERO = "ZERO"
    ONE = "ONE"
    PLUS = "PLUS"
    MINUS = "MINUS"
    PLUS_I = "PLUS_I"
    MINUS_I = "MINUS_I"

    @property
    def statevector(self) -> npt.NDArray[np.complex128]:
        """Return the reference state vector of this basic state."""
        return Statevec(getattr(BasicStates, self.value)).psi

    @staticmethod
    def try_from_statevector(psi: npt.NDArray[np.complex128], atol: float = 1e-6) -> BasicState | None:
        """Return the matching basic state (up to global phase), or ``None``."""
        vec = np.asarray(psi, dtype=np.complex128).ravel()
        for basic_state in BasicState:
            ref = basic_state.statevector
            # |<ref|psi>| == 1 iff equal up to a global phase.
            if abs(abs(np.vdot(ref, vec)) - 1.0) < atol:
                return basic_state
        return None


# Clifford gates that, applied to |0>, prepare each basic state.
BASIC_STATE_TO_CLIFFORD = {
    BasicState.ZERO: [Clifford.Z],
    BasicState.ONE: [Clifford.X],
    BasicState.PLUS: [Clifford.H],
    BasicState.MINUS: [Clifford.H, Clifford.Z],
    BasicState.PLUS_I: [Clifford.H, Clifford.S],
    BasicState.MINUS_I: [Clifford.H, Clifford.S, Clifford.Z],
}


def basic_state_to_clifford_gates(basic_state: BasicState) -> list[Clifford]:
    """Return the Clifford gates preparing ``basic_state`` from |0>."""
    return BASIC_STATE_TO_CLIFFORD[basic_state]


def pauli_measurement_to_clifford_gates(measurement: PauliMeasurement) -> list[Clifford]:
    """Return Cliffords rotating the given Pauli basis onto the computational Z basis."""
    match measurement.sign, measurement.axis:
        case Sign.PLUS, Axis.X:
            return [Clifford.H]
        case Sign.MINUS, Axis.X:
            return [Clifford.H, Clifford.Z]
        case Sign.PLUS, Axis.Y:
            return [Clifford.H, Clifford.S]
        case Sign.MINUS, Axis.Y:
            return [Clifford.H, Clifford.S, Clifford.Z]
        case Sign.PLUS, Axis.Z:
            return []
        case Sign.MINUS, Axis.Z:
            return [Clifford.X]
        case never:
            assert_never(never)


def pattern_to_stim_circuit(
    pattern: Pattern,
    noise_model: NoiseModel | None = None,
    input_state: dict[int, BasicState] | BasicState = BasicState.PLUS,
    fixed_states: dict[int, BasicState] | None = None,
) -> tuple[stim.Circuit, dict[int, int]]:
    """Transpile a Pauli MBQC pattern into a stim circuit and a node -> column map."""
    circuit = stim.Circuit()
    if isinstance(input_state, BasicState):
        for clifford in basic_state_to_clifford_gates(input_state):
            circuit.append(str(clifford), targets=pattern.input_nodes)  # type: ignore[call-overload]
    else:
        other_nodes = set(input_state.keys()) - set(pattern.input_nodes)
        if other_nodes:
            raise ValueError(f"Not input states: {other_nodes}")
        for node in pattern.input_nodes:
            basic_state = input_state[node]
            for clifford in basic_state_to_clifford_gates(basic_state):
                circuit.append(str(clifford), targets=[node])  # type: ignore[call-overload]
    if noise_model is None:
        actual_pattern: list[CommandOrNoise] = list(pattern)
    else:
        actual_pattern = noise_model.input_nodes(pattern.input_nodes)
        actual_pattern.extend(noise_model.transpile(list(pattern)))
    measure_count = 0
    measure_indices: dict[int, int] = {}

    def get_target(node: int) -> stim.GateTarget:
        return stim.target_rec(measure_indices[node] - measure_count)

    for cmd in actual_pattern:
        if cmd.kind == CommandKind.N:
            basic_state_or_none = None if fixed_states is None else fixed_states.get(cmd.node)
            if basic_state_or_none is None:
                basic_state_or_none = BasicState.try_from_statevector(Statevec(cmd.state).psi)
                if basic_state_or_none is None:
                    raise ValueError(f"Non-Pauli preparation: {cmd}")
            for clifford in basic_state_to_clifford_gates(basic_state_or_none):
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.E:
            circuit.append("CZ", cmd.nodes)  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.M:
            for node in cmd.s_domain:
                circuit.append("CX", [get_target(node), cmd.node])  # type: ignore[call-overload]
            for node in cmd.t_domain:
                circuit.append("CZ", [get_target(node), cmd.node])  # type: ignore[call-overload]
            measurement = cmd.measurement.try_to_pauli()
            if measurement is None:
                raise ValueError(f"Non-Pauli measurement: {cmd}")
            cliffords = pauli_measurement_to_clifford_gates(measurement)
            for clifford in reversed(cliffords):
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
            circuit.append("M", [cmd.node])  # type: ignore[call-overload]
            for clifford in cliffords:
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
            measure_indices[cmd.node] = measure_count
            measure_count += 1
        elif cmd.kind == CommandKind.X:
            for node in cmd.domain:
                circuit.append("CX", [get_target(node), cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.Z:
            for node in cmd.domain:
                circuit.append("CZ", [get_target(node), cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.C:
            circuit.append(str(cmd.clifford), [cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.ApplyNoise:
            match cmd.noise:
                case DepolarisingNoise(prob=prob):
                    (q,) = cmd.nodes
                    circuit.append("DEPOLARIZE1", [q], prob)
                case TwoQubitDepolarisingNoise(prob=prob):
                    (q0, q1) = cmd.nodes
                    circuit.append("DEPOLARIZE2", [q0, q1], prob)
                case SinglePauliNoise(prob=prob, error_type=error_type):
                    (q,) = cmd.nodes
                    if error_type == "X":
                        circuit.append("X_ERROR", [q], prob)
                    elif error_type == "Z":
                        circuit.append("Z_ERROR", [q], prob)
                    else:
                        raise ValueError(f"Unsupported single-Pauli: {error_type} and {cmd.nodes}")
                case _:
                    raise ValueError(f"Unsupported noise: {cmd.noise} and {cmd.nodes}")

    return circuit, measure_indices

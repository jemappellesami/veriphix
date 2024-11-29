import graphix.command
import numpy as np
from graphix.random_objects import rand_circuit, Circuit
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.states import BasicStates

from veriphix.client import Client, Secrets
from noise_model import VBQCNoiseModel

class TestVBQC:
    def test_trap_delegated(self, fx_rng: np.random.Generator):
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()

        # don't forget to add in the output nodes that are not initially measured!
        for onode in pattern.output_nodes:
            pattern.add(graphix.command.M(node=onode))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_runs = client.create_test_runs()
        for run in test_runs:
            backend = StatevectorBackend()
            trap_outcomes = client.delegate_test_run(backend=backend, run=run)
            assert trap_outcomes == [0 for _ in run.traps_list]

    def test_noiseless(self):
        circuit = Circuit(3)
        circuit.h(0)
        circuit.h(2)
        # circuit.h(2)
        # circuit.h(1)
        circuit.h(1)
        circuit.rz(1, np.pi/4)
        # circuit.cnot(0,1)
        circuit.cnot(0,2)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        backend = DensityMatrixBackend()

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_runs = client.create_test_runs()
        noise_model = VBQCNoiseModel(
            measure_error_prob=0,
            entanglement_error_prob=0,
            x_error_prob=0,
            z_error_prob=0,
            measure_channel_prob=0)
        for run in test_runs:
            backend = DensityMatrixBackend()
            client.refresh_randomness()
            trap_outcomes = client.delegate_test_run(backend=backend, run=run, noise_model=noise_model)
            assert sum(trap_outcomes) == 0

    def test_noisy(self):
        circuit = Circuit(3)
        circuit.h(0)
        circuit.h(2)
        # circuit.h(2)
        # circuit.h(1)
        circuit.h(1)
        circuit.rz(1, np.pi/4)
        # circuit.cnot(0,1)
        circuit.cnot(0,2)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        backend = DensityMatrixBackend()

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_runs = client.create_test_runs()
        noise_model = VBQCNoiseModel(
            measure_error_prob=1,
            entanglement_error_prob=1,
            x_error_prob=1,
            z_error_prob=1,
            measure_channel_prob=1)
        total_trap_failures = 0
        for run in test_runs:
            backend = DensityMatrixBackend()
            client.refresh_randomness()
            trap_outcomes = client.delegate_test_run(backend=backend, run=run, noise_model=noise_model)
            total_trap_failures += sum(trap_outcomes)
        assert total_trap_failures > 0
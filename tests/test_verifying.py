import numpy as np
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from stim import PauliString

from veriphix.client import Client
from veriphix.verifying import TestRun, merge


class TestVerifying:
    def test_delegate_test(self, fx_rng: np.random.Generator) -> None:
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        client = Client(pattern=pattern, rng=fx_rng)

        for _ in range(10):
            # Test noiseless trap delegation
            trap_size = fx_rng.integers(len(client.nodes))
            random_nodes = [client.nodes[i] for i in fx_rng.choice(len(client.nodes), size=trap_size, replace=False)]

            random_multi_qubit_trap = frozenset(random_nodes)
            # Only one trap
            traps = frozenset({random_multi_qubit_trap})

            test_run = TestRun(client=client, traps=traps)
            backend = StatevectorBackend()
            outcomes = test_run.delegate(backend=backend, rng=fx_rng).trap_outcomes

            for trap in traps:
                assert outcomes[trap] == 0

    def test_merge_does_not_mutate_its_arguments(self) -> None:
        # `merge` used to accumulate into `strings[0]` and hand it back, so merging
        # rewrote one of the strings it was given. Callers that keep a reference to
        # their own strings -- to read each trap's sign, say -- would silently see
        # the merged value instead of their own.
        strings = [PauliString("+X_"), PauliString("-_Z")]
        originals = [str(string) for string in strings]

        merged = merge(strings)

        assert [str(string) for string in strings] == originals
        assert merged is not strings[0]
        assert str(merged) == "-XZ"

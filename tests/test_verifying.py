import networkx as nx
import numpy as np
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend

from veriphix.client import Client
from veriphix.verifying import TestRun


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

    def test_traps_keep_their_own_stabilizer_sign(self, fx_rng: np.random.Generator) -> None:
        """Each trap must be checked against its own stabilizer sign.

        `build_common_stabilizer` merges every trap's conjugated measurement string
        into a single common stabilizer, whose sign is the *product* of the individual
        signs and so says nothing about any one trap. Checking a trap against that
        product inverts the verdict of every trap whose own sign differs from it: an
        honest server gets rejected, and a server cheating on that trap gets accepted.

        The graph here is an ordinary transpiled random circuit, and it is connected.
        Trap {0, 1, 2} conjugates to a sign -1 string and trap {5} to a sign +1 one;
        they share node 4, where both carry Z, which is what lets them merge. Each
        verifies correctly on its own, so both together must as well.
        """
        circuit = rand_circuit(2, 1, fx_rng)
        pattern = circuit.transpile().pattern
        client = Client(pattern=pattern, rng=fx_rng)
        trap_a = frozenset({0, 1, 2})
        trap_b = frozenset({5})

        # Premises of the test: a connected graph, and two traps of opposite sign.
        # The pattern depends on the seed of `fx_rng`; if that changes, pick another pair.
        assert nx.is_connected(client.graph)
        assert TestRun(client=client, traps=frozenset({trap_a})).stabilizer.sign == -1
        assert TestRun(client=client, traps=frozenset({trap_b})).stabilizer.sign == 1

        # An honest server must be accepted whichever traps share a run.
        for traps in (frozenset({trap_a}), frozenset({trap_b}), frozenset({trap_a, trap_b})):
            test_run = TestRun(client=client, traps=traps)
            outcomes = test_run.delegate(backend=StatevectorBackend(), rng=fx_rng).trap_outcomes
            assert sum(outcomes.values()) == 0, f"honest server rejected for traps {sorted(sorted(t) for t in traps)}"

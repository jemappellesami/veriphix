import networkx as nx
import numpy as np
from graphix import Measurement, OpenGraph
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
        r"""Each trap must be checked against its own stabilizer sign.

        The resource graph has two disconnected components::

            2       3          4
             \     /           |
              \   /            |
                0              1

        Trap A = {0, 2, 3} conjugates to a sign -1 string, trap B = {1} to a sign
        +1 one, and each verifies correctly on its own. Placed in the same test
        run they are merged into a single stabilizer carrying the *product* of
        the signs, so applying that one sign to every trap inverts trap B and
        rejects an honest server.

        The two components are disconnected, so trap B's measured parity cannot
        depend on whether trap A is being tested alongside it: only the
        bookkeeping differs between the runs below.
        """
        og = OpenGraph(
            graph=nx.Graph([(0, 2), (0, 3), (1, 4)]),
            input_nodes=[0, 1],
            output_nodes=[2, 3, 4],
            measurements={0: Measurement.XY(0), 1: Measurement.XY(0)},
        )
        client = Client(pattern=og.to_pattern(), rng=fx_rng)
        trap_a = frozenset({0, 2, 3})
        trap_b = frozenset({1})

        # The premise of the test: the two traps really do carry opposite signs.
        assert TestRun(client=client, traps=frozenset({trap_a})).stabilizer.sign == -1
        assert TestRun(client=client, traps=frozenset({trap_b})).stabilizer.sign == 1

        # An honest server must be accepted whichever traps share a run.
        for traps in (frozenset({trap_a}), frozenset({trap_b}), frozenset({trap_a, trap_b})):
            test_run = TestRun(client=client, traps=traps)
            outcomes = test_run.delegate(backend=StatevectorBackend(), rng=fx_rng).trap_outcomes
            assert sum(outcomes.values()) == 0, f"honest server rejected for traps {sorted(sorted(t) for t in traps)}"

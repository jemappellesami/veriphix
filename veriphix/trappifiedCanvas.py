from __future__ import annotations

import random
from typing import TYPE_CHECKING

import graphix.command
import graphix.ops
import graphix.pattern
import graphix.pauli
import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.simulator
import stim

if TYPE_CHECKING:
    import networkx as nx
    from graphix.states import State

    Trap = set[int]


class TrappifiedCanvas:
    """
    Trappified canvas according to
    Kapourniotis, T., Kashefi, E., Leichtle, D., Music, L., & Ollivier, H. (2024). Unifying quantum verification and error-detection: theory and tools for optimisations. Quantum Science and Technology, 9(3), 035036.
    https://arxiv.org/abs/2206.00631

    A trap is a set of nodes (i_1, ..., i_t) that are used in a verification protocol, to verify that the sum of their outcomes is 0 modulo 2.

    A trappified canvas is specified by a set of traps and the description of a state on which an honest execution of the protocol should yield the expected outcomes for all traps.

    This class uses the stabilizer formalism, to generate from a list of traps, the appropriate input state.

    To one trappified canvas corresponds a list of traps, and a resulting stabilizer expressed in a Pauli string. The input that satisfies all traps is a +1 eigenstate of that operator.
    """

    def __init__(self, graph: nx.Graph, traps_list: set[Trap]) -> None:
        self.graph = graph
        self.traps_list = traps_list

        self.trap_stabilizers = [self.compute_trap_stabilizer(trap) for trap in self.traps_list]
        self.stabilizer = self.merge_pauli_list(self.trap_stabilizers)
        dummies_coins = self.generate_coins_dummies()
        self.coins = self.generate_coins_trap_qubits(coins=dummies_coins)
        self.states = self.generate_eigenstate()

    def common_eigenstate(self, stabilizer_1: stim.PauliString, stabilizer_2: stim.PauliString) -> bool:
        """
        Returns `True` if the two stabilizers have a common eigenstate
        """
        match_X = set(stabilizer_1.pauli_indices("X")).issubset(
            set(stabilizer_2.pauli_indices("X")).union(set(stabilizer_2.pauli_indices("I")))
        )
        match_Y = set(stabilizer_1.pauli_indices("Y")).issubset(
            set(stabilizer_2.pauli_indices("Y")).union(set(stabilizer_2.pauli_indices("I")))
        )
        match_Z = set(stabilizer_1.pauli_indices("Z")).issubset(
            set(stabilizer_2.pauli_indices("Z")).union(set(stabilizer_2.pauli_indices("I")))
        )
        return match_X and match_Y and match_Z

    def merge(self, stabilizer_1: stim.PauliString, stabilizer_2: stim.PauliString) -> stim.PauliString:
        result = stabilizer_2.sign * stabilizer_1
        # We can iterate through the support of stab2 as they are supposed to have equal support
        for node in stabilizer_2.pauli_indices("XYZ"):
            result[node] = stabilizer_2[node]
        return result

    def fill_canonical_stabilizer(self, chain: stim.PauliString, node: int) -> None:
        chain[node] = "X"
        for n in self.graph.neighbors(node):
            chain[n] = "Z"

    # become unused?
    def get_canonical_stabilizer(self, node: int) -> stim.PauliString:
        chain = stim.PauliString(len(self.graph.nodes))
        self.fill_canonical_stabilizer(chain, node)
        return chain

    def compute_trap_stabilizer(self, trap: set[int]) -> stim.PauliString:
        trap_stabilizer = stim.PauliString(len(self.graph.nodes))
        for node in trap:
            self.fill_canonical_stabilizer(trap_stabilizer, node)
        return trap_stabilizer

    def merge_pauli_list(self, pauli_list) -> stim.PauliString:
        """
        If the protocol is coherent, all the stabilizers should be able to merge in one Pauli string.
        """
        # Initial flag to track if any merging happens in the iteration
        merged = True

        while merged:
            merged = False
            i = 0

            while i < len(pauli_list) - 1:
                j = i + 1

                while j < len(pauli_list):
                    if self.common_eigenstate(pauli_list[i], pauli_list[j]):
                        # Merge the two PauliStrings
                        merged_pauli = self.merge(pauli_list[i], pauli_list[j])

                        # Replace the first element with the merged result
                        pauli_list[i] = merged_pauli

                        # Remove the second element
                        del pauli_list[j]

                        # Set merged flag to True to indicate a merge happened
                        merged = True

                        # Break the inner loop since we need to recheck the new list
                        break
                    else:
                        j += 1

                if not merged:
                    i += 1

        return pauli_list[0]

    @property
    def trap_qubits(self):
        return [node for trap in self.traps_list for node in trap]

    @property
    def dummy_qubits(self):
        return [neighbor for trap in self.traps_list for node in trap for neighbor in list(self.graph.neighbors(node))]

    def generate_eigenstate(self) -> list[State]:
        states = []
        for node in sorted(self.graph.nodes):
            operator = graphix.pauli.TABLE[self.stabilizer[node]][0][0]
            states.append(operator.get_eigenstate(eigenvalue=self.coins[node]))
        return states

    def generate_coins_dummies(self):
        coins = dict()
        for node in self.graph.nodes:
            if node not in self.trap_qubits:
                coins[node] = random.randint(0, 1)
            else:
                coins[node] = 0
        return coins

    def generate_coins_trap_qubits(self, coins):
        for node in self.trap_qubits:
            coins[node] = sum(coins[n] for n in self.graph.neighbors(node)) % 2
        return coins

    def __str__(self) -> str:
        return f"List of traps: {self.traps_list}\nStabilizer: {self.stabilizer}"

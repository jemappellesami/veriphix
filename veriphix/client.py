from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import graphix.command
import graphix.ops
import graphix.pattern
import graphix.pauli
import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.simulator
import networkx as nx
import numpy as np
from graphix.clifford import Clifford
from graphix.command import BaseM, CommandKind, MeasureUpdate
from graphix.measurements import Measurement
from graphix.ops import Ops
from graphix.pattern import Pattern
from graphix.pauli import Pauli
from graphix.sim.statevec import StatevectorBackend
from graphix.simulator import MeasureMethod, PatternSimulator
from graphix.states import BasicStates

from veriphix.trappifiedCanvas import TrappifiedCanvas

if TYPE_CHECKING:
    from graphix.sim.base_backend import Backend

    from veriphix.trappifiedCanvas import Trap


# TODO update docstring
"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""


@dataclass
class TrappifiedRun:
    input_state: list
    tested_qubits: list[int]
    stabilizer: Pauli


@dataclass
class Secret_a:
    a: dict[int, int]
    a_N: dict[int, int]


@dataclass
class Secrets:
    r: bool = False
    a: bool = False
    theta: bool = False


@dataclass
class SecretDatas:
    r: dict[int, int]
    a: Secret_a
    theta: dict[int, int]

    # NOTE: not a class method?
    @staticmethod
    def from_secrets(secrets: Secrets, graph, input_nodes, output_nodes):
        node_list, edge_list = graph
        r = {}
        if secrets.r:
            # Need to generate the random bit for each measured qubit, 0 for the rest (output qubits)
            for node in node_list:
                r[node] = np.random.randint(0, 2) if node not in output_nodes else 0

        theta = {}
        if secrets.theta:
            # Create theta secret for all non-output nodes (measured qubits)
            for node in node_list:
                theta[node] = np.random.randint(0, 8) if node not in output_nodes else 0  # Expressed in pi/4 units
                ## TODO:
        a = {}
        a_N = {}
        if secrets.a:
            # Create `a` secret for all
            # order is Z(theta) X |+>
            for node in node_list:
                a[node] = np.random.randint(0, 2) if node not in input_nodes else 0

            # After all the `a` secrets have been generated, the `a_N` value can be
            # computed from the graph topology
            for i in node_list:
                a_N_value = 0
                for j in node_list:
                    if (i, j) in edge_list or (j, i) in edge_list:
                        a_N_value ^= a[j]
                a_N[i] = a_N_value

        return SecretDatas(r, Secret_a(a, a_N), theta)


@dataclass
class ByProduct:
    z_domain: list[int]
    x_domain: list[int]


def get_byproduct_db(pattern: Pattern) -> dict[int, ByProduct]:
    byproduct_db = dict()
    for node in pattern.output_nodes:
        byproduct_db[node] = ByProduct(z_domain=[], x_domain=[])

    for cmd in pattern.correction_commands():
        if cmd.node in pattern.output_nodes:
            if cmd.kind == CommandKind.Z:
                byproduct_db[cmd.node].z_domain = cmd.domain
            if cmd.kind == CommandKind.X:
                byproduct_db[cmd.node].x_domain = cmd.domain
    return byproduct_db


def remove_flow(pattern):
    clean_pattern = Pattern(pattern.input_nodes)
    for cmd in pattern:
        if cmd.kind in (CommandKind.X, CommandKind.Z):
            # If byproduct, remove it so it's not done by the server
            continue
        if cmd.kind == CommandKind.M:
            # If measure, remove measure parameters
            new_cmd = graphix.command.BaseM(node=cmd.node)
        else:
            new_cmd = cmd
        clean_pattern.add(new_cmd)
    return clean_pattern


def prepared_nodes_as_input_nodes(pattern: Pattern) -> Pattern:
    input_nodes = pattern.input_nodes
    seq = []
    for cmd in pattern:
        if cmd.kind == CommandKind.N:
            input_nodes.append(cmd.node)
        else:
            seq.append(cmd)
    result = Pattern(input_nodes=input_nodes)
    result.extend(seq)
    return result


class Client:
    def __init__(self, pattern, input_state=None, measure_method_cls=None, secrets: None | Secrets = None) -> None:
        self.initial_pattern = pattern

        self.input_nodes = self.initial_pattern.input_nodes.copy()
        self.output_nodes = self.initial_pattern.output_nodes.copy()
        self.graph = self.initial_pattern.get_graph()
        self.nodes_list = self.graph[0]

        # Copy the pauli-preprocessed nodes' measurement outcomes
        self.results = pattern.results.copy()
        if measure_method_cls is None:
            measure_method_cls = ClientMeasureMethod
        self.measure_method = measure_method_cls(self)

        self.measurement_db = {measure.node: measure for measure in pattern.get_measurement_commands()}
        self.byproduct_db = get_byproduct_db(pattern)
        # print("byprod_db", self.byproduct_db)

        # self.secrets_bool : bool -> self.secrets is not None
        # self.secrets_type : Secrets -> self.secrets
        # self.secrets : SecretDatas-> self.secret_datas
        self.secrets = secrets
        if secrets is None:
            self.secrets_bool = False
            secrets = Secrets()

        self.secret_datas = SecretDatas.from_secrets(secrets, self.graph, self.input_nodes, self.output_nodes)

        pattern_without_flow = remove_flow(pattern)
        self.clean_pattern = prepared_nodes_as_input_nodes(pattern_without_flow)

        self.input_state = input_state if input_state is not None else [BasicStates.PLUS for _ in self.input_nodes]

    def refresh_randomness(self) -> None:
        "method to refresh random randomness using parameters from Clinent instatiation."

        # refresh only if secrets bool is True; False is no randomness at all.
        if self.secrets is not None:
            self.secret_datas = SecretDatas.from_secrets(self.secrets, self.graph, self.input_nodes, self.output_nodes)

    def blind_qubits(self, backend: Backend) -> None:
        def z_rotation(theta) -> np.array:
            return np.array([[1, 0], [0, np.exp(1j * theta * np.pi / 4)]])

        def x_blind(a) -> Pauli:
            return Pauli.X if (a == 1) else Pauli.I

        for node in self.nodes_list:
            theta = self.secret_datas.theta.get(node, 0)
            a = self.secret_datas.a.a.get(node, 0)
            if a:
                backend.apply_single(node=node, op=x_blind(a).matrix)
            if theta:
                backend.apply_single(node=node, op=z_rotation(theta))

    def prepare_states(self, backend: Backend) -> None:
        # First prepare inputs
        backend.add_nodes(nodes=self.input_nodes, data=self.input_state)

        # Then iterate over auxiliaries required to blind
        outer_nodes = set(self.input_nodes + self.output_nodes)
        aux_nodes = [node for node in self.nodes_list if node not in outer_nodes]
        aux_data = [BasicStates.PLUS for _ in aux_nodes]
        backend.add_nodes(nodes=aux_nodes, data=aux_data)


        ## TODO: revoir ça
        # Prepare outputs
        output_data = []
        for node in self.output_nodes:
            r_value = self.secret_datas.r.get(node, 0)
            a_N_value = self.secret_datas.a.a_N.get(node, 0)
            output_data.append(BasicStates.PLUS if r_value ^ a_N_value == 0 else BasicStates.MINUS)
        backend.add_nodes(nodes=self.output_nodes, data=output_data)

    def create_test_runs(self) -> list[TrappifiedCanvas]:
        """
        Creates test runs according to FK12 protocol of
        Fitzsimons, J. F., & Kashefi, E. (2017). Unconditionally verifiable blind quantum computation. Physical Review A, 96(1), 012303.
        https://arxiv.org/abs/1203.5217

        The graph is partitioned in `k` colors.
        Each color is associated to a test run, or a Trappified Canvas.

        For a color, single-qubit traps are created for each node of that color.
        """

        # TODO: just pattern.get_graph()
        graph = nx.Graph()
        nodes, edges = self.graph
        graph.add_edges_from(edges)
        graph.add_nodes_from(nodes)

        # Create the graph coloring
        coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
        colors = set(coloring.values())
        nodes_by_color = {c: [] for c in colors}
        for node in sorted(graph.nodes):
            color = coloring[node]
            nodes_by_color[color].append(node)

        # Create the test runs : one per color
        runs: list[TrappifiedCanvas] = []
        for color in colors:
            # 1 color = 1 test run = 1 set of traps
            traps_list = []
            for colored_node in nodes_by_color[color]:
                trap_qubits = [colored_node]  # single-qubit trap
                trap: Trap = set(trap_qubits)
                traps_list.append(trap)

            ## TODO
            # In here, traps_list is a list of traps that are compatible already, because they are of the same color
            # So we can just merge them all already
            # And assume that TrappifiedCanvas needs to be instanciated with a traps_list that corresponds to a coloring.
            # and TC just merges them assuming they are all mergeable.
            trappified_canvas = TrappifiedCanvas(graph, traps_list=traps_list)

            runs.append(trappified_canvas)
        return runs

    def delegate_test_run(self, backend: Backend, run: TrappifiedCanvas, **kwargs) -> list[int]:
        # The state is entirely prepared and blinded by the client before being sent to the server
        # StateVectorBackend because noiseless preparation
        preparation_backend = StatevectorBackend()
        preparation_backend.add_nodes(nodes=sorted(self.graph[0]), data=run.states)
        self.blind_qubits(preparation_backend)

        backend.add_nodes(nodes=sorted(self.graph[0]), data=preparation_backend.state)

        tmp_measurement_db = self.measurement_db.copy()
        # Modify the pattern to be all X-basis measurements, no shifts/signalling updates
        # Warning should only work for BQP ie classical output
        for node in self.measurement_db:
            self.measurement_db[node] = graphix.command.M(node=node)

        # TODO add measurements on output nodes?

        sim = PatternSimulator(
            backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method, **kwargs
        )
        sim.run(input_state=None)

        trap_outcomes = []
        for trap in run.traps_list:
            outcomes = [self.results[component] for component in trap]  # here
            trap_outcome = sum(outcomes) % 2
            trap_outcomes.append(trap_outcome)

        self.measurement_db = tmp_measurement_db

        return trap_outcomes

    def delegate_pattern(self, backend: Backend, **kwargs) -> None:
        self.prepare_states(backend)
        self.blind_qubits(backend)
        sim = PatternSimulator(
            backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method, **kwargs
        )
        sim.run(input_state=None)
        self.decode_output_state(backend)

    def decode_output_state(self, backend: Backend):
        for node in self.output_nodes:
            z_decoding, x_decoding = self.decode_output(node)
            # print(f"decoding bits {z_decoding} and {x_decoding}")
            if z_decoding:
                backend.apply_single(node=node, op=Ops.Z)
            if x_decoding:
                backend.apply_single(node=node, op=Ops.X)

    def get_secrets_size(self):
        secrets_size = {}
        for secret in self.secret_datas:
            secrets_size[secret] = len(self.secret_datas[secret])
        return secrets_size

    def decode_output(self, node):
        z_decoding = sum(self.results[z_dep] for z_dep in self.byproduct_db[node].z_domain) % 2
        z_decoding ^= self.secret_datas.r.get(node, 0)
        x_decoding = sum(self.results[x_dep] for x_dep in self.byproduct_db[node].x_domain) % 2
        x_decoding ^= self.secret_datas.a.a.get(node, 0)
        return z_decoding, x_decoding


class ClientMeasureMethod(MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client

    def get_measurement_description(self, cmd: BaseM) -> Measurement:
        parameters = self.__client.measurement_db[cmd.node]

        # print("Client measurement db ", self.__client.measurement_db)
        r_value = self.__client.secret_datas.r.get(cmd.node, 0)
        theta_value = self.__client.secret_datas.theta.get(cmd.node, 0)
        a_value = self.__client.secret_datas.a.a.get(cmd.node, 0)
        a_N_value = self.__client.secret_datas.a.a_N.get(cmd.node, 0)
        # print("parameters.", parameters)
        # print("secrets", self.__client.secrets)
        # extract signals for adaptive angle
        s_signal = sum(self.__client.results[j] for j in parameters.s_domain)
        t_signal = sum(self.__client.results[j] for j in parameters.t_domain)
        measure_update = MeasureUpdate.compute(parameters.plane, s_signal % 2 == 1, t_signal % 2 == 1, Clifford.I)
        # print("meas update", measure_update)
        angle = parameters.angle * np.pi
        angle = angle * measure_update.coeff + measure_update.add_term
        angle = (-1) ** a_value * angle + theta_value * np.pi / 4 + np.pi * (r_value + a_N_value)
        # angle = angle * measure_update.coeff + measure_update.add_term
        return Measurement(plane=measure_update.new_plane, angle=angle)

    def get_measure_result(self, node: int) -> bool:
        raise ValueError("Server cannot have access to measurement results")

    def set_measure_result(self, node: int, result: bool) -> None:
        if self.__client.secret_datas.r:
            result ^= self.__client.secret_datas.r[node]
        self.__client.results[node] = result

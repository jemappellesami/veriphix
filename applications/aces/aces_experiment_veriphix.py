"""ACES (Averaged Circuit Eigenvalue Sampling) experiment, ported to current veriphix.

Reproduces the legacy gospel-based ACES experiment against this repo's veriphix +
graphix. See ``PORT_NOTES.md`` for the old -> new mapping table and the behavioural
differences forced by the new APIs.

Run (Stim backend, defaults)::

    python aces_experiment_veriphix.py

Fast smoke check (seconds)::

    python aces_experiment_veriphix.py --smoke
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing import freeze_support
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import typer
from graphix import Pattern, command
from graphix.command import CommandKind
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod, PrepareMethod
from numpy.random import PCG64, Generator
from stim_pauli_preprocessing import BasicState, pattern_to_stim_circuit

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12
from veriphix.sampling_circuits.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from veriphix.uncorrelated_depolarising_noise_model import UncorrelatedDepolarisingNoiseModel

if TYPE_CHECKING:
    from graphix.command import BaseN
    from graphix.noise_models.noise_model import NoiseModel
    from graphix.sim.base_backend import Backend
    from graphix.states import State

logger = logging.getLogger(__name__)

# Result-table entry shape: (samples, measure_indices, traps_list), where
# samples has shape (nshots, n_measured), measure_indices maps node -> column,
# and traps_list is a list of (trap,) tuples.
ResultTable = list[tuple[object, object, list[tuple[int, ...]]]]


class Method(Enum):
    Stim = "stim"
    Graphix = "graphix"
    Veriphix = "veriphix"


def state_to_basic_state(state: State) -> BasicState:
    bs = BasicState.try_from_statevector(Statevec(state).psi)
    if bs is None:
        raise ValueError(f"Not a basic state: {state}")
    return bs


def x_basis_measurement_pattern(clean_pattern: Pattern) -> Pattern:
    """Rewrite every (bare) measurement of a flow-removed pattern as an X-basis ``M``.

    The current ``veriphix.client.remove_flow`` strips measurements down to bare
    ``BaseM(node)`` (no basis), which the stim transpiler cannot read. The trap test
    measures every node in the X basis (θ-blinding off), so we re-attach a full
    ``command.M(node)`` (default ``Measurement.X``, no feed-forward).
    """
    pattern = Pattern(input_nodes=clean_pattern.input_nodes)
    for cmd in clean_pattern:
        if cmd.kind == CommandKind.M:
            pattern.add(command.M(node=cmd.node))
        else:
            pattern.add(cmd)
    return pattern


@dataclass
class SingleSimulation:
    order: ConstructionOrder
    nqubits: int
    nlayers: int
    noise_model: NoiseModel
    nshots: int
    jumps: int
    method: Method


@dataclass
class FixedPrepareMethod(PrepareMethod):
    states: dict[int, State]

    def prepare(self, backend: Backend, cmd: BaseN, rng: Generator | None = None) -> None:
        backend.add_nodes(nodes=[cmd.node], data=self.states[cmd.node])


def perform_single_simulation(
    params: SingleSimulation,
) -> list[tuple[ConstructionOrder, bool, ResultTable]]:
    fx_bg = PCG64(42)

    rng = Generator(fx_bg.jumped(params.jumps))  # Use the jumped rng

    pattern = generate_random_pauli_pattern(
        nqubits=params.nqubits, nlayers=params.nlayers, order=params.order, rng=rng
    )

    noise_model = params.noise_model

    # Original flow-carrying pattern -> Client (with blinding fully off); the client
    # adds measurements on the output nodes and exposes the flow-removed clean pattern.
    secrets = Secrets(r=False, a=False, theta=False)
    colours = get_bipartite_coloring(pattern)
    client = Client(pattern=pattern, secrets=secrets, protocol=FK12(manual_colouring=list(colours)))

    # Flow-removed pattern measured in X on every node, used for the efficient backends.
    client_pattern = x_basis_measurement_pattern(client.clean_pattern)

    test_runs = client.test_runs

    outcomes = []

    for i, run in enumerate(test_runs):
        traps_list = [tuple(trap) for trap in run.traps]
        if params.method == Method.Veriphix:
            raise NotImplementedError(
                "The Veriphix method requires a StimBackend port to the current graphix "
                "Backend ABC, which is out of scope for this port (see PORT_NOTES.md). "
                "Use --method stim (the efficient default)."
            )
        if params.method == Method.Graphix:
            assert params.nshots == 1
            measure_method = DefaultMeasureMethod()
            prepare_method = FixedPrepareMethod(dict(run.input_state))
            input_state = [run.input_state[node] for node in client_pattern.input_nodes]
            client_pattern.simulate_pattern(
                backend="densitymatrix",
                input_state=input_state,
                prepare_method=prepare_method,
                measure_method=measure_method,
                noise_model=noise_model,
            )
            results: ResultTable = [
                (
                    [measure_method.results],
                    list(range(len(measure_method.results))),
                    traps_list,
                )
            ]
        else:
            input_state_dict: dict[int, BasicState] = {}
            fixed_states: dict[int, BasicState] = {}
            for node, state in run.input_state.items():
                basic_state = state_to_basic_state(state)
                if node in client_pattern.input_nodes:
                    input_state_dict[node] = basic_state
                else:
                    fixed_states[node] = basic_state
            circuit, measure_indices = pattern_to_stim_circuit(
                client_pattern,
                input_state=input_state_dict,
                noise_model=noise_model,
                fixed_states=fixed_states,
            )
            sample = circuit.compile_sampler().sample(shots=params.nshots)
            results = [(sample, measure_indices, traps_list)]

        outcomes.append((params.order, bool(i), results))

    return outcomes


@dataclass
class SimulationResult:
    canonical: ResultTable
    deviant: ResultTable


def perform_simulation(
    nqubits: int,
    nlayers: int,
    noise_model: NoiseModel,
    nshots: int,
    ncircuits: int,
    method: Method,
) -> SimulationResult:
    jobs = [
        SingleSimulation(
            order=order,
            nqubits=nqubits,
            nlayers=nlayers,
            noise_model=noise_model,
            nshots=nshots,
            method=method,
            jumps=circuit * 2 + int(order == ConstructionOrder.Deviant),
        )
        for circuit in range(ncircuits)
        for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant)
    ]

    logger.debug(f"nb jobs to run: {len(jobs)}")
    outcomes = list(map(perform_single_simulation, jobs))

    test_outcome_table_canonical: ResultTable = []
    test_outcome_table_deviant: ResultTable = []

    for outcome in outcomes:
        for order, _col, results in outcome:
            if order == ConstructionOrder.Canonical:
                test_outcome_table_canonical.extend(results)
            elif order == ConstructionOrder.Deviant:
                test_outcome_table_deviant.extend(results)

    return SimulationResult(test_outcome_table_canonical, test_outcome_table_deviant)


def compute_failure_probabilities(
    nnodes: int,
    results_table: ResultTable,
) -> npt.NDArray[np.float64]:
    occurrences = np.zeros(nnodes, dtype=np.int64)
    occurrences_one = np.zeros(nnodes, dtype=np.int64)

    for samples, measure_indices, traps_list in results_table:
        nsamples = len(samples)
        ones = np.array(samples).sum(axis=0)
        for (trap,) in traps_list:
            occurrences[trap] += nsamples
            occurrences_one[trap] += ones[measure_indices[trap]]

    return occurrences_one / occurrences


def generate_equations(pattern: Pattern) -> dict[int, set[frozenset[int]]]:
    nodes = list(pattern.extract_graph().nodes)
    result: dict[int, set[frozenset[int]]] = {node: set() for node in nodes}
    active_nodes = {node: {node} for node in nodes}
    for cmd in reversed(list(pattern)):
        if cmd.kind == CommandKind.E:
            u, v = cmd.nodes
            edge = frozenset({u, v})
            for target in active_nodes[u] | active_nodes[v]:
                result[target].add(edge)
            active_nodes[u].add(v)
            active_nodes[v].add(u)
    return result


@dataclass
class EdgeDependency:
    edge: frozenset[int]
    order: ConstructionOrder
    measure_index: int
    previous_edges: frozenset[int]


def generate_edge_dependencies(nqubits: int, nlayers: int) -> list[EdgeDependency]:
    equations = {}
    for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant):
        pattern = generate_random_pauli_pattern(nqubits, nlayers, order=order)
        equations[order] = generate_equations(pattern)
    result: list[EdgeDependency] = []
    known: set[frozenset[int]] = set()
    known_indices: dict[frozenset[int], int] = {}
    while True:
        new_element = False
        for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant):
            for measure_index, lambdas in equations[order].items():
                left = lambdas - known
                try:
                    (edge,) = left
                except ValueError:
                    pass
                else:
                    previous_edges = frozenset(
                        {known_indices[lam] for lam in lambdas if lam != edge}
                    )
                    dependency = EdgeDependency(
                        edge, order, measure_index, previous_edges
                    )
                    known.add(edge)
                    known_indices[edge] = len(result)
                    result.append(dependency)
                    new_element = True
        if not new_element:
            break
    return result


def compute_aces_postprocessing_iteratively(
    nnodes: int, dependencies: list[EdgeDependency], results: SimulationResult
) -> dict[frozenset[int], float]:
    start = time.time()
    pi = {
        ConstructionOrder.Canonical: compute_failure_probabilities(
            nnodes, results.canonical
        ),
        ConstructionOrder.Deviant: compute_failure_probabilities(
            nnodes, results.deviant
        ),
    }
    logger.info(f"Failure probabilities in {time.time() - start:.4f} seconds.")
    result_log: list[float] = []
    for dependency in dependencies:
        pi_value = math.log(1 - 2 * pi[dependency.order][dependency.measure_index])
        for edge in dependency.previous_edges:
            pi_value -= result_log[edge]
        result_log.append(pi_value)
    return {
        dependency.edge: math.exp(v) for dependency, v in zip(dependencies, result_log, strict=True)
    }


def compute_probabilities_difference_can(
    failure_proba_can_result: dict[int, float],
    n_nodes: int,
) -> list[float]:
    return [1 - 2 * failure_proba_can_result[k] for k in range(n_nodes)]


def compute_probabilities_difference_dev(
    failure_proba_dev_result: dict[int, float],
    n_nodes: int,
    n_qubits: int,
) -> list[float]:
    return [
        1 - 2 * failure_proba_dev_result[k]
        for k in range(n_nodes)
        if (k % n_qubits) % 2 == 0 and (k // n_qubits) % 2 == 1
    ]


def generate_qubit_edge_matrix_from_pattern(
    pattern: Pattern, nodes: list[int], edges: list[frozenset[int]]
) -> npt.NDArray[np.int64]:
    equations = generate_equations(pattern)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    matrix = np.zeros((len(nodes), len(edges)), dtype=np.int64)
    for row, node in enumerate(nodes):
        lambdas = equations[node]
        for edge in lambdas:
            matrix[row, edge_index[edge]] = 1
    return matrix


def generate_qubit_edge_matrix(
    nqubits: int, nlayers: int
) -> tuple[list[frozenset[int]], npt.NDArray[np.int64]]:
    pattern_can = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=ConstructionOrder.Canonical
    )
    pattern_dev = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=ConstructionOrder.Deviant
    )

    edges_set = {frozenset(e) for e in pattern_can.extract_graph().edges}
    assert {frozenset(e) for e in pattern_dev.extract_graph().edges} == edges_set
    edges = list(edges_set)

    nnodes = nqubits * (4 * nlayers + 1)
    nodes_can = list(range(nnodes))
    nodes_dev = [
        row + col * nqubits
        for col in range(1, 4 * nlayers + 1, 2)
        for row in range(0, nqubits, 2)
    ]

    qubit_edge_matrix = generate_qubit_edge_matrix_from_pattern(
        pattern_can, nodes_can, edges
    )
    qubit_edge_matrix_dev = generate_qubit_edge_matrix_from_pattern(
        pattern_dev, nodes_dev, edges
    )

    # Stack the matrices together to form a single system
    lhs = np.vstack((qubit_edge_matrix, qubit_edge_matrix_dev))
    logger.debug(f"{lhs.shape=}")
    return edges, lhs


def compute_aces_postprocessing(
    nqubits: int, nnodes: int, nlayers: int, results: SimulationResult
) -> dict[frozenset[int], float]:
    logger.info("Computing failure probabilities...")
    failure_proba_can_final = compute_failure_probabilities(nnodes, results.canonical)
    failure_proba_dev_all = compute_failure_probabilities(nnodes, results.deviant)

    # computing circuit eigenvalues for both orders
    # deviant has been filtered to remove redundancy
    failure_proba_can = compute_probabilities_difference_can(
        failure_proba_can_final, nnodes
    )
    failure_proba_dev = compute_probabilities_difference_dev(
        failure_proba_dev_all,
        nnodes,
        nqubits,
    )

    # convert to numpy arrays for later processing
    py_failure_proba_can = np.array(failure_proba_can, dtype=np.float64)
    py_failure_proba_dev = np.array(failure_proba_dev, dtype=np.float64)

    logger.info("Setting up ACES...")
    edges, lhs = generate_qubit_edge_matrix(nqubits, nlayers)
    rhs = np.concatenate((py_failure_proba_can, py_failure_proba_dev))

    log_rhs = np.log(rhs)  # log constant vectors

    log_params, *_ = np.linalg.lstsq(lhs, log_rhs, rcond=None)

    logger.info("Calculating the lambdas...")
    return dict(zip(edges, np.exp(log_params), strict=True))


def generate_plot(
    inferred_lambdas: list[float],
    vline: float,
    target: Path = Path("plot.png"),
) -> None:
    """generate plot based on data: absolute value.
    vline for position of theoretical expectation
    """

    plt.figure(figsize=(10, 6))

    # Create histogram with density curve
    plt.hist(
        inferred_lambdas,
        bins="auto",
        color="#2ecc71",
        edgecolor="#27ae60",
        alpha=0.7,
        density=True,
    )

    # Add KDE plot
    sns.kdeplot(  # type: ignore[no-untyped-call]
        inferred_lambdas,
        color="#34495e",
        linewidth=2,
        label="KDE",
    )

    # Add reference lines
    plt.axvline(vline, color="red", linestyle="--", linewidth=1.5)
    plt.axvline(
        np.mean(inferred_lambdas),
        color="#3498db",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean ({np.mean(inferred_lambdas):.2f})",
    )
    plt.axvline(
        np.median(inferred_lambdas),
        color="#9b59b6",
        linestyle="-",
        linewidth=1.5,
        label=f"Median ({np.median(inferred_lambdas):.2f})",
    )

    # Formatting
    plt.title(r"ACES", fontsize=14)
    if vline != 0:
        plt.xlabel(r"${\hat \lambda}$", fontsize=12)
    else:
        plt.xlabel(r"${\hat \lambda} - \lambda_{\rm th}$", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    sns.despine()  # type: ignore[no-untyped-call]

    # Add statistical annotations
    stats_text = (
        f"Total edges: {len(inferred_lambdas)}\n"
        f"Min: {np.min(inferred_lambdas):.3f}\n"
        f"Max: {np.max(inferred_lambdas):.3f}\n"
        f"Std: {np.std(inferred_lambdas):.3f}"
    )
    plt.text(
        0.75,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.9},
    )

    plt.tight_layout()
    plt.savefig(target)


def cli(
    nqubits: int = 5,
    nlayers: int = 10,
    depol_prob: float = 0.001,
    nshots: int = 10000,
    ncircuits: int = 1,
    verbose: bool = False,
    method: Method | None = None,
    smoke: bool = False,
    target: Path = Path("plot.png"),
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)

    # Fast smoke configuration: runs in a few seconds, just enough to revalidate the port.
    if smoke:
        nqubits, nlayers, nshots = 2, 3, 4000

    nnodes = nqubits * ((4 * nlayers) + 1)

    # typer does not support default values for Enum
    if method is None:
        method = Method.Stim

    noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=depol_prob)

    logger.info("Starting simulations...")
    start = time.time()
    results = perform_simulation(
        nqubits=nqubits,
        nlayers=nlayers,
        noise_model=noise_model,
        nshots=nshots,
        ncircuits=ncircuits,
        method=method,
    )

    logger.info(f"Simulation finished in {time.time() - start:.4f} seconds.")
    start = time.time()
    dependencies = generate_edge_dependencies(nqubits, nlayers)
    inferred_lambdas = compute_aces_postprocessing_iteratively(
        nnodes, dependencies, results
    ).values()

    logger.info(f"Lambda inferred in {time.time() - start:.4f} seconds.")

    # expected theoretical value of the lambdas
    lambda_expected = 1 - 4 / 3 * depol_prob
    # compute difference between inferred and theoretical values
    lambda_diff: list[float] = [l - lambda_expected for l in inferred_lambdas]

    logger.info(
        f"Inferred lambda: mean={np.mean(list(inferred_lambdas)):.6f} "
        f"theory={lambda_expected:.6f} (n_edges={len(lambda_diff)})"
    )

    logger.info("Plotting the result...")

    # absolute plot
    generate_plot(list(inferred_lambdas), vline=1 - 4 * depol_prob / 3, target=target)

    # difference wrt theoretical expectation
    generate_plot(list(lambda_diff), vline=0, target=Path("plot_diff.png"))
    logger.info("Done!")


if __name__ == "__main__":
    freeze_support()
    typer.run(cli)

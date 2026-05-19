import json
import random
from pathlib import Path

import typer
from graphix.sim.statevec import StatevectorBackend

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.verifying import TrappifiedSchemeParameters

CIRCUITS_DIR = Path("applications/circuits")
TABLE_PATH = CIRCUITS_DIR / "table.json"

app = typer.Typer()


def load_pattern_from_circuit(circuit_label: str):
    with (CIRCUITS_DIR / circuit_label).open() as f:
        circuit = read_qasm(f)
    pattern = transpile(circuit)
    pattern.minimize_space()
    return pattern


def find_correct_value(circuit_name: str) -> int:
    with TABLE_PATH.open() as f:
        table = json.load(f)
    return round(table[circuit_name])


@app.command()
def main(
    comp_rounds: int = typer.Option(20, help="Number of computation rounds"),
    test_rounds: int = typer.Option(20, help="Number of test rounds"),
    threshold: int = typer.Option(5, help="Trap failure threshold"),
    blind: bool = typer.Option(True, help="Enable blinding"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    rng = __import__("numpy").random.default_rng(seed)

    with TABLE_PATH.open() as f:
        table = json.load(f)

    circuit_label = random.choice(list(table.keys()))
    typer.echo(f"Circuit: {circuit_label}")

    pattern = load_pattern_from_circuit(circuit_label)
    secrets = Secrets(r=blind, a=blind, theta=blind)
    parameters = TrappifiedSchemeParameters(
        comp_rounds=comp_rounds,
        test_rounds=test_rounds,
        threshold=threshold,
    )

    client = Client(pattern=pattern, secrets=secrets, parameters=parameters, rng=rng)

    canvas = client.sample_canvas(rng=rng)
    outcomes = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=rng)
    traps_decision, computation_decision, result_analysis = client.analyze_outcomes(canvas, outcomes)

    expected = find_correct_value(circuit_label)

    typer.echo(f"Traps passed: {traps_decision}")
    typer.echo(f"Failed test rounds: {result_analysis.nr_failed_test_rounds}/{test_rounds}")
    typer.echo(f"Computation decision: {computation_decision}")
    typer.echo(f"Expected answer: {expected}")
    if computation_decision is not None:
        match = int(computation_decision) == expected
        typer.echo(f"Correct: {match}")


if __name__ == "__main__":
    app()

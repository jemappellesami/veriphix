"""run.py — local one-shot driver for the noise-learning experiment.

Runs simulate.py (10 circuits, shared Gaussian-region noise model) then plot.py.

Usage
-----
    python applications/noise_learning/run.py
    python applications/noise_learning/run.py --n-circuits 10 --n-test-rounds 100 --force
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(add_completion=False)
HERE = Path(__file__).parent


@app.command()
def main(
    n_circuits:    Annotated[int,  typer.Option(help="Number of circuits to average over")] = 10,
    n_test_rounds: Annotated[int,  typer.Option(help="Test rounds per circuit")]            = 100,
    force:         Annotated[bool, typer.Option(help="Re-run circuits even if CSV exists")] = False,
) -> None:
    sim_cmd = [
        sys.executable, str(HERE / "simulate.py"),
        "--n-circuits", str(n_circuits),
        "--n-test-rounds", str(n_test_rounds),
    ]
    if force:
        sim_cmd.append("--force")
    typer.echo(f"$ {' '.join(sim_cmd)}")
    subprocess.run(sim_cmd, check=True)

    plot_cmd = [
        sys.executable, str(HERE / "plot.py"),
        "--n-test-rounds", str(n_test_rounds),
    ]
    typer.echo(f"$ {' '.join(plot_cmd)}")
    subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    app()

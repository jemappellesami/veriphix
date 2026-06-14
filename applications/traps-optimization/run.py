"""run.py — one-shot driver for the trap-optimisation experiment.

Runs optimize.py (solve Problem 1 on the learned heatmap) then plot.py.
No quantum simulation — both steps are fast.

Usage
-----
    python applications/traps-optimization/run.py
    python applications/traps-optimization/run.py --threshold 0.1
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
    threshold:   Annotated[float, typer.Option(help="Failure-rate threshold for a node to be 'noisy'")] = 0.05,
    heatmap_dir: Annotated[Path,  typer.Option(help="Learned-heatmap results dir")]                      = Path("applications/noise_learning/results"),
) -> None:
    opt_cmd = [
        sys.executable, str(HERE / "optimize.py"),
        "--threshold", str(threshold),
        "--heatmap-dir", str(heatmap_dir),
    ]
    typer.echo(f"$ {' '.join(opt_cmd)}")
    subprocess.run(opt_cmd, check=True)

    plot_cmd = [sys.executable, str(HERE / "plot.py")]
    typer.echo(f"$ {' '.join(plot_cmd)}")
    subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    app()

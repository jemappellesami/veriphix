"""hotgate.py — local convenience runner (no SLURM).

Runs all circuits sequentially by calling simulate.py for each one, then
calls plot.py.  For cluster use, run simulate.py via submit.sh instead.

Usage
-----
    python applications/gospel/hotgate/hotgate.py
    python applications/gospel/hotgate/hotgate.py \\
        --n-circuits 100 --n-test-rounds 100 --p-ent 2e-3
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(add_completion=False)

HERE = Path(__file__).parent


@app.command()
def main(
    n_circuits:    Annotated[int,   typer.Option(help="Number of circuits to process")] = 100,
    circuits_dir:  Annotated[Path,  typer.Option(help="Directory with .qasm circuit files")] = Path("applications/gospel/circuits/circuits-3-6"),
    n_test_rounds: Annotated[int,   typer.Option(help="Test rounds per circuit")]              = 100,
    p_ent:         Annotated[float, typer.Option(help="Depolarising entanglement error")]      = 2e-3,
    base_seed:     Annotated[int,   typer.Option(help="Base RNG seed")]                        = 42,
    out_dir:       Annotated[Path,  typer.Option(help="Directory for per-circuit CSVs")]       = Path("applications/gospel/hotgate/results"),
    out_plot:      Annotated[Path,  typer.Option(help="Output heatmap PDF")]                   = Path("applications/gospel/hotgate/heatmap.pdf"),
    force:         Annotated[bool,  typer.Option(help="Re-run circuits even if CSV exists")]   = False,
) -> None:
    t_total = time.monotonic()

    typer.echo(f"Running {n_circuits} circuits  (p_ent={p_ent:.1e}, rounds={n_test_rounds})")
    typer.echo(f"Results → {out_dir}")

    simulate_py = HERE / "simulate.py"

    for idx in range(n_circuits):
        out_csv = out_dir / f"circuit_{idx:03d}.csv"
        if out_csv.exists() and not force:
            typer.echo(f"  [{idx+1}/{n_circuits}] skip (already done)")
            continue

        elapsed = time.monotonic() - t_total
        m_el, s_el = divmod(int(elapsed), 60)
        typer.echo(f"  [{idx+1}/{n_circuits}] elapsed {m_el}m{s_el:02d}s", nl=False)

        cmd = [
            sys.executable, str(simulate_py),
            "--circuit-idx",    str(idx),
            "--circuits-dir",   str(circuits_dir),
            "--n-test-rounds",  str(n_test_rounds),
            "--p-ent",          str(p_ent),
            "--base-seed",      str(base_seed),
            "--out-dir",        str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            typer.echo(f"\n[ERROR] circuit {idx} failed:\n{result.stderr}")
            raise typer.Exit(1)

        # Print the last line of simulate.py output (timing info).
        last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        typer.echo(f"  →  {last_line}")

    total = time.monotonic() - t_total
    m, s = divmod(int(total), 60)
    typer.echo(f"\nAll done in {m}m{s:02d}s.  Plotting …\n")

    plot_py = HERE / "plot.py"
    cmd = [
        sys.executable, str(plot_py),
        "--results-dir",  str(out_dir),
        "--circuits-dir", str(circuits_dir),
        "--out",          str(out_plot),
        "--p-ent",        str(p_ent),
        "--n-test-rounds", str(n_test_rounds),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"[ERROR] plot.py failed:\n{result.stderr}")
        raise typer.Exit(1)
    typer.echo(result.stdout.strip())


if __name__ == "__main__":
    app()

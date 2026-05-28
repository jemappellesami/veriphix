"""Full circuit pipeline: generate + BQP-sample in one shot.

Steps
-----
1. For each (nqubits, depth) in the sweep, call the sampling_circuits CLI to
   generate raw circuits into  circuits/<circuits-n-d>/
2. For each generated folder, filter to BQP-hard circuits (prob < bqp_error or
   1 - prob < bqp_error) and copy n_shots of them into
   sampled_circuits/<circuits-n-d-{bqp_tag}>/

Usage
-----
    python applications/gospel/circuits_pipeline.py \\
        --nqubits-min 3 --nqubits-max 5 \\
        --depth-min 5   --depth-max 7   \\
        --bqp-error 0.1 --n-shots 100
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)

BASE          = Path("applications/gospel")
CIRCUITS_BASE = BASE / "circuits-2"
SAMPLED_BASE  = BASE / "sampled_circuits-2"

# ── circuit generation defaults ───────────────────────────────────────────────
NCIRCUITS   = 1000
P_GATE      = 0.5
P_CNOT      = 0.25
P_CNOT_FLIP = 0.5
P_RX        = 0.5
GEN_SEED    = 1729


def _bqp_tag(bqp_error: float) -> str:
    return f"{bqp_error:.0e}".replace("-0", "-").replace("+0", "")


# ── step 1: generate ──────────────────────────────────────────────────────────

def generate_circuits(nqubits: int, depth: int) -> Path:
    target = CIRCUITS_BASE / f"circuits-{nqubits}-{depth}"
    if target.exists():
        typer.echo(f"  [gen] Skipping {target.name} (already exists)")
        return target

    cmd = [
        sys.executable, "-m", "veriphix.sampling_circuits.sampling_circuits",
        "--ncircuits",   str(NCIRCUITS),
        "--nqubits",     str(nqubits),
        "--depth",       str(depth),
        "--p-gate",      str(P_GATE),
        "--p-cnot",      str(P_CNOT),
        "--p-cnot-flip", str(P_CNOT_FLIP),
        "--p-rx",        str(P_RX),
        "--seed",        str(GEN_SEED),
        "--target",      str(target),
    ]
    typer.echo(f"  [gen] Generating n={nqubits}, d={depth} → {target.name}")
    subprocess.run(cmd, check=True)
    return target


# ── step 2: BQP-sample ────────────────────────────────────────────────────────

def sample_bqp(source: Path, bqp_error: float, n_shots: int, seed: int) -> Path | None:
    with (source / "table.json").open() as f:
        table: dict[str, float] = json.load(f)

    candidates = [
        fname for fname, prob in table.items()
        if prob < bqp_error or 1 - prob < bqp_error
    ]

    if not candidates:
        typer.echo(f"  [bqp] Skipping {source.name}: no circuits satisfy bqp_error={bqp_error}")
        return None

    n = min(n_shots, len(candidates))
    if n < n_shots:
        typer.echo(f"  [bqp] Warning: requested {n_shots} but only {len(candidates)} candidates")

    dest = SAMPLED_BASE / f"{source.name}-{_bqp_tag(bqp_error)}"
    if dest.exists():
        typer.echo(f"  [bqp] Skipping {dest.name} (already exists)")
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    sampled = random.Random(seed).sample(candidates, n)
    sampled_table: dict[str, float] = {}
    for fname in sampled:
        shutil.copy(source / fname, dest / fname)
        sampled_table[fname] = table[fname]

    with (dest / "table.json").open("w") as f:
        json.dump(sampled_table, f, indent=2)

    typer.echo(f"  [bqp] {source.name} → {dest.name}  ({len(sampled)} circuits, pool {len(candidates)}/{len(table)})")
    return dest


# ── pipeline ──────────────────────────────────────────────────────────────────

@app.command()
def main(
    nqubits_min: Annotated[int,   typer.Option(help="Min nqubits (inclusive)")] = 3,
    nqubits_max: Annotated[int,   typer.Option(help="Max nqubits (inclusive)")] = 5,
    depth_min:   Annotated[int,   typer.Option(help="Min depth (inclusive)")]   = 5,
    depth_max:   Annotated[int,   typer.Option(help="Max depth (inclusive)")]   = 7,
    bqp_error:   Annotated[float, typer.Option(help="BQP error threshold")]     = 0.1,
    n_shots:     Annotated[int,   typer.Option(help="Circuits to sample per (n,d) cell")] = 100,
    seed:        Annotated[int,   typer.Option(help="Random seed for BQP sampling")]      = 0,
) -> None:
    CIRCUITS_BASE.mkdir(parents=True, exist_ok=True)
    SAMPLED_BASE.mkdir(parents=True, exist_ok=True)

    nqubits_range = range(nqubits_min, nqubits_max + 1)
    depth_range   = range(depth_min,   depth_max   + 1)
    total         = len(nqubits_range) * len(depth_range)

    typer.echo(
        f"\nPipeline: nqubits={list(nqubits_range)}  depth={list(depth_range)}"
        f"  bqp_error={bqp_error}  n_shots={n_shots}  ({total} cells)\n"
    )

    for nqubits in nqubits_range:
        for depth in depth_range:
            typer.echo(f"── n={nqubits}, d={depth} ──")
            source = generate_circuits(nqubits, depth)
            sample_bqp(source, bqp_error, n_shots, seed)

    typer.echo("\nDone.")


if __name__ == "__main__":
    app()

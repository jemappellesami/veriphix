"""
Goal of the script

being given a folder of circuits like `circuits-n-d` and parameters `bqp_error`, and `n_circuits`, sample randomly `n_circuits` circuits from the circuits from the folder that satisfy the following property:
their value in tables.json either satisfies `value<bqp_error`, or `1-value < bqp_error`

write them in `sampled_circuits/circuits-n-d-bqp` where now `bqp` is the BQP error

"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


CIRCUITS_BASE = Path("applications/gospel/circuits")


def sample_folder(source: Path, bqp_error: float, n_circuits: int, output_base: Path, seed: int) -> None:
    with (source / "table.json").open() as f:
        table: dict[str, float] = json.load(f)

    candidates = [
        filename for filename, prob in table.items()
        if prob < bqp_error or 1 - prob < bqp_error
    ]

    if not candidates:
        typer.echo(f"  Skipping {source.name}: no circuits satisfy bqp_error={bqp_error}")
        return

    n = min(n_circuits, len(candidates))
    if n < n_circuits:
        typer.echo(f"  Warning: requested {n_circuits} but only {len(candidates)} candidates — using all of them")

    rng = random.Random(seed)
    sampled = rng.sample(candidates, n)

    bqp_tag = f"{bqp_error:.0e}".replace("-0", "-").replace("+0", "")
    dest = output_base / f"{source.name}-{bqp_tag}"
    dest.mkdir(parents=True, exist_ok=True)

    sampled_table: dict[str, float] = {}
    for filename in sampled:
        shutil.copy(source / filename, dest / filename)
        sampled_table[filename] = table[filename]

    with (dest / "table.json").open("w") as f:
        json.dump(sampled_table, f, indent=2)

    typer.echo(f"  {source.name} → {dest.name}  ({len(sampled)} circuits, pool {len(candidates)}/{len(table)})")


@app.command()
def main(
    bqp_error: Annotated[float, typer.Option(help="BQP error threshold")],
    n_circuits: Annotated[int, typer.Option(help="Number of circuits to sample per folder")],
    circuits_base: Annotated[Path, typer.Option(help="Base directory containing circuits-n-d folders")] = CIRCUITS_BASE,
    output_base: Annotated[Path, typer.Option(help="Base output directory")] = Path("applications/gospel/sampled_circuits"),
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
) -> None:
    sources = sorted(p for p in circuits_base.iterdir() if p.is_dir() and (p / "table.json").exists())

    if not sources:
        typer.echo(f"No circuit folders found in {circuits_base}")
        raise typer.Exit(1)

    bqp_tag = f"{bqp_error:.0e}".replace("-0", "-").replace("+0", "")
    typer.echo(f"Sampling bqp_error={bqp_error} ({bqp_tag}) across {len(sources)} folders → {output_base}")

    for source in sources:
        sample_folder(source, bqp_error, n_circuits, output_base, seed)


if __name__ == "__main__":
    app()
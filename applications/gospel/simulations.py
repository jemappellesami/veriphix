"""Gospel benchmark — Stage 2: noise sweep over BQP-sampled circuits.

For each sampled folder (circuits-n-d-bqp) and each entanglement error rate,
runs VBQC test rounds and records the trap failure rate to a CSV.

Usage
-----
    python applications/gospel/simulations.py
    python applications/gospel/simulations.py --test-rounds 50 --out-csv results.csv
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from graphix.noise_models import DepolarisingNoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import RandomTraps
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.verifying import TrappifiedSchemeParameters

app = typer.Typer(add_completion=False)

SAMPLED_BASE = Path("applications/gospel/sampled_circuits")

ENT_ERRORS: list[float] = list(np.logspace(-6, -1, num=10))

CSV_HEADER = ["p_ent", "width", "depth", "bqp_error", "circuit_label", "traps_passed", "nr_failed_test_rounds", "test_rounds"]

FOLDER_RE = re.compile(r"^circuits-(\d+)-(\d+)-(.+)$")


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _load_done(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Return the set of (p_ent, width, depth, bqp_error, circuit_label) already in the CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        return {
            (row["p_ent"], row["width"], row["depth"], row["bqp_error"], row["circuit_label"])
            for row in csv.DictReader(f)
        }


def _open_csv(path: Path) -> tuple[csv.DictWriter, object]:
    is_new = not path.exists() or path.stat().st_size == 0
    fh = path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
    if is_new:
        writer.writeheader()
    return writer, fh


def _load_pattern(circuit_path: Path):
    with circuit_path.open() as f:
        circuit = read_qasm(f)
    pattern = transpile(circuit)
    pattern.minimize_space()
    return pattern


def _parse_folder(name: str) -> tuple[int, int, str] | None:
    m = FOLDER_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


# ── simulation ────────────────────────────────────────────────────────────────

@app.command()
def main(
    sampled_base: Annotated[Path,  typer.Option(help="Directory of sampled circuit folders")] = SAMPLED_BASE,
    test_rounds:  Annotated[int,   typer.Option(help="Test rounds per canvas")]                = 100,
    out_csv:      Annotated[Path,  typer.Option(help="Output CSV path")]                       = Path("applications/gospel/gospel_results.csv"),
    seed:         Annotated[int,   typer.Option(help="RNG seed")]                              = 42,
) -> None:
    rng = np.random.default_rng(seed)

    folders = sorted(
        p for p in sampled_base.iterdir()
        if p.is_dir() and (p / "table.json").exists() and _parse_folder(p.name)
    )

    if not folders:
        typer.echo(f"No sampled circuit folders found in {sampled_base}")
        raise typer.Exit(1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_csv)
    if done:
        typer.echo(f"Resuming: {len(done)} runs already in {out_csv}")
    csv_writer, csv_fh = _open_csv(out_csv)

    total = len(folders) * len(ENT_ERRORS)
    grand_cell = 0
    t_run_start = time.monotonic()

    try:
        for folder in folders:
            parsed = _parse_folder(folder.name)
            assert parsed is not None
            width, depth, bqp_tag = parsed

            with (folder / "table.json").open() as f:
                table: dict[str, float] = json.load(f)

            circuit_files = [folder / fname for fname in sorted(table) if fname.endswith(".qasm")]

            typer.echo(f"\n{'='*64}")
            typer.echo(f"Folder: {folder.name}  (n={width}, d={depth}, bqp={bqp_tag}, {len(circuit_files)} circuits)")
            typer.echo(f"{'='*64}")

            t_folder_start = time.monotonic()

            for p_idx, p_ent in enumerate(ENT_ERRORS):
                grand_cell += 1
                t_cell_start = time.monotonic()

                noise_model = DepolarisingNoiseModel(
                    entanglement_error_prob=p_ent,
                    measure_error_prob=0.0,
                    x_error_prob=0.0,
                    z_error_prob=0.0,
                    measure_channel_prob=0.0,
                )

                for c_idx, circuit_path in enumerate(circuit_files):
                    if (str(p_ent), str(width), str(depth), bqp_tag, circuit_path.name) in done:
                        continue

                    elapsed = time.monotonic() - t_cell_start
                    avg = elapsed / c_idx if c_idx > 0 else 0.0
                    eta = avg * (len(circuit_files) - c_idx)
                    typer.echo(
                        f"  p_ent={p_ent:.1e} [{p_idx+1}/{len(ENT_ERRORS)}]"
                        f"  circuit {c_idx+1}/{len(circuit_files)}"
                        f"  elapsed {_fmt_time(elapsed)}  ETA {_fmt_time(eta)}",
                        nl=False,
                    )
                    typer.echo("\r", nl=False)

                    pattern = _load_pattern(circuit_path)

                    client = Client(
                        pattern=pattern,
                        secrets=Secrets(a=True, r=True, theta=True),
                        protocol=RandomTraps(),
                        parameters=TrappifiedSchemeParameters(
                            comp_rounds=0, test_rounds=test_rounds, threshold=0
                        ),
                        rng=rng,
                    )

                    canvas = client.sample_canvas(rng=rng)
                    outcomes = client.delegate_canvas(
                        canvas=canvas,
                        backend_cls=DensityMatrixBackend,
                        noise_model=noise_model,
                        rng=rng,
                    )
                    traps_ok, _, result_analysis = client.analyze_outcomes(canvas, outcomes)

                    csv_writer.writerow({
                        "p_ent":                  p_ent,
                        "width":                  width,
                        "depth":                  depth,
                        "bqp_error":              bqp_tag,
                        "circuit_label":          circuit_path.name,
                        "traps_passed":           traps_ok,
                        "nr_failed_test_rounds":  result_analysis.nr_failed_test_rounds,
                        "test_rounds":            test_rounds,
                    })
                    csv_fh.flush()  # type: ignore[union-attr]

                cell_time      = time.monotonic() - t_cell_start
                folder_elapsed = time.monotonic() - t_folder_start
                run_elapsed    = time.monotonic() - t_run_start
                avg_grand      = run_elapsed / grand_cell
                eta_total      = avg_grand * (total - grand_cell)

                typer.echo(
                    f"  p_ent={p_ent:.1e} [{p_idx+1}/{len(ENT_ERRORS)}] done"
                    f"  | cell {_fmt_time(cell_time)}"
                    f"  folder {_fmt_time(folder_elapsed)}"
                    f"  total {_fmt_time(run_elapsed)}"
                    f"  ETA {_fmt_time(eta_total)}"
                )

    finally:
        csv_fh.close()  # type: ignore[union-attr]

    typer.echo(f"\nDone. Results saved to {out_csv}")
    typer.echo(f"Total runtime: {_fmt_time(time.monotonic() - t_run_start)}")


if __name__ == "__main__":
    app()

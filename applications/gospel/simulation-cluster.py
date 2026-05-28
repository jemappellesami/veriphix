"""Gospel benchmark — cluster-friendly version with Dask parallelism.

Each (circuit, p_ent) pair is submitted as an independent Dask future, so the
work fans out across all available workers — either local cores or SLURM nodes.
Results are written to CSV as futures complete (not in submission order).

Usage — local (uses all CPU cores)
-----------------------------------
    python applications/gospel/simulation-cluster.py

Usage — SLURM cluster
---------------------
    python applications/gospel/simulation-cluster.py \\
        --walltime 2 --memory 8 --cores 4 --port 8787 --scale 20
"""
from __future__ import annotations

import csv
import dataclasses
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import dask.distributed
import numpy as np
import typer
from dask_jobqueue import SLURMCluster
from graphix.noise_models import DepolarisingNoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile, get_bipartite_coloring
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.verifying import TrappifiedSchemeParameters

app = typer.Typer(add_completion=False)

SAMPLED_BASE = Path("applications/gospel/sampled_circuits")
ENT_ERRORS: list[float] = list(np.logspace(-6, -1, num=10))
# Removing the first couple of them
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
FOLDER_RE = re.compile(r"^circuits-(\d+)-(\d+)-(.+)$")


# ── result types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Result:
    p_ent: float
    width: int
    depth: int
    bqp_error: str
    circuit_label: str
    traps_passed: bool
    nr_failed_test_rounds: int
    test_rounds: int
    elapsed_s: float = 0.0  # wall-time for this run; not written to CSV


@dataclass(frozen=True)
class Failure:
    p_ent: float
    width: int
    depth: int
    bqp_error: str
    circuit_label: str
    error: str
    elapsed_s: float = 0.0


CSV_FIELDS = [f.name for f in dataclasses.fields(Result) if f.name != "elapsed_s"]

FolderKey = tuple[int, int, str]


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


# ── run unit ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Run:
    circuit_path: str          # str so it is trivially serialisable by Dask
    width: int
    depth: int
    bqp_error: str
    p_ent: float
    test_rounds: int
    seed: int

    def execute(self) -> Result | Failure:
        t0 = time.monotonic()
        circuit_path = Path(self.circuit_path)
        try:
            with circuit_path.open() as f:
                circuit = read_qasm(f)
            pattern = transpile(circuit)
            pattern.minimize_space()

            rng = np.random.default_rng(self.seed)
            noise_model = DepolarisingNoiseModel(
                entanglement_error_prob=self.p_ent,
                measure_error_prob=0.0,
                x_error_prob=0.0,
                z_error_prob=0.0,
                measure_channel_prob=0.0,
            )

            client = Client(
                pattern=pattern,
                secrets=Secrets(a=True, r=True, theta=True),
                protocol=FK12(manual_colouring=get_bipartite_coloring(pattern=pattern)),
                parameters=TrappifiedSchemeParameters(
                    comp_rounds=0, test_rounds=self.test_rounds, threshold=0
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

            return Result(
                p_ent=self.p_ent,
                width=self.width,
                depth=self.depth,
                bqp_error=self.bqp_error,
                circuit_label=circuit_path.name,
                traps_passed=traps_ok,
                nr_failed_test_rounds=result_analysis.nr_failed_test_rounds,
                test_rounds=self.test_rounds,
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            return Failure(
                p_ent=self.p_ent,
                width=self.width,
                depth=self.depth,
                bqp_error=self.bqp_error,
                circuit_label=Path(self.circuit_path).name,
                error=str(exc),
                elapsed_s=time.monotonic() - t0,
            )


# ── cluster helpers ───────────────────────────────────────────────────────────

def _get_cluster(
    walltime: int | None,
    memory: int | None,
    cores: int | None,
    port: int | None,
    scale: int | None,
) -> dask.distributed.deploy.cluster.Cluster:
    if walltime is None and memory is None and cores is None:
        cluster: dask.distributed.deploy.cluster.Cluster = dask.distributed.LocalCluster()
    else:
        for name, val in [("--walltime", walltime), ("--memory", memory), ("--cores", cores), ("--port", port), ("--scale", scale)]:
            if val is None:
                raise ValueError(f"{name} is required for SLURM")
        cluster = SLURMCluster(
            account="inria",
            queue="cpu_devel",
            cores=cores,
            memory=f"{memory}GB",
            walltime=f"{walltime}:00:00",
            scheduler_options={"dashboard_address": f":{port}"},
        )
    if scale is not None:
        cluster.scale(scale)
    return cluster


def _load_done(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Return the set of (p_ent, width, depth, bqp_error, circuit_label) already in the CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        return {
            (row["p_ent"], row["width"], row["depth"], row["bqp_error"], row["circuit_label"])
            for row in csv.DictReader(f)
        }


def _parse_folder(name: str) -> tuple[int, int, str] | None:
    m = FOLDER_RE.match(name)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


# ── main ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    sampled_base: Annotated[Path, typer.Option()] = SAMPLED_BASE,
    test_rounds:  Annotated[int,  typer.Option()] = 100,
    out_csv:      Annotated[Path, typer.Option()] = Path("applications/gospel/gospel_results_cluster.csv"),
    seed:         Annotated[int,  typer.Option()] = 42,
    walltime:     Annotated[int | None, typer.Option(help="SLURM: walltime in hours")] = None,
    memory:       Annotated[int | None, typer.Option(help="SLURM: memory in GB")]      = None,
    cores:        Annotated[int | None, typer.Option(help="SLURM: cores per job")]     = None,
    port:         Annotated[int | None, typer.Option(help="SLURM: dashboard port")]    = None,
    scale:        Annotated[int | None, typer.Option(help="Number of workers")]        = None,
) -> None:
    folders = sorted(
        p for p in sampled_base.iterdir()
        if p.is_dir() and (p / "table.json").exists() and _parse_folder(p.name)
    )
    if not folders:
        typer.echo(f"No sampled circuit folders found in {sampled_base}")
        raise typer.Exit(1)

    # Build all runs, each with a unique seed derived from the master seed
    ss = np.random.SeedSequence(seed)
    runs: list[Run] = []
    for folder in folders:
        parsed = _parse_folder(folder.name)
        assert parsed is not None
        width, depth, bqp_tag = parsed
        with (folder / "table.json").open() as f:
            table: dict[str, float] = json.load(f)
        circuit_files = sorted(folder / fname for fname in table if fname.endswith(".qasm"))
        for p_ent in ENT_ERRORS:
            for circuit_path in circuit_files:
                child_seed = int(ss.spawn(1)[0].generate_state(1)[0])
                runs.append(Run(
                    circuit_path=str(circuit_path),
                    width=width,
                    depth=depth,
                    bqp_error=bqp_tag,
                    p_ent=p_ent,
                    test_rounds=test_rounds,
                    seed=child_seed,
                ))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_csv)
    if done:
        typer.echo(f"Resuming: {len(done)} runs already in {out_csv}")
    runs = [
        r for r in runs
        if (str(r.p_ent), str(r.width), str(r.depth), r.bqp_error, Path(r.circuit_path).name) not in done
    ]

    typer.echo(f"{len(runs)} runs to submit across {len(folders)} folders × {len(ENT_ERRORS)} noise levels")

    is_new = not out_csv.exists() or out_csv.stat().st_size == 0

    cluster = _get_cluster(walltime, memory, cores, port, scale)
    dask_client = dask.distributed.Client(cluster)

    futures = [dask_client.submit(Run.execute, run, pure=False) for run in runs]

    # Per-folder totals (keyed by (width, depth, bqp_error))
    folder_total: dict[FolderKey, int] = defaultdict(int)
    folder_done:  dict[FolderKey, int] = defaultdict(int)
    folder_time:  dict[FolderKey, float] = defaultdict(float)
    for r in runs:
        folder_total[(r.width, r.depth, r.bqp_error)] += 1

    n_ok = n_fail = 0
    loop_start = time.monotonic()

    with out_csv.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()

        for fut in dask.distributed.as_completed(futures):
            try:
                report = fut.result()
            except Exception as exc:
                typer.echo(f"Future error: {exc}")
                continue

            fkey: FolderKey = (report.width, report.depth, report.bqp_error)
            folder_done[fkey] += 1
            folder_time[fkey] += report.elapsed_s

            wall_elapsed = time.monotonic() - loop_start
            n_done = n_ok + n_fail + 1
            throughput = n_done / wall_elapsed          # runs/s (reflects parallelism)
            remaining = len(runs) - n_done
            eta_str = _fmt_duration(remaining / throughput) if throughput > 0 else "?"

            f_done = folder_done[fkey]
            f_total = folder_total[fkey]
            f_avg = folder_time[fkey] / f_done

            if isinstance(report, Result):
                writer.writerow({k: v for k, v in dataclasses.asdict(report).items() if k in CSV_FIELDS})
                csvfile.flush()
                n_ok += 1
                typer.echo(
                    f"[{n_ok+n_fail}/{len(runs)}] "
                    f"n={report.width} d={report.depth} p={report.p_ent:.1e} "
                    f"{report.circuit_label}  "
                    f"traps={'✓' if report.traps_passed else '✗'}  "
                    f"failed={report.nr_failed_test_rounds}/{report.test_rounds}  "
                    f"t={report.elapsed_s:.1f}s  "
                    f"folder {f_done}/{f_total} avg={f_avg:.1f}s/run  "
                    f"ETA {eta_str}"
                )
            elif isinstance(report, Failure):
                n_fail += 1
                typer.echo(
                    f"✗ [{n_ok+n_fail}/{len(runs)}] {report.circuit_label} p={report.p_ent:.1e}  "
                    f"t={report.elapsed_s:.1f}s  ETA {eta_str}: {report.error}"
                )

    typer.echo(f"\nDone. {n_ok} results, {n_fail} failures → {out_csv}")


if __name__ == "__main__":
    app()

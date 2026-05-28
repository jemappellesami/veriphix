"""Gospel benchmark — trap-level failure extraction.

Same Dask parallelism as simulation.py, but instead of one row per
(circuit, p_ent), writes one row per (circuit, p_ent, round, trap) —
capturing exactly which trap fired in each test round.

CSV schema
----------
p_ent, width, depth, bqp_error, circuit_label,
round_idx, trap_nodes, outcome

- trap_nodes : JSON array of sorted qubit indices, e.g. [3] or [3, 7, 12]
- outcome    : 0 (pass) or 1 (fired)

Works for any protocol (FK12 single-qubit traps, RandomTraps multi-qubit traps,
Dummyless multi-qubit traps) — trap_nodes encodes the size unambiguously.

Usage — local
-------------
    python applications/gospel/benchmark/simulation-extract.py

Usage — SLURM cluster
---------------------
    python applications/gospel/benchmark/simulation-extract.py \\
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
from dask_jobqueue.slurm import SLURMCluster
from graphix.noise_models import DepolarisingNoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12, get_bipartite_coloring
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.verifying import TestResult, TrappifiedSchemeParameters

app = typer.Typer(add_completion=False)

SAMPLED_BASE = Path("applications/gospel/sampled_circuits-small")
ENT_ERRORS: list[float] = list(np.logspace(-6, -1, num=10))
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
ENT_ERRORS.pop(0)
ENT_ERRORS.pop()
ENT_ERRORS.pop()
ENT_ERRORS.pop()
ENT_ERRORS.pop()
FOLDER_RE = re.compile(r"^circuits-(\d+)-(\d+)-(.+)$")


# ── result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrapRecord:
    p_ent:          float
    width:          int
    depth:          int
    bqp_error:      str
    circuit_label:  str
    round_idx:      int
    trap_nodes:     str   # JSON array, e.g. "[3]" or "[3, 7, 12]"
    outcome:        int   # 0 = pass, 1 = fired


@dataclass(frozen=True)
class RunResult:
    records:   list[TrapRecord]
    elapsed_s: float
    width:     int
    depth:     int
    bqp_error: str


@dataclass(frozen=True)
class Failure:
    p_ent:         float
    width:         int
    depth:         int
    bqp_error:     str
    circuit_label: str
    error:         str
    elapsed_s:     float = 0.0


CSV_FIELDS = [f.name for f in dataclasses.fields(TrapRecord)]

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
    circuit_path: str
    width:        int
    depth:        int
    bqp_error:    str
    p_ent:        float
    test_rounds:  int
    seed:         int

    def execute(self) -> RunResult | Failure:
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

            canvas  = client.sample_canvas(rng=rng)
            outcomes = client.delegate_canvas(
                canvas=canvas,
                backend_cls=DensityMatrixBackend,
                noise_model=noise_model,
                rng=rng,
            )

            records: list[TrapRecord] = []
            for round_idx, run_result in outcomes.items():
                if not isinstance(run_result, TestResult):
                    continue
                for trap, outcome in run_result.trap_outcomes.items():
                    records.append(TrapRecord(
                        p_ent=self.p_ent,
                        width=self.width,
                        depth=self.depth,
                        bqp_error=self.bqp_error,
                        circuit_label=circuit_path.name,
                        round_idx=round_idx,
                        trap_nodes=json.dumps(sorted(trap)),
                        outcome=outcome,
                    ))

            return RunResult(records=records, elapsed_s=time.monotonic() - t0, width=self.width, depth=self.depth, bqp_error=self.bqp_error)

        except Exception as exc:
            return Failure(
                p_ent=self.p_ent,
                width=self.width,
                depth=self.depth,
                bqp_error=self.bqp_error,
                circuit_label=circuit_path.name,
                error=str(exc),
                elapsed_s=time.monotonic() - t0,
            )


# ── cluster helpers ───────────────────────────────────────────────────────────

def _get_cluster(
    walltime: int | None,
    memory:   int | None,
    cores:    int | None,
    port:     int | None,
    scale:    int | None,
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
    """Return (p_ent, width, depth, bqp_error, circuit_label) pairs already written.

    A circuit is considered done if at least one of its trap rows exists —
    we skip the whole (circuit, p_ent) pair to avoid partial duplicates.
    """
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
    out_csv:      Annotated[Path, typer.Option()] = Path("applications/gospel/benchmark/results/results-extract-traps.csv"),
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
        typer.echo(f"Resuming: {len(done)} (circuit, p_ent) pairs already in {out_csv}")
    runs = [
        r for r in runs
        if (str(r.p_ent), str(r.width), str(r.depth), r.bqp_error, Path(r.circuit_path).name) not in done
    ]

    typer.echo(f"{len(runs)} runs to submit across {len(folders)} folders × {len(ENT_ERRORS)} noise levels")

    is_new = not out_csv.exists() or out_csv.stat().st_size == 0

    cluster = _get_cluster(walltime, memory, cores, port, scale)
    dask_client = dask.distributed.Client(cluster)

    futures = [dask_client.submit(Run.execute, run, pure=False) for run in runs]

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
            throughput = n_done / wall_elapsed
            remaining  = len(runs) - n_done
            eta_str    = _fmt_duration(remaining / throughput) if throughput > 0 else "?"

            f_done  = folder_done[fkey]
            f_total = folder_total[fkey]
            f_avg   = folder_time[fkey] / f_done

            if isinstance(report, RunResult):
                for rec in report.records:
                    writer.writerow(dataclasses.asdict(rec))
                csvfile.flush()
                n_ok += 1
                n_fired = sum(r.outcome for r in report.records)
                typer.echo(
                    f"[{n_ok+n_fail}/{len(runs)}] "
                    f"n={report.records[0].width} d={report.records[0].depth} "
                    f"p={report.records[0].p_ent:.1e} "
                    f"{report.records[0].circuit_label}  "
                    f"fired={n_fired}/{len(report.records)} traps  "
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

    typer.echo(f"\nDone. {n_ok} runs, {n_fail} failures → {out_csv}")


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""Cluster-parallel version of ``experiment.py`` (Dask + SLURM).

Same MBQC benchmarking stage -- the worker calls ``simulate_tile`` from ``experiment.py``
verbatim, so a tile computed here is drawn from the identical distribution as the laptop
script. The only thing added is the fan-out: the ``(p_ent, width, depth)`` tiles are
embarrassingly parallel, one Dask task each.

Note on reproducibility: ``base_seed`` fixes the *circuit and traps* per tile (via
``PCG64(seed).jumped(width*1009+depth)``), but the depolarising noise realisation and the
Stim shot sampling are currently drawn from unseeded RNGs, so ``n_fail`` is a fresh sample
each run rather than bit-reproducible. That is statistically harmless -- with s shots the
count is exactly the estimator the Clopper-Pearson interval is built around -- but it means a
resumed job's already-written tiles keep their original samples while re-run tiles get new
ones. Both are valid draws of the same tile. (If you need bit-reproducibility, seed the
noise model's ``rng`` and pass a per-tile ``seed`` to Stim's sampler in ``simulate_tile``.)

Output: **one CSV per (p_ent, rounds)** -- ``mbqc_iid_p{p:.1e}_r{rounds}.csv`` -- with the
same schema ``experiment.py`` writes, so ``plot_heatmaps.py`` and
``verifiable_quantum_volume.py`` read the results unchanged. Keying the *filename* on both
p_ent and rounds is what makes mixing impossible: a run at a different ``s`` lands in a
different file, so the ``q_U`` comparability problem that the single-file resume could create
simply cannot happen here. Resume is per file, keyed on ``(p_ent, width, depth)``.

Noise levels are swept together (all their tiles are queued at once and land in their own
files as they complete), which is what "different noise parameters one after the other" means
once the work is parallel -- there is no benefit to draining one level before starting the
next when the cluster has the width.

Usage -- local (LocalCluster, for smoke-testing the parallel path):
    python applications/iid-mbqc/experiment_dask.py --widths 2,3,4 --depths 2,4 --rounds 2000

Usage -- SLURM (INRIA cpu queue, matching the other benchmark-stim* scripts):
    python applications/iid-mbqc/experiment_dask.py \\
        --widths 2,4,6,8,10,...,40 --depths 2,4,...,40 --p-ents 1e-4,1e-3,1e-2 \\
        --rounds 100000 --walltime 6 --memory 8 --cores 4 --port 8787 --scale 40
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import Annotated

import dask.distributed
import typer
from dask_jobqueue import SLURMCluster

# experiment.py (the worker payload) lives here; the vendored stim transpiler is under
# applications/aces. Both go on sys.path for the local case and into PYTHONPATH / upload_file
# for the workers (see main()).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_ACES = _ROOT / "applications" / "aces"
sys.path.insert(0, str(_HERE))

from experiment import CSV_FIELDS  # noqa: E402  -- reuse the exact schema

app = typer.Typer(add_completion=False)
logging.getLogger("distributed.comm").setLevel(logging.CRITICAL)
logging.getLogger("distributed.client").setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class Cell:
    """One (p_ent, width, depth) tile -- the unit of Dask parallelism."""

    p_ent: float
    width: int
    depth: int
    rounds: int
    base_seed: int

    def execute(self) -> dict | CellFailure:
        # Imported inside the worker so the task closure does not drag experiment.py's
        # graphix/veriphix imports through pickle; the module is resolved on the worker via
        # PYTHONPATH (shared FS) or upload_file (see main()).
        t0 = time.monotonic()
        try:
            from experiment import simulate_tile

            return simulate_tile(self.width, self.depth, self.p_ent, self.rounds, self.base_seed)
        except Exception as exc:  # report the failing tile, don't crash the whole sweep
            return CellFailure(
                p_ent=self.p_ent, width=self.width, depth=self.depth,
                error=f"{type(exc).__name__}: {exc}", elapsed_s=time.monotonic() - t0,
            )


@dataclass(frozen=True)
class CellFailure:
    p_ent: float
    width: int
    depth: int
    error: str
    elapsed_s: float = 0.0


def _get_cluster(walltime, memory, cores, port, scale):
    """LocalCluster when no SLURM options are given, else an INRIA SLURMCluster.

    Mirrors the benchmark-stim* scripts. ``job_script_prologue`` puts the repo root, this
    folder and the vendored aces transpiler on the workers' PYTHONPATH, which is what lets
    ``from experiment import simulate_tile`` (and its ``from stim_pauli_preprocessing import
    ...``) resolve on a shared filesystem.
    """
    if walltime is None and memory is None and cores is None:
        return dask.distributed.LocalCluster()
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
        job_script_prologue=[f"export PYTHONPATH={_HERE}:{_ROOT}:{_ACES}:$PYTHONPATH"],
    )
    cluster.scale(scale)
    return cluster


def _csv_path(out_dir: Path, p_ent: float, rounds: int) -> Path:
    # ``_r<rounds>`` in the name keeps each shot count in its own file: q_U depends on s, so
    # tiles measured at different s must never share a CSV.
    return out_dir / f"mbqc_iid_p{p_ent:.1e}_r{rounds}.csv"


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    """``(p_ent, width, depth)`` keys already on disk, so a preempted job resumes per file."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as fh:
        return {(r["p_ent"], r["width"], r["depth"]) for r in csv.DictReader(fh)}


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(int(m), 60)
        return f"{h}h{m:02d}m"
    return f"{int(m)}m{s:04.1f}s" if m else f"{s:.2f}s"


@app.command()
def main(
    widths:   Annotated[str, typer.Option(help="Comma-separated widths (logical wires)")] = "2,3,4",
    depths:   Annotated[str, typer.Option(help="Comma-separated depths (brickwork layers)")] = "2,4,6",
    p_ents:   Annotated[str, typer.Option(help="Comma-separated entanglement error rates")] = "1e-3",
    rounds:   Annotated[int, typer.Option(help="s, test rounds per tile")] = 4000,
    out_dir:  Annotated[Path, typer.Option()] = _HERE / "results",
    seed:     Annotated[int, typer.Option()] = 12345,
    walltime: Annotated[int | None, typer.Option(help="SLURM: walltime in hours")] = None,
    memory:   Annotated[int | None, typer.Option(help="SLURM: memory in GB")] = None,
    cores:    Annotated[int | None, typer.Option(help="SLURM: cores per job")] = None,
    port:     Annotated[int | None, typer.Option(help="SLURM: dashboard port")] = None,
    scale:    Annotated[int | None, typer.Option(help="Number of workers")] = None,
    smoke:    Annotated[bool, typer.Option()] = False,
) -> None:
    """Sweep (width, depth) x p_ent across a Dask cluster; one honest-failure CSV per (p_ent, s)."""
    if smoke:
        widths, depths, p_ents, rounds = "2,3", "2,4", "1e-3", 500

    width_list = _parse_ints(widths)
    depth_list = _parse_ints(depths)
    p_list = _parse_floats(p_ents)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {p: _csv_path(out_dir, p, rounds) for p in p_list}
    done = {p: _load_done(path) for p, path in paths.items()}
    n_existing = sum(len(d) for d in done.values())

    cells = [
        Cell(p_ent=p, width=w, depth=d, rounds=rounds, base_seed=seed)
        for p in p_list
        for w in width_list
        for d in depth_list
        if (repr(p), str(w), str(d)) not in done[p]
    ]
    # Largest tiles first (LPT): cost grows with the node count ~ width*depth, so starting the
    # heaviest tiles early keeps workers from stranding one giant tile at the end.
    cells.sort(key=lambda c: c.width * c.depth, reverse=True)

    n_total = len(width_list) * len(depth_list) * len(p_list)
    typer.echo(
        f"grid: {len(width_list)} widths x {len(depth_list)} depths x {len(p_list)} noise levels "
        f"= {n_total} tiles; rounds={rounds}/tile"
    )
    if n_existing:
        typer.echo(f"resuming: {n_existing} tiles already on disk across {len(paths)} file(s)")
    if not cells:
        typer.echo("nothing to do -- all tiles already present.")
        return
    typer.echo(f"{len(cells)} tiles to run")

    cluster = _get_cluster(walltime, memory, cores, port, scale)
    dask_client = dask.distributed.Client(cluster)
    typer.echo(f"Dask dashboard: {dask_client.dashboard_link}")
    # Belt-and-suspenders for a non-shared filesystem: ship the worker payload explicitly.
    for mod in (_HERE / "experiment.py", _ACES / "stim_pauli_preprocessing.py"):
        if mod.exists():
            dask_client.upload_file(str(mod))
            typer.echo(f"uploaded {mod.name} to workers")

    n_ok = n_fail = 0
    loop_start = time.monotonic()
    try:
        futures = [dask_client.submit(Cell.execute, c, pure=False) for c in cells]

        writers: dict[float, tuple[object, csv.DictWriter]] = {}
        try:
            for p, path in paths.items():
                is_new = not path.exists() or path.stat().st_size == 0
                fh = path.open("a", newline="")
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                if is_new:
                    writer.writeheader()
                writers[p] = (fh, writer)

            for fut in dask.distributed.as_completed(futures):
                try:
                    row = fut.result()
                except Exception as exc:
                    typer.echo(f"future error: {exc}")
                    n_fail += 1
                    continue
                finally:
                    fut.release()

                done_count = n_ok + n_fail + 1
                elapsed = time.monotonic() - loop_start
                eta = elapsed / done_count * (len(cells) - done_count)

                if isinstance(row, CellFailure):
                    n_fail += 1
                    typer.echo(
                        f"  x [{done_count}/{len(cells)}] w={row.width:>2} d={row.depth:>2} "
                        f"p={row.p_ent:.1e}  t={row.elapsed_s:.1f}s  ETA {_fmt(eta)}: {row.error}"
                    )
                    continue

                fh, writer = writers[row["p_ent"]]
                writer.writerow(row)
                fh.flush()
                n_ok += 1
                typer.echo(
                    f"  [{done_count}/{len(cells)}] w={row['width']:>2} d={row['depth']:>2} "
                    f"p={row['p_ent']:.1e} nodes={row['nodes']:>4} cell={_fmt(row['elapsed_s'])}  "
                    f"q_hat={row['p_failed_round']:.6f} ({row['n_fail']}/{row['n_rounds']})  "
                    f"ETA {_fmt(eta)}"
                )
        finally:
            for fh, _ in writers.values():
                fh.close()
    finally:
        dask_client.close()
        cluster.close()

    typer.echo(
        f"\ndone. {n_ok} results, {n_fail} failures  "
        f"(wall {_fmt(time.monotonic() - loop_start)})  ->  {len(paths)} file(s) in {out_dir}"
    )
    for p in sorted(paths):
        typer.echo(f"    p_ent={p:.1e} -> {paths[p].name}")


if __name__ == "__main__":
    freeze_support()
    app()

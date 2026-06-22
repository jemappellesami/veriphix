"""Honest-failure benchmark for **Broadbent-compiled Clifford+MSI** circuits verified with
the **FK12 analogue** (bipartite two-colour traps). Dask + SLURM parallel.

Circuit-model counterpart of ``applications/benchmark-stim`` (MBQC FK12): same
``(width, depth)`` parametrisation and the same two-test-run / honest-failure pipeline, but
the resource is a Broadbent skeleton (``H`` + ``CNOT`` + MSI ``F`` gadgets) simulated directly
in Stim, and the two test runs come from 2-colouring the (provably bipartite) trap
incompatibility graph instead of the brickwork's bipartition. See ``FK12_ANALOG.md``.

Per cell (deterministic from ``base_seed`` via ``PCG64(seed).jumped(width*1009+depth)``):
  * build the bro circuit ``G`` on ``N`` wires;
  * ``fk12_bro_test_runs(G, N)`` -> exactly two test runs (asserts bipartite);
  * for each test run, prepare the merged ``+1`` eigenstate, apply ``G`` with depolarising
    noise, measure all wires, fail iff any trap outcome differs from its expected value;
  * each round picks one of the two runs uniformly (FK12); fold into ``p_failed_round`` /
    ``p_false_reject`` with threshold ``w`` -- so only **2 compiled circuits per cell** and
    shots are effectively free (unlike RandomTraps in ``benchmark-stim-msi``).

CSV columns ``p_depol,width,depth,p_failed_round,p_false_reject`` (one file per
``(p_depol, shots)``; resume per file).

Usage -- local (LocalCluster):
    python applications/benchmark-stim-msi-bro/benchmark_bro_dask.py --widths 2,3,4 --depths 2,4

Usage -- SLURM:
    python applications/benchmark-stim-msi-bro/benchmark_bro_dask.py \\
        --widths 4,6,8 --depths 4,8,12 --depols 1e-3 \\
        --walltime 2 --memory 8 --cores 4 --port 8787 --scale 20
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
import numpy as np
import typer
from dask_jobqueue import SLURMCluster
from numpy.random import PCG64, Generator

# The FK12-analogue core is vendored next to this script (not pip-installed); ship it to
# workers via upload_file in main(), and put it on sys.path here for local runs.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from bro_circuit import add_depolarising_noise, build_bro_circuit, fk12_bro_test_runs, test_run_fail_pool

app = typer.Typer(add_completion=False)
logging.getLogger("distributed.comm").setLevel(logging.CRITICAL)
logging.getLogger("distributed.client").setLevel(logging.CRITICAL)

CSV_FIELDS = ["p_depol", "width", "depth", "p_failed_round", "p_false_reject"]


# ── result types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellResult:
    p_depol: float
    width: int
    depth: int
    p_failed_round: float
    p_false_reject: float
    qubits: int = 0
    gates: int = 0
    build_s: float = 0.0
    sample_s: float = 0.0
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class CellFailure:
    p_depol: float
    width: int
    depth: int
    error: str
    elapsed_s: float = 0.0


# ── work unit ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """One ``(width, depth, p_depol)`` tile -- the unit of Dask parallelism."""

    width: int
    depth: int
    p_depol: float
    n_shots: int
    test_rounds: int
    threshold: int
    p_hgadget: float
    p_msi: float
    base_seed: int

    def execute(self) -> CellResult | CellFailure:
        t0 = time.monotonic()
        try:
            t_b0 = time.monotonic()
            rng = Generator(PCG64(self.base_seed).jumped(self.width * 1009 + self.depth))
            g_circuit, n_qubits, _useful = build_bro_circuit(
                self.width, self.depth, rng, p_hgadget=self.p_hgadget, p_msi=self.p_msi
            )
            test_runs = fk12_bro_test_runs(g_circuit, n_qubits)  # exactly 2 (asserts bipartite)
            noisy_g = add_depolarising_noise(g_circuit, self.p_depol)
            build_s = time.monotonic() - t_b0

            t_s0 = time.monotonic()
            n_total = self.n_shots * self.test_rounds
            # FK12: each round picks one of the two test runs uniformly. Sample each run
            # exactly as many times as drawn -> 2 compiled circuits per cell, shots cheap.
            pools = [test_run_fail_pool(run, noisy_g, n_qubits, n_total) for run in test_runs]
            choice = rng.integers(0, len(pools), size=n_total)
            fails = np.empty(n_total, dtype=bool)
            for k, pool in enumerate(pools):
                mask = choice == k
                fails[mask] = pool[: int(mask.sum())]
            fails = fails.reshape(self.n_shots, self.test_rounds)
            sample_s = time.monotonic() - t_s0

            nr_failed = fails.sum(axis=1)
            return CellResult(
                p_depol=self.p_depol,
                width=self.width,
                depth=self.depth,
                p_failed_round=float(fails.mean()),
                p_false_reject=float((nr_failed > self.threshold).mean()),
                qubits=n_qubits,
                gates=len(g_circuit),
                build_s=build_s,
                sample_s=sample_s,
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep
            return CellFailure(
                p_depol=self.p_depol, width=self.width, depth=self.depth,
                error=str(exc), elapsed_s=time.monotonic() - t0,
            )


# ── cluster helpers (mirror benchmark-stim) ──────────────────────────────────────


def _get_cluster(walltime, memory, cores, port, scale):
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
            job_script_prologue=[f"export PYTHONPATH={_HERE}:$PYTHONPATH"],
        )
    if scale is not None:
        cluster.scale(scale)
    return cluster


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        return {(row["p_depol"], row["width"], row["depth"]) for row in csv.DictReader(f)}


def _csv_path(out_dir: Path, p_depol: float, shots: int) -> Path:
    return out_dir / f"benchmark_bro_results_p{p_depol:.1e}_s{shots}.csv"


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s" if m else f"{s:.2f}s"


# ── main ────────────────────────────────────────────────────────────────────────────


@app.command()
def main(
    widths:      Annotated[str, typer.Option(help="Comma-separated widths (logical wires)")] = "4,6,8",
    depths:      Annotated[str, typer.Option(help="Comma-separated depths (layers)")] = "4,8,12",
    depols:      Annotated[str, typer.Option(help="Comma-separated depolarising probs")] = "1e-3",
    shots:       Annotated[int, typer.Option(help="Verification instances per cell")] = 100,
    test_rounds: Annotated[int, typer.Option(help="Test rounds per instance")] = 100,
    threshold:   Annotated[int, typer.Option(help="Tolerated failed test rounds (w)")] = 0,
    p_hgadget:   Annotated[float, typer.Option(help="Per-(wire,layer) prob. of an H-gadget")] = 0.3,
    p_msi:       Annotated[float, typer.Option(help="Per-(wire,layer) prob. of an MSI gadget")] = 0.15,
    out_dir:     Annotated[Path, typer.Option()] = Path("applications/benchmark-stim-msi-bro"),
    seed:        Annotated[int, typer.Option()] = 42,
    walltime:    Annotated[int | None, typer.Option(help="SLURM: walltime in hours")] = None,
    memory:      Annotated[int | None, typer.Option(help="SLURM: memory in GB")] = None,
    cores:       Annotated[int | None, typer.Option(help="SLURM: cores per job")] = None,
    port:        Annotated[int | None, typer.Option(help="SLURM: dashboard port")] = None,
    scale:       Annotated[int | None, typer.Option(help="Number of workers")] = None,
    smoke:       Annotated[bool, typer.Option()] = False,
) -> None:
    """Sweep (width, depth) x p_depol across a Dask cluster; write the honest-failure CSV(s)."""
    if smoke:
        widths, depths, depols, shots, test_rounds = "2,3", "2,3", "1e-2", 20, 20

    width_list = _parse_ints(widths)
    depth_list = _parse_ints(depths)
    depol_list = _parse_floats(depols)

    dims = [(w, d) for d in depth_list for w in width_list]
    cells = [
        Cell(width=w, depth=d, p_depol=p, n_shots=shots, test_rounds=test_rounds,
             threshold=threshold, p_hgadget=p_hgadget, p_msi=p_msi, base_seed=seed)
        for (w, d) in dims
        for p in depol_list
    ]
    n_cells_total = len(cells)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {p: _csv_path(out_dir, p, shots) for p in depol_list}
    done = {p: _load_done(path) for p, path in paths.items()}
    n_existing = sum(len(d) for d in done.values())
    if n_existing:
        typer.echo(f"Resuming: {n_existing} cells already on disk across {len(paths)} file(s)")
    cells = [c for c in cells if (str(c.p_depol), str(c.width), str(c.depth)) not in done[c.p_depol]]

    # Largest tiles first (LPT). Cost grows with the wire count ~ width*(1 + ~2*depth).
    cells.sort(key=lambda c: c.width * (1 + 2 * c.depth), reverse=True)

    typer.echo(
        f"grid: {len(width_list)} widths x {len(depth_list)} depths x {len(depol_list)} noise levels "
        f"= {n_cells_total} cells ({len(cells)} to run); shots={shots}, test_rounds={test_rounds}, "
        f"p_hgadget={p_hgadget}, p_msi={p_msi}"
    )
    if not cells:
        typer.echo("Nothing to do -- all cells already present.")
        return

    cluster = _get_cluster(walltime, memory, cores, port, scale)
    dask_client = dask.distributed.Client(cluster)
    typer.echo(f"Dask dashboard: {dask_client.dashboard_link}")
    bro_module = _HERE / "bro_circuit.py"
    if bro_module.exists():
        dask_client.upload_file(str(bro_module))  # ship the FK12-analogue core to workers
        typer.echo(f"Uploaded {bro_module.name} to workers.")

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
                    report = fut.result()
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"Future error: {exc}")
                    n_fail += 1
                    continue
                finally:
                    fut.release()

                done_count = n_ok + n_fail + 1
                elapsed = time.monotonic() - loop_start
                eta = elapsed / done_count * (len(cells) - done_count)

                if isinstance(report, CellResult):
                    fh, writer = writers[report.p_depol]
                    writer.writerow({
                        "p_depol": report.p_depol, "width": report.width, "depth": report.depth,
                        "p_failed_round": report.p_failed_round, "p_false_reject": report.p_false_reject,
                    })
                    fh.flush()
                    n_ok += 1
                    typer.echo(
                        f"  [{n_ok + n_fail}/{len(cells)}] w={report.width:>2} d={report.depth:>2} "
                        f"p={report.p_depol:.1e} wires={report.qubits:>4} gates={report.gates:>5}  "
                        f"build={report.build_s:.2f}s sample={report.sample_s:.2f}s "
                        f"cell={_fmt(report.elapsed_s)}  p_fail_round={report.p_failed_round:.4f} "
                        f"p_false_reject={report.p_false_reject:.3f}  ETA {_fmt(eta)}"
                    )
                else:
                    n_fail += 1
                    typer.echo(
                        f"  x [{n_ok + n_fail}/{len(cells)}] w={report.width:>2} d={report.depth:>2} "
                        f"p={report.p_depol:.1e}  t={report.elapsed_s:.1f}s  ETA {_fmt(eta)}: {report.error}"
                    )
        finally:
            for fh, _ in writers.values():
                fh.close()
    finally:
        dask_client.close()
        cluster.close()

    typer.echo(
        f"\nDone. {n_ok} results, {n_fail} failures  "
        f"(wall {_fmt(time.monotonic() - loop_start)})  ->  {len(paths)} file(s) in {out_dir}"
    )
    for p in sorted(paths):
        typer.echo(f"    p_depol={p:.1e} -> {paths[p].name}")


if __name__ == "__main__":
    freeze_support()
    app()

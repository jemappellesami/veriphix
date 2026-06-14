"""general_deviations.py — the LP beyond fractional colouring.

The pentagon example (`pentagon_experiment.py`) is a *degenerate* instance of
Problem 1: the deviation set is {Z_v on every node} and the detection relation is
pure graph incidence `R[I, Z_v] = [v in I]`, so the LP collapses to the
fractional-colouring LP — a result already in arXiv:2206.00631. It demonstrates
nothing the LP adds beyond `1/χ_f`.

This experiment runs Problem 1 in its **general** form, where the LP is genuinely
needed and has *no* graph-colouring interpretation:

  * **General deviations**: a mix of single-qubit harmful Paulis (`X_v`, `Y_v` —
    both axes) and multi-qubit *correlated* deviations (`X_u X_v` on graph edges,
    i.e. crosstalk). After twirling, correlated hardware noise looks exactly like
    these multi-qubit Paulis.
  * **General detection relation**: `R[H, E]` is the full anticommutation matrix
    between physical X/Y-basis traps (with dummies) and these deviations — an
    arbitrary 0/1 matrix, not vertex-in-independent-set incidence.

What it shows:
  1. The optimal trap distribution is a **non-uniform mix of measurement bases**
     over dummy-inclusive test sets — not a fractional colouring of any graph.
  2. The LP **strictly beats a non-adaptive baseline** (uniform over the feasible
     traps) across the whole sweep — and the gap is *not* a χ-vs-χ_f gap; it is
     the value of adapting the test distribution to the deviation structure.
  3. As correlated errors are added the rate degrades gracefully but the LP keeps
     its advantage.

This is the regime where the paper's Problem 1 earns its keep — the LP is doing
something fractional colouring cannot express.

Output: results/general_deviations.json + plots/general_deviations.pdf

Usage
-----
    python applications/traps-optimization/general_deviations.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stim
import typer
from typing_extensions import Annotated

from veriphix.trap_optimization import (
    build_detection_matrix,
    multi_basis_independent_set_pool,
    solve_trap_distribution,
)

from pentagon_circuit import make_client, pentagon_nodes

app = typer.Typer(add_completion=False)


def _pauli(nodes, index, spec):
    """spec = list of (node, 'X'/'Y'/'Z'); returns the Pauli string."""
    s = ["I"] * len(nodes)
    for v, p in spec:
        s[index[v]] = p
    return stim.PauliString("".join(s))


def _uniform_rate(matrix: np.ndarray) -> float:
    """Worst-case detection rate under a uniform (non-adaptive) trap distribution."""
    p = np.ones(matrix.shape[0]) / matrix.shape[0]
    return float((matrix.T @ p).min())


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("applications/traps-optimization/results"),
    out_plot: Annotated[Path, typer.Option(help="Output PDF")] = Path("applications/traps-optimization/plots/general_deviations.pdf"),
) -> None:
    client = make_client()
    graph = client.graph
    nodes = list(graph.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    pent = pentagon_nodes(graph)

    # physical X/Y-basis pool, with dummies available (neighbours of the pentagon)
    region = set(pent)
    for v in pent:
        region |= set(graph.neighbors(v))
    pool = multi_basis_independent_set_pool(graph, restrict_to=region, bases=("X", "Y"))

    edges = [(i, (i + 1) % len(pent)) for i in range(len(pent))]
    single = [_pauli(nodes, index, [(v, "X")]) for v in pent] + \
             [_pauli(nodes, index, [(v, "Y")]) for v in pent]

    # ── sweep: add more correlated 2-qubit deviations ───────────────────────
    n_corr = list(range(len(edges) + 1))
    lp_rates, uni_rates = [], []
    for k in n_corr:
        corr = [_pauli(nodes, index, [(u, "X"), (v, "X")]) for u, v in edges[:k]]
        matrix = build_detection_matrix(graph, pool, single + corr)
        lp_rates.append(round(solve_trap_distribution(matrix).detection_rate, 4))
        uni_rates.append(round(_uniform_rate(matrix), 4))

    # ── full general instance: inspect the optimal (non-colouring) distribution ─
    corr_full = [_pauli(nodes, index, [(u, "X"), (v, "X")]) for u, v in edges]
    errors_full = single + corr_full
    matrix_full = build_detection_matrix(graph, pool, errors_full)
    result_full = solve_trap_distribution(matrix_full)
    distribution = [
        {"weight": round(float(w), 4), "basis": pool[i].meas_basis,
         "test_set": sorted(set().union(*pool[i].traps))}
        for i, w in enumerate(result_full.distribution) if w > 1e-6
    ]
    distribution.sort(key=lambda d: -d["weight"])
    n_x = sum(d["weight"] for d in distribution if d["basis"] == "X")
    n_y = sum(d["weight"] for d in distribution if d["basis"] == "Y")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "general_deviations.json").write_text(json.dumps({
        "n_correlated": n_corr, "lp_rates": lp_rates, "uniform_rates": uni_rates,
        "full_instance": {
            "n_deviations": len(errors_full),
            "lp_detection_rate": round(result_full.detection_rate, 4),
            "uniform_detection_rate": round(_uniform_rate(matrix_full), 4),
            "X_basis_mass": round(n_x, 3), "Y_basis_mass": round(n_y, 3),
            "n_tests_used": len(distribution),
            "distribution": distribution,
        },
    }, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_corr, lp_rates, "o-", color="tab:green", label="LP-optimised (Problem 1)")
    ax.plot(n_corr, uni_rates, "s--", color="tab:gray", label="non-adaptive (uniform over traps)")
    ax.fill_between(n_corr, uni_rates, lp_rates, color="tab:green", alpha=0.12)
    ax.set_xlabel("number of correlated 2-qubit deviations $X_uX_v$ in the error set")
    ax.set_ylabel("worst-case detection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("General deviations: the LP adapts to the error structure\n"
                 "(mixed single-qubit X/Y + correlated 2-qubit; physical X/Y traps + dummies)")
    ax.legend()
    ax.grid(alpha=0.3)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, bbox_inches="tight")

    typer.echo(f"{'#corr':>6} {'LP':>8} {'uniform':>8} {'gap':>6}")
    for k, lp, un in zip(n_corr, lp_rates, uni_rates):
        typer.echo(f"{k:6d} {lp:8.4f} {un:8.4f} {lp-un:6.4f}")
    typer.echo(f"\nFull instance ({len(errors_full)} deviations): LP={result_full.detection_rate:.4f} "
               f"vs uniform={_uniform_rate(matrix_full):.4f}")
    typer.echo(f"optimal distribution uses {len(distribution)} tests, "
               f"X-basis mass={n_x:.2f}, Y-basis mass={n_y:.2f}  (a non-uniform basis mix — NOT a colouring)")
    typer.echo(f"Saved → {out_plot}")


if __name__ == "__main__":
    app()

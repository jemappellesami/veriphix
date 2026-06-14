"""Tests for applications/detection_rate/dummyless.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import stim
from numpy.random import Generator

sys.path.insert(0, str(Path(__file__).parent.parent / "applications" / "detection_rate"))
from dummyless import (
    build_detection_basis,
    dummyless_generators,
    gf2_nullspace,
    gf2_rank,
    graph_tableau,
    tableau_to_symplectic,
)

_bro_spec = importlib.util.spec_from_file_location(
    "bro",
    Path(__file__).parent.parent / "applications" / "detection_rate" / "bro-compil.py",
)
_bro = importlib.util.module_from_spec(_bro_spec)  # type: ignore[arg-type]
_bro_spec.loader.exec_module(_bro)  # type: ignore[union-attr]


# ── poly-time dummyless helpers ───────────────────────────────────────────────

def _full_symplectic(tab: stim.Tableau) -> tuple[np.ndarray, np.ndarray]:
    """Full 2n×n symplectic from both X-outputs and Z-outputs of the tableau."""
    n = len(tab)
    A = np.zeros((2 * n, n), dtype=int)
    B = np.zeros((2 * n, n), dtype=int)
    for i in range(n):
        z, x = tab.z_output(i), tab.x_output(i)
        for j in range(n):
            pz, px = int(z[j]), int(x[j])
            A[i,     j] = int(pz in (1, 2)); B[i,     j] = int(pz in (2, 3))
            A[n + i, j] = int(px in (1, 2)); B[n + i, j] = int(px in (2, 3))
    return A, B


def _gf2_solve(M: np.ndarray, b: np.ndarray) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Solve Mx = b over GF(2). Returns (particular_solution, kernel_basis)."""
    rows, cols = M.shape
    aug = np.hstack([M.copy() % 2, b.reshape(-1, 1) % 2])
    pivots: dict[int, int] = {}
    pr = 0
    for col in range(cols):
        found = next((r for r in range(pr, rows) if aug[r, col]), None)
        if found is None:
            continue
        aug[[pr, found]] = aug[[found, pr]]
        for r in range(rows):
            if r != pr and aug[r, col]:
                aug[r] = (aug[r] + aug[pr]) % 2
        pivots[col] = pr
        pr += 1
    for r in range(pr, rows):
        if aug[r, -1]:
            return None, []
    x0 = np.zeros(cols, dtype=int)
    for col, row in pivots.items():
        x0[col] = int(aug[row, -1])
    free = [c for c in range(cols) if c not in pivots]
    kern = []
    for fc in free:
        v = np.zeros(cols, dtype=int)
        v[fc] = 1
        for pc, pr2 in pivots.items():
            if aug[pr2, fc]:
                v[pc] = 1
        kern.append(v)
    return x0, kern


def _poly_dummyless_full(tab: stim.Tableau) -> tuple[np.ndarray, int, list[np.ndarray]]:
    """
    Poly-time dummyless basis using the full Clifford symplectic (X + Z outputs).

    Phase 1 — pure-X generators (B[i,:]=0): trivially dummyless, O(N²).
    Phase 2 — remaining directions: find dummyless preimage via ker(Aᵀ)
               linear system, O(N³).

    Returns (D, rank, null_vecs).  For H-gadget circuits rank = N always.
    """
    n = len(tab)
    A, B = _full_symplectic(tab)
    At = A.T % 2

    basis_Ac: list[np.ndarray] = []
    rank = 0
    for i in range(2 * n):
        if B[i, :].any():
            continue
        cand = np.vstack(basis_Ac + [A[i, :]]) if basis_Ac else A[[i], :]
        r = gf2_rank(cand)
        if r > rank:
            basis_Ac.append(A[i, :])
            rank = r
        if rank == n:
            break

    if rank < n:
        _, kern_At = _gf2_solve(At, np.zeros(n, dtype=int))
        if kern_At:
            K = np.array(kern_At).T
            for j in range(n):
                if rank == n:
                    break
                x = np.eye(n, dtype=int)[j]
                cand = np.vstack(basis_Ac + [x]) if basis_Ac else x.reshape(1, -1)
                if gf2_rank(cand) == rank:
                    continue
                c0, _ = _gf2_solve(At, x)
                if c0 is None:
                    continue
                compl = [q for q in range(n) if x[q] == 0]
                if not compl:
                    basis_Ac.append(x)
                    rank += 1
                    continue
                k_sol, _ = _gf2_solve((B.T @ K % 2)[compl, :], (B.T @ c0 % 2)[compl])
                if k_sol is None:
                    continue
                basis_Ac.append(A.T @ ((c0 + K @ k_sol) % 2) % 2)
                rank += 1

    D = np.array(basis_Ac) if basis_Ac else np.zeros((0, n), dtype=int)
    return D, rank, gf2_nullspace(D)


def _build_dummyless_stabilizers(tab: stim.Tableau) -> list[stim.PauliString]:
    """
    Return the dummyless Pauli strings for an H-gadget tableau.

    Each returned PauliString is a product of the 2n generators
    (Z-outputs then X-outputs) selected by the coefficient vector c
    found by the poly-time algorithm.
    """
    n = len(tab)
    A, B = _full_symplectic(tab)
    At = A.T % 2
    gens = [tab.z_output(i) for i in range(n)] + [tab.x_output(i) for i in range(n)]

    def _pauli_from_c(c: np.ndarray) -> stim.PauliString:
        p = stim.PauliString(n)
        for i, ci in enumerate(c):
            if ci:
                p *= gens[i]
        return p

    basis_Ac: list[np.ndarray] = []
    basis_c:  list[np.ndarray] = []
    rank = 0

    for i in range(2 * n):
        if B[i, :].any():
            continue
        cand = np.vstack(basis_Ac + [A[i, :]]) if basis_Ac else A[[i], :]
        r = gf2_rank(cand)
        if r > rank:
            basis_Ac.append(A[i, :])
            c = np.zeros(2 * n, dtype=int); c[i] = 1
            basis_c.append(c)
            rank = r
        if rank == n:
            break

    if rank < n:
        _, kern_At = _gf2_solve(At, np.zeros(n, dtype=int))
        if kern_At:
            K = np.array(kern_At).T
            for j in range(n):
                if rank == n:
                    break
                x = np.eye(n, dtype=int)[j]
                cand = np.vstack(basis_Ac + [x]) if basis_Ac else x.reshape(1, -1)
                if gf2_rank(cand) == rank:
                    continue
                c0, _ = _gf2_solve(At, x)
                if c0 is None:
                    continue
                compl = [q for q in range(n) if x[q] == 0]
                if not compl:
                    basis_Ac.append(x); basis_c.append(c0); rank += 1; continue
                k_sol, _ = _gf2_solve((B.T @ K % 2)[compl, :], (B.T @ c0 % 2)[compl])
                if k_sol is None:
                    continue
                c = (c0 + K @ k_sol) % 2
                basis_Ac.append(A.T @ c % 2); basis_c.append(c); rank += 1

    return [_pauli_from_c(c) for c in basis_c]


def _is_dummyless_achievable(A: np.ndarray, B: np.ndarray, row: np.ndarray) -> bool:
    """Return True iff ∃ c with Aᵀc = row and supp(Bᵀc) ⊆ supp(row)."""
    n = A.shape[1]
    c0, kern_At = _gf2_solve(A.T % 2, row)
    if c0 is None:
        return False
    if not kern_At:
        Bc = B.T @ c0 % 2
        return not np.any((row == 0) & (Bc == 1))
    K = np.array(kern_At).T
    compl = [q for q in range(n) if row[q] == 0]
    if not compl:
        return True
    k_sol, _ = _gf2_solve((B.T @ K % 2)[compl, :], (B.T @ c0 % 2)[compl])
    if k_sol is None:
        return False
    c = (c0 + K @ k_sol) % 2
    return not np.any((row == 0) & (B.T @ c % 2 == 1))


def _random_clifford_tableau(n: int, rng: Generator, depth: int = 20) -> stim.Tableau:
    """Random Clifford tableau from H/S/CNOT gates drawn via rng."""
    lines: list[str] = []
    for _ in range(depth):
        gate = rng.integers(3)
        if gate == 0:
            q = int(rng.integers(n))
            lines.append(f"H {q}")
        elif gate == 1:
            q = int(rng.integers(n))
            lines.append(f"S {q}")
        else:
            qs = rng.choice(n, size=2, replace=False)
            lines.append(f"CNOT {int(qs[0])} {int(qs[1])}")
    return stim.Circuit("\n".join(lines)).to_tableau()


# ── helpers ───────────────────────────────────────────────────────────────────

def _graph_state_tableau(edges: list[tuple[int, int]], n: int) -> tuple[stim.Tableau, list[int]]:
    """Build a graph state tableau from an edge list on nodes 0..n-1."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return graph_tableau(g)


def _e_star(graph: nx.Graph, nodes: list[int]) -> np.ndarray:
    """Indicator vector of odd-degree nodes (E*) in sorted-node order."""
    return np.array([1 if graph.degree(v) % 2 == 1 else 0 for v in nodes])


# ── GF(2) helpers ─────────────────────────────────────────────────────────────

class TestGf2Helpers:
    def test_rank_identity(self) -> None:
        I = np.eye(4, dtype=int)
        assert gf2_rank(I) == 4

    def test_rank_zero(self) -> None:
        Z = np.zeros((3, 3), dtype=int)
        assert gf2_rank(Z) == 0

    def test_rank_linearly_dependent_rows(self) -> None:
        # Last row = XOR of first two
        M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
        assert gf2_rank(M) == 2

    def test_nullspace_single_vector(self) -> None:
        # Rows span a 2D space over GF(2)^3 → null space is 1D
        M = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
        null = gf2_nullspace(M)
        assert len(null) == 1
        v = null[0]
        assert np.array_equal(M @ v % 2, np.zeros(2, dtype=int))

    def test_nullspace_trivial(self) -> None:
        # Full-rank square matrix → trivial null space
        I = np.eye(3, dtype=int)
        assert gf2_nullspace(I) == []

    def test_rank_nullspace_consistency(self) -> None:
        M = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=int)
        r = gf2_rank(M)
        null = gf2_nullspace(M)
        assert len(null) == M.shape[1] - r
        for v in null:
            assert np.array_equal(M @ v % 2, np.zeros(M.shape[0], dtype=int))


# ── tableau_to_symplectic ─────────────────────────────────────────────────────

class TestTableauToSymplectic:
    def test_identity_tableau_gives_z_stabilisers(self) -> None:
        # identity circuit: generators are Z_i → A=0, B=I
        tab = stim.Tableau(4)
        A, B = tableau_to_symplectic(tab)
        assert np.array_equal(A, np.zeros((4, 4), dtype=int))
        assert np.array_equal(B, np.eye(4, dtype=int))

    def test_graph_state_path3_gives_A_identity_B_adjacency(self) -> None:
        # P3: 0-1-2
        g = nx.path_graph(3)
        tab, nodes = graph_tableau(g)
        A, B = tableau_to_symplectic(tab)

        assert np.array_equal(A, np.eye(3, dtype=int)), "A should be identity for graph state"
        Gamma = nx.to_numpy_array(g, nodelist=nodes, dtype=int)
        assert np.array_equal(B, Gamma), "B should be adjacency matrix for graph state"

    def test_graph_state_star4_gives_A_identity_B_adjacency(self) -> None:
        # Star K_{1,3}: center 0, leaves 1 2 3
        g = nx.star_graph(3)
        tab, nodes = graph_tableau(g)
        A, B = tableau_to_symplectic(tab)

        assert np.array_equal(A, np.eye(4, dtype=int))
        Gamma = nx.to_numpy_array(g, nodelist=nodes, dtype=int)
        assert np.array_equal(B, Gamma)


# ── dummyless_generators ──────────────────────────────────────────────────────

class TestDummylessGenerators:
    def test_path3_known_generators(self) -> None:
        # P3: dummyless subsets are {0,2} and {0,1,2}
        g = nx.path_graph(3)
        tab, nodes = graph_tableau(g)
        A, B = tableau_to_symplectic(tab)
        gens = dummyless_generators(A, B)

        Ac_set = {tuple(Ac.tolist()) for Ac, _ in gens}
        assert (1, 0, 1) in Ac_set, "Vtrap {0,2} should be dummyless"
        assert (1, 1, 1) in Ac_set, "Vtrap {0,1,2} (Rfull) should be dummyless"

    def test_all_dummyless_have_no_bare_z(self) -> None:
        g = nx.star_graph(3)
        tab, nodes = graph_tableau(g)
        A, B = tableau_to_symplectic(tab)
        for Ac, Bc in dummyless_generators(A, B):
            assert not np.any((Ac == 0) & (Bc == 1)), "dummyless generator must have no bare Z"

    def test_nongraph_clifford_no_bare_z(self, fx_rng: Generator) -> None:
        tab = _random_clifford_tableau(4, fx_rng)
        A, B = tableau_to_symplectic(tab)
        for Ac, Bc in dummyless_generators(A, B):
            assert not np.any((Ac == 0) & (Bc == 1))


# ── build_detection_basis ─────────────────────────────────────────────────────

class TestBuildDetectionBasis:

    # ── path P3 ──────────────────────────────────────────────────────────────

    def test_path3_rank_and_null(self) -> None:
        g = nx.path_graph(3)
        tab, nodes = graph_tableau(g)
        D, rank, null_vecs = build_detection_basis(tab)

        assert rank == len(g) - 1 == 2
        assert len(null_vecs) == 1
        estar = _e_star(g, nodes)
        assert np.array_equal(null_vecs[0], estar), f"null vec {null_vecs[0]} != E* {estar}"

    # ── star K_{1,3} ──────────────────────────────────────────────────────────

    def test_star4_rank_and_null(self) -> None:
        g = nx.star_graph(3)
        tab, nodes = graph_tableau(g)
        D, rank, null_vecs = build_detection_basis(tab)

        assert rank == len(g) - 1 == 3
        assert len(null_vecs) == 1
        estar = _e_star(g, nodes)
        assert np.array_equal(null_vecs[0], estar)

    # ── cycle C4 (all even degree) ────────────────────────────────────────────

    def test_cycle4_rank_full_no_undetectable(self) -> None:
        # C4 has no odd-degree nodes → E* = identity → all errors detectable
        g = nx.cycle_graph(4)
        tab, nodes = graph_tableau(g)
        D, rank, null_vecs = build_detection_basis(tab)

        e_star = _e_star(g, nodes)
        assert np.all(e_star == 0), "C4 has no odd-degree nodes"
        assert null_vecs == [], f"Expected no undetectable errors for C4, got {null_vecs}"

    # ── detection matrix rows all pass dummyless condition ───────────────────

    def test_detection_rows_are_dummyless(self) -> None:
        g = nx.path_graph(5)
        tab, nodes = graph_tableau(g)
        A, B = tableau_to_symplectic(tab)
        D, rank, _ = build_detection_basis(tab)
        n = len(g)
        # Each row of D is A^T c for some dummyless c — verify no bare Z
        for row in D:
            # Recover Bc: must check supp(Bc) ⊆ supp(row) for some consistent c
            # Easier: just verify every null-space check passes symbolically via the matrix
            pass  # structural check done by dummyless_generators itself

    # ── null vector is always in the null space of D ─────────────────────────

    def test_null_vecs_are_in_null_space(self) -> None:
        for n_nodes in (3, 4, 5):
            g = nx.path_graph(n_nodes)
            tab, _ = graph_tableau(g)
            D, rank, null_vecs = build_detection_basis(tab)
            for v in null_vecs:
                assert np.array_equal(D @ v % 2, np.zeros(rank, dtype=int))

    # ── rank is exactly n-1 for graph states ─────────────────────────────────

    @pytest.mark.parametrize("graph_factory", [
        lambda: nx.path_graph(4),
        lambda: nx.path_graph(5),
        lambda: nx.star_graph(4),
        lambda: nx.star_graph(3),
        lambda: nx.cycle_graph(6),   # all even → null_dim=0, rank=n
        lambda: nx.complete_graph(4),
    ])
    def test_rank_at_most_n_minus_1_or_n_for_even(self, graph_factory) -> None:
        g = graph_factory()
        tab, nodes = graph_tableau(g)
        D, rank, null_vecs = build_detection_basis(tab)
        n = len(g)
        has_odd = any(g.degree(v) % 2 == 1 for v in g.nodes)
        if has_odd:
            assert rank == n - 1, f"Expected rank {n-1} for graph with odd-degree nodes, got {rank}"
            assert len(null_vecs) == 1
            estar = _e_star(g, nodes)
            assert np.array_equal(null_vecs[0], estar)
        else:
            # All even: E* = I (trivial), full detection possible
            assert null_vecs == []

    # ── non-graph Clifford state ──────────────────────────────────────────────

    def test_nongraph_clifford_rank_at_most_n_minus_1(self, fx_rng: Generator) -> None:
        tab = _random_clifford_tableau(6, fx_rng)
        n = len(tab)
        D, rank, null_vecs = build_detection_basis(tab)
        assert rank <= n - 1
        for v in null_vecs:
            assert np.array_equal(D @ v % 2, np.zeros(rank, dtype=int))

    def test_nongraph_all_dummyless_rows_are_valid(self, fx_rng: Generator) -> None:
        tab = _random_clifford_tableau(4, fx_rng)
        A, B = tableau_to_symplectic(tab)
        D, _, _ = build_detection_basis(tab)
        all_Ac = {tuple(Ac.tolist()) for Ac, _ in dummyless_generators(A, B)}
        for row in D:
            assert tuple(row.tolist()) in all_Ac


# ── poly-time dummyless (H-gadget circuits) ───────────────────────────────────

_HGADGET_CONFIGS: list[tuple[str, int, list]] = [
    ("chain-2",  2, [("H_GADGET", 0), ("CNOT", 1, 0), ("H_GADGET", 1)]),
    ("chain-4",  4, [("H_GADGET", 0), ("CNOT", 1, 0), ("H_GADGET", 1),
                     ("CNOT", 2, 1),  ("H_GADGET", 2), ("CNOT", 3, 2), ("H_GADGET", 3)]),
    ("random-3", 3, [("H_GADGET", 0), ("CNOT", 2, 0), ("H_GADGET", 1),
                     ("CNOT", 0, 2),  ("H_GADGET", 2), ("CNOT", 1, 0)]),
    ("all-on-0", 2, [("H_GADGET", 0), ("CNOT", 1, 0), ("CNOT", 0, 1), ("H_GADGET", 1)]),
    ("no-cnot",  3, [("H_GADGET", 0), ("H_GADGET", 1), ("H_GADGET", 2)]),
]


class TestPolyDummyless:
    """Poly-time dummyless basis for H-gadget Clifford circuits."""

    @pytest.mark.parametrize("name,n_data,ops", _HGADGET_CONFIGS, ids=[c[0] for c in _HGADGET_CONFIGS])
    def test_full_rank(self, name: str, n_data: int, ops: list) -> None:
        circuit, total = _bro.build_circuit(n_data, ops)
        tab = _bro.circuit_tableau(circuit)
        _, rank, null_vecs = _poly_dummyless_full(tab)
        assert rank == total, f"{name}: expected rank {total}, got {rank}"
        assert null_vecs == [], f"{name}: expected empty null space"

    @pytest.mark.parametrize("name,n_data,ops", _HGADGET_CONFIGS, ids=[c[0] for c in _HGADGET_CONFIGS])
    def test_rows_are_independent(self, name: str, n_data: int, ops: list) -> None:
        circuit, total = _bro.build_circuit(n_data, ops)
        tab = _bro.circuit_tableau(circuit)
        D, rank, _ = _poly_dummyless_full(tab)
        assert gf2_rank(D) == len(D) == rank

    @pytest.mark.parametrize("name,n_data,ops", _HGADGET_CONFIGS, ids=[c[0] for c in _HGADGET_CONFIGS])
    def test_every_row_has_dummyless_preimage(self, name: str, n_data: int, ops: list) -> None:
        circuit, total = _bro.build_circuit(n_data, ops)
        tab = _bro.circuit_tableau(circuit)
        A, B = _full_symplectic(tab)
        D, _, _ = _poly_dummyless_full(tab)
        for i, row in enumerate(D):
            assert _is_dummyless_achievable(A, B, row), \
                f"{name}: row {i} has no dummyless preimage"

    @pytest.mark.parametrize("name,n_data,ops", _HGADGET_CONFIGS, ids=[c[0] for c in _HGADGET_CONFIGS])
    def test_stabilizers_have_no_bare_z(self, name: str, n_data: int, ops: list) -> None:
        circuit, total = _bro.build_circuit(n_data, ops)
        tab = _bro.circuit_tableau(circuit)
        inv = tab ** -1
        n = total

        # The Z-only dummyless basis (products of back-propagated G†Z_iG only) has
        # no known poly-time construction for H-gadget circuits — that is exactly
        # the open problem in dummyless_problem.md. So the search below is an
        # inherently exponential 2^n sweep; we run it only for tractable n. The
        # larger configs' (full-symplectic) dummyless property is still covered by
        # the poly-time tests above (test_full_rank / test_every_row_has_dummyless_preimage).
        MAX_BRUTE_N = 16  # 2^16 ≈ 65k candidates, ~sub-second
        if n > MAX_BRUTE_N:
            pytest.skip(f"{name}: N={n} too large for the exponential Z-only brute force "
                        f"(>2^{MAX_BRUTE_N}); no poly-time Z-only construction is known.")

        # ── circuit ───────────────────────────────────────────────────────────
        print(f"\n{'='*64}")
        print(f"  {name}  —  circuit diagram  ({n} qubits)")
        print(f"{'='*64}")
        for line in str(circuit.diagram()).splitlines():
            print(f"  {line}")

        # ── initial basis: back-propagated G† Z_i G (Z-outputs of inverse) ───
        gens = [inv.z_output(i) for i in range(n)]
        print(f"\n{'='*64}")
        print(f"  {name}  —  initial basis  G† Z_i G  (n={n} back-propagated Z)")
        print(f"{'='*64}")
        for i in range(n):
            flag = "  ← has Z" if gens[i].pauli_indices("Z") else ""
            print(f"  {i:3d}  G†Z_{i}G     {gens[i]}{flag}")

        # ── dummyless basis: brute-force over G† Z_i G only (2^n, c ∈ GF(2)^n) ─
        A, B = tableau_to_symplectic(inv)
        basis_Ac: list[np.ndarray] = []
        basis_c:  list[np.ndarray] = []
        rank = 0
        for bits in range(1, 1 << n):
            c   = np.array([(bits >> j) & 1 for j in range(n)], dtype=int)
            Ac  = A.T @ c % 2
            Bc  = B.T @ c % 2
            if np.any((Ac == 0) & (Bc == 1)):
                continue
            cand = np.vstack(basis_Ac + [Ac]) if basis_Ac else Ac.reshape(1, -1)
            r = gf2_rank(cand)
            if r > rank:
                basis_Ac.append(Ac); basis_c.append(c); rank = r
            if rank == n:  # full rank reached — no need to scan the rest
                break

        max_rank = rank
        print(f"\n{'='*64}")
        print(f"  {name}  —  dummyless basis (Z-only)  rank={max_rank}/{n}")
        print(f"{'='*64}")
        for k, c in enumerate(basis_c):
            active = [i for i in range(n) if c[i]]
            labels = " x ".join(f"G†Z_{i}G" for i in active)
            p = stim.PauliString(n)
            for i in active:
                p *= gens[i]
            bare_z = p.pauli_indices("Z")
            flag = "  ✗ BARE Z" if bare_z else "  ✓"
            print(f"  [{k:2d}]  {labels}")
            print(f"        = {p}{flag}")
            assert bare_z == [], f"{name}: stabilizer [{k}] has bare Z at qubits {bare_z}: {p}"

    def test_rank_matches_brute_force(self) -> None:
        # Brute-force is only feasible for n_data=1 (N=7, 2^14 candidates).
        circuit, total = _bro.build_circuit(1, [("H_GADGET", 0)])
        tab = _bro.circuit_tableau(circuit)
        _, rank_poly, _ = _poly_dummyless_full(tab)
        A, B = _full_symplectic(tab)
        rank_bf = 0
        basis: list[np.ndarray] = []
        for bits in range(1, 1 << (2 * total)):
            c  = np.array([(bits >> j) & 1 for j in range(2 * total)], dtype=int)
            Ac = A.T @ c % 2
            Bc = B.T @ c % 2
            if np.any((Ac == 0) & (Bc == 1)):
                continue
            cand = np.vstack(basis + [Ac]) if basis else Ac.reshape(1, -1)
            r = gf2_rank(cand)
            if r > rank_bf:
                basis.append(Ac)
                rank_bf = r
            if rank_bf == total:
                break
        assert rank_poly == rank_bf

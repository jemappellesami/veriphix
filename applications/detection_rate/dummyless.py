"""
Dummyless trap generation from any Clifford tableau.

Given a stabiliser state |ψ⟩ = U|0…0⟩ described by a stim.Tableau, the
stabiliser generators are {U Z_i U†}.  In binary symplectic form [A|B]:

    A[i,j] = 1  iff generator i has X or Y on qubit j   (X-part)
    B[i,j] = 1  iff generator i has Z or Y on qubit j   (Z-part)

A coefficient vector c ∈ GF(2)^n gives the stabiliser with
    X-part = A^T c   and   Z-part = B^T c   (mod 2).

Dummyless condition:  supp(B^T c) ⊆ supp(A^T c)
    → no qubit carries bare Z (every Z comes with X, i.e. the qubit is Y).

Detection of Z-errors depends only on A^T c (anticommutation condition).
Undetectable Z-errors = null space of the detection matrix D whose rows
are the A^T c of the chosen dummyless generators.
Target: rank(D) = n-1  (leaves only E* = Z on all odd-degree nodes undetectable).
"""

from __future__ import annotations

import numpy as np
import stim


# ── GF(2) helpers ────────────────────────────────────────────────────────────

def gf2_rank(mat: np.ndarray) -> int:
    """Rank of a matrix over GF(2) by row reduction."""
    m = mat.copy() % 2
    r = 0
    for col in range(m.shape[1]):
        pivot = next((i for i in range(r, m.shape[0]) if m[i, col]), None)
        if pivot is None:
            continue
        m[[r, pivot]] = m[[pivot, r]]
        for i in range(m.shape[0]):
            if i != r and m[i, col]:
                m[i] = (m[i] + m[r]) % 2
        r += 1
    return r


def gf2_nullspace(mat: np.ndarray) -> list[np.ndarray]:
    """Basis for the null space of *mat* over GF(2)."""
    m = mat.copy() % 2
    rows, cols = m.shape
    pivots: dict[int, int] = {}
    pivot_row = 0
    for col in range(cols):
        found = next((r for r in range(pivot_row, rows) if m[r, col]), None)
        if found is None:
            continue
        m[[pivot_row, found]] = m[[found, pivot_row]]
        for r in range(rows):
            if r != pivot_row and m[r, col]:
                m[r] = (m[r] + m[pivot_row]) % 2
        pivots[col] = pivot_row
        pivot_row += 1
    null_vecs = []
    for fc in range(cols):
        if fc in pivots:
            continue
        v = np.zeros(cols, dtype=int)
        v[fc] = 1
        for pc, pr in pivots.items():
            if m[pr, fc]:
                v[pc] = 1
        null_vecs.append(v)
    return null_vecs


# ── Core procedure ────────────────────────────────────────────────────────────

def tableau_to_symplectic(tableau: stim.Tableau) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the binary symplectic [A | B] from a Clifford tableau.

    The stabiliser generators of *tableau*|0…0⟩ are {tableau.z_output(i)}.
    Returns A and B, each of shape (n, n) over GF(2).
    """
    n = len(tableau)
    A = np.zeros((n, n), dtype=int)
    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        gen = tableau.z_output(i)   # U Z_i U†
        for j in range(n):
            p = int(gen[j])          # 0=I  1=X  2=Y  3=Z
            A[i, j] = int(p in (1, 2))
            B[i, j] = int(p in (2, 3))
    return A, B


def dummyless_generators(A: np.ndarray, B: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Find all non-trivial dummyless stabilisers of the state with symplectic [A|B].

    A coefficient vector c ∈ GF(2)^n is dummyless iff

        supp(B^T c) ⊆ supp(A^T c)   (no qubit has bare Z)

    Returns a list of (Ac, Bc) pairs where Ac = A^T c and Bc = B^T c (mod 2).

    Complexity: O(2^n · n).  Practical for n ≤ ~20.
    """
    n = A.shape[0]
    result = []
    for bits in range(1, 1 << n):
        c  = np.array([(bits >> j) & 1 for j in range(n)], dtype=int)
        Ac = A.T @ c % 2
        Bc = B.T @ c % 2
        if not np.any((Ac == 0) & (Bc == 1)):
            result.append((Ac, Bc))
    return result


def build_detection_basis(
    tableau: stim.Tableau,
) -> tuple[np.ndarray, int, list[np.ndarray]]:
    """
    Maximal GF(2)-independent dummyless detection basis for any Clifford tableau.

    Parameters
    ----------
    tableau : stim.Tableau
        The Clifford circuit preparing the stabiliser state from |0…0⟩.

    Returns
    -------
    D : np.ndarray
        Detection matrix whose rows are the X-parts (A^T c) of the chosen
        independent dummyless generators.
    rank : int
        GF(2) rank of D.  Equals n-1 when the detection scheme leaves only
        the harmless error E* undetectable.
    null_vecs : list[np.ndarray]
        Null space basis of D — the undetectable Z-errors.  For graph states
        with the correct basis, this contains exactly E* (Z on odd-degree nodes).
    """
    A, B = tableau_to_symplectic(tableau)
    basis: list[np.ndarray] = []
    rank = 0
    for Ac, _ in dummyless_generators(A, B):
        candidate = np.vstack(basis + [Ac])
        new_rank = gf2_rank(candidate)
        if new_rank > rank:
            basis.append(Ac)
            rank = new_rank
    n = len(tableau)
    D = np.array(basis) if basis else np.zeros((0, n), dtype=int)
    return D, rank, gf2_nullspace(D)


# ── Convenience: build tableau from a networkx graph ─────────────────────────

def graph_tableau(graph) -> tuple[stim.Tableau, list]:
    """
    Return the Clifford tableau for the graph state of *graph* together with
    the sorted node list (which defines the qubit-index mapping).

    Applies H on every qubit then CZ on every edge.
    """
    import networkx as nx  # local import so the module doesn't require nx at top level

    nodes = sorted(graph.nodes)
    idx   = {v: i for i, v in enumerate(nodes)}
    circ  = stim.Circuit()
    for v in nodes:
        circ.append("H", [idx[v]])
    for u, v in graph.edges:
        circ.append("CZ", [idx[u], idx[v]])
    return circ.to_tableau(), nodes

"""Broadbent-compiled Clifford+MSI circuits and the **FK12 analogue** for verifying them.

This is the reusable core of ``benchmark-stim-msi-bro``. It provides two things:

1. ``build_bro_circuit`` -- a generator of **Broadbent-skeleton** Clifford+MSI circuits,
   parametrised by ``(width, depth)`` exactly like the MBQC brickwork benchmark.
2. ``fk12_bro_test_runs`` -- the **FK12 analogue for the circuit model**: the trap
   incompatibility graph of a Broadbent-compiled circuit is *bipartite* (χ = 2), so the
   single-qubit traps merge into exactly **two test runs** with no graph-colouring search.

See ``FK12_ANALOG.md`` in this folder for the full derivation and the provenance of the
bipartite claim (verified in ``applications/quasi_graph/``).

The Broadbent skeleton
----------------------
Broadbent (2018) compiles every Hadamard as ``H = HTTHTTHTTH`` and every phase as ``P = TT``,
turning the Clifford part of any circuit into **only ``H`` and ``CNOT``** with all non-Clifford
content pushed into magic-state injections (MSI). The Clifford skeleton of an ``H`` is a
6-ancilla gadget (``H`` + ``CNOT`` only); each ``T`` is an MSI gadget ``F = SWAP . CNOT``
(magic-free in test rounds). We build circuits from exactly these pieces.

Why the trap graph is bipartite (mechanism)
-------------------------------------------
Traps are computational-basis: ``stab_q = G† Z_q G``. Two traps are *compatible* (mergeable
into one test run) iff their Pauli strings commute on **every** index. ``H`` is the only gate
that exchanges ``X <-> Z``; ``CNOT`` preserves Pauli type per wire
(``Z_c->Z_c, Z_t->Z_c Z_t``; ``X_c->X_c X_t, X_t->X_t``). So in an ``H + CNOT`` circuit each
wire's Pauli type is fixed by *Hadamard parity*, giving exactly two trap classes -> the
incompatibility graph is bipartite with χ = 2 (the X-test and Z-test of Broadbent). This is
**not** true for ``CZ`` (which mixes ``X_a -> X_a Z_b``); Broadbent compiles to ``CNOT + H``
precisely to stay in the non-mixing regime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import stim
from numpy.random import Generator

# Per-qubit prep gates (from |0>) for the +1 eigenstate of a Pauli *letter* (sign ignored;
# the sign is tracked per trap as the expected outcome). I,Z -> |0>; X -> |+>; Y -> |+i>.
_PREP: dict[int, tuple[str, ...]] = {0: (), 1: ("H",), 2: ("H", "S"), 3: ()}
_2Q_GATES = frozenset({"CX", "CY", "CZ", "XCX", "XCZ", "YCX", "YCZ", "SWAP", "ISWAP"})


# ── circuit generation ───────────────────────────────────────────────────────────


def build_bro_circuit(
    width: int,
    depth: int,
    rng: Generator,
    *,
    p_hgadget: float = 0.3,
    p_msi: float = 0.15,
) -> tuple[stim.Circuit, int, list[int]]:
    """Build a random **Broadbent-skeleton** Clifford+MSI circuit on ``width`` logical wires.

    Each of ``depth`` layers does, per logical wire, an optional **H-gadget** (prob.
    ``p_hgadget``) or **MSI gadget** (prob. ``p_msi``), followed by a brickwork of ``CNOT``
    entangling gates between neighbouring wires. ``holder[r]`` tracks the physical qubit that
    currently carries logical role ``r`` (gadgets teleport the role onto a fresh ancilla),
    mirroring ``applications/quasi_graph``'s ``build_broadbent``.

    All gates are ``H`` and ``CNOT`` (the MSI ``SWAP`` is three ``CNOT``s), so the trap graph
    is bipartite (see module docstring). Returns ``(circuit, n_qubits, useful_qubits)`` where
    ``useful_qubits`` are the role-0 output wire and the MSI-ancilla wires (the paper's ``Q``);
    the benchmark traps *all* wires, of which ``Q`` is a subset.

    The circuit carries no measurements (deferred): it is the unitary ``G`` on ``n_qubits``.
    """
    circuit = stim.Circuit()
    holder = {r: r for r in range(width)}
    nxt = width
    useful: list[int] = [0]  # role-0 output wire

    def h_gadget(r: int) -> None:
        nonlocal nxt
        d = holder[r]
        a = list(range(nxt, nxt + 6))
        nxt += 6
        # Clifford skeleton of H = HTTHTTHTTH: 6 ancillas, 6 CNOTs, 4 Hadamards.
        for gate, *qs in (
            ("H", d), ("CX", a[0], d), ("CX", a[1], a[0]), ("H", a[1]), ("CX", a[2], a[1]),
            ("CX", a[3], a[2]), ("H", a[3]), ("CX", a[4], a[3]), ("CX", a[5], a[4]), ("H", a[5]),
        ):
            circuit.append(gate, list(qs))
        holder[r] = a[5]  # role teleported onto the last ancilla

    def msi_gadget(r: int) -> None:
        nonlocal nxt
        d = holder[r]
        a = nxt
        nxt += 1
        circuit.append("CX", [a, d])    # F = SWAP . CNOT (ancilla a is the injection wire)
        circuit.append("SWAP", [a, d])
        holder[r] = a
        useful.append(a)

    for layer in range(depth):
        for r in range(width):
            u = rng.random()
            if u < p_hgadget:
                h_gadget(r)
            elif u < p_hgadget + p_msi:
                msi_gadget(r)
        for i in range(layer % 2, width - 1, 2):  # brickwork CNOT entangling layer
            a, b = holder[i], holder[i + 1]
            circuit.append("CX", [a, b] if rng.random() < 0.5 else [b, a])

    return circuit, nxt, sorted(set(useful))


def add_depolarising_noise(g_circuit: stim.Circuit, p_depol: float) -> stim.Circuit:
    """Server-side depolarising noise after every gate (``DEPOLARIZE1`` / ``DEPOLARIZE2``)."""
    out = stim.Circuit()
    for inst in g_circuit:
        out.append(inst)
        targets = [tg.value for tg in inst.targets_copy()]
        if inst.name in _2Q_GATES:
            for i in range(0, len(targets), 2):
                out.append("DEPOLARIZE2", [targets[i], targets[i + 1]], p_depol)
        else:
            for q in targets:
                out.append("DEPOLARIZE1", [q], p_depol)
    return out


# ── FK12 analogue: bipartite single-qubit traps ──────────────────────────────────


@dataclass(frozen=True)
class BroTestRun:
    """One colour of the bipartite trap graph = one FK12-style test run.

    * ``prep_letters[k]`` -- the merged Pauli letter (0=I,1=X,2=Y,3=Z) the input must be a
      ``+1`` eigenstate of on wire ``k`` (well-defined because same-colour traps commute on
      every index, so they agree on each non-identity letter).
    * ``traps`` -- the wires whose computational-basis outcome this run checks.
    * ``expected[i]`` -- the deterministic honest outcome of ``traps[i]``: the sign of that
      trap's canonical stabiliser ``stab_q`` (0 if ``+``, 1 if ``-``). The round *fails* iff
      any measured ``outcome_q`` differs from ``expected_q``.
    """

    prep_letters: np.ndarray  # (n_qubits,) uint8
    traps: np.ndarray         # (k,) int
    expected: np.ndarray      # (k,) uint8


def canonical_z_traps(circuit: stim.Circuit, n_qubits: int) -> list[stim.PauliString]:
    """The canonical computational-basis trap basis ``stab_q = G† Z_q G`` for every wire.

    ``inv.z_output(q)`` reads ``G† Z_q G`` straight off the tableau column (no per-trap
    conjugation), so all traps cost ``O(n_qubits^2)`` total rather than ``O(n_qubits^3)``.
    """
    inv = circuit.to_tableau().inverse()  # G^{-1} = G†; z_output(q) = G† Z_q G
    return [inv.z_output(q) for q in range(n_qubits)]


def _expand_to_h_cz(circuit: stim.Circuit) -> list[tuple]:
    """Rewrite the circuit over ``{H, CZ}``: ``CX c t = H t . CZ c t . H t``; ``SWAP = 3 CX``."""
    ops: list[tuple] = []
    for inst in circuit:
        name = inst.name
        tg = [t.value for t in inst.targets_copy()]
        if name == "H":
            ops.extend(("H", q) for q in tg)
        elif name == "CX":
            for i in range(0, len(tg), 2):
                c, t = tg[i], tg[i + 1]
                ops += [("H", t), ("CZ", c, t), ("H", t)]
        elif name == "SWAP":
            for i in range(0, len(tg), 2):
                a, b = tg[i], tg[i + 1]
                for cc, tt in ((a, b), (b, a), (a, b)):
                    ops += [("H", tt), ("CZ", cc, tt), ("H", tt)]
        else:
            raise ValueError(f"unexpected gate in a Broadbent skeleton: {name}")
    return ops


def segment_colours(circuit: stim.Circuit, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """First- and last-segment colours of every wire in **O(gates)** -- no tableau, no graph.

    The correct realisation of "a Hadamard flips the colour". Rewrite to ``{H, CZ}`` and split
    each wire's timeline into **segments** cut at every ``H``: consecutive segments are joined
    (so they take opposite colours -- the flip), and each ``CZ`` joins the two wires' *current*
    segments (``CNOT = H.CZ.H`` couples them). The resulting **segment graph** is a union of
    paths and edges, hence bipartite; BFS 2-colours it. This is exactly the quasi-graph of
    ``applications/reviews/notes_verification.md``.

    Returns ``(col_first, col_last)``: ``col_first[k]`` / ``col_last[k]`` are the colours of
    wire ``k``'s *first* (input) and *last* (output) segment -- its input/output basis type,
    which differ iff wire ``k`` sees an odd number of Hadamards (Broadbent's X<->Z swap).

    A naive "count H's on physical wire ``q`` mod 2" does **not** work: H-parity colours each
    gadget correctly but only up to an independent per-gadget flip, which the data ``CNOT``s
    then make globally inconsistent. The segment graph is what propagates that coupling.
    """
    seg = list(range(n_qubits))
    first = list(range(n_qubits))
    adj: dict[int, list[int]] = {i: [] for i in range(n_qubits)}
    nxt = n_qubits
    for op in _expand_to_h_cz(circuit):
        if op[0] == "H":
            a = op[1]
            adj[nxt] = [seg[a]]
            adj[seg[a]].append(nxt)
            seg[a] = nxt
            nxt += 1
        else:
            _, a, b = op
            adj[seg[a]].append(seg[b])
            adj[seg[b]].append(seg[a])

    colour: dict[int, int] = {}
    for start in adj:
        if start in colour:
            continue
        colour[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in colour:
                    colour[v] = colour[u] ^ 1
                    queue.append(v)
                elif colour[v] == colour[u]:
                    raise ValueError("segment graph not bipartite (not a Broadbent skeleton?)")
    col_first = np.array([colour[first[w]] for w in range(n_qubits)], dtype=np.int8)
    col_last = np.array([colour[seg[w]] for w in range(n_qubits)], dtype=np.int8)
    return col_first, col_last


def segment_two_colouring(circuit: stim.Circuit, n_qubits: int) -> np.ndarray:
    """The trap 2-colouring (last-segment colour per wire); see :func:`segment_colours`."""
    return segment_colours(circuit, n_qubits)[1]


def two_colouring(stabs: list[stim.PauliString], n_qubits: int) -> np.ndarray:
    """**Oracle** 2-colouring from the trap incompatibility graph (``O(N^3)``; for tests).

    Edge ``(i, j)`` iff ``stab_i`` and ``stab_j`` *anti-commute at some index*. For a Broadbent
    skeleton the graph is bipartite, so BFS 2-colours it -- but building the adjacency is cubic.
    The production path uses :func:`segment_two_colouring` (linear) instead; this stays as the
    correctness oracle that the structural colouring is checked against.
    """
    letters = np.array([[s[k] for k in range(n_qubits)] for s in stabs], dtype=np.uint8)
    x = (letters == 1) | (letters == 2)  # X or Y has an x-bit
    z = (letters == 3) | (letters == 2)  # Z or Y has a z-bit

    color = np.full(n_qubits, -1, dtype=np.int8)
    for start in range(n_qubits):
        if color[start] >= 0:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            anti = ((x[u][None, :] & z) ^ (z[u][None, :] & x)).any(axis=1)
            anti[u] = False
            for v in np.flatnonzero(anti):
                if color[v] < 0:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    raise ValueError("trap incompatibility graph is not bipartite (not a Broadbent skeleton?)")
    return color


def fk12_bro_test_runs(circuit: stim.Circuit, n_qubits: int) -> list[BroTestRun]:
    """The **FK12 analogue**: the two Broadbent test runs, built in **O(gates)** -- no tableau.

    This is Broadbent's X-test / Z-test in closed form. Feed each wire ``|0>`` (Z) or ``|+>``
    (X); a Hadamard locally swaps ``X <-> Z``; the two runs are the two segment colours. From
    the segment colouring ``(col_first, col_last)`` (:func:`segment_colours`):

      * **test run ``C`` in {0, 1}:** trap wires ``= {q : col_last[q] == C}``, each with
        **expected outcome 0**;
      * **input prep:** wire ``k`` is ``|+>`` (X) iff ``col_first[k] != C``, else ``|0>`` (Z).

    This needs no stabiliser computation: it relies on the verified invariant that every
    canonical trap ``stab_q = G† Z_q G`` of a Broadbent skeleton (real Clifford ``<H, CNOT>``,
    Z-basis traps) is **sign-+** and **Y-free**, so the prep is purely ``|0>``/``|+>`` and all
    expected outcomes are 0. Verified noiseless-deterministic, and identical ``p_failed_round``
    to the tableau path. The tableau-based :func:`canonical_z_traps` / :func:`two_colouring`
    remain as the correctness oracle in the test suite.
    """
    col_first, col_last = segment_colours(circuit, n_qubits)
    runs: list[BroTestRun] = []
    for c in (0, 1):
        traps = np.flatnonzero(col_last == c)
        prep = np.where(col_first != c, 1, 3).astype(np.uint8)  # |+>=X if col_first != C else |0>=Z
        runs.append(BroTestRun(prep_letters=prep, traps=traps, expected=np.zeros(traps.size, dtype=np.uint8)))
    return runs


def test_run_fail_pool(
    test_run: BroTestRun, noisy_g: stim.Circuit, n_qubits: int, n_shots: int
) -> np.ndarray:
    """Batch-sample ``n_shots`` honest-but-noisy rounds of one test run.

    Prepares the merged ``+1`` eigenstate, applies the noisy ``G``, measures all wires, and
    returns a boolean ``(n_shots,)``: ``True`` where some trap's outcome differs from its
    expected (deterministic) honest value.
    """
    circuit = stim.Circuit()
    for k in range(n_qubits):
        for gate in _PREP[int(test_run.prep_letters[k])]:
            circuit.append(gate, [k])
    circuit += noisy_g
    circuit.append("M", list(range(n_qubits)))  # record index == wire index

    samples = np.asarray(circuit.compile_sampler().sample(shots=n_shots))
    fail = np.zeros(n_shots, dtype=bool)
    for q, exp in zip(test_run.traps, test_run.expected):
        fail |= (samples[:, q] ^ exp).astype(bool)
    return fail

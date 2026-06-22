"""Dummyless stabilizer search in the *circuit model* (Broadbent compilation).

Background (see `applications/reviews/notes_verification.md`, §"Pauli noise detection
in Clifford circuits via stabilizer testing"):

* A Clifford `C` on `N` wires is tested by stabilizer testing. For a chosen output
  measurement basis `P` (X or Z) on each wire, the **canonical stabilizer basis** is

      S_i = C^dag P_i C ,   i = 0 .. N-1 .

  Preparing a +1-eigenstate of `S_i` at the input and checking the parity of the
  `P`-measurements at the output certifies the `i`-th output observable.

* A stabilizer is **dummyless** if its Pauli string contains **no Z** (only I, X, Y).
  Then any +1-eigenstate is a product of **XY-plane** single-qubit states — no `|0>`
  "dummy" qubits are needed.

* The canonical basis is in general *not* dummyless. But test runs compose: the XOR of
  outcomes corresponds to the **product** of stabilizers. So we may change basis by
  multiplying canonical stabilizers together. The quest:

      from {S_0, ..., S_{N-1}}, build as many **linearly independent dummyless**
      stabilizers as possible (products of the S_i). In MBQC / graph states `n-1`
      always suffices; is it reachable in the circuit model too?

Important subtlety: "dummyless" (no Z, but Y allowed) is **not** a linear subspace of
the symplectic space, because `X * Y = iZ` at a shared index. So this is a genuine
*combinatorial* search over subsets of the canonical basis, not plain linear algebra.

This module provides:
  * `build_broadbent(...)`        -- build a Broadbent-compiled Clifford skeleton
  * `single_hadamard_gadget()`    -- the 7-wire minimal working example
  * `canonical_basis(...)`        -- S_i = C^dag P_i C
  * `Stab`                        -- a stabilizer that remembers which S_i it is built
                                     from, supports `*`, and knows if it is dummyless
  * `Workspace`                   -- the thing you actually play with: `ws.s[i]`,
                                     `ws.combine(0, 1, 4, 6)`, `ws.search_dummyless()`,
                                     `ws.eigenstate(stab)`, ...

Only depends on `stim` (+ the std lib).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import stim

# --------------------------------------------------------------------------- #
#  Circuit building -- the Broadbent compilation skeleton                      #
# --------------------------------------------------------------------------- #

# Broadbent realises H = HTTHTTHTTH; dropping the T-rotations (which are identity
# in a test run) leaves the Clifford *skeleton*: 6 ancillas, 6 CNOTs, 4 Hadamards.
# The data qubit's role is teleported down the chain and exits on the 6th ancilla.
#
# Two equivalent ancilla conventions (they share the same output-side stabilizers,
# since H CNOT H = CZ):
#   * "broadbent"  -- CNOT data gates + CNOT-based gadget (the compilation skeleton)
#   * "quasi_graph"-- CZ   data gates + CZ-based   gadget (the notes' H-gadget)


def _gadget_lines(style: str, d: int, anc: tuple[int, ...]) -> list[str]:
    """Lines for one H-gadget on data wire `d` with 6 fresh ancillas `anc`.

    The output role exits on `anc[5]` (the 6th ancilla).
    """
    a1, a2, a3, a4, a5, a6 = anc
    if style == "broadbent":  # CNOT-based skeleton
        return [
            f"H {d}",
            f"CNOT {a1} {d}", f"CNOT {a2} {a1}", f"H {a2}",
            f"CNOT {a3} {a2}", f"CNOT {a4} {a3}", f"H {a4}",
            f"CNOT {a5} {a4}", f"CNOT {a6} {a5}", f"H {a6}",
        ]
    if style == "quasi_graph":  # CZ-based gadget (notes §H-gadget)
        return [
            f"CZ {d} {a1}", f"H {a1}",
            f"CZ {a1} {a2}", f"CZ {a2} {a3}", f"H {a3}",
            f"CZ {a3} {a4}", f"CZ {a4} {a5}", f"H {a5}",
            f"CZ {a5} {a6}", f"H {a6}",
        ]
    raise ValueError(f"unknown style {style!r} (use 'broadbent' or 'quasi_graph')")


def build_broadbent(
    num_roles: int,
    logical_ops: list[tuple],
    style: str = "broadbent",
) -> tuple[stim.Circuit, int, dict[int, list[int]]]:
    """Build a Broadbent-compiled Clifford skeleton.

    Parameters
    ----------
    num_roles
        Number of logical qubits ("roles") the computation starts with.
    logical_ops
        A list of operations on *roles*:
          * ("E", r, s) -- entangling gate between roles r and s
                           (CNOT if style="broadbent", CZ if style="quasi_graph")
          * ("H", r)    -- a Hadamard on role r, realised by an H-gadget.
        A role is teleported by its H-gadget onto a fresh ancilla wire, so the wire
        currently *holding* a role changes over time.
    style
        "broadbent" (CNOT skeleton) or "quasi_graph" (CZ skeleton).

    Returns
    -------
    (circuit, num_wires, trajectories)
        `trajectories[r]` is the list of wires that ever held role `r`
        (its first entry is `r`, the last is the final holder).
    """
    lines: list[str] = []
    holder = {r: r for r in range(num_roles)}        # role -> current physical wire
    traj: dict[int, list[int]] = {r: [r] for r in range(num_roles)}
    nxt = num_roles                                  # next free ancilla wire

    for op in logical_ops:
        kind = op[0]
        if kind in ("E", "CNOT", "CZ"):
            r, s = op[1], op[2]
            gate = "CZ" if style == "quasi_graph" else "CNOT"
            lines.append(f"{gate} {holder[r]} {holder[s]}")
        elif kind == "H":
            r = op[1]
            d = holder[r]
            anc = tuple(range(nxt, nxt + 6))
            nxt += 6
            lines += _gadget_lines(style, d, anc)
            traj[r].extend(anc)
            holder[r] = anc[5]                       # role now lives on the 6th ancilla
        else:
            raise ValueError(f"unknown logical op {op!r}")

    # Pad with identity across every wire so stim's tableau spans all `nxt` qubits.
    header = "I " + " ".join(map(str, range(nxt)))
    circuit = stim.Circuit("\n".join([header, *lines]))
    return circuit, nxt, traj


def single_hadamard_gadget(style: str = "broadbent") -> tuple[stim.Circuit, int, dict[int, list[int]]]:
    """The minimal working example: one H-gadget on a single role -> 7 wires."""
    return build_broadbent(1, [("H", 0)], style=style)


# --------------------------------------------------------------------------- #
#  Canonical stabilizer basis                                                  #
# --------------------------------------------------------------------------- #

def canonical_basis(circuit: stim.Circuit, basis: str = "Z") -> list[stim.PauliString]:
    """Canonical basis S_i = C^dag P_i C for every wire i (P = Z or X).

    `Z` is Broadbent's computational-basis convention (the Broadbent skeleton is
    naturally dummyless-rich here). `X` is the MBQC convention of the notes.
    """
    sim = stim.TableauSimulator()
    sim.do_circuit(circuit)
    inv = sim.current_inverse_tableau()              # = G^-1 = C^dag (forward C = G)
    n = circuit.num_qubits
    out = inv.z_output if basis == "Z" else inv.x_output
    return [out(i) for i in range(n)]


# --------------------------------------------------------------------------- #
#  Stabilizer wrapper that remembers its provenance                            #
# --------------------------------------------------------------------------- #

_PAULI_CHAR = {0: "_", 1: "X", 2: "Y", 3: "Z"}


def is_dummyless(pauli: stim.PauliString) -> bool:
    """True iff the Pauli string contains no Z (only I, X, Y)."""
    return all(pauli[k] != 3 for k in range(len(pauli)))


def _bits(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if (mask >> i) & 1]


@dataclass(frozen=True)
class Stab:
    """A stabilizer that remembers it is a product of canonical generators.

    `support` is a bitmask over canonical-basis indices: bit i set <=> S_i is a
    factor. Multiplying two `Stab`s XORs their supports (GF(2)) and multiplies the
    Pauli strings -- exactly the "compose two test runs" operation.
    """

    pauli: stim.PauliString
    support: int

    def __mul__(self, other: "Stab") -> "Stab":
        return Stab(self.pauli * other.pauli, self.support ^ other.support)

    @property
    def is_dummyless(self) -> bool:
        return is_dummyless(self.pauli)

    @property
    def factors(self) -> list[int]:
        """Indices of the canonical generators whose product this is."""
        return _bits(self.support)

    @property
    def weight(self) -> int:
        """Number of non-identity qubits."""
        return sum(self.pauli[k] != 0 for k in range(len(self.pauli)))

    def pauli_str(self) -> str:
        return str(self.pauli)

    def __repr__(self) -> str:
        flag = "dummyless ✓" if self.is_dummyless else "has Z     ✗"
        prov = "·".join(f"S{i}" for i in self.factors) or "I"
        return f"{self.pauli}   [{flag}]   = {prov}"


# --------------------------------------------------------------------------- #
#  GF(2) linear algebra on the support bitmasks                                #
# --------------------------------------------------------------------------- #

def gf2_independent(masks: list[int]) -> list[int]:
    """Return a maximal linearly independent subset of `masks` (greedy, order-keeping)."""
    basis: list[int] = []
    kept: list[int] = []
    for m in masks:
        x = m
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
            basis.sort(reverse=True)
            kept.append(m)
    return kept


def gf2_rank(masks: list[int]) -> int:
    basis: list[int] = []
    for m in masks:
        x = m
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
            basis.sort(reverse=True)
    return len(basis)


# --------------------------------------------------------------------------- #
#  The Workspace -- what you actually play with                                #
# --------------------------------------------------------------------------- #

class Workspace:
    """Hold a circuit, its canonical basis, and tools to manipulate stabilizers.

    Quick start::

        ws = Workspace.single_hadamard()      # 7-wire Broadbent H-gadget, Z-basis
        ws.show_basis()                        # print S_0 .. S_6
        print(ws.combine(0, 1, 4, 6))          # try the product S0·S1·S4·S6
        print(ws.s[0] * ws.s[1])               # same idea, operator style
        gens = ws.search_dummyless()           # find a maximal independent dummyless set
    """

    def __init__(self, circuit: stim.Circuit, basis: str = "Z",
                 trajectories: dict[int, list[int]] | None = None, name: str = ""):
        self.circuit = circuit
        self.basis = basis
        self.name = name
        self.N = circuit.num_qubits
        self.trajectories = trajectories or {}
        self._S = canonical_basis(circuit, basis)
        self.s: list[Stab] = [Stab(self._S[i], 1 << i) for i in range(self.N)]

    # ----- constructors --------------------------------------------------- #

    @classmethod
    def single_hadamard(cls, basis: str = "Z", style: str = "broadbent") -> "Workspace":
        circ, _, traj = single_hadamard_gadget(style=style)
        return cls(circ, basis=basis, trajectories=traj,
                   name=f"single H-gadget ({style}, {basis}-basis)")

    @classmethod
    def from_logical(cls, num_roles: int, logical_ops: list[tuple],
                     basis: str = "Z", style: str = "broadbent") -> "Workspace":
        circ, _, traj = build_broadbent(num_roles, logical_ops, style=style)
        return cls(circ, basis=basis, trajectories=traj,
                   name=f"{num_roles} roles, {len(logical_ops)} ops ({style}, {basis}-basis)")

    # ----- combining stabilizers ------------------------------------------ #

    def combine(self, *idxs: int) -> Stab:
        """Product S_{i1} · S_{i2} · ... of canonical generators (your main tool)."""
        flat: list[int] = []
        for x in idxs:
            flat.extend(x) if isinstance(x, (list, tuple, set, frozenset)) else flat.append(x)
        out = Stab(stim.PauliString(self.N), 0)      # identity
        for i in flat:
            out = out * self.s[i]
        return out

    def __getitem__(self, i: int) -> Stab:
        return self.s[i]

    def __len__(self) -> int:
        return self.N

    # ----- inspection ----------------------------------------------------- #

    def show_basis(self) -> None:
        print(f"# {self.name or 'circuit'}  -- N = {self.N} wires, {self.basis}-basis")
        print(f"# canonical basis  S_i = C^dag {self.basis}_i C")
        for i, st in enumerate(self.s):
            flag = "dummyless" if st.is_dummyless else "has Z"
            print(f"  S_{i} = {st.pauli}   ({flag})")

    # ----- dummyless search ----------------------------------------------- #

    def all_dummyless(self, max_terms: int | None = None) -> list[Stab]:
        """Exhaustively enumerate every non-empty product of generators that is
        dummyless. Cost is 2^N; fine for the MWE, slow beyond ~20 wires."""
        max_terms = max_terms or self.N
        found: list[Stab] = []
        for r in range(1, max_terms + 1):
            for combo in itertools.combinations(range(self.N), r):
                st = self.combine(*combo)
                if st.is_dummyless:
                    found.append(st)
        return found

    def search_dummyless(self, verbose: bool = True) -> list[Stab]:
        """Greedily extract a maximal *linearly independent* set of dummyless
        stabilizers, preferring low-weight / few-term combinations first."""
        cand = self.all_dummyless()
        cand.sort(key=lambda st: (bin(st.support).count("1"), st.weight, st.support))
        basis: list[int] = []
        chosen: list[Stab] = []
        for st in cand:
            x = st.support
            for b in basis:
                x = min(x, x ^ b)
            if x:
                basis.append(x)
                basis.sort(reverse=True)
                chosen.append(st)
        if verbose:
            print(f"# {self.name or 'circuit'}")
            print(f"# dummyless stabilizers found (independent): "
                  f"{len(chosen)} of n-1 = {self.N - 1} target (n = {self.N})")
            for st in chosen:
                print("  " + repr(st))
        return chosen

    # ----- check a user-proposed set -------------------------------------- #

    def check_set(self, combos: list) -> dict:
        """Given a list of combos (each a Stab or an iterable of generator indices),
        report how many are dummyless and their GF(2) rank."""
        stabs = [c if isinstance(c, Stab) else self.combine(*c) for c in combos]
        n_dummyless = sum(st.is_dummyless for st in stabs)
        rank = gf2_rank([st.support for st in stabs])
        return {
            "count": len(stabs),
            "dummyless": n_dummyless,
            "all_dummyless": n_dummyless == len(stabs),
            "rank": rank,
            "independent": rank == len(stabs),
            "stabs": stabs,
        }

    # ----- generate the +1-eigenstate (input states) ---------------------- #

    def eigenstate(self, stab: Stab) -> dict[int, dict]:
        """For a *dummyless* stabilizer, return the product +1-eigenstate as
        single-qubit XY-plane states: {wire: {"pauli", "angle_pi", "ket"}}.

        `angle_pi` is the Bloch angle phi (state (|0> + e^{i phi}|1>)/sqrt2) in units
        of pi. All angles are in {0, 1/2, 1, 3/2} -> the four XY-plane states, never
        a computational-basis "dummy".
        """
        if not stab.is_dummyless:
            raise ValueError("eigenstate is only XY-plane (dummyless) for no-Z stabilizers")
        sign = stab.pauli.sign
        if sign not in (1, -1):
            raise ValueError(f"non-real stabilizer sign {sign!r}; not a valid ±1 stabilizer")

        # per-qubit +1 eigenstates of X (|+>) and Y (|+i>); I is free, take |+>.
        plus = {  # (pauli, plus-eigenvalue state)
            0: ("I", 0.0, "|+>"),
            1: ("X", 0.0, "|+>"),
            2: ("Y", 0.5, "|+i>"),
        }
        minus = {
            1: ("X", 1.0, "|->"),
            2: ("Y", 1.5, "|-i>"),
        }
        paulis = [stab.pauli[k] for k in range(self.N)]
        states = {}
        for k, p in enumerate(paulis):
            name, ang, ket = plus[p]
            states[k] = {"pauli": name, "angle_pi": ang, "ket": ket}

        # Fix the global sign by flipping ONE non-identity qubit if needed.
        if sign == -1:
            flip = next((k for k, p in enumerate(paulis) if p != 0), None)
            if flip is None:
                raise ValueError("cannot represent -I as a +1 eigenstate")
            name, ang, ket = minus[paulis[flip]]
            states[flip] = {"pauli": name, "angle_pi": ang, "ket": ket}
        return states

    def eigenstate_str(self, stab: Stab) -> str:
        st = self.eigenstate(stab)
        return "  ".join(f"q{k}:{v['ket']}" for k, v in st.items())

    # ----- the graph-based construction, transferred to circuit stabilizers - #
    #
    # The notes' polynomial graph algorithm (and `veriphix.protocols.Dummyless`):
    #   1. Rfull = product of ALL canonical stabilizers is dummyless.
    #   2. "even-degree" nodes: R\v = Rfull * S_v stays dummyless -> a generator.
    #   3. "odd-degree" nodes: paired up via paths through even-degree nodes.
    # That algorithm assumes *graph-state* stabilizers S_v = X_v . prod_{w~v} Z_w.
    # Our circuit stabilizers are NOT of that form, but the same three moves
    # transfer with one change: "even degree" becomes a LOCAL predicate.

    def rfull(self) -> Stab:
        """Rfull = product of all canonical generators. Dummyless whenever every
        qubit has an odd number of X-owners (true for Broadbent H-gadget chains)."""
        return self.combine(*range(self.N))

    def x_owners(self) -> dict[int, list[int]]:
        """qubit -> list of generators that carry an X/Y (not Z) at that qubit."""
        return {k: [i for i in range(self.N) if self._S[i][k] in (1, 2)] for k in range(self.N)}

    def removable_singles(self) -> tuple[list[int], list[int]]:
        """Generalised "even-degree" test.

        Returns (removable, disagree). `R\\i = Rfull * S_i` is dummyless iff S_i
        *agrees* with Rfull (same X vs Y) at every qubit S_i owns (is X-type on);
        a Z appears exactly where they disagree. Pure-Z generators are always
        removable. The disagree set is the circuit analogue of the odd-degree
        nodes -- it is the X-owners of Rfull's Y positions, and is even in size.
        """
        rfull = self.rfull()
        removable, disagree = [], []
        for i in range(self.N):
            (removable if (rfull * self.s[i]).is_dummyless else disagree).append(i)
        return removable, disagree

    def incompatibility_graph(self) -> nx.Graph:
        """Edge i--j iff S_i, S_j do NOT commute on every qubit index (the
        trap-merging incompatibility graph). NB: its *degree* does NOT predict
        removability -- only its path structure is useful for pairing."""
        g = nx.Graph()
        g.add_nodes_from(range(self.N))
        for i, j in itertools.combinations(range(self.N), 2):
            if not all(not (a and b and a != b)
                       for a, b in zip(list(self._S[i]), list(self._S[j]))):
                g.add_edge(i, j)
        return g

    def pairing_pool(self, max_traj: int = 13) -> list[Stab]:
        """Candidate dummyless generators that PAIR UP the disagree set -- the
        circuit analogue of `odd_pair_generators_bfs`'s odd-node spanning tree.

        Two kinds of "edges" in the disagree pairing graph:

        * within-gadget paths: a disagree pair at the two ends of an H-gadget is
          connected *through that gadget's own ancilla chain*. Each role's
          trajectory is a short 1-D chain, so we enumerate dummyless products over
          its wires (bounded -> polynomial when gadget-depth per role is bounded).
        * cross-gadget direct pairs: `Rfull * S_i * S_j` for disagree i,j -- the
          CNOT-coupling pairs (e.g. two ends that annihilate the same charge).

        Driving the paths by the **role trajectories** (not the incompatibility
        graph) is the key fix: the incompatibility graph connects the wrong
        endpoints and its longer paths pick up Zs.
        """
        rfull = self.rfull()
        _, disagree = self.removable_singles()
        pool: list[Stab] = []
        # within-gadget chain paths
        for traj in self.trajectories.values():
            wires = sorted(set(traj))
            if len(wires) > max_traj:
                continue
            for r in range(2, len(wires) + 1):
                for combo in itertools.combinations(wires, r):
                    st = self.combine(*combo)
                    if st.is_dummyless:
                        pool.append(st)
        # cross-gadget direct pairs
        for i, j in itertools.combinations(disagree, 2):
            st = rfull * self.s[i] * self.s[j]
            if st.is_dummyless:
                pool.append(st)
        return pool

    def constructive_dummyless(self, verbose: bool = True, exhaustive: bool = True) -> dict:
        """Build a dummyless set the graph-algorithm way, in named phases:

          A  Rfull + every removable single  (polynomial; the transferred graph step)
          B  trajectory-driven pairing of the disagree set (`pairing_pool`) -- the
             generalised odd-pair construction (polynomial for bounded gadget depth)
          C  residual completion from the exhaustive dummyless pool (the part that is
             cheap only for small N -- this is the open "is it poly for all Cliffords?")

        Phase B reaches n-1 on H-gadget chains AND CNOT-*chain* circuits (whenever
        Rfull stays dummyless and the pairing graph is rich enough). It does NOT
        always suffice -- e.g. a CNOT *ring* makes Rfull non-dummyless and the whole
        Rfull-anchored construction breaks; that residual is the open question.

        Returns a report dict with the generators and the rank reached after each phase.
        """
        rfull = self.rfull()
        removable, disagree = self.removable_singles()

        gens: list[Stab] = []
        sup: list[int] = []

        def add(st: Stab) -> bool:
            if gf2_rank(sup + [st.support]) > len(sup):
                gens.append(st)
                sup.append(st.support)
                return True
            return False

        # Phase A ----------------------------------------------------------- #
        add(rfull)
        for i in removable:
            add(rfull * self.s[i])
        rank_A = len(sup)

        # Phase B ----------------------------------------------------------- #
        for st in sorted(self.pairing_pool(), key=lambda s: s.weight):
            if len(sup) == self.N - 1:
                break
            add(st)
        rank_B = len(sup)

        # Phase C ----------------------------------------------------------- #
        if exhaustive and len(sup) < self.N - 1:
            for st in sorted(self.all_dummyless(), key=lambda s: s.weight):
                if len(sup) == self.N - 1:
                    break
                add(st)
        rank_C = len(sup)

        report = {
            "rfull": rfull,
            "removable": removable,
            "disagree": disagree,
            "generators": gens,
            "rank_A": rank_A,
            "rank_B": rank_B,
            "rank_C": rank_C,
            "target": self.N - 1,
        }
        if verbose:
            print(f"# {self.name or 'circuit'}  (N = {self.N}, target n-1 = {self.N - 1})")
            print(f"  Rfull = {rfull.pauli}   dummyless = {rfull.is_dummyless}")
            print(f"  removable singles (generalised even-degree): {removable}")
            print(f"  disagree set      (generalised odd-degree) : {disagree}")
            print(f"  rank after A (Rfull + removables)         : {rank_A}")
            print(f"  rank after B (+ trajectory pairing, poly) : {rank_B}"
                  + ("   <- B reached n-1" if rank_B == self.N - 1 else ""))
            print(f"  rank after C (+ exhaustive residual)      : {rank_C}"
                  + ("   <- needed brute force" if rank_C > rank_B else ""))
        return report


# --------------------------------------------------------------------------- #
#  Self-check when run directly                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ws = Workspace.single_hadamard()
    ws.show_basis()
    print()
    ws.search_dummyless()

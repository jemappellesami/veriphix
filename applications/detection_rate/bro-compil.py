"""
Clifford circuit generator using CNOTs and H-gadgets, via stim.

H-gadget on data qubit d (uses 6 new ancillas a1..a6):
  H(d)
  CNOT(a1 -> d)
  CNOT(a2 -> a1)
  H(a2)
  CNOT(a3 -> a2)
  CNOT(a4 -> a3)
  H(a4)
  CNOT(a5 -> a4)
  CNOT(a6 -> a5)
  H(a6)
"""

import stim
import numpy as np


# ---------------------------------------------------------------------------
# Circuit building
# ---------------------------------------------------------------------------

def apply_h_gadget(lines: list[str], data: int, next_anc: int) -> int:
    """
    Append H-gadget instruction strings to *lines*.
    Returns updated next_anc (= old next_anc + 6).
    """
    a1, a2, a3, a4, a5, a6 = range(next_anc, next_anc + 6)

    lines += [
        f"H {data}",
        f"CNOT {a1} {data}",
        f"CNOT {a2} {a1}",
        f"H {a2}",
        f"CNOT {a3} {a2}",
        f"CNOT {a4} {a3}",
        f"H {a4}",
        f"CNOT {a5} {a4}",
        f"CNOT {a6} {a5}",
        f"H {a6}",
    ]
    return next_anc + 6


Op = tuple  # ('CNOT', ctrl, tgt) | ('H_GADGET', qubit)


def build_circuit(n: int, ops: list[Op]) -> tuple[stim.Circuit, int]:
    """
    Build a stim circuit starting with *n* data qubits.

    ops  –  list of:
        ('CNOT', control, target)
        ('H_GADGET', qubit)

    Returns (circuit, total_qubits).
    """
    lines: list[str] = []
    next_anc: int = n

    for op in ops:
        kind = op[0]
        if kind == "CNOT":
            ctrl, tgt = int(op[1]), int(op[2])
            lines.append(f"CNOT {ctrl} {tgt}")
        elif kind == "H_GADGET":
            qubit = int(op[1])
            next_anc = apply_h_gadget(lines, qubit, next_anc)
        else:
            raise ValueError(f"Unknown operation: {kind!r}")

    circuit = stim.Circuit("\n".join(lines)) if lines else stim.Circuit()
    return circuit, next_anc


# ---------------------------------------------------------------------------
# Tableau helpers
# ---------------------------------------------------------------------------

def circuit_tableau(circuit: stim.Circuit) -> stim.Tableau:
    """Return the unitary tableau U of the circuit (forward action)."""
    sim = stim.TableauSimulator()
    sim.do_circuit(circuit)
    return sim.current_inverse_tableau() ** -1


def print_tableau(tab: stim.Tableau, total_qubits: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Stabilizer tableau  ({total_qubits} qubits)")
    print(f"{'='*60}")
    print("X outputs  U X_i U†:")
    for i in range(total_qubits):
        print(f"  X_{i} -> {tab.x_output(i)}")
    print("\nZ outputs  U Z_i U†:")
    for i in range(total_qubits):
        print(f"  Z_{i} -> {tab.z_output(i)}")


# ---------------------------------------------------------------------------
# Commutation relations for Z_i
# ---------------------------------------------------------------------------

def z_commutation(tab: stim.Tableau, total_qubits: int) -> np.ndarray:  # type: ignore[type-arg]
    """
    For each pair (i,j) compute whether the evolved operators
        Ẑ_i = U Z_i U†   and   Ẑ_j = U Z_j U†
    commute (0) or anticommute (1).

    Returns an (n x n) integer matrix.
    """
    evolved = [tab.z_output(i) for i in range(total_qubits)]

    mat = np.zeros((total_qubits, total_qubits), dtype=int)
    for i in range(total_qubits):
        for j in range(total_qubits):
            mat[i, j] = 0 if evolved[i].commutes(evolved[j]) else 1
    return mat


def print_commutation(mat: np.ndarray, total_qubits: int) -> None:  # type: ignore[type-arg]
    n = total_qubits
    w = max(3, len(str(n - 1)) + 2)
    print(f"\n{'='*60}")
    print("  Commutation of evolved Ẑ_i = U Z_i U†")
    print("  C = commute   A = anticommute")
    print(f"{'='*60}")

    header = " " * (w + 1) + "".join(f"Z{j:<{w}}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"Z{i:<{w}}" + "".join(
            ("C" if mat[i, j] == 0 else "A") + " " * w for j in range(n)
        )
        print(row)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if mat[i, j]]
    print()
    if pairs:
        print("Anticommuting pairs:")
        for i, j in pairs:
            print(f"  Ẑ_{i}  ✕  Ẑ_{j}")
    else:
        print("All evolved Ẑ_i mutually commute.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = 2   # initial data qubits: 0, 1

    ops: list[Op] = [
        ("H_GADGET", 0),   # qubit 0  → ancillas 2..7
        ("CNOT",     1, 0),
        ("H_GADGET", 1),   # qubit 1  → ancillas 8..13
    ]

    print(f"Initial qubits : {n}  (indices 0..{n-1})")
    print(f"Operations     : {ops}")

    circuit, total = build_circuit(n, ops)

    print(f"\nTotal qubits   : {total}")
    print(f"\n--- stim circuit ---\n{circuit}")

    tab = circuit_tableau(circuit)
    print_tableau(tab, total)

    mat = z_commutation(tab, total)
    print_commutation(mat, total)

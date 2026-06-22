"""Minimal working example: one Broadbent H-gadget (7 wires) -> 6 dummyless stabilizers.

Run:  ./.venv/bin/python applications/dummyless_circuit_model/minimal_example.py

This walks through the whole story on the smallest non-trivial circuit:
  1. build the Broadbent H-gadget skeleton (data wire 0 + 6 ancillas = 7 wires),
  2. print the canonical stabilizer basis S_i = C^dag Z_i C,
  3. show that the canonical basis is *not* dummyless (every S_i has a Z),
  4. find n-1 = 6 linearly independent *dummyless* stabilizers by combining them,
  5. print the XY-plane input state each dummyless stabilizer asks for.
"""

from dummyless import Workspace


def main() -> None:
    # 1. The minimal circuit: a single H-gadget on one role -> 7 wires.
    ws = Workspace.single_hadamard(basis="Z", style="broadbent")

    print("=" * 70)
    print("1. CIRCUIT")
    print("=" * 70)
    print(ws.circuit)
    print(f"\n  total wires n = {ws.N}   (role 0 starts on wire 0, exits on wire "
          f"{ws.trajectories[0][-1]})")
    print(f"  role-0 trajectory through the gadget: {ws.trajectories[0]}")

    # 2 + 3. Canonical basis -- note every generator carries a Z.
    print("\n" + "=" * 70)
    print("2. CANONICAL STABILIZER BASIS  S_i = C^dag Z_i C")
    print("=" * 70)
    ws.show_basis()
    print("\n  -> none of the S_i is dummyless: each needs a Z-eigenstate (a 'dummy').")

    # A couple of hand-built combinations, to show the tool.
    print("\n" + "=" * 70)
    print("3. TRYING COMBINATIONS BY HAND")
    print("=" * 70)
    for combo in [(0, 1), (0, 1, 2), (0, 2, 3, 6), (0, 2, 4, 6)]:
        st = ws.combine(*combo)
        print(f"  S{combo} -> {st!r}")

    # 4. The search: a maximal independent dummyless set.
    print("\n" + "=" * 70)
    print("4. SEARCH: maximal linearly independent dummyless set")
    print("=" * 70)
    gens = ws.search_dummyless()

    # 5. The XY-plane input state each one requires (no |0> dummies anywhere).
    print("\n" + "=" * 70)
    print("5. XY-PLANE INPUT STATE FOR EACH DUMMYLESS STABILIZER")
    print("=" * 70)
    for st in gens:
        print(f"  {st.pauli}  <-  {ws.eigenstate_str(st)}")

    print("\n" + "=" * 70)
    report = ws.check_set(gens)
    print(f"RESULT: {report['rank']} independent dummyless stabilizers "
          f"(target n-1 = {ws.N - 1}). "
          + ("REACHED ✓" if report["rank"] == ws.N - 1 else "NOT reached ✗"))
    print("=" * 70)


if __name__ == "__main__":
    main()

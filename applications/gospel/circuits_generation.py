"""
Goal of this script: when triggered, generates random circuits with command like

`python -m veriphix.sampling_circuits.sampling_circuits --ncircuits 10 --nqubits 4 --depth 5 --p-gate 0.5 --p-cnot 0.25 --p-cnot-flip 0.5 --p-rx 0.5 --seed 1729 --target circuits`

but nqubits and depth are subject to the following sweep:
- nqubits from 2 to 4
- depth from 2 to 4 as well

and output dir for circuits is 'circuits-n-d' where $n$ is nqubits and $d$ is depth

when running this file, it should generate circuits in applications/gospel/circuits/

"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CIRCUITS_BASE = Path("applications/gospel/circuits")

NQUBITS_RANGE = range(3, 6)
DEPTH_RANGE   = range(5, 8)

NCIRCUITS  = 1000
P_GATE     = 0.5
P_CNOT     = 0.25
P_CNOT_FLIP = 0.5
P_RX       = 0.5
SEED       = 1729

CIRCUITS_BASE.mkdir(parents=True, exist_ok=True)

for n in NQUBITS_RANGE:
    for d in DEPTH_RANGE:
        target = CIRCUITS_BASE / f"circuits-{n}-{d}"
        if target.exists():
            print(f"Skipping {target} (already exists)")
            continue

        cmd = [
            sys.executable, "-m", "veriphix.sampling_circuits.sampling_circuits",
            "--ncircuits",   str(NCIRCUITS),
            "--nqubits",     str(n),
            "--depth",       str(d),
            "--p-gate",      str(P_GATE),
            "--p-cnot",      str(P_CNOT),
            "--p-cnot-flip", str(P_CNOT_FLIP),
            "--p-rx",        str(P_RX),
            "--seed",        str(SEED),
            "--target",      str(target),
        ]

        print(f"Generating n={n}, d={d} → {target}")
        subprocess.run(cmd, check=True)

print("Done.")

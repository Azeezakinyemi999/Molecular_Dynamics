#!/usr/bin/env python3
"""Partial-Hessian vibrational frequency calculation via ASE + MACE."""
import json, os
import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator

STRUCTURE  = '/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/adsorption/phase2_h/results/h_atom_s_1_relaxed.lammps'
MACE_MODEL = '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mh-1.model'
OUTDIR     = '/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/vibrations/ni_bulk_test/hopa_s_1_IS'
DELTA      = 0.01
DEVICE     = 'cpu'

os.makedirs(OUTDIR, exist_ok=True)

# ── Load structure ────────────────────────────────────────────────────
atoms = read(STRUCTURE, format="lammps-data", atom_style="atomic")
atoms.wrap()
calc  = MACECalculator(
    model_paths=MACE_MODEL, device=DEVICE, default_dtype="float64", head="omat_pbe"
)
atoms.calc = calc

# ── Identify H and 6 nearest metal neighbours ─────────────────────────
syms      = np.array(atoms.get_chemical_symbols())
pos       = atoms.get_positions()
h_indices = np.where(syms == "H")[0]
if len(h_indices) != 1:
    raise RuntimeError(
        "Expected exactly 1 H atom, found " + str(len(h_indices))
    )
h_idx     = int(h_indices[0])
metal_idx = np.where(syms != "H")[0]
dists     = np.linalg.norm(pos[metal_idx] - pos[h_idx], axis=1)
nearest6  = metal_idx[np.argsort(dists)[:6]].tolist()
indices   = [h_idx] + nearest6
print("Displacing", len(indices), "atoms: H @", h_idx, "metals @", nearest6)

# ── Finite-difference vibrations ──────────────────────────────────────
vib_name = os.path.join(OUTDIR, "vib")
vib = Vibrations(atoms, indices=indices, delta=DELTA, name=vib_name)
vib.run()
vib.summary()

# ── Extract real and imaginary frequencies ────────────────────────────
# ASE returns complex cm^-1: imaginary modes have f.imag != 0
freqs_raw      = vib.get_frequencies()
freqs_real_cm1 = []
freqs_imag_cm1 = []
for f in freqs_raw:
    if abs(f.imag) > 1e-6:
        freqs_imag_cm1.append(float(abs(f.imag)))
    elif f.real > 0:
        freqs_real_cm1.append(float(f.real))
    else:
        # near-zero or slightly negative — treat as imaginary
        freqs_imag_cm1.append(float(abs(f.real)))

result = {
    "structure":            STRUCTURE,
    "n_atoms_displaced":    len(indices),
    "h_index":              h_idx,
    "metal_indices":        nearest6,
    "delta_ang":            DELTA,
    "frequencies_real_cm1": freqs_real_cm1,
    "frequencies_imag_cm1": freqs_imag_cm1,
}

out_json = os.path.join(OUTDIR, "vib_frequencies.json")
with open(out_json, "w") as fh:
    json.dump(result, fh, indent=2)
print("Wrote:", out_json)
print("Real modes (", len(freqs_real_cm1), "):", freqs_real_cm1[:5], "...")
print("Imag modes (", len(freqs_imag_cm1), "):", freqs_imag_cm1)

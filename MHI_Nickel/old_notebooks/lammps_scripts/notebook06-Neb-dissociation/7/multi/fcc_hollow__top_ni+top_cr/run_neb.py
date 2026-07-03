#!/usr/bin/env python3
# Multi-pathway NEB: fcc_hollow__top_ni+top_cr
label = "fcc_hollow__top_ni+top_cr"
# IS: H2* at fcc_hollow  (true label: Ni_atop_physisorbed)
# FS: 2H* at top_ni + top_cr
# FS true labels: MoMoNi_hollow_hcp + CrNiNi_hollow_hcp
# Graph distance between FS sites: -1

import os, sys, numpy as np, torch
from ase.io import read
from ase.mep import NEB
from ase.optimize import MDMin
from ase.calculators.singlepoint import SinglePointCalculator
from mace.calculators import MACECalculator

# Strip CUDA stubs from LD path — causes segfault on some nodes
_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in _ld.split(":") if "stubs" not in p)

MACE_MODEL      = "/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mp-0b2-medium.model"
NEB_IS_FILE     = "structures/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/neb_initial.lammps"
FS_RELAXED_DATA = "structures/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/neb_final_relaxed.lammps"
Z_FREEZE_CUTOFF = 22.115
E_IS            = -2398.19537413707
N_IMAGES        = 18
NEB_FMAX        = 0.05
N1_STEPS        = 5000
NEB_STEPS       = 10000
SPRING_CONST    = 2.0
N1_FMAX         = NEB_FMAX * 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sys.stdout.flush()

class MACEFrozenCalc(MACECalculator):
    def get_forces(self, atoms=None):
        forces = super().get_forces(atoms).copy()
        a = self.atoms if atoms is None else atoms
        forces[a.get_positions()[:, 2] < Z_FREEZE_CUTOFF] = 0.0
        return forces

def make_calc():
    """Fresh calculator instance for each intermediate image."""
    return MACEFrozenCalc(
        model_paths=MACE_MODEL,
        device=device,
        default_dtype="float64",
    )

# Read E_FS from FS minimization log
fs_log = "results/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/fs_min.log"
E_FS = None
with open(fs_log) as f:
    for line in f:
        if "pe_final_eV" in line and ":" in line:
            try: E_FS = float(line.split(":")[1].strip())
            except: pass
if E_FS is None:
    raise FileNotFoundError(f"FS log not found or no energy: {fs_log}")
print(f"E_IS = {E_IS:.6f} eV")
print(f"E_FS = {E_FS:.6f} eV")
print(f"dE   = {E_FS - E_IS:.4f} eV")
sys.stdout.flush()

print("Loading IS and FS...")
is_raw = read(NEB_IS_FILE,     format="lammps-data", style="atomic")
fs_raw = read(FS_RELAXED_DATA, format="lammps-data", style="atomic")

is_end = is_raw.copy()
is_end.calc = SinglePointCalculator(is_end)
is_end.calc.results = {"energy": E_IS, "forces": np.zeros((len(is_end), 3))}

fs_end = fs_raw.copy()
fs_end.calc = SinglePointCalculator(fs_end)
fs_end.calc.results = {"energy": E_FS, "forces": np.zeros((len(fs_end), 3))}

images = [is_end]
for _ in range(N_IMAGES):
    img = is_raw.copy()
    img.calc = make_calc()
    images.append(img)
images.append(fs_end)

print(f"Images: {len(images)} total ({N_IMAGES} intermediate + IS + FS)")
sys.stdout.flush()

neb = NEB(images, climb=False, k=SPRING_CONST, method="aseneb")
neb.interpolate(method="idpp")

print("IDPP done. H-H distances:")
for i, img in enumerate(images):
    lbl = "IS" if i==0 else ("FS" if i==len(images)-1 else f"img_{i}")
    p = img.get_positions()
    s = np.array(img.get_chemical_symbols())
    h = p[s=="H"]
    hh = np.linalg.norm(h[0]-h[1]) if len(h)==2 else 0.0
    print(f"  {lbl:6s}: H-H = {hh:.4f} A")
sys.stdout.flush()

print(f"\nPhase 1: regular NEB (5000 steps)")
sys.stdout.flush()
opt1 = MDMin(neb, logfile="results/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/neb_phase1.log", dt=0.05)
opt1.run(fmax=N1_FMAX, steps=N1_STEPS)
print(f"Phase 1 done: {opt1.nsteps} steps")
sys.stdout.flush()

for img in images[1:-1]:
    img.set_momenta(np.zeros_like(img.get_momenta()))

neb.climb = True
print(f"\nPhase 2: CINEB (10000 steps, fmax=0.050 eV/A)")
sys.stdout.flush()
opt2 = MDMin(neb, logfile="results/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/neb_phase2.log", dt=0.02)
try:
    opt2.run(fmax=NEB_FMAX, steps=NEB_STEPS)
except Exception as e:
    print(f"WARNING: phase 2 raised: {e}")
    sys.stdout.flush()
print(f"Phase 2 done: {opt2.nsteps} steps")
sys.stdout.flush()

with open("results/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/neb_phase2.log") as _f:
    _last = [l for l in _f if l.strip()][-1]
fmax_final = float(_last.split()[-1])
converged  = fmax_final < NEB_FMAX
print(f"fmax: {fmax_final:.4f} eV/A  Converged: {converged}")
sys.stdout.flush()

energies = []
for img in images:
    try:    energies.append(img.get_potential_energy())
    except: energies.append(None)

valid  = [e for e in energies if e is not None]
E_a    = max(valid) - E_IS
dE_rxn = E_FS - E_IS

print(f"\nNEB Results: {label}")
print(f"  E_IS    : {E_IS:.6f} eV")
print(f"  E_FS    : {E_FS:.6f} eV")
print(f"  E_a     : {E_a:.4f} eV")
print(f"  delta_E : {dE_rxn:.4f} eV")
print(f"  fmax    : {fmax_final:.4f} eV/A")
print(f"  Converged: {converged}")
sys.stdout.flush()

barrier_file = "results/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/neb_barrier.txt"
path_file    = "results/notebook06-Neb-dissociation/7/multi/fcc_hollow__top_ni+top_cr/neb_path.dat"

with open(barrier_file, "w", encoding="utf-8") as f:
    f.write("IS : H2* at fcc_hollow\n")
    f.write("IS true label : Ni_atop_physisorbed\n")
    f.write("FS : 2H* at top_ni + top_cr\n")
    f.write("FS true label 1 : MoMoNi_hollow_hcp\n")
    f.write("FS true label 2 : CrNiNi_hollow_hcp\n")
    f.write("Graph distance  : -1\n")
    f.write(f"fmax_final  : {fmax_final:.4f} eV/A\n")
    f.write(f"Converged   : {converged}\n\n")
    f.write(f"E_IS    = {E_IS:.6f} eV\n")
    f.write(f"E_FS    = {E_FS:.6f} eV\n")
    f.write(f"E_a     = {E_a:.4f} eV\n")
    f.write(f"delta_E = {dE_rxn:.4f} eV\n\n")
    f.write("Image energies:\n")
    for i, e in enumerate(energies):
        lbl = "IS" if i==0 else ("FS" if i==len(energies)-1 else f"img_{i}")
        if e is not None:
            f.write(f"  {lbl:6s}: {e:.6f} eV  ({e-E_IS:+.4f} eV)\n")

with open(path_file, "w", encoding="utf-8") as f:
    f.write("# image  E(eV)  dE_from_IS(eV)\n")
    n = len(energies)
    for i, e in enumerate(energies):
        if e is not None:
            f.write(f"{i/(n-1):.4f}  {e:.6f}  {e-E_IS:+.4f}\n")

print("Saved: " + barrier_file)
print("Saved: " + path_file)

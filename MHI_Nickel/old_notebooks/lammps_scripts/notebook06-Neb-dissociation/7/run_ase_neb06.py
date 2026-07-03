#!/usr/bin/env python3
# ASE CINEB: H2 dissociation on Hastelloy N (111) — Notebook 06
# IS: H2* at ads_top_b.log
# FS: 2H* at top_mo + top_ni
# Interpolation: IDPP [4] — avoids atom clashes during dissociation
# Two-phase: regular NEB (phase 1) then CINEB (phase 2)
# Refs: NEB [1,2]; CINEB [3]; IDPP [4]; MACE [7]

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

# ── Paths and parameters injected from notebook ──────────
MACE_MODEL      = "/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mp-0b2-medium.model"
NEB_IS_FILE     = "structures/notebook06-Neb-dissociation/7/neb_initial.lammps"
FS_RELAXED_DATA = "structures/notebook06-Neb-dissociation/7/neb_final_relaxed.lammps"
Z_FREEZE_CUTOFF = 22.115  # A — bottom layers frozen
E_IS            = -2398.20109933816             # eV — from NB05 minimization
E_FS            = -2399.4497586065             # eV — from NB06 FS minimization
N_IMAGES        = 18       # intermediate images
NEB_FMAX        = 0.05         # eV/A — CINEB convergence
N1_STEPS        = 5000         # phase 1 steps
NEB_STEPS       = 10000        # phase 2 CINEB steps
SPRING_CONST    = 2.0     # elastic band, to keep image together
N1_FMAX         = NEB_FMAX * 3       # looser fmax for phase 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sys.stdout.flush()

# ── Freeze bottom slab layers — H atoms are above cutoff ─
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

# ── Load IS and FS structures ─────────────────────────────
print("Loading IS and FS...")
is_raw = read(NEB_IS_FILE,     format="lammps-data", style="atomic")
fs_raw = read(FS_RELAXED_DATA, format="lammps-data", style="atomic")

# Endpoints use SinglePointCalculator — energies pinned, no re-evaluation
is_end = is_raw.copy()
is_end.calc = SinglePointCalculator(is_end)
is_end.calc.results = {"energy": E_IS, "forces": np.zeros((len(is_end), 3))}

fs_end = fs_raw.copy()
fs_end.calc = SinglePointCalculator(fs_end)
fs_end.calc.results = {"energy": E_FS, "forces": np.zeros((len(fs_end), 3))}

# ── Build image chain — intermediates start from IS geometry
images = [is_end]
for _ in range(N_IMAGES):
    img = is_raw.copy()
    img.calc = make_calc()
    images.append(img)
images.append(fs_end)

print(f"Images: {len(images)} total ({N_IMAGES} intermediate + IS + FS)")
sys.stdout.flush()

# ── IDPP interpolation — threads H atoms apart above surface
neb = NEB(images, climb=False, k=1.0, method="aseneb")
neb.interpolate(method="idpp")

print("IDPP interpolation done. H-H distances along path:")
for i, img in enumerate(images):
    lbl = "IS" if i == 0 else ("FS" if i == len(images)-1 else f"img_{i}")
    p = img.get_positions()
    s = np.array(img.get_chemical_symbols())
    h = p[s == "H"]
    hh = np.linalg.norm(h[0]-h[1]) if len(h) == 2 else 0.0
    print(f"  {lbl:6s}: H-H = {hh:.4f} A")
sys.stdout.flush()

# ── Phase 1: regular NEB — moves images onto MEP ─────────
print(f"\nPhase 1: regular NEB ({N1_STEPS} steps, fmax={N1_FMAX:.3f} eV/A)")
sys.stdout.flush()
opt1 = MDMin(neb, logfile="results/notebook06-Neb-dissociation/7/neb06_phase1.log", dt=0.05)
opt1.run(fmax=N1_FMAX, steps=N1_STEPS)
print(f"Phase 1 done: {opt1.nsteps} steps")
sys.stdout.flush()

# Reset momenta before CINEB — prevents phase 1 velocity carryover
for img in images[1:-1]:
    img.set_momenta(np.zeros_like(img.get_momenta()))

# ── Phase 2: CINEB — pushes highest image to saddle point ─
neb.climb = True
print(f"\nPhase 2: CINEB ({NEB_STEPS} steps, fmax={NEB_FMAX:.3f} eV/A)")
sys.stdout.flush()
opt2 = MDMin(neb, logfile="results/notebook06-Neb-dissociation/7/neb06_phase2.log", dt=0.02)
try:
    opt2.run(fmax=NEB_FMAX, steps=NEB_STEPS)
except Exception as e:
    print(f"WARNING: phase 2 raised: {e}")
    sys.stdout.flush()
print(f"Phase 2 done: {opt2.nsteps} steps")
sys.stdout.flush()

# Read true fmax from phase 2 log — avoids NEB-projected force artifact
with open("results/notebook06-Neb-dissociation/7/neb06_phase2.log", "r") as _f:
    _last = [l for l in _f if l.strip()][-1]
fmax_final = float(_last.split()[-1])
converged  = fmax_final < NEB_FMAX
print(f"fmax from log: {fmax_final:.4f} eV/A  Converged: {converged}")
sys.stdout.flush()

# ── Extract barrier and reaction energy ───────────────────
energies = []
for img in images:
    try:    energies.append(img.get_potential_energy())
    except: energies.append(None)

valid   = [e for e in energies if e is not None]
E_a     = max(valid) - E_IS
dE      = E_FS - E_IS

print(f"\nNEB06 Results")
print(f"  E_IS     : {E_IS:.6f} eV")
print(f"  E_FS     : {E_FS:.6f} eV")
print(f"  E_a      : {E_a:.4f} eV")
print(f"  delta_E  : {dE:.4f} eV")
print(f"  fmax     : {fmax_final:.4f} eV/A")
print(f"  Converged: {converged}")
sys.stdout.flush()

# ── Write intermediate image structures ──────────────────
from ase.io import write as ase_write
img_dir = "structures/notebook06-Neb-dissociation/7/neb_images"
os.makedirs(img_dir, exist_ok=True)

for i, img in enumerate(images[1:-1], start=1):
    img.info["energy_eV"]  = energies[i]
    img.info["dE_from_IS"] = energies[i] - E_IS
    img.info["image_index"] = i
    fpath = os.path.join(img_dir, f"neb_img_{i:02d}.lammps")
    ase_write(fpath, img, format="lammps-data")

# Full trajectory (IS + intermediates + FS) for OVITO
traj_path = "results/notebook06-Neb-dissociation/7/neb_path.extxyz"
ase_write(traj_path, images, format="extxyz")

print(f"Written {len(images)-2} intermediate structures to {img_dir}")
print(f"Trajectory written to {traj_path}")
sys.stdout.flush()
# ── Write output files ────────────────────────────────────
barrier_file = "results/notebook06-Neb-dissociation/7/neb_barrier.txt"
path_file    = "results/notebook06-Neb-dissociation/7/neb_path.dat"

with open(barrier_file, "w", encoding="utf-8") as f:
    f.write("IS : H2* at ads_top_b.log\n")
    f.write("FS : 2H* at top_mo + top_ni\n")
    f.write("Interpolation: IDPP\n")
    f.write(f"N images    : {len(images)} ({N_IMAGES} intermediate + IS + FS)\n")
    f.write(f"fmax_final  : {fmax_final:.4f} eV/A\n")
    f.write(f"Converged   : {converged}\n\n")
    f.write(f"E_IS    = {E_IS:.6f} eV\n")
    f.write(f"E_FS    = {E_FS:.6f} eV\n")
    f.write(f"E_a     = {E_a:.4f} eV\n")
    f.write(f"delta_E = {dE:.4f} eV\n\n")
    f.write("Image energies:\n")
    for i, e in enumerate(energies):
        lbl = "IS" if i == 0 else ("FS" if i == len(energies)-1 else f"img_{i}")
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

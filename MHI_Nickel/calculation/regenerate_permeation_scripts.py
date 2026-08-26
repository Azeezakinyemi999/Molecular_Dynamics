#!/usr/bin/env python3
"""
regenerate_permeation_scripts.py
================================
Standalone regeneration of permeation_run_{stem}.py for every metal, so the
generated orchestrators pick up the reframed permeation pipeline (dissociation-
seeded subsurface entry, two-layer KMC, per-environment solubility, Arrhenius
outputs). Mirrors regenerate_neb_scripts.py's per-metal pattern and the config
embedded in the currently-deployed permeation_run_*.py headers (read off
Hastelloy_N_1234's header), written as a standalone script instead of going
through pipeline.ipynb so it can't accidentally trigger the notebook's later
submission cells.

ONLY calls generate_permeation_scripts() per metal (a pure file-write). It does
not submit anything and does not touch neb_run/diffusivity_run/pipeline_run.
Safe to re-run: generate_permeation_scripts always overwrites its own output
deterministically from the same inputs.

Config faithfully mirrors production (40x40 KMC grid, 500k steps, 40-point log
pressure sweep). Edit the numeric block below if you want a lighter validation
run before committing to the full production grid.

Usage
-----
    python regenerate_permeation_scripts.py

After this, wrap + submit with wrap_permeation_runs_west.py (a metal at a time).
DH_DISS_EV / DH_ENTRY_EV are left None: the orchestrator auto-extracts them from
the metal's dissociation ranked_barriers.json + Hop A rate dict at run time.
"""
import os
import re as _re
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import (
    SLURM_DEFAULTS, BASE_DIR, N_REPLICAS, SPRING_CONST, NEB_FTOL,
    ELEM_STR_7, E2T_7, MASSES_7, ELEM_STR_10, E2T_10, MASSES_10,
)
from models.permeation_workflow import generate_permeation_scripts
from models.structure import is_pure_bcc_structure

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

# ── numeric config (mirrors the deployed permeation_run_*.py headers) ────────
TEMPERATURES  = [400, 600, 800]
N_H_VALUES    = [1, 3, 5, 10]
OPERATING_P_HIGH_PA = 1.0e6   # feed-side H2 [Pa]; was max of the KMC sweep grid
OPERATING_P_LOW_PA  = 0.0     # permeate side, fully swept [Pa]
A0_M          = 3.52e-10
L_M           = 1e-3
NX            = 40
NY            = 40
SEED          = 42
KMC_MAX_STEPS = 500000
N_IMAGES      = N_REPLICAS
SPRING_K      = SPRING_CONST
NEB_FTOL_VAL  = NEB_FTOL

GPU_SLURM = dict(SLURM_DEFAULTS, partition='sharing', gpu='a100:1',
                 cpus_per_task=8,  time='01:00:00')
NEB_SLURM = dict(SLURM_DEFAULTS, partition='short', gpu=None,
                 cpus_per_task=16, time='12:00:00')
VIB_SLURM = dict(SLURM_DEFAULTS, partition='short', gpu=None,
                 cpus_per_task=8,  time='06:00:00')

INPUT_STRUCTURES = [
    os.path.join(WORK_DIR, 'input_structure/Hastelloy_N_7_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Cr_oxide_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Hastelloy_N_42_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Hastelloy_N_111_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Hastelloy_N_1234_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Hastelloy_N_12345_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Al_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Fe_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/Ni_supercell.lammps'),
    os.path.join(WORK_DIR, 'input_structure/bestsqs3.lmp'),
    os.path.join(WORK_DIR, 'input_structure/Ni_oxide_supercell.lammps'),
]


def classify_metal(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if 'oxide' in stem:
        return 'oxide'
    if any(k in stem for k in ('hastelloy', 'bestsqs', 'sqs', 'alloy')):
        return 'alloy'
    return 'pure'


# Same skip rule as regenerate_neb_scripts.py: no surface NEB was produced for
# these, so there is no upstream for permeation either.
SKIP_OXIDE_STEMS = {'Ni_oxide_supercell'}

perm_scripts = {}
for _struct_path in INPUT_STRUCTURES:
    stem   = os.path.splitext(os.path.basename(_struct_path))[0]
    mtype  = classify_metal(_struct_path)

    skip_reason = None
    if stem in SKIP_OXIDE_STEMS:
        skip_reason = 'polar oxide termination (GitHub #5)'
    elif mtype == 'pure' and is_pure_bcc_structure(_struct_path):
        skip_reason = 'BCC surface/subsurface untested (GitHub #6)'
    if skip_reason:
        print(f'  [SKIP] {stem}: {skip_reason}')
        continue

    _out = os.path.join(WORK_DIR, f'permeation_run_{stem}.py')
    generate_permeation_scripts(
        work_dir          = WORK_DIR,
        stem              = stem,
        relaxed_slab_path = os.path.join(WORK_DIR, f'slabs/{stem}/phase2_relax/relaxed_slab.lammps'),
        surface_sites_json= os.path.join(WORK_DIR, f'slabs/{stem}/phase3_sites/surface_sites.json'),
        phase2_h_dir      = os.path.join(WORK_DIR, f'adsorption/{stem}/phase2_h/results'),
        sub_neb_dir       = os.path.join(WORK_DIR, f'neb_subsurface/{stem}'),
        vib_dir           = os.path.join(WORK_DIR, f'vibrations/{stem}'),
        results_dir       = os.path.join(WORK_DIR, f'results/{stem}'),
        temperatures      = TEMPERATURES,
        n_h_values        = N_H_VALUES,
        operating_p_high_pa = OPERATING_P_HIGH_PA,
        operating_p_low_pa  = OPERATING_P_LOW_PA,
        a0_m              = A0_M,
        l_m               = L_M,
        dh_diss_ev        = None,     # auto-extracted at run time
        dh_entry_ev       = None,     # auto-extracted at run time
        nx                = NX,
        ny                = NY,
        seed              = SEED,
        kmc_max_steps     = KMC_MAX_STEPS,
        gpu_slurm_cfg     = GPU_SLURM,
        neb_slurm_cfg     = NEB_SLURM,
        vib_slurm_cfg     = VIB_SLURM,
        n_images          = N_IMAGES,
        spring_const      = SPRING_K,
        neb_ftol          = NEB_FTOL_VAL,
        out_py            = _out,
        elem_str          = ELEM_STR_10 if mtype == 'oxide' else ELEM_STR_7,
        e2t               = E2T_10      if mtype == 'oxide' else E2T_7,
        masses            = MASSES_10   if mtype == 'oxide' else MASSES_7,
        metal_type        = mtype,
    )
    perm_scripts[stem] = _out
    print(f'Written: {_out}  (metal_type={mtype!r})')

print(f'\n{len(perm_scripts)} permeation script(s) regenerated. Nothing was submitted.')
print('Next: wrap + submit a metal whose Part 1 (surface NEB) and Part 3 '
      '(diffusivity) are complete:')
print('  python wrap_permeation_runs_west.py')

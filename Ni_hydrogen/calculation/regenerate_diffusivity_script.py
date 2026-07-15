#!/usr/bin/env python3
"""
regenerate_diffusivity_script.py
=================================
Standalone regeneration of diffusivity_run.py, mirroring the exact config
currently embedded in the deployed diffusivity_run.py (read directly off
its own header block on 2026-07-15) -- written as a standalone script
instead of going through pipeline.ipynb, specifically to avoid touching
that notebook's later cells (Part 2 permeation generation, Part B
pipeline_run.py generation/submission) while a NEB run is still active
on the cluster.

Why this is needed now: models/diffusivity_workflow.py gained two fixes
since diffusivity_run.py was last generated --
  1. Phase 1a (bare-bulk min) and Phase 1b-B (bulk+H min) now submit via
     write_chained_slurm_job with periodic LAMMPS restart checkpoints,
     so a wall-time timeout self-resubmits instead of failing outright
     (previously a one-shot write_slurm_job with no recovery path).
  2. generate_diffusivity_scripts()'s short_gpu_time default was raised
     from '00:20:00' to '01:00:00' to match what was already deployed
     (this script passes it explicitly anyway, so it's frozen against
     any future default drift regardless).
Neither fix is live until diffusivity_run.py is regenerated -- this
script does that, changing nothing else. All other config values
(INPUT_STRUCTURES, N_H/T grids, NVT/NPT wall times, MD step counts) are
copied verbatim from the currently-deployed file so this is a pure
"pick up the fixes" regeneration, not a config change.

Only WRITES diffusivity_run.py -- does not touch diffusivity_run.sh,
does not submit anything. Safe to re-run: generate_diffusivity_scripts
always overwrites its own output deterministically from the same inputs.

Usage
-----
    python regenerate_diffusivity_script.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import (
    ELEM_STR_7, E2T_7, MASSES_7, ELEM_STR_10, E2T_10, MASSES_10,
    BASE_DIR,
)
from models.diffusivity_workflow import generate_diffusivity_scripts

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

# ── Verbatim from the currently-deployed diffusivity_run.py's header ────────
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
N_H_VALUES   = [1, 3, 5, 10]
TEMPERATURES = [400, 600, 800]

NVT_WALL_TIME = '24:00:00'
CUTOFF        = '23:55:00'
GPU_PARTITION = 'multigpu'
GPU_TIME      = '24:00:00'

SHORT_GPU_PARTITION = 'sharing'
SHORT_GPU_TIME      = '01:00:00'
SHORT_GPU_CUTOFF    = '00:55:00'
NPT_GPU_PARTITION   = 'gpu'
NPT_GPU_TIME        = '08:00:00'
NPT_GPU_CUTOFF      = '07:55:00'

TIMESTEP_PS   = 0.0005
TAU_T_PS      = 0.1
N_EQUIL_STEPS = 2_000_000
N_PROD_STEPS  = 5_000_000
THERMO_EVERY  = 1000
DUMP_EVERY    = 1000
VELOCITY_SEED = 42
RESTART_EVERY = 100_000

NPT_HEAT_STEPS = 100_000
NPT_PROD_STEPS = 500_000
NPT_BARO_DAMP  = 1.0
NPT_DUMP_EVERY = 1000

MIN_ETOL    = 0.0
MIN_FTOL    = 1e-8
MIN_MAXITER = 50000
MIN_MAXEVAL = 500000

# NEW this run -- did not exist when diffusivity_run.py was last generated.
MIN_RESTART_EVERY = 2000

# ── Metal classification -- same rule as regenerate_neb_scripts.py's
# classify_metal(), which is what produced the deployed file's METAL_TABLE
# (oxide stems get the 9-type O-inclusive table, everything else the
# 8-type table) ────────────────────────────────────────────────────────────
def classify_metal(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return 'oxide' if 'oxide' in stem else 'alloy'

METAL_TABLE = {}
for _struct_path in INPUT_STRUCTURES:
    _stem  = os.path.splitext(os.path.basename(_struct_path))[0]
    _mtype = classify_metal(_struct_path)
    METAL_TABLE[_stem] = {
        'elem_str': ELEM_STR_10 if _mtype == 'oxide' else ELEM_STR_7,
        'e2t':      E2T_10      if _mtype == 'oxide' else E2T_7,
        'masses':   MASSES_10   if _mtype == 'oxide' else MASSES_7,
    }

out_py = generate_diffusivity_scripts(
    input_structures=INPUT_STRUCTURES,
    n_h_values=N_H_VALUES,
    temperatures=TEMPERATURES,
    work_dir=WORK_DIR,
    nvt_wall_time=NVT_WALL_TIME,
    cutoff=CUTOFF,
    gpu_partition=GPU_PARTITION,
    gpu_time=GPU_TIME,
    timestep_ps=TIMESTEP_PS,
    tau_t_ps=TAU_T_PS,
    n_equil_steps=N_EQUIL_STEPS,
    n_prod_steps=N_PROD_STEPS,
    thermo_every=THERMO_EVERY,
    dump_every=DUMP_EVERY,
    velocity_seed=VELOCITY_SEED,
    restart_every=RESTART_EVERY,
    short_gpu_partition=SHORT_GPU_PARTITION,
    short_gpu_time=SHORT_GPU_TIME,
    short_gpu_cutoff=SHORT_GPU_CUTOFF,
    npt_gpu_partition=NPT_GPU_PARTITION,
    npt_gpu_time=NPT_GPU_TIME,
    npt_gpu_cutoff=NPT_GPU_CUTOFF,
    npt_heat_steps=NPT_HEAT_STEPS,
    npt_prod_steps=NPT_PROD_STEPS,
    npt_baro_damp=NPT_BARO_DAMP,
    npt_dump_every=NPT_DUMP_EVERY,
    min_etol=MIN_ETOL,
    min_ftol=MIN_FTOL,
    min_maxiter=MIN_MAXITER,
    min_maxeval=MIN_MAXEVAL,
    min_restart_every=MIN_RESTART_EVERY,
    metal_table=METAL_TABLE,
    out_py=os.path.join(WORK_DIR, 'diffusivity_run.py'),
)
print(f'Written: {out_py}')

#!/usr/bin/env python3
"""
regenerate_diffusivity_scripts.py
===================================
Per-metal counterpart to regenerate_diffusivity_script.py: writes
diffusivity_run_{stem}.py for each metal/structure separately, via
write_diffusivity_run_script() (models/diffusivity_workflow.py, added in
c7f03ca), instead of one combined diffusivity_run.py that walks every
structure serially in a single process. Mirrors regenerate_neb_scripts.py's
per-metal loop, so diffusivity can run as N independent concurrent
orchestrator processes -- same as neb_run_{stem}.py / permeation_run_{stem}.py.

generate_diffusivity_scripts() (the combined-script path) and the existing
diffusivity_run.py are untouched by this -- this only adds the per-metal
alternative. pipeline_run.py keeps using the combined script exactly as
today; wiring this per-metal path into the orchestrator is a separate,
later step (same caveat write_diffusivity_run_script()'s own docstring
already states).

All config values (N_H_VALUES, TEMPERATURES, wall times, MD step counts)
are copied verbatim from regenerate_diffusivity_script.py, so this is a
pure "per-metal instead of combined" alternative, not a config change.

Only WRITES diffusivity_run_{stem}.py per metal -- does not submit
anything. Safe to re-run: write_diffusivity_run_script always overwrites
its own output deterministically from the same inputs.

Usage
-----
    python regenerate_diffusivity_scripts.py

After this, wrap each script in its own trackable SLURM job with
wrap_diffusivity_runs_west.py (mirrors wrap_neb_runs_west.py), rather than
launching bare nohup processes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import (
    ELEM_STR_7, E2T_7, MASSES_7, ELEM_STR_10, E2T_10, MASSES_10,
    BASE_DIR,
)
from models.diffusivity_workflow import write_diffusivity_run_script

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

# ── Verbatim from regenerate_diffusivity_script.py ───────────────────────────
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

MIN_ETOL          = 0.0
MIN_FTOL          = 1e-8
MIN_MAXITER       = 50000
MIN_MAXEVAL       = 500000
MIN_RESTART_EVERY = 2000

# ── Metal classification -- same rule as regenerate_diffusivity_script.py's
# classify_metal() (oxide stems get the 10-type O-inclusive table, everything
# else the 7-type table). Diffusivity has no surface/slab step, so unlike
# regenerate_neb_scripts.py's classify_metal() there is no 'pure' vs 'alloy'
# distinction and no BCC/polar-oxide skip logic -- every structure below
# gets a script. ─────────────────────────────────────────────────────────────
def classify_metal(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return 'oxide' if 'oxide' in stem else 'alloy'

diffusivity_scripts = {}
for _struct_path in INPUT_STRUCTURES:
    _stem  = os.path.splitext(os.path.basename(_struct_path))[0]
    _mtype = classify_metal(_struct_path)
    _out = os.path.join(WORK_DIR, f'diffusivity_run_{_stem}.py')
    write_diffusivity_run_script(
        struct_path         = _struct_path,
        stem                = _stem,
        n_h_values          = N_H_VALUES,
        temperatures        = TEMPERATURES,
        work_dir            = WORK_DIR,
        nvt_wall_time       = NVT_WALL_TIME,
        cutoff              = CUTOFF,
        gpu_partition       = GPU_PARTITION,
        gpu_time            = GPU_TIME,
        timestep_ps         = TIMESTEP_PS,
        tau_t_ps            = TAU_T_PS,
        n_equil_steps       = N_EQUIL_STEPS,
        n_prod_steps        = N_PROD_STEPS,
        thermo_every        = THERMO_EVERY,
        dump_every          = DUMP_EVERY,
        velocity_seed       = VELOCITY_SEED,
        restart_every       = RESTART_EVERY,
        short_gpu_partition = SHORT_GPU_PARTITION,
        short_gpu_time      = SHORT_GPU_TIME,
        short_gpu_cutoff    = SHORT_GPU_CUTOFF,
        npt_gpu_partition   = NPT_GPU_PARTITION,
        npt_gpu_time        = NPT_GPU_TIME,
        npt_gpu_cutoff      = NPT_GPU_CUTOFF,
        npt_heat_steps      = NPT_HEAT_STEPS,
        npt_prod_steps      = NPT_PROD_STEPS,
        npt_baro_damp       = NPT_BARO_DAMP,
        npt_dump_every      = NPT_DUMP_EVERY,
        min_etol            = MIN_ETOL,
        min_ftol            = MIN_FTOL,
        min_maxiter         = MIN_MAXITER,
        min_maxeval         = MIN_MAXEVAL,
        min_restart_every   = MIN_RESTART_EVERY,
        elem_str            = ELEM_STR_10 if _mtype == 'oxide' else ELEM_STR_7,
        e2t                 = E2T_10      if _mtype == 'oxide' else E2T_7,
        masses              = MASSES_10   if _mtype == 'oxide' else MASSES_7,
        out_py              = _out,
    )
    diffusivity_scripts[_stem] = _out
    print(f'Written: {_out}  (metal_type={_mtype!r})')

print(f'\n{len(diffusivity_scripts)} script(s) regenerated. Nothing was submitted.')
print('Next: wrap each in its own trackable SLURM job:')
print('  python wrap_diffusivity_runs_west.py')

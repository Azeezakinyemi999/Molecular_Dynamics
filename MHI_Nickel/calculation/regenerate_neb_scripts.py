#!/usr/bin/env python3
"""
regenerate_neb_scripts.py
==========================
Standalone regeneration of neb_run_{stem}.py for every metal, mirroring
pipeline.ipynb's cells 2 (imports), 5 (config/METAL_CONFIGS/E_H2_GAS), and
8 (the write_neb_run_script loop) exactly -- written as a standalone script
instead of asking for interactive notebook cell execution, specifically to
avoid any risk of accidentally running a cell past 8 (e.g. cell 15, which
would touch pipeline_run.py's own generation/submission).

This script ONLY calls write_neb_run_script() per metal (a pure file-write,
confirmed no submission happens inside it) and reads the already-cached H2
reference energy (does not resubmit that job). It does not touch
diffusivity_run.py, permeation, or pipeline_run.py in any way.

Safe to re-run: write_neb_run_script always overwrites its own output
deterministically from the same inputs.

Usage
-----
    python regenerate_neb_scripts.py

After this, launch each regenerated script manually (a metal at a time, or
all at once, in the background) -- do NOT go through pipeline_run.py again,
since it already used its one shot at launching these and won't do so a
second time (confirmed: it never re-invokes a metal's neb_run_{stem}.py
after its initial subprocess.Popen).
"""
import os
import re as _re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import (
    PAIR_STYLE, MACE_MODEL_LAMMPS, PAIR_SUFFIX, LAMMPS_CMD, KOKKOS_FLAGS,
    ELEM_STR_7, E2T_7, MASSES_7, ELEM_STR_10, E2T_10, MASSES_10,
    SLURM_DEFAULTS, BASE_DIR, N_REPLICAS, SPRING_CONST, NEB_FTOL,
)
from models.neb_workflow import write_neb_run_script, calculate_ref_adsorbate_energy
from models.structure import is_pure_bcc_structure

# ── cell 5 (Part 1 + metal classification + E_H2_GAS) ───────────────────────
WORK_DIR = os.path.join(BASE_DIR, 'calculation')

MILLER         = (1, 1, 1)
LAYERS         = 12
VACUUM         = 15.0
LAT_REPEAT     = (5, 6)
SEP_MIN        = 2.5
SEP_MAX        = 5.0
GRAPH_DIST_MIN = 2
PROX_CUTOFF    = 3.0
N_IMAGES       = N_REPLICAS
SPRING_K       = SPRING_CONST
NEB_FTOL_VAL   = NEB_FTOL
H_HEIGHT       = 1.5
Z_FREEZE_CUTOFF  = None
SURF_TIMESTEP_PS = 0.0005

NEB_GPU_SLURM = dict(SLURM_DEFAULTS, partition='gpu', time='08:00:00')
MIN_SLURM     = dict(SLURM_DEFAULTS, partition='sharing', time='01:00:00')
NEB_CPU_SLURM = dict(SLURM_DEFAULTS, partition='short',
                     gpu=None, cpus_per_task=18, time='12:00:00')
NEB_VIB_SLURM = dict(SLURM_DEFAULTS, partition='short',
                     gpu=None, cpus_per_task=8,  time='06:00:00')

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

METAL_CONFIGS = []
for _struct_path in INPUT_STRUCTURES:
    _stem  = os.path.splitext(os.path.basename(_struct_path))[0]
    _mtype = classify_metal(_struct_path)
    METAL_CONFIGS.append({
        'struct_path': _struct_path,
        'stem':        _stem,
        'type':        _mtype,
        'elem_str':    ELEM_STR_10 if _mtype == 'oxide' else ELEM_STR_7,
        'e2t':         E2T_10      if _mtype == 'oxide' else E2T_7,
        'masses':      MASSES_10   if _mtype == 'oxide' else MASSES_7,
    })

SKIP_OXIDE_STEMS = {'Ni_oxide_supercell'}
for _cfg in METAL_CONFIGS:
    _skip_reason = None
    if _cfg['stem'] in SKIP_OXIDE_STEMS:
        _skip_reason = 'polar oxide termination (GitHub #5)'
    elif _cfg['type'] == 'pure' and is_pure_bcc_structure(_cfg['struct_path']):
        _skip_reason = 'BCC surface/subsurface untested (GitHub #6)'
    _cfg['skip_surface'] = _skip_reason is not None
    _cfg['skip_reason']  = _skip_reason

# dry_run=True here only means "don't submit if the cache is missing" --
# production has been running for days, so this must read the existing
# cache, not attempt a new submission. Hard-stop below if that assumption
# is wrong, rather than silently propagating E_H2_GAS=None into every
# metal's regenerated script.
E_H2_GAS = calculate_ref_adsorbate_energy(
    adsorbate='H2', outdir=os.path.join(WORK_DIR, 'adsorption', 'ref_energies'),
    pair_style=PAIR_STYLE, mace_model=MACE_MODEL_LAMMPS, pair_suffix=PAIR_SUFFIX,
    elem_str=ELEM_STR_7, e2t=E2T_7, masses=MASSES_7,
    lammps_cmd=LAMMPS_CMD, kokkos_flags=KOKKOS_FLAGS,
    slurm_opts=MIN_SLURM, dry_run=True,
)
if E_H2_GAS is None:
    print('FATAL: E_H2_GAS came back None -- the H2 reference cache is '
          'missing or unreadable. Refusing to regenerate scripts with a '
          'missing reference energy. Investigate before re-running.')
    sys.exit(1)
print(f'E_H2_GAS = {E_H2_GAS} eV (read from cache)')
print(f'{len(METAL_CONFIGS)} metals loaded.\n')

# ── cell 8 (regeneration loop) ───────────────────────────────────────────────
neb_scripts = {}
for cfg in METAL_CONFIGS:
    stem = cfg['stem']
    if cfg.get('skip_surface'):
        print(f'  [SKIP] {stem}: {cfg["skip_reason"]}')
        continue
    _out = os.path.join(WORK_DIR, f'neb_run_{stem}.py')
    _seed_m   = _re.search(r'(\d+)', stem)
    slab_seed = int(_seed_m.group(1)) if _seed_m else 7
    write_neb_run_script(
        bulk_min_path   = os.path.join(WORK_DIR, f'structures/bulk_min_{stem}.lammps'),
        work_dir        = WORK_DIR,
        e_h2_gas        = E_H2_GAS,
        slab_dir        = os.path.join(WORK_DIR, 'slabs', stem),
        ads_dir         = os.path.join(WORK_DIR, 'adsorption', stem),
        neb_dir         = os.path.join(WORK_DIR, 'neb', stem),
        miller          = MILLER,
        layers          = LAYERS,
        vacuum          = VACUUM,
        lat_repeat      = LAT_REPEAT,
        z_freeze_cutoff = Z_FREEZE_CUTOFF,
        surf_timestep   = SURF_TIMESTEP_PS,
        sep_min         = SEP_MIN,
        sep_max         = SEP_MAX,
        graph_dist_min  = GRAPH_DIST_MIN,
        prox_cutoff     = PROX_CUTOFF,
        n_images        = N_IMAGES,
        spring_const    = SPRING_K,
        neb_ftol        = NEB_FTOL_VAL,
        h_height        = H_HEIGHT,
        elem_str        = cfg['elem_str'],
        e2t             = cfg['e2t'],
        masses          = cfg['masses'],
        gpu_slurm_cfg   = NEB_GPU_SLURM,
        min_slurm_cfg   = MIN_SLURM,
        neb_slurm_cfg   = NEB_CPU_SLURM,
        vib_slurm_cfg   = NEB_VIB_SLURM,
        out_py          = _out,
        metal_type      = cfg['type'],
        slab_seed       = slab_seed,
    )
    neb_scripts[stem] = _out
    print(f'Written: {_out}  (metal_type={cfg["type"]!r}  slab_seed={slab_seed})')

print(f'\n{len(neb_scripts)} script(s) regenerated. Nothing was submitted.')
print('Next: launch each one manually, e.g.:')
for stem, path in neb_scripts.items():
    print(f'  nohup python {path} > {WORK_DIR}/neb_run_{stem}.log 2>&1 &')

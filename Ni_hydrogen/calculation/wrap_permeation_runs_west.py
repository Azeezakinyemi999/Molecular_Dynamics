#!/usr/bin/env python3
"""
wrap_permeation_runs_west.py
============================
Wraps each already-regenerated permeation_run_{stem}.py (from
regenerate_permeation_scripts.py) in its own small, trackable SLURM job on the
`west` partition -- the same partition/resource shape pipeline_orch uses
(cpus_per_task=4, 30-day wall time), since permeation_run.py is a long-lived
orchestrator that submits Hop A/B FS-min + NEB arrays, vibration jobs, and runs
the KMC sweeps, waiting on each. Mirrors wrap_neb_runs_west.py exactly.

Only WRITES the .sh wrapper files -- does not submit anything. Prints the
sbatch commands to run manually, mirroring the generate-then-you-submit pattern
used across this pipeline.

Only submit a metal whose upstream is complete: Part 1 (surface/dissociation
NEB -> slabs/{stem}/phase3_sites/surface_sites.json, neb/{stem}/ranked_barriers.json,
adsorption/{stem}/phase2_h/results/h_atom_*) and Part 3 (diffusivity ->
results/{stem}_{n_h}H/diffusivity_arrhenius.json, results/lattice_params_vs_T.json).
Use check_permeation_maps.py first to confirm readiness.

Usage
-----
    python wrap_permeation_runs_west.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import BASE_DIR, SLURM_DEFAULTS
from models.create_slurm import write_slurm_job

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

# Matches pipeline.ipynb cell 5's PIPE_* literals exactly.
WEST_SLURM = dict(
    SLURM_DEFAULTS,
    partition='west',
    gpu=None,
    cpus_per_task=4,
    time='30-00:00:00',
)

perm_run_scripts = sorted(glob.glob(os.path.join(WORK_DIR, 'permeation_run_*.py')))
if not perm_run_scripts:
    print(f'No permeation_run_*.py found in {WORK_DIR} -- run '
          f'regenerate_permeation_scripts.py first.')
    sys.exit(1)

print(f'Found {len(perm_run_scripts)} regenerated script(s):')
sh_paths = []
for py_path in perm_run_scripts:
    stem = os.path.basename(py_path)[len('permeation_run_'):-len('.py')]
    sh_path = os.path.join(WORK_DIR, f'slurm_permeation_run_{stem}.sh')
    write_slurm_job(
        job_name=f'perm_run_{stem}',
        slurm_config=WEST_SLURM,
        out_path=sh_path,
        commands=[f'python {py_path}'],
    )
    sh_paths.append(sh_path)
    print(f'  {stem}: {sh_path}')

print(f'\n{len(sh_paths)} wrapper(s) written. Nothing was submitted.')
print('Next: submit ONLY a metal whose Part 1 + Part 3 are complete, e.g.:')
for sh_path in sh_paths:
    print(f'  sbatch {sh_path}')

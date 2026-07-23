#!/usr/bin/env python3
"""
wrap_diffusivity_runs_west.py
===============================
Wraps each already-regenerated diffusivity_run_{stem}.py (from
regenerate_diffusivity_scripts.py) in its own small, trackable SLURM job on
the `west` partition -- the same partition/resource shape pipeline_orch
itself uses (cpus_per_task=4, 30-day wall time), so each metal's
diffusivity orchestrator shows up in squeue individually instead of
running invisibly as a login-node nohup process. Mirrors
wrap_neb_runs_west.py exactly, but for the per-metal diffusivity scripts.

Only WRITES the .sh wrapper files -- does not submit anything. Prints the
sbatch commands to run manually, mirroring the rest of this session's
generate-then-you-submit pattern.

Usage
-----
    python wrap_diffusivity_runs_west.py
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

# 'diffusivity_run_*.py' -- the underscore after 'run' excludes the
# combined diffusivity_run.py (regenerate_diffusivity_script.py's output),
# which has no stem suffix and is wrapped separately by
# wrap_diffusivity_run_west.py (singular).
diffusivity_run_scripts = sorted(
    glob.glob(os.path.join(WORK_DIR, 'diffusivity_run_*.py'))
)
if not diffusivity_run_scripts:
    print(f'No diffusivity_run_*.py found in {WORK_DIR} -- run '
          f'regenerate_diffusivity_scripts.py first.')
    sys.exit(1)

print(f'Found {len(diffusivity_run_scripts)} regenerated script(s):')
sh_paths = []
for py_path in diffusivity_run_scripts:
    stem = os.path.basename(py_path)[len('diffusivity_run_'):-len('.py')]
    sh_path = os.path.join(WORK_DIR, f'slurm_diffusivity_run_{stem}.sh')
    write_slurm_job(
        job_name=f'diffusivity_run_{stem}',
        slurm_config=WEST_SLURM,
        out_path=sh_path,
        commands=[f'python {py_path}'],
    )
    sh_paths.append(sh_path)
    print(f'  {stem}: {sh_path}')

print(f'\n{len(sh_paths)} wrapper(s) written. Nothing was submitted.')
print('Next: submit each one, e.g.:')
for sh_path in sh_paths:
    print(f'  sbatch {sh_path}')

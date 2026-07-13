#!/usr/bin/env python3
"""
wrap_diffusivity_run_west.py
=============================
Wraps the already-regenerated diffusivity_run.py in its own small,
trackable SLURM job on the `west` partition -- mirrors
wrap_neb_runs_west.py exactly, but for diffusivity_run.py specifically.

diffusivity_run.py has no standalone .sh wrapper of its own: under
pipeline_orch it only ever runs as a direct subprocess.Popen child of
pipeline_run.py's single SLURM allocation, never as its own submitted job.
With pipeline_orch not currently driving it, it needs its own trackable
job to resume Hastelloy_N_7's stalled NVT chains (and pick up any other
already-completed-but-unmarked phases via the .done backfill).

Only WRITES the .sh wrapper file -- does not submit anything. Prints the
sbatch command to run manually, mirroring the rest of this session's
generate-then-you-submit pattern.

Usage
-----
    python wrap_diffusivity_run_west.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import BASE_DIR, SLURM_DEFAULTS
from models.create_slurm import write_slurm_job

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

# Matches wrap_neb_runs_west.py / pipeline.ipynb cell 5's PIPE_* literals exactly.
WEST_SLURM = dict(
    SLURM_DEFAULTS,
    partition='west',
    gpu=None,
    cpus_per_task=4,
    time='30-00:00:00',
)

py_path = os.path.join(WORK_DIR, 'diffusivity_run.py')
if not os.path.exists(py_path):
    print(f'{py_path} not found -- regenerate it first (pipeline.ipynb cell 10).')
    sys.exit(1)

sh_path = os.path.join(WORK_DIR, 'slurm_diffusivity_run.sh')
write_slurm_job(
    job_name='diffusivity_run',
    slurm_config=WEST_SLURM,
    out_path=sh_path,
    commands=[f'python {py_path}'],
)

print(f'Wrapper written: {sh_path}')
print('Nothing was submitted.')
print(f'Next: sbatch {sh_path}')

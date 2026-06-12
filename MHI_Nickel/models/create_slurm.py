"""
models/create_slurm.py
======================
Utilities for writing and submitting SLURM batch scripts used across
all MHI_Nickel notebooks.

Covers:
- Single jobs      (NB02, NB04, NB04b)
- Multi-job loops  (NB05, NB06, NB10)
- Array jobs       (NB05b, NB06a, NB06b)
- Queue-filling    (auto_submit_*.sh equivalent)

Usage
-----
import sys; sys.path.insert(0, '.')
from models.create_slurm import write_slurm_job, submit_slurm_job, auto_submit
"""

import glob
import os
import subprocess
import time


# ═══════════════════════════════════════════════════════════════════════════
# SLURM SCRIPT WRITER
# ═══════════════════════════════════════════════════════════════════════════

def write_slurm_job(
    job_name,
    script_path,
    output_log,
    slurm_config,
    out_path,
    runner='lmp',
    lammps_cmd=None,
    kokkos_flags=None,
    lammps_log=None,
    array_range=None,
    concurrent=None,
    extra_env_vars=None,
):
    """
    Write a SLURM batch script (.sh) to disk and return its path.

    Supports both LAMMPS (lmp) and Python runners, optional SLURM array
    jobs, and per-call environment variable overrides.

    Parameters
    ----------
    job_name : str
        SLURM job name passed to ``--job-name``.
    script_path : str
        Path to the LAMMPS input script or Python file executed inside
        the job.
    output_log : str
        Path pattern for SLURM stdout.  Use ``%j`` for single jobs or
        ``%A_%a`` for array jobs.
    slurm_config : dict
        Cluster settings with keys: ``partition``, ``ntasks``,
        ``cpus_per_task``, ``gpu``, ``time``, ``conda_env``,
        ``cuda_version``, ``openmpi_ver``, ``ld_paths`` (list of str).
    out_path : str
        Destination path for the written ``.sh`` file.  Parent
        directories are created automatically.
    runner : {'lmp', 'python'}, optional
        Executable to call inside the job body.  Default is ``'lmp'``.
    lammps_cmd : str, optional
        Full path to the LAMMPS binary.  Required when
        ``runner='lmp'``.
    kokkos_flags : list of str, optional
        Extra flags inserted before ``-in`` when calling LAMMPS,
        e.g. ``['-k', 'on', 'g', '1', '-sf', 'kk']``.
        Ignored when ``runner='python'``.
    lammps_log : str, optional
        Path passed to ``-log`` when calling LAMMPS.  If ``None``,
        no ``-log`` flag is added.  Ignored when ``runner='python'``.
    array_range : tuple of int, optional
        ``(start, end)`` for ``--array=start-end``.  If ``None``, no
        array directive is written.
    concurrent : int, optional
        ``%N`` throttle appended to the array directive, e.g.
        ``--array=0-99%4``.  Only used when ``array_range`` is set.
    extra_env_vars : dict, optional
        Additional ``export KEY=VALUE`` lines written after the
        ``LD_LIBRARY_PATH`` exports, e.g. ``{'SEED': '42'}``.

    Returns
    -------
    out_path : str
        Path to the written SLURM script.

    Raises
    ------
    ValueError
        If ``runner='lmp'`` but ``lammps_cmd`` is not provided.
    """
    if runner == 'lmp' and lammps_cmd is None:
        raise ValueError("lammps_cmd is required when runner='lmp'.")

    sc = slurm_config

    # ── LD_LIBRARY_PATH exports ───────────────────────────────────
    ld_lines = '\n'.join(
        f'export LD_LIBRARY_PATH={p}:$LD_LIBRARY_PATH'
        for p in sc.get('ld_paths', [])
    )

    # ── Extra environment variables ───────────────────────────────
    env_lines = ''
    if extra_env_vars:
        env_lines = '\n'.join(
            f'export {k}={v}' for k, v in extra_env_vars.items()
        )

    # ── Array directive ───────────────────────────────────────────
    array_line = ''
    if array_range is not None:
        s, e = array_range
        throttle = f'%{concurrent}' if concurrent else ''
        array_line = f'#SBATCH --array={s}-{e}{throttle}'

    # ── Executable line ───────────────────────────────────────────
    if runner == 'lmp':
        kk = ' '.join(kokkos_flags) if kokkos_flags else ''
        log_flag = f'-log {lammps_log}' if lammps_log else ''
        exec_line = f'{lammps_cmd} {kk} -in {script_path} {log_flag}'.strip()
    else:
        exec_line = f'python {script_path}'

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={sc['ntasks']}
#SBATCH --cpus-per-task={sc['cpus_per_task']}
#SBATCH --gres=gpu:{sc['gpu']}
#SBATCH --partition={sc['partition']}
#SBATCH --time={sc['time']}
#SBATCH --output={output_log}
{array_line}

module load OpenMPI/{sc['openmpi_ver']}
module load cuda/{sc['cuda_version']}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate {sc['conda_env']}
{ld_lines}
{env_lines}
cd {os.getcwd()}

echo "Node: $(hostname)  Start: $(date)"
{exec_line}
echo "End: $(date)"
"""

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    os.chmod(out_path, 0o755)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# SLURM JOB SUBMITTER
# ═══════════════════════════════════════════════════════════════════════════

def submit_slurm_job(slurm_path, extra_args=None, dry_run=False):
    """
    Submit a SLURM script with ``sbatch`` and return the job ID.

    Parameters
    ----------
    slurm_path : str
        Path to the ``.sh`` file to submit.
    extra_args : list of str, optional
        Additional arguments prepended to the ``sbatch`` call, e.g.
        ``['--array=2,5']`` for targeted array resubmissions.
    dry_run : bool, optional
        If ``True``, print the command without submitting.
        Default ``False``.

    Returns
    -------
    job_id : str or None
        Submitted SLURM job ID, or ``None`` on failure or dry run.
    """
    cmd = ['sbatch'] + (extra_args or []) + [slurm_path]

    if dry_run:
        print(f'[dry-run] {" ".join(cmd)}')
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f'Submitted {slurm_path} → Job {job_id}')
        return job_id

    print(f'Submission failed: {result.stderr.strip()}')
    return None


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-SUBMIT QUEUE FILLER
# ═══════════════════════════════════════════════════════════════════════════

def auto_submit(
    array_script,
    index_file,
    result_dir,
    result_pattern,
    n_total,
    job_name,
    queue_max=8,
    concurrent=4,
    interval=60,
    pre_submit_scripts=None,
):
    """
    Continuously submit SLURM array jobs, keeping the queue filled.

    Mirrors the logic of ``auto_submit_*.sh``: polls ``squeue``,
    submits chunks to fill up to ``queue_max``, waits for all tasks
    to finish, then reports missing outputs.  Run this in a long-lived
    process on the cluster login node (e.g. inside a ``screen`` or
    ``nohup`` session).

    Parameters
    ----------
    array_script : str
        Path to the SLURM array ``.sh`` script to submit.
    index_file : str
        Text file with one label per line (one per array task).
    result_dir : str
        Directory where output files are written by the jobs.
    result_pattern : str
        Glob pattern relative to ``result_dir`` used to detect task
        completion, e.g. ``'ads_s_*.log'`` or
        ``'*/neb_barrier.txt'``.
    n_total : int
        Total number of array tasks (0-indexed: 0 to n_total-1).
    job_name : str
        SLURM job name used to filter ``squeue`` output (``-n`` flag).
    queue_max : int, optional
        Maximum number of tasks to keep in the queue at once.
        Default ``8``.
    concurrent : int, optional
        ``%N`` throttle per ``sbatch --array`` submission.
        Default ``4``.
    interval : int, optional
        Seconds to sleep when the queue is full or while draining.
        Default ``60``.
    pre_submit_scripts : list of str, optional
        SLURM scripts submitted once before the main loop, e.g.
        reference ``clean_slab`` and ``h2_gas`` jobs in NB05b.

    Returns
    -------
    missing_ids : list of int
        Task indices whose output files were not found after all jobs
        complete.  Empty list means full success.

    Notes
    -----
    Uses ``os.environ['USER']`` to scope ``squeue`` queries.  Ensure
    ``$USER`` is set correctly on the cluster.
    """
    user = os.environ.get('USER', '')

    def _queue_count():
        r = subprocess.run(
            ['squeue', '-u', user, '-h', '-t', 'pending,running',
             '-r', '-n', job_name],
            capture_output=True, text=True,
        )
        lines = [l for l in r.stdout.strip().splitlines() if l]
        return len(lines)

    # ── Pre-submit reference jobs ─────────────────────────────────
    if pre_submit_scripts:
        for s in pre_submit_scripts:
            subprocess.run(['sbatch', s])
            time.sleep(2)

    # ── Main submission loop ──────────────────────────────────────
    next_task = 0
    print(f'Auto-submit: {n_total} tasks, queue_max={queue_max}, '
          f'concurrent={concurrent}')

    while next_task < n_total:
        n_queued = _queue_count()
        room = queue_max - n_queued

        if room > 0:
            end_task = min(next_task + room - 1, n_total - 1)
            print(f'  [{time.strftime("%H:%M:%S")}] '
                  f'Queue={n_queued}, submitting {next_task}–{end_task}')
            subprocess.run([
                'sbatch',
                f'--array={next_task}-{end_task}%{concurrent}',
                array_script,
            ])
            next_task = end_task + 1
            time.sleep(5)
        else:
            print(f'  [{time.strftime("%H:%M:%S")}] '
                  f'Queue full ({n_queued}/{queue_max}), waiting {interval}s...')
            time.sleep(interval)

    # ── Wait for drain ────────────────────────────────────────────
    print('All submitted. Waiting for queue to drain...')
    while True:
        n = _queue_count()
        if n == 0:
            break
        print(f'  [{time.strftime("%H:%M:%S")}] {n} tasks remaining...')
        time.sleep(interval)

    # ── Completion check ──────────────────────────────────────────
    with open(index_file) as f:
        labels = [line.strip() for line in f if line.strip()]

    completed = glob.glob(os.path.join(result_dir, result_pattern))
    completed_basenames = {os.path.basename(c) for c in completed}

    missing_ids = [
        i for i, label in enumerate(labels)
        if not any(
            label in c or label in os.path.basename(c)
            for c in completed
        )
    ]

    n_ok = len(labels) - len(missing_ids)
    print(f'\nDone.  {n_ok}/{len(labels)} tasks completed.')

    if missing_ids:
        ids_str = ','.join(map(str, missing_ids))
        print(f'Missing IDs : {ids_str}')
        print(f'Resubmit   : sbatch --array={ids_str}%{concurrent} {array_script}')

    return missing_ids
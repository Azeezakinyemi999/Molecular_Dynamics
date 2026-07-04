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
    slurm_config,
    out_path,
    commands=None,
    script_path=None,
    output_log=None,
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

    Supports arbitrary shell commands (``commands`` list), single
    LAMMPS/Python scripts (``script_path`` + ``runner``), optional
    SLURM array jobs, and per-call environment variable overrides.

    Parameters
    ----------
    job_name : str
        SLURM job name passed to ``--job-name``.
    slurm_config : dict
        Cluster settings with keys: ``partition``, ``ntasks``,
        ``cpus_per_task``, ``gpu``, ``time``, ``conda_env``,
        ``cuda_version``, ``openmpi_ver``, ``ld_paths`` (list of str).
    out_path : str
        Destination path for the written ``.sh`` file.  Parent
        directories are created automatically.
    commands : list of str, optional
        Arbitrary shell commands written verbatim into the job body,
        one per line.  Takes precedence over ``script_path``/``runner``
        when provided.  Example::

            commands=[
                f'{LAMMPS_CMD} {kk} -in min.lammps -log min.log',
                f'{LAMMPS_CMD} {kk} -in nvt.lammps -log nvt.log',
            ]
    script_path : str, optional
        Path to the LAMMPS input script or Python file executed inside
        the job.  Used only when ``commands`` is ``None``.
    output_log : str, optional
        Path pattern for SLURM stdout.  Use ``%j`` for single jobs or
        ``%A_%a`` for array jobs.  Defaults to ``out_path`` with
        ``_%j.out`` appended.
    runner : {'lmp', 'python'}, optional
        Executable to call when ``commands`` is ``None``.
        Default is ``'lmp'``.
    lammps_cmd : str, optional
        Full path to the LAMMPS binary.  Required when
        ``runner='lmp'`` and ``commands`` is ``None``.
    kokkos_flags : list of str, optional
        Extra flags inserted before ``-in`` when calling LAMMPS.
        Ignored when ``commands`` is provided.
    lammps_log : str, optional
        Path passed to ``-log``.  Ignored when ``commands`` is provided.
    array_range : tuple of int, optional
        ``(start, end)`` for ``--array=start-end``.
    concurrent : int, optional
        ``%N`` throttle appended to the array directive.
    extra_env_vars : dict, optional
        Additional ``export KEY=VALUE`` lines, e.g. ``{'SEED': '42'}``.

    Returns
    -------
    out_path : str
        Path to the written SLURM script.

    Raises
    ------
    ValueError
        If neither ``commands`` nor ``script_path`` is provided.
        If ``runner='lmp'``, ``commands`` is ``None``, and
        ``lammps_cmd`` is not provided.
    """
    if commands is None and script_path is None:
        raise ValueError('Provide either commands or script_path.')
    if commands is None and runner == 'lmp' and lammps_cmd is None:
        raise ValueError("lammps_cmd is required when runner='lmp' and commands is None.")

    sc = slurm_config

    # ── Default output log ────────────────────────────────────────
    if output_log is None:
        base = os.path.splitext(out_path)[0]
        output_log = f'{base}_%j.out'

    # ── LD_LIBRARY_PATH exports ───────────────────────────────────
    ld_lines = '\n'.join(
        f'export LD_LIBRARY_PATH={p}:$LD_LIBRARY_PATH'
        for p in sc.get('ld_paths', [])
    )
    # ── LD_PRELOAD ────────────────────────────────────────────
    preload_line = ''
    if sc.get('ld_preload'):
        preload_line = f'export LD_PRELOAD={sc["ld_preload"]}'

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

    # ── Executable block ──────────────────────────────────────────
    if commands is not None:
        exec_line = '\n'.join(commands)
    elif runner == 'lmp':
        kk = ' '.join(kokkos_flags) if kokkos_flags else ''
        log_flag = f'-log {lammps_log}' if lammps_log else ''
        exec_line = f'{lammps_cmd} {kk} -in {script_path} {log_flag}'.strip()
    else:
        exec_line = f'python {script_path}'

    gpu_line   = f'#SBATCH --gres=gpu:{sc["gpu"]}' if sc.get('gpu') else ''
    cuda_line  = f'module load cuda/{sc["cuda_version"]}' if sc.get('cuda_version') else ''

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={sc['ntasks']}
#SBATCH --cpus-per-task={sc['cpus_per_task']}
{gpu_line}
#SBATCH --partition={sc['partition']}
#SBATCH --time={sc['time']}
#SBATCH --output={output_log}
#SBATCH --exclude=d3204
{array_line}

module load OpenMPI/{sc['openmpi_ver']}
{cuda_line}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate {sc['conda_env']}
{ld_lines}
{preload_line}
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

def submit_slurm_job(slurm_path, extra_args=None, dry_run=False, dependency=None):
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
    dependency : str or None, optional
        SLURM dependency expression, e.g. ``'afterok:12345'``.
        Passed as ``--dependency=<value>``.  Default ``None``.

    Returns
    -------
    job_id : str or None
        Submitted SLURM job ID, or ``None`` on failure or dry run.
    """
    dep_args = [f'--dependency={dependency}'] if dependency else []
    cmd = ['sbatch'] + dep_args + (extra_args or []) + [slurm_path]

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


# ═══════════════════════════════════════════════════════════════════════════
# JOB STATUS CHECKER
# ═══════════════════════════════════════════════════════════════════════════

def check_jobs(
    job_ids: dict,
    verbose: bool = True,
) -> dict:
    """
    Poll ``squeue`` and return the current status of a set of jobs.

    Parameters
    ----------
    job_ids : dict
        ``{label: job_id_str}`` mapping — e.g.
        ``{300: '123456', 400: '123457'}``.
    verbose : bool, optional
        If ``True``, print a status table.  Default ``True``.

    Returns
    -------
    dict
        ``{label: status}`` where status is one of
        ``'running'``, ``'pending'``, ``'done'``, or ``'unknown'``.
    """
    # Get all jobs currently in the queue for this user
    user = os.environ.get('USER', '')
    result = subprocess.run(
        ['squeue', '-u', user, '-h', '-o', '%i %T'],
        capture_output=True, text=True,
    )
    queued = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            queued[parts[0]] = parts[1].lower()  # job_id → state

    statuses = {}
    for label, jid in job_ids.items():
        jid_str = str(jid)
        if jid_str in queued:
            statuses[label] = queued[jid_str]   # 'running' or 'pending'
        else:
            statuses[label] = 'done'             # not in queue → finished

    if verbose:
        print(f'{"Label":>10s}  {"Job ID":>10s}  {"Status"}')
        print('-' * 35)
        for label, jid in job_ids.items():
            print(f'{str(label):>10s}  {str(jid):>10s}  {statuses[label]}')
        n_done = sum(1 for s in statuses.values() if s == 'done')
        print(f'\n{n_done}/{len(job_ids)} jobs done.')

    return statuses


# ═══════════════════════════════════════════════════════════════════════════
# BLOCKING WAIT FOR JOBS
# ═══════════════════════════════════════════════════════════════════════════

def wait_for_jobs(
    job_ids: dict,
    poll_interval: int = 60,
    verbose: bool = True,
) -> None:
    """
    Block until all jobs in ``job_ids`` have left the SLURM queue.

    Call this before any post-processing cell that depends on job
    outputs (e.g. before parsing logs in NB05, NB06, NB10).

    Parameters
    ----------
    job_ids : dict
        ``{label: job_id_str}`` — same format as :func:`check_jobs`.
    poll_interval : int, optional
        Seconds between ``squeue`` polls.  Default ``60``.
    verbose : bool, optional
        Print status updates while waiting.  Default ``True``.
    """
    import time as _time

    if verbose:
        print(f'Waiting for {len(job_ids)} jobs to finish '
              f'(polling every {poll_interval}s)...')

    while True:
        statuses = check_jobs(job_ids, verbose=False)
        n_done = sum(1 for s in statuses.values() if s == 'done')
        n_total = len(job_ids)

        if verbose:
            still_running = [
                f'{l}({s})' for l, s in statuses.items() if s != 'done'
            ]
            print(f'  [{_time.strftime("%H:%M:%S")}] '
                  f'{n_done}/{n_total} done'
                  + (f'  — waiting: {" ".join(still_running)}' if still_running else ''))

        if n_done == n_total:
            if verbose:
                print('All jobs completed.')
            return

        _time.sleep(poll_interval)


# ═══════════════════════════════════════════════════════════════════════════
# CHAINED / SELF-RESUBMITTING JOB  (for runs that exceed max wall time)
# ═══════════════════════════════════════════════════════════════════════════

def _hms_to_seconds(hms: str) -> int:
    """Convert 'D-HH:MM:SS' or 'HH:MM:SS' to total integer seconds."""
    hms = hms.strip()
    days = 0
    if '-' in hms:
        d_part, hms = hms.split('-', 1)
        days = int(d_part)
    parts = hms.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        raise ValueError(f'Cannot parse time string: {hms!r}')
    return days * 86400 + h * 3600 + m * 60 + s


def write_chained_slurm_job(
    job_name,
    slurm_config,
    out_path,
    first_commands,
    restart_commands=None,
    restart_glob=None,
    cutoff=None,
    flush_wait=30,
    work_dir=None,
    output_log=None,
    extra_env_vars=None,
    n_equil=None,
    equil_restart_commands=None,
    prod_restart_commands=None,
):
    """
    Write a self-resubmitting SLURM script for simulations that exceed
    the cluster's maximum wall time.

    The same ``.sh`` file is used for every leg of the run.  On each
    execution the script:

    1. Globs for an existing LAMMPS restart file (``restart_glob``).
    2. Runs ``first_commands`` (leg 1, fresh start) or
       ``restart_commands`` (legs 2+, from checkpoint) wrapped inside
       ``timeout`` so it exits cleanly at ``cutoff``.
    3. Inspects the exit code:

       * **0** — LAMMPS converged (force/energy tolerance met).
         Prints "complete" and exits without resubmitting.
       * **124** — ``timeout`` fired (wall-time limit approaching).
         Waits ``flush_wait`` seconds for LAMMPS to finish writing the
         last restart file, then calls ``sbatch`` on the original script
         path (baked in at generation time — ``$0`` under sbatch is the
         spooled copy in ``/var/spool/slurmd``) to
         chain the next leg.
       * **other** — real error; exits with that code so SLURM marks
         the job as FAILED (no silent infinite loop).

    Compared to :func:`write_slurm_job` this function:

    * Uses a bash heredoc for command blocks — no quoting issues with
      paths or LAMMPS flags that contain single quotes.
    * Accepts an explicit ``work_dir`` instead of baking in
      ``os.getcwd()`` (which would be wrong when leg 2 is submitted by
      SLURM, not the notebook).
    * Does **not** support array jobs — chained runs are single jobs
      submitted serially.

    Parameters
    ----------
    job_name : str
        SLURM ``--job-name``.
    slurm_config : dict
        Cluster settings — same dict used by :func:`write_slurm_job`.
        Keys: ``partition``, ``ntasks``, ``cpus_per_task``, ``gpu``,
        ``time``, ``conda_env``, ``cuda_version``, ``openmpi_ver``,
        ``ld_paths``.
    out_path : str
        Destination ``.sh`` path.  Parent directories are created.
        The script is made executable (``chmod 755``).
    first_commands : list of str
        Shell commands for leg 1 (no restart file present), e.g.::

            [f'{LAMMPS_CMD} {kk} -in nvt.lammps -log nvt.log']
    restart_commands : list of str or None
        Shell commands for legs 2+ when no phase detection is needed.
        Ignored when ``n_equil`` is provided.
    restart_glob : str
        Glob pattern used to detect an existing LAMMPS checkpoint, e.g.
        ``'results/500K/nvt_500K.*.restart'``.
    n_equil : int or None, optional
        When provided, enables phase-aware restart selection.  The step
        number is extracted from the checkpoint filename; if it is less
        than ``n_equil``, ``equil_restart_commands`` is used, otherwise
        ``prod_restart_commands``.
    equil_restart_commands : list of str or None, optional
        Commands for mid-equil restarts (step < n_equil).  Required when
        ``n_equil`` is set.
    prod_restart_commands : list of str or None, optional
        Commands for mid-production restarts (step >= n_equil).  Required
        when ``n_equil`` is set.
    cutoff : str
        Wall-time at which the job should self-terminate and resubmit,
        in ``'HH:MM:SS'`` format.  Must be shorter than
        ``slurm_config['time']`` to leave time for the resubmission.
        Example: ``'23:55:00'`` with ``slurm_config['time']='24:00:00'``.
    flush_wait : int, optional
        Seconds to sleep after ``timeout`` fires before resubmitting,
        to allow LAMMPS to finish flushing the restart file to disk.
        Default ``30``.
    work_dir : str or None, optional
        Absolute path to ``cd`` into at job start.  If ``None``, no
        ``cd`` is written and all paths in the commands must be
        absolute.  Default ``None``.
    output_log : str or None, optional
        SLURM stdout pattern.  Defaults to ``{out_path_stem}_%j.out``.
    extra_env_vars : dict or None, optional
        ``{KEY: VALUE}`` pairs exported after the standard environment
        setup, e.g. ``{'OMP_NUM_THREADS': '1'}``.

    Returns
    -------
    out_path : str
        Path to the written script.

    Example
    -------
    >>> from models.config import SLURM_DEFAULTS, LAMMPS_CMD, KOKKOS_FLAGS
    >>> kk = ' '.join(KOKKOS_FLAGS)
    >>> cfg = SLURM_DEFAULTS | {'time': '24:00:00'}
    >>> write_chained_slurm_job(
    ...     job_name         = 'nvt_500K',
    ...     slurm_config     = cfg,
    ...     out_path         = 'slurm_scripts/500K/nvt_chain.sh',
    ...     first_commands   = [f'{LAMMPS_CMD} {kk} -in nvt.lammps -log nvt.log'],
    ...     restart_commands = [f'{LAMMPS_CMD} {kk} -in nvt_restart.lammps -log nvt.log'],
    ...     restart_glob     = 'results/500K/nvt_500K.*.restart',
    ...     cutoff           = '23:55:00',
    ...     work_dir         = '/projects/westgroup/akinyemi.az/MHI_Nickel',
    ... )
    """
    sc = slurm_config
    cutoff_sec = _hms_to_seconds(cutoff)

    # ── output log ────────────────────────────────────────────────
    if output_log is None:
        base = os.path.splitext(os.path.abspath(out_path))[0]
        output_log = f'{base}_%j.out'

    # ── LD_LIBRARY_PATH ───────────────────────────────────────────
    ld_lines = '\n'.join(
        f'export LD_LIBRARY_PATH={p}:$LD_LIBRARY_PATH'
        for p in sc.get('ld_paths', [])
    )
    preload_line = f'export LD_PRELOAD={sc["ld_preload"]}' if sc.get('ld_preload') else ''

    # ── extra env vars ────────────────────────────────────────────
    env_lines = ''
    if extra_env_vars:
        env_lines = '\n'.join(
            f'export {k}={v}' for k, v in extra_env_vars.items()
        )

    # ── working directory ─────────────────────────────────────────
    cd_line = f'cd {work_dir}' if work_dir else ''

    # ── command heredocs ──────────────────────────────────────────
    # Using heredoc (<<'EOF') avoids ALL quoting/escaping issues with
    # paths, LAMMPS flags, or single quotes inside the commands.
    first_block = '\n'.join(first_commands)

    # Build the script as a list of lines to avoid f-string conflicts
    # with bash $ variables.
    lines = [
        '#!/bin/bash',
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --ntasks={sc["ntasks"]}',
        f'#SBATCH --cpus-per-task={sc["cpus_per_task"]}',
        f'#SBATCH --gres=gpu:{sc["gpu"]}' if sc.get('gpu') else '',
        f'#SBATCH --partition={sc["partition"]}',
        f'#SBATCH --time={sc["time"]}',
        f'#SBATCH --output={output_log}',
        '#SBATCH --exclude=d3204',
        '',
        '# ── Environment ─────────────────────────────────────────',
        f'module load OpenMPI/{sc["openmpi_ver"]}',
        f'module load cuda/{sc["cuda_version"]}' if sc.get('cuda_version') else '',
        'source ~/miniforge3/etc/profile.d/conda.sh',
        f'conda activate {sc["conda_env"]}',
        ld_lines,
        preload_line,
        env_lines,
        cd_line,
        '',
        '# ── Paths and config ─────────────────────────────────────',
        # $0 under sbatch is the spooled copy in /var/spool/slurmd, so the
        # .done sentinel and resubmission must use the real script path.
        f'SCRIPT_PATH="{os.path.abspath(out_path)}"',
        f'RESTART_GLOB="{restart_glob}"',
        f'CUTOFF_SEC={cutoff_sec}',
        f'FLUSH_WAIT={flush_wait}',
        '',
        'echo "Job: $SLURM_JOB_ID  |  Node: $(hostname)  |  $(date)"',
        '',
        '# ── Select command block ─────────────────────────────────',
        '# ls -t glob | head -1 returns the most recent restart file.',
        '# If the glob matches nothing, RESTART_FILE is empty.',
        'RESTART_FILE=$(ls -t ${RESTART_GLOB} 2>/dev/null | head -1)',
        '',
    ]

    if n_equil is not None:
        # Phase-aware restart: detect whether checkpoint is in equil or prod
        equil_block = '\n'.join(equil_restart_commands)
        prod_block  = '\n'.join(prod_restart_commands)
        lines += [
            f'N_EQUIL={n_equil}',
            '',
            'if [ -n "$RESTART_FILE" ]; then',
            '    # Extract the step number from the checkpoint filename.',
            '    LAST_STEP=$(basename "$RESTART_FILE" .restart | grep -oE \'[0-9]+$\')',
            '    if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -lt "$N_EQUIL" ]; then',
            '        echo "Equil-phase restart (step=$LAST_STEP < n_equil=$N_EQUIL)."',
            f'        timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<\'EQUIL_RST_BLOCK\'',
            equil_block,
            'EQUIL_RST_BLOCK',
            '    else',
            '        echo "Prod-phase restart (step=$LAST_STEP >= n_equil=$N_EQUIL)."',
            f'        timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<\'PROD_RST_BLOCK\'',
            prod_block,
            'PROD_RST_BLOCK',
            '    fi',
            'else',
            '    echo "No restart file found -- starting fresh."',
            f'    timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<\'FIRST_BLOCK\'',
            first_block,
            'FIRST_BLOCK',
            'fi',
        ]
    else:
        # Legacy two-branch behaviour (no phase detection)
        restart_block = '\n'.join(restart_commands)
        lines += [
            'if [ -n "$RESTART_FILE" ]; then',
            '    echo "Restart file found: $RESTART_FILE -- using restart commands."',
            f'    timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<\'LAMMPS_BLOCK\'',
            restart_block,
            'LAMMPS_BLOCK',
            'else',
            '    echo "No restart file found -- starting fresh."',
            f'    timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<\'LAMMPS_BLOCK\'',
            first_block,
            'LAMMPS_BLOCK',
            'fi',
        ]

    lines += [
        '',
        'EXIT_CODE=$?',
        'echo "Command block exited with code $EXIT_CODE at $(date)"',
        '',
        '# ── Handle exit code ─────────────────────────────────────',
        'if [ "$EXIT_CODE" -eq 0 ]; then',
        '    echo "LAMMPS converged (exit 0). Simulation complete."',
        '    touch "${SCRIPT_PATH}.done"',
        '    exit 0',
        '',
        'elif [ "$EXIT_CODE" -eq 124 ]; then',
        f'    echo "Timeout fired. Waiting ${{FLUSH_WAIT}}s for LAMMPS to flush restart files..."',
        '    sleep "$FLUSH_WAIT"',
        '    echo "Resubmitting: sbatch $SCRIPT_PATH"',
        '    NEW_JOB=$(sbatch "$SCRIPT_PATH")',
        '    echo "  --> $NEW_JOB"',
        '    exit 0',
        '',
        'else',
        '    echo "Job failed with exit code $EXIT_CODE. Not resubmitting."',
        '    exit $EXIT_CODE',
        'fi',
        '',
    ]

    script = '\n'.join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write(script)
    os.chmod(out_path, 0o755)
    return out_path
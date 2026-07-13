"""
tests/test_create_slurm.py
==========================
Unit tests for models/create_slurm.py.

Covers:
  _hms_to_seconds           — 'HH:MM:SS' / 'D-HH:MM:SS' → int seconds
  write_slurm_job           — single-job SLURM script writer (all modes)
  write_chained_slurm_job   — edge cases not covered by test_nvt_restart.py
  check_jobs                — squeue parsing, incl. SLURM array job ids

All tests are offline — no sbatch, no squeue, no cluster (subprocess.run
is mocked for check_jobs tests).
"""

import os
import stat
import sys
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.create_slurm import (
    write_slurm_job,
    write_chained_slurm_job,
    check_jobs,
    submit_with_retry,
    _hms_to_seconds,
)


# ─── shared SLURM config ──────────────────────────────────────────────────────

_SLURM_CFG = {
    'partition':     'test-gpu',
    'ntasks':        1,
    'cpus_per_task': 4,
    'gpu':           'a100:1',
    'time':          '01:00:00',
    'openmpi_ver':   '4.1.6',
    'cuda_version':  '12.3.0',
    'conda_env':     'mace-env',
    'ld_paths':      ['/fake/lib1', '/fake/lib2'],
}

_CUTOFF   = '00:55:00'   # 3300 seconds
_LAMMPS   = '/cluster/bin/lmp'
_SCRIPT   = '/work/nvt.lammps'
_LOG      = '/work/nvt.log'

# ─── helper ───────────────────────────────────────────────────────────────────

def _lines_with(text, kw):
    return [l for l in text.splitlines() if kw in l]


# ═════════════════════════════════════════════════════════════════════════════
# 1. _hms_to_seconds
# ═════════════════════════════════════════════════════════════════════════════

class TestHmsToSeconds:

    def test_one_second(self):
        assert _hms_to_seconds('00:00:01') == 1

    def test_one_minute(self):
        assert _hms_to_seconds('00:01:00') == 60

    def test_one_hour(self):
        assert _hms_to_seconds('01:00:00') == 3600

    def test_24_hours(self):
        assert _hms_to_seconds('24:00:00') == 86400

    def test_mixed_hms(self):
        # 2 h 30 m 15 s = 7200 + 1800 + 15 = 9015
        assert _hms_to_seconds('02:30:15') == 9015

    def test_cutoff_value(self):
        # 23:55:00 = 23*3600 + 55*60 = 82800 + 3300 = 86100
        assert _hms_to_seconds('23:55:00') == 86100

    def test_zero(self):
        assert _hms_to_seconds('00:00:00') == 0

    def test_days_prefix(self):
        # 1-00:00:00 = 86400
        assert _hms_to_seconds('1-00:00:00') == 86400

    def test_days_plus_hms(self):
        # 1-12:30:45 = 86400 + 43200 + 1800 + 45 = 131445
        assert _hms_to_seconds('1-12:30:45') == 131445

    def test_two_part_mm_ss(self):
        # 'MM:SS' form (no hours) → (0, MM, SS)
        assert _hms_to_seconds('02:30') == 150

    def test_leading_whitespace_stripped(self):
        assert _hms_to_seconds(' 01:00:00') == 3600

    def test_invalid_format_raises(self):
        with pytest.raises((ValueError, IndexError)):
            _hms_to_seconds('not-a-time')


# ═════════════════════════════════════════════════════════════════════════════
# 2. write_slurm_job
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteSlurmJob:

    @pytest.fixture()
    def cmd_script(self, tmp_path):
        out = str(tmp_path / 'test_job.sh')
        write_slurm_job(
            job_name='my_job',
            slurm_config=_SLURM_CFG,
            out_path=out,
            commands=['echo hello', 'echo world'],
        )
        return out, open(out).read()

    # ── file / return value ───────────────────────────────────────────────

    def test_file_created(self, cmd_script):
        out, _ = cmd_script
        assert os.path.exists(out)

    def test_returns_path(self, tmp_path):
        out = str(tmp_path / 'ret.sh')
        ret = write_slurm_job('j', _SLURM_CFG, out, commands=['echo'])
        assert ret == out

    def test_script_executable(self, cmd_script):
        out, _ = cmd_script
        mode = os.stat(out).st_mode
        assert mode & stat.S_IXUSR

    def test_parent_dirs_created(self, tmp_path):
        out = str(tmp_path / 'a' / 'b' / 'job.sh')
        write_slurm_job('j', _SLURM_CFG, out, commands=['echo'])
        assert os.path.exists(out)

    # ── SBATCH header ─────────────────────────────────────────────────────

    def test_shebang(self, cmd_script):
        _, content = cmd_script
        assert content.startswith('#!/bin/bash')

    def test_job_name_in_header(self, cmd_script):
        _, content = cmd_script
        assert '#SBATCH --job-name=my_job' in content

    def test_ntasks_in_header(self, cmd_script):
        _, content = cmd_script
        assert '#SBATCH --ntasks=1' in content

    def test_cpus_per_task_in_header(self, cmd_script):
        _, content = cmd_script
        assert '#SBATCH --cpus-per-task=4' in content

    def test_partition_in_header(self, cmd_script):
        _, content = cmd_script
        assert '#SBATCH --partition=test-gpu' in content

    def test_time_in_header(self, cmd_script):
        _, content = cmd_script
        assert '#SBATCH --time=01:00:00' in content

    def test_gpu_line_present(self, cmd_script):
        _, content = cmd_script
        assert '#SBATCH --gres=gpu:a100:1' in content

    def test_gpu_line_absent_when_none(self, tmp_path):
        cfg = {**_SLURM_CFG, 'gpu': None}
        out = str(tmp_path / 'nogpu.sh')
        write_slurm_job('j', cfg, out, commands=['echo'])
        content = open(out).read()
        assert '--gres' not in content

    def test_default_output_log(self, cmd_script):
        out, content = cmd_script
        stem = os.path.splitext(out)[0]
        assert f'{stem}_%j.out' in content

    def test_custom_output_log(self, tmp_path):
        out = str(tmp_path / 'j.sh')
        custom_log = '/work/logs/job_%j.out'
        write_slurm_job('j', _SLURM_CFG, out,
                        commands=['echo'], output_log=custom_log)
        assert custom_log in open(out).read()

    # ── environment setup ─────────────────────────────────────────────────

    def test_ld_paths_exported(self, cmd_script):
        _, content = cmd_script
        assert 'export LD_LIBRARY_PATH=/fake/lib1' in content
        assert 'export LD_LIBRARY_PATH=/fake/lib2' in content

    def test_cuda_module_loaded(self, cmd_script):
        _, content = cmd_script
        assert 'module load cuda/12.3.0' in content

    def test_no_cuda_module_when_absent(self, tmp_path):
        cfg = {**_SLURM_CFG}
        cfg.pop('cuda_version', None)
        out = str(tmp_path / 'nocuda.sh')
        write_slurm_job('j', cfg, out, commands=['echo'])
        assert 'module load cuda' not in open(out).read()

    def test_openmpi_module_loaded(self, cmd_script):
        _, content = cmd_script
        assert 'module load OpenMPI/4.1.6' in content

    def test_conda_activated(self, cmd_script):
        _, content = cmd_script
        assert 'conda activate mace-env' in content

    def test_extra_env_vars_exported(self, tmp_path):
        out = str(tmp_path / 'envvars.sh')
        write_slurm_job('j', _SLURM_CFG, out,
                        commands=['echo'],
                        extra_env_vars={'MY_VAR': '42', 'SEED': '7'})
        content = open(out).read()
        assert 'export MY_VAR=42' in content
        assert 'export SEED=7' in content

    def test_ld_preload_written(self, tmp_path):
        cfg = {**_SLURM_CFG, 'ld_preload': '/fake/preload.so'}
        out = str(tmp_path / 'preload.sh')
        write_slurm_job('j', cfg, out, commands=['echo'])
        assert 'export LD_PRELOAD=/fake/preload.so' in open(out).read()

    # ── commands mode ─────────────────────────────────────────────────────

    def test_commands_written_verbatim(self, cmd_script):
        _, content = cmd_script
        assert 'echo hello' in content
        assert 'echo world' in content

    def test_multi_command_order(self, tmp_path):
        out = str(tmp_path / 'order.sh')
        write_slurm_job('j', _SLURM_CFG, out,
                        commands=['cmd_one', 'cmd_two', 'cmd_three'])
        lines = open(out).read().splitlines()
        pos = {c: next(i for i, l in enumerate(lines) if c in l)
               for c in ('cmd_one', 'cmd_two', 'cmd_three')}
        assert pos['cmd_one'] < pos['cmd_two'] < pos['cmd_three']

    # ── lmp runner mode ───────────────────────────────────────────────────

    def test_lmp_runner_command_line(self, tmp_path):
        out = str(tmp_path / 'lmp.sh')
        write_slurm_job('j', _SLURM_CFG, out,
                        script_path=_SCRIPT,
                        runner='lmp',
                        lammps_cmd=_LAMMPS,
                        lammps_log=_LOG)
        content = open(out).read()
        assert _LAMMPS in content
        assert f'-in {_SCRIPT}' in content
        assert f'-log {_LOG}' in content

    def test_lmp_runner_with_kokkos_flags(self, tmp_path):
        out = str(tmp_path / 'kk.sh')
        kk = ['-k', 'on', 'g', '1', '-sf', 'kk']
        write_slurm_job('j', _SLURM_CFG, out,
                        script_path=_SCRIPT,
                        runner='lmp',
                        lammps_cmd=_LAMMPS,
                        kokkos_flags=kk)
        content = open(out).read()
        assert '-k on g 1 -sf kk' in content

    def test_lmp_no_log_flag_when_absent(self, tmp_path):
        out = str(tmp_path / 'nolog.sh')
        write_slurm_job('j', _SLURM_CFG, out,
                        script_path=_SCRIPT,
                        runner='lmp',
                        lammps_cmd=_LAMMPS)
        assert '-log' not in open(out).read()

    # ── python runner mode ────────────────────────────────────────────────

    def test_python_runner(self, tmp_path):
        out = str(tmp_path / 'py.sh')
        write_slurm_job('j', _SLURM_CFG, out,
                        script_path='/work/run.py',
                        runner='python')
        assert 'python /work/run.py' in open(out).read()

    # ── array mode ────────────────────────────────────────────────────────

    def test_array_directive_present(self, tmp_path):
        out = str(tmp_path / 'arr.sh')
        write_slurm_job('j', _SLURM_CFG, out,
                        commands=['echo'],
                        array_range=(0, 9))
        assert '#SBATCH --array=0-9' in open(out).read()

    def test_array_with_concurrent_throttle(self, tmp_path):
        out = str(tmp_path / 'arr_throttle.sh')
        write_slurm_job('j', _SLURM_CFG, out,
                        commands=['echo'],
                        array_range=(0, 19),
                        concurrent=4)
        assert '#SBATCH --array=0-19%4' in open(out).read()

    def test_no_array_directive_when_not_set(self, cmd_script):
        _, content = cmd_script
        assert '--array' not in content

    # ── error handling ────────────────────────────────────────────────────

    def test_raises_when_no_commands_no_script(self, tmp_path):
        out = str(tmp_path / 'bad.sh')
        with pytest.raises(ValueError):
            write_slurm_job('j', _SLURM_CFG, out)

    def test_raises_lmp_runner_without_lammps_cmd(self, tmp_path):
        out = str(tmp_path / 'bad2.sh')
        with pytest.raises(ValueError):
            write_slurm_job('j', _SLURM_CFG, out,
                            script_path=_SCRIPT, runner='lmp')


# ═════════════════════════════════════════════════════════════════════════════
# 3. write_chained_slurm_job — edge cases
# (Phase-aware and legacy behaviour is tested in test_nvt_restart.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteChainedSlurmJobEdgeCases:

    @pytest.fixture()
    def legacy_script(self, tmp_path):
        out = str(tmp_path / 'chain.sh')
        write_chained_slurm_job(
            job_name='chain_test',
            slurm_config=_SLURM_CFG,
            out_path=out,
            first_commands=['echo first'],
            restart_commands=['echo restart'],
            restart_glob='checkpoints/*.restart',
            cutoff=_CUTOFF,
        )
        return out, open(out).read()

    # ── file / return value ───────────────────────────────────────────────

    def test_file_created(self, legacy_script):
        out, _ = legacy_script
        assert os.path.exists(out)

    def test_returns_path(self, tmp_path):
        out = str(tmp_path / 'chain.sh')
        ret = write_chained_slurm_job(
            'j', _SLURM_CFG, out,
            first_commands=['echo'],
            restart_commands=['echo'],
            restart_glob='*.restart',
            cutoff=_CUTOFF,
        )
        assert ret == out

    def test_script_executable(self, legacy_script):
        out, _ = legacy_script
        assert os.stat(out).st_mode & stat.S_IXUSR

    # ── cutoff converted to seconds ───────────────────────────────────────

    def test_cutoff_sec_embedded(self, legacy_script):
        _, content = legacy_script
        # '00:55:00' = 3300 seconds
        assert 'CUTOFF_SEC=3300' in content

    # ── restart glob ──────────────────────────────────────────────────────

    def test_restart_glob_embedded(self, legacy_script):
        _, content = legacy_script
        assert 'RESTART_GLOB="checkpoints/*.restart"' in content

    # ── SCRIPT_PATH self-reference ────────────────────────────────────────

    def test_script_path_baked_absolute(self, legacy_script):
        # $0 under sbatch is the spooled copy in /var/spool/slurmd, so the
        # .done/.failed sentinels and resubmission must use the real script
        # path baked in at generation time — never realpath "$0".
        out, content = legacy_script
        assert 'realpath "$0"' not in content
        assert f'SCRIPT_PATH="{os.path.abspath(out)}"' in content

    # ── flush wait ────────────────────────────────────────────────────────

    def test_default_flush_wait(self, legacy_script):
        _, content = legacy_script
        assert 'FLUSH_WAIT=30' in content

    def test_custom_flush_wait(self, tmp_path):
        out = str(tmp_path / 'fw.sh')
        write_chained_slurm_job(
            'j', _SLURM_CFG, out,
            first_commands=['echo'],
            restart_commands=['echo'],
            restart_glob='*.restart',
            cutoff=_CUTOFF,
            flush_wait=60,
        )
        assert 'FLUSH_WAIT=60' in open(out).read()

    # ── work_dir ──────────────────────────────────────────────────────────

    def test_work_dir_written(self, tmp_path):
        out = str(tmp_path / 'wd.sh')
        work = '/projects/mywork'
        write_chained_slurm_job(
            'j', _SLURM_CFG, out,
            first_commands=['echo'],
            restart_commands=['echo'],
            restart_glob='*.restart',
            cutoff=_CUTOFF,
            work_dir=work,
        )
        assert f'cd {work}' in open(out).read()

    def test_no_work_dir_no_cd(self, legacy_script):
        _, content = legacy_script
        # Only 'cd' that could appear is from the work_dir line;
        # with no work_dir the cd line is empty/absent
        cd_lines = [l for l in content.splitlines()
                    if l.strip().startswith('cd ') and '/projects' not in l]
        assert not cd_lines

    # ── extra env vars ────────────────────────────────────────────────────

    def test_extra_env_vars_written(self, tmp_path):
        out = str(tmp_path / 'env.sh')
        write_chained_slurm_job(
            'j', _SLURM_CFG, out,
            first_commands=['echo'],
            restart_commands=['echo'],
            restart_glob='*.restart',
            cutoff=_CUTOFF,
            extra_env_vars={'OMP_NUM_THREADS': '1'},
        )
        assert 'export OMP_NUM_THREADS=1' in open(out).read()

    # ── ld_preload ────────────────────────────────────────────────────────

    def test_ld_preload_written(self, tmp_path):
        cfg = {**_SLURM_CFG, 'ld_preload': '/lib/libfoo.so'}
        out = str(tmp_path / 'preload.sh')
        write_chained_slurm_job(
            'j', cfg, out,
            first_commands=['echo'],
            restart_commands=['echo'],
            restart_glob='*.restart',
            cutoff=_CUTOFF,
        )
        assert 'export LD_PRELOAD=/lib/libfoo.so' in open(out).read()

    # ── custom output_log ─────────────────────────────────────────────────

    def test_custom_output_log(self, tmp_path):
        out = str(tmp_path / 'chain.sh')
        custom = '/logs/chain_%j.out'
        write_chained_slurm_job(
            'j', _SLURM_CFG, out,
            first_commands=['echo'],
            restart_commands=['echo'],
            restart_glob='*.restart',
            cutoff=_CUTOFF,
            output_log=custom,
        )
        assert custom in open(out).read()

    # ── exit-code logic ───────────────────────────────────────────────────

    def test_exit_0_block_present(self, legacy_script):
        _, content = legacy_script
        assert '"$EXIT_CODE" -eq 0' in content

    def test_exit_124_block_present(self, legacy_script):
        _, content = legacy_script
        assert '"$EXIT_CODE" -eq 124' in content

    def test_sbatch_resubmit_present(self, legacy_script):
        _, content = legacy_script
        assert 'sbatch "$SCRIPT_PATH"' in content

    # ── legacy vs phase-aware branching ───────────────────────────────────

    def test_legacy_has_no_n_equil(self, legacy_script):
        _, content = legacy_script
        assert 'N_EQUIL=' not in content

    def test_phase_aware_has_n_equil(self, tmp_path):
        out = str(tmp_path / 'pa.sh')
        write_chained_slurm_job(
            'j', _SLURM_CFG, out,
            first_commands=['echo fresh'],
            equil_restart_commands=['echo equil'],
            prod_restart_commands=['echo prod'],
            restart_glob='*.restart',
            cutoff=_CUTOFF,
            n_equil=2_000_000,
        )
        content = open(out).read()
        assert 'N_EQUIL=2000000' in content
        assert 'LAST_STEP=' in content
        assert '"$LAST_STEP" -lt "$N_EQUIL"' in content


# ═══════════════════════════════════════════════════════════════════════════
# check_jobs — squeue parsing, including SLURM array job ids
# ═══════════════════════════════════════════════════════════════════════════

def _mock_squeue(stdout):
    return patch('models.create_slurm.subprocess.run',
                 return_value=MagicMock(stdout=stdout))


class TestCheckJobsArrayIds:
    """Regression tests for a real bug found on the cluster: an array job
    submitted via write_slurm_job(array_range=...) is tracked by its bare
    parent job id (e.g. '8175100'), but squeue reports each array TASK as
    '<jid>_<task_id>' (or a compact pending range '<jid>_[0-2]') -- the
    bare id never appears verbatim. check_jobs' exact-match lookup missed
    this entirely, so wait_for_jobs() returned 'done' on its very first
    poll for any array submission, even with every task still queued —
    Hop A/B NEB's real barriers were silently collected before the NEB
    jobs had actually finished."""

    def test_array_job_fully_running_is_not_done(self):
        with _mock_squeue('8175100_1 RUNNING\n8175100_2 PENDING\n8175100_3 RUNNING\n'):
            statuses = check_jobs({'hopa_neb': '8175100'}, verbose=False)
        assert statuses['hopa_neb'] != 'done'

    def test_array_job_compact_pending_range_is_not_done(self):
        # SLURM shows an untouched pending array as one compact line
        # before any task has started.
        with _mock_squeue('8175100_[1-3] PENDING\n'):
            statuses = check_jobs({'hopa_neb': '8175100'}, verbose=False)
        assert statuses['hopa_neb'] != 'done'

    def test_array_job_all_tasks_finished_is_done(self):
        with _mock_squeue(''):
            statuses = check_jobs({'hopa_neb': '8175100'}, verbose=False)
        assert statuses['hopa_neb'] == 'done'

    def test_array_job_partially_finished_is_not_done(self):
        # 1 of 3 tasks left the queue; 2 still remain -- must still wait.
        with _mock_squeue('8175100_2 RUNNING\n'):
            statuses = check_jobs({'hopa_neb': '8175100'}, verbose=False)
        assert statuses['hopa_neb'] != 'done'

    def test_non_array_job_running_unaffected(self):
        with _mock_squeue('999999 RUNNING\n'):
            statuses = check_jobs({'slab_min': '999999'}, verbose=False)
        assert statuses['slab_min'] == 'running'

    def test_non_array_job_done_unaffected(self):
        with _mock_squeue(''):
            statuses = check_jobs({'slab_min': '999999'}, verbose=False)
        assert statuses['slab_min'] == 'done'

    def test_does_not_confuse_unrelated_job_id_prefix(self):
        # '81751000' must not be mistaken for a task of array '8175100'.
        with _mock_squeue('81751000 RUNNING\n'):
            statuses = check_jobs({'hopa_neb': '8175100'}, verbose=False)
        assert statuses['hopa_neb'] == 'done'

    def test_mixed_array_and_non_array_jobs_tracked_independently(self):
        with _mock_squeue('8175100_1 RUNNING\n999999 RUNNING\n'):
            statuses = check_jobs(
                {'hopa_neb': '8175100', 'slab_min': '999999'}, verbose=False)
        assert statuses['hopa_neb'] != 'done'
        assert statuses['slab_min'] != 'done'


class TestSubmitWithRetry:
    """submit_with_retry() wraps submit_slurm_job() with a fixed-interval
    backoff for transient account-wide failures (QOSMaxSubmitJobPerUserLimit)
    that a single-job (non-array) submission has no other protection
    against -- unlike auto_submit()'s arrays, which throttle queue depth
    via polling, a bare submit_slurm_job() call previously gave up
    immediately on any sbatch failure."""

    def test_succeeds_immediately_without_sleeping(self, monkeypatch):
        calls = []
        monkeypatch.setattr('models.create_slurm.submit_slurm_job',
                             lambda *a, **kw: (calls.append(1), '12345')[1])
        sleep_calls = []
        monkeypatch.setattr('models.create_slurm.time.sleep',
                             lambda s: sleep_calls.append(s))

        job_id = submit_with_retry('job.sh')

        assert job_id == '12345'
        assert len(calls) == 1
        assert sleep_calls == []

    def test_retries_then_succeeds(self, monkeypatch):
        results = iter([None, None, '999'])
        monkeypatch.setattr('models.create_slurm.submit_slurm_job',
                             lambda *a, **kw: next(results))
        sleep_calls = []
        monkeypatch.setattr('models.create_slurm.time.sleep',
                             lambda s: sleep_calls.append(s))

        job_id = submit_with_retry('job.sh', retry_interval=30)

        assert job_id == '999'
        assert sleep_calls == [30, 30]

    def test_exhausts_retries_and_returns_none(self, monkeypatch):
        call_count = []
        monkeypatch.setattr(
            'models.create_slurm.submit_slurm_job',
            lambda *a, **kw: (call_count.append(1), None)[1],
        )
        sleep_calls = []
        monkeypatch.setattr('models.create_slurm.time.sleep',
                             lambda s: sleep_calls.append(s))

        job_id = submit_with_retry('job.sh', max_retries=3, retry_interval=5)

        assert job_id is None
        assert len(call_count) == 3
        # Sleeps only happen BETWEEN attempts, never after the last one.
        assert sleep_calls == [5, 5]

    def test_passes_through_extra_args_and_dependency(self, monkeypatch):
        captured = {}
        def _fake_submit(slurm_path, extra_args=None, dependency=None):
            captured['slurm_path'] = slurm_path
            captured['extra_args'] = extra_args
            captured['dependency'] = dependency
            return 'jid'
        monkeypatch.setattr('models.create_slurm.submit_slurm_job', _fake_submit)

        submit_with_retry('job.sh', extra_args=['--array=1,2'], dependency='afterok:1')

        assert captured['slurm_path'] == 'job.sh'
        assert captured['extra_args'] == ['--array=1,2']
        assert captured['dependency'] == 'afterok:1'

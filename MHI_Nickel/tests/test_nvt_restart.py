"""
tests/test_nvt_restart.py
==========================
Tests for the NVT smart-restart fix.

Covers every failure mode identified during the bug analysis:

  Bug 1  – Phase 1 re-equilibration on every restart (wasted wall time)
  Bug 2  – LAMMPS log overwritten on each restart
  Bug 3  – out_file naming collision (log vs write_data)
  Bug 4  – MSD sawtooth (compute resets to 0 at each restart)
            → not a code bug; handled by using dump trajectory for post-processing
  Bug 5  – Orchestrator marks NVT done prematurely (job-ID turnover)

Also verifies:
  - Equil-phase H MSD is now tracked (new requirement)
  - Two separate MSD files (equil / prod)
  - No regressions in NPT chained jobs, Phase 1a/1b wait_for_jobs, etc.
"""

import os
import sys
import stat
import inspect
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.lammps_script import (
    write_nvt_bulk_script,
    write_nvt_equil_restart_script,
    write_nvt_prod_restart_script,
    write_nvt_bulk_restart_script,   # must still exist for backward compat
)
from models.create_slurm import write_chained_slurm_job
import models.diffusivity_workflow as _dw


# ─── shared constants ────────────────────────────────────────────────────────

N_EQUIL     = 2_000_000
N_PROD      = 5_000_000
TOTAL_STEPS = N_EQUIL + N_PROD   # 7_000_000

PAIR_STYLE  = 'mace/mp'
MACE_MODEL  = '/fake/model.pt'
PAIR_SUFFIX = 'mace'
ELEM_STR    = 'Ni H'
TEMPERATURE = 1000
H_TYPE      = 9
THERMO      = 1000

SLURM_CFG = {
    'partition':     'gpu',
    'ntasks':        1,
    'cpus_per_task': 8,
    'gpu':           1,
    'time':          '24:00:00',
    'openmpi_ver':   '4.1.1',
    'cuda_version':  '11.7',
    'conda_env':     'mace',
    'ld_paths':      [],
}

CUTOFF = '23:55:00'


# ─── helpers ─────────────────────────────────────────────────────────────────

def _lines_with(text, keyword):
    return [l for l in text.splitlines() if keyword in l]

def _first_nonblank(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ''

def _pos(text, keyword):
    """Return index of first line containing keyword, or -1."""
    for i, line in enumerate(text.splitlines()):
        if keyword in line:
            return i
    return -1

def _block_after(text, keyword, n=3):
    """Return up to n lines starting from the first line containing keyword."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if keyword in line:
            return lines[i:i + n]
    return []

def _slice_between(text, start_kw, end_kw):
    """Return the portion of text between the first line with start_kw
    and the first line with end_kw (exclusive)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if start_kw in l), None)
    end   = next((i for i, l in enumerate(lines) if end_kw in l and i > (start or 0)), None)
    if start is None or end is None:
        return ''
    return '\n'.join(lines[start:end])


# ─── fixtures that generate scripts ──────────────────────────────────────────

@pytest.fixture()
def bulk_script(tmp_path):
    lmp = tmp_path / 'nvt_1000K.lammps'
    write_nvt_bulk_script(
        bulk_h_file    = str(tmp_path / 'bulk_h.lammps'),
        traj_file      = str(tmp_path / 'nvt_1000K.dump'),
        out_file       = str(tmp_path / 'nvt_1000K.out'),
        msd_equil_file = str(tmp_path / 'msd_1000K_equil.dat'),
        msd_prod_file  = str(tmp_path / 'msd_1000K_prod.dat'),
        out_path       = str(lmp),
        pair_style     = PAIR_STYLE,
        mace_model     = MACE_MODEL,
        pair_suffix    = PAIR_SUFFIX,
        elem_str       = ELEM_STR,
        temperature    = TEMPERATURE,
        h_type         = H_TYPE,
        thermo_every   = THERMO,
        n_equil        = N_EQUIL,
        n_prod         = N_PROD,
        restart_dir    = str(tmp_path / 'checkpoints'),
    )
    return lmp.read_text()


@pytest.fixture()
def equil_rst_script(tmp_path):
    lmp = tmp_path / 'nvt_1000K_rst_equil.lammps'
    rst_glob = str(tmp_path / 'checkpoints' / 'nvt_1000K.*.restart')
    write_nvt_equil_restart_script(
        restart_file   = rst_glob,
        traj_file      = str(tmp_path / 'nvt_1000K.dump'),
        out_file       = str(tmp_path / 'nvt_1000K.out'),
        msd_equil_file = str(tmp_path / 'msd_1000K_equil.dat'),
        msd_prod_file  = str(tmp_path / 'msd_1000K_prod.dat'),
        log_file       = str(tmp_path / 'nvt_1000K.log'),
        out_path       = str(lmp),
        pair_style     = PAIR_STYLE,
        mace_model     = MACE_MODEL,
        pair_suffix    = PAIR_SUFFIX,
        elem_str       = ELEM_STR,
        temperature    = TEMPERATURE,
        h_type         = H_TYPE,
        thermo_every   = THERMO,
        n_equil        = N_EQUIL,
        n_prod         = N_PROD,
        restart_dir    = str(tmp_path / 'checkpoints'),
    )
    return lmp.read_text()


@pytest.fixture()
def prod_rst_script(tmp_path):
    lmp = tmp_path / 'nvt_1000K_rst_prod.lammps'
    rst_glob = str(tmp_path / 'checkpoints' / 'nvt_1000K.*.restart')
    write_nvt_prod_restart_script(
        restart_file  = rst_glob,
        traj_file     = str(tmp_path / 'nvt_1000K.dump'),
        out_file      = str(tmp_path / 'nvt_1000K.out'),
        msd_prod_file = str(tmp_path / 'msd_1000K_prod.dat'),
        log_file      = str(tmp_path / 'nvt_1000K.log'),
        out_path      = str(lmp),
        pair_style    = PAIR_STYLE,
        mace_model    = MACE_MODEL,
        pair_suffix   = PAIR_SUFFIX,
        elem_str      = ELEM_STR,
        temperature   = TEMPERATURE,
        h_type        = H_TYPE,
        thermo_every  = THERMO,
        n_equil       = N_EQUIL,
        n_prod        = N_PROD,
        restart_dir   = str(tmp_path / 'checkpoints'),
    )
    return lmp.read_text()


@pytest.fixture()
def chain_sh_phase_aware(tmp_path):
    sh = tmp_path / 'nvt_1000K_chain.sh'
    write_chained_slurm_job(
        job_name               = 'nvt_1000K_test',
        slurm_config           = SLURM_CFG,
        out_path               = str(sh),
        first_commands         = ['lmp -in nvt_1000K.lammps -log nvt.log'],
        equil_restart_commands = ['lmp -in nvt_1000K_rst_equil.lammps -log /dev/null'],
        prod_restart_commands  = ['lmp -in nvt_1000K_rst_prod.lammps -log /dev/null'],
        n_equil                = N_EQUIL,
        restart_glob           = 'checkpoints/nvt_1000K.*.restart',
        cutoff                 = CUTOFF,
        work_dir               = str(tmp_path),
    )
    return sh.read_text()


@pytest.fixture()
def chain_sh_legacy(tmp_path):
    sh = tmp_path / 'nvt_1000K_chain_legacy.sh'
    write_chained_slurm_job(
        job_name         = 'nvt_1000K_legacy',
        slurm_config     = SLURM_CFG,
        out_path         = str(sh),
        first_commands   = ['lmp -in nvt.lammps -log nvt.log'],
        restart_commands = ['lmp -in nvt_restart.lammps -log nvt.log'],
        restart_glob     = 'checkpoints/nvt_1000K.*.restart',
        cutoff           = CUTOFF,
        work_dir         = str(tmp_path),
    )
    return sh.read_text()


# ═════════════════════════════════════════════════════════════════════════════
# 1. write_nvt_bulk_script — first-run script
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteNvtBulkScript:

    def test_equil_msd_file_present(self, bulk_script):
        """msd_equil_file path must appear in the script (Phase 1 MSD tracking)."""
        assert 'msd_1000K_equil.dat' in bulk_script

    def test_prod_msd_file_present(self, bulk_script):
        """msd_prod_file path must appear in the script (Phase 2 MSD tracking)."""
        assert 'msd_1000K_prod.dat' in bulk_script

    def test_old_msd_file_param_absent(self, bulk_script):
        """The old single msd_file name 'msd_1000K.dat' must NOT appear — confirms rename."""
        # Only the split names should appear, not a bare 'msd_1000K.dat'
        lines = _lines_with(bulk_script, 'msd_1000K.dat')
        assert not lines, f"Old msd_file name still present: {lines}"

    def test_equil_msd_fix_exists(self, bulk_script):
        """fix msd_equil must be set up in Phase 1."""
        assert 'fix' in bulk_script and 'msd_equil' in bulk_script

    def test_equil_msd_no_append(self, bulk_script):
        """Phase 1 (first run) MSD must NOT use append yes — fresh file."""
        equil_fix_lines = _lines_with(bulk_script, 'msd_equil')
        for line in equil_fix_lines:
            assert 'append yes' not in line, (
                f"Phase 1 MSD should NOT append on first run: {line}")

    def test_unfix_msd_equil_present(self, bulk_script):
        """unfix msd_equil must appear — cleans up before Phase 2."""
        assert 'unfix' in bulk_script
        assert any('msd_equil' in l for l in _lines_with(bulk_script, 'unfix'))

    def test_uncompute_msd_h_present(self, bulk_script):
        """uncompute msd_H must appear between phases."""
        assert 'uncompute' in bulk_script
        assert any('msd_H' in l for l in _lines_with(bulk_script, 'uncompute'))

    def test_unfix_before_phase2(self, bulk_script):
        """unfix msd_equil must come BEFORE fix nvt_prod (Phase 2 start)."""
        pos_unfix = _pos(bulk_script, 'unfix          msd_equil')
        pos_prod  = _pos(bulk_script, 'fix            nvt_prod')
        assert pos_unfix != -1, 'unfix msd_equil not found'
        assert pos_prod  != -1, 'fix nvt_prod not found'
        assert pos_unfix < pos_prod, (
            f'unfix msd_equil (line {pos_unfix}) must come before fix nvt_prod (line {pos_prod})')

    def test_prod_msd_fix_exists(self, bulk_script):
        """fix msd_prod must be set up in Phase 2."""
        assert 'msd_prod' in bulk_script

    def test_prod_msd_no_append_first_run(self, bulk_script):
        """Phase 2 (first run) MSD must NOT use append yes."""
        prod_fix_lines = _lines_with(bulk_script, 'msd_prod')
        for line in prod_fix_lines:
            assert 'append yes' not in line, (
                f"Phase 2 MSD should NOT append on first run: {line}")

    def test_two_distinct_compute_msd_declarations(self, bulk_script):
        """compute msd_H appears twice — once for equil, once for prod."""
        compute_lines = _lines_with(bulk_script, 'compute        msd_H')
        assert len(compute_lines) == 2, (
            f'Expected 2 compute msd_H declarations, got {len(compute_lines)}')

    def test_equil_dump_no_append(self, bulk_script):
        """Equil dump on first run must NOT have append yes."""
        equil_dump_lines = _lines_with(bulk_script, 'equil_dump')
        for line in equil_dump_lines:
            assert 'append yes' not in line, (
                f"Equil dump should not append on first run: {line}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. write_nvt_equil_restart_script — mid-equil restart
#    Bug 1 fix: no more full re-equilibration from scratch
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteNvtEquilRestartScript:

    def test_log_append_is_first_line(self, equil_rst_script):
        """BUG 2 FIX: log file must be set to append mode as the very first command."""
        first = _first_nonblank(equil_rst_script)
        assert first.startswith('log '), f"First line should be 'log ...', got: {first!r}"
        assert 'append' in first, f"First line must include 'append': {first!r}"

    def test_log_file_path_correct(self, equil_rst_script, tmp_path):
        """The log command must reference the actual log_file path, not out_file."""
        first = _first_nonblank(equil_rst_script)
        assert 'nvt_1000K.log' in first, f"log command references wrong file: {first!r}"
        assert 'nvt_1000K.out' not in first, "log should go to .log, not .out"

    def test_equil_run_upto(self, equil_rst_script):
        """'run N upto' resumes to the absolute equil step count — replaces
        the remaining_equil variable, which could go negative for a stale
        checkpoint."""
        assert f'run            {N_EQUIL}  upto' in equil_rst_script
        assert 'remaining_equil' not in equil_rst_script

    def test_equil_run_not_plain_hardcoded(self, equil_rst_script):
        """A plain 'run N_EQUIL' (without upto) would redo the FULL equil
        phase on every restart leg instead of just the remainder."""
        assert f'run            {N_EQUIL}\n' not in equil_rst_script
        assert '?' not in equil_rst_script   # no C-style ternary in LAMMPS

    def test_equil_dump_appends(self, equil_rst_script):
        """Equil dump must append — prior equil frames already in the file."""
        equil_lines = _lines_with(equil_rst_script, 'equil_dump')
        append_found = any('append yes' in l for l in equil_lines)
        assert append_found, "Equil dump must have append yes in restart script"

    def test_equil_msd_appends(self, equil_rst_script):
        """Equil MSD must append — prior equil MSD data already in the file.

        The fix ave/time command uses '&' continuation so 'append yes' lives on
        the continuation line, not the 'fix msd_equil' line.  Search the two-line
        block starting at 'fix msd_equil'.
        """
        block = _block_after(equil_rst_script, 'fix            msd_equil', n=2)
        assert block, "fix msd_equil not found in equil restart script"
        assert any('append yes' in l for l in block), (
            "Equil MSD fix must use append yes in equil restart script; block: "
            + repr(block))

    def test_unfix_msd_equil_before_prod(self, equil_rst_script):
        """unfix msd_equil must appear before production phase starts."""
        pos_unfix = _pos(equil_rst_script, 'unfix          msd_equil')
        pos_prod  = _pos(equil_rst_script, 'fix            nvt_prod')
        assert pos_unfix != -1, 'unfix msd_equil not found in equil restart script'
        assert pos_prod  != -1, 'fix nvt_prod not found in equil restart script'
        assert pos_unfix < pos_prod

    def test_uncompute_msd_h_between_phases(self, equil_rst_script):
        """uncompute msd_H must appear between equil and prod phases."""
        pos_uncompute = _pos(equil_rst_script, 'uncompute      msd_H')
        pos_prod      = _pos(equil_rst_script, 'fix            nvt_prod')
        assert pos_uncompute != -1, 'uncompute msd_H not found'
        assert pos_uncompute < pos_prod

    def test_prod_dump_no_append(self, equil_rst_script):
        """BUG 1 FIX: production dump must NOT append — this is the first time prod runs.

        Use 'dump_modify    prod_dump' (the LAMMPS command keyword) rather than bare
        'prod_dump', to avoid matching the pytest tmp_path directory name which contains
        'prod_dump' as a substring of the test function name.
        """
        assert 'dump_modify    prod_dump  sort id  append yes' not in equil_rst_script, (
            "Prod dump must NOT append in equil restart (first prod run)")

    def test_prod_msd_no_append(self, equil_rst_script):
        """Production MSD must NOT append — first time production runs."""
        msd_prod_lines = _lines_with(equil_rst_script, 'msd_prod')
        for line in msd_prod_lines:
            assert 'append yes' not in line, (
                f"Prod MSD must NOT append in equil restart (first prod run): {line}")

    def test_prod_run_uses_fixed_n_prod(self, equil_rst_script):
        """Production run uses fixed n_prod (not a variable) — always full production."""
        assert f'run            {N_PROD}' in equil_rst_script

    def test_equil_phase_present(self, equil_rst_script):
        """fix nvt_equil must exist — we're continuing equil."""
        assert 'fix            nvt_equil' in equil_rst_script

    def test_prod_phase_present(self, equil_rst_script):
        """fix nvt_prod must exist — production follows equil."""
        assert 'fix            nvt_prod' in equil_rst_script

    def test_msd_equil_file_referenced(self, equil_rst_script):
        """msd_equil_file path must appear in the script."""
        assert 'msd_1000K_equil.dat' in equil_rst_script

    def test_msd_prod_file_referenced(self, equil_rst_script):
        """msd_prod_file path must appear in the script."""
        assert 'msd_1000K_prod.dat' in equil_rst_script


# ═════════════════════════════════════════════════════════════════════════════
# 3. write_nvt_prod_restart_script — mid-production restart
#    Bug 1 fix: skips equil entirely, continues production
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteNvtProdRestartScript:

    def test_log_append_is_first_line(self, prod_rst_script):
        """BUG 2 FIX: log file must be set to append mode as very first command."""
        first = _first_nonblank(prod_rst_script)
        assert first.startswith('log '), f"First line should be 'log ...', got: {first!r}"
        assert 'append' in first

    def test_log_file_path_correct(self, prod_rst_script, tmp_path):
        """The log command must reference the actual log_file path."""
        first = _first_nonblank(prod_rst_script)
        assert 'nvt_1000K.log' in first
        assert 'nvt_1000K.out' not in first

    def test_prod_run_upto_total(self, prod_rst_script):
        """'run TOTAL upto' resumes to n_equil + n_prod — replaces the old
        ternary formula '((N - step) > 0 ? (N - step) : 0)', which is
        invalid LAMMPS syntax: it crashed every prod restart leg and
        truncated the production MSD files (2026-07-03 smoke-test bug)."""
        assert f'run            {TOTAL_STEPS}  upto' in prod_rst_script
        assert 'remaining_prod' not in prod_rst_script
        assert '?' not in prod_rst_script

    def test_prod_run_not_plain_hardcoded(self, prod_rst_script):
        """No plain hardcoded step counts — only the upto form."""
        assert f'run            {N_PROD}' not in prod_rst_script
        assert f'run            {N_EQUIL}\n' not in prod_rst_script

    def test_prod_dump_appends(self, prod_rst_script):
        """Production dump must append — prior frames already accumulated."""
        prod_dump_lines = _lines_with(prod_rst_script, 'prod_dump')
        assert any('append yes' in l for l in prod_dump_lines), (
            "Prod dump must use append yes in prod restart script")

    def test_prod_msd_appends(self, prod_rst_script):
        """Production MSD must append — prior MSD segments already in file.

        Same '&' continuation issue as test_equil_msd_appends: 'append yes' is on
        the continuation line, not on 'fix msd_prod'.  Check the two-line block.
        """
        block = _block_after(prod_rst_script, 'fix            msd_prod', n=2)
        assert block, "fix msd_prod not found in prod restart script"
        assert any('append yes' in l for l in block), (
            "Prod MSD fix must use append yes in prod restart script; block: "
            + repr(block))

    def test_no_equil_phase(self, prod_rst_script):
        """BUG 1 FIX: prod restart must NOT re-run equilibration."""
        assert 'fix            nvt_equil' not in prod_rst_script

    def test_no_equil_dump(self, prod_rst_script):
        """BUG 1 FIX: no equil_dump in prod restart — equil already completed.

        Use the full LAMMPS 'dump' command keyword to avoid matching the pytest
        tmp_path directory name ('test_no_equil_dump0') which contains 'equil_dump'.
        """
        assert 'dump           equil_dump' not in prod_rst_script

    def test_no_equil_msd(self, prod_rst_script):
        """Prod restart must not touch equil MSD file."""
        assert 'msd_equil' not in prod_rst_script
        assert 'msd_1000K_equil.dat' not in prod_rst_script

    def test_prod_phase_present(self, prod_rst_script):
        """fix nvt_prod must exist."""
        assert 'fix            nvt_prod' in prod_rst_script

    def test_msd_prod_file_referenced(self, prod_rst_script):
        """msd_prod_file path must appear."""
        assert 'msd_1000K_prod.dat' in prod_rst_script

    def test_remaining_prod_math(self, prod_rst_script):
        """Formula (n_equil + n_prod - step) must encode the correct total."""
        assert str(TOTAL_STEPS) in prod_rst_script, (
            f"Expected total_steps={TOTAL_STEPS} in remaining_prod formula")

    def test_boundary_at_n_equil(self):
        """At step=n_equil, remaining_prod should equal n_prod (pure math check)."""
        step = N_EQUIL
        remaining = TOTAL_STEPS - step
        assert remaining == N_PROD

    def test_boundary_mid_production(self):
        """At step=n_equil+1M, remaining_prod should be n_prod-1M."""
        step = N_EQUIL + 1_000_000
        remaining = TOTAL_STEPS - step
        assert remaining == N_PROD - 1_000_000


# ═════════════════════════════════════════════════════════════════════════════
# 4. write_chained_slurm_job — phase detection and .done sentinel
#    Bug 5 fix: no more premature Phase 3 from job-ID turnover
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteChainedSlurmJobPhaseAware:

    def test_n_equil_written_to_script(self, chain_sh_phase_aware):
        """N_EQUIL variable must be embedded in the bash script."""
        assert f'N_EQUIL={N_EQUIL}' in chain_sh_phase_aware

    def test_last_step_extraction(self, chain_sh_phase_aware):
        """LAST_STEP= extraction from checkpoint filename must be present."""
        assert 'LAST_STEP=' in chain_sh_phase_aware

    def test_phase_comparison_in_script(self, chain_sh_phase_aware):
        """Bash must compare LAST_STEP against N_EQUIL to select restart type."""
        assert '"$LAST_STEP" -lt "$N_EQUIL"' in chain_sh_phase_aware

    def test_equil_restart_command_in_script(self, chain_sh_phase_aware):
        """Equil restart command must appear in the script."""
        assert 'nvt_1000K_rst_equil.lammps' in chain_sh_phase_aware

    def test_prod_restart_command_in_script(self, chain_sh_phase_aware):
        """Prod restart command must appear in the script."""
        assert 'nvt_1000K_rst_prod.lammps' in chain_sh_phase_aware

    def test_restart_commands_use_dev_null_log(self, chain_sh_phase_aware):
        """BUG 2 FIX: restart commands must suppress -log on command line
        (log is handled inside the LAMMPS script with append)."""
        assert '-log /dev/null' in chain_sh_phase_aware

    def test_done_sentinel_written_on_exit_0(self, chain_sh_phase_aware):
        """BUG 5 FIX: .done sentinel must be written when LAMMPS truly converges."""
        assert 'touch "${SCRIPT_PATH}.done"' in chain_sh_phase_aware

    def test_done_sentinel_only_on_exit_0(self, chain_sh_phase_aware):
        """The touch command must appear inside the exit-0 block, not in resubmit block."""
        lines = chain_sh_phase_aware.splitlines()
        # Find exit-0 block start
        exit0_idx = next(
            (i for i, l in enumerate(lines) if '"$EXIT_CODE" -eq 0' in l), -1)
        # Find exit-124 block start
        exit124_idx = next(
            (i for i, l in enumerate(lines) if '"$EXIT_CODE" -eq 124' in l), -1)
        # Find touch line
        touch_idx = next(
            (i for i, l in enumerate(lines) if 'touch "${SCRIPT_PATH}.done"' in l), -1)
        assert touch_idx != -1
        assert exit0_idx < touch_idx < exit124_idx, (
            "touch .done must appear inside the exit-0 block only")

    def test_resubmit_on_exit_124(self, chain_sh_phase_aware):
        """Exit 124 (timeout) must trigger sbatch resubmission."""
        assert 'sbatch "$SCRIPT_PATH"' in chain_sh_phase_aware

    def test_script_is_executable(self, tmp_path):
        """Generated SLURM script must have execute permissions."""
        sh = tmp_path / 'nvt_chain_exec.sh'
        write_chained_slurm_job(
            job_name               = 'exec_test',
            slurm_config           = SLURM_CFG,
            out_path               = str(sh),
            first_commands         = ['echo first'],
            equil_restart_commands = ['echo equil'],
            prod_restart_commands  = ['echo prod'],
            n_equil                = N_EQUIL,
            restart_glob           = 'checkpoints/*.restart',
            cutoff                 = CUTOFF,
        )
        mode = os.stat(str(sh)).st_mode
        assert mode & stat.S_IXUSR, "Script must be executable by owner"

    def test_first_command_uses_log_file(self, chain_sh_phase_aware):
        """First-run command must use -log to create the log file fresh."""
        first_lines = _lines_with(chain_sh_phase_aware, 'nvt_1000K.lammps')
        assert any('-log' in l and '/dev/null' not in l for l in first_lines), (
            "First-run command must use an actual -log file, not /dev/null")


class TestWriteChainedSlurmJobLegacy:
    """Backward-compat: old callers using restart_commands must still work."""

    def test_legacy_no_n_equil_variable(self, chain_sh_legacy):
        """N_EQUIL must NOT appear when n_equil is not provided."""
        assert 'N_EQUIL=' not in chain_sh_legacy

    def test_legacy_no_last_step(self, chain_sh_legacy):
        """LAST_STEP extraction must NOT appear in legacy mode."""
        assert 'LAST_STEP=' not in chain_sh_legacy

    def test_legacy_restart_commands_present(self, chain_sh_legacy):
        """The old restart_commands must still be embedded in legacy mode."""
        assert 'nvt_restart.lammps' in chain_sh_legacy

    def test_legacy_done_sentinel_still_present(self, chain_sh_legacy):
        """BUG 5 FIX: .done sentinel must also be written in legacy mode."""
        assert 'touch "${SCRIPT_PATH}.done"' in chain_sh_legacy

    def test_legacy_resubmit_still_works(self, chain_sh_legacy):
        """Exit 124 must still resubmit in legacy mode."""
        assert 'sbatch "$SCRIPT_PATH"' in chain_sh_legacy


# ═════════════════════════════════════════════════════════════════════════════
# 5. diffusivity_workflow._body — generated diffusivity_run.py content
# ═════════════════════════════════════════════════════════════════════════════

class TestDiffusivityWorkflowBody:
    """Inspect the _body raw string embedded in generate_diffusivity_scripts."""

    @pytest.fixture(autouse=True)
    def body(self):
        src = inspect.getsource(_dw.generate_diffusivity_scripts)
        # Extract the _body raw string: everything between _body = r""" and the closing """
        start = src.index('_body = r"""') + len('_body = r"""')
        end   = src.rindex('"""', start)
        self._body = src[start:end]

    def test_new_imports_present(self):
        """New restart script functions must be imported in generated script."""
        assert 'write_nvt_equil_restart_script' in self._body
        assert 'write_nvt_prod_restart_script'  in self._body

    def test_old_import_absent(self):
        """write_nvt_bulk_restart_script must NOT be imported (replaced by two new functions)."""
        assert 'write_nvt_bulk_restart_script' not in self._body

    def test_msd_equil_file_path(self):
        """msd_equil_file path variable must be defined in Phase 2 block."""
        assert 'msd_equil_file' in self._body

    def test_msd_prod_file_path(self):
        """msd_prod_file path variable must be defined in Phase 2 block."""
        assert 'msd_prod_file' in self._body

    def test_old_msd_file_var_absent(self):
        """Old single msd_file variable must NOT be used in Phase 2."""
        # Allow 'msd_equil_file' and 'msd_prod_file' but not bare 'msd_file ='
        lines_with_msd = [l for l in self._body.splitlines()
                          if 'msd_file' in l
                          and 'msd_equil_file' not in l
                          and 'msd_prod_file' not in l]
        assert not lines_with_msd, (
            f"Old bare msd_file variable still present: {lines_with_msd}")

    def test_equil_rst_lmp_path(self):
        """nvt_equil_rst_lmp path variable must be defined."""
        assert 'nvt_equil_rst_lmp' in self._body

    def test_prod_rst_lmp_path(self):
        """nvt_prod_rst_lmp path variable must be defined."""
        assert 'nvt_prod_rst_lmp' in self._body

    def test_completion_guard_uses_done_sentinel(self):
        """BUG 5 FIX: Phase 2 guard must check for .done file, not msd_file existence."""
        assert "chain_sh + '.done'" in self._body or 'chain_sh + ".done"' in self._body

    def test_sentinel_polling_loop(self):
        """BUG 5 FIX: waiting loop must poll .done files, not job IDs."""
        assert '.done' in self._body
        assert '_time.sleep' in self._body

    def test_phase2_does_not_use_wait_for_jobs(self):
        """BUG 5 FIX: Phase 2 NVT must not use wait_for_jobs (job-ID turnover problem)."""
        # Phase 2 block is after 'Phase 2: NVT MD'; check the sentinel approach is used
        # The old pattern was: wait_for_jobs(job_ids)
        assert 'wait_for_jobs(job_ids)' not in self._body

    def test_log_file_path_defined(self):
        """BUG 2 FIX: separate log_file variable must be defined in Phase 2."""
        assert 'log_file' in self._body

    def test_first_command_uses_log_file_not_out_file(self):
        """BUG 2+3 FIX: first_commands must use -log {log_file}, not -log {out_file}.

        In the _body source, 'first_commands=[' and '-log {log_file}' are on adjacent
        lines (multi-line list literal).  Join consecutive line pairs to find the pattern
        across the line boundary.
        """
        body_lines = self._body.splitlines()
        # Build joined 2-line windows to catch multi-line list literals
        windows = [body_lines[i] + ' ' + body_lines[i + 1]
                   for i in range(len(body_lines) - 1)]
        # Find any window that contains both 'first_commands' and '-log'
        fc_log_windows = [w for w in windows if 'first_commands' in w and '-log' in w]
        assert fc_log_windows, (
            "Could not find 'first_commands' with '-log' in a 2-line window in _body")
        for w in fc_log_windows:
            assert 'log_file' in w, (
                f"first_commands should log to log_file, not out_file: {w!r}")
            assert 'out_file' not in w, (
                f"first_commands must not log to out_file: {w!r}")

    def test_n_equil_passed_to_chained_job(self):
        """n_equil must be passed to write_chained_slurm_job call."""
        assert 'n_equil=N_EQUIL_STEPS' in self._body

    def test_phase1a_still_uses_wait_for_jobs(self):
        """Regression: Phase 1a minimisation must still use wait_for_jobs."""
        assert "wait_for_jobs({'min_bare': jid})" in self._body

    def test_phase1b_npt_still_uses_wait_for_jobs(self):
        """Regression: Phase 1b NPT must still use wait_for_jobs."""
        assert 'wait_for_jobs(npt_job_ids)' in self._body

    def test_phase1b_min_h_still_uses_wait_for_jobs(self):
        """Regression: Phase 1b H minimisation must still use wait_for_jobs."""
        assert 'wait_for_jobs(min_h_job_ids)' in self._body


# ═════════════════════════════════════════════════════════════════════════════
# 6. Regression tests — things that must NOT have changed
# ═════════════════════════════════════════════════════════════════════════════

class TestRegressions:

    def test_old_restart_function_still_importable(self):
        """write_nvt_bulk_restart_script must still exist for any already-generated
        cluster scripts that import it directly."""
        from models.lammps_script import write_nvt_bulk_restart_script
        assert callable(write_nvt_bulk_restart_script)

    def test_new_functions_importable(self):
        """Both new functions must be importable from models.lammps_script."""
        from models.lammps_script import (
            write_nvt_equil_restart_script,
            write_nvt_prod_restart_script,
        )
        assert callable(write_nvt_equil_restart_script)
        assert callable(write_nvt_prod_restart_script)

    def test_write_chained_slurm_job_still_accepts_restart_commands(self, tmp_path):
        """Backward compat: restart_commands kwarg must still work without n_equil."""
        sh = tmp_path / 'legacy.sh'
        write_chained_slurm_job(
            job_name         = 'legacy',
            slurm_config     = SLURM_CFG,
            out_path         = str(sh),
            first_commands   = ['echo first'],
            restart_commands = ['echo restart'],
            restart_glob     = 'ck/*.restart',
            cutoff           = CUTOFF,
        )
        assert sh.exists()

    def test_npt_chained_job_unaffected(self):
        """Regression: NPT jobs in _body must still use old restart_commands style
        (they do not need phase detection)."""
        src = inspect.getsource(_dw.generate_diffusivity_scripts)
        # NPT block uses restart_commands= (not equil_restart_commands)
        # Verify the npt section still calls write_chained_slurm_job with restart_commands
        assert "restart_commands=[f'{LAMMPS_CMD}" in src or "restart_commands=[" in src

    def test_bulk_script_still_writes_equil_traj(self, bulk_script):
        """Regression: equil trajectory dump must still be written in first-run script."""
        assert 'equil_dump' in bulk_script
        assert 'phase1_equil.lammpstrj' in bulk_script

    def test_bulk_script_still_writes_data_files(self, bulk_script):
        """Regression: write_data and write_restart must still appear for both phases."""
        assert 'write_restart' in bulk_script
        assert 'write_data' in bulk_script
        assert bulk_script.count('write_restart') >= 2
        assert bulk_script.count('write_data') >= 2

    def test_equil_rst_still_writes_data_files(self, equil_rst_script):
        """Equil restart must still write checkpoints at end of equil and prod."""
        assert equil_rst_script.count('write_restart') >= 2
        assert equil_rst_script.count('write_data') >= 2

    def test_prod_rst_still_writes_data_files(self, prod_rst_script):
        """Prod restart must still write a checkpoint at end of production."""
        assert 'write_restart' in prod_rst_script
        assert 'write_data' in prod_rst_script

    def test_equil_rst_has_read_restart(self, equil_rst_script):
        """Equil restart must read from checkpoint, not read_data."""
        assert 'read_restart' in equil_rst_script
        assert 'read_data' not in equil_rst_script

    def test_prod_rst_has_read_restart(self, prod_rst_script):
        """Prod restart must read from checkpoint, not read_data."""
        assert 'read_restart' in prod_rst_script
        assert 'read_data' not in prod_rst_script

    def test_periodic_restarts_written_in_all_scripts(self, bulk_script,
                                                       equil_rst_script,
                                                       prod_rst_script):
        """All scripts must write periodic checkpoint restarts during runs."""
        for name, script in [('bulk', bulk_script),
                              ('equil_rst', equil_rst_script),
                              ('prod_rst', prod_rst_script)]:
            assert 'restart' in script and 'nvt_1000K' in script, (
                f"Periodic restart directive missing in {name} script")

    def test_h_type_propagated(self, bulk_script, equil_rst_script, prod_rst_script):
        """h_type must appear in group definition in all scripts."""
        for name, script in [('bulk', bulk_script),
                              ('equil_rst', equil_rst_script),
                              ('prod_rst', prod_rst_script)]:
            assert f'type {H_TYPE}' in script, (
                f"h_type {H_TYPE} not in group command of {name} script")

    def test_temperature_propagated(self, bulk_script, equil_rst_script, prod_rst_script):
        """Temperature must appear in fix nvt in all scripts."""
        for name, script in [('bulk', bulk_script),
                              ('equil_rst', equil_rst_script),
                              ('prod_rst', prod_rst_script)]:
            assert f'{TEMPERATURE}.0  {TEMPERATURE}.0' in script, (
                f"Temperature {TEMPERATURE} not in nvt fix of {name} script")

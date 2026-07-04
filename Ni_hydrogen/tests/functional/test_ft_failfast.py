"""
tests/functional/test_ft_failfast.py
=====================================
Category A/B functional tests — fail-fast hardening of the SLURM workflow.

Guards the fixes from the 2026-07 pipeline smoke-testing campaign:
  * chain scripts: absolute SCRIPT_PATH (not $0 — spooled copy under
    sbatch), .failed sentinel on real failure, sentinel cleared per leg
  * generated diffusivity_run.py: _require_file output validation after
    every wait_for_jobs, .failed detection in the NVT polling loop
  * NVT restart scripts: 'run N upto' (LAMMPS has no C-style ternary)
  * Arrhenius post-processing survives negative/NaN diffusivities

No LAMMPS, no SLURM, no GPU, no cluster files required.
"""

import os
import sys
import pathlib

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import MagicMock

for _m in ('acat', 'acat.adsorption_sites'):
    sys.modules.setdefault(_m, MagicMock())
for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.patches',
           'matplotlib.cm', 'matplotlib.ticker', 'matplotlib.transforms',
           'matplotlib.colors', 'matplotlib.scale', 'matplotlib._path',
           'matplotlib._api', 'matplotlib.cbook', 'matplotlib.rcsetup'):
    sys.modules.setdefault(_m, MagicMock())

from models.create_slurm import write_chained_slurm_job
from models.diffusivity_workflow import generate_diffusivity_scripts
from models.lammps_script import (
    write_nvt_prod_restart_script,
    write_nvt_equil_restart_script,
)

_SLURM = dict(partition='sharing', ntasks=1, cpus_per_task=8,
              gpu='a100:1', time='00:20:00', conda_env='env',
              cuda_version='12.3.0', openmpi_ver='4.1.6', ld_paths=[])


def _chain_script(tmp_path):
    out = str(tmp_path / 'chain.sh')
    write_chained_slurm_job(
        job_name='t', slurm_config=_SLURM, out_path=out,
        first_commands=['lmp -in a.lammps'],
        equil_restart_commands=['lmp -in b.lammps'],
        prod_restart_commands=['lmp -in c.lammps'],
        n_equil=500, restart_glob='ck/*.restart',
        cutoff='00:05:00', work_dir=str(tmp_path),
    )
    return out, open(out).read()


# ─── chain script sentinels ───────────────────────────────────────────────────

def test_chain_script_path_is_absolute_not_dollar_zero(tmp_path):
    """$0 under sbatch is the spooled copy in /var/spool/slurmd — the .done
    sentinel then never lands next to the script and orchestrators hang."""
    out, s = _chain_script(tmp_path)
    assert 'realpath "$0"' not in s
    assert f'SCRIPT_PATH="{os.path.abspath(out)}"' in s


def test_chain_script_failed_sentinel(tmp_path):
    _, s = _chain_script(tmp_path)
    assert 'touch "${SCRIPT_PATH}.failed"' in s, \
        'real failures must leave a .failed sentinel for the orchestrator'
    assert 'rm -f "${SCRIPT_PATH}.failed"' in s, \
        'each leg must clear a stale sentinel before running'
    # ordering: the clear happens before the command block, the touch after
    assert s.index('rm -f "${SCRIPT_PATH}.failed"') \
        < s.index('touch "${SCRIPT_PATH}.failed"')


def test_chain_script_done_sentinel_still_present(tmp_path):
    _, s = _chain_script(tmp_path)
    assert 'touch "${SCRIPT_PATH}.done"' in s


# ─── generated diffusivity_run.py hardening ──────────────────────────────────

@pytest.fixture(scope='module')
def diffusivity_run_src(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('diffgen')
    out = str(tmp / 'diffusivity_run.py')
    generate_diffusivity_scripts(
        input_structures=['/x/Ni.lammps'], n_h_values=[1], temperatures=[600],
        work_dir=str(tmp), nvt_wall_time='00:20:00', cutoff='00:05:00',
        gpu_partition='sharing', gpu_time='00:20:00', timestep_ps=0.001,
        tau_t_ps=0.1, n_equil_steps=500, n_prod_steps=2000, thermo_every=100,
        dump_every=100, velocity_seed=1, restart_every=1000, out_py=out,
    )
    return open(out).read()


def test_diffusivity_run_compiles(diffusivity_run_src):
    compile(diffusivity_run_src, 'diffusivity_run.py', 'exec')


def test_diffusivity_run_validates_outputs(diffusivity_run_src):
    """wait_for_jobs counts FAILED jobs as done — every phase must verify
    its output file exists before the next phase consumes it."""
    src = diffusivity_run_src
    assert 'def _require_file' in src
    # one check per phase: bare min, NPT (per T), bulk+H min (per T)
    assert src.count('_require_file(') >= 4


def test_diffusivity_run_detects_failed_chains(diffusivity_run_src):
    """A failed NVT chain previously hung the orchestrator forever."""
    src = diffusivity_run_src
    assert "+ '.failed'" in src or '".failed"' in src
    assert 'RuntimeError' in src


# ─── NVT restart scripts: no ternary, run-upto ────────────────────────────────

def _rst_common(tmp_path, writer, name, **extra):
    out = str(tmp_path / name)
    writer(
        restart_file='ck/*.restart', traj_file='t.dump', out_file='o.out',
        log_file='l.log', out_path=out,
        pair_style='mliap unified m.pt 0', mace_model='m.pt', pair_suffix='',
        elem_str='Ni H', temperature=600, n_equil=500, n_prod=2000,
        **extra,
    )
    return open(out).read()


def test_prod_restart_uses_run_upto_not_ternary(tmp_path):
    s = _rst_common(tmp_path, write_nvt_prod_restart_script,
                    'rst_prod.lammps', msd_prod_file='mp.dat')
    assert 'run            2500  upto' in s
    assert '?' not in s, 'LAMMPS variable formulas have no C-style ternary'
    assert 'remaining_prod' not in s


def test_equil_restart_uses_run_upto(tmp_path):
    s = _rst_common(tmp_path, write_nvt_equil_restart_script,
                    'rst_equil.lammps',
                    msd_equil_file='me.dat', msd_prod_file='mp.dat')
    assert 'run            500  upto' in s
    assert 'remaining_equil' not in s


# ─── Arrhenius guard: negative / invalid D must not crash ─────────────────────

_TABLE_HEADER = (
    '# Hydrogen self-diffusivity from MSD linear fit\n'
    '#      T_K        D (m2/s)    sigma_D (m2/s)        R2\n'
    '# ========  ==============  ================  ========\n'
)


def test_arrhenius_skips_gracefully_on_negative_D(tmp_path):
    """The 2026-07-03 smoke test crashed here: log(D<0) → NaN fit and
    matplotlib rejected negative yerr. Now: warn, drop, return NaNs."""
    from models.diffusivity_post_processing import run_arrhenius_pipeline

    table = tmp_path / 'diffusivity_table.txt'
    table.write_text(_TABLE_HEADER +
                     '     600.0   -1.640714e-10      1.388015e-10    0.2589\n'
                     '     700.0    1.298756e-09      4.806639e-10    0.6460\n')

    arr = run_arrhenius_pipeline(str(table), outdir=str(tmp_path))
    assert np.isnan(arr['Ea']) and np.isnan(arr['D0'])
    assert arr['fig'] is None and arr['plot_file'] is None
    # arrays still returned so the caller can write its JSON
    assert len(arr['T_arr']) == 2


def test_arrhenius_fits_after_dropping_bad_point(tmp_path, monkeypatch):
    """3 temps, one invalid: fit proceeds on the 2 valid points."""
    import models.diffusivity_post_processing as dpp

    table = tmp_path / 'diffusivity_table.txt'
    table.write_text(_TABLE_HEADER +
                     '     600.0    1.640714e-10      1.388015e-10    0.80\n'
                     '     650.0   -5.000000e-10      2.000000e-10    0.10\n'
                     '     700.0    1.298756e-09      4.806639e-10    0.65\n')

    monkeypatch.setattr(dpp, 'plot_arrhenius',
                        lambda *a, **k: 'FIG_STUB')
    arr = dpp.run_arrhenius_pipeline(str(table), outdir=str(tmp_path))
    assert np.isfinite(arr['Ea']) and arr['D0'] > 0
    assert arr['fig'] == 'FIG_STUB'

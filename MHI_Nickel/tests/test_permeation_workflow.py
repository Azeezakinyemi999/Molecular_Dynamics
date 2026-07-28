"""
tests/test_permeation_workflow.py
==================================
Tests for models/permeation_workflow.py — offline, no LAMMPS or SLURM.

Covers:
  generate_permeation_scripts  — embedded config, phase labels, body imports
  generate_permeation_sh       — SBATCH directives, optional mem/time flags
  load_barrier_summary         — DataFrame from jobs JSON + barrier files
  load_rate_summary            — DataFrame from rate_dict JSON files
  load_kmc_sweeps              — dict from permeation_sweep JSON files
  load_permeability_results    — dict from permeability JSON files
  plot_bottleneck              — returns None when no data; saves PNG otherwise
  plot_permeation_summary      — schema-current 3-panel figure; back-compat + errors
"""

import json
import math
import os
import pathlib
import sys
import pytest
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')   # must precede any pyplot import

from models.permeation_workflow import (
    generate_permeation_scripts,
    generate_permeation_sh,
    load_barrier_summary,
    load_rate_summary,
    load_kmc_sweeps,
    load_permeability_results,
    plot_bottleneck,
    plot_permeation_summary,
)


# ── shared config ────────────────────────────────────────────────────────────

_PERM_CFG = dict(
    work_dir='/work',
    stem='test_metal',
    relaxed_slab_path='/work/slab/relaxed.lammps',
    surface_sites_json='/work/slab/surface_sites.json',
    phase2_h_dir='/work/neb/h_atoms',
    sub_neb_dir='/work/sub_neb',
    vib_dir='/work/vib',
    results_dir='/work/results/permeation',
    temperatures=[600, 800, 1000],
    n_h_values=[1, 3, 5, 10],
    p_vals_pa=[100.0, 1000.0, 10000.0],
    a0_m=3.52e-10,
    l_m=1e-3,
    dh_diss_ev=0.2,
    dh_entry_ev=0.15,
    nx=10,
    ny=10,
    seed=42,
    kmc_max_steps=100_000,
    gpu_slurm_cfg={'partition': 'gpu', 'time': '24:00:00'},
    neb_slurm_cfg={'partition': 'gpu', 'time': '12:00:00'},
    vib_slurm_cfg={'partition': 'cpu', 'time': '06:00:00'},
    n_images=7,
    spring_const=5.0,
    neb_ftol=0.05,
)

_SH_CFG = dict(
    orch_job_name='perm_orch',
    orch_partition='cpu',
    orch_cpus_per_task=4,
    orch_mem='8G',
    orch_time='12:00:00',
    orch_openmpi_ver='4.1.4',
    orch_cuda_version='12.0',
    orch_conda_env='ase-env',
    orch_ld_paths=['/usr/local/lib'],
    work_dir='/work',
    out_py='/work/permeation_run.py',
)


# ── file-writing helpers ──────────────────────────────────────────────────────

def _write_barrier(path, E_abs=0.45, E_des=0.15, delta_E=0.30,
                   fmax=0.05, converged=True):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(
        f'E_IS: -100.5\n'
        f'E_FS: -100.2\n'
        f'E_abs: {E_abs}\n'
        f'E_des: {E_des}\n'
        f'delta_E: {delta_E}\n'
        f'fmax_final: {fmax}\n'
        f'converged: {converged}\n'
    )


def _write_jobs_json(path, hop, jobs):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(jobs, indent=2))


def _write_rate_dict_json(path, T=600):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        'hopa_Ni3Mo': {
            'k_forward': 1.23e10, 'k_reverse': 3.45e11,
            'Ea_raw': 0.45, 'Ea_zpe': 0.43,
            'Ed_raw': 0.15, 'Ed_zpe': 0.13,
            'nu': 1.2e13, 'delta_e': 0.30, 'T_K': float(T),
        }
    }
    pathlib.Path(path).write_text(json.dumps(data, indent=2))


def _write_sweep_json(path, T=600):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    P_vals = [1000.0, 4000.0, 9000.0, 16000.0]
    J_vals = [2.5 * math.sqrt(p) for p in P_vals]
    data = {
        'P_vals': P_vals,
        'J_vals': J_vals,
        'C0_vals': [1e22, 2e22, 3e22, 4e22],
        'sqrt_P_vals': [math.sqrt(p) for p in P_vals],
        'converged': [True, True, True, True],
        'theta_vals': [0.1, 0.12, 0.14, 0.16],
        't_total_vals': [1e-6] * 4,
        'n_steps_vals': [1000] * 4,
        'T_K': float(T), 'D_m2s': 1e-10, 'a0_m': 3.52e-10,
    }
    pathlib.Path(path).write_text(json.dumps(data, indent=2))


def _write_permeability_json(path, T=600):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        'T_K': float(T), 'D0_m2s': 1e-7, 'E_D_eV': 0.4,
        'dH_sol_eV': 0.3, 'a0_m': 3.52e-10,
        'dH_diss_eV': 0.2, 'dH_entry_eV': 0.15,
        'option1': {'S': 1e22, 'Phi': 1e12, 'J': 1e15},
        'option2': {'S': 2e22, 'Phi': 2e12, 'J': 2e15},
        'option3': {'S': 3e22, 'Phi': 3e12, 'J': 3e15,
                    'S_std': 1e21, 'n_converged': 4},
        'P_high_Pa': 16000.0, 'L_m': 1e-3,
    }
    pathlib.Path(path).write_text(json.dumps(data, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# 1. generate_permeation_scripts
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratePermeationScripts:

    @pytest.fixture()
    def gen_result(self, tmp_path):
        out_py = str(tmp_path / 'permeation_run.py')
        ret    = generate_permeation_scripts(**_PERM_CFG, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        return ret, out_py, content

    def test_file_created(self, gen_result):
        _, out_py, _ = gen_result
        assert pathlib.Path(out_py).exists()

    def test_returns_out_py_path(self, gen_result):
        ret, out_py, _ = gen_result
        assert ret == out_py

    def test_work_dir_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'WORK_DIR' in content

    def test_temperatures_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'TEMPERATURES' in content
        assert '600' in content and '1000' in content

    def test_pressure_vals_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'P_VALS_PA' in content
        assert '10000.0' in content

    def test_n_h_values_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'N_H_VALUES' in content
        assert '[1, 3, 5, 10]' in content

    def test_no_diffusivity_placeholder_embedded(self, gen_result):
        """D0/Ea are loaded per-(stem, n_H) at runtime from Part 3's real
        fit — no global placeholder constant should be injected at all."""
        _, _, content = gen_result
        assert 'D0_M2S' not in content
        assert 'E_D_EV' not in content

    def test_kmc_grid_params_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'NX' in content and 'NY' in content
        assert 'SEED' in content and 'KMC_MAX_STEPS' in content

    def test_neb_params_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'N_IMAGES' in content
        assert 'SPRING_K' in content
        assert 'NEB_FTOL_VAL' in content

    def test_key_imports_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'from models.tst_rates import' in content
        assert 'from models.permeation import' in content

    def test_connect_to_surface_cell_is_flattened(self, gen_result):
        """Regression test: connect_to_surface/_periodic_xy_distance expect
        a flat [Lx, Ly, Lz] array. ase.Atoms.get_cell() alone returns a 3x3
        Cell object, whose cell[0] is a row vector, not a scalar -- this
        crashed with 'TypeError: type numpy.ndarray doesn't define
        __round__ method' the first time this script ever actually ran."""
        _, _, content = gen_result
        assert '_slab_atoms.get_cell().diagonal()' in content

    def test_dedup_is_labels_uses_real_energies(self, gen_result):
        """Regression test: Hop A's e_is must be the real absolute relaxed
        energy from Part 1's h_min_{sid}.log, not a hardcoded 0.0 -- a 0.0
        placeholder silently produced ~100+ eV 'barriers' the first time
        this ever ran for real, since 0.0 sits nowhere near the slab's true
        energy scale (~-125 eV). Post-Part-1 the entry H* are seeded from
        dissociation products via collect_entry_h_sources /
        build_surface_sub1_sub2_map, whose (sid, is_path, e_is) triples
        carry that same real e_is."""
        _, _, content = gen_result
        assert 'collect_entry_h_sources(' in content
        assert 'build_surface_sub1_sub2_map(' in content
        # e_is threaded from the path map, never a 0.0 placeholder
        assert "e['e_is']" in content
        assert "h_atom_'), p, 0.0)" not in content

    def test_all_six_phase_labels_in_body(self, gen_result):
        _, _, content = gen_result
        for phase in ('Phase 1', 'Phase 2', 'Phase 3',
                      'Phase 4', 'Phase 5', 'Phase 6'):
            assert phase in content, f'{phase!r} not found in generated file'

    def test_hop_a_and_hop_b_neb_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'Hop A NEB' in content
        assert 'Hop B NEB' in content

    def test_richardson_flux_called_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'richardson_flux' in content

    def test_dh_values_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'DH_DISS_EV' in content
        assert 'DH_ENTRY_EV' in content

    def test_status_tracking_structure_in_body(self, gen_result):
        """Structural shape of the success-tracking fix: _PERM_STATUS must be
        initialized before the n_H loop, every `continue` path that skips a
        result must record into it, and the final decision must be based on
        whether anything was actually written — not an unconditional success
        message. See [[project_pipeline_test_bugs]]."""
        _, _, content = gen_result
        status_init_idx  = content.index("_PERM_STATUS = {")
        loop_idx         = content.index('for _n_h in N_H_VALUES:')
        ready_skip_idx   = content.index("_PERM_STATUS['n_h_skipped'].append")
        written_idx      = content.index("_PERM_STATUS['permeability_written'].append")
        status_write_idx = content.index("permeation_status.json")
        exit_check_idx   = content.index("if not _PERM_STATUS['permeability_written']:")
        exit_call_idx    = content.index('sys.exit(1)', exit_check_idx)
        else_idx         = content.index('else:', exit_call_idx)

        assert status_init_idx < loop_idx < ready_skip_idx < written_idx
        assert written_idx < status_write_idx < exit_check_idx < exit_call_idx < else_idx


class TestPermeationSuccessTracking:
    """Behavioral proof that a metal producing zero permeability results
    now exits nonzero, instead of silently reporting success — the exact
    gap pipeline_run.py's `_rc2 != 0` check relied on being meaningful.
    See [[project_pipeline_test_bugs]]."""

    def _tail_slice(self, content):
        marker = '_P_HIGH = max(P_VALS_PA)'
        idx = content.index(marker)
        return content[idx:]

    def test_zero_results_exits_nonzero_and_records_reasons(self, tmp_path):
        out_py = str(tmp_path / 'permeation_run.py')
        _cfg = {**_PERM_CFG, 'n_h_values': [1, 3]}
        generate_permeation_scripts(**_cfg, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        tail = self._tail_slice(content)

        results_dir = str(tmp_path / 'results')
        os.makedirs(results_dir, exist_ok=True)
        status_path = os.path.join(results_dir, 'permeation_status.json')

        def _fake_resolve(work_dir, stem, n_h):
            return {
                'ready': False,
                'nh_dir': os.path.join(results_dir, f'{stem}_{n_h}H'),
                'message': f'no diffusivity_arrhenius.json for n_H={n_h}',
            }

        ns = {
            'os': os, 'json': json, 'sys': sys,
            'N_H_VALUES': [1, 3],
            'STEM': 'test_metal',
            'WORK_DIR': str(tmp_path / 'work'),
            'RESULTS_DIR': results_dir,
            'resolve_nh_diffusivity': _fake_resolve,
            'P_VALS_PA': _PERM_CFG['p_vals_pa'],
            '_PHASE6_READY': True,
        }
        with pytest.raises(SystemExit) as exc_info:
            exec(compile(tail, out_py, 'exec'), ns)
        assert exc_info.value.code == 1

        assert os.path.exists(status_path)
        status = json.loads(pathlib.Path(status_path).read_text())
        assert status['permeability_written'] == []
        assert {s['n_h'] for s in status['n_h_skipped']} == {1, 3}
        assert all('no diffusivity_arrhenius.json' in s['reason']
                   for s in status['n_h_skipped'])

    def test_some_results_produced_exits_cleanly(self, tmp_path):
        """Every other test in this class only exercises the all-fail
        path. This is the one proof that the 'else' branch (some/all n_H
        succeed -> permeability_written non-empty -> no sys.exit) actually
        runs end-to-end without crashing, not just structurally present."""
        import itertools
        import numpy as np
        from models.checkpoint import is_done, mark_done

        out_py = str(tmp_path / 'permeation_run.py')
        _cfg = {**_PERM_CFG, 'n_h_values': [1, 3], 'temperatures': [600]}
        generate_permeation_scripts(**_cfg, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        tail = self._tail_slice(content)

        results_dir = str(tmp_path / 'results')
        os.makedirs(results_dir, exist_ok=True)
        pathlib.Path(os.path.join(results_dir, 'rate_dict_T600K.json')).write_text('{}')

        def _fake_resolve(work_dir, stem, n_h):
            nh_dir = os.path.join(results_dir, f'{stem}_{n_h}H')
            return {
                'ready': True,
                'nh_dir': nh_dir,
                'D0_m2s': 1e-9,
                'E_D_eV': 0.3,
                'dilute_note': None,
            }

        ns = {
            'os': os, 'json': json, 'sys': sys, 'itertools': itertools, 'np': np,
            'is_done': is_done, 'mark_done': mark_done,
            'N_H_VALUES': [1, 3],
            'STEM': 'test_metal',
            'WORK_DIR': str(tmp_path / 'work'),
            'RESULTS_DIR': results_dir,
            'resolve_nh_diffusivity': _fake_resolve,
            'TEMPERATURES': [600],
            '_a0_dict': {600: 3.5e-10},
            '_slab_species': ['Ni'],
            '_sid2comp': {},
            '_kmc_composition': None,
            '_diss_vib': {},
            '_DISS_JSON': str(tmp_path / 'nonexistent_diss.json'),
            '_NU_DISS': 1e13,
            '_PHASE6_READY': True,
            '_DH_SOL': 0.25,
            '_DH_DISS_USED': 0.2,
            '_DH_ENTRY_USED': 0.15,
            '_KB_EV': 8.617333262e-5,
            'P_VALS_PA': _cfg['p_vals_pa'],
            'A0_M': _cfg['a0_m'],
            'L_M': _cfg['l_m'],
            'NX': _cfg['nx'], 'NY': _cfg['ny'], 'SEED': _cfg['seed'],
            'KMC_MAX_STEPS': _cfg['kmc_max_steps'],
            'arrhenius_diffusivity': lambda D0, Ea, T: 1e-10,
            'sweep_pressure': lambda **kw: {'converged': [True] * len(kw['P_vals_Pa'])},
            'fit_solubility_from_kmc': lambda sw: {'S_mean': 1e-3, 'S_std': 1e-4, 'n_converged': 3},
            'lattice_site_S0': lambda a0: 1e28,
            'permeability': lambda D, S: D * S,
            'richardson_flux': lambda Phi, Ph, Pl, L: Phi * Ph / L,
            'parse_barrier_file': lambda p: {},
            # Part 6: env-keyed rate assembly + per-env solubility (all defined
            # before the _P_HIGH tail marker in the real body, so injected here)
            '_hopa_vib': {}, '_hopb_vib': {}, '_fs_freq_sets': [],
            '_dh_sol_by_env': {'Ni6_oct': {'dH_sol_eV': 0.25, 'w_env': 1.0, 'n_sites': 1}},
            '_sub1_env_comp': None, '_sub2_env_comp': None,
            'env_rate_dict': lambda hv, T: ({}, {}),
            'solubility_by_environment': lambda d, S0, T: 1e-3,
            'solubility_env_rel_err': lambda d, T, S0_rel_err=0.0: 0.1,
            'solubility_from_rates': lambda kd, kds, ke, kx, a0, T: 1e-3,
            'vibrational_S0': lambda a0, T, freqs: 1e23,
            'build_dh_sol_by_env': lambda *a, **kw: {},
            'classify_sieverts_regime': lambda P, th, converged=None, **kw: {
                'regime': 'sieverts_compatible', 'theta_exponent': 0.5,
                'theta_max': 0.2},
            'check_sieverts_law': lambda P, J, plot=False: {'r_squared': 0.99},
            # error-propagation-current return schemas (signatures accept the
            # y_err / *_err kwargs the body now passes)
            'fit_arrhenius': lambda T, y, yerr=None: {
                'prefactor': 1e28, 'Ea_eV': 0.25, 'r2': 0.99,
                'n_points': len(list(T)),
                'prefactor_rel_err': 0.1, 'Ea_err_eV': 0.02},
            'permeability_arrhenius': lambda D0, ED, S0, dH, **kw: {
                'Phi0': D0 * S0, 'E_phi_eV': ED + dH,
                'Phi0_rel_err': 0.1, 'Phi0_factor': 1.1, 'E_phi_err_eV': 0.02},
        }

        # Full success: must run to completion with no SystemExit.
        exec(compile(tail, out_py, 'exec'), ns)

        status_path = os.path.join(results_dir, 'permeation_status.json')
        assert os.path.exists(status_path)
        status = json.loads(pathlib.Path(status_path).read_text())
        assert status['n_h_skipped'] == []
        assert len(status['permeability_written']) == 2  # 2 n_H x 1 T

        for n_h in (1, 3):
            perm_f = os.path.join(results_dir, f'test_metal_{n_h}H', 'permeability_T600K.json')
            assert os.path.exists(perm_f)
            payload = json.loads(pathlib.Path(perm_f).read_text())
            assert payload['n_H'] == n_h


# ═══════════════════════════════════════════════════════════════════════════
# 2. generate_permeation_sh
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratePermeationSh:

    @pytest.fixture()
    def sh_result(self, tmp_path):
        out_sh  = str(tmp_path / 'permeation_run.sh')
        ret     = generate_permeation_sh(**_SH_CFG, out_sh=out_sh)
        content = pathlib.Path(out_sh).read_text()
        return ret, out_sh, content

    def test_file_created(self, sh_result):
        _, out_sh, _ = sh_result
        assert pathlib.Path(out_sh).exists()

    def test_returns_out_sh_path(self, sh_result):
        ret, out_sh, _ = sh_result
        assert ret == out_sh

    def test_sbatch_job_name_present(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --job-name=perm_orch' in content

    def test_sbatch_partition_present(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --partition=cpu' in content

    def test_sbatch_mem_present_when_set(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --mem=8G' in content

    def test_sbatch_time_present_when_set(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --time=12:00:00' in content

    def test_sbatch_mem_absent_when_empty(self, tmp_path):
        out_sh = str(tmp_path / 'run.sh')
        generate_permeation_sh(**{**_SH_CFG, 'orch_mem': ''}, out_sh=out_sh)
        assert '#SBATCH --mem=' not in pathlib.Path(out_sh).read_text()

    def test_sbatch_time_absent_when_empty(self, tmp_path):
        out_sh = str(tmp_path / 'run2.sh')
        generate_permeation_sh(**{**_SH_CFG, 'orch_time': ''}, out_sh=out_sh)
        assert '#SBATCH --time=' not in pathlib.Path(out_sh).read_text()

    def test_conda_activate_present(self, sh_result):
        _, _, content = sh_result
        assert 'conda activate ase-env' in content

    def test_python_run_line_present(self, sh_result):
        _, _, content = sh_result
        assert 'python /work/permeation_run.py' in content


# ═══════════════════════════════════════════════════════════════════════════
# 3. load_barrier_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadBarrierSummary:

    def test_returns_empty_when_no_jobs_json(self, tmp_path):
        result = load_barrier_summary(str(tmp_path / 'sub_neb'))
        assert isinstance(result, pd.DataFrame) and result.empty

    def test_returns_dataframe_when_jobs_present(self, tmp_path):
        bf = str(tmp_path / 'sub_neb' / 'hopa' / 'Ni3Mo' / 'barrier.txt')
        _write_barrier(bf)
        jobs = [{'sid': 'Ni3Mo', 'barrier_file': bf}]
        _write_jobs_json(str(tmp_path / 'sub_neb' / 'hopa' / 'hopa_jobs.json'),
                         'hopa', jobs)
        result = load_barrier_summary(str(tmp_path / 'sub_neb'))
        assert not result.empty

    def test_sid_and_hop_columns_present(self, tmp_path):
        bf = str(tmp_path / 'sub_neb' / 'hopa' / 'Ni3Mo' / 'barrier.txt')
        _write_barrier(bf)
        jobs = [{'sid': 'Ni3Mo', 'barrier_file': bf}]
        _write_jobs_json(str(tmp_path / 'sub_neb' / 'hopa' / 'hopa_jobs.json'),
                         'hopa', jobs)
        result = load_barrier_summary(str(tmp_path / 'sub_neb'))
        assert 'sid' in result.columns and 'hop' in result.columns

    def test_e_abs_column_has_correct_value(self, tmp_path):
        bf = str(tmp_path / 'sub_neb' / 'hopa' / 'X' / 'barrier.txt')
        _write_barrier(bf, E_abs=0.38)
        jobs = [{'sid': 'X', 'barrier_file': bf}]
        _write_jobs_json(str(tmp_path / 'sub_neb' / 'hopa' / 'hopa_jobs.json'),
                         'hopa', jobs)
        result = load_barrier_summary(str(tmp_path / 'sub_neb'))
        assert result['E_abs'].iloc[0] == pytest.approx(0.38)

    def test_missing_barrier_file_excluded(self, tmp_path):
        jobs = [{'sid': 'ghost', 'barrier_file': '/nonexistent/barrier.txt'}]
        _write_jobs_json(str(tmp_path / 'sub_neb' / 'hopa' / 'hopa_jobs.json'),
                         'hopa', jobs)
        result = load_barrier_summary(str(tmp_path / 'sub_neb'))
        assert result.empty


# ═══════════════════════════════════════════════════════════════════════════
# 4. load_rate_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadRateSummary:

    def test_returns_empty_when_no_files(self, tmp_path):
        result = load_rate_summary(str(tmp_path), [600, 800])
        assert isinstance(result, pd.DataFrame) and result.empty

    def test_returns_dataframe_when_file_exists(self, tmp_path):
        _write_rate_dict_json(str(tmp_path / 'rate_dict_T600K.json'), T=600)
        result = load_rate_summary(str(tmp_path), [600])
        assert not result.empty

    def test_label_column_present(self, tmp_path):
        _write_rate_dict_json(str(tmp_path / 'rate_dict_T600K.json'), T=600)
        result = load_rate_summary(str(tmp_path), [600])
        assert 'label' in result.columns

    def test_t_k_column_populated(self, tmp_path):
        _write_rate_dict_json(str(tmp_path / 'rate_dict_T600K.json'), T=600)
        result = load_rate_summary(str(tmp_path), [600])
        assert (result['T_K'] == 600).all()

    def test_k_forward_column_present(self, tmp_path):
        _write_rate_dict_json(str(tmp_path / 'rate_dict_T600K.json'), T=600)
        result = load_rate_summary(str(tmp_path), [600])
        assert 'k_forward' in result.columns
        assert result['k_forward'].iloc[0] == pytest.approx(1.23e10, rel=1e-6)

    def test_missing_temperature_file_skipped(self, tmp_path):
        _write_rate_dict_json(str(tmp_path / 'rate_dict_T600K.json'), T=600)
        # 800K file not present
        result = load_rate_summary(str(tmp_path), [600, 800])
        assert len(result) == 1   # only 600K entry


# ═══════════════════════════════════════════════════════════════════════════
# 5. load_kmc_sweeps
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadKmcSweeps:

    def test_returns_empty_dict_when_no_files(self, tmp_path):
        result = load_kmc_sweeps(str(tmp_path), [600, 800])
        assert result == {}

    def test_returns_dict_with_temperature_key(self, tmp_path):
        _write_sweep_json(str(tmp_path / 'permeation_sweep_T600K.json'), T=600)
        result = load_kmc_sweeps(str(tmp_path), [600])
        assert 600 in result

    def test_loaded_content_has_p_vals(self, tmp_path):
        _write_sweep_json(str(tmp_path / 'permeation_sweep_T600K.json'), T=600)
        result = load_kmc_sweeps(str(tmp_path), [600])
        assert 'P_vals' in result[600]
        assert len(result[600]['P_vals']) == 4

    def test_missing_temperature_excluded(self, tmp_path):
        _write_sweep_json(str(tmp_path / 'permeation_sweep_T600K.json'), T=600)
        result = load_kmc_sweeps(str(tmp_path), [600, 800])
        assert 600 in result
        assert 800 not in result


# ═══════════════════════════════════════════════════════════════════════════
# 6. load_permeability_results
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadPermeabilityResults:

    def test_returns_empty_dict_when_no_files(self, tmp_path):
        result = load_permeability_results(str(tmp_path), [600, 800])
        assert result == {}

    def test_returns_dict_with_temperature_key(self, tmp_path):
        _write_permeability_json(str(tmp_path / 'permeability_T600K.json'), T=600)
        result = load_permeability_results(str(tmp_path), [600])
        assert 600 in result

    def test_option_keys_present_in_loaded_data(self, tmp_path):
        _write_permeability_json(str(tmp_path / 'permeability_T600K.json'), T=600)
        result = load_permeability_results(str(tmp_path), [600])
        for opt in ('option1', 'option2', 'option3'):
            assert opt in result[600]

    def test_missing_temperature_excluded(self, tmp_path):
        _write_permeability_json(str(tmp_path / 'permeability_T600K.json'), T=600)
        result = load_permeability_results(str(tmp_path), [600, 800])
        assert 600 in result and 800 not in result


# ═══════════════════════════════════════════════════════════════════════════
# 7. plot_bottleneck
# ═══════════════════════════════════════════════════════════════════════════

class TestPlotBottleneck:

    def test_returns_none_when_no_sweep_files(self, tmp_path):
        result = plot_bottleneck(str(tmp_path), [600, 800])
        assert result is None

    def test_returns_none_when_sweeps_have_fewer_than_two_converged_points(self, tmp_path):
        # only 1 converged point → can't fit Sieverts → skips
        data = {
            'P_vals': [1000.0, 4000.0],
            'J_vals': [79.1, 158.1],
            'converged': [True, False],   # only 1 converged
        }
        pathlib.Path(tmp_path / 'permeation_sweep_T600K.json').write_text(
            json.dumps(data))
        result = plot_bottleneck(str(tmp_path), [600])
        assert result is None

    def test_saves_png_when_valid_sweep_data_present(self, tmp_path):
        _write_sweep_json(str(tmp_path / 'permeation_sweep_T600K.json'), T=600)
        out = plot_bottleneck(str(tmp_path), [600])
        assert out is not None
        assert pathlib.Path(out).exists()
        assert out.endswith('bottleneck.png')


# ═══════════════════════════════════════════════════════════════════════════
# 7b. plot_permeation_summary — schema-current 3-panel figure
# ═══════════════════════════════════════════════════════════════════════════

def _write_summary_inputs(d, T_list=(600, 800), with_errors=True, with_regime=True):
    """Write the JSON that plot_permeation_summary consumes into dir ``d``."""
    def _sol_route(S0, dH):
        r = {'available': True, 'S0': S0, 'dH_sol_eV': dH,
             'T_K_arr': list(T_list),
             'S_arr': [S0 * math.exp(-dH / (8.617333262e-5 * T)) for T in T_list],
             'r2': 0.99}
        if with_errors:
            r['S0_rel_err'] = 0.3
            r['dH_sol_err_eV'] = 0.05
        return r

    def _perm_route(Phi0, Eph):
        r = {'available': True, 'Phi0': Phi0, 'E_phi_eV': Eph, 'r2_S': 0.99}
        if with_errors:
            r['Phi0_rel_err'] = 0.4
            r['Phi0_factor'] = math.exp(0.4)
            r['E_phi_err_eV'] = 0.1
            r['J_rel_err_by_T'] = {str(int(T)): 0.5 for T in T_list}
        return r

    pathlib.Path(d, 'solubility_arrhenius.json').write_text(json.dumps({
        'routes': {'geometric': _sol_route(1e29, -0.05),
                   'vibrational': _sol_route(1e24, -0.10),
                   'detailed_balance': _sol_route(5e28, 0.02)}}))
    pathlib.Path(d, 'permeability_arrhenius.json').write_text(json.dumps({
        'routes': {'geometric': _perm_route(1e12, 0.35),
                   'vibrational': _perm_route(1e8, 0.30),
                   'detailed_balance': _perm_route(5e11, 0.42)}}))
    for T in T_list:
        P = [10.0 ** e for e in range(-4, 5)]
        th = [min(0.99, 1e-3 * (p ** 0.5)) for p in P]
        pathlib.Path(d, f'permeation_sweep_T{int(T)}K.json').write_text(json.dumps({
            'P_vals': P, 'theta_vals': th, 'T_K': T}))
        pm = {'T_K': T, 'option1': {}, 'option2': {}}
        if with_regime:
            pm['sieverts_regime'] = {'regime': 'sieverts_compatible',
                                     'theta_exponent': 0.5}
        pathlib.Path(d, f'permeability_T{int(T)}K.json').write_text(json.dumps(pm))


class TestPlotPermeationSummary:

    def test_returns_none_when_arrhenius_json_missing(self, tmp_path):
        # sweeps present but no Arrhenius fits → nothing to summarise
        _write_sweep_json(str(tmp_path / 'permeation_sweep_T600K.json'), T=600)
        assert plot_permeation_summary(str(tmp_path), [600]) is None

    def test_saves_png_with_full_schema(self, tmp_path):
        _write_summary_inputs(str(tmp_path), T_list=(600, 800),
                              with_errors=True, with_regime=True)
        out = plot_permeation_summary(str(tmp_path), [600, 800])
        assert out is not None
        assert pathlib.Path(out).exists()
        assert out.endswith('permeation_summary.png')

    def test_back_compat_without_error_or_regime_fields(self, tmp_path):
        # pre-schema data: no *_err_eV, no Phi0_rel_err, no sieverts_regime.
        # Must still render (bands collapse to zero width, labels omit ±).
        _write_summary_inputs(str(tmp_path), T_list=(600, 800),
                              with_errors=False, with_regime=False)
        out = plot_permeation_summary(str(tmp_path), [600, 800])
        assert out is not None
        assert pathlib.Path(out).exists()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Entry H* sourcing — moved to collect_entry_h_sources
# ═══════════════════════════════════════════════════════════════════════════
# collect_dedup_is_labels was removed in the Part 1 reframing: Hop A is now
# seeded from dissociation products, not a wholesale h_atom_* glob. Its
# replacement, collect_entry_h_sources (models/neb_subsurface.py), is covered
# by TestCollectEntryHSources in tests/test_neb_subsurface.py.

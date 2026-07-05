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
    collect_dedup_is_labels,
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
        """Regression test: Hop A's e_is must come from
        collect_dedup_is_labels() (real absolute energies from Part 1's
        h_min_{sid}.log), not a hardcoded 0.0 -- a 0.0 placeholder
        silently produced ~100+ eV 'barriers' the first time this ever
        ran for real, since 0.0 sits nowhere near the slab's true energy
        scale (~-125 eV)."""
        _, _, content = gen_result
        assert 'collect_dedup_is_labels(PHASE2_H_DIR)' in content
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
# 8. collect_dedup_is_labels
# ═══════════════════════════════════════════════════════════════════════════

def _write_h_site(phase2_dir, sid, pe_final_eV):
    pathlib.Path(phase2_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(phase2_dir) / f'h_atom_{sid}_relaxed.lammps').write_text('dummy')
    (pathlib.Path(phase2_dir) / f'h_min_{sid}.log').write_text(
        f'  pe_final_eV     : {pe_final_eV}\n'
        f'  fmax_eV_per_Ang : 4.08e-07\n'
    )


class TestCollectDedupIsLabels:

    def test_uses_absolute_energy_not_placeholder(self, tmp_path):
        """Regression test: e_is must be the real relaxed total energy
        (~-125 eV for this structure), never a hardcoded 0.0 -- a 0.0
        placeholder silently produced ~100+ eV Hop A 'barriers' the first
        time this code ever ran for real, since 0.0 sits nowhere near
        this slab's actual energy scale."""
        _write_h_site(tmp_path, 's_0', -125.489223)
        labels = collect_dedup_is_labels(str(tmp_path))
        assert len(labels) == 1
        sid, path, e_is = labels[0]
        assert sid == 's_0'
        assert e_is == pytest.approx(-125.489223)
        assert e_is != 0.0

    def test_multiple_sites_sorted_by_sid(self, tmp_path):
        _write_h_site(tmp_path, 's_1', -125.489223)
        _write_h_site(tmp_path, 's_0', -125.489223)
        labels = collect_dedup_is_labels(str(tmp_path))
        assert [l[0] for l in labels] == ['s_0', 's_1']

    def test_missing_log_skipped_not_fabricated(self, tmp_path):
        """A site with a relaxed structure but no parseable log must be
        skipped entirely -- never assigned a fabricated e_is."""
        pathlib.Path(tmp_path).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(tmp_path) / 'h_atom_s_0_relaxed.lammps').write_text('dummy')
        # no h_min_s_0.log written
        labels = collect_dedup_is_labels(str(tmp_path))
        assert labels == []

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert collect_dedup_is_labels(str(tmp_path)) == []

    def test_is_path_points_to_relaxed_structure(self, tmp_path):
        _write_h_site(tmp_path, 's_0', -125.489223)
        sid, path, e_is = collect_dedup_is_labels(str(tmp_path))[0]
        assert path.endswith('h_atom_s_0_relaxed.lammps')
        assert os.path.exists(path)

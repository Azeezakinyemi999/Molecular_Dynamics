"""
tests/test_diffusivity_workflow.py
===================================
Tests for models/diffusivity_workflow.py — offline, no LAMMPS or SLURM.

Covers:
  generate_diffusivity_scripts  — embedded config, body structure, cutoff calc
  generate_orchestrator_sh      — SBATCH directives, chmod, optional time line
  load_diffusivity_results      — DataFrame columns, missing-file handling
  qc_minimizations              — empty return when no log files present
  qc_nvt_thermo                 — empty return when no thermo files present
"""

import os
import pathlib
import sys
import pytest
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')   # must precede any pyplot import

from models.diffusivity_workflow import (
    generate_diffusivity_scripts,
    generate_orchestrator_sh,
    load_diffusivity_results,
    qc_minimizations,
    qc_nvt_thermo,
)
from models.diffusivity_post_processing import save_diffusivity_table


# ── shared config ────────────────────────────────────────────────────────────

_STRUCTS = ['/data/Ni3Mo.lammps', '/data/Ni.lammps']
_NH      = [1, 4]
_TEMPS   = [600, 800, 1000]

_GEN_CFG = dict(
    input_structures=_STRUCTS,
    n_h_values=_NH,
    temperatures=_TEMPS,
    work_dir='/work',
    nvt_wall_time='48:00:00',
    cutoff='47:50:00',
    gpu_partition='gpu_p100',
    gpu_time='48:00:00',
    timestep_ps=0.0005,
    tau_t_ps=0.1,
    n_equil_steps=2_000_000,
    n_prod_steps=7_000_000,
    thermo_every=1000,
    dump_every=1000,
    velocity_seed=12345,
    restart_every=10000,
)

_SH_CFG = dict(
    orch_job_name='diff_orch',
    orch_partition='cpu',
    orch_cpus_per_task=4,
    orch_mem='8G',
    orch_time='24:00:00',
    orch_openmpi_ver='4.1.4',
    orch_cuda_version='12.0',
    orch_conda_env='ase-env',
    orch_ld_paths=['/usr/local/lib'],
    work_dir='/work',
    out_py='/work/diffusivity_run.py',
)


# ── helper: write a valid diffusivity_table.txt in the expected path ─────────

def _make_table(results_root, struct_stem, n_h,
                T_vals=None, D_vals=None, D_errs=None, R2_vals=None):
    T_vals  = T_vals  or [600., 800., 1000.]
    D_vals  = D_vals  or [1e-10, 3e-10, 7e-10]
    D_errs  = D_errs  or [1e-11, 2e-11, 5e-11]
    R2_vals = R2_vals or [0.998, 0.995, 0.990]
    run_name  = f'{struct_stem}_{n_h}H'
    table_dir = pathlib.Path(results_root) / run_name / 'analysis'
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = str(table_dir / 'diffusivity_table.txt')
    save_diffusivity_table(T_vals, D_vals, D_errs, R2_vals, table_path)
    return table_path


# ═══════════════════════════════════════════════════════════════════════════
# 1. generate_diffusivity_scripts
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateDiffusivityScripts:

    @pytest.fixture()
    def gen_result(self, tmp_path):
        out_py = str(tmp_path / 'scripts' / 'diffusivity_run.py')
        ret = generate_diffusivity_scripts(**_GEN_CFG, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        return ret, out_py, content

    def test_file_created(self, gen_result):
        _, out_py, _ = gen_result
        assert pathlib.Path(out_py).exists()

    def test_returns_out_py_path(self, gen_result):
        ret, out_py, _ = gen_result
        assert ret == out_py

    def test_input_structures_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'Ni3Mo.lammps' in content
        assert 'Ni.lammps' in content

    def test_temperatures_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'TEMPERATURES' in content
        assert '600' in content
        assert '1000' in content

    def test_n_equil_steps_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'N_EQUIL_STEPS' in content
        assert '2000000' in content

    def test_n_prod_steps_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'N_PROD_STEPS' in content
        assert '7000000' in content

    def test_key_imports_present(self, gen_result):
        _, _, content = gen_result
        assert 'from models.lammps_script import' in content
        assert 'from models.diffusivity_post_processing import' in content

    def test_phase1a_label_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'Phase 1a' in content

    def test_phase2_label_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'Phase 2' in content

    def test_phase3_label_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'Phase 3' in content

    def test_short_gpu_cutoff_auto_computed(self, tmp_path):
        # short_gpu_time='04:00:00' → 4*3600 - 300 = 14100 s → '03:55:00'
        out_py = str(tmp_path / 's' / 'run.py')
        generate_diffusivity_scripts(**_GEN_CFG, short_gpu_time='04:00:00',
                                     short_gpu_cutoff=None, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        assert "'03:55:00'" in content

    def test_short_gpu_cutoff_custom_used_when_provided(self, tmp_path):
        out_py = str(tmp_path / 's' / 'run.py')
        generate_diffusivity_scripts(**_GEN_CFG, short_gpu_cutoff='02:30:00',
                                     out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        assert "'02:30:00'" in content

    def test_metal_table_embedded(self, tmp_path):
        out_py = str(tmp_path / 's' / 'run.py')
        metal_table = {'Ni3Mo': {'elem_str': 'Ni Mo H', 'e2t': {'Ni': 1}}}
        generate_diffusivity_scripts(**_GEN_CFG, metal_table=metal_table,
                                     out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        assert 'METAL_TABLE' in content
        assert 'Ni3Mo' in content

    def test_arrhenius_json_save_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'diffusivity_arrhenius.json' in content

    def test_lattice_params_json_save_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'lattice_params_vs_T.json' in content


# ═══════════════════════════════════════════════════════════════════════════
# 2. generate_orchestrator_sh
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateOrchestratorSh:

    @pytest.fixture()
    def sh_result(self, tmp_path):
        out_sh = str(tmp_path / 'scripts' / 'diffusivity_run.sh')
        ret = generate_orchestrator_sh(**_SH_CFG, out_sh=out_sh)
        content = pathlib.Path(out_sh).read_text()
        return ret, out_sh, content

    def test_file_created(self, sh_result):
        _, out_sh, _ = sh_result
        assert pathlib.Path(out_sh).exists()

    def test_returns_out_sh_path(self, sh_result):
        ret, out_sh, _ = sh_result
        assert ret == out_sh

    def test_file_is_executable(self, sh_result):
        _, out_sh, _ = sh_result
        assert os.access(out_sh, os.X_OK)

    def test_sbatch_job_name_present(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --job-name=diff_orch' in content

    def test_sbatch_partition_present(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --partition=cpu' in content

    def test_conda_activate_present(self, sh_result):
        _, _, content = sh_result
        assert 'conda activate ase-env' in content

    def test_python_run_line_present(self, sh_result):
        _, _, content = sh_result
        assert 'python /work/diffusivity_run.py' in content

    def test_sbatch_time_line_present_when_set(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --time=24:00:00' in content

    def test_sbatch_time_line_absent_when_empty(self, tmp_path):
        out_sh = str(tmp_path / 's' / 'run.sh')
        generate_orchestrator_sh(**{**_SH_CFG, 'orch_time': ''}, out_sh=out_sh)
        content = pathlib.Path(out_sh).read_text()
        assert '#SBATCH --time=' not in content

    def test_ld_library_path_present(self, sh_result):
        _, _, content = sh_result
        assert '/usr/local/lib' in content


# ═══════════════════════════════════════════════════════════════════════════
# 3. load_diffusivity_results
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadDiffusivityResults:

    def test_returns_empty_when_no_files(self, tmp_path):
        result = load_diffusivity_results(
            _STRUCTS, _NH, str(tmp_path / 'results')
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        rr = str(tmp_path / 'results')
        _make_table(rr, 'Ni3Mo', 1)
        result = load_diffusivity_results(['/data/Ni3Mo.lammps'], [1], rr)
        for col in ('T_K', 'D', 'sigma_D', 'R2'):
            assert col in result.columns

    def test_struct_column_populated(self, tmp_path):
        rr = str(tmp_path / 'results')
        _make_table(rr, 'Ni3Mo', 1)
        result = load_diffusivity_results(['/data/Ni3Mo.lammps'], [1], rr)
        assert 'struct' in result.columns
        assert (result['struct'] == 'Ni3Mo').all()

    def test_n_h_column_populated(self, tmp_path):
        rr = str(tmp_path / 'results')
        _make_table(rr, 'Ni3Mo', 4)
        result = load_diffusivity_results(['/data/Ni3Mo.lammps'], [4], rr)
        assert 'n_h' in result.columns
        assert (result['n_h'] == 4).all()

    def test_row_count_matches_temperatures(self, tmp_path):
        rr = str(tmp_path / 'results')
        _make_table(rr, 'Ni3Mo', 1, T_vals=[600., 800., 1000.])
        result = load_diffusivity_results(['/data/Ni3Mo.lammps'], [1], rr)
        assert len(result) == 3

    def test_missing_file_silently_skipped(self, tmp_path):
        rr = str(tmp_path / 'results')
        # Only create table for Ni, not Ni3Mo
        _make_table(rr, 'Ni', 1)
        result = load_diffusivity_results(
            ['/data/Ni3Mo.lammps', '/data/Ni.lammps'], [1], rr
        )
        # Ni3Mo missing — only Ni rows returned
        assert not result.empty
        assert (result['struct'] == 'Ni').all()


# ═══════════════════════════════════════════════════════════════════════════
# 4. qc_minimizations
# ═══════════════════════════════════════════════════════════════════════════

class TestQcMinimizations:

    def test_returns_empty_when_no_log_files(self, tmp_path):
        result = qc_minimizations(
            ['/data/Ni3Mo.lammps'], [1], str(tmp_path / 'results')
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_for_empty_struct_list(self, tmp_path):
        result = qc_minimizations([], [], str(tmp_path))
        assert result.empty


# ═══════════════════════════════════════════════════════════════════════════
# 5. qc_nvt_thermo
# ═══════════════════════════════════════════════════════════════════════════

class TestQcNvtThermo:

    def test_returns_empty_when_no_thermo_files(self, tmp_path):
        result = qc_nvt_thermo(
            ['/data/Ni3Mo.lammps'], [1], [600, 800],
            str(tmp_path / 'results'),
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_for_empty_struct_list(self, tmp_path):
        result = qc_nvt_thermo([], [], [], str(tmp_path))
        assert result.empty

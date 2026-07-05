"""
tests/test_neb_workflow.py
===========================
Tests for models/neb_workflow.py — offline, no LAMMPS or SLURM.

Covers:
  write_neb_run_script      — embedded config, phase labels, checkpoint guard
  write_neb_orchestrator_sh — SBATCH headers, conda, python run line
  load_neb_results          — DataFrame from pairs JSON; fallback directory scan
  collect_neb_results       — converged / missing counts; ranked JSON written
  plot_barrier_heatmap      — returns None for empty DataFrame
  plot_mep_overlay          — returns None for empty DataFrame
"""

import json
import os
import pathlib
import sys
from unittest.mock import MagicMock
import pytest
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# acat and models.surface_graph are not installed in the offline test env;
# mock them before neb_workflow is imported so the module-level imports succeed.
for _m in ('acat', 'acat.adsorption_sites'):
    sys.modules[_m] = MagicMock()
sys.modules['models.surface_graph'] = MagicMock()

import matplotlib
matplotlib.use('Agg')   # must precede any pyplot import

from models.neb_workflow import (
    write_neb_run_script,
    write_neb_orchestrator_sh,
    load_neb_results,
    collect_neb_results,
    plot_barrier_heatmap,
    plot_mep_overlay,
)


# ── shared config ────────────────────────────────────────────────────────────

_NEB_CFG = dict(
    bulk_min_path='/work/bulk_min.lammps',
    work_dir='/work/calculation',
    e_h2_gas=-6.77,
    slab_dir='/work/slabs',
    ads_dir='/work/ads',
    neb_dir='/work/neb',
    miller=(1, 1, 1),
    layers=12,
    vacuum=15.0,
    lat_repeat=(5, 6),
    sep_min=2.5,
    sep_max=6.0,
    graph_dist_min=2,
    prox_cutoff=5.0,
    n_images=18,
    spring_const=1.0,
    neb_ftol=0.05,
    h_height=1.5,
    gpu_slurm_cfg={'partition': 'multigpu', 'time': '24:00:00'},
    neb_slurm_cfg={'partition': 'short', 'time': '12:00:00'},
)

_SH_CFG = dict(
    orch_job_name='neb_orch',
    orch_partition='cpu',
    orch_cpus_per_task=4,
    orch_time='48:00:00',
    orch_openmpi_ver='4.1.4',
    orch_cuda_version='12.0',
    orch_conda_env='ase-env',
    orch_ld_paths=['/usr/local/lib'],
    out_py='/work/neb_run.py',
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


def _write_pairs_json(path, jobs):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(jobs, indent=2))


def _make_job(label, barrier_file, path_file=''):
    return {
        'label':        label,
        'is_site':      'site0',
        'fs_site1':     'site1',
        'fs_site2':     'site2',
        'E_IS':         -100.5,
        'E_FS':         -100.2,
        'delta_E':      0.30,
        'is_fs_dist':   2.8,
        'barrier_file': barrier_file,
        'path_file':    path_file,
        'neb_script':   '/work/neb/run_neb.py',
        'min_script':   '/work/neb/min_fs.lammps',
        'fsmin_sh':     '/work/neb/slurm_fsmin.sh',
        'neb_sh':       '/work/neb/slurm_neb.sh',
        'job_dir':      '/work/neb',
        'is_true_label': 'L1_hcp',
        'fs_true_label1': 'Ni_fcc',
        'fs_true_label2': 'Mo_hcp',
        'graph_dist':   3,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. write_neb_run_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNebRunScript:

    @pytest.fixture()
    def gen_result(self, tmp_path):
        out_py = str(tmp_path / 'scripts' / 'neb_run.py')
        ret = write_neb_run_script(**_NEB_CFG, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        return ret, out_py, content

    def test_file_created(self, gen_result):
        _, out_py, _ = gen_result
        assert pathlib.Path(out_py).exists()

    def test_returns_out_py_path(self, gen_result):
        ret, out_py, _ = gen_result
        assert ret == out_py

    def test_creates_parent_directory(self, tmp_path):
        out_py = str(tmp_path / 'a' / 'b' / 'neb_run.py')
        write_neb_run_script(**_NEB_CFG, out_py=out_py)
        assert pathlib.Path(out_py).exists()

    def test_bulk_min_path_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'BULK_MIN_PATH' in content
        assert '/work/bulk_min.lammps' in content

    def test_work_dir_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'WORK_DIR' in content
        assert '/work/calculation' in content

    def test_neb_dir_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'NEB_DIR' in content
        assert '/work/neb' in content

    def test_miller_and_neb_params_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'MILLER' in content
        assert 'N_IMAGES' in content
        assert 'SPRING_CONST' in content
        assert 'NEB_FTOL' in content

    def test_filter_params_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'SEP_MIN' in content
        assert 'SEP_MAX' in content
        assert 'PROX_CUTOFF' in content
        assert 'GRAPH_DIST_MIN' in content

    def test_ranked_barriers_checkpoint_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'ranked_barriers.json' in content

    def test_collect_neb_results_actually_called(self, gen_result):
        """Regression test for GitHub #7: ranked_barriers.json referenced
        in the checkpoint-guard path string is not proof it ever gets
        written — collect_neb_results(NEB_DIR) must actually be called
        after the NEB array completes, or the file never materialises
        and Phase E / Part 2's DH_DISS_EV auto-extraction silently starve."""
        _, _, content = gen_result
        assert 'collect_neb_results' in content
        assert 'collect_neb_results(NEB_DIR)' in content
        # must be imported from models.neb_workflow, not left undefined
        assert 'from models.neb_workflow import' in content
        import_block_start = content.index('from models.neb_workflow import')
        import_block = content[import_block_start:import_block_start + 200]
        assert 'collect_neb_results' in import_block

    def test_phase_e_job_dir_includes_neb_subfolder(self, gen_result):
        """Regression test: orchestrate_neb writes per-job dirs under
        {outdir}/neb/{label}/ (NEB_DIR already IS that outdir), so Phase E
        must reconstruct job_dir as NEB_DIR/neb/{label} — not NEB_DIR/{label}
        directly, which silently found no IS/TS structures for any job."""
        _, _, content = gen_result
        assert "os.path.join(NEB_DIR, 'neb', _lbl_e)" in content
        assert "os.path.join(NEB_DIR, _lbl_e)" not in content

    def test_orchestrate_full_neb_workflow_called(self, gen_result):
        _, _, content = gen_result
        assert 'orchestrate_full_neb_workflow' in content

    def test_phase_e_vibrations_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'Phase E' in content or 'Vibrational frequencies' in content or 'vibrations_diss' in content

    def test_auto_submit_called_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'auto_submit' in content

    def test_fsmin_auto_submit_result_dir_includes_neb_subfolder(self, gen_result):
        """Regression test: neb_final_relaxed.lammps lands at
        {NEB_DIR}/neb/{label}/ (orchestrate_neb's own 'neb' subfolder
        under the outdir it was given — NEB_DIR itself), so auto_submit's
        completion-check result_dir must be NEB_DIR/neb, not NEB_DIR
        directly — otherwise its glob never matches anything and every
        FS-min task is misreported as missing regardless of real status."""
        _, _, content = gen_result
        assert "os.path.join(NEB_DIR, 'neb')" in content

    def test_vib_slurm_cfg_defaults_to_neb_slurm_cfg(self, tmp_path):
        out_py = str(tmp_path / 'neb_run.py')
        write_neb_run_script(**_NEB_CFG, vib_slurm_cfg=None, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        # VIB_SLURM_CFG should be present (falls back to neb_slurm_cfg)
        assert 'VIB_SLURM_CFG' in content

    def test_e_h2_gas_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'E_H2_GAS' in content
        assert '-6.77' in content


# ═══════════════════════════════════════════════════════════════════════════
# 2. write_neb_orchestrator_sh
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNebOrchestratorSh:

    @pytest.fixture()
    def sh_result(self, tmp_path):
        out_sh = str(tmp_path / 'scripts' / 'neb_run.sh')
        ret = write_neb_orchestrator_sh(**_SH_CFG, out_sh=out_sh)
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
        assert '#SBATCH --job-name=neb_orch' in content

    def test_sbatch_partition_present(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --partition=cpu' in content

    def test_module_load_openmpi_present(self, sh_result):
        _, _, content = sh_result
        assert 'OpenMPI/4.1.4' in content

    def test_conda_activate_present(self, sh_result):
        _, _, content = sh_result
        assert 'conda activate ase-env' in content

    def test_python_run_line_present(self, sh_result):
        _, _, content = sh_result
        assert 'python /work/neb_run.py' in content

    def test_ld_library_path_present(self, sh_result):
        _, _, content = sh_result
        assert '/usr/local/lib' in content

    def test_default_time_used_when_none(self, tmp_path):
        out_sh = str(tmp_path / 'neb_run.sh')
        write_neb_orchestrator_sh(**{**_SH_CFG, 'orch_time': None}, out_sh=out_sh)
        content = pathlib.Path(out_sh).read_text()
        assert '48:00:00' in content


# ═══════════════════════════════════════════════════════════════════════════
# 3. load_neb_results
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadNebResults:

    def test_returns_empty_when_no_files(self, tmp_path):
        result = load_neb_results(str(tmp_path))
        assert isinstance(result, pd.DataFrame) and result.empty

    def test_returns_dataframe_with_barrier_file(self, tmp_path):
        bf = str(tmp_path / 'job_A' / 'neb_barrier.txt')
        _write_barrier(bf, E_abs=0.45)
        jobs = [_make_job('job_A', bf)]
        _write_pairs_json(str(tmp_path / 'neb_pairs.json'), jobs)
        result = load_neb_results(str(tmp_path))
        assert not result.empty

    def test_e_abs_column_present(self, tmp_path):
        bf = str(tmp_path / 'job_A' / 'neb_barrier.txt')
        _write_barrier(bf, E_abs=0.38)
        jobs = [_make_job('job_A', bf)]
        _write_pairs_json(str(tmp_path / 'neb_pairs.json'), jobs)
        result = load_neb_results(str(tmp_path))
        assert 'E_abs' in result.columns
        assert result['E_abs'].iloc[0] == pytest.approx(0.38)

    def test_missing_barrier_file_skipped(self, tmp_path):
        jobs = [_make_job('ghost', '/nonexistent/neb_barrier.txt')]
        _write_pairs_json(str(tmp_path / 'neb_pairs.json'), jobs)
        result = load_neb_results(str(tmp_path))
        assert result.empty

    def test_fallback_directory_scan_when_no_pairs_json(self, tmp_path):
        label_dir = tmp_path / 'job_X'
        bf = str(label_dir / 'neb_barrier.txt')
        _write_barrier(bf, E_abs=0.50)
        result = load_neb_results(str(tmp_path))
        assert not result.empty
        assert result['E_abs'].iloc[0] == pytest.approx(0.50)

    def test_sorted_by_e_abs_ascending(self, tmp_path):
        for name, ea in [('j1', 0.60), ('j2', 0.30)]:
            _write_barrier(str(tmp_path / name / 'neb_barrier.txt'), E_abs=ea)
            jobs = [_make_job(name, str(tmp_path / name / 'neb_barrier.txt'))]
        # Two separate reads since each test is independent — just use the scan path
        for name, ea in [('k1', 0.60), ('k2', 0.30)]:
            _write_barrier(str(tmp_path / name / 'neb_barrier.txt'), E_abs=ea)
        result = load_neb_results(str(tmp_path))
        assert list(result['E_abs']) == sorted(list(result['E_abs']))


# ═══════════════════════════════════════════════════════════════════════════
# 4. collect_neb_results
# ═══════════════════════════════════════════════════════════════════════════

class TestCollectNebResults:

    def test_counts_missing_when_barrier_file_absent(self, tmp_path):
        neb_subdir = tmp_path / 'neb'
        neb_subdir.mkdir()
        jobs = [_make_job('ghost', str(neb_subdir / 'ghost' / 'neb_barrier.txt'))]
        _write_pairs_json(str(neb_subdir / 'neb_pairs.json'), jobs)
        result = collect_neb_results(str(tmp_path))
        assert result['n_missing'] == 1
        assert result['n_converged'] == 0

    def test_counts_converged_when_barrier_file_present(self, tmp_path):
        neb_subdir = tmp_path / 'neb'
        neb_subdir.mkdir()
        bf = str(neb_subdir / 'job_A' / 'neb_barrier.txt')
        _write_barrier(bf, converged=True)
        jobs = [_make_job('job_A', bf)]
        _write_pairs_json(str(neb_subdir / 'neb_pairs.json'), jobs)
        result = collect_neb_results(str(tmp_path))
        assert result['n_converged'] == 1
        assert result['n_missing'] == 0

    def test_counts_failed_when_not_converged(self, tmp_path):
        neb_subdir = tmp_path / 'neb'
        neb_subdir.mkdir()
        bf = str(neb_subdir / 'job_B' / 'neb_barrier.txt')
        _write_barrier(bf, converged=False)
        jobs = [_make_job('job_B', bf)]
        _write_pairs_json(str(neb_subdir / 'neb_pairs.json'), jobs)
        result = collect_neb_results(str(tmp_path))
        assert result['n_failed'] == 1

    def test_ranked_json_written(self, tmp_path):
        neb_subdir = tmp_path / 'neb'
        neb_subdir.mkdir()
        bf = str(neb_subdir / 'job_A' / 'neb_barrier.txt')
        _write_barrier(bf)
        jobs = [_make_job('job_A', bf)]
        _write_pairs_json(str(neb_subdir / 'neb_pairs.json'), jobs)
        result = collect_neb_results(str(tmp_path), outdir=str(tmp_path))
        assert pathlib.Path(result['ranked_json']).exists()

    def test_results_list_has_one_entry_per_job(self, tmp_path):
        neb_subdir = tmp_path / 'neb'
        neb_subdir.mkdir()
        bf1 = str(neb_subdir / 'j1' / 'neb_barrier.txt')
        bf2 = str(neb_subdir / 'j2' / 'neb_barrier.txt')
        _write_barrier(bf1, E_abs=0.40)
        _write_barrier(bf2, E_abs=0.55)
        jobs = [_make_job('j1', bf1), _make_job('j2', bf2)]
        _write_pairs_json(str(neb_subdir / 'neb_pairs.json'), jobs)
        result = collect_neb_results(str(tmp_path))
        assert len(result['results']) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 5. plot_barrier_heatmap
# ═══════════════════════════════════════════════════════════════════════════

class TestPlotBarrierHeatmap:

    def test_returns_none_for_empty_dataframe(self, tmp_path):
        result = plot_barrier_heatmap(pd.DataFrame(), str(tmp_path))
        assert result is None

    def test_saves_png_for_nonempty_dataframe(self, tmp_path):
        df = pd.DataFrame([{
            'is_label': 'L1_hcp', 'fs_label1': 'Ni', 'fs_label2': 'Mo',
            'graph_dist': 3, 'n_grouped': 1,
            'E_abs': 0.45, 'E_des': 0.15, 'delta_E': 0.30,
            'converged': True, 'fmax_final': 0.04,
            'barrier_file': '', 'path_file': '',
        }])
        result = plot_barrier_heatmap(df, str(tmp_path))
        assert result is not None
        assert pathlib.Path(result).exists()
        assert result.endswith('barrier_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════
# 6. plot_mep_overlay
# ═══════════════════════════════════════════════════════════════════════════

class TestPlotMepOverlay:

    def test_returns_none_for_empty_dataframe(self, tmp_path):
        result = plot_mep_overlay(pd.DataFrame(), str(tmp_path))
        assert result is None

    def test_saves_png_for_nonempty_dataframe_with_no_path_files(self, tmp_path):
        df = pd.DataFrame([{
            'is_label': 'L1_hcp', 'fs_label1': 'Ni', 'fs_label2': 'Mo',
            'graph_dist': 3, 'n_grouped': 1,
            'E_abs': 0.45, 'E_des': 0.15, 'delta_E': 0.30,
            'converged': True, 'fmax_final': 0.04,
            'barrier_file': '', 'path_file': '/nonexistent/path.dat',
        }])
        result = plot_mep_overlay(df, str(tmp_path))
        # path file doesn't exist → no paths plotted → still saves an empty figure
        assert result is not None
        assert pathlib.Path(result).exists()
        assert result.endswith('mep_overlay.png')

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
from unittest.mock import MagicMock
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

    def test_arrhenius_json_path_is_per_run_not_shared(self, gen_result):
        """Each (stem, n_H) must write its own diffusivity_arrhenius.json —
        previously this was shared across all n_H for a structure, so every
        H-concentration but the last silently overwrote the others' fits."""
        _, _, content = gen_result
        assert "os.path.join(dirs['root'], 'diffusivity_arrhenius.json')" in content
        # must NOT be the old shared per-stem-only path
        assert "os.path.join(WORK_DIR, 'results', struct_stem, 'diffusivity_arrhenius.json')" not in content

    def test_lattice_params_json_save_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'lattice_params_vs_T.json' in content

    def test_per_combo_try_except_wraps_full_body(self, gen_result):
        """The try/except must span the WHOLE per-(struct, n_H) body — from
        right after `run_name = ...` through the final Arrhenius-json print —
        not just a leading fragment of it. A partial wrap would leave later
        phases (NVT, post-processing) able to crash the entire run again."""
        _, _, content = gen_result
        run_name_idx = content.index("run_name = f'{struct_stem}_{n_h}H'")
        try_idx      = content.index('try:', run_name_idx)
        except_idx   = content.index('except Exception as _exc:', try_idx)
        saved_idx    = content.rindex("print(f'  Saved diffusivity_arrhenius.json")
        assert run_name_idx < try_idx < saved_idx < except_idx

    def test_failure_manifest_and_nonzero_exit_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'diffusivity_failures.json' in content
        assert 'sys.exit(1)' in content


class TestDiffusivityFailureIsolation:
    """Behavioral proof (not just generated-text matching) that one
    structure's failure no longer kills the whole diffusivity_run.py run —
    see [[project_pipeline_test_bugs]] on the pre-fix diffusivity_workflow.py
    main loop having zero per-structure isolation."""

    def test_one_bad_structure_does_not_block_others(self, tmp_path, monkeypatch):
        import json

        work_dir = str(tmp_path / 'work')
        os.makedirs(work_dir, exist_ok=True)
        out_py = str(tmp_path / 'diffusivity_run.py')

        struct_a = str(tmp_path / 'A_supercell.lammps')
        struct_b = str(tmp_path / 'B_supercell.lammps')
        pathlib.Path(struct_a).write_text('dummy')
        pathlib.Path(struct_b).write_text('dummy')

        generate_diffusivity_scripts(
            input_structures=[struct_a, struct_b],
            n_h_values=[1, 3],
            temperatures=[400],
            work_dir=work_dir,
            nvt_wall_time='24:00:00',
            cutoff='23:55:00',
            gpu_partition='multigpu',
            gpu_time='24:00:00',
            timestep_ps=0.0005,
            tau_t_ps=0.1,
            n_equil_steps=100,
            n_prod_steps=100,
            thermo_every=10,
            dump_every=10,
            velocity_seed=42,
            restart_every=100,
            out_py=out_py,
        )

        # Stub every SLURM/LAMMPS side-effecting call as a no-op. None of
        # them ever produce `bulk_min.lammps`, so Phase 1a's `_require_file`
        # check deterministically raises for EVERY (struct, n_H) combo —
        # proving the loop keeps going regardless, rather than proving
        # anything about Phase 1a itself (already covered by the real
        # cluster smoke test).
        monkeypatch.setattr('models.create_slurm.write_slurm_job', lambda **kw: None)
        monkeypatch.setattr('models.create_slurm.submit_slurm_job', lambda *a, **kw: 'fakejid')
        monkeypatch.setattr('models.create_slurm.wait_for_jobs', lambda *a, **kw: None)
        monkeypatch.setattr('models.lammps_script.write_minimization_script', lambda **kw: None)

        src = pathlib.Path(out_py).read_text()
        with pytest.raises(SystemExit) as exc_info:
            exec(compile(src, out_py, 'exec'), {'__name__': '__main__'})
        assert exc_info.value.code == 1

        failures_path = os.path.join(work_dir, 'diffusivity_failures.json')
        assert os.path.exists(failures_path)
        failures = json.loads(pathlib.Path(failures_path).read_text())

        # All 4 (struct, n_H) combos were attempted despite each one failing —
        # the pre-fix behavior would have stopped after the very first one.
        assert len(failures) == 4
        run_names = {f['run_name'] for f in failures}
        assert run_names == {
            'A_supercell_1H', 'A_supercell_3H',
            'B_supercell_1H', 'B_supercell_3H',
        }
        assert all('Expected output missing' in f['error'] for f in failures)


class TestSharedBareBulkAndNpt:
    """Regression tests for the GitHub issue: Phase 1a (bare-bulk min) and
    Phase 1b Step A (NPT) were redone once per n_H instead of once per
    structure, and never reused pipeline.ipynb's pre-pipeline
    bulk_min_{stem}.lammps. See [[project_pipeline_test_bugs]]."""

    def test_min_bare_out_uses_shared_pre_pipeline_path(self, tmp_path):
        out_py = str(tmp_path / 'diffusivity_run.py')
        generate_diffusivity_scripts(**_GEN_CFG, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        assert "os.path.join(WORK_DIR, 'structures', f'bulk_min_{struct_stem}.lammps')" in content
        assert "os.path.join(dirs['structures'], 'bulk_min.lammps')" not in content

    def test_npt_uses_its_own_gpu_partition_not_short_gpu(self, tmp_path):
        """NPT is real chained MD (heat+prod), same category as slab
        surface relaxation -- not a quick one-shot minimization like
        bare-bulk/bulk+H min, which stay on SHORT_GPU_SLURM_CFG. Regression
        test for the GitHub issue on gpu vs sharing partition placement."""
        out_py = str(tmp_path / 'diffusivity_run.py')
        generate_diffusivity_scripts(
            **_GEN_CFG, npt_gpu_partition='gpu', npt_gpu_time='08:00:00', out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        assert "NPT_GPU_PARTITION    = 'gpu'" in content
        assert "NPT_GPU_TIME         = '08:00:00'" in content
        npt_call_idx = content.index("job_name=f'npt_{T}K_{struct_stem}'")
        npt_call_block = content[npt_call_idx:npt_call_idx + 200]
        assert 'slurm_config=NPT_GPU_SLURM_CFG' in npt_call_block
        assert 'slurm_config=SHORT_GPU_SLURM_CFG' not in npt_call_block

    def test_bare_bulk_and_bulk_h_min_still_use_short_gpu(self, tmp_path):
        out_py = str(tmp_path / 'diffusivity_run.py')
        generate_diffusivity_scripts(**_GEN_CFG, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        min_bare_idx = content.index("job_name=f'min_bare_{struct_stem}'")
        assert 'slurm_config=SHORT_GPU_SLURM_CFG' in content[min_bare_idx:min_bare_idx + 200]
        min_h_idx = content.index("job_name=f'min_h_{T}K_{run_name}'")
        assert 'slurm_config=SHORT_GPU_SLURM_CFG' in content[min_h_idx:min_h_idx + 200]

    def test_shared_block_precedes_n_h_loop(self, tmp_path):
        out_py = str(tmp_path / 'diffusivity_run.py')
        generate_diffusivity_scripts(**_GEN_CFG, out_py=out_py)
        content = pathlib.Path(out_py).read_text()
        shared_idx = content.index('Shared, n_H-independent')
        loop_idx   = content.index('for n_h in N_H_VALUES:')
        assert shared_idx < loop_idx

    def test_shared_block_computed_once_across_multiple_n_h(self, tmp_path, monkeypatch):
        """The actual compute-savings claim: with 2 n_H values for one
        structure, Phase 1a + NPT must submit exactly one min_bare job and
        one NPT job total -- not one pair per n_H (which would be 2 of
        each, 4 total)."""
        import json

        work_dir = str(tmp_path / 'work')
        os.makedirs(work_dir, exist_ok=True)
        out_py = str(tmp_path / 'diffusivity_run.py')

        struct_a = str(tmp_path / 'A_supercell.lammps')
        pathlib.Path(struct_a).write_text('dummy')

        generate_diffusivity_scripts(
            input_structures=[struct_a],
            n_h_values=[1, 3],
            temperatures=[400],
            work_dir=work_dir,
            nvt_wall_time='24:00:00',
            cutoff='23:55:00',
            gpu_partition='multigpu',
            gpu_time='24:00:00',
            timestep_ps=0.0005,
            tau_t_ps=0.1,
            n_equil_steps=100,
            n_prod_steps=100,
            thermo_every=10,
            dump_every=10,
            velocity_seed=42,
            restart_every=100,
            out_py=out_py,
        )

        bulk_min_path = os.path.join(work_dir, 'structures', 'bulk_min_A_supercell.lammps')
        npt_final = f'{os.path.splitext(bulk_min_path)[0]}_npt_final_400K.lammps'
        npt_dump  = os.path.join(work_dir, 'results', 'A_supercell', 'structures', 'npt_boxdims_400K.dat')

        def _fake_wait(jobs):
            if 'min_bare' in jobs:
                os.makedirs(os.path.dirname(bulk_min_path), exist_ok=True)
                pathlib.Path(bulk_min_path).write_text('dummy minimized')
            elif any(k.startswith('npt_') for k in jobs):
                pathlib.Path(npt_final).write_text('dummy npt final')
                os.makedirs(os.path.dirname(npt_dump), exist_ok=True)
                pathlib.Path(npt_dump).write_text(
                    '# Step Lx Ly Lz\n1 17.5 17.5 17.5\n2 17.5 17.5 17.5\n')

        submit_mock = MagicMock(return_value='fakejid')
        wait_mock = MagicMock(side_effect=_fake_wait)
        monkeypatch.setattr('models.create_slurm.submit_slurm_job', submit_mock)
        monkeypatch.setattr('models.create_slurm.write_slurm_job', lambda **kw: None)
        monkeypatch.setattr('models.create_slurm.write_chained_slurm_job', lambda **kw: None)
        monkeypatch.setattr('models.create_slurm.wait_for_jobs', wait_mock)
        monkeypatch.setattr('models.lammps_script.write_minimization_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_npt_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_npt_restart_script', lambda **kw: None)
        monkeypatch.setattr(
            'models.structure.insert_hydrogen',
            lambda **kw: (_ for _ in ()).throw(RuntimeError('stop after shared block')))

        src = pathlib.Path(out_py).read_text()
        with pytest.raises(SystemExit):
            exec(compile(src, out_py, 'exec'), {'__name__': '__main__'})

        # 1 min_bare submit + 1 npt_400K submit = 2 total, regardless of
        # there being 2 n_H values. The pre-fix behavior would submit both
        # once per n_H (4 total).
        assert submit_mock.call_count == 2
        assert wait_mock.call_count == 2

        failures = json.loads(pathlib.Path(work_dir, 'diffusivity_failures.json').read_text())
        # Both n_H values reach (and fail at) Phase 1b Step B independently
        # -- proves the shared block itself succeeded via reuse, and
        # per-n_H failure isolation still works on top of it.
        assert len(failures) == 2
        assert all('stop after shared block' in f['error'] for f in failures)

    def test_two_structures_two_n_h_all_succeed_end_to_end(self, tmp_path, monkeypatch):
        """Full happy-path exec across 2 structures x 2 n_H, everything
        mocked to succeed. Every other test in this module only ever
        exercises a failure branch -- this is the one proof that the
        hoisted shared-block refactor doesn't crash, produces output for
        every (structure, n_H) combo, and doesn't cross-contaminate the
        shared _a0_by_T/npt_final_paths between structures A and B."""
        import json as _json

        work_dir = str(tmp_path / 'work')
        os.makedirs(work_dir, exist_ok=True)
        out_py = str(tmp_path / 'diffusivity_run.py')

        stems = ['A_supercell', 'B_supercell']
        struct_paths = {}
        for stem in stems:
            p = str(tmp_path / f'{stem}.lammps')
            pathlib.Path(p).write_text(f'dummy {stem}')
            struct_paths[stem] = p

        generate_diffusivity_scripts(
            input_structures=[struct_paths[s] for s in stems],
            n_h_values=[1, 3],
            temperatures=[600],
            work_dir=work_dir,
            nvt_wall_time='24:00:00',
            cutoff='23:55:00',
            gpu_partition='multigpu',
            gpu_time='24:00:00',
            timestep_ps=0.0005,
            tau_t_ps=0.1,
            n_equil_steps=100,
            n_prod_steps=100,
            thermo_every=10,
            dump_every=10,
            velocity_seed=42,
            restart_every=100,
            out_py=out_py,
        )

        def _bulk_min_path(stem):
            return os.path.join(work_dir, 'structures', f'bulk_min_{stem}.lammps')

        def _npt_final_path(stem, T=600):
            return f'{os.path.splitext(_bulk_min_path(stem))[0]}_npt_final_{T}K.lammps'

        def _npt_dump_path(stem, T=600):
            return os.path.join(work_dir, 'results', stem, 'structures', f'npt_boxdims_{T}K.dat')

        _state = {'min_bare_calls': 0, 'npt_calls': 0}
        _insert_hydrogen_calls = []

        def _fake_submit(sh_path):
            return f'jid_{os.path.basename(sh_path)}'

        def _fake_wait(jobs):
            if 'min_bare' in jobs:
                stem = stems[_state['min_bare_calls']]
                _state['min_bare_calls'] += 1
                p = _bulk_min_path(stem)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                pathlib.Path(p).write_text(f'minimized {stem}')
            elif any(k.startswith('npt_') for k in jobs):
                stem = stems[_state['npt_calls']]
                _state['npt_calls'] += 1
                pathlib.Path(_npt_final_path(stem)).write_text(f'npt final {stem}')
                dump_p = _npt_dump_path(stem)
                os.makedirs(os.path.dirname(dump_p), exist_ok=True)
                pathlib.Path(dump_p).write_text(
                    '# Step Lx Ly Lz\n1 17.5 17.5 17.5\n2 17.5 17.5 17.5\n')
            # else: min_h_* / NVT chain waits -- their outputs are created
            # synchronously by the write_minimization_script/
            # write_chained_slurm_job mocks below, so no action needed here.

        def _fake_write_min(**kw):
            pathlib.Path(kw['min_output']).write_text('dummy minimized')

        def _fake_write_chained(**kw):
            pathlib.Path(kw['out_path'] + '.done').write_text('done')

        def _fake_insert_hydrogen(**kw):
            _insert_hydrogen_calls.append(kw['bulk_min_path'])
            out_dir = kw['out_dir']
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, 'bulk_h.lammps')
            pathlib.Path(p).write_text('dummy bulk+H')
            return p, [0.0, 0.0, 0.0], 1.8

        monkeypatch.setattr('models.create_slurm.submit_slurm_job', _fake_submit)
        monkeypatch.setattr('models.create_slurm.write_slurm_job', lambda **kw: None)
        monkeypatch.setattr('models.create_slurm.write_chained_slurm_job', _fake_write_chained)
        monkeypatch.setattr('models.create_slurm.wait_for_jobs', _fake_wait)
        monkeypatch.setattr('models.lammps_script.write_minimization_script', _fake_write_min)
        monkeypatch.setattr('models.lammps_script.write_npt_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_npt_restart_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_nvt_bulk_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_nvt_equil_restart_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_nvt_prod_restart_script', lambda **kw: None)
        monkeypatch.setattr('models.structure.insert_hydrogen', _fake_insert_hydrogen)
        monkeypatch.setattr('models.diffusivity_post_processing.run_diffusivity_pipeline',
                            lambda **kw: {'D': 1e-10, 'sigma_D': 1e-11, 'R2': 0.99})
        monkeypatch.setattr('models.diffusivity_post_processing.save_diffusivity_table',
                            lambda *a, **kw: None)
        monkeypatch.setattr('models.diffusivity_post_processing.run_arrhenius_pipeline',
                            lambda **kw: {
                                'Ea': 0.4, 'D0': 1e-7, 'D0_err': 1e-8, 'Ea_err': 0.01, 'R2': 0.98,
                                'T_arr': [600.0], 'D_arr': [1e-10], 'D_err_arr': [1e-11],
                            })

        src = pathlib.Path(out_py).read_text()
        # Full success: must run to completion with no exception/SystemExit.
        exec(compile(src, out_py, 'exec'), {'__name__': '__main__'})

        # Shared block ran exactly once per structure (2), not once per
        # (structure, n_H) (which would be 4) -- the actual fix.
        assert _state['min_bare_calls'] == 2
        assert _state['npt_calls'] == 2

        failures = _json.loads(pathlib.Path(work_dir, 'diffusivity_failures.json').read_text())
        assert failures == []

        for stem in stems:
            for n_h in (1, 3):
                arr_path = os.path.join(work_dir, 'results', f'{stem}_{n_h}H',
                                        'diffusivity_arrhenius.json')
                assert os.path.exists(arr_path), f'missing {arr_path}'

        # No cross-contamination: each structure's H-insertion must use
        # its OWN NPT final structure, never the other structure's.
        assert _insert_hydrogen_calls.count(_npt_final_path('A_supercell')) == 2  # n_H=1,3
        assert _insert_hydrogen_calls.count(_npt_final_path('B_supercell')) == 2

    def test_corrupt_npt_dump_caught_not_a_crash(self, tmp_path, monkeypatch):
        """Realistic production failure mode: SLURM reports the NPT job
        exited 0 (so _require_file's existence check passes), but the
        LAMMPS run itself crashed partway and left an empty/header-only
        dump file. get_lattice_parameter_from_dump raises ValueError on
        this -- must be caught by the shared block's try/except (not
        propagate and kill the whole script), marking all n_H for that
        structure as failed with a clear message."""
        import json as _json

        work_dir = str(tmp_path / 'work')
        os.makedirs(work_dir, exist_ok=True)
        out_py = str(tmp_path / 'diffusivity_run.py')

        struct_a = str(tmp_path / 'A_supercell.lammps')
        pathlib.Path(struct_a).write_text('dummy')

        generate_diffusivity_scripts(
            input_structures=[struct_a],
            n_h_values=[1, 3],
            temperatures=[600],
            work_dir=work_dir,
            nvt_wall_time='24:00:00',
            cutoff='23:55:00',
            gpu_partition='multigpu',
            gpu_time='24:00:00',
            timestep_ps=0.0005,
            tau_t_ps=0.1,
            n_equil_steps=100,
            n_prod_steps=100,
            thermo_every=10,
            dump_every=10,
            velocity_seed=42,
            restart_every=100,
            out_py=out_py,
        )

        bulk_min_path = os.path.join(work_dir, 'structures', 'bulk_min_A_supercell.lammps')
        npt_final = f'{os.path.splitext(bulk_min_path)[0]}_npt_final_600K.lammps'
        npt_dump  = os.path.join(work_dir, 'results', 'A_supercell', 'structures', 'npt_boxdims_600K.dat')

        def _fake_wait(jobs):
            if 'min_bare' in jobs:
                os.makedirs(os.path.dirname(bulk_min_path), exist_ok=True)
                pathlib.Path(bulk_min_path).write_text('dummy minimized')
            elif any(k.startswith('npt_') for k in jobs):
                # SLURM/LAMMPS "succeeded" but wrote nothing but a header --
                # exactly what a mid-run crash that still exits 0 looks like.
                pathlib.Path(npt_final).write_text('dummy npt final')
                os.makedirs(os.path.dirname(npt_dump), exist_ok=True)
                pathlib.Path(npt_dump).write_text('# Step Lx Ly Lz\n')

        monkeypatch.setattr('models.create_slurm.submit_slurm_job', MagicMock(return_value='fakejid'))
        monkeypatch.setattr('models.create_slurm.write_slurm_job', lambda **kw: None)
        monkeypatch.setattr('models.create_slurm.write_chained_slurm_job', lambda **kw: None)
        monkeypatch.setattr('models.create_slurm.wait_for_jobs', _fake_wait)
        monkeypatch.setattr('models.lammps_script.write_minimization_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_npt_script', lambda **kw: None)
        monkeypatch.setattr('models.lammps_script.write_npt_restart_script', lambda **kw: None)

        src = pathlib.Path(out_py).read_text()
        # Must NOT propagate the ValueError -- only sys.exit(1) via the
        # normal _FAILURES manifest path.
        with pytest.raises(SystemExit) as exc_info:
            exec(compile(src, out_py, 'exec'), {'__name__': '__main__'})
        assert exc_info.value.code == 1

        failures = _json.loads(pathlib.Path(work_dir, 'diffusivity_failures.json').read_text())
        assert len(failures) == 2  # both n_H=1 and n_H=3 skipped for A_supercell
        assert all('No data rows found' in f['error'] for f in failures)


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

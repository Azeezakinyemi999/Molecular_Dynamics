"""
tests/test_neb_subsurface.py
==============================
Unit tests for models/neb_subsurface.py -- Hop A/B NEB orchestration.

Covers the changes made this session (previously had zero direct test
coverage despite being modified three times):
  - no-op-stub skip pattern for already-done FS-min/NEB per-site SLURM
    commands (mirrors Section C's FS-min convention in neb_workflow.py)
  - the 0-index/1-index off-by-one fix in the fsmin/neb array generation
  - submit_with_retry() wired into the auto_submit()-throttled arrays
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.config import SLURM_DEFAULTS, ELEM_STR_7, E2T_7, MASSES_7


def _is_lammps(tmp_path, name='is.lammps'):
    from ase import Atoms
    from ase.io import write as ase_write
    atoms = Atoms(
        'Ni8H',
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0],
                   [0, 0, 2.5], [2.5, 0, 2.5], [0, 2.5, 2.5], [2.5, 2.5, 2.5],
                   [1.25, 1.25, 1.25]],
        cell=[10, 10, 10], pbc=[True, True, False],
    )
    p = str(tmp_path / name)
    ase_write(p, atoms, format='lammps-data', masses=True, specorder=['Ni', 'H'])
    return p


class TestOrchestrateHopaNebSkip:

    def _run(self, tmp_path, monkeypatch, *, pre_create_fs_relaxed=False,
              pre_create_neb_barrier=False):
        from models.neb_subsurface import orchestrate_hopa_neb

        is_path = _is_lammps(tmp_path)
        outdir = tmp_path / 'sub_neb'

        if pre_create_fs_relaxed or pre_create_neb_barrier:
            job_dir = outdir / 'hopa' / 's_0'
            job_dir.mkdir(parents=True)
            if pre_create_fs_relaxed:
                (job_dir / 'sub1_fs_relaxed.lammps').write_text('fake relaxed FS')
            if pre_create_neb_barrier:
                (job_dir / 'neb_barrier.txt').write_text('fake barrier')

        subsurface_sites = [{'site_id': 'ss1_a', 'position': [1.25, 1.25, -1.5],
                              'layer_classification': 'subsurface_1'}]
        surface_connections = [('ss1_a', 's_0', 0.8)]

        # orchestrate_hopa_neb/orchestrate_hopb_neb import these locally
        # (inside the function body), so they must be patched at their
        # source module -- patching models.neb_subsurface.X has no effect
        # since that name is never bound at module level there.
        write_min_calls = []
        monkeypatch.setattr(
            'models.lammps_script.write_adsorbate_min_script',
            lambda **kw: write_min_calls.append(kw),
        )
        monkeypatch.setattr('models.ase_neb.run_neb_pipeline',
                             lambda **kw: str(tmp_path / 'run_neb.py'))

        write_slurm_calls = []
        monkeypatch.setattr(
            'models.create_slurm.write_slurm_job',
            lambda **kw: (write_slurm_calls.append(kw),
                          str(tmp_path / f"{kw['job_name']}.sh"))[1],
        )
        write_chained_calls = []
        monkeypatch.setattr(
            'models.create_slurm.write_chained_slurm_job',
            lambda **kw: (write_chained_calls.append(kw), str(tmp_path / 'slurm_neb.sh'))[1],
        )

        result = orchestrate_hopa_neb(
            dedup_is_labels=[('s_0', is_path, -50.0)],
            subsurface_graph=(None, subsurface_sites),
            surface_connections=surface_connections,
            outdir=str(outdir),
            masses=MASSES_7, e2t=E2T_7, elem_str=ELEM_STR_7,
            slurm_opts=dict(SLURM_DEFAULTS, partition='sharing', time='01:00:00'),
            neb_slurm_opts=dict(SLURM_DEFAULTS, partition='short', time='12:00:00',
                                 gpu=None, cpus_per_task=18),
            n_images=4, dry_run=True,
            z_freeze_cutoff=5.0,
        )
        return result, write_min_calls, write_slurm_calls, write_chained_calls

    def test_fsmin_no_op_stub_when_already_done(self, tmp_path, monkeypatch):
        _, _, write_slurm_calls, _ = self._run(
            tmp_path, monkeypatch, pre_create_fs_relaxed=True)
        fsmin_call = next(c for c in write_slurm_calls if c['job_name'] == 'fsmin_hopa_s_0')
        assert any('skip' in c.lower() for c in fsmin_call['commands'])
        assert not any('lmp' in c.lower() or 'LAMMPS' in c for c in fsmin_call['commands'])

    def test_fsmin_real_command_when_not_done(self, tmp_path, monkeypatch):
        _, _, write_slurm_calls, _ = self._run(
            tmp_path, monkeypatch, pre_create_fs_relaxed=False)
        fsmin_call = next(c for c in write_slurm_calls if c['job_name'] == 'fsmin_hopa_s_0')
        assert not any('skip' in c.lower() for c in fsmin_call['commands'])

    def test_neb_no_op_stub_when_already_done(self, tmp_path, monkeypatch):
        _, _, write_slurm_calls, write_chained_calls = self._run(
            tmp_path, monkeypatch, pre_create_fs_relaxed=True, pre_create_neb_barrier=True)
        assert write_chained_calls == []
        neb_call = next(c for c in write_slurm_calls if c['job_name'] == 'neb_hopa_s_0')
        assert any('skip' in c.lower() for c in neb_call['commands'])

    def test_neb_real_chained_job_when_not_done(self, tmp_path, monkeypatch):
        _, _, _, write_chained_calls = self._run(
            tmp_path, monkeypatch, pre_create_fs_relaxed=False, pre_create_neb_barrier=False)
        assert len(write_chained_calls) == 1

    def test_array_uses_zero_indexed_range_and_offset_sed(self, tmp_path, monkeypatch):
        # auto_submit() submits array chunks 0-indexed; job_index.txt is
        # read via `sed -n Np` (1-indexed). Both the embedded array_range
        # and the sed offset must account for this, or the last site is
        # silently dropped and task 0 wastes a slot on a nonexistent line
        # (the exact bug already fixed for neb_workflow.py's arrays).
        _, _, write_slurm_calls, _ = self._run(tmp_path, monkeypatch)
        for array_name in ('hopa_fsmin_array', 'hopa_neb_array'):
            array_call = next(c for c in write_slurm_calls if c['job_name'] == array_name)
            assert array_call['array_range'] == (0, 0)   # 1 site -> indices [0, 0]
            assert any('SLURM_ARRAY_TASK_ID+1' in c for c in array_call['commands'])


class TestOrchestrateHopbNebSkip:
    """Mirrors TestOrchestrateHopaNebSkip for Hop B -- same fix pattern,
    separate function (orchestrate_hopb_neb), independently verified."""

    def _run(self, tmp_path, monkeypatch, *, pre_create_fs_relaxed=False,
              pre_create_neb_barrier=False):
        from models.neb_subsurface import orchestrate_hopb_neb

        hopa_outdir = tmp_path / 'sub_neb' / 'hopa'
        hopa_job_dir = hopa_outdir / 's_0'
        hopa_job_dir.mkdir(parents=True)
        sub1_relaxed = _is_lammps(hopa_job_dir, name='sub1_fs_relaxed.lammps')
        (hopa_job_dir / 'fs_min.log').write_text(
            'Step PotEng\n0 -50.0\n1 -50.5\nLoop time'
        )

        outdir = tmp_path / 'sub_neb'
        if pre_create_fs_relaxed or pre_create_neb_barrier:
            job_dir = outdir / 'hopb' / 's_0'
            job_dir.mkdir(parents=True)
            if pre_create_fs_relaxed:
                (job_dir / 'sub2_fs_relaxed.lammps').write_text('fake relaxed FS')
            if pre_create_neb_barrier:
                (job_dir / 'neb_barrier.txt').write_text('fake barrier')

        subsurface_sites = [
            {'site_id': 'ss1_a', 'position': [1.25, 1.25, -1.5],
             'layer_classification': 'subsurface_1'},
            {'site_id': 'ss2_a', 'position': [1.25, 1.25, -3.5],
             'layer_classification': 'subsurface_2'},
        ]
        import networkx as nx
        G = nx.Graph()
        for s in subsurface_sites:
            G.add_node(s['site_id'], **s)
        G.add_edge('ss1_a', 'ss2_a')

        # orchestrate_hopa_neb/orchestrate_hopb_neb import these locally
        # (inside the function body), so they must be patched at their
        # source module -- patching models.neb_subsurface.X has no effect
        # since that name is never bound at module level there.
        write_min_calls = []
        monkeypatch.setattr(
            'models.lammps_script.write_adsorbate_min_script',
            lambda **kw: write_min_calls.append(kw),
        )
        monkeypatch.setattr('models.ase_neb.run_neb_pipeline',
                             lambda **kw: str(tmp_path / 'run_neb.py'))

        write_slurm_calls = []
        monkeypatch.setattr(
            'models.create_slurm.write_slurm_job',
            lambda **kw: (write_slurm_calls.append(kw),
                          str(tmp_path / f"{kw['job_name']}.sh"))[1],
        )
        write_chained_calls = []
        monkeypatch.setattr(
            'models.create_slurm.write_chained_slurm_job',
            lambda **kw: (write_chained_calls.append(kw), str(tmp_path / 'slurm_neb.sh'))[1],
        )

        hopa_jobs = [{'sid': 's_0', 'ss1_id': 'ss1_a'}]
        result = orchestrate_hopb_neb(
            hopa_jobs=hopa_jobs,
            hopa_outdir=str(hopa_outdir),
            subsurface_graph=(G, subsurface_sites),
            outdir=str(outdir),
            masses=MASSES_7, e2t=E2T_7, elem_str=ELEM_STR_7,
            slurm_opts=dict(SLURM_DEFAULTS, partition='sharing', time='01:00:00'),
            neb_slurm_opts=dict(SLURM_DEFAULTS, partition='short', time='12:00:00',
                                 gpu=None, cpus_per_task=18),
            n_images=4, dry_run=True,
            z_freeze_cutoff=5.0,
        )
        return result, write_min_calls, write_slurm_calls, write_chained_calls

    def test_fsmin_no_op_stub_when_already_done(self, tmp_path, monkeypatch):
        _, _, write_slurm_calls, _ = self._run(
            tmp_path, monkeypatch, pre_create_fs_relaxed=True)
        fsmin_call = next(c for c in write_slurm_calls if c['job_name'] == 'fsmin_hopb_s_0')
        assert any('skip' in c.lower() for c in fsmin_call['commands'])

    def test_neb_no_op_stub_when_already_done(self, tmp_path, monkeypatch):
        _, _, write_slurm_calls, write_chained_calls = self._run(
            tmp_path, monkeypatch, pre_create_fs_relaxed=True, pre_create_neb_barrier=True)
        assert write_chained_calls == []
        neb_call = next(c for c in write_slurm_calls if c['job_name'] == 'neb_hopb_s_0')
        assert any('skip' in c.lower() for c in neb_call['commands'])

    def test_neb_real_chained_job_when_not_done(self, tmp_path, monkeypatch):
        _, _, _, write_chained_calls = self._run(
            tmp_path, monkeypatch, pre_create_fs_relaxed=False, pre_create_neb_barrier=False)
        assert len(write_chained_calls) == 1

    def test_array_uses_zero_indexed_range_and_offset_sed(self, tmp_path, monkeypatch):
        _, _, write_slurm_calls, _ = self._run(tmp_path, monkeypatch)
        for array_name in ('hopb_fsmin_array', 'hopb_neb_array'):
            array_call = next(c for c in write_slurm_calls if c['job_name'] == array_name)
            assert array_call['array_range'] == (0, 0)
            assert any('SLURM_ARRAY_TASK_ID+1' in c for c in array_call['commands'])

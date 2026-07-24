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

    def test_job_dict_carries_sub1_env(self, tmp_path, monkeypatch):
        # Part 2: the sub1 oct-site environment must be persisted in each Hop A
        # job dict so Part 6 can key k_entry/k_exit by environment.
        result, _, _, _ = self._run(tmp_path, monkeypatch)
        assert 'sub1_env' in result['jobs'][0]


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

    def test_job_dict_carries_sub1_and_sub2_env(self, tmp_path, monkeypatch):
        # Part 2: Hop B job dicts must carry both oct-site environments; sub2_env
        # is the per-environment key Part 6 uses for the deeper (sub1→sub2)
        # entry/exit rates in the two-layer KMC.
        result, _, _, _ = self._run(tmp_path, monkeypatch)
        assert 'sub1_env' in result['jobs'][0]
        assert 'sub2_env' in result['jobs'][0]


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — subsurface entry maps (surface → sub1 → sub2)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_subsurface_graph():
    """(G, subsurface_sites) with two sub1 (one mapped, one unmatched) + one sub2."""
    import networkx as nx
    subsurface_sites = [
        {'site_id': 'ss_0', 'position': [1.0, 1.0, 10.0],
         'layer_classification': 'subsurface_1', 'composition_label': 'Ni6_oct'},
        {'site_id': 'ss_1', 'position': [1.0, 1.0, 8.0],
         'layer_classification': 'subsurface_2', 'composition_label': 'Ni5Mo_oct'},
        {'site_id': 'ss_2', 'position': [3.0, 3.0, 10.0],
         'layer_classification': 'subsurface_1', 'composition_label': 'Ni4Mo2_oct'},
    ]
    G = nx.Graph()
    for s in subsurface_sites:
        G.add_node(s['site_id'], position=s['position'],
                   layer_classification=s['layer_classification'],
                   composition_label=s['composition_label'])
    G.add_edge('ss_0', 'ss_1')          # ss_0 (sub1) → ss_1 (sub2)
    # ss_2 has no subsurface_2 neighbour on purpose (unmatched case)
    return G, subsurface_sites


class TestBuildSub1Sub2Map:

    def test_matched_and_unmatched(self, tmp_path):
        from models.neb_subsurface import build_sub1_sub2_map
        out = str(tmp_path / 'sub1_sub2_map.json')
        m = build_sub1_sub2_map(_fake_subsurface_graph(), out_json=out)

        assert m['ss_0']['sub2_id'] == 'ss_1'
        assert m['ss_0']['sub1_env'] == 'Ni6_oct'
        assert m['ss_0']['sub2_env'] == 'Ni5Mo_oct'
        # unmatched sub1 recorded, not dropped
        assert 'ss_2' in m
        assert m['ss_2']['sub2_id'] is None
        assert 'reason' in m['ss_2']
        assert os.path.exists(out)

    def test_only_sub1_sites_are_keys(self, tmp_path):
        from models.neb_subsurface import build_sub1_sub2_map
        m = build_sub1_sub2_map(_fake_subsurface_graph())
        # ss_1 is a sub2 site — must not appear as a top-level (sub1) key
        assert set(m.keys()) == {'ss_0', 'ss_2'}


def _write_h_site(dirpath, sid, pe):
    """Write a minimal h_atom_{sid}_relaxed.lammps + h_min_{sid}.log pair."""
    os.makedirs(dirpath, exist_ok=True)
    open(os.path.join(dirpath, f'h_atom_{sid}_relaxed.lammps'), 'w').write('fake\n')
    open(os.path.join(dirpath, f'h_min_{sid}.log'), 'w').write(f'pe_final_eV: {pe}\n')


class TestCollectEntryHSources:

    def _ranked(self, dirpath, jobs):
        os.makedirs(dirpath, exist_ok=True)
        import json as _j
        with open(os.path.join(dirpath, 'ranked_barriers.json'), 'w') as f:
            _j.dump(jobs, f)

    def test_only_converged_products_seed_entry(self, tmp_path):
        from models.neb_subsurface import collect_entry_h_sources
        neb_dir = str(tmp_path / 'neb')
        ph2 = str(tmp_path / 'phase2_h')
        self._ranked(neb_dir, [
            {'label': 's_9__s_1+s_2',  'fs_site1': 's_1', 'fs_site2': 's_2', 'converged': True},
            {'label': 's_9__s_2+s_3',  'fs_site1': 's_2', 'fs_site2': 's_3', 'converged': True},
            {'label': 's_9__s_4+s_5',  'fs_site1': 's_4', 'fs_site2': 's_5', 'converged': False},
        ])
        for sid, pe in [('s_1', -100.1), ('s_2', -100.2), ('s_3', -100.3)]:
            _write_h_site(ph2, sid, pe)

        out = str(tmp_path / 'entry_h_sources.json')
        triples = collect_entry_h_sources(neb_dir, ph2, out_json=out)
        sids = [t[0] for t in triples]
        # s_1, s_2, s_3 from converged runs; s_2 deduped once; s_4/s_5 excluded (unconverged)
        assert sids == ['s_1', 's_2', 's_3']
        # provenance: s_2 appears in two runs
        import json as _j
        recs = {r['surface_sid']: r for r in _j.load(open(out))}
        assert len(recs['s_2']['source_diss_runs']) == 2

    def test_missing_structure_or_log_skipped(self, tmp_path):
        from models.neb_subsurface import collect_entry_h_sources
        neb_dir = str(tmp_path / 'neb')
        ph2 = str(tmp_path / 'phase2_h')
        self._ranked(neb_dir, [
            {'label': 's_9__s_1+s_2', 'fs_site1': 's_1', 'fs_site2': 's_2', 'converged': True},
        ])
        _write_h_site(ph2, 's_1', -100.1)   # s_2 deliberately absent
        triples = collect_entry_h_sources(neb_dir, ph2)
        assert [t[0] for t in triples] == ['s_1']


class TestBuildSurfaceSub1Sub2Map:

    def test_two_h_to_same_sub1_collapses_to_lower_energy(self, tmp_path):
        from models.neb_subsurface import build_surface_sub1_sub2_map
        graph = _fake_subsurface_graph()
        sub1_sub2 = {'ss_0': {'sub2_id': 'ss_1', 'sub2_env': 'Ni5Mo_oct',
                              'sub1_env': 'Ni6_oct'}}
        # s_a and s_b both map to sub1 ss_0; s_b is lower-energy IS
        entry_sources = [('s_a', '/p/h_atom_s_a.lammps', -100.0),
                         ('s_b', '/p/h_atom_s_b.lammps', -100.5)]
        surface_connections = [('ss_0', 's_a', 0.5), ('ss_0', 's_b', 0.6)]

        pm = build_surface_sub1_sub2_map(entry_sources, surface_connections,
                                         sub1_sub2, graph)
        assert len(pm) == 1                       # collapsed to one Hop A
        assert pm[0]['surface_sid'] == 's_b'      # lower-energy IS kept
        assert pm[0]['collapsed_from'] == ['s_a']
        assert pm[0]['sub1_id'] == 'ss_0'
        assert pm[0]['sub2_id'] == 'ss_1'

    def test_distinct_sub1_kept_separate(self, tmp_path):
        from models.neb_subsurface import build_surface_sub1_sub2_map
        graph = _fake_subsurface_graph()
        sub1_sub2 = {
            'ss_0': {'sub2_id': 'ss_1', 'sub2_env': 'Ni5Mo_oct', 'sub1_env': 'Ni6_oct'},
            'ss_2': {'sub2_id': None, 'sub1_env': 'Ni4Mo2_oct'},
        }
        entry_sources = [('s_a', '/p/a.lammps', -100.0), ('s_b', '/p/b.lammps', -100.5)]
        surface_connections = [('ss_0', 's_a', 0.5), ('ss_2', 's_b', 0.6)]
        pm = build_surface_sub1_sub2_map(entry_sources, surface_connections,
                                         sub1_sub2, graph)
        assert len(pm) == 2
        by_sid = {e['surface_sid']: e for e in pm}
        assert by_sid['s_b']['sub2_id'] is None   # sub1 with no sub2 carried through

    def test_records_site_type_and_via_connected(self, tmp_path):
        from models.neb_subsurface import build_surface_sub1_sub2_map
        graph = _fake_subsurface_graph()
        sub1_sub2 = {'ss_0': {'sub2_id': 'ss_1', 'sub2_env': 'Ni5Mo_oct',
                              'sub1_env': 'Ni6_oct', 'sub1_type': 'oct', 'sub2_type': 'oct'}}
        pm = build_surface_sub1_sub2_map(
            [('s_a', '/p/a.lammps', -100.0)], [('ss_0', 's_a', 0.5)], sub1_sub2, graph)
        assert pm[0]['via'] == 'connected'
        assert pm[0]['sub1_type'] == 'oct'          # from composition_label suffix
        assert pm[0]['far_mapping'] is False

    def test_nearest_fallback_when_not_connected(self, tmp_path):
        # Finding A: an entry H* with no surface connection is mapped to its
        # nearest sub1 (never dropped) when entry_mapping='nearest'.
        from models.neb_subsurface import build_surface_sub1_sub2_map
        graph = _fake_subsurface_graph()   # ss_0@(1,1), ss_2@(3,3) are sub1
        sub1_sub2 = {'ss_0': {'sub2_id': 'ss_1', 'sub2_env': 'Ni5Mo_oct', 'sub1_env': 'Ni6_oct'}}
        surf_data = {'sites': [{'site_id': 's_c', 'position': [1.2, 1.1, 14.0]}]}
        pm = build_surface_sub1_sub2_map(
            [('s_c', '/p/c.lammps', -100.0)], [], sub1_sub2, graph,
            entry_mapping='nearest', surface_sites_data=surf_data, cell=[10.0, 10.0, 10.0])
        assert len(pm) == 1
        assert pm[0]['sub1_id'] == 'ss_0'           # nearest sub1 by xy
        assert pm[0]['via'] == 'nearest'
        assert pm[0]['xy_offset_ang'] < 0.5

    def test_directly_below_still_drops_unconnected(self, tmp_path):
        from models.neb_subsurface import build_surface_sub1_sub2_map
        graph = _fake_subsurface_graph()
        surf_data = {'sites': [{'site_id': 's_c', 'position': [1.2, 1.1, 14.0]}]}
        pm = build_surface_sub1_sub2_map(
            [('s_c', '/p/c.lammps', -100.0)], [], {}, graph,
            entry_mapping='directly_below', surface_sites_data=surf_data, cell=[10.0, 10.0, 10.0])
        assert pm == []                              # dropped (old behaviour)

    def test_far_mapping_flagged(self, tmp_path):
        from models.neb_subsurface import build_surface_sub1_sub2_map
        graph = _fake_subsurface_graph()
        sub1_sub2 = {'ss_0': {'sub2_id': 'ss_1'}}
        # s_far@(6,6): nearest sub1 is ss_2@(3,3) at ~4.24 Å (xy) -> > 3.0 -> flagged
        surf_data = {'sites': [{'site_id': 's_far', 'position': [6.0, 6.0, 14.0]}]}
        pm = build_surface_sub1_sub2_map(
            [('s_far', '/p/f.lammps', -100.0)], [], sub1_sub2, graph,
            entry_mapping='nearest', surface_sites_data=surf_data, cell=[10.0, 10.0, 10.0],
            far_offset_ang=3.0)
        assert pm[0]['far_mapping'] is True
        assert pm[0]['xy_offset_ang'] > 3.0


class TestInterstitialEnvAndReclassify:

    def test_env_label_and_alias(self):
        from models.neb_subsurface import interstitial_env_label, oct_env_label
        assert interstitial_env_label({'composition_label': 'Al4_tet'}) == 'Al4_tet'
        assert oct_env_label is interstitial_env_label           # back-compat alias
        assert interstitial_env_label({}) == 'unknown_env'

    def test_site_type_of(self):
        from models.neb_subsurface import site_type_of
        assert site_type_of({'site_type': 'tet'}) == 'tet'       # explicit field
        assert site_type_of({'composition_label': 'Ni6_oct'}) == 'oct'   # from suffix
        assert site_type_of({'composition_label': 'Al4_tet'}) == 'tet'
        assert site_type_of({}) == 'unknown'

    def test_classify_relaxed_h_env_octahedral(self, tmp_path):
        # 6 Ni octahedrally around one H (all within the 2.2 Å cutoff) -> Ni6_oct
        from ase import Atoms
        from ase.io import write as ase_write
        from models.neb_subsurface import classify_relaxed_h_env
        c = 10.0
        atoms = Atoms(
            'Ni6H',
            positions=[[c/2 - 2, c/2, c/2], [c/2 + 2, c/2, c/2],
                       [c/2, c/2 - 2, c/2], [c/2, c/2 + 2, c/2],
                       [c/2, c/2, c/2 - 2], [c/2, c/2, c/2 + 2],
                       [c/2, c/2, c/2]],
            cell=[c, c, c], pbc=[True, True, False])
        p = str(tmp_path / 'relaxed.lammps')
        ase_write(p, atoms, format='lammps-data', masses=True, specorder=['Ni', 'H'])
        rc = classify_relaxed_h_env(p)
        assert rc['site_type'] == 'oct'
        assert rc['coord_count'] == 6
        assert rc['env'] == 'Ni6_oct'

    def test_classify_relaxed_h_env_missing_file(self, tmp_path):
        from models.neb_subsurface import classify_relaxed_h_env
        rc = classify_relaxed_h_env(str(tmp_path / 'nope.lammps'))
        assert rc['site_type'] == 'unknown' and rc['env'] == 'unknown_env'

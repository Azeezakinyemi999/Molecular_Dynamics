"""
tests/functional/test_ft_subsurface_neb.py
============================================
Category A + C functional tests — subsurface hop script generation.

Tests:
1. connect_to_surface() — pure geometric function, no files needed
2. orchestrate_hopa_neb(dry_run=True) — generates Hop A NEB scripts

No LAMMPS, no SLURM, no GPU required.

Fixture strategy for orchestrate_hopa_neb:
- 2×2 Ni FCC(111) slab (5 layers) built with ASE
- IS file: slab + 1 H on surface written via add_adsorbate (correct MASSES format)
- Synthetic networkx graph with 1 subsurface_1 site
- Synthetic surface_connections linking that site to the surface IS site
"""

import json
import os
import sys
import pathlib
import math

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import MagicMock

for _m in ('acat', 'acat.adsorption_sites'):
    sys.modules.setdefault(_m, MagicMock())
sys.modules.setdefault('models.surface_graph', MagicMock())
for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.patches',
           'matplotlib.cm', 'matplotlib.ticker', 'matplotlib.transforms',
           'matplotlib.colors', 'matplotlib.scale', 'matplotlib._path',
           'matplotlib._api', 'matplotlib.cbook', 'matplotlib.rcsetup'):
    sys.modules.setdefault(_m, MagicMock())

import networkx as nx
from ase.build import fcc111
from ase.io import write as ase_write

from models.config import E2T_7, MASSES_7, ELEM_STR_7
from models.structure import add_adsorbate
from models.subsurface_graph import connect_to_surface
from models.neb_subsurface import orchestrate_hopa_neb


# ─── shared slab fixture ─────────────────────────────────────────────────────

def _make_slab_and_is(tmp_path):
    """
    Build a 2×2 Ni FCC(111) slab (5 layers, 20 atoms) and an IS file
    (slab + 1 H on the top surface).  Returns:
        (slab, slab_path, is_path, surface_z, cell)
    """
    slab = fcc111('Ni', size=(2, 2, 5), vacuum=10.0)
    slab.wrap()

    slab_path = str(tmp_path / 'slab.lammps')
    ase_write(slab_path, slab, format='lammps-data', atom_style='atomic')

    pos = slab.get_positions()
    surface_z = float(pos[:, 2].max())
    cell = slab.get_cell().lengths()  # [Lx, Ly, Lz]

    surf_idx = [int(i) for i in range(len(slab))
                if abs(pos[i, 2] - surface_z) < 0.5]
    sx = float(pos[surf_idx[0], 0])
    sy = float(pos[surf_idx[0], 1])

    is_path = str(tmp_path / 'h_s0_relaxed.lammps')
    add_adsorbate(
        slab_path=slab_path,
        site_position=[sx, sy],
        species='H',
        masses=MASSES_7,
        e2t=E2T_7,
        out_path=is_path,
    )

    return slab, slab_path, is_path, surface_z, cell, surf_idx


# ═════════════════════════════════════════════════════════════════════════════
# 1. connect_to_surface — pure geometric tests
# ═════════════════════════════════════════════════════════════════════════════

class TestConnectToSurface:
    """
    connect_to_surface(subsurface_sites, surface_sites_data, cell, xy_tol)

    Only sites with layer_classification='subsurface_1' should appear in
    the connections list.  XY distances are computed with periodic BC.
    """

    def _make_inputs(self, cell):
        """
        3-element list cell [Lx, Ly, Lz].

        subsurface site ss_0 is at (0.5, 0.5) — subsurface_1, close to s_0.
        subsurface site ss_1 is at (0.5, 0.5) — subsurface_2, same XY, MUST be excluded.
        Surface site s_0 is at (0.5, 0.5) — directly above ss_0.
        Surface site s_1 is at (Lx/2, Ly/2) — far from ss_0 in centre of cell.
        """
        Lx, Ly, _ = cell
        subsurface_sites = [
            {'site_id': 'ss_0', 'layer_classification': 'subsurface_1',
             'position': [0.5, 0.5, 5.0]},
            {'site_id': 'ss_1', 'layer_classification': 'subsurface_2',
             'position': [0.5, 0.5, 3.0]},
        ]
        surface_sites_data = {
            'sites': [
                {'site_id': 's_0', 'level1': {
                    'position': [0.5, 0.5, 8.0],
                    'site_type': 'fcc', 'composition': {}, 'full_label': '', 'atom_indices': [],
                }, 'level2': {}, 'level3': []},
                {'site_id': 's_1', 'level1': {
                    'position': [Lx / 2, Ly / 2, 8.0],
                    'site_type': 'fcc', 'composition': {}, 'full_label': '', 'atom_indices': [],
                }, 'level2': {}, 'level3': []},
            ]
        }
        return subsurface_sites, surface_sites_data

    @pytest.fixture()
    def connections(self):
        cell = np.array([5.0, 5.0, 25.0])
        subsurface_sites, surface_sites_data = self._make_inputs(cell)
        return connect_to_surface(subsurface_sites, surface_sites_data, cell, xy_tol=1.5)

    def test_returns_list(self, connections):
        assert isinstance(connections, list)

    def test_only_subsurface_1_connected(self, connections):
        # ss_1 has layer_classification='subsurface_2' — must NOT appear
        ss_ids_in_result = {c[0] for c in connections}
        assert 'ss_1' not in ss_ids_in_result, (
            "connect_to_surface returned a subsurface_2 site — "
            "only subsurface_1 should be connected"
        )

    def test_subsurface_1_matched_to_closest_surface_site(self, connections):
        # ss_0 at (0.5, 0.5) — closest surface site is s_0 also at (0.5, 0.5)
        matched_surface_ids = {c[1] for c in connections if c[0] == 'ss_0'}
        # s_0 must be in the connections for ss_0
        assert 's_0' in matched_surface_ids, (
            f"ss_0 not matched to s_0; matched to: {matched_surface_ids}"
        )

    def test_xy_dist_is_float(self, connections):
        for c in connections:
            assert isinstance(c[2], float), (
                f"xy_dist {c[2]!r} is not a float — connections must be (str, str, float)"
            )

    def test_no_connection_for_site_beyond_tol(self):
        # Tight tol so only exact matches connect
        cell = np.array([10.0, 10.0, 30.0])
        subsurface_sites = [
            {'site_id': 'ss_far', 'layer_classification': 'subsurface_1',
             'position': [0.5, 0.5, 5.0]},
        ]
        surface_sites_data = {
            'sites': [
                {'site_id': 's_far', 'level1': {
                    'position': [4.0, 4.0, 8.0],   # >3.5 Å away in XY
                    'site_type': 'fcc', 'composition': {}, 'full_label': '', 'atom_indices': [],
                }, 'level2': {}, 'level3': []},
            ]
        }
        conns = connect_to_surface(subsurface_sites, surface_sites_data,
                                   cell, xy_tol=1.5)
        # (0.5,0.5) vs (4.0,4.0) → min periodic dist = min(3.5, 10-3.5)=3.5 > 1.5 → no match
        assert len(conns) == 0, (
            f"Expected no connections with xy_tol=1.5 but got: {conns}"
        )

    def test_periodic_distance_wraps_correctly(self):
        # Site at (0.1, 0.1) and surface site at (Lx-0.1, Ly-0.1).
        # Euclidean dist = sqrt((Lx-0.2)^2 + (Ly-0.2)^2) >> 0,
        # but periodic dist = 0.2*sqrt(2) ≈ 0.28 Å < 1.5 Å.
        cell = np.array([5.0, 5.0, 25.0])
        subsurface_sites = [
            {'site_id': 'ss_wrap', 'layer_classification': 'subsurface_1',
             'position': [0.1, 0.1, 5.0]},
        ]
        surface_sites_data = {
            'sites': [
                {'site_id': 's_wrap', 'level1': {
                    'position': [4.9, 4.9, 8.0],
                    'site_type': 'fcc', 'composition': {}, 'full_label': '', 'atom_indices': [],
                }, 'level2': {}, 'level3': []},
            ]
        }
        conns = connect_to_surface(subsurface_sites, surface_sites_data,
                                   cell, xy_tol=1.5)
        assert len(conns) == 1, (
            f"Periodic distance should match (0.1,0.1) to (4.9,4.9) in 5×5 cell, got: {conns}"
        )
        _, _, xy_dist = conns[0]
        assert math.isclose(xy_dist, 0.2 * 2**0.5, rel_tol=0.01), (
            f"Periodic xy_dist={xy_dist:.4f} Å, expected ~{0.2*2**0.5:.4f} Å"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. orchestrate_hopa_neb(dry_run=True) — script generation
# ═════════════════════════════════════════════════════════════════════════════

class TestOrchestrateHopaNebDryRun:
    """
    orchestrate_hopa_neb(dry_run=True) must generate per-job directories
    with the FS-min LAMMPS script, ASE NEB script, and SLURM scripts.
    Array scripts and job_index.txt are written to the hopa/ directory.
    """

    @pytest.fixture()
    def hopa_result(self, tmp_path):
        slab, slab_path, is_path, surface_z, cell, surf_idx = _make_slab_and_is(tmp_path)
        pos = slab.get_positions()
        sx = float(pos[surf_idx[0], 0])
        sy = float(pos[surf_idx[0], 1])

        sub1_z = surface_z - 2.1
        ss_id = 'ss_0'

        G = nx.Graph()
        G.add_node(ss_id, site_id=ss_id, layer_classification='subsurface_1',
                   position=[sx, sy, sub1_z])
        subsurface_sites = [
            {'site_id': ss_id, 'layer_classification': 'subsurface_1',
             'position': [sx, sy, sub1_z]},
        ]

        surface_connections = [(ss_id, 's_0', 0.0)]
        dedup_is_labels = [('s_0', is_path, -2270.0)]

        result = orchestrate_hopa_neb(
            dedup_is_labels=dedup_is_labels,
            subsurface_graph=(G, subsurface_sites),
            surface_connections=surface_connections,
            outdir=str(tmp_path),
            masses=MASSES_7,
            e2t=E2T_7,
            elem_str=ELEM_STR_7,
            dry_run=True,
        )
        return result, tmp_path / 'hopa'

    def test_status_is_generated(self, hopa_result):
        result, _ = hopa_result
        assert result['status'] == 'generated'

    def test_n_jobs_is_one(self, hopa_result):
        result, _ = hopa_result
        assert result['n_jobs'] == 1

    def test_fsmin_array_script_created(self, hopa_result):
        result, hopa_dir = hopa_result
        assert pathlib.Path(result['fsmin_array']).exists(), (
            "hopa_fsmin_array.sh was not created"
        )

    def test_neb_array_script_created(self, hopa_result):
        result, hopa_dir = hopa_result
        assert pathlib.Path(result['neb_array']).exists(), (
            "hopa_neb_array.sh was not created"
        )

    def test_job_index_created(self, hopa_result):
        _, hopa_dir = hopa_result
        assert (hopa_dir / 'job_index.txt').exists()

    def test_job_index_contains_sid(self, hopa_result):
        result, hopa_dir = hopa_result
        content = (hopa_dir / 'job_index.txt').read_text()
        assert 's_0' in content

    def test_jobs_json_created(self, hopa_result):
        result, hopa_dir = hopa_result
        assert pathlib.Path(result['jobs_json']).exists()

    def test_jobs_json_has_required_keys(self, hopa_result):
        result, _ = hopa_result
        required_keys = {
            'sid', 'is_path', 'e_is', 'ss1_id', 'sub1_xyz',
            'fs_raw', 'fs_relaxed', 'fsmin_script', 'neb_script',
            'fsmin_sh', 'neb_sh', 'barrier_file', 'path_file', 'job_dir',
        }
        job = result['jobs'][0]
        missing = required_keys - job.keys()
        assert not missing, f"hopa_jobs.json entry missing keys: {missing}"

    def test_per_job_directory_created(self, hopa_result):
        _, hopa_dir = hopa_result
        assert (hopa_dir / 's_0').is_dir()

    def test_fsmin_lammps_script_created(self, hopa_result):
        result, _ = hopa_result
        assert pathlib.Path(result['jobs'][0]['fsmin_script']).exists(), (
            "min_fs.lammps not created in job dir"
        )

    def test_fsmin_slurm_created_with_sharing_partition(self, hopa_result):
        result, _ = hopa_result
        sh_path = pathlib.Path(result['jobs'][0]['fsmin_sh'])
        assert sh_path.exists()
        content = sh_path.read_text()
        assert 'sharing' in content, (
            "Hop A SLURM FS-min script does not use 'sharing' partition"
        )

    def test_neb_run_script_created(self, hopa_result):
        result, _ = hopa_result
        assert pathlib.Path(result['jobs'][0]['neb_script']).exists(), (
            "run_hopa.py not created in job dir"
        )

    def test_neb_run_script_uses_extxyz(self, hopa_result):
        result, _ = hopa_result
        content = pathlib.Path(result['jobs'][0]['neb_script']).read_text()
        assert '.extxyz' in content, (
            "run_hopa.py does not use .extxyz for trajectory export"
        )
        assert '.lammpstrj' not in content, (
            "run_hopa.py still uses .lammpstrj (old broken format)"
        )

    def test_fs_raw_lammps_created(self, hopa_result):
        result, _ = hopa_result
        assert pathlib.Path(result['jobs'][0]['fs_raw']).exists(), (
            "sub1_fs_raw.lammps not created — build_hopa_fs may have failed"
        )

    def test_sub1_xyz_is_list_of_3_floats(self, hopa_result):
        result, _ = hopa_result
        xyz = result['jobs'][0]['sub1_xyz']
        assert len(xyz) == 3
        for v in xyz:
            assert isinstance(v, float), f"sub1_xyz element {v!r} is not float"

    def test_no_jobs_when_no_surface_connection(self, tmp_path):
        # If surface_connections is empty, the function should return n_jobs=0
        slab, slab_path, is_path, surface_z, cell, surf_idx = _make_slab_and_is(tmp_path)
        G = nx.Graph()
        subsurface_sites = []
        result = orchestrate_hopa_neb(
            dedup_is_labels=[('s_0', is_path, -2270.0)],
            subsurface_graph=(G, subsurface_sites),
            surface_connections=[],   # no connections → skip all IS sites
            outdir=str(tmp_path),
            masses=MASSES_7,
            e2t=E2T_7,
            elem_str=ELEM_STR_7,
            dry_run=True,
        )
        assert result['n_jobs'] == 0

"""
tests/test_surface_graph.py
============================
Unit tests for models/surface_graph.py — fully offline.

`build_surface_graph` requires ACAT + a real slab file and is skipped.
All other public functions are tested with a small synthetic NetworkX graph.

CRITICAL: acat must be mocked before importing surface_graph — it is
imported at the module level.

Covers:
  get_site_neighbors      — site-site BFS traversal by shell depth
  _get_atom_neighbors     — shell-1 / shell-2 atom neighbours via graph edges
  build_site_environment  — Level 1/2/3 environment dict from graph + mock slab
  save_surface_sites      — JSON serialisation of graph + environments
  visualize_surface_graph — matplotlib figure returned (Agg backend)
"""

import json
import pathlib
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock

# ── CRITICAL: ensure real models.surface_graph is imported ───────────────────
# test_neb_workflow.py (alphabetically earlier) sets
#   sys.modules['models.surface_graph'] = MagicMock()
# to prevent ImportError when neb_workflow.py imports surface_graph.
# We must remove that stale mock so Python re-imports the real module here.
if isinstance(sys.modules.get('models.surface_graph'), MagicMock):
    del sys.modules['models.surface_graph']

# Mock acat so that surface_graph.py's module-level imports succeed offline.
for _m in ('acat', 'acat.adsorption_sites', 'acat.settings'):
    sys.modules[_m] = MagicMock()

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')

import networkx as nx

from models.surface_graph import (
    get_site_neighbors,
    _get_atom_neighbors,
    build_site_environment,
    save_surface_sites,
    visualize_surface_graph,
    ELEMENT_COLORS,
    SITE_COLORS,
    SITE_MARKERS,
)


# ── synthetic graph fixture ───────────────────────────────────────────────────
#
# 4-atom top layer on a 10x10 cell:
#   a_0  Ni  (0.0, 0.0, 5.0)
#   a_1  Ni  (2.5, 0.0, 5.0)
#   a_2  Ni  (0.0, 2.5, 5.0)
#   a_3  Mo  (2.5, 2.5, 5.0)
#
# Atom-atom edges (dist 2.5 < cutoff 3.2):
#   a_0 — a_1, a_0 — a_2, a_1 — a_3, a_2 — a_3
#
# Site nodes:
#   s_0: hollow (centroid of a_0,a_1,a_2) at (0.83, 0.83, 5.2)
#   s_1: ontop  (above a_0)               at (0.00, 0.00, 5.5)
#
# Site-atom edges:
#   s_0 → a_0, a_1, a_2
#   s_1 → a_0
#
# Site-site edge:
#   s_0 ↔ s_1  (shared atom a_0)
#
# Full-slab positions (indices 0-3 are top layer; 4 is subsurface Mo):
# ─────────────────────────────────────────────────────────────────────────────

_POSITIONS = np.array([
    [0.0,  0.0,  5.0],   # 0  Ni
    [2.5,  0.0,  5.0],   # 1  Ni
    [0.0,  2.5,  5.0],   # 2  Ni
    [2.5,  2.5,  5.0],   # 3  Mo
    [0.83, 0.83, 2.5],   # 4  Mo  (subsurface)
])
_SYMS = np.array(['Ni', 'Ni', 'Ni', 'Mo', 'Mo'])
_CELL = np.array([10.0, 10.0, 10.0])


def _make_graph():
    G = nx.Graph()
    G.graph['cell']          = _CELL.tolist()
    G.graph['z_max']         = 5.0
    G.graph['top_layer_tol'] = 1.8
    G.graph['seed']          = 99

    atom_defs = [
        ('a_0', 'Ni', 0.0,  0.0,  5.0, 0),
        ('a_1', 'Ni', 2.5,  0.0,  5.0, 1),
        ('a_2', 'Ni', 0.0,  2.5,  5.0, 2),
        ('a_3', 'Mo', 2.5,  2.5,  5.0, 3),
    ]
    for nid, elem, x, y, z, full_idx in atom_defs:
        G.add_node(nid, node_type='atom', element=elem,
                   position=[x, y, z], x=x, y=y, z=z, layer=0,
                   color=ELEMENT_COLORS.get(elem, '#AAAAAA'),
                   size=124)

    # atom-atom edges (periodic distance < 3.2)
    edges_aa = [('a_0', 'a_1', 2.5), ('a_0', 'a_2', 2.5),
                ('a_1', 'a_3', 2.5), ('a_2', 'a_3', 2.5)]
    for u, v, d in edges_aa:
        G.add_edge(u, v, edge_type='atom-atom', distance=d)

    # site nodes
    G.add_node('s_0', node_type='site', site_type='hollow',
               composition='NiNiNi', label='NiNiNi_hollow_fcc',
               position=[0.83, 0.83, 5.2], x=0.83, y=0.83, z=5.2,
               atom_indices=(0, 1, 2), subsurf_element='',
               color=SITE_COLORS['hollow'], marker=SITE_MARKERS['hollow'])
    G.add_node('s_1', node_type='site', site_type='ontop',
               composition='Ni', label='Ni_atop',
               position=[0.0, 0.0, 5.5], x=0.0, y=0.0, z=5.5,
               atom_indices=(0,), subsurf_element='',
               color=SITE_COLORS['ontop'], marker=SITE_MARKERS['ontop'])

    # site-atom edges
    for ai in ('a_0', 'a_1', 'a_2'):
        G.add_edge('s_0', ai, edge_type='site-atom')
    G.add_edge('s_1', 'a_0', edge_type='site-atom')

    # site-site edge
    G.add_edge('s_0', 's_1', edge_type='site-site', shared_atoms=[0])

    return G


def _make_slab_mock():
    slab = MagicMock()
    slab.get_positions.return_value = _POSITIONS.copy()
    slab.get_chemical_symbols.return_value = list(_SYMS)
    slab.cell.diagonal.return_value = _CELL.copy()
    return slab


@pytest.fixture
def G():
    return _make_graph()


@pytest.fixture
def slab():
    return _make_slab_mock()


# ═══════════════════════════════════════════════════════════════════════════
# 1. get_site_neighbors
# ═══════════════════════════════════════════════════════════════════════════

class TestGetSiteNeighbors:

    def test_shell1_from_s0(self, G):
        nbrs = get_site_neighbors(G, 's_0', shell=1)
        assert 's_1' in nbrs
        assert 's_0' not in nbrs

    def test_shell1_from_s1(self, G):
        nbrs = get_site_neighbors(G, 's_1', shell=1)
        assert 's_0' in nbrs

    def test_shell2_superset_of_shell1(self, G):
        s1 = set(get_site_neighbors(G, 's_0', shell=1))
        s2 = set(get_site_neighbors(G, 's_0', shell=2))
        assert s1.issubset(s2)

    def test_node_not_in_graph_returns_empty(self, G):
        assert get_site_neighbors(G, 's_999', shell=1) == []

    def test_query_node_not_in_result(self, G):
        nbrs = get_site_neighbors(G, 's_0', shell=2)
        assert 's_0' not in nbrs


# ═══════════════════════════════════════════════════════════════════════════
# 2. _get_atom_neighbors
# ═══════════════════════════════════════════════════════════════════════════

class TestGetAtomNeighbors:

    def test_shell1_of_a0_contains_a1_and_a2(self, G):
        shell1, _ = _get_atom_neighbors(0, G, _POSITIONS, _SYMS, _CELL)
        shell1_indices = {n['index'] for n in shell1}
        assert 1 in shell1_indices
        assert 2 in shell1_indices

    def test_shell2_of_a0_contains_a3(self, G):
        _, shell2 = _get_atom_neighbors(0, G, _POSITIONS, _SYMS, _CELL)
        shell2_indices = {n['index'] for n in shell2}
        assert 3 in shell2_indices

    def test_excluded_indices_not_in_shell1(self, G):
        shell1, _ = _get_atom_neighbors(0, G, _POSITIONS, _SYMS, _CELL,
                                         exclude_indices={1})
        shell1_indices = {n['index'] for n in shell1}
        assert 1 not in shell1_indices

    def test_each_neighbor_has_required_keys(self, G):
        shell1, shell2 = _get_atom_neighbors(0, G, _POSITIONS, _SYMS, _CELL)
        for n in shell1 + shell2:
            for key in ('index', 'element', 'distance', 'shell'):
                assert key in n

    def test_node_not_in_graph_returns_empty_shells(self, G):
        shell1, shell2 = _get_atom_neighbors(4, G, _POSITIONS, _SYMS, _CELL)
        assert shell1 == []
        assert shell2 == []


# ═══════════════════════════════════════════════════════════════════════════
# 3. build_site_environment
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildSiteEnvironment:

    @pytest.fixture
    def envs(self, G, slab):
        return build_site_environment(G, slab)

    def test_returns_dict_keyed_by_site_node(self, envs):
        assert 's_0' in envs
        assert 's_1' in envs

    def test_level1_has_expected_keys(self, envs):
        l1 = envs['s_0']['level1']
        for key in ('site_id', 'site_type', 'composition', 'full_label',
                    'position', 'constituent_atoms'):
            assert key in l1

    def test_level1_constituent_atoms_matches_atom_indices(self, envs):
        l1 = envs['s_0']['level1']
        # s_0 has atom_indices=(0,1,2)
        indices = {ca['index'] for ca in l1['constituent_atoms']}
        assert indices == {0, 1, 2}

    def test_level2_keyed_by_atom_index(self, envs):
        l2 = envs['s_0']['level2']
        for ai in (0, 1, 2):
            assert str(ai) in l2

    def test_level2_has_shell_counts(self, envs):
        l2 = envs['s_0']['level2']
        for ai_str, info in l2.items():
            assert 'n_shell1' in info
            assert 'n_shell2' in info

    def test_level3_contains_neighboring_sites(self, envs):
        # s_0 and s_1 are connected by site-site edge
        l3 = envs['s_0']['level3']
        l3_ids = [x['site_id'] for x in l3]
        assert 's_1' in l3_ids

    def test_level3_shared_atoms_listed(self, envs):
        l3 = envs['s_0']['level3']
        nbr_s1 = next(x for x in l3 if x['site_id'] == 's_1')
        assert len(nbr_s1['shared_atoms']) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. save_surface_sites
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveSurfaceSites:

    @pytest.fixture
    def saved(self, tmp_path, G, slab):
        envs = build_site_environment(G, slab)
        save_path = str(tmp_path / 'results' / 'surface_sites.json')
        output = save_surface_sites(G, envs, slab, save_path, seed=99)
        data   = json.loads(pathlib.Path(save_path).read_text())
        return output, save_path, data

    def test_file_created(self, saved):
        _, save_path, _ = saved
        assert pathlib.Path(save_path).exists()

    def test_returns_dict(self, saved):
        output, _, _ = saved
        assert isinstance(output, dict)

    def test_metadata_block_present(self, saved):
        _, _, data = saved
        assert 'metadata' in data
        assert data['metadata']['seed'] == 99

    def test_sites_list_present(self, saved):
        _, _, data = saved
        assert 'sites' in data
        assert len(data['sites']) == 2

    def test_surface_atoms_present(self, saved):
        _, _, data = saved
        assert 'surface_atoms' in data

    def test_site_ids_in_output(self, saved):
        _, _, data = saved
        site_ids = {s['site_id'] for s in data['sites']}
        assert 's_0' in site_ids
        assert 's_1' in site_ids

    def test_site_type_counts_in_metadata(self, saved):
        _, _, data = saved
        counts = data['metadata']['site_type_counts']
        assert 'hollow' in counts
        assert 'ontop' in counts


# ═══════════════════════════════════════════════════════════════════════════
# 5. visualize_surface_graph
# ═══════════════════════════════════════════════════════════════════════════

class TestVisualizeSurfaceGraph:

    def test_returns_figure(self, tmp_path, G, slab):
        import matplotlib.pyplot as plt
        fig = visualize_surface_graph(
            G, slab,
            selected_site='s_0',
            save_path=str(tmp_path / 'graph.png'),
            seed=99,
        )
        assert fig is not None
        plt.close('all')

    def test_output_file_created(self, tmp_path, G, slab):
        import matplotlib.pyplot as plt
        out = str(tmp_path / 'graph.png')
        visualize_surface_graph(G, slab, selected_site='s_0',
                                 save_path=out, seed=99)
        assert pathlib.Path(out).exists()
        plt.close('all')

    def test_auto_selects_hollow_when_site_none(self, tmp_path, G, slab):
        """Passing selected_site=None should auto-select first hollow site."""
        import matplotlib.pyplot as plt
        fig = visualize_surface_graph(
            G, slab, selected_site=None,
            save_path=str(tmp_path / 'g2.png'), seed=99)
        assert fig is not None
        plt.close('all')

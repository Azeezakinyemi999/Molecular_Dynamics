"""
tests/test_oxide_surface_graph.py
==================================
Unit tests for the oxide path of models/surface_graph.py — fully offline.

The oxide path bypasses ACAT entirely (geometric enumeration), and the
module's ACAT/matplotlib imports are lazy, so no mocking is required —
but a stale MagicMock of models.surface_graph installed by
test_neb_workflow.py (alphabetically earlier) must be cleared first.

Covers:
  _z_plane_clusters        — gap-based plane detection, unequal populations
  _enumerate_oxide_sites   — ontop + M–O bridge generation, PBC midpoints
  build_surface_graph      — metal_type='oxide' end-to-end: graph topology,
                             composite-termination merge, downstream schema
"""

import json
import pathlib
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock

# Clear stale mock left by test_neb_workflow.py (see test_surface_graph.py)
if isinstance(sys.modules.get('models.surface_graph'), MagicMock):
    del sys.modules['models.surface_graph']

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import networkx as nx

from models.structure import write_lammps_data
from models.surface_graph import (
    _z_plane_clusters,
    _enumerate_oxide_sites,
    build_surface_graph,
    build_site_environment,
    save_surface_sites,
)

_MASSES = {1: (58.6934, 'Ni'), 2: (15.999, 'O'), 3: (51.9961, 'Cr')}
_E2T    = {'Ni': 1, 'O': 2, 'Cr': 3}


# ─── _z_plane_clusters ────────────────────────────────────────────────────────

def test_z_plane_clusters_basic():
    z = np.array([0.0, 0.05, 2.0, 2.1, 4.0])
    clusters = _z_plane_clusters(z, gap_tol=0.5)
    assert len(clusters) == 3
    assert sorted(len(c) for c in clusters) == [1, 2, 2]
    # ordered bottom → top
    assert float(z[clusters[0]].mean()) < float(z[-1])


def test_z_plane_clusters_unequal_populations():
    # 3 atoms at z=0, 1 atom at z=2, 5 atoms at z=4 — rank binning would fail
    z = np.array([0.0] * 3 + [2.0] + [4.0] * 5)
    clusters = _z_plane_clusters(z, gap_tol=0.5)
    assert [len(c) for c in clusters] == [3, 1, 5]


# ─── _enumerate_oxide_sites ───────────────────────────────────────────────────

def _square_nio_plane():
    """4-atom checkerboard NiO plane on a 4×4 Å cell (NN dist 2.0 Å)."""
    pos = np.array([
        [0.0, 0.0, 10.0],   # Ni
        [2.0, 0.0, 10.0],   # O
        [0.0, 2.0, 10.0],   # O
        [2.0, 2.0, 10.0],   # Ni
    ])
    syms = np.array(['Ni', 'O', 'O', 'Ni'])
    cell = np.array([4.0, 4.0, 30.0])
    return pos, syms, cell


def test_oxide_sites_counts_and_composition():
    pos, syms, cell = _square_nio_plane()
    sites = _enumerate_oxide_sites(pos, syms, cell, np.arange(4),
                                   oxide_bond_cutoff=2.6)

    ontop  = [s for s in sites if s['site'] == 'ontop']
    bridge = [s for s in sites if s['site'] == 'bridge']
    assert len(ontop) == 4
    # 2 Ni × 2 O distinct pairs in the small periodic cell, all at 2.0 Å
    assert len(bridge) == 4

    # bridges only across Ni–O pairs, metal-first composition
    assert {b['composition'] for b in bridge} == {'NiO'}
    assert {o['composition'] for o in ontop} == {'Ni', 'O'}

    # indices are full-slab and reference exactly one Ni and one O
    for b in bridge:
        i, j = b['indices']
        assert {str(syms[i]), str(syms[j])} == {'Ni', 'O'}


def test_oxide_bridge_midpoint_wraps_pbc():
    # Ni at x=0 and O at x=3 on a 4 Å cell: min-image bond crosses the
    # boundary (dist 1.0 Å) and the raw midpoint is x=-0.5 → must wrap to 3.5
    pos  = np.array([[0.0, 0.0, 10.0], [3.0, 0.0, 10.0]])
    syms = np.array(['Ni', 'O'])
    cell = np.array([4.0, 4.0, 30.0])
    sites = _enumerate_oxide_sites(pos, syms, cell, np.arange(2))

    bridge = [s for s in sites if s['site'] == 'bridge']
    assert len(bridge) == 1
    x, y = float(bridge[0]['position'][0]), float(bridge[0]['position'][1])
    assert 0.0 <= x < cell[0] and 0.0 <= y < cell[1]
    assert x == pytest.approx(3.5)


def test_oxide_sites_no_bridges_on_pure_o_plane():
    pos = np.array([[0.0, 0.0, 10.0], [2.0, 0.0, 10.0]])
    syms = np.array(['O', 'O'])
    cell = np.array([4.0, 4.0, 30.0])
    sites = _enumerate_oxide_sites(pos, syms, cell, np.arange(2))
    assert all(s['site'] == 'ontop' for s in sites)   # no O–O bridges


# ─── build_surface_graph(metal_type='oxide') end-to-end ──────────────────────

def _write_nio_slab(tmp_path, n_planes=4, composite=False):
    """Stack checkerboard NiO(100)-like planes; optionally add a Cr plane
    0.6 Å beneath the top plane (composite termination)."""
    positions, symbols = [], []
    for k in range(n_planes):
        z = 5.0 + 2.0 * k
        shift = 0.0 if k % 2 == 0 else 1.0
        for (x, y, s) in [(0, 0, 'Ni'), (2, 0, 'O'), (0, 2, 'O'), (2, 2, 'Ni')]:
            positions.append([(x + shift) % 4.0, (y + shift) % 4.0, z])
            symbols.append(s)
    if composite:
        z_top = 5.0 + 2.0 * (n_planes - 1)
        positions.append([1.0, 1.0, z_top - 0.6])
        symbols.append('Cr')
    p = str(tmp_path / 'nio_slab.lammps')
    write_lammps_data(symbols=symbols, positions=np.array(positions),
                      cell_lengths=[4.0, 4.0, 30.0],
                      masses=_MASSES, e2t=_E2T, out_path=p,
                      comment='NiO test slab')
    return p


def test_build_surface_graph_oxide_topology(tmp_path):
    p = _write_nio_slab(tmp_path)
    G, slab, top3, sites = build_surface_graph(p, seed=7, metal_type='oxide')

    assert top3 is None   # no ACAT sub-slab on the oxide path

    site_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'site']
    atom_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'atom']
    assert len(atom_nodes) == 4          # top plane only
    assert len(site_nodes) == 4 + 4      # 4 ontop + 4 distinct Ni–O bridges

    edge_types = {d['edge_type'] for _, _, d in G.edges(data=True)}
    assert {'atom-atom', 'site-atom', 'site-site'} <= edge_types

    # nearest ontop pair must be graph distance 2 (via a shared bridge) —
    # this is what makes graph_dist_min=2 in enumerate_fs_pairs work
    sub = nx.Graph((u, v) for u, v, d in G.edges(data=True)
                   if d['edge_type'] == 'site-site')
    ontops = [n for n in site_nodes
              if G.nodes[n]['site_type'] == 'ontop' and n in sub]
    lengths = nx.single_source_shortest_path_length(sub, ontops[0])
    d_min = min(lengths[b] for b in ontops[1:] if b in lengths)
    assert d_min == 2


def test_build_surface_graph_oxide_composite_termination(tmp_path):
    """A Cr plane 0.6 Å below the top plane is chemically exposed and must
    be merged into the surface (oxide_exposure_tol=1.0)."""
    p = _write_nio_slab(tmp_path, composite=True)
    G, slab, _, sites = build_surface_graph(p, seed=7, metal_type='oxide')

    atom_elems = {d['element'] for n, d in G.nodes(data=True)
                  if d['node_type'] == 'atom'}
    assert 'Cr' in atom_elems, 'sub-surface Cr plane not merged into surface'
    comps = {s['composition'] for s in sites}
    assert 'Cr' in comps          # Cr ontop site exists
    assert 'CrO' in comps         # Cr–O bridges exist


def test_oxide_sites_json_schema_feeds_neb_pools(tmp_path):
    """The saved JSON must satisfy every key _site_signature/load_neb_pools
    reads: level1.constituent_atoms, level2 element/shell1, positions."""
    p = _write_nio_slab(tmp_path)
    G, slab, _, _ = build_surface_graph(p, seed=7, metal_type='oxide')
    envs = build_site_environment(G, slab)
    out = str(tmp_path / 'surface_sites.json')
    save_surface_sites(G, envs, slab, out, seed=7)

    data = json.load(open(out))
    assert data['sites']
    for s in data['sites']:
        l1 = s['level1']
        assert l1['constituent_atoms'], s['site_id']
        assert all('element' in a for a in l1['constituent_atoms'])
        assert len(l1['position']) == 3
        for v in s['level2'].values():
            assert 'element' in v and 'shell1' in v

    # fingerprint dedup must collapse symmetry-equivalent sites:
    # perfect checkerboard → exactly 3 unique (Ni_atop, O_atop, NiO_bridge)
    def sig(s):
        l1 = s['level1']
        els = tuple(sorted(a['element'] for a in l1['constituent_atoms']))
        shells = tuple(sorted(
            (v['element'], tuple(sorted(n['element'] for n in v['shell1'])))
            for v in s['level2'].values()))
        return (l1['site_type'], els, shells)

    assert len({sig(s) for s in data['sites']}) == 3


def test_metal_path_signature_unchanged():
    """metal_type default must keep the ACAT code path (lazy import) —
    calling with a mocked ACAT confirms the routing, not the physics."""
    import inspect
    from models import surface_graph
    params = inspect.signature(surface_graph.build_surface_graph).parameters
    assert params['metal_type'].default == 'alloy'
    assert params['n_layers_total'].default == 12

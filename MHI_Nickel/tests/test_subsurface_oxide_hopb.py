"""
tests/test_subsurface_oxide_hopb.py
====================================
Unit tests for the oxide path of models/subsurface_graph.py and the
Hop B sub1–sub2 edge fix — fully offline (scipy Voronoi runs locally).

Covers:
  _identify_layers_by_gaps — gap-based layers, unequal plane populations
  classify_site            — keep_unclassified → 'interstitial' retention
  build_subsurface_graph   — metal_type='oxide' auto layer indices +
                             keep-all interstitials
  Hop B fix                — subsurface-subsurface edges exist for BOTH
                             metal and oxide paths; find_sub2_neighbor works
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from ase.build import fcc111

from models.structure import write_lammps_data
from models.subsurface_graph import (
    _identify_layers_by_gaps,
    classify_site,
    build_subsurface_graph,
)

_MASSES = {1: (58.6934, 'Ni'), 2: (15.999, 'O')}
_E2T    = {'Ni': 1, 'O': 2}


# ─── _identify_layers_by_gaps ─────────────────────────────────────────────────

def test_gap_layers_contract():
    """Same contract as _identify_layers: 1-indexed bottom→top + mean z."""
    positions = np.array([
        [0, 0, 0.0], [1, 0, 0.1],          # layer 1 (2 atoms)
        [0, 0, 2.0],                        # layer 2 (1 atom)
        [0, 0, 4.0], [1, 0, 4.1], [2, 0, 3.9],   # layer 3 (3 atoms)
    ], dtype=float)
    layer_map, layer_z = _identify_layers_by_gaps(positions, gap_tol=0.5)

    assert sorted(layer_z) == [1, 2, 3]
    assert layer_map[0] == 1 and layer_map[2] == 2 and layer_map[5] == 3
    assert layer_z[1] == pytest.approx(0.05)
    assert layer_z[3] == pytest.approx(4.0, abs=0.1)
    # every atom assigned
    assert sorted(layer_map) == list(range(6))


# ─── classify_site keep_unclassified ──────────────────────────────────────────

def _two_atom_env():
    """A site with only 2 coordinating atoms — outside all oct/tet rules."""
    site_pos  = np.array([0.0, 0.0, 0.0])
    atoms     = np.array([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]])
    elements  = np.array(['Ni', 'O'])
    cell      = np.array([20.0, 20.0, 20.0])
    return site_pos, atoms, elements, cell


def test_classify_site_default_discards_unknown():
    clf = classify_site(*_two_atom_env(), cutoff=2.0)
    assert clf['site_type'] == 'unknown'
    assert clf['composition_label'].startswith('unknown_n')


def test_classify_site_keep_unclassified_retains_interstitial():
    clf = classify_site(*_two_atom_env(), cutoff=2.0, keep_unclassified=True)
    assert clf['site_type'] == 'interstitial'
    assert 'interstitial' in clf['composition_label']
    assert 'unknown' not in clf['composition_label']


# ─── shared slab + sites fixtures ─────────────────────────────────────────────

def _write_slab_and_sites(tmp_path, atoms, name):
    """Write a slab plus a minimal surface_sites.json (ontop on top atoms)."""
    slab_path = str(tmp_path / f'{name}.lammps')
    write_lammps_data(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell_lengths=atoms.cell.diagonal(),
        masses=_MASSES, e2t=_E2T, out_path=slab_path, comment=name,
    )
    pos = atoms.get_positions()
    z_top = pos[:, 2].max()
    sites = [
        {'site_id': f's_{k}',
         'level1': {'site_type': 'ontop', 'full_label': 'test_atop',
                    'position': [float(p[0]), float(p[1]), float(p[2]) + 1.5]}}
        for k, p in enumerate(pos[pos[:, 2] > z_top - 0.3])
    ]
    sites_path = str(tmp_path / f'{name}_sites.json')
    json.dump({'sites': sites, 'surface_atoms': []}, open(sites_path, 'w'))
    return slab_path, sites_path


def _count_edges(G, edge_type):
    return sum(1 for _, _, d in G.edges(data=True)
               if d.get('edge_type') == edge_type)


# ─── metal path: Hop B edges (the dead-code fix) ─────────────────────────────

def test_metal_hopb_sub1_sub2_edges(tmp_path):
    slab = fcc111('Ni', size=(2, 2, 12), vacuum=10.0)
    slab_path, sites_path = _write_slab_and_sites(tmp_path, slab, 'ni12')

    G, sub_sites = build_subsurface_graph(slab_path, sites_path, seed=42)

    lc = {s['layer_classification'] for s in sub_sites}
    assert 'subsurface_1' in lc and 'subsurface_2' in lc

    n_s12 = _count_edges(G, 'subsurface-subsurface')
    assert n_s12 > 0, 'Hop B fix regressed: no sub1-sub2 edges on metal path'

    # every subsurface_1 site must reach a subsurface_2 neighbour
    from models.neb_subsurface import find_sub2_neighbor
    sub1 = [s for s in sub_sites
            if s['layer_classification'] == 'subsurface_1']
    assert sub1
    for s in sub1:
        find_sub2_neighbor(G, s['site_id'], sub_sites)   # raises if none

    # metal path must still discard unclassified sites
    assert all(s['site_type'] in ('oct', 'tet') for s in sub_sites)


def test_metal_hopb_auto_derivation_matches_historical_10_11(tmp_path):
    """N=12 must auto-derive subsurface_layers=(10, 11) — the historical
    hardcoded default — through the REAL build_subsurface_graph(), not a
    reimplementation of the formula."""
    slab = fcc111('Ni', size=(2, 2, 12), vacuum=10.0)
    slab_path, sites_path = _write_slab_and_sites(tmp_path, slab, 'ni12b')

    _, sub_sites = build_subsurface_graph(slab_path, sites_path, seed=42)

    layer_numbers = {s['layer_classification']: s['layer_number']
                     for s in sub_sites}
    assert layer_numbers.get('subsurface_1') == 11
    assert layer_numbers.get('subsurface_2') == 10


def test_metal_thin_slab_warns_and_degrades_without_crashing(tmp_path, capsys):
    """N=3: n_frozen=round(3/3)=1, subsurface_1=2 (valid), subsurface_2=1
    (collides with the frozen bottom layer). build_subsurface_graph must
    warn, drop subsurface_2 entirely, and keep working for subsurface_1 —
    not crash. This is the exact degenerate case the old smoke test's
    3-layer slab hit silently before the Bug 14 fix (this is what
    justified raising the smoke test's SLAB_LAYERS from 3 to 6)."""
    slab = fcc111('Ni', size=(2, 2, 3), vacuum=10.0)
    slab_path, sites_path = _write_slab_and_sites(tmp_path, slab, 'ni3')

    G, sub_sites = build_subsurface_graph(slab_path, sites_path, seed=42)
    captured = capsys.readouterr()

    assert 'WARNING' in captured.out
    assert 'subsurface_2' in captured.out

    lc = {s['layer_classification'] for s in sub_sites}
    assert 'subsurface_1' in lc
    assert 'subsurface_2' not in lc   # dropped, not silently mis-assigned

    # Hop B has no edges — degraded, not crashed
    assert _count_edges(G, 'subsurface-subsurface') == 0

    # find_sub2_neighbor must raise per-job (the existing, already-tested
    # graceful-skip behaviour in orchestrate_hopb_neb), not crash the graph
    from models.neb_subsurface import find_sub2_neighbor
    sub1 = [s for s in sub_sites if s['layer_classification'] == 'subsurface_1']
    assert sub1
    with pytest.raises(ValueError, match='No subsurface_2 neighbor'):
        find_sub2_neighbor(G, sub1[0]['site_id'], sub_sites)


# ─── oxide path ───────────────────────────────────────────────────────────────

def _nio_rocksalt_slab():
    from ase.build import bulk, surface
    nio = bulk('NiO', 'rocksalt', a=4.17, cubic=True)
    slab = surface(nio, (1, 0, 0), 6, vacuum=10.0).repeat((2, 2, 1))
    slab.wrap()
    return slab


def test_oxide_subsurface_graph(tmp_path):
    slab = _nio_rocksalt_slab()
    slab_path, sites_path = _write_slab_and_sites(tmp_path, slab, 'nio6')

    G, sub_sites = build_subsurface_graph(
        slab_path, sites_path, seed=42, metal_type='oxide')

    # gap detection found the real planes and auto-derived the layer roles
    lc = {}
    for s in sub_sites:
        lc[s['layer_classification']] = lc.get(s['layer_classification'], 0) + 1
    assert lc.get('subsurface_1', 0) > 0
    assert lc.get('subsurface_2', 0) > 0

    # sub1 sits between the top plane and sub2 (first/second plane below)
    z_planes = sorted({round(float(z), 2)
                       for z in slab.get_positions()[:, 2]})
    z_top = z_planes[-1]
    for s in sub_sites:
        if s['layer_classification'] == 'subsurface_1':
            assert s['position'][2] < z_top

    # keep-all: nothing was discarded as 'unknown'
    assert all(s['site_type'] != 'unknown' for s in sub_sites)

    # Hop B edges exist on the oxide path too
    assert _count_edges(G, 'subsurface-subsurface') > 0
    assert _count_edges(G, 'surface-subsurface') > 0


def test_oxide_interstitials_retained(tmp_path):
    """The FCC oct/tet coordination rules generally do not match oxide
    voids — the oxide path must keep them as 'interstitial' rather than
    silently dropping them (which would leave Hop A with no targets)."""
    slab = _nio_rocksalt_slab()
    slab_path, sites_path = _write_slab_and_sites(tmp_path, slab, 'nio6b')

    _, sub_sites = build_subsurface_graph(
        slab_path, sites_path, seed=42, metal_type='oxide')

    assert len(sub_sites) > 0
    allowed = {'oct', 'tet', 'interstitial'}
    assert {s['site_type'] for s in sub_sites} <= allowed

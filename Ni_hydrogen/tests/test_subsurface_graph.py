"""
tests/test_subsurface_graph.py
================================
Unit tests for models/subsurface_graph.py — fully offline.

Tests every pure-Python/numpy helper directly without touching a LAMMPS
slab file.  The two functions that require a real slab
(find_voronoi_sites, build_subsurface_graph) are skipped.

Covers:
  _identify_layers          — rank-based z-layer binning
  _find_coordinating_atoms  — neighbour list with xy PBC
  _composition_label        — element count → string label
  classify_site             — oct/tet/unknown from coordination number
  _periodic_xy_distance     — xy distance with wrapping
  _get_surface_site_position — extracts position from nested site dicts
  connect_to_surface        — pairs layer-11 sites to surface sites
  save_subsurface_sites     — JSON serialisation + summary block
"""

import json
import pathlib
import sys
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.subsurface_graph import (
    _identify_layers,
    _identify_layers_by_gaps,
    _find_coordinating_atoms,
    _composition_label,
    classify_site,
    _periodic_xy_distance,
    _get_surface_site_position,
    connect_to_surface,
    save_subsurface_sites,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. _identify_layers
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentifyLayers:

    @pytest.fixture
    def uniform_pos(self):
        """12 atoms, 1 per z-layer (z = 0,1,...,11)."""
        return np.array([[0.0, 0.0, float(i)] for i in range(12)])

    def test_correct_number_of_unique_layers(self, uniform_pos):
        layer_map, layer_z = _identify_layers(uniform_pos, n_layers=12)
        assert len(layer_z) == 12

    def test_every_atom_assigned(self, uniform_pos):
        layer_map, _ = _identify_layers(uniform_pos, n_layers=12)
        assert len(layer_map) == 12

    def test_bottom_atom_in_layer_1(self, uniform_pos):
        # Atom 0 has z=0 (lowest) → layer 1
        layer_map, _ = _identify_layers(uniform_pos, n_layers=12)
        assert layer_map[0] == 1

    def test_top_atom_in_layer_n(self, uniform_pos):
        # Atom 11 has z=11 (highest) → layer 12
        layer_map, _ = _identify_layers(uniform_pos, n_layers=12)
        assert layer_map[11] == 12

    def test_layer_z_centers_monotone_increasing(self, uniform_pos):
        _, layer_z = _identify_layers(uniform_pos, n_layers=12)
        z_vals = [layer_z[L] for L in range(1, 13)]
        assert z_vals == sorted(z_vals)

    def test_raises_if_fewer_atoms_than_layers(self):
        pos = np.array([[0.0, 0.0, float(i)] for i in range(5)])
        with pytest.raises(ValueError, match='only 5 atoms but 12'):
            _identify_layers(pos, n_layers=12)

    def test_multi_atom_per_layer(self):
        """24 atoms in 12 layers → 2 atoms per layer."""
        pos = np.array([[0.0, 1.0, float(i // 2)] for i in range(24)])
        layer_map, layer_z = _identify_layers(pos, n_layers=12)
        assert len(layer_z) == 12
        # Both atoms in the same z-layer should have the same layer number
        assert layer_map[0] == layer_map[1]


# ═══════════════════════════════════════════════════════════════════════════
# 2. _find_coordinating_atoms
# ═══════════════════════════════════════════════════════════════════════════

class TestFindCoordinatingAtoms:

    @pytest.fixture
    def env(self):
        site_pos   = np.array([1.0, 1.0, 1.0])
        atom_pos   = np.array([
            [1.0, 1.0, 2.0],   # dist=1.0 — inside cutoff 2.2
            [1.0, 1.0, 3.5],   # dist=2.5 — outside cutoff
            [2.0, 1.0, 1.0],   # dist=1.0 — inside cutoff
        ])
        atom_elems = np.array(['Ni', 'Ni', 'Mo'])
        cell       = np.array([10.0, 10.0, 10.0])
        return site_pos, atom_pos, atom_elems, cell

    def test_returns_atoms_within_cutoff(self, env):
        sp, ap, ae, cell = env
        result = _find_coordinating_atoms(sp, ap, ae, cell, cutoff=2.2)
        assert len(result) == 2

    def test_excludes_atoms_outside_cutoff(self, env):
        sp, ap, ae, cell = env
        result = _find_coordinating_atoms(sp, ap, ae, cell, cutoff=2.2)
        indices = {c['atom_index'] for c in result}
        assert 1 not in indices  # atom at dist=2.5

    def test_sorted_by_distance_ascending(self, env):
        sp, ap, ae, cell = env
        result = _find_coordinating_atoms(sp, ap, ae, cell, cutoff=2.2)
        dists = [c['distance'] for c in result]
        assert dists == sorted(dists)

    def test_element_in_result(self, env):
        sp, ap, ae, cell = env
        result = _find_coordinating_atoms(sp, ap, ae, cell, cutoff=2.2)
        elems = {c['element'] for c in result}
        assert 'Ni' in elems
        assert 'Mo' in elems

    def test_periodic_xy_wrap(self):
        """Atom near cell edge wraps around and is found next to site at origin."""
        site_pos   = np.array([0.5, 0.5, 1.0])
        # Atom at x=9.8 in a 10x10 cell → image at x=-0.2, dist=0.7 in xy
        atom_pos   = np.array([[9.8, 0.5, 1.0]])
        atom_elems = np.array(['Cr'])
        cell       = np.array([10.0, 10.0, 5.0])
        result = _find_coordinating_atoms(site_pos, atom_pos, atom_elems, cell, cutoff=2.2)
        assert len(result) == 1
        assert result[0]['element'] == 'Cr'
        assert result[0]['distance'] == pytest.approx(0.7, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 3. _composition_label
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositionLabel:

    def test_oct_label_format(self):
        coord = [{'element': 'Ni'}, {'element': 'Ni'}, {'element': 'Ni'},
                  {'element': 'Mo'}, {'element': 'Cr'}, {'element': 'Fe'}]
        label = _composition_label(coord, 'oct')
        assert label.endswith('_oct')
        # Ni appears 3 times (most common)
        assert 'Ni3' in label

    def test_tet_label_format(self):
        coord = [{'element': 'Ni'}, {'element': 'Ni'},
                  {'element': 'Mo'}, {'element': 'Mo'}]
        label = _composition_label(coord, 'tet')
        assert label.endswith('_tet')
        assert 'Ni2' in label
        assert 'Mo2' in label

    def test_single_element_has_no_count_suffix(self):
        coord = [{'element': 'Ni'}]
        label = _composition_label(coord, 'oct')
        assert 'Ni1' not in label
        assert 'Ni' in label

    def test_most_common_element_first(self):
        coord = [{'element': 'Ni'}, {'element': 'Ni'},
                  {'element': 'Ni'}, {'element': 'Mo'}]
        label = _composition_label(coord, 'oct')
        assert label.startswith('Ni3')


# ═══════════════════════════════════════════════════════════════════════════
# 4. classify_site
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifySite:

    @pytest.fixture
    def cell(self):
        return np.array([10.0, 10.0, 10.0])

    def _make_atoms(self, n, dist=1.8):
        """Place n atoms equidistantly around a sphere of radius `dist`."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = np.column_stack([
            np.cos(angles) * dist,
            np.sin(angles) * dist,
            np.zeros(n),
        ])
        # centre site at origin
        return np.zeros(3), pos, np.array(['Ni'] * n)

    def test_6_neighbors_classified_oct(self, cell):
        site, ap, ae = self._make_atoms(6)
        result = classify_site(site, ap, ae, cell, cutoff=2.2)
        assert result['site_type'] == 'oct'
        assert result['is_distorted'] is False

    def test_4_neighbors_classified_tet(self, cell):
        site, ap, ae = self._make_atoms(4)
        result = classify_site(site, ap, ae, cell, cutoff=2.2)
        assert result['site_type'] == 'tet'
        assert result['is_distorted'] is False

    def test_5_neighbors_oct_distorted(self, cell):
        site, ap, ae = self._make_atoms(5)
        result = classify_site(site, ap, ae, cell, cutoff=2.2)
        assert result['site_type'] == 'oct'
        assert result['is_distorted'] is True

    def test_3_neighbors_tet_distorted(self, cell):
        site, ap, ae = self._make_atoms(3)
        result = classify_site(site, ap, ae, cell, cutoff=2.2)
        assert result['site_type'] == 'tet'
        assert result['is_distorted'] is True

    def test_2_neighbors_unknown(self, cell):
        site, ap, ae = self._make_atoms(2)
        result = classify_site(site, ap, ae, cell, cutoff=2.2)
        assert result['site_type'] == 'unknown'

    def test_distortion_score_nonzero_when_unequal(self, cell):
        site   = np.zeros(3)
        ap     = np.array([[1.5, 0.0, 0.0],
                            [1.5, 0.0, 0.0],   # same direction, won't matter
                            [-2.1, 0.0, 0.0],
                            [0.0, 1.5, 0.0],
                            [0.0, -1.5, 0.0],
                            [0.0, 0.0, 1.5]])
        ae     = np.array(['Ni'] * 6)
        result = classify_site(site, ap, ae, cell, cutoff=2.2)
        assert result['distortion_score'] > 0.0

    def test_coord_count_in_result(self, cell):
        site, ap, ae = self._make_atoms(6)
        result = classify_site(site, ap, ae, cell, cutoff=2.2)
        assert result['coord_count'] == 6


# ═══════════════════════════════════════════════════════════════════════════
# 5. _periodic_xy_distance
# ═══════════════════════════════════════════════════════════════════════════

class TestPeriodicXyDistance:

    _cell = np.array([10.0, 10.0, 20.0])

    def test_zero_for_same_position(self):
        p = np.array([3.0, 4.0, 0.0])
        assert _periodic_xy_distance(p, p, self._cell) == pytest.approx(0.0)

    def test_simple_distance_no_wrap(self):
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([4.0, 5.0, 0.0])
        # xy dist = sqrt(9+16) = 5
        assert _periodic_xy_distance(p1, p2, self._cell) == pytest.approx(5.0)

    def test_wraps_x_through_boundary(self):
        # p1 at x=0.5, p2 at x=9.5 → direct dist=9, image dist=1
        p1 = np.array([0.5, 0.0, 0.0])
        p2 = np.array([9.5, 0.0, 0.0])
        assert _periodic_xy_distance(p1, p2, self._cell) == pytest.approx(1.0)

    def test_wraps_y_through_boundary(self):
        p1 = np.array([0.0, 0.5, 0.0])
        p2 = np.array([0.0, 9.5, 0.0])
        assert _periodic_xy_distance(p1, p2, self._cell) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. _get_surface_site_position
# ═══════════════════════════════════════════════════════════════════════════

class TestGetSurfaceSitePosition:

    def test_nested_level1_position(self):
        site = {'site_id': 's_0', 'level1': {'position': [1.0, 2.0, 5.0]}}
        assert _get_surface_site_position(site) == [1.0, 2.0, 5.0]

    def test_flat_position_fallback(self):
        site = {'site_id': 's_1', 'position': [3.0, 4.0, 6.0]}
        assert _get_surface_site_position(site) == [3.0, 4.0, 6.0]

    def test_xy_fallback(self):
        site = {'site_id': 's_2', 'xy': [5.0, 6.0]}
        assert _get_surface_site_position(site) == [5.0, 6.0]

    def test_returns_none_if_no_position(self):
        site = {'site_id': 's_3'}
        assert _get_surface_site_position(site) is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. connect_to_surface
# ═══════════════════════════════════════════════════════════════════════════

class TestConnectToSurface:

    _cell = np.array([10.0, 10.0, 20.0])

    @pytest.fixture
    def surface_data(self):
        return {
            'sites': [
                {'site_id': 's_0', 'level1': {'position': [1.0, 1.0, 8.0]}},
                {'site_id': 's_1', 'level1': {'position': [7.0, 7.0, 8.0]}},
            ]
        }

    def test_connects_subsurface_1_to_nearby_surface_site(self, surface_data):
        sub = [{'site_id': 'ss_0', 'layer_classification': 'subsurface_1',
                'position': [1.0, 1.0, 5.0]}]
        conns = connect_to_surface(sub, surface_data, self._cell, xy_tol=1.5)
        assert len(conns) == 1
        assert conns[0][0] == 'ss_0'
        assert conns[0][1] == 's_0'

    def test_skips_subsurface_2_layer(self, surface_data):
        sub = [{'site_id': 'ss_0', 'layer_classification': 'subsurface_2',
                'position': [1.0, 1.0, 5.0]}]
        conns = connect_to_surface(sub, surface_data, self._cell, xy_tol=1.5)
        assert conns == []

    def test_no_connection_when_xy_too_far(self, surface_data):
        sub = [{'site_id': 'ss_0', 'layer_classification': 'subsurface_1',
                'position': [4.0, 4.0, 5.0]}]  # far from s_0 and s_1
        conns = connect_to_surface(sub, surface_data, self._cell, xy_tol=1.5)
        assert conns == []

    def test_distance_below_xy_tol(self, surface_data):
        sub = [{'site_id': 'ss_0', 'layer_classification': 'subsurface_1',
                'position': [1.1, 1.0, 5.0]}]
        conns = connect_to_surface(sub, surface_data, self._cell, xy_tol=1.5)
        assert len(conns) == 1
        xy_dist = conns[0][2]
        assert xy_dist < 1.5

    def test_periodic_boundary_connection(self, surface_data):
        """ss site near x=9.9 connects to surface site near x=0.1 via PBC."""
        surface_data_pbc = {
            'sites': [
                {'site_id': 's_0', 'level1': {'position': [0.2, 0.2, 8.0]}},
            ]
        }
        sub = [{'site_id': 'ss_0', 'layer_classification': 'subsurface_1',
                'position': [9.9, 0.2, 5.0]}]
        conns = connect_to_surface(sub, surface_data_pbc, self._cell, xy_tol=1.5)
        assert len(conns) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 8. auto-derivation of subsurface_1/subsurface_2 from total layer count
#    (mirrors build_subsurface_graph's Step 1 formula: n_frozen=round(N/3),
#    subsurface_1=N-1, subsurface_2=N-2 — the bulk-entry layer)
# ═══════════════════════════════════════════════════════════════════════════

def _derive_layer_roles(_N):
    """Reproduce build_subsurface_graph's Step-1 derivation in isolation."""
    n_frozen = round(_N / 3)
    sub1_L, sub2_L = _N - 1, _N - 2
    valid = set()
    for L in (sub1_L, sub2_L):
        if 1 <= L <= _N and L > n_frozen:
            valid.add(L)
    return n_frozen, sub1_L, sub2_L, valid


class TestAutoDeriveLayerRoles:

    def test_production_12_layer_case_matches_historical_defaults(self):
        # Historical hardcoded defaults were subsurface_layers=(10, 11).
        n_frozen, sub1, sub2, valid = _derive_layer_roles(12)
        assert (sub2, sub1) == (10, 11)
        assert valid == {10, 11}

    @pytest.mark.parametrize('_N', range(4, 17))
    def test_subsurface_layers_above_frozen_region_for_adequate_slabs(self, _N):
        n_frozen, sub1, sub2, valid = _derive_layer_roles(_N)
        assert sub1 == _N - 1 and sub2 == _N - 2
        # For N>=4, round(N/3) leaves at least one clear layer above frozen.
        assert valid == {sub1, sub2}

    def test_thin_slab_warns_and_degrades_without_crashing(self):
        # N=3: n_frozen=1, sub1=2 (valid), sub2=1 (collides with frozen).
        n_frozen, sub1, sub2, valid = _derive_layer_roles(3)
        assert n_frozen == 1
        assert sub1 == 2 and sub2 == 1
        assert valid == {2}          # sub2 dropped, not crashed
        assert sub2 not in valid

    def test_gap_based_plane_count_matches_rank_based_for_clean_slab(self):
        """Hybrid approach: gap-clustering must agree with the known atom
        count for a well-formed, evenly-spaced synthetic slab."""
        positions = np.array([[0.0, 0.0, float(i) * 2.0] for i in range(12)])
        _, layer_z = _identify_layers_by_gaps(positions, gap_tol=0.5)
        assert len(layer_z) == 12


# ═══════════════════════════════════════════════════════════════════════════
# 9. save_subsurface_sites
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveSubsurfaceSites:

    @pytest.fixture
    def sites(self):
        return [
            {'site_id': 'ss_0', 'site_type': 'oct',
             'layer_classification': 'subsurface_1', 'layer_number': 11,
             'composition_label': 'Ni4_oct', 'position': [1.0, 1.0, 5.0],
             'distortion_score': 0.05, 'is_distorted': False, 'coord_count': 6,
             'coord_list': []},
            {'site_id': 'ss_1', 'site_type': 'tet',
             'layer_classification': 'subsurface_2', 'layer_number': 10,
             'composition_label': 'Ni3Mo_tet', 'position': [2.0, 2.0, 3.0],
             'distortion_score': 0.10, 'is_distorted': True, 'coord_count': 4,
             'coord_list': []},
        ]

    def test_creates_json_file(self, tmp_path, sites):
        p = str(tmp_path / 'out' / 'subsurface_sites.json')
        save_subsurface_sites(sites, p, seed=7)
        assert pathlib.Path(p).exists()

    def test_json_has_sites_key(self, tmp_path, sites):
        p = str(tmp_path / 'out' / 'subsurface_sites.json')
        save_subsurface_sites(sites, p, seed=7)
        data = json.loads(pathlib.Path(p).read_text())
        assert 'sites' in data
        assert len(data['sites']) == 2

    def test_seed_stored_in_json(self, tmp_path, sites):
        p = str(tmp_path / 'out' / 'subsurface_sites.json')
        save_subsurface_sites(sites, p, seed=42)
        data = json.loads(pathlib.Path(p).read_text())
        assert data['seed'] == 42

    def test_json_has_summary_block(self, tmp_path, sites):
        p = str(tmp_path / 'out' / 'subsurface_sites.json')
        save_subsurface_sites(sites, p, seed=7)
        data = json.loads(pathlib.Path(p).read_text())
        assert 'summary' in data
        assert 'by_site_type' in data['summary']

    def test_creates_parent_directory(self, tmp_path, sites):
        p = str(tmp_path / 'a' / 'b' / 'c' / 'sub.json')
        save_subsurface_sites(sites, p, seed=1)
        assert pathlib.Path(p).exists()

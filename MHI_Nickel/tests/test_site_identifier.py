"""
tests/test_site_identifier.py
==============================
Unit tests for models/site_identifier.py — fully offline.

site_identifier.py only imports numpy and ase.io, so all helpers are
testable with synthetic numpy arrays or small ASE Atoms objects built
directly in the test (no LAMMPS files needed).

Covers:
  _classify_site_type       — coordination-number → site-type mapping
  _check_hcp_fcc            — HCP vs FCC discrimination from subsurface xy
  _get_surface_atoms        — surface mask + z_max computation
  _get_subsurface_atoms     — subsurface mask
  _physisorbed_site         — physisorption site labelling
  identify_adsorption_site  — full pipeline with ASE Atoms objects
"""

import numpy as np
import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from ase import Atoms

from models.site_identifier import (
    _classify_site_type,
    _check_hcp_fcc,
    _get_surface_atoms,
    _get_subsurface_atoms,
    _physisorbed_site,
    identify_adsorption_site,
    ADSORBATE_REGISTRY,
)


# ── synthetic Atoms builders ──────────────────────────────────────────────────

def _atop_atoms():
    """1 Ni surface atom + 1 H directly above it — atop/chemisorbed."""
    atoms = Atoms('NiH', positions=[[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 2.0]])
    atoms.cell = [10.0, 10.0, 20.0]
    return atoms


def _bridge_atoms():
    """2 Ni + 1 H above the midpoint — bridge/chemisorbed."""
    atoms = Atoms('NiNiH', positions=[[0.0, 0.0, 0.0],
                                       [2.5, 0.0, 0.0],
                                       [1.25, 0.0, 2.0]])
    atoms.cell = [10.0, 10.0, 20.0]
    return atoms


def _hollow_atoms():
    """3 Ni in equilateral triangle + H above centroid — hollow/chemisorbed."""
    a = 2.5
    h = a * np.sqrt(3) / 2
    # centroid_y = h/3 = a*sqrt(3)/6
    cy = h / 3
    atoms = Atoms('NiNiNiH', positions=[
        [0.0,  0.0, 0.0],
        [a,    0.0, 0.0],
        [a/2,  h,   0.0],
        [a/2,  cy,  1.5],   # directly above centroid
    ])
    atoms.cell = [10.0, 10.0, 20.0]
    return atoms


def _physisorbed_atoms():
    """1 Ni + 1 H far above — physisorbed."""
    atoms = Atoms('NiH', positions=[[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 4.0]])   # z_above=4.0 > phys_z_thresh=2.5
    atoms.cell = [10.0, 10.0, 20.0]
    return atoms


# ═══════════════════════════════════════════════════════════════════════════
# 1. _classify_site_type
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifySiteType:

    def test_0_neighbors_unknown(self):
        assert _classify_site_type(0) == 'unknown'

    def test_1_neighbor_atop(self):
        assert _classify_site_type(1) == 'atop'

    def test_2_neighbors_bridge(self):
        assert _classify_site_type(2) == 'bridge'

    def test_3_neighbors_hollow(self):
        assert _classify_site_type(3) == 'hollow'

    def test_4_neighbors_hollow_4fold(self):
        assert _classify_site_type(4) == 'hollow_4fold'

    def test_5_neighbors_5fold_label(self):
        assert _classify_site_type(5) == '5fold'


# ═══════════════════════════════════════════════════════════════════════════
# 2. _check_hcp_fcc
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckHcpFcc:

    def _make_args(self, sub_xy_offset):
        """bind_pos at (1,1), sub atom at (1+ox, 1+oy)."""
        bind_pos  = np.array([1.0, 1.0, 2.0])
        pos       = np.array([[1.0 + sub_xy_offset[0],
                                1.0 + sub_xy_offset[1],
                                0.0]])
        syms      = np.array(['Ni'])
        sub_mask  = np.array([True])
        return bind_pos, pos, syms, sub_mask

    def test_returns_hcp_when_sub_atom_nearby(self):
        bp, pos, syms, sub = self._make_args([0.1, 0.0])
        assert _check_hcp_fcc(bp, pos, syms, sub, hcp_cutoff=1.5) == 'hcp'

    def test_returns_fcc_when_sub_atom_far(self):
        bp, pos, syms, sub = self._make_args([2.0, 0.0])
        assert _check_hcp_fcc(bp, pos, syms, sub, hcp_cutoff=1.5) == 'fcc'

    def test_returns_none_when_no_subsurface_atoms(self):
        bind_pos = np.array([0.0, 0.0, 1.0])
        pos      = np.zeros((4, 3))
        syms     = np.array(['Ni', 'Ni', 'Ni', 'Ni'])
        sub_mask = np.array([False, False, False, False])
        assert _check_hcp_fcc(bind_pos, pos, syms, sub_mask) is None

    def test_threshold_boundary(self):
        """Exactly at cutoff — dist == hcp_cutoff → fcc (strict <)."""
        bp, pos, syms, sub = self._make_args([1.5, 0.0])
        result = _check_hcp_fcc(bp, pos, syms, sub, hcp_cutoff=1.5)
        assert result == 'fcc'


# ═══════════════════════════════════════════════════════════════════════════
# 3. _get_surface_atoms
# ═══════════════════════════════════════════════════════════════════════════

class TestGetSurfaceAtoms:

    def test_identifies_top_atoms_as_surface(self):
        pos  = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, 5.0],   # surface
                          [0.0, 0.0, 5.1]])  # surface
        syms = np.array(['Ni', 'Ni', 'Ni', 'Ni'])
        mask, z_max = _get_surface_atoms(pos, syms, z_surf_tol=2.0)
        assert z_max == pytest.approx(5.1)
        assert mask[2] and mask[3]
        assert not mask[0]

    def test_excludes_adsorbate_element_from_z_max(self):
        pos  = np.array([[0.0, 0.0, 0.0],   # Ni — surface
                          [0.0, 0.0, 4.0]])  # H — adsorbate
        syms = np.array(['Ni', 'H'])
        mask, z_max = _get_surface_atoms(
            pos, syms, z_surf_tol=2.0, adsorbate_elements={'H'})
        # z_max must come from non-H atoms only
        assert z_max == pytest.approx(0.0)
        assert mask[0]
        assert not mask[1]

    def test_z_surf_tol_controls_thickness(self):
        pos  = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 3.0],
                          [0.0, 0.0, 5.0]])
        syms = np.array(['Ni', 'Ni', 'Ni'])
        mask_tight, _ = _get_surface_atoms(pos, syms, z_surf_tol=1.0)
        mask_wide,  _ = _get_surface_atoms(pos, syms, z_surf_tol=3.0)
        assert mask_tight.sum() < mask_wide.sum()

    def test_no_adsorbate_set_includes_all_top_atoms(self):
        pos  = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        syms = np.array(['Ni', 'Ni'])
        mask, z_max = _get_surface_atoms(pos, syms, z_surf_tol=2.0)
        assert mask[1]


# ═══════════════════════════════════════════════════════════════════════════
# 4. _get_subsurface_atoms
# ═══════════════════════════════════════════════════════════════════════════

class TestGetSubsurfaceAtoms:

    def test_returns_correct_subsurface_mask(self):
        # z_max=5, z_surf_tol=2, sub_depth=2.5 → range (0.5, 3.0)
        pos  = np.array([[0.0, 0.0, z] for z in [0.0, 1.0, 2.0, 3.5, 5.0]])
        syms = np.array(['Ni'] * 5)
        mask = _get_subsurface_atoms(pos, syms, z_max=5.0,
                                      z_surf_tol=2.0, sub_depth=2.5)
        # Atoms at z=1.0 and z=2.0 are in (0.5, 3.0)
        assert mask[1] and mask[2]
        assert not mask[0]   # z=0.0 < 0.5
        assert not mask[3]   # z=3.5 > 3.0
        assert not mask[4]   # z=5.0 > 3.0

    def test_empty_when_no_atoms_in_range(self):
        pos  = np.array([[0.0, 0.0, z] for z in [0.0, 5.0]])
        syms = np.array(['Ni', 'Ni'])
        # range: (5-2-2.5, 5-2) = (0.5, 3.0) → atom at z=0 excluded
        mask = _get_subsurface_atoms(pos, syms, z_max=5.0,
                                      z_surf_tol=2.0, sub_depth=2.5)
        assert not mask.any()

    def test_boundaries_exclusive(self):
        """Atoms exactly on the boundaries are NOT included (strict > / <)."""
        pos  = np.array([[0.0, 0.0, 0.5],   # exactly at z_lo=0.5
                          [0.0, 0.0, 3.0]])  # exactly at z_hi=3.0
        syms = np.array(['Ni', 'Ni'])
        mask = _get_subsurface_atoms(pos, syms, z_max=5.0,
                                      z_surf_tol=2.0, sub_depth=2.5)
        assert not mask.any()


# ═══════════════════════════════════════════════════════════════════════════
# 5. _physisorbed_site
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysisorbedSite:

    @pytest.fixture
    def surface(self):
        # Equilateral triangle with side a=5 Å — all inter-atom distances 5 Å.
        # This ensures the circumradius (~2.89 Å) and midpoint distance (2.5 Å)
        # both exceed the default phys_xy_tol=1.8 Å, avoiding spurious 'atop'.
        surf_pos  = np.array([[0.0,  0.0,  0.0],
                               [5.0,  0.0,  0.0],
                               [2.5,  4.33, 0.0]])
        surf_syms = np.array(['Ni', 'Mo', 'Ni'])
        surf_idx  = np.array([0, 1, 2])
        return surf_pos, surf_syms, surf_idx

    def test_atop_when_very_close_in_xy(self, surface):
        surf_pos, surf_syms, surf_idx = surface
        bind_pos = np.array([0.1, 0.1, 4.0])   # ~0.14 Å from atom 0 in xy
        site_type, comp, label, nbrs = _physisorbed_site(
            bind_pos, surf_pos, surf_syms, surf_idx,
            n_neighbors=3, phys_xy_tol=1.8)
        assert site_type == 'atop'
        assert 'Ni' in comp
        assert 'atop' in label

    def test_near_bridge_when_equidistant_to_two_atoms(self, surface):
        surf_pos, surf_syms, surf_idx = surface
        # Midpoint of atom 0 [0,0] and atom 1 [5,0] → d1=d2=2.5 > 1.8 (not atop)
        # d3 to atom 2 [2.5, 4.33] = 4.33, d3-d1=1.83 > 0.8 → near_bridge
        bind_pos = np.array([2.5, 0.0, 4.0])
        site_type, comp, label, nbrs = _physisorbed_site(
            bind_pos, surf_pos, surf_syms, surf_idx,
            n_neighbors=3, phys_xy_tol=1.8)
        assert site_type == 'near_bridge'

    def test_near_hollow_when_equidistant_to_three_atoms(self, surface):
        surf_pos, surf_syms, surf_idx = surface
        # Circumcenter of equilateral triangle: [2.5, 4.33/3] = [2.5, 1.44]
        # circumradius = 5/√3 ≈ 2.89 > 1.8 (not atop); d1=d2=d3 → near_hollow
        bind_pos = np.array([2.5, 1.443, 4.0])
        site_type, comp, label, nbrs = _physisorbed_site(
            bind_pos, surf_pos, surf_syms, surf_idx,
            n_neighbors=3, phys_xy_tol=1.8)
        assert site_type == 'near_hollow'

    def test_label_contains_physisorbed(self, surface):
        surf_pos, surf_syms, surf_idx = surface
        bind_pos = np.array([0.1, 0.1, 4.0])
        _, _, label, _ = _physisorbed_site(
            bind_pos, surf_pos, surf_syms, surf_idx)
        assert 'physisorbed' in label

    def test_returns_neighbors_list(self, surface):
        surf_pos, surf_syms, surf_idx = surface
        bind_pos = np.array([0.1, 0.1, 4.0])
        _, _, _, nbrs = _physisorbed_site(
            bind_pos, surf_pos, surf_syms, surf_idx, n_neighbors=3)
        assert isinstance(nbrs, list)
        assert len(nbrs) == 3
        # Each neighbor is (element, xy_dist, slab_idx)
        for elem, dist, idx in nbrs:
            assert isinstance(elem, str)
            assert isinstance(dist, float)


# ═══════════════════════════════════════════════════════════════════════════
# 6. identify_adsorption_site (full pipeline, ASE Atoms as input)
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentifyAdsorptionSite:

    def test_returns_list(self):
        result = identify_adsorption_site(_atop_atoms(), 'H')
        assert isinstance(result, list)
        assert len(result) == 1

    def test_atop_chemisorbed(self):
        result = identify_adsorption_site(_atop_atoms(), 'H')
        s = result[0]
        assert s['mode'] == 'chemisorbed'
        assert s['site_type'] == 'atop'
        assert 'Ni' in s['composition']

    def test_bridge_chemisorbed(self):
        result = identify_adsorption_site(_bridge_atoms(), 'H')
        s = result[0]
        assert s['mode'] == 'chemisorbed'
        assert s['site_type'] == 'bridge'
        assert s['n_bonded'] == 2

    def test_hollow_chemisorbed(self):
        result = identify_adsorption_site(_hollow_atoms(), 'H')
        s = result[0]
        assert s['mode'] == 'chemisorbed'
        assert s['site_type'] == 'hollow'
        assert s['n_bonded'] == 3

    def test_physisorbed_detected_in_auto_mode(self):
        result = identify_adsorption_site(
            _physisorbed_atoms(), 'H',
            mode='auto', phys_z_thresh=2.5)
        s = result[0]
        assert s['mode'] == 'physisorbed'

    def test_z_above_surface_field_populated(self):
        result = identify_adsorption_site(_atop_atoms(), 'H')
        s = result[0]
        assert 'z_above_surface' in s
        assert s['z_above_surface'] == pytest.approx(2.0, abs=0.05)

    def test_label_field_is_string(self):
        result = identify_adsorption_site(_atop_atoms(), 'H')
        assert isinstance(result[0]['label'], str)

    def test_invalid_adsorbate_raises_value_error(self):
        with pytest.raises((ValueError, RuntimeError)):
            identify_adsorption_site(_atop_atoms(), 'UNKNOWN')

    def test_adsorbate_registry_has_expected_entries(self):
        for key in ('H', 'H2', 'CO', 'OH'):
            assert key in ADSORBATE_REGISTRY

"""
tests/test_kmc.py
==================
Tests for models/kmc.py — offline, no LAMMPS required.

All tests run the KMC code on small synthetic grids.  Physics formulas are
checked against analytic values.  np.random.seed is set before BKL tests for
reproducibility.
"""

import math
import sys
import pathlib
import pytest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.kmc import (
    _M_H2_KG,
    _KB_J,
    gas_strike_rate,
    drain_rate,
    make_grid,
    grid_neighbors,
    element_pair,
    surface_coverage,
    subsurface_population,
    subsurface_concentration,
    build_event_list,
    _execute_event,
    kmc_step,
    run_kmc,
)


# ── shared test parameters ──────────────────────────────────────────────────

_P   = 1_000.0     # Pa
_T   = 600.0       # K
_D   = 1e-10       # m²/s
_A0  = 3.52e-10    # m  (Ni lattice parameter)

_RATES = {
    'k_diss':      {('Ni', 'Ni'): 0.5},
    'k_des':       {('Ni', 'Ni'): 1e6},
    'k_surf_diff': {('Ni', 'Ni'): 1e8},
    'k_entry':     {'Ni': 1e7},
    'k_exit':      {'Ni': 1e6},
}


def _all_ni_grid(nx=4, ny=4):
    grid = make_grid(nx, ny, composition={'Ni': 1.0}, seed=0)
    return grid


# ═══════════════════════════════════════════════════════════════════════════
# 1. gas_strike_rate
# ═══════════════════════════════════════════════════════════════════════════

class TestGasStrikeRate:

    def test_returns_positive_float(self):
        R = gas_strike_rate(_P, _T, _M_H2_KG, (_A0 / math.sqrt(2.0)) ** 2)
        assert R > 0.0

    def test_exact_formula(self):
        area = (_A0 / math.sqrt(2.0)) ** 2
        expected = _P * area / math.sqrt(2.0 * math.pi * _M_H2_KG * _KB_J * _T)
        assert gas_strike_rate(_P, _T, _M_H2_KG, area) == pytest.approx(expected, rel=1e-8)

    def test_linear_in_pressure(self):
        area = 1e-20
        R1 = gas_strike_rate(1000.0, _T, _M_H2_KG, area)
        R2 = gas_strike_rate(2000.0, _T, _M_H2_KG, area)
        assert R2 == pytest.approx(2.0 * R1, rel=1e-8)

    def test_linear_in_area(self):
        R1 = gas_strike_rate(_P, _T, _M_H2_KG, 1e-20)
        R2 = gas_strike_rate(_P, _T, _M_H2_KG, 2e-20)
        assert R2 == pytest.approx(2.0 * R1, rel=1e-8)

    def test_decreases_with_temperature(self):
        area = 1e-20
        R_lo = gas_strike_rate(_P, 300.0, _M_H2_KG, area)
        R_hi = gas_strike_rate(_P, 1200.0, _M_H2_KG, area)
        assert R_hi < R_lo

    def test_inverse_sqrt_temperature_dependence(self):
        area = 1e-20
        R1 = gas_strike_rate(_P, 400.0, _M_H2_KG, area)
        R2 = gas_strike_rate(_P, 1600.0, _M_H2_KG, area)
        # R ∝ 1/√T → R(T1)/R(T2) = √(T2/T1)
        assert R1 / R2 == pytest.approx(math.sqrt(1600.0 / 400.0), rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 2. drain_rate
# ═══════════════════════════════════════════════════════════════════════════

class TestDrainRate:

    def test_returns_positive_float(self):
        assert drain_rate(_D, _A0) > 0.0

    def test_exact_formula(self):
        dx = _A0 / math.sqrt(2.0)
        expected = _D / (dx * dx)
        assert drain_rate(_D, _A0) == pytest.approx(expected, rel=1e-8)

    def test_linear_in_diffusivity(self):
        k1 = drain_rate(1e-10, _A0)
        k2 = drain_rate(2e-10, _A0)
        assert k2 == pytest.approx(2.0 * k1, rel=1e-8)

    def test_inverse_square_in_a0(self):
        k1 = drain_rate(_D, 1e-10)
        k2 = drain_rate(_D, 2e-10)
        assert k1 / k2 == pytest.approx(4.0, rel=1e-8)

    def test_larger_a0_smaller_rate(self):
        assert drain_rate(_D, 4e-10) < drain_rate(_D, 2e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 3. make_grid
# ═══════════════════════════════════════════════════════════════════════════

class TestMakeGrid:

    def test_returns_dict_with_required_keys(self):
        g = make_grid(3, 3)
        for k in ('surface_elem', 'surface_occ', 'subsurface_occ', 'nx', 'ny'):
            assert k in g

    def test_surface_elem_shape(self):
        g = make_grid(5, 7)
        assert g['surface_elem'].shape == (5, 7)

    def test_surface_occ_all_zero_initially(self):
        g = make_grid(4, 4)
        assert g['surface_occ'].sum() == 0

    def test_subsurface_occ_all_zero_initially(self):
        g = make_grid(4, 4)
        assert g['subsurface_occ'].sum() == 0

    def test_nx_ny_stored_correctly(self):
        g = make_grid(6, 8)
        assert g['nx'] == 6 and g['ny'] == 8

    def test_elements_from_composition(self):
        g = make_grid(10, 10, composition={'Mo': 1.0}, seed=0)
        assert np.all(g['surface_elem'] == 'Mo')

    def test_default_composition_hastelloy_n(self):
        g = make_grid(20, 20, seed=0)
        elems = set(g['surface_elem'].flatten())
        assert elems <= {'Ni', 'Mo', 'Cr', 'Fe'}

    def test_reproducible_with_same_seed(self):
        g1 = make_grid(5, 5, seed=99)
        g2 = make_grid(5, 5, seed=99)
        assert np.all(g1['surface_elem'] == g2['surface_elem'])

    def test_different_seeds_give_different_grids(self):
        g1 = make_grid(10, 10, seed=1)
        g2 = make_grid(10, 10, seed=2)
        assert not np.all(g1['surface_elem'] == g2['surface_elem'])


# ═══════════════════════════════════════════════════════════════════════════
# 4. grid_neighbors
# ═══════════════════════════════════════════════════════════════════════════

class TestGridNeighbors:

    def test_returns_four_neighbors(self):
        assert len(grid_neighbors(2, 2, 5, 5)) == 4

    def test_interior_neighbors_correct(self):
        nb = grid_neighbors(2, 2, 5, 5)
        assert (1, 2) in nb
        assert (3, 2) in nb
        assert (2, 1) in nb
        assert (2, 3) in nb

    def test_x_boundary_wraps(self):
        # site (0, 2) in a 5x5 grid — left neighbor should be (4, 2)
        nb = grid_neighbors(0, 2, 5, 5)
        assert (4, 2) in nb

    def test_y_boundary_wraps(self):
        # site (2, 0) — top neighbor should be (2, 4)
        nb = grid_neighbors(2, 0, 5, 5)
        assert (2, 4) in nb

    def test_corner_site_wraps_both_axes(self):
        nb = grid_neighbors(0, 0, 5, 5)
        assert (4, 0) in nb
        assert (0, 4) in nb

    def test_all_neighbors_in_valid_range(self):
        nx, ny = 6, 8
        for i, j in [(0, 0), (5, 7), (3, 4)]:
            for ni, nj in grid_neighbors(i, j, nx, ny):
                assert 0 <= ni < nx
                assert 0 <= nj < ny


# ═══════════════════════════════════════════════════════════════════════════
# 5. element_pair
# ═══════════════════════════════════════════════════════════════════════════

class TestElementPair:

    def _grid_with_elems(self, e00, e01):
        g = _all_ni_grid(2, 2)
        g['surface_elem'][0, 0] = e00
        g['surface_elem'][0, 1] = e01
        return g

    def test_same_element_returns_pair(self):
        g = self._grid_with_elems('Ni', 'Ni')
        assert element_pair(g, 0, 0, 0, 1) == ('Ni', 'Ni')

    def test_alphabetic_ordering_ni_mo(self):
        g = self._grid_with_elems('Ni', 'Mo')
        pair = element_pair(g, 0, 0, 0, 1)
        assert pair == ('Mo', 'Ni')

    def test_alphabetic_ordering_mo_ni(self):
        g = self._grid_with_elems('Mo', 'Ni')
        pair = element_pair(g, 0, 0, 0, 1)
        assert pair == ('Mo', 'Ni')

    def test_symmetric_lookup(self):
        g = self._grid_with_elems('Fe', 'Cr')
        p1 = element_pair(g, 0, 0, 0, 1)
        p2 = element_pair(g, 0, 1, 0, 0)
        assert p1 == p2

    def test_returns_tuple(self):
        g = _all_ni_grid(2, 2)
        assert isinstance(element_pair(g, 0, 0, 0, 1), tuple)


# ═══════════════════════════════════════════════════════════════════════════
# 6. surface_coverage / subsurface_population / concentration
# ═══════════════════════════════════════════════════════════════════════════

class TestGridQueries:

    def test_coverage_zero_for_empty_grid(self):
        g = _all_ni_grid(4, 4)
        assert surface_coverage(g) == pytest.approx(0.0)

    def test_coverage_one_for_full_grid(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][:] = 1
        assert surface_coverage(g) == pytest.approx(1.0)

    def test_coverage_partial(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['surface_occ'][1, 1] = 1
        assert surface_coverage(g) == pytest.approx(2 / 16)

    def test_subsurface_population_empty(self):
        g = _all_ni_grid(4, 4)
        assert subsurface_population(g) == 0

    def test_subsurface_population_counted(self):
        g = _all_ni_grid(4, 4)
        g['subsurface_occ'][0, 0] = 1
        g['subsurface_occ'][2, 3] = 1
        assert subsurface_population(g) == 2

    def test_concentration_zero_for_empty_subsurface(self):
        g = _all_ni_grid(4, 4)
        assert subsurface_concentration(g, _A0) == pytest.approx(0.0)

    def test_concentration_positive_when_occupied(self):
        g = _all_ni_grid(4, 4)
        g['subsurface_occ'][0, 0] = 1
        assert subsurface_concentration(g, _A0) > 0.0

    def test_concentration_exact_formula(self):
        # C = N_sub / (nx*ny * a0^3/sqrt(2))
        g = _all_ni_grid(4, 4)
        g['subsurface_occ'][0, 0] = 1
        g['subsurface_occ'][1, 2] = 1
        g['subsurface_occ'][3, 3] = 1
        n_sub = 3
        oct_vol_per_site = (_A0 ** 3) / math.sqrt(2.0)
        expected = n_sub / (4 * 4 * oct_vol_per_site)
        assert subsurface_concentration(g, _A0) == pytest.approx(expected, rel=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 7. build_event_list
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildEventList:

    def _events_of_kind(self, events, kind):
        return [e for e in events if e['kind'] == kind]

    def test_empty_grid_no_enter_exit_drain(self):
        g = _all_ni_grid(4, 4)
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert len(self._events_of_kind(events, 'enter')) == 0
        assert len(self._events_of_kind(events, 'exit')) == 0
        assert len(self._events_of_kind(events, 'drain')) == 0

    def test_empty_surface_generates_adsorb_events(self):
        g = _all_ni_grid(4, 4)
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert len(self._events_of_kind(events, 'adsorb')) > 0

    def test_occupied_surface_site_generates_enter_event(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        enter_evts = self._events_of_kind(events, 'enter')
        assert any((0, 0) in e['sites'] for e in enter_evts)

    def test_occupied_subsurface_generates_drain_event(self):
        g = _all_ni_grid(4, 4)
        g['subsurface_occ'][2, 2] = 1
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        drain_evts = self._events_of_kind(events, 'drain')
        assert any((2, 2) in e['sites'] for e in drain_evts)

    def test_occupied_subsurface_empty_surface_generates_exit_event(self):
        g = _all_ni_grid(4, 4)
        g['subsurface_occ'][1, 1] = 1
        # surface (1,1) is empty, so exit event should be generated
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        exit_evts = self._events_of_kind(events, 'exit')
        assert any((1, 1) in e['sites'] for e in exit_evts)

    def test_adjacent_occupied_pair_generates_desorb_event(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['surface_occ'][0, 1] = 1
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        desorb_evts = self._events_of_kind(events, 'desorb')
        assert len(desorb_evts) > 0

    def test_occupied_site_adjacent_empty_generates_surf_diff(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][2, 2] = 1
        # (2,3) and other neighbours are empty → diff events
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        diff_evts = self._events_of_kind(events, 'surf_diff')
        assert len(diff_evts) > 0

    def test_no_adsorb_when_k_diss_absent(self):
        g = _all_ni_grid(4, 4)
        rates_no_diss = {k: v for k, v in _RATES.items() if k != 'k_diss'}
        events = build_event_list(g, rates_no_diss, _P, _T, _D, _A0)
        assert len(self._events_of_kind(events, 'adsorb')) == 0

    def test_all_event_rates_positive(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['subsurface_occ'][1, 1] = 1
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert all(e['rate'] > 0.0 for e in events)

    def test_enter_not_generated_if_subsurface_occupied(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['subsurface_occ'][0, 0] = 1  # both occupied — no enter possible
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        enter_at_00 = [e for e in events if e['kind'] == 'enter' and (0, 0) in e['sites']]
        assert len(enter_at_00) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. _execute_event
# ═══════════════════════════════════════════════════════════════════════════

class TestExecuteEvent:

    def test_adsorb_occupies_both_sites(self):
        g = _all_ni_grid(4, 4)
        _execute_event(g, {'kind': 'adsorb', 'sites': [(0, 0), (0, 1)], 'rate': 1.0})
        assert g['surface_occ'][0, 0] == 1
        assert g['surface_occ'][0, 1] == 1

    def test_desorb_clears_both_sites(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][1, 1] = 1
        g['surface_occ'][1, 2] = 1
        _execute_event(g, {'kind': 'desorb', 'sites': [(1, 1), (1, 2)], 'rate': 1.0})
        assert g['surface_occ'][1, 1] == 0
        assert g['surface_occ'][1, 2] == 0

    def test_surf_diff_moves_h_to_empty_site(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][2, 2] = 1
        _execute_event(g, {'kind': 'surf_diff', 'sites': [(2, 2), (2, 3)], 'rate': 1.0})
        assert g['surface_occ'][2, 2] == 0
        assert g['surface_occ'][2, 3] == 1

    def test_enter_moves_h_to_subsurface(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        _execute_event(g, {'kind': 'enter', 'sites': [(0, 0)], 'rate': 1.0})
        assert g['surface_occ'][0, 0] == 0
        assert g['subsurface_occ'][0, 0] == 1

    def test_exit_moves_h_back_to_surface(self):
        g = _all_ni_grid(4, 4)
        g['subsurface_occ'][3, 3] = 1
        _execute_event(g, {'kind': 'exit', 'sites': [(3, 3)], 'rate': 1.0})
        assert g['subsurface_occ'][3, 3] == 0
        assert g['surface_occ'][3, 3] == 1

    def test_drain_clears_subsurface_site(self):
        g = _all_ni_grid(4, 4)
        g['subsurface_occ'][1, 3] = 1
        _execute_event(g, {'kind': 'drain', 'sites': [(1, 3)], 'rate': 1.0})
        assert g['subsurface_occ'][1, 3] == 0

    def test_adsorb_does_not_touch_subsurface(self):
        g = _all_ni_grid(4, 4)
        _execute_event(g, {'kind': 'adsorb', 'sites': [(0, 0), (0, 1)], 'rate': 1.0})
        assert g['subsurface_occ'].sum() == 0


# ═══════════════════════════════════════════════════════════════════════════
# 9. kmc_step
# ═══════════════════════════════════════════════════════════════════════════

class TestKmcStep:

    def test_empty_event_list_returns_zero(self):
        g = _all_ni_grid(4, 4)
        dt = kmc_step(g, [])
        assert dt == 0.0

    def test_returns_positive_dt_when_events_present(self):
        np.random.seed(42)
        g = _all_ni_grid(4, 4)
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        dt = kmc_step(g, events)
        assert dt > 0.0

    def test_grid_mutated_after_step(self):
        np.random.seed(42)
        g = _all_ni_grid(4, 4)
        events = build_event_list(g, _RATES, _P, _T, _D, _A0)
        occ_before = g['surface_occ'].copy()
        kmc_step(g, events)
        # adsorb events should have changed occupancy
        assert not np.all(g['surface_occ'] == occ_before)

    def test_dt_scales_inversely_with_total_rate(self):
        # event list with rate 1.0 vs rate 2.0 → dt halved
        np.random.seed(10)
        g1 = _all_ni_grid(2, 2)
        g2 = _all_ni_grid(2, 2)
        e1 = [{'kind': 'drain', 'sites': [(0, 0)], 'rate': 1.0}]
        e2 = [{'kind': 'drain', 'sites': [(0, 0)], 'rate': 2.0}]
        g1['subsurface_occ'][0, 0] = 1
        g2['subsurface_occ'][0, 0] = 1
        # seed ensures same u1 draw → dt ∝ 1/Q
        np.random.seed(99)
        dt1 = kmc_step(g1, e1)
        np.random.seed(99)
        dt2 = kmc_step(g2, e2)
        assert dt1 == pytest.approx(2.0 * dt2, rel=1e-8)

    def test_zero_total_rate_returns_zero(self):
        g = _all_ni_grid(4, 4)
        events = [{'kind': 'drain', 'sites': [(0, 0)], 'rate': 0.0}]
        dt = kmc_step(g, events)
        assert dt == 0.0

    def test_reproducible_with_seed(self):
        g1 = _all_ni_grid(4, 4)
        g2 = _all_ni_grid(4, 4)
        events1 = build_event_list(g1, _RATES, _P, _T, _D, _A0)
        events2 = build_event_list(g2, _RATES, _P, _T, _D, _A0)
        np.random.seed(7)
        dt1 = kmc_step(g1, events1)
        np.random.seed(7)
        dt2 = kmc_step(g2, events2)
        assert dt1 == pytest.approx(dt2, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 10. run_kmc
# ═══════════════════════════════════════════════════════════════════════════

class TestRunKmc:

    def test_returns_dict_with_required_keys(self):
        np.random.seed(0)
        g = _all_ni_grid(4, 4)
        result = run_kmc(g, _RATES, _P, _T, _D, _A0, n_steps=5)
        for k in ('t_arr', 'theta_arr', 'n_sub_arr'):
            assert k in result

    def test_array_length_is_n_steps_plus_one(self):
        np.random.seed(0)
        g = _all_ni_grid(4, 4)
        result = run_kmc(g, _RATES, _P, _T, _D, _A0, n_steps=10)
        assert len(result['t_arr']) == 11
        assert len(result['theta_arr']) == 11

    def test_t_arr_starts_at_zero(self):
        np.random.seed(0)
        g = _all_ni_grid(4, 4)
        result = run_kmc(g, _RATES, _P, _T, _D, _A0, n_steps=5)
        assert result['t_arr'][0] == pytest.approx(0.0)

    def test_t_arr_monotonically_nondecreasing(self):
        np.random.seed(0)
        g = _all_ni_grid(4, 4)
        result = run_kmc(g, _RATES, _P, _T, _D, _A0, n_steps=20)
        assert np.all(np.diff(result['t_arr']) >= 0.0)

    def test_initial_theta_zero_for_empty_grid(self):
        np.random.seed(0)
        g = _all_ni_grid(4, 4)
        result = run_kmc(g, _RATES, _P, _T, _D, _A0, n_steps=5)
        assert result['theta_arr'][0] == pytest.approx(0.0)

    def test_zero_steps_returns_length_one_arrays(self):
        np.random.seed(0)
        g = _all_ni_grid(4, 4)
        result = run_kmc(g, _RATES, _P, _T, _D, _A0, n_steps=0)
        assert len(result['t_arr']) == 1
        assert len(result['theta_arr']) == 1
        assert result['t_arr'][0] == pytest.approx(0.0)

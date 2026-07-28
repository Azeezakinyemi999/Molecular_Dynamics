"""
tests/test_kmc.py
==================
Tests for models/kmc.py — offline, no LAMMPS required.

Three-layer engine (surface / sub1 / sub2). All tests run the KMC code on
small synthetic grids.  Physics formulas are checked against analytic values.
np.random.seed is set before BKL tests for reproducibility.
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
    _N_A,
    _rate_lookup,
    _mean_of,
    gas_strike_rate,
    drain_rate,
    make_grid,
    grid_neighbors,
    element_pair,
    surface_coverage,
    sub1_population,
    sub2_population,
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

# Inter-layer dicts keyed by env; with the degenerate make_grid default the
# sub1/sub2 env of an all-Ni grid is 'Ni', so 'Ni'-keyed rates resolve.
_RATES = {
    'k_diss':       {('Ni', 'Ni'): 0.5},
    'k_des':        {('Ni', 'Ni'): 1e6},
    'k_surf_diff':  {('Ni', 'Ni'): 1e8},
    'k_entry':      {'Ni': 1e7},
    'k_exit':       {'Ni': 1e6},
    'k_hopB_entry': {'Ni': 5e6},
    'k_hopB_exit':  {'Ni': 5e5},
}


def _all_ni_grid(nx=4, ny=4):
    return make_grid(nx, ny, composition={'Ni': 1.0}, seed=0)


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
        assert drain_rate(2e-10, _A0) == pytest.approx(2.0 * drain_rate(1e-10, _A0), rel=1e-8)

    def test_inverse_square_in_a0(self):
        assert drain_rate(_D, 1e-10) / drain_rate(_D, 2e-10) == pytest.approx(4.0, rel=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 3. _rate_lookup / _mean_of  (per-env lookup with fallback)
# ═══════════════════════════════════════════════════════════════════════════

class TestRateLookup:

    def test_hit_returns_value(self):
        assert _rate_lookup({'Ni6_oct': 3.0, 'Ni5Mo_oct': 5.0}, 'Ni6_oct', 99.0) == 3.0

    def test_miss_returns_fallback_mean_not_zero(self):
        g = {'Ni6_oct': 2.0, 'Ni5Mo_oct': 4.0}
        # unknown env → fallback mean (3.0), never silent 0.0
        assert _rate_lookup(g, 'Ni4Mo2_oct', _mean_of(g)) == pytest.approx(3.0)

    def test_empty_group_returns_zero(self):
        assert _rate_lookup({}, 'Ni6_oct', 0.0) == 0.0

    def test_mean_of(self):
        assert _mean_of({'a': 2.0, 'b': 4.0}) == pytest.approx(3.0)
        assert _mean_of({}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. make_grid
# ═══════════════════════════════════════════════════════════════════════════

class TestMakeGrid:

    def test_returns_dict_with_required_keys(self):
        g = make_grid(3, 3)
        for k in ('surface_elem', 'surface_occ', 'sub1_occ', 'sub2_occ',
                  'sub1_env', 'sub2_env', 'nx', 'ny'):
            assert k in g

    def test_layer_shapes(self):
        g = make_grid(5, 7)
        for k in ('surface_elem', 'surface_occ', 'sub1_occ', 'sub2_occ',
                  'sub1_env', 'sub2_env'):
            assert g[k].shape == (5, 7)

    def test_all_occ_zero_initially(self):
        g = make_grid(4, 4)
        assert g['surface_occ'].sum() == 0
        assert g['sub1_occ'].sum() == 0
        assert g['sub2_occ'].sum() == 0

    def test_env_defaults_to_surface_element(self):
        g = make_grid(6, 6, composition={'Mo': 1.0}, seed=0)
        assert np.all(g['surface_elem'] == 'Mo')
        assert np.all(g['sub1_env'] == 'Mo')     # degenerate: env == element
        assert np.all(g['sub2_env'] == 'Mo')

    def test_env_composition_draws_labels(self):
        g = make_grid(20, 20, composition={'Ni': 1.0}, seed=0,
                      sub1_env_composition={'Ni6_oct': 1.0})
        assert np.all(g['sub1_env'] == 'Ni6_oct')

    def test_env_labels_not_truncated(self):
        # object dtype must preserve long labels verbatim (no fixed-width cut)
        g = make_grid(4, 4, composition={'Ni': 1.0}, seed=0,
                      sub1_env_composition={'Ni3MoCrFe_oct': 1.0})
        assert g['sub1_env'][0, 0] == 'Ni3MoCrFe_oct'

    def test_default_composition_hastelloy_n(self):
        g = make_grid(20, 20, seed=0)
        assert set(g['surface_elem'].flatten()) <= {'Ni', 'Mo', 'Cr', 'Fe'}

    def test_reproducible_with_same_seed(self):
        g1 = make_grid(5, 5, seed=99)
        g2 = make_grid(5, 5, seed=99)
        assert np.all(g1['surface_elem'] == g2['surface_elem'])


# ═══════════════════════════════════════════════════════════════════════════
# 5. grid_neighbors / element_pair
# ═══════════════════════════════════════════════════════════════════════════

class TestGridNeighbors:

    def test_returns_four_neighbors(self):
        assert len(grid_neighbors(2, 2, 5, 5)) == 4

    def test_interior_neighbors_correct(self):
        nb = grid_neighbors(2, 2, 5, 5)
        assert (1, 2) in nb and (3, 2) in nb and (2, 1) in nb and (2, 3) in nb

    def test_boundary_wraps(self):
        assert (4, 2) in grid_neighbors(0, 2, 5, 5)
        assert (2, 4) in grid_neighbors(2, 0, 5, 5)


class TestElementPair:

    def _grid_with_elems(self, e00, e01):
        g = _all_ni_grid(2, 2)
        g['surface_elem'][0, 0] = e00
        g['surface_elem'][0, 1] = e01
        return g

    def test_alphabetic_ordering(self):
        g = self._grid_with_elems('Ni', 'Mo')
        assert element_pair(g, 0, 0, 0, 1) == ('Mo', 'Ni')

    def test_symmetric_lookup(self):
        g = self._grid_with_elems('Fe', 'Cr')
        assert element_pair(g, 0, 0, 0, 1) == element_pair(g, 0, 1, 0, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. grid queries
# ═══════════════════════════════════════════════════════════════════════════

class TestGridQueries:

    def test_coverage_partial(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['surface_occ'][1, 1] = 1
        assert surface_coverage(g) == pytest.approx(2 / 16)

    def test_sub1_and_sub2_population(self):
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][0, 0] = 1
        g['sub2_occ'][1, 1] = 1
        g['sub2_occ'][2, 2] = 1
        assert sub1_population(g) == 1
        assert sub2_population(g) == 2
        assert subsurface_population(g) == 3      # total across both layers

    def test_concentration_uses_sub2_only(self):
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][0, 0] = 1                   # sub1 must NOT contribute to C0
        assert subsurface_concentration(g, _A0) == pytest.approx(0.0)
        g['sub2_occ'][0, 0] = 1
        assert subsurface_concentration(g, _A0) > 0.0

    def test_concentration_exact_formula(self):
        g = _all_ni_grid(4, 4)
        g['sub2_occ'][0, 0] = 1
        g['sub2_occ'][1, 2] = 1
        expected = 2 / (4 * 4 * (_A0 ** 3) / math.sqrt(2.0)) / _N_A   # mol H per m^3
        assert subsurface_concentration(g, _A0) == pytest.approx(expected, rel=1e-8)

    def test_concentration_layer_selects_sub1(self):
        # layer='sub1' reports the first-subsurface occupancy (the dissolved
        # reference); the default ('sub2') must not see it.
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][0, 0] = 1
        assert subsurface_concentration(g, _A0) == pytest.approx(0.0)
        assert subsurface_concentration(g, _A0, layer='sub1') > 0.0
        expected = 1 / (4 * 4 * (_A0 ** 3) / math.sqrt(2.0)) / _N_A   # mol H per m^3
        assert subsurface_concentration(g, _A0, layer='sub1') == pytest.approx(expected, rel=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 7. build_event_list
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildEventList:

    def _of_kind(self, events, kind):
        return [e for e in events if e['kind'] == kind]

    def test_empty_grid_only_adsorb(self):
        g = _all_ni_grid(4, 4)
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert len(self._of_kind(ev, 'adsorb')) > 0
        for k in ('enter', 'exit', 'hopB_enter', 'hopB_exit', 'drain'):
            assert len(self._of_kind(ev, k)) == 0

    def test_kdiss_mislabelled_pair_still_adsorbs_via_fallback(self):
        # A k_diss/k_des keyed by a pair that does NOT match the grid element
        # (the ('s','s') mislabelling that silently zeroed adsorption and left
        # the whole KMC inert) must still fire adsorb events via the per-class
        # mean-fallback -- never a silent empty grid on a known element.
        g = _all_ni_grid(4, 4)
        rd = dict(_RATES)
        rd['k_diss'] = {('s', 's'): 0.5}
        rd['k_des']  = {('s', 's'): 1e6}
        ev = build_event_list(g, rd, _P, _T, _D, _A0)
        assert len(self._of_kind(ev, 'adsorb')) > 0

    def test_empty_kdiss_stays_inert(self):
        # The fallback never fabricates a rate: a genuinely-empty k_diss yields
        # no adsorb events (mean of {} is 0.0).
        g = _all_ni_grid(4, 4)
        rd = dict(_RATES)
        rd['k_diss'] = {}
        ev = build_event_list(g, rd, _P, _T, _D, _A0)
        assert len(self._of_kind(ev, 'adsorb')) == 0

    def test_surface_occupied_generates_enter(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert any((0, 0) in e['sites'] for e in self._of_kind(ev, 'enter'))

    def test_sub1_occupied_empty_surface_generates_exit(self):
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][1, 1] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert any((1, 1) in e['sites'] for e in self._of_kind(ev, 'exit'))

    def test_sub1_occupied_empty_sub2_generates_hopB_enter(self):
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][2, 2] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert any((2, 2) in e['sites'] for e in self._of_kind(ev, 'hopB_enter'))

    def test_sub2_occupied_empty_sub1_generates_hopB_exit(self):
        g = _all_ni_grid(4, 4)
        g['sub2_occ'][3, 3] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert any((3, 3) in e['sites'] for e in self._of_kind(ev, 'hopB_exit'))

    def test_sub2_occupied_generates_drain(self):
        g = _all_ni_grid(4, 4)
        g['sub2_occ'][2, 2] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert any((2, 2) in e['sites'] for e in self._of_kind(ev, 'drain'))

    def test_sub1_occupied_does_not_drain(self):
        # drain is from sub2 only — an occupied sub1 with empty sub2 must not drain
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][2, 2] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert len(self._of_kind(ev, 'drain')) == 0

    def test_adjacent_occupied_pair_desorbs(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['surface_occ'][0, 1] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert len(self._of_kind(ev, 'desorb')) > 0

    def test_occupied_site_adjacent_empty_diffuses(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][2, 2] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert len(self._of_kind(ev, 'surf_diff')) > 0

    def test_no_adsorb_when_k_diss_absent(self):
        g = _all_ni_grid(4, 4)
        rates = {k: v for k, v in _RATES.items() if k != 'k_diss'}
        ev = build_event_list(g, rates, _P, _T, _D, _A0)
        assert len(self._of_kind(ev, 'adsorb')) == 0

    def test_all_event_rates_positive(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['sub1_occ'][1, 1] = 1
        g['sub2_occ'][2, 2] = 1
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert all(e['rate'] > 0.0 for e in ev)

    def test_enter_not_generated_if_sub1_occupied(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        g['sub1_occ'][0, 0] = 1     # both occupied — no enter possible
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert not any(e['kind'] == 'enter' and (0, 0) in e['sites'] for e in ev)

    def test_entry_rate_resolves_per_env(self):
        # Two distinct sub1 environments get distinct entry rates.
        g = make_grid(1, 2, composition={'Ni': 1.0}, seed=0)
        g['sub1_env'][0, 0] = 'Ni6_oct'
        g['sub1_env'][0, 1] = 'Ni5Mo_oct'
        g['surface_occ'][:] = 1
        rates = dict(_RATES, k_entry={'Ni6_oct': 1e7, 'Ni5Mo_oct': 3e7})
        ev = build_event_list(g, rates, _P, _T, _D, _A0)
        by_site = {e['sites'][0]: e['rate'] for e in ev if e['kind'] == 'enter'}
        assert by_site[(0, 0)] == pytest.approx(1e7)
        assert by_site[(0, 1)] == pytest.approx(3e7)

    def test_unknown_env_falls_back_to_mean_not_zero(self):
        g = make_grid(1, 1, composition={'Ni': 1.0}, seed=0)
        g['sub1_env'][0, 0] = 'Ni4Mo2_oct'      # not in k_entry
        g['surface_occ'][0, 0] = 1
        rates = dict(_RATES, k_entry={'Ni6_oct': 2e7, 'Ni5Mo_oct': 4e7})
        ev = build_event_list(g, rates, _P, _T, _D, _A0)
        enter = [e for e in ev if e['kind'] == 'enter']
        assert len(enter) == 1
        assert enter[0]['rate'] == pytest.approx(3e7)   # mean(2e7, 4e7), not 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 8. _execute_event
# ═══════════════════════════════════════════════════════════════════════════

class TestExecuteEvent:

    def test_adsorb_occupies_both_surface_sites(self):
        g = _all_ni_grid(4, 4)
        _execute_event(g, {'kind': 'adsorb', 'sites': [(0, 0), (0, 1)], 'rate': 1.0})
        assert g['surface_occ'][0, 0] == 1 and g['surface_occ'][0, 1] == 1

    def test_desorb_clears_both_surface_sites(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][1, 1] = 1
        g['surface_occ'][1, 2] = 1
        _execute_event(g, {'kind': 'desorb', 'sites': [(1, 1), (1, 2)], 'rate': 1.0})
        assert g['surface_occ'][1, 1] == 0 and g['surface_occ'][1, 2] == 0

    def test_surf_diff_moves_h(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][2, 2] = 1
        _execute_event(g, {'kind': 'surf_diff', 'sites': [(2, 2), (2, 3)], 'rate': 1.0})
        assert g['surface_occ'][2, 2] == 0 and g['surface_occ'][2, 3] == 1

    def test_enter_surface_to_sub1(self):
        g = _all_ni_grid(4, 4)
        g['surface_occ'][0, 0] = 1
        _execute_event(g, {'kind': 'enter', 'sites': [(0, 0)], 'rate': 1.0})
        assert g['surface_occ'][0, 0] == 0 and g['sub1_occ'][0, 0] == 1

    def test_exit_sub1_to_surface(self):
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][3, 3] = 1
        _execute_event(g, {'kind': 'exit', 'sites': [(3, 3)], 'rate': 1.0})
        assert g['sub1_occ'][3, 3] == 0 and g['surface_occ'][3, 3] == 1

    def test_hopB_enter_sub1_to_sub2(self):
        g = _all_ni_grid(4, 4)
        g['sub1_occ'][2, 1] = 1
        _execute_event(g, {'kind': 'hopB_enter', 'sites': [(2, 1)], 'rate': 1.0})
        assert g['sub1_occ'][2, 1] == 0 and g['sub2_occ'][2, 1] == 1

    def test_hopB_exit_sub2_to_sub1(self):
        g = _all_ni_grid(4, 4)
        g['sub2_occ'][1, 3] = 1
        _execute_event(g, {'kind': 'hopB_exit', 'sites': [(1, 3)], 'rate': 1.0})
        assert g['sub2_occ'][1, 3] == 0 and g['sub1_occ'][1, 3] == 1

    def test_drain_clears_sub2(self):
        g = _all_ni_grid(4, 4)
        g['sub2_occ'][1, 3] = 1
        _execute_event(g, {'kind': 'drain', 'sites': [(1, 3)], 'rate': 1.0})
        assert g['sub2_occ'][1, 3] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 9. kmc_step
# ═══════════════════════════════════════════════════════════════════════════

class TestKmcStep:

    def test_empty_event_list_returns_zero(self):
        assert kmc_step(_all_ni_grid(4, 4), []) == 0.0

    def test_returns_positive_dt_when_events_present(self):
        np.random.seed(42)
        g = _all_ni_grid(4, 4)
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        assert kmc_step(g, ev) > 0.0

    def test_grid_mutated_after_step(self):
        np.random.seed(42)
        g = _all_ni_grid(4, 4)
        ev = build_event_list(g, _RATES, _P, _T, _D, _A0)
        before = g['surface_occ'].copy()
        kmc_step(g, ev)
        assert not np.all(g['surface_occ'] == before)

    def test_dt_scales_inversely_with_total_rate(self):
        g1 = _all_ni_grid(2, 2); g1['sub2_occ'][0, 0] = 1
        g2 = _all_ni_grid(2, 2); g2['sub2_occ'][0, 0] = 1
        e1 = [{'kind': 'drain', 'sites': [(0, 0)], 'rate': 1.0}]
        e2 = [{'kind': 'drain', 'sites': [(0, 0)], 'rate': 2.0}]
        np.random.seed(99); dt1 = kmc_step(g1, e1)
        np.random.seed(99); dt2 = kmc_step(g2, e2)
        assert dt1 == pytest.approx(2.0 * dt2, rel=1e-8)

    def test_zero_total_rate_returns_zero(self):
        assert kmc_step(_all_ni_grid(4, 4),
                        [{'kind': 'drain', 'sites': [(0, 0)], 'rate': 0.0}]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 10. run_kmc
# ═══════════════════════════════════════════════════════════════════════════

class TestRunKmc:

    def test_returns_dict_with_required_keys(self):
        np.random.seed(0)
        result = run_kmc(_all_ni_grid(4, 4), _RATES, _P, _T, _D, _A0, n_steps=5)
        for k in ('t_arr', 'theta_arr', 'n_sub_arr'):
            assert k in result

    def test_array_length_is_n_steps_plus_one(self):
        np.random.seed(0)
        result = run_kmc(_all_ni_grid(4, 4), _RATES, _P, _T, _D, _A0, n_steps=10)
        assert len(result['t_arr']) == 11 and len(result['theta_arr']) == 11

    def test_t_arr_monotonically_nondecreasing(self):
        np.random.seed(0)
        result = run_kmc(_all_ni_grid(4, 4), _RATES, _P, _T, _D, _A0, n_steps=20)
        assert np.all(np.diff(result['t_arr']) >= 0.0)

    def test_initial_state_zero(self):
        np.random.seed(0)
        result = run_kmc(_all_ni_grid(4, 4), _RATES, _P, _T, _D, _A0, n_steps=5)
        assert result['t_arr'][0] == pytest.approx(0.0)
        assert result['theta_arr'][0] == pytest.approx(0.0)
        assert result['n_sub_arr'][0] == 0

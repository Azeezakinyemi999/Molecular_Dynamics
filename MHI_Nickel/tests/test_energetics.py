"""
tests/test_energetics.py
=========================
Tests for models/energetics.py — offline, no LAMMPS or matplotlib required.

Energy formulas are tested against analytical values; site-ranking tests use
hand-crafted dicts; summarise_neb tests use synthetic barrier/path files.
"""

import sys
import pathlib
import pytest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.energetics import (
    KB_EV,
    calc_binding_energy,
    calc_dissociation_barrier,
    calc_reaction_energy,
    mu_h_to_pressure,
    pressure_to_mu_h,
    rank_sites,
    best_n_sites,
    summarise_neb,
)


# ── synthetic file helpers ──────────────────────────────────────────────────

def _write_barrier(path, E_IS=-100.5, E_FS=-100.2, E_abs=0.45, E_des=0.15,
                   delta_E=0.3, fmax=0.05, converged=True):
    text = (
        f'E_IS: {E_IS}\n'
        f'E_FS: {E_FS}\n'
        f'E_abs: {E_abs}\n'
        f'E_des: {E_des}\n'
        f'delta_E: {delta_E}\n'
        f'fmax_final: {fmax}\n'
        f'converged: {converged}\n'
    )
    pathlib.Path(path).write_text(text)


def _write_path(path, n=5):
    lines = ['# image_fraction  E_eV  dE_from_IS_eV\n']
    fracs = np.linspace(0, 1, n)
    # simple hill: 0 → 0.45 peak at centre → 0.3
    dE = np.array([0.0, 0.25, 0.45, 0.35, 0.30])[:n]
    E_base = -100.5
    for f, de in zip(fracs, dE):
        lines.append(f'{f:.4f}  {E_base + de:.4f}  {de:.4f}\n')
    pathlib.Path(path).write_text(''.join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# 1. calc_binding_energy
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcBindingEnergy:

    def test_basic_formula(self):
        result = calc_binding_energy(e_with_h=-10.0, e_clean=-8.0,
                                     e_h2_gas=-6.0, n_h2_ref=0.5)
        assert result == pytest.approx(-10.0 - (-8.0) - 0.5 * (-6.0))

    def test_default_n_h2_ref_is_half(self):
        e_bind = calc_binding_energy(-10.0, -8.0, -6.0)
        # default n_h2_ref = 0.5
        assert e_bind == pytest.approx(-10.0 + 8.0 + 3.0)

    def test_n_h2_ref_one_subtracts_full_h2(self):
        e_bind = calc_binding_energy(-10.0, -8.0, -6.0, n_h2_ref=1.0)
        assert e_bind == pytest.approx(-10.0 + 8.0 + 6.0)

    def test_negative_for_stable_adsorption(self):
        # slab+H much lower in energy than clean + H₂
        e_bind = calc_binding_energy(-15.0, -10.0, -5.0, n_h2_ref=0.5)
        assert e_bind < 0.0

    def test_positive_for_unstable_binding(self):
        # slab+H higher in energy than clean + H₂
        e_bind = calc_binding_energy(-9.0, -10.0, -5.0, n_h2_ref=0.5)
        assert e_bind > 0.0

    def test_zero_when_balanced(self):
        # zero when e_with_h = e_clean + n_h2_ref * e_h2_gas
        # -10.0 + 0.5*(-2.0) = -11.0
        e_bind = calc_binding_energy(-11.0, -10.0, -2.0, n_h2_ref=0.5)
        assert e_bind == pytest.approx(0.0)

    def test_n_h2_ref_one_vs_half_differ_by_half_h2(self):
        e1 = calc_binding_energy(-10.0, -8.0, -6.0, n_h2_ref=0.5)
        e2 = calc_binding_energy(-10.0, -8.0, -6.0, n_h2_ref=1.0)
        # difference = -0.5 * e_h2_gas = 3.0
        assert e2 - e1 == pytest.approx(-0.5 * (-6.0))

    def test_returns_float(self):
        result = calc_binding_energy(-10.0, -8.0, -6.0)
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════════════
# 2. calc_dissociation_barrier
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcDissociationBarrier:

    def test_basic_formula(self):
        assert calc_dissociation_barrier(e_ts=-5.0, e_is=-5.5) == pytest.approx(0.5)

    def test_zero_when_ts_equals_is(self):
        assert calc_dissociation_barrier(-5.0, -5.0) == pytest.approx(0.0)

    def test_negative_barrier_if_ts_below_is(self):
        # can be negative if the "TS" is not a true saddle point
        assert calc_dissociation_barrier(-6.0, -5.0) == pytest.approx(-1.0)

    def test_returns_float(self):
        assert isinstance(calc_dissociation_barrier(-5.0, -5.5), float)

    def test_large_barrier(self):
        assert calc_dissociation_barrier(0.0, -3.0) == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. calc_reaction_energy
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcReactionEnergy:

    def test_basic_formula(self):
        assert calc_reaction_energy(e_fs=-5.2, e_is=-5.5) == pytest.approx(0.3)

    def test_negative_for_exothermic(self):
        assert calc_reaction_energy(-6.0, -5.0) < 0.0

    def test_positive_for_endothermic(self):
        assert calc_reaction_energy(-4.0, -5.0) > 0.0

    def test_zero_when_is_equals_fs(self):
        assert calc_reaction_energy(-5.0, -5.0) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(calc_reaction_energy(-5.0, -5.5), float)


# ═══════════════════════════════════════════════════════════════════════════
# 4. mu_h_to_pressure
# ═══════════════════════════════════════════════════════════════════════════

class TestMuHToPressure:

    _e_h2 = -6.7

    def test_pressure_one_when_mu_equals_half_e_h2(self):
        # exponent = 2*(0.5*e_h2 - 0.5*e_h2) / (kB*T) = 0 → P=1
        mu_h = 0.5 * self._e_h2
        P = mu_h_to_pressure(mu_h, self._e_h2, T=300.)
        assert P == pytest.approx(1.0, rel=1e-6)

    def test_higher_mu_higher_pressure(self):
        P1 = mu_h_to_pressure(-3.5, self._e_h2, T=300.)
        P2 = mu_h_to_pressure(-3.0, self._e_h2, T=300.)
        assert P2 > P1

    def test_lower_mu_lower_pressure(self):
        P = mu_h_to_pressure(-4.5, self._e_h2, T=300.)
        assert P < 1.0

    def test_positive_pressure_always(self):
        for mu in [-5.0, -4.0, -3.5, -3.0]:
            assert mu_h_to_pressure(mu, self._e_h2, T=300.) > 0.0

    def test_exact_value(self):
        mu_h = -3.5
        T = 300.
        expected = np.exp(2.0 * (mu_h - 0.5 * self._e_h2) / (KB_EV * T))
        assert mu_h_to_pressure(mu_h, self._e_h2, T) == pytest.approx(expected, rel=1e-8)

    def test_higher_t_smaller_pressure_for_same_mu(self):
        # for mu_h < 0.5*e_h2 (negative exponent), higher T → larger |kBT| → smaller exp
        mu_h = -4.0  # below 0.5*e_h2 = -3.35 → exponent negative
        P_low_T  = mu_h_to_pressure(mu_h, self._e_h2, T=300.)
        P_high_T = mu_h_to_pressure(mu_h, self._e_h2, T=1000.)
        assert P_high_T > P_low_T  # closer to 1 (less negative exponent scaled by larger T)

    def test_returns_float(self):
        assert isinstance(mu_h_to_pressure(-3.5, self._e_h2, T=300.), float)


# ═══════════════════════════════════════════════════════════════════════════
# 5. pressure_to_mu_h
# ═══════════════════════════════════════════════════════════════════════════

class TestPressureToMuH:

    _e_h2 = -6.7

    def test_pressure_one_gives_half_e_h2(self):
        # ln(1) = 0 → mu_h = 0.5 * e_h2
        mu = pressure_to_mu_h(1.0, self._e_h2, T=300.)
        assert mu == pytest.approx(0.5 * self._e_h2)

    def test_higher_pressure_higher_mu(self):
        mu1 = pressure_to_mu_h(1.0, self._e_h2, T=300.)
        mu2 = pressure_to_mu_h(10.0, self._e_h2, T=300.)
        assert mu2 > mu1

    def test_exact_value(self):
        P = 5.0; T = 500.
        expected = 0.5 * self._e_h2 + 0.5 * KB_EV * T * np.log(P)
        assert pressure_to_mu_h(P, self._e_h2, T) == pytest.approx(expected, rel=1e-8)

    def test_returns_float(self):
        assert isinstance(pressure_to_mu_h(1.0, self._e_h2, T=300.), float)


# ═══════════════════════════════════════════════════════════════════════════
# 6. mu_h_to_pressure / pressure_to_mu_h round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestMuHPressureRoundTrip:

    _e_h2 = -6.7

    def test_mu_to_p_to_mu_recovers_original(self):
        mu_h = -3.8
        P = mu_h_to_pressure(mu_h, self._e_h2, T=400.)
        mu_back = pressure_to_mu_h(P, self._e_h2, T=400.)
        assert mu_back == pytest.approx(mu_h, rel=1e-8)

    def test_p_to_mu_to_p_recovers_original(self):
        P = 1e-3
        mu = pressure_to_mu_h(P, self._e_h2, T=600.)
        P_back = mu_h_to_pressure(mu, self._e_h2, T=600.)
        assert P_back == pytest.approx(P, rel=1e-8)

    def test_round_trip_at_multiple_temperatures(self):
        mu_h = -3.6
        for T in [300., 500., 800.]:
            P = mu_h_to_pressure(mu_h, self._e_h2, T)
            mu_back = pressure_to_mu_h(P, self._e_h2, T)
            assert mu_back == pytest.approx(mu_h, rel=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 7. rank_sites
# ═══════════════════════════════════════════════════════════════════════════

class TestRankSites:

    def test_empty_dict_returns_empty_list(self):
        assert rank_sites({}) == []

    def test_single_site_returns_list_of_one(self):
        result = rank_sites({'top': -0.3})
        assert len(result) == 1
        assert result[0] == ('top', -0.3)

    def test_most_stable_first(self):
        sites = {'top': -0.3, 'fcc': -0.7, 'bridge': -0.5}
        ranked = rank_sites(sites)
        assert ranked[0][0] == 'fcc'

    def test_sorted_ascending_by_energy(self):
        sites = {'a': -0.1, 'b': -0.9, 'c': -0.5}
        ranked = rank_sites(sites)
        energies = [e for _, e in ranked]
        assert energies == sorted(energies)

    def test_all_sites_present(self):
        sites = {'top': -0.3, 'fcc': -0.7, 'bridge': -0.5}
        ranked = rank_sites(sites)
        assert len(ranked) == 3

    def test_returns_list_of_tuples(self):
        ranked = rank_sites({'x': -0.5})
        assert isinstance(ranked, list)
        assert isinstance(ranked[0], tuple)

    def test_original_dict_not_mutated(self):
        sites = {'a': -0.5, 'b': -0.3}
        original_keys = list(sites.keys())
        rank_sites(sites)
        assert list(sites.keys()) == original_keys


# ═══════════════════════════════════════════════════════════════════════════
# 8. best_n_sites
# ═══════════════════════════════════════════════════════════════════════════

class TestBestNSites:

    _sites = {
        'top_Ni':  -0.30,
        'bridge':  -0.55,
        'fcc':     -0.72,
        'hcp':     -0.68,
        'subsurface': -0.40,
    }

    _coords = {
        'top_Ni':     np.array([0.0, 0.0]),
        'bridge':     np.array([1.0, 0.0]),
        'fcc':        np.array([2.0, 0.0]),
        'hcp':        np.array([2.1, 0.0]),   # very close to fcc
        'subsurface': np.array([5.0, 5.0]),
    }

    def test_default_n_two_returns_two(self):
        result = best_n_sites(self._sites)
        assert len(result) == 2

    def test_n_one_returns_most_stable(self):
        result = best_n_sites(self._sites, n=1)
        assert len(result) == 1
        assert result[0][0] == 'fcc'

    def test_n_larger_than_sites_returns_all(self):
        result = best_n_sites(self._sites, n=100)
        assert len(result) == len(self._sites)

    def test_without_separation_returns_top_n_by_energy(self):
        result = best_n_sites(self._sites, n=2)
        labels = [lbl for lbl, _ in result]
        assert 'fcc' in labels
        assert 'hcp' in labels

    def test_raises_valueerror_if_sep_set_without_coords(self):
        with pytest.raises(ValueError, match='site_coords'):
            best_n_sites(self._sites, n=2, min_separation=1.5)

    def test_separation_filter_excludes_close_sites(self):
        # fcc and hcp are 0.1 Å apart — min_separation=1.0 should exclude hcp
        result = best_n_sites(self._sites, n=2,
                              min_separation=1.0, site_coords=self._coords)
        labels = [lbl for lbl, _ in result]
        assert 'fcc' in labels
        assert 'hcp' not in labels

    def test_separation_zero_includes_all(self):
        result = best_n_sites(self._sites, n=5,
                              min_separation=0.0, site_coords=self._coords)
        assert len(result) == 5

    def test_result_sorted_by_energy(self):
        result = best_n_sites(self._sites, n=3)
        energies = [e for _, e in result]
        assert energies == sorted(energies)

    def test_separation_filter_picks_next_best_after_exclusion(self):
        # fcc (-0.72) chosen first; hcp excluded (0.1 Å away); next should be bridge (-0.55)
        result = best_n_sites(self._sites, n=2,
                              min_separation=1.0, site_coords=self._coords)
        labels = [lbl for lbl, _ in result]
        assert 'bridge' in labels

    def test_skips_sites_not_in_coords_when_filter_active(self):
        # site_coords missing 'top_Ni' — it should be skipped
        coords_partial = {k: v for k, v in self._coords.items() if k != 'top_Ni'}
        result = best_n_sites(self._sites, n=5,
                              min_separation=0.0, site_coords=coords_partial)
        labels = [lbl for lbl, _ in result]
        assert 'top_Ni' not in labels

    def test_single_valid_site_after_separation_filter(self):
        # two sites 0.1 Å apart; min_separation=1.0 → only the better one survives
        sites  = {'a': -0.80, 'b': -0.75}
        coords = {'a': np.array([0.0, 0.0]), 'b': np.array([0.05, 0.0])}
        result = best_n_sites(sites, n=2, min_separation=1.0, site_coords=coords)
        assert len(result) == 1
        assert result[0][0] == 'a'


# ═══════════════════════════════════════════════════════════════════════════
# 9. summarise_neb
# ═══════════════════════════════════════════════════════════════════════════

class TestSummariseNeb:

    @pytest.fixture()
    def neb_files(self, tmp_path):
        barrier = str(tmp_path / 'neb_barrier.txt')
        path    = str(tmp_path / 'neb_path.dat')
        _write_barrier(barrier, E_abs=0.45, E_des=0.15, delta_E=0.30,
                       fmax=0.045, converged=True)
        _write_path(path, n=5)
        return barrier, path

    def test_returns_dict(self, neb_files):
        barrier, path = neb_files
        result = summarise_neb(barrier, path)
        assert isinstance(result, dict)

    def test_required_keys_present(self, neb_files):
        barrier, path = neb_files
        result = summarise_neb(barrier, path)
        for key in ('Ea', 'E_des', 'delta_E', 'ts_index', 'converged',
                    'fmax_final', 'rxn_coord', 'energies'):
            assert key in result

    def test_ea_from_barrier_file(self, neb_files):
        barrier, path = neb_files
        result = summarise_neb(barrier, path)
        assert result['Ea'] == pytest.approx(0.45)

    def test_e_des_from_barrier_file(self, neb_files):
        barrier, path = neb_files
        result = summarise_neb(barrier, path)
        assert result['E_des'] == pytest.approx(0.15)

    def test_delta_e_from_barrier_file(self, neb_files):
        barrier, path = neb_files
        result = summarise_neb(barrier, path)
        assert result['delta_E'] == pytest.approx(0.30)

    def test_converged_true(self, neb_files):
        barrier, path = neb_files
        result = summarise_neb(barrier, path)
        assert result['converged'] is True

    def test_rxn_coord_loaded_from_path_file(self, neb_files):
        barrier, path = neb_files
        result = summarise_neb(barrier, path)
        assert result['rxn_coord'] is not None
        assert len(result['rxn_coord']) == 5

    def test_rxn_coord_none_when_path_file_missing(self, tmp_path):
        barrier = str(tmp_path / 'neb_barrier.txt')
        _write_barrier(barrier)
        result = summarise_neb(barrier, str(tmp_path / 'nonexistent.dat'))
        assert result['rxn_coord'] is None

    def test_ea_none_when_barrier_file_missing(self, tmp_path):
        path = str(tmp_path / 'neb_path.dat')
        _write_path(path)
        result = summarise_neb(str(tmp_path / 'nonexistent.txt'), path)
        assert result['Ea'] is None

    def test_converged_false_parsed(self, tmp_path):
        barrier = str(tmp_path / 'barrier_notconv.txt')
        path    = str(tmp_path / 'path.dat')
        _write_barrier(barrier, converged=False)
        _write_path(path)
        result = summarise_neb(barrier, path)
        assert result['converged'] is False

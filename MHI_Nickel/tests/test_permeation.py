"""
tests/test_permeation.py
========================
Tests for models/permeation.py — offline, no LAMMPS or KMC run required
for the unit tests (sweep_pressure is not tested here; it wraps run_kmc_to_steady_state).

Covers:
  fick_flux, check_sieverts_law, arrhenius_diffusivity,
  lattice_site_S0, solubility_from_rates, fit_solubility_from_kmc,
  sieverts_solubility, permeability, richardson_flux
"""

import json
import math
import sys
import pathlib
import pytest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.permeation import (
    _KB_EV,
    _KB_J,
    _M_H2_KG,
    fick_flux,
    check_sieverts_law,
    arrhenius_diffusivity,
    lattice_site_S0,
    solubility_from_rates,
    fit_solubility_from_kmc,
    sieverts_solubility,
    permeability,
    richardson_flux,
    resolve_nh_diffusivity,
)


# ── shared fixtures ──────────────────────────────────────────────────────────

_A0  = 3.52e-10   # Ni lattice constant [m]
_T   = 600.0      # K


def _sweep(P_vals, C0_vals, converged=None):
    """Minimal sweep_result dict for fit_solubility_from_kmc tests."""
    if converged is None:
        converged = [True] * len(P_vals)
    return {
        'P_vals':      list(P_vals),
        'C0_vals':     list(C0_vals),
        'converged':   list(converged),
        'sqrt_P_vals': [float(np.sqrt(p)) for p in P_vals],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. fick_flux
# ═══════════════════════════════════════════════════════════════════════════

class TestFickFlux:

    def test_basic_formula(self):
        # J = D * C0 / L when C_low = 0
        D, C0, L = 1e-10, 1e25, 1e-3
        assert fick_flux(D, C0, L) == pytest.approx(D * C0 / L, rel=1e-10)

    def test_with_nonzero_c_low(self):
        D, C0, C_low, L = 1e-10, 2e25, 1e25, 1e-3
        expected = D * (C0 - C_low) / L
        assert fick_flux(D, C0, L, C_low_m3=C_low) == pytest.approx(expected, rel=1e-10)

    def test_raises_for_zero_thickness(self):
        with pytest.raises(ValueError):
            fick_flux(1e-10, 1e25, 0.0)

    def test_raises_for_negative_thickness(self):
        with pytest.raises(ValueError):
            fick_flux(1e-10, 1e25, -1e-3)

    def test_linear_in_diffusivity(self):
        J1 = fick_flux(1e-10, 1e25, 1e-3)
        J2 = fick_flux(2e-10, 1e25, 1e-3)
        assert J2 == pytest.approx(2.0 * J1, rel=1e-10)

    def test_inversely_proportional_to_thickness(self):
        J1 = fick_flux(1e-10, 1e25, 1e-3)
        J2 = fick_flux(1e-10, 1e25, 2e-3)
        assert J1 == pytest.approx(2.0 * J2, rel=1e-10)

    def test_equal_concentrations_gives_zero_flux(self):
        assert fick_flux(1e-10, 5e24, 1e-3, C_low_m3=5e24) == pytest.approx(0.0, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# 2. check_sieverts_law
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckSievertsFit:

    def _perfect_sieverts(self):
        P_vals = [1000.0, 4000.0, 9000.0, 16000.0]
        J_vals = [2.5 * math.sqrt(p) for p in P_vals]   # J ∝ √P exactly
        return P_vals, J_vals

    def test_returns_required_keys(self):
        P, J = self._perfect_sieverts()
        result = check_sieverts_law(P, J, plot=False)
        for k in ('slope', 'intercept', 'r_squared', 'is_sieverts'):
            assert k in result

    def test_perfect_linear_gives_r2_one(self):
        P, J = self._perfect_sieverts()
        result = check_sieverts_law(P, J, plot=False)
        assert result['r_squared'] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_linear_is_sieverts_true(self):
        P, J = self._perfect_sieverts()
        assert check_sieverts_law(P, J, plot=False)['is_sieverts'] is True

    def test_nonlinear_j_not_sieverts(self):
        P_vals = [1000.0, 4000.0, 9000.0, 16000.0]
        # J proportional to P (quadratic in √P) → R² < 0.98
        J_vals = [p * 0.001 for p in P_vals]
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        assert result['is_sieverts'] is False
        assert result['r_squared'] < 0.98

    def test_slope_matches_coefficient(self):
        P_vals = [1000.0, 4000.0, 9000.0, 16000.0]
        slope_true = 2.5
        J_vals = [slope_true * math.sqrt(p) for p in P_vals]
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        assert result['slope'] == pytest.approx(slope_true, rel=1e-6)

    def test_threshold_at_0_98(self):
        # R² threshold is exactly 0.98
        P, J = self._perfect_sieverts()
        result = check_sieverts_law(P, J, plot=False)
        assert result['r_squared'] >= 0.98

    def test_constant_j_r2_is_one(self):
        # all J identical → ss_tot = 0; code returns the special-case r2 = 1.0
        P_vals = [1000.0, 4000.0, 9000.0, 16000.0]
        J_vals = [5.0, 5.0, 5.0, 5.0]
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        assert result['r_squared'] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. arrhenius_diffusivity
# ═══════════════════════════════════════════════════════════════════════════

class TestArrheniusDiffusivity:

    def test_exact_formula(self):
        D0, E_D, T = 1e-7, 0.4, 600.0
        expected = D0 * math.exp(-E_D / (_KB_EV * T))
        assert arrhenius_diffusivity(D0, E_D, T) == pytest.approx(expected, rel=1e-8)

    def test_raises_for_zero_temperature(self):
        with pytest.raises(ValueError):
            arrhenius_diffusivity(1e-7, 0.4, 0.0)

    def test_raises_for_negative_temperature(self):
        with pytest.raises(ValueError):
            arrhenius_diffusivity(1e-7, 0.4, -100.0)

    def test_higher_temperature_gives_higher_d(self):
        D0, E_D = 1e-7, 0.4
        assert arrhenius_diffusivity(D0, E_D, 800.0) > arrhenius_diffusivity(D0, E_D, 600.0)

    def test_zero_barrier_returns_d0(self):
        D0 = 1e-7
        assert arrhenius_diffusivity(D0, 0.0, 600.0) == pytest.approx(D0, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 4. lattice_site_S0
# ═══════════════════════════════════════════════════════════════════════════

class TestLatticeSiteS0:

    def test_exact_formula(self):
        a0 = 3.52e-10
        expected = 4.0 / (a0 ** 3)
        assert lattice_site_S0(a0) == pytest.approx(expected, rel=1e-10)

    def test_returns_positive_float(self):
        assert lattice_site_S0(_A0) > 0.0

    def test_inversely_proportional_to_a0_cubed(self):
        S1 = lattice_site_S0(1e-10)
        S2 = lattice_site_S0(2e-10)
        assert S1 / S2 == pytest.approx(8.0, rel=1e-8)

    def test_smaller_lattice_higher_density(self):
        assert lattice_site_S0(3.0e-10) > lattice_site_S0(4.0e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 5. solubility_from_rates
# ═══════════════════════════════════════════════════════════════════════════

class TestSolubilityFromRates:

    _K_DISS  = 0.5          # dimensionless sticking factor
    _K_DES   = 1e6          # s⁻¹
    _K_ENTRY = 1e7          # s⁻¹
    _K_EXIT  = 1e6          # s⁻¹

    def _call(self, **kw):
        defaults = dict(
            k_diss=self._K_DISS, k_des_s1=self._K_DES,
            k_entry_s1=self._K_ENTRY, k_exit_s1=self._K_EXIT,
            a0_m=_A0, T_K=_T,
        )
        defaults.update(kw)
        return solubility_from_rates(**defaults)

    def test_returns_positive_float(self):
        assert self._call() > 0.0

    def test_raises_for_zero_temperature(self):
        with pytest.raises(ValueError):
            self._call(T_K=0.0)

    def test_raises_for_zero_k_des(self):
        with pytest.raises(ValueError):
            self._call(k_des_s1=0.0)

    def test_raises_for_zero_k_exit(self):
        with pytest.raises(ValueError):
            self._call(k_exit_s1=0.0)

    def test_higher_k_entry_gives_higher_s(self):
        S1 = self._call(k_entry_s1=1e7)
        S2 = self._call(k_entry_s1=2e7)
        assert S2 > S1

    def test_higher_k_exit_gives_lower_s(self):
        S1 = self._call(k_exit_s1=1e6)
        S2 = self._call(k_exit_s1=2e6)
        assert S2 < S1

    def test_exact_formula(self):
        k_diss, k_des, k_entry, k_exit = 0.5, 1e6, 1e7, 1e6
        rho_oct = 4.0 / (_A0 ** 3)
        A_site  = (_A0 / math.sqrt(2.0)) ** 2
        denom   = k_des * math.sqrt(2.0 * math.pi * _M_H2_KG * _KB_J * _T)
        expected = rho_oct * (k_entry / k_exit) * math.sqrt(k_diss * A_site / denom)
        got = solubility_from_rates(k_diss, k_des, k_entry, k_exit, _A0, _T)
        assert got == pytest.approx(expected, rel=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 6. fit_solubility_from_kmc
# ═══════════════════════════════════════════════════════════════════════════

class TestFitSolubilityFromKmc:

    def test_returns_required_keys(self):
        result = fit_solubility_from_kmc(_sweep([1000.0], [1e22]))
        for k in ('S_vals', 'P_vals', 'S_mean', 'S_std', 'n_converged'):
            assert k in result

    def test_s_vals_equal_c0_over_sqrt_p(self):
        P = [1000.0, 4000.0]
        C0 = [2e22, 4e22]
        result = fit_solubility_from_kmc(_sweep(P, C0))
        for i, (p, c) in enumerate(zip(P, C0)):
            assert result['S_vals'][i] == pytest.approx(c / math.sqrt(p), rel=1e-8)

    def test_s_mean_correct(self):
        P = [1000.0, 4000.0]
        C0 = [1e22, 2e22]
        S0 = 1e22 / math.sqrt(1000.0)
        S1 = 2e22 / math.sqrt(4000.0)
        result = fit_solubility_from_kmc(_sweep(P, C0))
        assert result['S_mean'] == pytest.approx((S0 + S1) / 2.0, rel=1e-8)

    def test_non_converged_excluded_from_mean(self):
        P = [1000.0, 4000.0]
        C0 = [1e22, 9e22]         # second point is wild — should be excluded
        conv = [True, False]
        result = fit_solubility_from_kmc(_sweep(P, C0, converged=conv))
        expected_S = 1e22 / math.sqrt(1000.0)
        assert result['S_mean'] == pytest.approx(expected_S, rel=1e-8)
        assert result['n_converged'] == 1

    def test_non_converged_appears_as_none_in_s_vals(self):
        P = [1000.0, 4000.0]
        C0 = [1e22, 9e22]
        conv = [True, False]
        result = fit_solubility_from_kmc(_sweep(P, C0, converged=conv))
        assert result['S_vals'][0] is not None
        assert result['S_vals'][1] is None

    def test_all_non_converged_gives_zero_mean(self):
        P = [1000.0]
        C0 = [1e22]
        result = fit_solubility_from_kmc(_sweep(P, C0, converged=[False]))
        assert result['S_mean'] == pytest.approx(0.0)
        assert result['n_converged'] == 0

    def test_single_converged_gives_zero_std(self):
        result = fit_solubility_from_kmc(_sweep([1000.0], [1e22]))
        assert result['S_std'] == pytest.approx(0.0)

    def test_zero_pressure_point_excluded_even_if_converged(self):
        # P=0 is skipped by `if conv and P > 0` even when converged=True
        result = fit_solubility_from_kmc(_sweep([0.0, 1000.0], [1e22, 1e22]))
        assert result['n_converged'] == 1
        assert result['S_vals'][0] is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. sieverts_solubility
# ═══════════════════════════════════════════════════════════════════════════

class TestSievertsSolubility:

    def test_exact_formula(self):
        dH, S0, T = 0.25, 1e28, 600.0
        expected = S0 * math.exp(-dH / (_KB_EV * T))
        assert sieverts_solubility(dH, S0, T) == pytest.approx(expected, rel=1e-8)

    def test_raises_for_zero_temperature(self):
        with pytest.raises(ValueError):
            sieverts_solubility(0.25, 1e28, 0.0)

    def test_zero_enthalpy_returns_s0(self):
        S0 = 1e28
        assert sieverts_solubility(0.0, S0, 600.0) == pytest.approx(S0, rel=1e-10)

    def test_higher_dh_gives_lower_s(self):
        S0 = 1e28
        S1 = sieverts_solubility(0.1, S0, 600.0)
        S2 = sieverts_solubility(0.4, S0, 600.0)
        assert S2 < S1

    def test_higher_temperature_gives_higher_s_for_endothermic(self):
        # positive ΔH_sol → higher T → larger S (less Boltzmann suppression)
        S0 = 1e28
        S_lo = sieverts_solubility(0.3, S0, 600.0)
        S_hi = sieverts_solubility(0.3, S0, 1200.0)
        assert S_hi > S_lo


# ═══════════════════════════════════════════════════════════════════════════
# 8. permeability
# ═══════════════════════════════════════════════════════════════════════════

class TestPermeability:

    def test_exact_product(self):
        D, S = 1e-10, 3e22
        assert permeability(D, S) == pytest.approx(D * S, rel=1e-10)

    def test_linear_in_diffusivity(self):
        S = 3e22
        assert permeability(2e-10, S) == pytest.approx(2.0 * permeability(1e-10, S), rel=1e-10)

    def test_linear_in_solubility(self):
        D = 1e-10
        assert permeability(D, 6e22) == pytest.approx(2.0 * permeability(D, 3e22), rel=1e-10)

    def test_returns_positive_float(self):
        assert permeability(1e-10, 1e22) > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 9. richardson_flux
# ═══════════════════════════════════════════════════════════════════════════

class TestRichardsonFlux:

    _PHI = 1e12   # atoms·m⁻¹·s⁻¹·Pa^(−½)
    _L   = 1e-3   # m

    def test_exact_formula(self):
        J = richardson_flux(self._PHI, 10000.0, 1000.0, self._L)
        expected = self._PHI * (math.sqrt(10000.0) - math.sqrt(1000.0)) / self._L
        assert J == pytest.approx(expected, rel=1e-8)

    def test_zero_low_pressure(self):
        J = richardson_flux(self._PHI, 10000.0, 0.0, self._L)
        expected = self._PHI * math.sqrt(10000.0) / self._L
        assert J == pytest.approx(expected, rel=1e-8)

    def test_raises_for_zero_thickness(self):
        with pytest.raises(ValueError):
            richardson_flux(self._PHI, 10000.0, 1000.0, 0.0)

    def test_raises_for_negative_thickness(self):
        with pytest.raises(ValueError):
            richardson_flux(self._PHI, 10000.0, 1000.0, -1e-3)

    def test_linear_in_permeability(self):
        J1 = richardson_flux(1e12, 10000.0, 0.0, self._L)
        J2 = richardson_flux(2e12, 10000.0, 0.0, self._L)
        assert J2 == pytest.approx(2.0 * J1, rel=1e-10)

    def test_inversely_proportional_to_thickness(self):
        J1 = richardson_flux(self._PHI, 10000.0, 0.0, 1e-3)
        J2 = richardson_flux(self._PHI, 10000.0, 0.0, 2e-3)
        assert J1 == pytest.approx(2.0 * J2, rel=1e-10)

    def test_negative_p_low_clamped_to_zero(self):
        # max(P_low_Pa, 0.0) clamps negative low-side pressure to zero
        J_neg  = richardson_flux(self._PHI, 10000.0, -500.0, self._L)
        J_zero = richardson_flux(self._PHI, 10000.0,    0.0, self._L)
        assert J_neg == pytest.approx(J_zero, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 10. resolve_nh_diffusivity — Part 2 <-> Part 3 handoff, no-placeholder logic
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveNhDiffusivity:

    def _write_fit(self, tmp_path, stem, n_h, D0=None, Ea=None, omit=False):
        nh_dir = tmp_path / 'results' / f'{stem}_{n_h}H'
        nh_dir.mkdir(parents=True)
        if not omit:
            payload = {}
            if D0 is not None:
                payload['D0_m2s'] = D0
            if Ea is not None:
                payload['E_D_eV'] = Ea
            (nh_dir / 'diffusivity_arrhenius.json').write_text(json.dumps(payload))
        return nh_dir

    def test_missing_file_not_ready(self, tmp_path):
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        assert res['ready'] is False
        assert res['D0_m2s'] is None and res['E_D_eV'] is None
        assert 'not found' in res['message']

    def test_missing_file_reports_correct_path(self, tmp_path):
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 3)
        expected = str(tmp_path / 'results' / 'ni_bulk_test_3H'
                        / 'diffusivity_arrhenius.json')
        assert res['diff_file'] == expected

    def test_valid_fit_is_ready(self, tmp_path):
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=1.2e-8, Ea=0.35)
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        assert res['ready'] is True
        assert res['D0_m2s'] == pytest.approx(1.2e-8)
        assert res['E_D_eV'] == pytest.approx(0.35)
        assert res['message'] is None

    def test_nan_d0_not_ready(self, tmp_path):
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=float('nan'), Ea=0.35)
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        assert res['ready'] is False
        assert 'no valid D0/Ea' in res['message']

    def test_nan_ea_not_ready(self, tmp_path):
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=1.2e-8, Ea=float('nan'))
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        assert res['ready'] is False

    def test_missing_d0_key_not_ready(self, tmp_path):
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=None, Ea=0.35)
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        assert res['ready'] is False

    def test_missing_ea_key_not_ready(self, tmp_path):
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=1.2e-8, Ea=None)
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        assert res['ready'] is False

    def test_dilute_limit_n_h_1_has_no_caveat(self, tmp_path):
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=1.2e-8, Ea=0.35)
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        assert res['dilute_note'] is None

    def test_non_dilute_n_h_gt_1_has_caveat(self, tmp_path):
        self._write_fit(tmp_path, 'ni_bulk_test', 3, D0=1.2e-8, Ea=0.35)
        res = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 3)
        assert res['dilute_note'] is not None
        assert 'dilute' in res['dilute_note'].lower()
        assert 'n_H=3' in res['dilute_note']

    def test_different_n_h_use_independent_paths(self, tmp_path):
        # Direct regression proof for the per-n_H Arrhenius overwrite bug:
        # two different n_H values must resolve to two different directories.
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=1.0e-8, Ea=0.30)
        self._write_fit(tmp_path, 'ni_bulk_test', 3, D0=2.0e-8, Ea=0.40)
        res1 = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        res3 = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 3)
        assert res1['nh_dir'] != res3['nh_dir']
        assert res1['diff_file'] != res3['diff_file']
        assert res1['D0_m2s'] != res3['D0_m2s']
        assert res1['E_D_eV'] != res3['E_D_eV']

    def test_one_n_h_missing_does_not_affect_the_other(self, tmp_path):
        # Only n_H=1 has a fit written; n_H=3 must independently report
        # "not found" rather than accidentally reading n_H=1's file.
        self._write_fit(tmp_path, 'ni_bulk_test', 1, D0=1.0e-8, Ea=0.30)
        res1 = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 1)
        res3 = resolve_nh_diffusivity(str(tmp_path), 'ni_bulk_test', 3)
        assert res1['ready'] is True
        assert res3['ready'] is False

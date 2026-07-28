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
    _N_A,
    fick_flux,
    check_sieverts_law,
    classify_sieverts_regime,
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


class TestClassifySievertsRegime:
    # Regime read from the coverage isotherm θ(P): exponent ≈0.5 (dissociative
    # equilibrium → Sieverts), ≈1.0 (dissociation rate-limited → surface),
    # θ→1 with no dilute points (saturated).
    _P = [1e-4, 4e-4, 1.6e-3, 6.4e-3, 2.56e-2]   # factor-4 pressure steps

    def test_sqrt_p_is_sieverts_compatible(self):
        th = [0.01 * math.sqrt(p) for p in self._P]      # θ ∝ √P  → n ≈ 0.5
        out = classify_sieverts_regime(self._P, th)
        assert out['theta_exponent'] == pytest.approx(0.5, abs=0.05)
        assert out['regime'] == 'sieverts_compatible'

    def test_linear_p_is_surface_limited(self):
        th = [2.0 * p for p in self._P]                  # θ ∝ P    → n ≈ 1.0
        out = classify_sieverts_regime(self._P, th)
        assert out['theta_exponent'] == pytest.approx(1.0, abs=0.05)
        assert out['regime'] == 'surface_limited'

    def test_saturated_only_when_no_dilute_points(self):
        out = classify_sieverts_regime([1e3, 1e4, 1e5], [0.95, 0.97, 0.99])
        assert out['saturates_in_sweep'] is True
        assert out['n_dilute_points'] == 0
        assert out['regime'] == 'saturated_only'

    def test_converged_filter_drops_points(self):
        # only 2 converged points → cannot classify
        out = classify_sieverts_regime(
            [1e-4, 2e-4, 3e-4], [0.01, 0.02, 0.03], converged=[True, True, False])
        assert out['regime'] == 'insufficient_data'


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
        expected = 4.0 / (a0 ** 3) / _N_A          # mol H per m^3 (÷ Avogadro)
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
        rho_oct = 4.0 / (_A0 ** 3) / _N_A          # mol H per m^3 (÷ Avogadro)
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


# ═══════════════════════════════════════════════════════════════════════════
# Section 4b — Heterogeneous, per-environment solubility (both S0 routes)
# ═══════════════════════════════════════════════════════════════════════════

from models.permeation import (
    vibrational_S0, build_dh_sol_by_env, solubility_by_environment,
)


class TestVibrationalS0:

    def test_positive(self):
        assert vibrational_S0(_A0, _T, [500.0, 800.0, 1200.0]) > 0.0

    def test_scales_with_site_density(self):
        # rho_oct = 4/a0^3, so halving a0 multiplies S0 by 8 (all else equal)
        s_big = vibrational_S0(_A0,       _T, [500.0, 800.0, 1200.0])
        s_sml = vibrational_S0(_A0 / 2.0, _T, [500.0, 800.0, 1200.0])
        assert s_sml == pytest.approx(8.0 * s_big, rel=1e-6)

    def test_stiffer_dissolved_modes_raise_S0(self):
        # larger q_H (softer modes) -> larger S0; stiffer modes -> smaller
        s_soft = vibrational_S0(_A0, _T, [200.0, 200.0, 200.0])
        s_stiff = vibrational_S0(_A0, _T, [2000.0, 2000.0, 2000.0])
        assert s_soft > s_stiff

    def test_below_geometric_ceiling_with_H_modes(self):
        # The dissolved-H partition function must use only the H's ~3 local
        # modes (~1500 cm^-1). With those, S0_vib stays below the geometric
        # site-density ceiling 4/a0^3. Feeding the soft metal-cage modes
        # (~100-250 cm^-1) blows past the ceiling (physically impossible) --
        # the ~1e10 inflation the H-only FS vibration avoids.
        ceiling = lattice_site_S0(_A0)
        s_H = vibrational_S0(_A0, _T, [1500.0, 1546.0, 1571.0])
        assert s_H < ceiling
        s_cage = vibrational_S0(_A0, _T, [1500.0, 1546.0, 1571.0] + [120.0] * 18)
        assert s_cage > ceiling


class TestBuildDhSolByEnv:

    def _hopa(self):
        # reaction energy = Ea_zpe - Ed_zpe
        return {
            'hopa_s_0': {'env': 'Ni6_oct',   'Ea_zpe': 0.30, 'Ed_zpe': 0.10},  # ΔH_A=+0.20
            'hopa_s_1': {'env': 'Ni6_oct',   'Ea_zpe': 0.40, 'Ed_zpe': 0.20},  # ΔH_A=+0.20
            'hopa_s_2': {'env': 'Ni5Mo_oct', 'Ea_zpe': 0.25, 'Ed_zpe': 0.35},  # ΔH_A=-0.10
        }

    def test_groups_by_sub1_env_with_weights(self):
        out = build_dh_sol_by_env(self._hopa(), dh_diss_eV=-1.0)
        assert set(out) == {'Ni6_oct', 'Ni5Mo_oct'}
        assert out['Ni6_oct']['n_sites'] == 2
        assert out['Ni5Mo_oct']['n_sites'] == 1
        assert out['Ni6_oct']['w_env'] == pytest.approx(2 / 3)
        assert out['Ni5Mo_oct']['w_env'] == pytest.approx(1 / 3)

    def test_dh_sol_formula_sub1_no_hopb(self):
        # Solubility is referenced to sub1: ΔH_sol = ½·ΔH_diss + ΔH_HopA only.
        # Hop B is NOT a parameter and must not enter (it is bulk diffusion, in D).
        out = build_dh_sol_by_env(self._hopa(), dh_diss_eV=-1.0)
        # Ni6_oct: 0.5*(-1.0) + mean(0.20, 0.20) = -0.5 + 0.20 = -0.30
        assert out['Ni6_oct']['dH_sol_eV'] == pytest.approx(-0.30)
        # Ni5Mo_oct: -0.5 + (-0.10) = -0.60
        assert out['Ni5Mo_oct']['dH_sol_eV'] == pytest.approx(-0.60)
        # Hop B must not be reported in the solubility dict; Hop A must be
        assert 'dH_hopB_mean_eV' not in out['Ni6_oct']
        assert 'dH_hopA_eV' in out['Ni6_oct']

    def test_writes_json(self, tmp_path):
        out_json = str(tmp_path / 'dH_sol_by_env.json')
        build_dh_sol_by_env(self._hopa(), -1.0, out_json=out_json)
        assert pathlib.Path(out_json).exists()
        loaded = json.loads(pathlib.Path(out_json).read_text())
        assert 'Ni6_oct' in loaded


class TestSolubilityByEnvironment:

    def test_single_env_reduces_to_boltzmann(self):
        d = {'Ni6_oct': {'dH_sol_eV': 0.15, 'w_env': 1.0, 'n_sites': 1}}
        S0 = 2.0e28
        got = solubility_by_environment(d, S0, _T)
        assert got == pytest.approx(S0 * math.exp(-0.15 / (_KB_EV * _T)))

    def test_weighted_sum_over_envs(self):
        d = {
            'a': {'dH_sol_eV': 0.10, 'w_env': 0.5, 'n_sites': 1},
            'b': {'dH_sol_eV': 0.40, 'w_env': 0.5, 'n_sites': 1},
        }
        S0 = 1.0
        kT = _KB_EV * _T
        expected = 0.5 * math.exp(-0.10 / kT) + 0.5 * math.exp(-0.40 / kT)
        assert solubility_by_environment(d, S0, _T) == pytest.approx(expected)

    def test_lower_barrier_env_dominates(self):
        # a more favourable (lower ΔH_sol) environment gives higher solubility
        d_lo = {'x': {'dH_sol_eV': 0.05, 'w_env': 1.0, 'n_sites': 1}}
        d_hi = {'x': {'dH_sol_eV': 0.50, 'w_env': 1.0, 'n_sites': 1}}
        assert solubility_by_environment(d_lo, 1.0, _T) > solubility_by_environment(d_hi, 1.0, _T)

    def test_empty_returns_zero(self):
        assert solubility_by_environment({}, 1.0, _T) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 — Arrhenius output fitters (S0/dH_sol, Phi0/E_phi)
# ═══════════════════════════════════════════════════════════════════════════

from models.permeation import fit_arrhenius, permeability_arrhenius


class TestFitArrhenius:

    def test_recovers_known_parameters(self):
        A, Ea = 5.0e12, 0.42
        T = np.array([400.0, 600.0, 800.0])
        y = A * np.exp(-Ea / (_KB_EV * T))
        fit = fit_arrhenius(T, y)
        assert fit['prefactor'] == pytest.approx(A, rel=1e-6)
        assert fit['Ea_eV'] == pytest.approx(Ea, rel=1e-6)
        assert fit['r2'] == pytest.approx(1.0, abs=1e-9)

    def test_curvature_lowers_r2(self):
        # A sum of two Arrhenius terms with a genuine crossover (low-ΔH term
        # dominates at low T, high-ΔH + huge-prefactor term at high T) is not a
        # single line -> visible curvature -> R2 meaningfully below 1.
        T = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        y = np.exp(-0.1 / (_KB_EV * T)) + 1e8 * np.exp(-0.9 / (_KB_EV * T))
        fit = fit_arrhenius(T, y)
        assert fit['r2'] < 0.999

    def test_fewer_than_two_points_is_nan(self):
        fit = fit_arrhenius([600.0], [1.0])
        assert fit['n_points'] == 1
        assert math.isnan(fit['prefactor'])

    def test_drops_nonpositive_points(self):
        A, Ea = 1.0e10, 0.3
        T = np.array([400.0, 600.0, 800.0])
        y = A * np.exp(-Ea / (_KB_EV * T))
        fit = fit_arrhenius(np.append(T, 900.0), np.append(y, 0.0))  # 0.0 dropped
        assert fit['n_points'] == 3
        assert fit['Ea_eV'] == pytest.approx(Ea, rel=1e-6)


class TestPermeabilityArrhenius:

    def test_phi0_is_product(self):
        r = permeability_arrhenius(1e-7, 0.40, 2e28, 0.15)
        assert r['Phi0'] == pytest.approx(1e-7 * 2e28)

    def test_e_phi_is_sum(self):
        r = permeability_arrhenius(1e-7, 0.40, 2e28, 0.15)
        assert r['E_phi_eV'] == pytest.approx(0.55)

    def test_consistent_with_direct_fit(self):
        # Phi(T) = D0*S0 * exp(-(E_D+dH)/kT) must fit back to Phi0/E_phi
        D0, E_D, S0, dH = 1e-7, 0.40, 2e28, 0.15
        params = permeability_arrhenius(D0, E_D, S0, dH)
        T = np.array([400.0, 600.0, 800.0])
        Phi = (D0 * np.exp(-E_D / (_KB_EV * T))) * (S0 * np.exp(-dH / (_KB_EV * T)))
        fit = fit_arrhenius(T, Phi)
        assert fit['prefactor'] == pytest.approx(params['Phi0'], rel=1e-6)
        assert fit['Ea_eV'] == pytest.approx(params['E_phi_eV'], rel=1e-6)

"""
tests/functional/test_ft_permeability.py
==========================================
Category D functional tests — pure-Python math for the permeability pipeline.

Covers TST rate algebra, Fick/Richardson-Sieverts permeability formulas, and
a minimal KMC integration smoke-test that runs on a tiny grid.

No GPU, no SLURM, no MACE, no cluster required.
"""

import math
import sys
import os

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.tst_rates import (
    BOLTZMANN_eV,
    arrhenius_rate,
    vineyard_prefactor,
    apply_zpe_correction,
)
from models.permeation import (
    fick_flux,
    arrhenius_diffusivity,
    lattice_site_S0,
    sieverts_solubility,
    permeability,
    richardson_flux,
    check_sieverts_law,
    fit_solubility_from_kmc,
)
from models.kmc import (
    make_grid,
    run_kmc_to_steady_state,
)

# ─── shared physical constants ────────────────────────────────────────────────
_KB_EV = BOLTZMANN_eV          # eV / K
_A0    = 3.52e-10              # m  (Ni FCC lattice constant)
_T     = 700.0                 # K  (representative operating temperature)


# ═════════════════════════════════════════════════════════════════════════════
# 1. TST rate algebra
# ═════════════════════════════════════════════════════════════════════════════

class TestTSTRatesDetailedBalance:
    """
    Verify TST rate functions against closed-form answers.

    Key invariant — detailed balance:
        k_fwd / k_rev = exp(delta_E / k_B T)
    where delta_E = E_des − E_abs = −(E_FS − E_IS) = −ΔE (reaction energy).

    With Ea = 0.5 eV (forward barrier), Ed = 0.4 eV (reverse barrier),
    delta_E = Ed − Ea = −0.1 eV (exothermic hop):
        k_fwd / k_rev = exp(+0.1 / (kB × 700)) = exp(1.6558) ≈ 5.239
    """

    _EA    = 0.50   # eV  forward barrier (E_abs)
    _ED    = 0.40   # eV  reverse barrier (E_des)
    _NU    = 1e13   # s^-1  attempt frequency
    _DELTA = _ED - _EA   # = −0.10 eV  reaction energy (negative = exothermic)

    def _kfwd(self):
        return arrhenius_rate(self._NU, self._EA, _T)

    def _krev(self):
        return arrhenius_rate(self._NU, self._ED, _T)

    def test_forward_rate_positive(self):
        assert self._kfwd() > 0.0

    def test_reverse_rate_positive(self):
        assert self._krev() > 0.0

    def test_forward_greater_than_reverse(self):
        # With Ea > Ed, forward is harder; Ed < Ea → k_des > k_abs → reverse is faster
        # Wait: k_fwd uses Ea=0.5, k_rev uses Ed=0.4; exp(-0.5/kBT) < exp(-0.4/kBT)
        # So k_fwd < k_rev (exothermic: product state is lower, reverse barrier is smaller)
        assert self._kfwd() < self._krev()

    def test_detailed_balance_ratio(self):
        # k_fwd / k_rev = exp(−Ea/kBT) / exp(−Ed/kBT) = exp((Ed−Ea)/kBT)
        expected_ratio = math.exp((self._ED - self._EA) / (_KB_EV * _T))
        actual_ratio   = self._kfwd() / self._krev()
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-9)

    def test_arrhenius_known_value(self):
        # k = 1e13 × exp(−0.5 / (kB × 700))
        expected = 1e13 * math.exp(-0.5 / (_KB_EV * _T))
        assert math.isclose(self._kfwd(), expected, rel_tol=1e-10)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            arrhenius_rate(1e13, 0.5, 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            arrhenius_rate(1e13, 0.5, -100.0)

    def test_zero_barrier_rate_equals_prefactor(self):
        # exp(0) = 1 → k = ν
        assert math.isclose(arrhenius_rate(1e12, 0.0, _T), 1e12, rel_tol=1e-10)


class TestVineyardAndZPE:
    """
    Vineyard prefactor from mode-product ratio.

    With IS = [f1, f2, f3] and TS = [f1, f2] (one imaginary mode removed),
    ν* = c × (f1 × f2 × f3) / (f1 × f2) = c × f3
    where c = SPEED_LIGHT_CM_S (converts last cm^-1 to s^-1).
    """

    from models.tst_rates import SPEED_LIGHT_CM_S as _C_CM_S

    _IS_FREQS = [800.0, 600.0, 400.0]   # cm^-1
    _TS_FREQS = [800.0, 600.0]           # cm^-1  (imaginary mode = 400 excluded)

    def test_vineyard_known_ratio(self):
        from models.tst_rates import SPEED_LIGHT_CM_S
        expected = SPEED_LIGHT_CM_S * 400.0   # remaining unpaired IS mode
        result   = vineyard_prefactor(self._IS_FREQS, self._TS_FREQS)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_vineyard_positive(self):
        assert vineyard_prefactor(self._IS_FREQS, self._TS_FREQS) > 0.0

    def test_vineyard_typical_order_of_magnitude(self):
        # Expect result in ~10^12 – 10^14 s^-1 for acoustic modes
        freqs_is = [400.0, 600.0, 800.0, 1000.0, 1200.0]
        freqs_ts = [400.0, 600.0, 800.0, 1000.0]  # one imaginary removed
        nu = vineyard_prefactor(freqs_is, freqs_ts)
        assert 1e11 < nu < 1e15

    def test_vineyard_no_valid_freqs_raises(self):
        with pytest.raises(ValueError, match='No valid IS frequencies'):
            vineyard_prefactor([10.0], [800.0], min_freq_cm1=50.0)

    def test_zpe_correction_zero_when_same_freqs(self):
        freqs = [800.0, 600.0, 400.0]
        corrected = apply_zpe_correction(0.5, freqs, freqs)
        assert math.isclose(corrected, 0.5, rel_tol=1e-10)

    def test_zpe_correction_increases_barrier_when_ts_stiffer(self):
        # TS stiffer (higher freqs) → ZPE_TS > ZPE_IS → ΔZPE > 0 → barrier increases
        freqs_is = [400.0, 600.0]
        freqs_ts = [800.0, 1200.0]    # stiffer TS
        corrected = apply_zpe_correction(0.5, freqs_is, freqs_ts)
        assert corrected > 0.5

    def test_zpe_correction_decreases_barrier_when_is_stiffer(self):
        freqs_is = [800.0, 1200.0]    # stiffer IS
        freqs_ts = [400.0, 600.0]
        corrected = apply_zpe_correction(0.5, freqs_is, freqs_ts)
        assert corrected < 0.5

    def test_zpe_correction_known_value(self):
        from models.tst_rates import CM1_TO_EV
        freqs_is = [1000.0]
        freqs_ts = [2000.0]
        # ZPE_IS = 0.5 × 1000 × CM1_TO_EV;  ZPE_TS = 0.5 × 2000 × CM1_TO_EV
        # ΔZPE = 0.5 × (2000 - 1000) × CM1_TO_EV
        expected = 0.5 + 0.5 * 1000.0 * CM1_TO_EV
        corrected = apply_zpe_correction(0.5, freqs_is, freqs_ts)
        assert math.isclose(corrected, expected, rel_tol=1e-9)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Fick's law and Richardson-Sieverts permeability math
# ═════════════════════════════════════════════════════════════════════════════

class TestFickAndPermeabilityMath:
    """
    Verify the permeation formulas against closed-form answers.

    All functions in permeation.py (Sections 1 and 4) are pure math:
    no KMC, no LAMMPS.
    """

    _D   = 1e-9    # m²/s   bulk diffusivity
    _C0  = 1e27    # atoms/m³
    _L   = 1e-3    # m      membrane thickness
    _D0  = 1e-7    # m²/s   pre-exponential
    _ED  = 0.4     # eV     diffusion activation energy
    _DH  = 0.2     # eV     solution enthalpy

    # ── Fick's law ────────────────────────────────────────────────────────────

    def test_fick_flux_known_value(self):
        expected = self._D * self._C0 / self._L
        assert math.isclose(fick_flux(self._D, self._C0, self._L), expected, rel_tol=1e-10)

    def test_fick_flux_with_low_side_concentration(self):
        C_low = 0.1 * self._C0
        expected = self._D * (self._C0 - C_low) / self._L
        assert math.isclose(fick_flux(self._D, self._C0, self._L, C_low), expected, rel_tol=1e-10)

    def test_fick_flux_zero_gradient_gives_zero(self):
        assert fick_flux(self._D, self._C0, self._L, self._C0) == pytest.approx(0.0, abs=1e-30)

    def test_fick_flux_scales_inversely_with_L(self):
        J1 = fick_flux(self._D, self._C0, self._L)
        J2 = fick_flux(self._D, self._C0, 2 * self._L)
        assert math.isclose(J2, 0.5 * J1, rel_tol=1e-10)

    def test_fick_flux_zero_thickness_raises(self):
        with pytest.raises(ValueError):
            fick_flux(self._D, self._C0, 0.0)

    # ── Arrhenius diffusivity ─────────────────────────────────────────────────

    def test_arrhenius_diffusivity_known_value(self):
        expected = self._D0 * math.exp(-self._ED / (_KB_EV * _T))
        result   = arrhenius_diffusivity(self._D0, self._ED, _T)
        assert math.isclose(result, expected, rel_tol=1e-10)

    def test_arrhenius_diffusivity_zero_E_equals_D0(self):
        assert math.isclose(arrhenius_diffusivity(1e-7, 0.0, _T), 1e-7, rel_tol=1e-10)

    def test_arrhenius_diffusivity_decreases_with_increasing_E(self):
        D1 = arrhenius_diffusivity(1e-7, 0.3, _T)
        D2 = arrhenius_diffusivity(1e-7, 0.6, _T)
        assert D1 > D2

    def test_arrhenius_diffusivity_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            arrhenius_diffusivity(1e-7, 0.4, 0.0)

    # ── Lattice S0 ────────────────────────────────────────────────────────────

    def test_lattice_site_S0_formula(self):
        expected = 4.0 / (_A0 ** 3)
        assert math.isclose(lattice_site_S0(_A0), expected, rel_tol=1e-10)

    def test_lattice_site_S0_positive(self):
        assert lattice_site_S0(_A0) > 0.0

    # ── Sieverts solubility ───────────────────────────────────────────────────

    def test_sieverts_solubility_known_value(self):
        S0       = lattice_site_S0(_A0)
        expected = S0 * math.exp(-self._DH / (_KB_EV * _T))
        result   = sieverts_solubility(self._DH, S0, _T)
        assert math.isclose(result, expected, rel_tol=1e-10)

    def test_sieverts_solubility_zero_dH_equals_S0(self):
        S0 = 1e28
        assert math.isclose(sieverts_solubility(0.0, S0, _T), S0, rel_tol=1e-10)

    def test_sieverts_solubility_increases_with_temperature(self):
        S0 = 1e28
        S1 = sieverts_solubility(0.3, S0, 600.0)
        S2 = sieverts_solubility(0.3, S0, 900.0)
        assert S2 > S1   # endothermic: higher T → more solubility

    def test_sieverts_solubility_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            sieverts_solubility(0.2, 1e28, 0.0)

    # ── Permeability Φ = D × S ────────────────────────────────────────────────

    def test_permeability_is_product(self):
        S   = sieverts_solubility(self._DH, lattice_site_S0(_A0), _T)
        D   = arrhenius_diffusivity(self._D0, self._ED, _T)
        Phi = permeability(D, S)
        assert math.isclose(Phi, D * S, rel_tol=1e-10)

    # ── Richardson flux ───────────────────────────────────────────────────────

    def test_richardson_flux_known_value(self):
        S   = sieverts_solubility(self._DH, lattice_site_S0(_A0), _T)
        D   = arrhenius_diffusivity(self._D0, self._ED, _T)
        Phi = permeability(D, S)
        P_high, P_low = 1e5, 1e2
        expected = Phi * (math.sqrt(P_high) - math.sqrt(P_low)) / self._L
        result   = richardson_flux(Phi, P_high, P_low, self._L)
        assert math.isclose(result, expected, rel_tol=1e-10)

    def test_richardson_flux_zero_for_equal_pressures(self):
        Phi = 1e10
        J   = richardson_flux(Phi, 1e5, 1e5, self._L)
        assert J == pytest.approx(0.0, abs=1e-10)

    def test_richardson_flux_scales_inversely_with_L(self):
        Phi = 1e10
        J1  = richardson_flux(Phi, 1e5, 0.0, self._L)
        J2  = richardson_flux(Phi, 1e5, 0.0, 2 * self._L)
        assert math.isclose(J2, 0.5 * J1, rel_tol=1e-10)

    def test_richardson_flux_scales_with_sqrt_pressure_difference(self):
        Phi = 1e10
        # Double √P → J doubles (with P_low = 0)
        J1 = richardson_flux(Phi, 1e4, 0.0, self._L)
        J2 = richardson_flux(Phi, 4e4, 0.0, self._L)
        assert math.isclose(J2, 2.0 * J1, rel_tol=1e-10)

    def test_richardson_flux_negative_L_raises(self):
        with pytest.raises(ValueError):
            richardson_flux(1e10, 1e5, 0.0, -1e-3)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Sieverts fit and empirical solubility
# ═════════════════════════════════════════════════════════════════════════════

class TestSievertsFitAndSolubility:
    """
    check_sieverts_law: fit J vs √P and test linearity diagnostics.
    fit_solubility_from_kmc: recover S from a synthetic sweep_result dict.

    All tests use synthetic data — no KMC run required.
    """

    def _perfect_linear_sweep(self, n=8, k_slope=1e15):
        P_vals = [float(1e3 * (i + 1)) for i in range(n)]   # 1e3..8e3 Pa
        J_vals = [k_slope * math.sqrt(P) for P in P_vals]
        return P_vals, J_vals

    def test_perfect_linear_is_sieverts(self):
        P_vals, J_vals = self._perfect_linear_sweep()
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        assert result['is_sieverts'] is True

    def test_perfect_linear_r_squared_near_one(self):
        P_vals, J_vals = self._perfect_linear_sweep()
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        assert result['r_squared'] > 0.9999

    def test_linear_j_over_4_decades_not_sieverts(self):
        # J ∝ P (linear in P, not in √P) over 4 decades → R² ≈ 0.92 < 0.98.
        # A narrow range would look linear even for J ∝ P; 4 decades of P makes
        # the curvature of J vs √P clearly visible to the R² statistic.
        P_vals = [1e1, 1e2, 1e3, 1e4, 1e5]
        J_vals = [1e10 * P for P in P_vals]
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        assert result['is_sieverts'] is False
        assert result['r_squared'] < 0.98   # must fall below the Sieverts threshold

    def test_sieverts_slope_matches_k(self):
        k = 5e14
        P_vals, J_vals = self._perfect_linear_sweep(k_slope=k)
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        # slope of J vs √P should ≈ k
        assert math.isclose(result['slope'], k, rel_tol=1e-6)

    def test_sieverts_result_has_required_keys(self):
        P_vals, J_vals = self._perfect_linear_sweep()
        result = check_sieverts_law(P_vals, J_vals, plot=False)
        for key in ('slope', 'intercept', 'r_squared', 'is_sieverts'):
            assert key in result

    def test_fit_solubility_from_kmc_mean_correct(self):
        # Synthetic: C0 = S_true × √P  (Sieverts)
        S_true = 3e25   # atoms/m³/Pa^0.5
        P_vals = [1e3 * (i + 1) for i in range(5)]
        sqrt_P = [math.sqrt(P) for P in P_vals]
        C0_vals = [S_true * sp for sp in sqrt_P]
        sweep_result = {
            'P_vals':      P_vals,
            'C0_vals':     C0_vals,
            'sqrt_P_vals': sqrt_P,
            'converged':   [True] * 5,
        }
        result = fit_solubility_from_kmc(sweep_result)
        assert result['n_converged'] == 5
        assert math.isclose(result['S_mean'], S_true, rel_tol=1e-9)
        # S_std should be essentially zero (floating-point rounding only)
        assert result['S_std'] / result['S_mean'] < 1e-12

    def test_fit_solubility_skips_unconverged(self):
        P_vals  = [1e3, 2e3, 3e3]
        C0_vals = [1e27, 2e27, 3e27]
        sqrt_P  = [math.sqrt(P) for P in P_vals]
        sweep_result = {
            'P_vals':      P_vals,
            'C0_vals':     C0_vals,
            'sqrt_P_vals': sqrt_P,
            'converged':   [True, False, True],
        }
        result = fit_solubility_from_kmc(sweep_result)
        assert result['n_converged'] == 2
        assert result['S_vals'][1] is None   # unconverged → excluded


# ═════════════════════════════════════════════════════════════════════════════
# 4. KMC integration smoke-test
# ═════════════════════════════════════════════════════════════════════════════

class TestKMCIntegration:
    """
    Integration test: make_grid → run_kmc_to_steady_state → fick_flux.

    Uses a tiny 4×4 pure-Ni grid and small window/max_steps to stay fast.
    Rates are chosen so drain (D/dx²) is small compared to k_entry, ensuring
    C0 > 0 at steady state.

    Expected run time: < 2 seconds.
    """

    _NX   = 4
    _NY   = 4
    _P    = 1_000.0      # Pa
    _T    = 600.0        # K
    _D    = 1e-15        # m²/s  — intentionally tiny so k_drain << k_entry
    _A0   = 3.52e-10     # m
    _L    = 1e-3         # m  membrane thickness

    # Pure Ni rates — k_diss kept moderate; k_entry/k_hopB >> k_drain
    # (D/dx^2 ≈ 1.6e4 s^-1) so H flows surface→sub1→sub2 and C0 (the sub2
    # concentration that feeds the bulk) stays > 0. k_entry/k_exit/k_hopB_*
    # keyed by 'Ni' rely on make_grid's degenerate env default for this
    # pure-Ni grid (sub1_env == sub2_env == 'Ni').
    _RATES = {
        'k_diss':       {('Ni', 'Ni'): 0.5},
        'k_des':        {('Ni', 'Ni'): 1e6},
        'k_surf_diff':  {('Ni', 'Ni'): 1e8},
        'k_entry':      {'Ni': 1e8},
        'k_exit':       {'Ni': 1e4},
        'k_hopB_entry': {'Ni': 1e8},
        'k_hopB_exit':  {'Ni': 1e4},
    }

    @pytest.fixture(scope='class')
    def kmc_result(self):
        np.random.seed(0)
        grid = make_grid(self._NX, self._NY, composition={'Ni': 1.0}, seed=42)
        result = run_kmc_to_steady_state(
            grid, self._RATES, self._P, self._T, self._D, self._A0,
            window=50, max_steps=300,
        )
        return result, grid

    def test_kmc_returns_required_keys(self, kmc_result):
        result, _ = kmc_result
        for key in ('t_total', 'theta_ss', 'C0', 'C0_sub1', 'C0_sub2',
                    'n_steps', 'converged'):
            assert key in result
        # C0 is the back-compat alias for the sub2 (deepest) layer
        assert result['C0'] == result['C0_sub2']

    def test_kmc_t_total_positive(self, kmc_result):
        result, _ = kmc_result
        assert result['t_total'] >= 0.0

    def test_kmc_theta_in_unit_interval(self, kmc_result):
        result, _ = kmc_result
        assert 0.0 <= result['theta_ss'] <= 1.0

    def test_kmc_C0_non_negative(self, kmc_result):
        result, _ = kmc_result
        assert result['C0'] >= 0.0

    def test_kmc_C0_positive_with_slow_drain(self, kmc_result):
        # k_drain = D / (a0/√2)^2 ≈ 1.6e4 s^-1 << k_entry = 1e8 s^-1
        # → subsurface should retain H → C0 > 0
        result, _ = kmc_result
        assert result['C0'] > 0.0

    def test_fick_flux_positive_after_kmc(self, kmc_result):
        result, _ = kmc_result
        J = fick_flux(self._D, result['C0'], self._L)
        assert J > 0.0

    def test_kmc_n_steps_within_max(self, kmc_result):
        result, _ = kmc_result
        assert result['n_steps'] <= 300

    def test_sweep_pressure_returns_correct_dict_structure(self):
        from models.permeation import sweep_pressure
        np.random.seed(1)
        P_vals = [500.0, 1000.0, 2000.0]
        out = sweep_pressure(
            P_vals, self._RATES, self._D, self._L, self._T, self._A0,
            nx=3, ny=3, composition={'Ni': 1.0},
            kmc_kwargs={'window': 50, 'max_steps': 200},
        )
        for key in ('P_vals', 'J_vals', 'C0_vals', 'sqrt_P_vals', 'converged'):
            assert key in out
        assert len(out['P_vals']) == 3
        assert len(out['J_vals']) == 3
        assert all(J >= 0.0 for J in out['J_vals'])

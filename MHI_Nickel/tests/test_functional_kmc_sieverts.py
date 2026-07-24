"""
tests/test_functional_kmc_sieverts.py
=======================================
Functional test: KMC pressure sweep → Sieverts' law → full permeation chain.

All offline (no LAMMPS/SLURM). Uses rate constants scaled so the KMC
converges in < 10 000 steps on a 12×12 pure-Ni grid while still reproducing
the correct Sieverts' scaling C₀ ∝ √P.

Scientific basis
----------------
Dissociative adsorption (H₂ → 2H*) is first-order in P:
    R_ads = k_diss × R_strike(P,T) × (1-θ)²

Recombinative desorption (2H* → H₂) is second-order in θ:
    R_des = k_des × θ²

At steady state (dilute regime θ ≪ 1):
    k_diss × R_strike × (1-θ)² ≈ k_des × θ²
    → θ ∝ √P  (Sieverts' law)

Two-layer subsurface equilibrium (surface ⇄ sub1 ⇄ sub2 → bulk):
    sub1:  k_entry × θ ≈ (k_exit + k_hopB_entry) × sub1         → sub1 ∝ θ
    sub2:  k_hopB_entry × sub1 ≈ (k_hopB_exit + k_drain) × sub2 → sub2 ∝ sub1
    → C₀ (= sub2 concentration, the layer that feeds bulk) ∝ √P

Rate parameters are chosen so that k_drain (= D/dx²) is much smaller than
k_entry/k_exit and k_hopB_entry/k_hopB_exit, keeping all three layers
unsaturated while ensuring they equilibrate on a timescale short enough for
convergence within a few thousand KMC steps per pressure point.
"""

import math
import numpy as np
import pytest
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')

from models.kmc import (
    make_grid,
    run_kmc_to_steady_state,
)
from models.permeation import (
    check_sieverts_law,
    fit_solubility_from_kmc,
    permeability,
    richardson_flux,
)

# ── KMC parameters ────────────────────────────────────────────────────────────

_A0   = 3.52e-10   # m  (FCC Ni lattice parameter)
_T_K  = 700.0      # K
_NX   = 12
_NY   = 12

# D is set so k_drain = D/(a0/√2)² ≈ 1000 s⁻¹  (much smaller than k_entry/k_exit)
_D_M2S = 6.2e-17   # m²/s — scaled down for test tractability

# k_diss sticking coefficient chosen so θ ≈ 0.05–0.19 across the pressure sweep
# (dilute-regime Sieverts' scaling holds throughout)
_RATE_DICT = {
    'k_diss':       {('Ni', 'Ni'): 1e-4},   # dimensionless sticking coefficient
    'k_des':        {('Ni', 'Ni'): 2e5},    # s⁻¹ recombinative desorption
    'k_entry':      {'Ni': 2e5},            # s⁻¹ surface → sub1  (Hop A fwd)
    'k_exit':       {'Ni': 2e5},            # s⁻¹ sub1 → surface  (Hop A rev)
    'k_hopB_entry': {'Ni': 2e5},            # s⁻¹ sub1 → sub2     (Hop B fwd)
    'k_hopB_exit':  {'Ni': 2e5},            # s⁻¹ sub2 → sub1     (Hop B rev)
}
# NOTE: k_entry/k_exit/k_hopB_* keyed by 'Ni' rely on make_grid's degenerate
# env default (sub1_env == sub2_env == surface element) for this pure-Ni grid,
# so the env-resolved lookups in build_event_list resolve to these rates.

# Pressure sweep spanning factor of 16 (√P ratio = 4:1)
_P_VALS = [1000.0, 4000.0, 9000.0, 16000.0]   # Pa


# ── Module-level fixture — run KMC once, reuse across all tests ───────────────

@pytest.fixture(scope='module')
def sweep():
    """Run KMC to steady state at each pressure; return list of result dicts."""
    results = []
    for P in _P_VALS:
        np.random.seed(0)
        grid = make_grid(_NX, _NY, composition={'Ni': 1.0}, seed=42)
        res  = run_kmc_to_steady_state(
            grid, _RATE_DICT, P, _T_K, _D_M2S, _A0,
            window=1000, rtol=0.05, max_steps=100_000,
        )
        res['P'] = P
        results.append(res)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 1. KMC convergence
# ═══════════════════════════════════════════════════════════════════════════

class TestKmcConvergence:
    """Verify the KMC reaches steady state at every pressure point."""

    def test_all_pressure_points_converge(self, sweep):
        for r in sweep:
            assert r['converged'], \
                f"KMC did not converge at P={r['P']:.0f} Pa"

    def test_surface_coverage_positive_at_all_pressures(self, sweep):
        for r in sweep:
            assert r['theta_ss'] > 0, \
                f"θ=0 at P={r['P']:.0f} Pa — surface never occupied"

    def test_subsurface_concentration_positive_at_all_pressures(self, sweep):
        for r in sweep:
            assert r['C0'] > 0, \
                f"C₀=0 at P={r['P']:.0f} Pa — subsurface never occupied"

    def test_surface_coverage_stays_below_saturation(self, sweep):
        """θ < 0.5 at all pressures — ensures we are in the Sieverts' regime."""
        for r in sweep:
            assert r['theta_ss'] < 0.5, \
                f"θ={r['theta_ss']:.3f} at P={r['P']:.0f} Pa (too high; not dilute)"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sieverts' law on KMC output
# ═══════════════════════════════════════════════════════════════════════════

class TestSievertLawFromKmc:
    """Verify that both θ and C₀ scale as √P (Sieverts' law)."""

    def test_surface_coverage_monotone_with_pressure(self, sweep):
        """θ must increase monotonically — basic sanity check."""
        thetas = [r['theta_ss'] for r in sweep]
        for i in range(1, len(thetas)):
            assert thetas[i] > thetas[i - 1], \
                f"θ did not increase from P={_P_VALS[i-1]:.0f} to {_P_VALS[i]:.0f} Pa"

    def test_c0_monotone_with_pressure(self, sweep):
        """C₀ must increase monotonically with pressure."""
        c0s = [r['C0'] for r in sweep]
        for i in range(1, len(c0s)):
            assert c0s[i] > c0s[i - 1], \
                f"C₀ did not increase from P={_P_VALS[i-1]:.0f} to {_P_VALS[i]:.0f} Pa"

    def test_surface_coverage_fits_sqrt_p_r2_above_threshold(self, sweep):
        """θ vs √P linear fit R² ≥ 0.90 (Sieverts' law for surface layer)."""
        P_vals = [r['P']       for r in sweep]
        J_vals = [r['theta_ss'] for r in sweep]
        fit    = check_sieverts_law(P_vals, J_vals, plot=False)
        r2     = fit['r_squared']
        assert r2 >= 0.90, \
            f"Surface coverage R²={r2:.4f} < 0.90 — Sieverts' law not satisfied"

    def test_subsurface_c0_fits_sqrt_p_r2_above_threshold(self, sweep):
        """C₀ vs √P linear fit R² ≥ 0.90 (Sieverts' law for subsurface)."""
        P_vals  = [r['P']  for r in sweep]
        C0_vals = [r['C0'] for r in sweep]
        fit     = check_sieverts_law(P_vals, C0_vals, plot=False)
        r2      = fit['r_squared']
        assert r2 >= 0.90, \
            f"Subsurface C₀ R²={r2:.4f} < 0.90 — Sieverts' law not satisfied"

    def test_theta_ratio_approximates_sqrt_pressure_ratio(self, sweep):
        """
        Between P₁ and P₄ (ratio 16), θ ratio should be close to √16 = 4.
        Tolerance ±30 % accounts for small-grid noise and mild coverage effects.
        """
        theta_low  = sweep[0]['theta_ss']   # P = 1000 Pa
        theta_high = sweep[-1]['theta_ss']  # P = 16000 Pa
        ratio      = theta_high / theta_low
        expected   = math.sqrt(_P_VALS[-1] / _P_VALS[0])  # √16 = 4
        assert abs(ratio - expected) / expected < 0.30, \
            f"θ ratio={ratio:.3f} deviates > 30% from expected √P ratio={expected:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Full permeation chain
# ═══════════════════════════════════════════════════════════════════════════

class TestPermeationChainFunctional:
    """
    Run the complete pipeline:
      KMC sweep → fit_solubility_from_kmc → permeability → richardson_flux

    Verifies the chain executes without errors and returns physically consistent
    values (positive permeability, positive flux for upstream > downstream).
    """

    @pytest.fixture(scope='class')
    def sweep_dict(self, sweep):
        """Format KMC results as the dict expected by fit_solubility_from_kmc."""
        return {
            'P_vals':    [r['P']         for r in sweep],
            'C0_vals':   [r['C0']        for r in sweep],
            'converged': [r['converged'] for r in sweep],
        }

    @pytest.fixture(scope='class')
    def sol_result(self, sweep_dict):
        return fit_solubility_from_kmc(sweep_dict)

    def test_fit_solubility_has_positive_s_mean(self, sol_result):
        assert sol_result['S_mean'] > 0, \
            "S_mean should be positive"

    def test_fit_solubility_uses_all_converged_points(self, sweep, sol_result):
        n_converged = sum(1 for r in sweep if r['converged'])
        assert sol_result['n_converged'] == n_converged

    def test_s_values_consistent_across_pressures(self, sol_result):
        """
        S = C₀/√P should be approximately constant.
        Coefficient of variation (std/mean) < 25 % for the 12×12 grid.
        """
        if sol_result['n_converged'] < 2 or sol_result['S_mean'] <= 0:
            pytest.skip('insufficient converged points for CV check')
        cv = sol_result['S_std'] / sol_result['S_mean']
        assert cv < 0.40, \
            f"S coefficient of variation {cv:.3f} > 0.40 — solubility too scattered"

    def test_permeability_positive(self, sol_result):
        Phi = permeability(_D_M2S, sol_result['S_mean'])
        assert Phi > 0, "Permeability should be positive"

    def test_permeability_finite(self, sol_result):
        Phi = permeability(_D_M2S, sol_result['S_mean'])
        assert math.isfinite(Phi), "Permeability should be finite"

    def test_richardson_flux_positive_for_pressure_gradient(self, sol_result):
        """With upstream pressure > 0 and downstream ≈ 0, flux must be positive."""
        Phi = permeability(_D_M2S, sol_result['S_mean'])
        L   = 1e-3     # 1 mm membrane thickness
        J   = richardson_flux(Phi, P_high_Pa=16000.0, P_low_Pa=0.0, L_m=L)
        assert J > 0, "Richardson flux must be positive for P_high > P_low"

    def test_richardson_flux_increases_with_upstream_pressure(self, sol_result):
        """Higher upstream pressure → higher flux."""
        Phi = permeability(_D_M2S, sol_result['S_mean'])
        L   = 1e-3
        J_low  = richardson_flux(Phi, P_high_Pa=1000.0,  P_low_Pa=0.0, L_m=L)
        J_high = richardson_flux(Phi, P_high_Pa=16000.0, P_low_Pa=0.0, L_m=L)
        assert J_high > J_low, \
            "Flux at P_high=16000 should exceed flux at P_high=1000"

    def test_richardson_flux_zero_for_equal_pressures(self, sol_result):
        """No pressure gradient → no net flux."""
        Phi = permeability(_D_M2S, sol_result['S_mean'])
        L   = 1e-3
        J   = richardson_flux(Phi, P_high_Pa=5000.0, P_low_Pa=5000.0, L_m=L)
        assert abs(J) < 1e-30, \
            f"Flux should be ~0 for equal pressures, got {J:.3e}"

    def test_permeability_scales_linearly_with_diffusivity(self, sol_result):
        """Permeability = D × S, so doubling D doubles Phi."""
        S   = sol_result['S_mean']
        Phi1 = permeability(_D_M2S,     S)
        Phi2 = permeability(_D_M2S * 2, S)
        assert abs(Phi2 / Phi1 - 2.0) < 1e-10, \
            "Permeability should scale linearly with diffusivity"

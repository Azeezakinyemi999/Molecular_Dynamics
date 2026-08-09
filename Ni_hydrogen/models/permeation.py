"""
models/permeation.py
====================
Macroscopic H permeation flux from KMC steady-state concentrations.

Pipeline
--------
1. ``fick_flux``            — J = D (C0 − C_low) / L
2. ``sweep_pressure``       — run KMC to steady state at each pressure, collect J
3. ``check_sieverts_law``   — fit J vs √P; diagnose bulk vs surface bottleneck
4. Richardson-Sieverts permeability:
   a. ``arrhenius_diffusivity``   — D(T) = D₀ exp(−E_D / k_B T)
   b. Solubility S₀ via three routes:
        Option 1  ``lattice_site_S0``       — S₀ = 4/a₀³  (geometric maximum)
        Option 2  ``solubility_from_rates`` — S(T) from TST rates (detailed balance)
        Option 3  ``fit_solubility_from_kmc`` — empirical S = C0/√P from sweep
   c. ``sieverts_solubility`` — S(T) = S₀ exp(−ΔH_sol / k_B T)
   d. ``permeability``        — Φ = D × S  [mol·m⁻¹·s⁻¹·Pa^(−½)]
   e. ``richardson_flux``     — J = Φ (√P_high − √P_low) / L

Sieverts' law
-------------
For bulk-diffusion-limited transport, J ∝ √P (Richardson / Sieverts regime):
H₂ pressure sets the surface H concentration via C ∝ √P, and flux follows.
Deviation from linearity in J–√P space indicates surface kinetics are limiting.

Richardson-Sieverts permeability
---------------------------------
Φ(T) = D(T) · S(T)   where  S(T) = S₀ · exp(−ΔH_sol / k_B T)

ΔH_sol = ΔH_diss / 2 + ΔH_entry  (half H₂-dissociation + Hop A reaction energy)

For Option 2 (detailed balance from TST rates), the derivation equates adsorption
and desorption rates for (1/2)H₂ ↔ H*,  then entry/exit for H* ↔ H_sub::

    θ_eq ≈ √( k_diss · A_site · P / (k_des · √(2π m_H2 k_B T)) )
    C_sub = ρ_oct · (k_entry / k_exit) · θ_eq

    S(T) = C_sub/√P = ρ_oct · (k_entry/k_exit) · √( k_diss·A_site / (k_des·√(2πm_H2 k_BT)) )

k_diss is dimensionless (Boltzmann sticking, same convention as the KMC engine).
k_des, k_entry, k_exit are full TST rates in s⁻¹.
"""

from __future__ import annotations

import json
import os

import numpy as np

from models.kmc import make_grid, run_kmc_to_steady_state
from models.tst_rates import vib_partition_function, h2_gas_partition_function


# ---------------------------------------------------------------------------
# Section 1 — Fick's law
# ---------------------------------------------------------------------------

def fick_flux(
    D_m2s: float,
    C0_m3: float,
    L_m: float,
    C_low_m3: float = 0.0,
) -> float:
    """Steady-state permeation flux through a membrane of thickness L.

    .. math::

        J = D \\cdot \\frac{C_0 - C_{\\text{low}}}{L}

    Parameters
    ----------
    D_m2s : float
        Bulk diffusivity [m²/s].
    C0_m3 : float
        H concentration at the high-pressure surface [mol/m³].
    L_m : float
        Membrane thickness [m].
    C_low_m3 : float
        H concentration at the low-pressure surface [mol/m³].
        Default 0 (permeate side assumed H-free).

    Returns
    -------
    float
        Flux J [mol/(m²·s)].
    """
    if L_m <= 0:
        raise ValueError(f'Membrane thickness must be positive; got L_m={L_m}.')
    return D_m2s * (C0_m3 - C_low_m3) / L_m


# ---------------------------------------------------------------------------
# Section 2 — Pressure sweep
# ---------------------------------------------------------------------------

def pop_weighted_rate_ratio(fwd_by_env: dict, rev_by_env: dict, env_array=None) -> float:
    """Population-weighted mean of ``k_forward(env)/k_reverse(env)`` over grid sites.

    This is the dimensionless factor in the rate-ratio (dilute detailed-balance)
    subsurface occupancy ``C = ρ_oct·⟨k_fwd/k_rev⟩·θ`` — the continuous,
    count-free estimator that replaces integer occupancy counting for the
    subsurface layers. Counting floors at ~1 atom whenever the true equilibrium
    occupancy is ≪ 1 atom per grid (e.g. endothermic entry), losing all P/T
    dependence; the rate ratio does not.

    The ratio is averaged over the environments actually present on the grid
    (``env_array`` from ``make_grid``); an env absent from a rate dict falls back
    to that dict's mean, and a non-positive reverse rate contributes nothing.
    Returns 0.0 if no valid ratio exists — e.g. an empty entry-rate dict, i.e.
    no surface→sub1 channel at all.

    Parameters
    ----------
    fwd_by_env, rev_by_env : dict
        Environment-keyed forward (entry) and reverse (exit) rates [s⁻¹].
    env_array : ndarray or None
        Per-site environment labels for the layer. If ``None``, a uniform mean
        over the rate dict's environments is used.

    Returns
    -------
    float
        Population-weighted ⟨k_fwd/k_rev⟩ (dimensionless), or 0.0.
    """
    if not fwd_by_env or not rev_by_env:
        return 0.0
    _fm = sum(fwd_by_env.values()) / len(fwd_by_env)
    _rm = sum(rev_by_env.values()) / len(rev_by_env)
    envs = (list(np.asarray(env_array, dtype=object).ravel())
            if env_array is not None else list(fwd_by_env))
    ratios = []
    for e in envs:
        kr = rev_by_env.get(e, _rm)
        if kr and kr > 0.0:
            ratios.append(fwd_by_env.get(e, _fm) / kr)
    return float(np.mean(ratios)) if ratios else 0.0


def sweep_pressure(
    P_vals_Pa: list,
    rate_dict: dict,
    D_m2s: float,
    L_m: float,
    T_K: float,
    a0_m: float,
    nx: int,
    ny: int,
    composition: dict | None = None,
    seed: int = 42,
    kmc_kwargs: dict | None = None,
    sub1_env_composition: dict | None = None,
    sub2_env_composition: dict | None = None,
) -> dict:
    """Run KMC to steady state at each pressure and compute permeation flux.

    A fresh grid is created for each pressure point so that results are
    independent.  The same random seed is used for every grid so that
    composition realisations are identical across pressures.

    Parameters
    ----------
    P_vals_Pa : list of float
        H₂ partial pressures [Pa] to sweep over.
    rate_dict : dict
        KMC rate constants (see ``models/kmc.py`` module docstring).
    D_m2s : float
        Bulk diffusivity [m²/s].
    L_m : float
        Membrane thickness [m].
    T_K : float
        Temperature [K].
    a0_m : float
        FCC lattice constant [m].
    nx, ny : int
        KMC grid dimensions.
    composition : dict, optional
        Alloy element fractions.  Defaults to Hastelloy N.
    seed : int
        RNG seed for grid element assignment.
    kmc_kwargs : dict, optional
        Extra keyword arguments forwarded to ``run_kmc_to_steady_state``
        (e.g. ``window``, ``rtol``, ``max_steps``).
    sub1_env_composition, sub2_env_composition : dict, optional
        Oct-site environment fractions for the sub1/sub2 layers, forwarded to
        ``make_grid`` so entry/exit/hopB rates resolve per environment. When
        ``None`` each layer's env defaults to the surface element (see
        ``make_grid``).

    Returns
    -------
    dict
        Primary ``C0``/``J`` fields are the **rate-ratio** (dilute
        detailed-balance) occupancy ``ρ_oct·⟨k_entry/k_exit⟩·θ`` and its Fick
        flux — continuous and count-free, so they recover Sieverts √P scaling:
        ``{'C0_sub1_vals', 'C0_sub2_vals', 'C0_vals' (=sub2), 'J_sub1_vals',
        'J_vals' (=sub2)}``. Raw integer-count occupancy (the noise/quantization-
        limited diagnostic) is preserved under ``{'C0_sub1_count_vals',
        'C0_sub2_count_vals', 'J_sub1_count_vals', 'J_count_vals'}``. Plus
        ``{'P_vals', 'sqrt_P_vals', 'converged', 'theta_vals', 't_total_vals',
        'n_steps_vals', 'method'}``. ``C0_vals``/``J_vals`` alias sub2 for
        back-compat.
    """
    kw = kmc_kwargs or {}
    rho_oct = 4.0 / (a0_m ** 3) / _N_A            # oct-site density [mol/m³]
    k_entry = rate_dict.get('k_entry',      {})
    k_exit  = rate_dict.get('k_exit',       {})
    k_hbent = rate_dict.get('k_hopB_entry', {})
    k_hbext = rate_dict.get('k_hopB_exit',  {})

    P_out:        list[float] = []
    sqrtP_out:    list[float] = []
    conv_out:     list[bool]  = []
    theta_out:    list[float] = []
    t_total_out:  list[float] = []
    nsteps_out:   list[int]   = []
    # PRIMARY: rate-ratio (dilute detailed-balance) occupancy & Fick flux
    C0_sub1_out:  list[float] = []
    C0_sub2_out:  list[float] = []
    J_sub1_out:   list[float] = []
    J_out:        list[float] = []
    # DIAGNOSTIC: raw integer-count occupancy & its Fick flux
    C0_sub1_ct:   list[float] = []
    C0_sub2_ct:   list[float] = []
    J_sub1_ct:    list[float] = []
    J_ct:         list[float] = []

    for P in P_vals_Pa:
        grid = make_grid(nx, ny, composition=composition, seed=seed,
                         sub1_env_composition=sub1_env_composition,
                         sub2_env_composition=sub2_env_composition)
        ss    = run_kmc_to_steady_state(grid, rate_dict, P, T_K, D_m2s, a0_m, **kw)
        theta = float(ss['theta_ss'])

        # Rate-ratio occupancy: C = ρ_oct·⟨k_entry/k_exit⟩·θ (sub1), ×⟨k_hopB⟩
        # for sub2. Count-free and continuous — avoids the ~1-atom quantization
        # floor of integer counting when the true occupancy is ≪ 1 per grid.
        # θ ∝ √P in the dilute limit, so these recover Sieverts scaling.
        R1 = pop_weighted_rate_ratio(k_entry, k_exit, grid['sub1_env'])
        R2 = pop_weighted_rate_ratio(k_hbent, k_hbext, grid['sub2_env'])
        C1 = rho_oct * R1 * theta
        C2 = rho_oct * R1 * R2 * theta
        J1 = fick_flux(D_m2s, C1, L_m)
        J2 = fick_flux(D_m2s, C2, L_m)

        # Raw integer-count occupancy (diagnostic; noise/quantization-limited)
        C1c = float(ss['C0_sub1']); C2c = float(ss['C0_sub2'])
        J1c = fick_flux(D_m2s, C1c, L_m); J2c = fick_flux(D_m2s, C2c, L_m)

        P_out.append(float(P))
        sqrtP_out.append(float(np.sqrt(P)))
        conv_out.append(bool(ss['converged']))
        theta_out.append(theta)
        t_total_out.append(float(ss['t_total']))
        nsteps_out.append(int(ss['n_steps']))
        C0_sub1_out.append(float(C1)); C0_sub2_out.append(float(C2))
        J_sub1_out.append(float(J1));  J_out.append(float(J2))
        C0_sub1_ct.append(C1c); C0_sub2_ct.append(C2c)
        J_sub1_ct.append(float(J1c)); J_ct.append(float(J2c))

        print(
            f'  P={P:.2e} Pa | θ={theta:.4f} | '
            f'C0(sub1)={C1:.3e} C0(sub2)={C2:.3e} mol/m³ (rate-ratio; '
            f'count {C1c:.2e}/{C2c:.2e}) | J={J2:.3e} mol/m²/s | '
            f'converged={ss["converged"]}'
        )

    return {
        'P_vals':       P_out,
        'sqrt_P_vals':  sqrtP_out,
        'converged':    conv_out,
        'theta_vals':   theta_out,
        't_total_vals': t_total_out,
        'n_steps_vals': nsteps_out,
        # PRIMARY — rate-ratio occupancy & Fick flux (√P-scaling, count-free)
        'C0_sub1_vals': C0_sub1_out,
        'C0_sub2_vals': C0_sub2_out,
        'C0_vals':      C0_sub2_out,          # back-compat alias = sub2
        'J_sub1_vals':  J_sub1_out,
        'J_vals':       J_out,                # back-compat alias = sub2
        # DIAGNOSTIC — raw integer-count occupancy & flux (noise-limited)
        'C0_sub1_count_vals': C0_sub1_ct,
        'C0_sub2_count_vals': C0_sub2_ct,
        'J_sub1_count_vals':  J_sub1_ct,
        'J_count_vals':       J_ct,
        'method': ('C0/J = rate-ratio occupancy rho_oct*<k_entry/k_exit>*theta '
                   '(dilute detailed balance); *_count_vals are raw integer '
                   'counts, a noise-limited diagnostic'),
    }


# ---------------------------------------------------------------------------
# Section 3 — Sieverts' law check
# ---------------------------------------------------------------------------

def check_sieverts_law(
    P_vals_Pa: list,
    J_vals: list,
    plot: bool = True,
) -> dict:
    """Fit J vs √P and diagnose the rate-limiting step.

    A linear fit of J against √P with R² ≈ 1 indicates bulk-diffusion-limited
    transport (Sieverts' law).  Curvature (R² < 0.98) suggests surface
    adsorption/dissociation or subsurface entry kinetics are limiting.

    Parameters
    ----------
    P_vals_Pa : list of float
        Pressures [Pa].
    J_vals : list of float
        Corresponding fluxes [mol/(m²·s)].
    plot : bool
        If ``True``, display a J vs √P scatter + fit line using matplotlib.
        Silently skipped if matplotlib is unavailable.

    Returns
    -------
    dict
        ``{'slope': float, 'intercept': float, 'r_squared': float,
           'is_sieverts': bool}``
        ``is_sieverts`` is ``True`` when R² ≥ 0.98.
    """
    sqrt_P = np.sqrt(np.asarray(P_vals_Pa, dtype=float))
    J      = np.asarray(J_vals, dtype=float)

    # Linear fit: J = slope × √P + intercept
    coeffs   = np.polyfit(sqrt_P, J, 1)
    slope    = float(coeffs[0])
    intercept = float(coeffs[1])

    # R² from residuals
    J_pred   = slope * sqrt_P + intercept
    ss_res   = float(np.sum((J - J_pred) ** 2))
    ss_tot   = float(np.sum((J - J.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    is_sieverts = r2 >= 0.98

    print(f'Sieverts fit:  slope={slope:.3e}  intercept={intercept:.3e}  R²={r2:.4f}')
    if is_sieverts:
        print('  → R² ≥ 0.98: bulk-diffusion-limited (Sieverts law holds).')
    else:
        print('  → R² < 0.98: surface kinetics or subsurface entry are limiting.')

    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(sqrt_P, J, color='steelblue', zorder=3, label='KMC')
            x_fit = np.linspace(sqrt_P.min(), sqrt_P.max(), 200)
            ax.plot(x_fit, slope * x_fit + intercept, 'k--', label=f'fit R²={r2:.3f}')
            ax.set_xlabel('√P  [Pa^(1/2)]')
            ax.set_ylabel('J  [mol m⁻² s⁻¹]')
            ax.set_title('Sieverts\' law check')
            ax.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass

    return {
        'slope':      slope,
        'intercept':  intercept,
        'r_squared':  r2,
        'is_sieverts': is_sieverts,
    }


def classify_sieverts_regime(P_vals_Pa, theta_vals, converged=None,
                             dilute_theta_max: float = 0.4,
                             sat_theta: float = 0.85) -> dict:
    """Classify the Sieverts-law regime from the KMC coverage isotherm θ(P).

    Solubility itself is a thermodynamic quantity (computed from energies); the
    KMC's distinct job is to say **whether Sieverts' law even applies** to a
    given surface. Sieverts (C ∝ √P) holds only when the surface keeps its
    *dissociative-adsorption equilibrium* in the dilute limit. The coverage is
    the well-sampled KMC observable that reveals this (unlike the rare-event
    subsurface concentration, which is noise-limited):

    * ``θ ∝ √P`` at low P (log-log slope n ≈ 0.5) → dissociative equilibrium
      maintained → **sieverts_compatible** (flux ∝ √P, diffusion-limited).
    * ``θ ∝ P`` (n ≈ 1.0) → adsorption/dissociation rate-limiting (H drains
      before it can re-desorb) → **surface_limited**; Sieverts fails and the
      flux goes as P, not √P (e.g. slow-dissociation oxide surfaces).
    * ``θ`` plateaus near 1 with no dilute points sampled → **saturated_only**:
      the sweep sits above the dilute Sieverts window.

    Uses only the (well-sampled) surface coverage — never the rare-event
    subsurface count — so it is robust where the KMC flux is not.

    Parameters
    ----------
    P_vals_Pa : list of float
        Sweep pressures [Pa].
    theta_vals : list of float
        Steady-state surface coverage at each pressure (from the KMC).
    converged : list of bool, optional
        Per-point convergence flags; unconverged points are dropped.
    dilute_theta_max : float
        Upper θ bound defining the "dilute" window used to fit the low-P
        exponent.  Default 0.4.
    sat_theta : float
        θ above which the surface is treated as saturated.  Default 0.85.

    Returns
    -------
    dict
        ``{'regime', 'theta_exponent', 'theta_max', 'saturates_in_sweep',
           'n_dilute_points', 'note'}``.  ``regime`` ∈
        {``'sieverts_compatible'``, ``'surface_limited'``, ``'saturated_only'``,
        ``'insufficient_data'``}.
    """
    P  = np.asarray(P_vals_Pa, dtype=float)
    th = np.asarray(theta_vals, dtype=float)
    if converged is not None:
        keep = np.asarray(converged, dtype=bool)
        P, th = P[keep], th[keep]
    ok = (P > 0.0) & (th > 0.0)
    P, th = P[ok], th[ok]

    theta_max = float(th.max()) if th.size else 0.0
    saturated = theta_max >= sat_theta

    dil = th < dilute_theta_max
    n_dilute = int(dil.sum())               # genuinely dilute (low-θ) points
    if n_dilute >= 3:
        n = float(np.polyfit(np.log(P[dil]), np.log(th[dil]), 1)[0])
    elif P.size >= 3:
        # not enough dilute points — fall back to the lowest third of the sweep
        idx = np.argsort(P)[:max(3, P.size // 3)]
        n = float(np.polyfit(np.log(P[idx]), np.log(th[idx]), 1)[0])
    else:
        return {'regime': 'insufficient_data', 'theta_exponent': None,
                'theta_max': theta_max, 'saturates_in_sweep': saturated,
                'n_dilute_points': n_dilute,
                'note': 'need >=3 positive converged points to classify'}

    # exponent near 0.5 -> dissociative equilibrium (Sieverts); near 1.0 ->
    # dissociation rate-limited (surface-limited). Boundary midway at 0.75.
    if saturated and n_dilute < 3:
        regime = 'saturated_only'
    elif n < 0.75:
        regime = 'sieverts_compatible'
    else:
        regime = 'surface_limited'

    print(f'[Sieverts regime] θ~P^{n:.2f}  θ_max={theta_max:.2f}  -> {regime}')
    return {'regime': regime, 'theta_exponent': n, 'theta_max': theta_max,
            'saturates_in_sweep': saturated, 'n_dilute_points': n_dilute,
            'note': ('theta~P^n: n≈0.5 sieverts_compatible, n≈1.0 surface_limited; '
                     'theta_max near 1 = saturated (out of the dilute window)')}


# ---------------------------------------------------------------------------
# Section 4 — Richardson-Sieverts permeability
# ---------------------------------------------------------------------------

# Physical constants used throughout this section
_KB_EV   = 8.617333262e-5    # eV / K
_KB_J    = 1.380649e-23      # J / K
_M_H2_KG = 2.0 * 1.6735575e-27  # kg  (H₂ molecule)
_N_A     = 6.02214076e23     # Avogadro / mol  (counts -> mol H)


# ── Step G: units metadata for result payloads ──────────────────────────────────
# Maps result value-field name -> unit string, so written JSON carries units
# explicitly instead of relying on inconsistent key-name suffixes. All quantities
# are SI + per mol H, pressures in Pa (implicit P_ref = 1 Pa). Fields absent from
# a payload are simply skipped; string/bool/genuinely-dimensionless fields are
# either labelled 'dimensionless' or omitted.
RESULT_UNITS = {
    # energies
    'E_D_eV': 'eV', 'Ea_eV': 'eV', 'E_phi_eV': 'eV', 'E_phi_err_eV': 'eV',
    'dH_sol_eV': 'eV', 'dH_sol_err_eV': 'eV', 'dH_sol_mean_eV': 'eV',
    'dH_diss_eV': 'eV', 'dH_entry_eV': 'eV',
    # diffusivity, length, temperature, time
    'D_m2s': 'm^2 s^-1', 'D0_m2s': 'm^2 s^-1',
    'a0_m': 'm', 'L_m': 'm', 'T_K': 'K', 'T_K_arr': 'K', 't_total_vals': 's',
    # pressure
    'P_vals': 'Pa', 'P_high_Pa': 'Pa', 'P_dilute_Pa': 'Pa', 'sqrt_P_vals': 'Pa^0.5',
    # solubility  [mol H m^-3 Pa^-0.5]
    'S0': 'mol H m^-3 Pa^-0.5', 'S': 'mol H m^-3 Pa^-0.5', 'S_arr': 'mol H m^-3 Pa^-0.5',
    'S_mean': 'mol H m^-3 Pa^-0.5', 'S_std': 'mol H m^-3 Pa^-0.5',
    'S_vals': 'mol H m^-3 Pa^-0.5', 'S_sub1': 'mol H m^-3 Pa^-0.5',
    'S_sub2': 'mol H m^-3 Pa^-0.5',
    # permeability, flux, concentration
    'Phi': 'mol H m^-1 s^-1 Pa^-0.5', 'Phi0': 'mol H m^-1 s^-1 Pa^-0.5',
    'J': 'mol H m^-2 s^-1', 'J_vals': 'mol H m^-2 s^-1', 'J_sub1_vals': 'mol H m^-2 s^-1',
    'J_count_vals': 'mol H m^-2 s^-1', 'J_sub1_count_vals': 'mol H m^-2 s^-1',
    'sub1_at_Phigh': 'mol H m^-2 s^-1', 'sub2_at_Phigh': 'mol H m^-2 s^-1',
    'C0_vals': 'mol H m^-3', 'C0_sub1_vals': 'mol H m^-3', 'C0_sub2_vals': 'mol H m^-3',
    'C0_sub1_count_vals': 'mol H m^-3', 'C0_sub2_count_vals': 'mol H m^-3',
    # counts
    'n_H': 'count', 'n_env': 'count', 'n_converged': 'count', 'n_points': 'count',
    'n_dilute_points': 'count', 'n_steps_vals': 'count',
    # dimensionless (labelled explicitly so their absence is not ambiguous)
    'theta_vals': 'dimensionless', 'theta_dilute': 'dimensionless',
    'theta_max': 'dimensionless', 'theta_exponent': 'dimensionless',
    'S0_rel_err': 'dimensionless (fractional)', 'S_rel_err': 'dimensionless (fractional)',
    'Phi0_rel_err': 'dimensionless (fractional)', 'J_rel_err_by_T': 'dimensionless (fractional)',
    'Phi0_factor': 'dimensionless (x/div 1-sigma band)', 'w_env': 'dimensionless',
    'r2': 'dimensionless', 'r2_S': 'dimensionless', 'sieverts_r2': 'dimensionless',
}


def units_for(payload) -> dict:
    """Collect ``{field_name: unit_string}`` for every known value field appearing
    anywhere in a (possibly nested) result payload.

    Walks nested dicts/lists so nested value fields (e.g. ``option1['Phi']``) are
    covered by a single flat block. Fields not in :data:`RESULT_UNITS` are skipped
    (strings, bools, and genuinely-dimensionless-by-omission fields). Intended to
    be attached as ``payload['units'] = units_for(payload)`` just before writing.
    """
    found = {}

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in RESULT_UNITS and k not in found:
                    found[k] = RESULT_UNITS[k]
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    _walk(payload)
    return found


def arrhenius_diffusivity(D0_m2s: float, E_D_eV: float, T_K: float) -> float:
    """Temperature-dependent bulk diffusivity from an Arrhenius fit.

    .. math::

        D(T) = D_0 \\cdot \\exp\\!\\left(-\\frac{E_D}{k_B T}\\right)

    Parameters
    ----------
    D0_m2s : float
        Diffusivity pre-exponential [m²/s].  Fit from LAMMPS MSD Arrhenius plot.
    E_D_eV : float
        Diffusion activation energy [eV].
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        D(T) in m²/s.
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    return D0_m2s * np.exp(-E_D_eV / (_KB_EV * T_K))


def lattice_site_S0(a0_m: float) -> float:
    """Option 1 — S₀ from the FCC octahedral-site density.

    .. math::

        S_0 = \\frac{4}{a_0^3}  \\quad [\\text{mol m}^{-3}\\,\\text{Pa}^{-1/2}]

    This is the geometric maximum: if every oct site were filled at P = 1 Pa with
    no thermodynamic penalty (ΔH_sol = 0).  Multiply by exp(−ΔH_sol / k_B T) via
    :func:`sieverts_solubility` to obtain S(T).

    Parameters
    ----------
    a0_m : float
        FCC lattice constant [m].

    Returns
    -------
    float
        S₀ in mol m⁻³ Pa^(−½).
    """
    _S0 = 4.0 / (a0_m ** 3) / _N_A            # mol H per m^3 (÷ Avogadro)
    print(f'[S0 opt1] S0={_S0:.3e} mol·m⁻³·Pa⁻⁰·⁵  (a0={a0_m*1e10:.4f} Å)')
    return _S0


def solubility_from_rates(
    k_diss: float,
    k_des_s1: float,
    k_entry_s1: float,
    k_exit_s1: float,
    a0_m: float,
    T_K: float,
) -> float:
    """Option 2 — S(T) from TST rate constants via detailed balance.

    Derivation (dilute-limit equilibrium):

    1. Adsorption/desorption balance:
       θ_eq = √( k_diss · A_site · P / (k_des · √(2π m_H2 k_B T)) )

    2. Entry/exit balance:
       C_sub = ρ_oct · (k_entry / k_exit) · θ_eq

    Combining:  S(T) = C_sub / √P = ρ_oct · (k_entry / k_exit)
                       · √( k_diss · A_site / (k_des · √(2π m_H2 k_B T)) )

    Parameters
    ----------
    k_diss : float
        Dimensionless H₂ sticking coefficient (Boltzmann factor, same as KMC).
        This is *not* a rate; it is multiplied by the gas-strike rate inside
        the KMC engine.
    k_des_s1 : float
        H₂ desorption rate [s⁻¹] (TST full rate from dissociation NEB).
    k_entry_s1 : float
        H* → H_sub (Hop A forward) rate [s⁻¹].
    k_exit_s1 : float
        H_sub → H* (Hop A reverse) rate [s⁻¹].
    a0_m : float
        FCC lattice constant [m].
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        S(T) in mol m⁻³ Pa^(−½).

    Notes
    -----
    The site area for H₂ dissociation is A_site = (a₀ / √2)² = a₀² / 2
    (nearest-neighbour distance squared, consistent with the KMC engine's
    ``gas_strike_rate``).
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    if k_des_s1 <= 0 or k_exit_s1 <= 0:
        raise ValueError('k_des_s1 and k_exit_s1 must be positive.')

    rho_oct  = 4.0 / (a0_m ** 3) / _N_A     # oct-site density [mol·m⁻³] (÷ Avogadro)
    A_site   = (a0_m / np.sqrt(2.0)) ** 2  # dissociation site area [m²]

    # Denominator: k_des × √(2π m_H2 k_B T)  [s⁻¹ × kg·m/s = N]
    denom = k_des_s1 * np.sqrt(2.0 * np.pi * _M_H2_KG * _KB_J * T_K)

    S = rho_oct * (k_entry_s1 / k_exit_s1) * np.sqrt(k_diss * A_site / denom)
    print(f'[S opt2] S(T={T_K:.0f}K)={S:.3e} mol·m⁻³·Pa⁻⁰·⁵')
    return S


# ---------------------------------------------------------------------------
# Section 4b — Heterogeneous, per-environment thermodynamic solubility
#
# Solubility is an equilibrium property: it depends on the reaction ENERGIES
# along the connected chain (½ H₂ dissociation + Hop A + Hop B, to sub2), not
# the barriers. Different octahedral environments have different ΔH_sol, so the
# statistically-correct solubility is a population-weighted Boltzmann sum over
# environments — never a single collapsed representative. Two S₀ prefactor
# routes are offered: geometric (lattice_site_S0) and vibrational
# (vibrational_S0, reusing the dissolved-H FS modes + gas-phase H₂ reference).
# ---------------------------------------------------------------------------

def vibrational_S0(a0_m: float, T_K: float, freqs_dissolved_cm1,
                   P_ref_Pa: float = 1.0) -> float:
    """Vibrational (partition-function) solubility pre-exponential S₀.

    .. math::

        S_0 = \\rho_{\\text{oct}} \\cdot
              \\frac{q_{\\text{vib}}^{\\text{H,dissolved}}(T)}
                   {\\sqrt{q^{\\text{H}_2,\\text{gas}}(T, P_{\\text{ref}})}}

    The √ on the gas partition function reflects the H₂ → 2H stoichiometry
    (per dissolved H atom). The gas translational term carries a 1/P factor, so
    with ``P_ref_Pa = 1`` the resulting S₀ has units mol·m⁻³·Pa^(−½) and
    ``C₀ = S₀·exp(−ΔH_sol/kT)·√P`` with P in Pa (see
    :func:`solubility_by_environment` / :func:`sieverts_solubility`).

    Parameters
    ----------
    a0_m : float
        FCC lattice constant [m].
    T_K : float
        Temperature [K].
    freqs_dissolved_cm1 : iterable of float
        Real vibrational frequencies [cm⁻¹] of a dissolved H in its oct cage
        (the FS-endpoint modes from Part 2's vibration run).
    P_ref_Pa : float
        Reference pressure fixing the S₀ units. Default 1.0 Pa.

    Returns
    -------
    float
        S₀ in mol·m⁻³·Pa^(−½).
    """
    rho_oct = 4.0 / (a0_m ** 3) / _N_A       # mol H per m^3 (÷ Avogadro)
    q_H  = vib_partition_function(freqs_dissolved_cm1, T_K)
    q_H2 = h2_gas_partition_function(T_K, P_ref_Pa)['total']
    _S0 = rho_oct * q_H / np.sqrt(q_H2)
    print(f'[S0 vib] S0={_S0:.3e} mol·m⁻³·Pa⁻⁰·⁵  (q_H={q_H:.3f}, q_H2={q_H2:.3e})')
    return _S0


def build_dh_sol_by_env(hopa_vib: dict, dh_diss_eV: float,
                        dh_diss_err_eV: float = 0.0,
                        out_json: str | None = None) -> dict:
    """Per-environment solution enthalpy ΔH_sol(env), referenced to sub1.

    ``ΔH_sol(env) = ½·ΔH_diss + ΔH_HopA(env)``, built from the ZPE-corrected
    reaction energy of Hop A (reaction energy = forward barrier − reverse
    barrier = ``Ea_zpe − Ed_zpe``), grouped by the sub1 entry environment.
    Solubility is referenced to the first subsurface (sub1) dissolved site; the
    sub1→sub2 hop (Hop B) and deeper transport are treated as bulk diffusion
    (carried by D), so they are deliberately NOT included here. Population
    weight ``w_env`` is the fraction of Hop A pathways in each environment.

    Parameters
    ----------
    hopa_vib : dict
        ``{label: {env, Ea_zpe, Ed_zpe, ...}}`` from
        ``tst_rates.write_hop_vib_rates`` (env = sub1 env).
    dh_diss_eV : float
        H₂ dissociation reaction energy [eV] (per H₂).
    dh_diss_err_eV : float
        Absolute σ on ΔH_diss (e.g. the SEM over the dissociation pathways).
        Propagated into ``dH_sol_err_eV``; default 0.0.
    out_json : str, optional
        If given, written here as ``dH_sol_by_env.json``.

    Returns
    -------
    dict
        ``{env: {dH_sol_eV, dH_sol_err_eV, w_env, n_sites, dH_hopA_eV,
        dH_hopA_err_eV}}``.  Energy errors are absolute σ [eV]; ``dH_sol_err_eV``
        = √((½·σ_diss)² + σ_HopA²) with σ_HopA the within-env SEM.
    """
    def _rxn(r):
        a, d = r.get('Ea_zpe'), r.get('Ed_zpe')
        return None if (a is None or d is None) else float(a) - float(d)

    a_by_env: dict = {}
    for r in hopa_vib.values():
        env = r.get('env') or r.get('sub1_env')
        dr  = _rxn(r)
        if env is None or dr is None:
            continue
        a_by_env.setdefault(env, []).append(dr)

    total = sum(len(v) for v in a_by_env.values()) or 1
    out: dict = {}
    for env, deltas in a_by_env.items():
        dh_hopa = float(np.mean(deltas))
        # SEM of the Hop A reaction energy within this environment
        sem_hopa = (float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
                    if len(deltas) >= 2 else 0.0)
        # ΔH_sol = ½·ΔH_diss + ΔH_HopA  ->  σ = √((½·σ_diss)² + σ_HopA²) [absolute, eV]
        dh_sol_err = float(np.sqrt((0.5 * dh_diss_err_eV) ** 2 + sem_hopa ** 2))
        out[env] = {
            'dH_sol_eV':      0.5 * dh_diss_eV + dh_hopa,
            'dH_sol_err_eV':  dh_sol_err,
            'w_env':          len(deltas) / total,
            'n_sites':        len(deltas),
            'dH_hopA_eV':     dh_hopa,
            'dH_hopA_err_eV': sem_hopa,
        }

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'[dH_sol by env] {len(out)} environments → {out_json}')
    return out


def solubility_by_environment(dh_sol_by_env: dict, S0: float, T_K: float) -> float:
    """Population-weighted Boltzmann solubility over oct-site environments.

    .. math::

        S(T) = S_0 \\cdot \\sum_{\\text{env}} w_{\\text{env}}\\,
               e^{-\\Delta H_{\\text{sol}}(\\text{env}) / k_B T}

    ``S0`` supplies the prefactor (ρ_oct-bearing) — either the geometric
    :func:`lattice_site_S0` or the vibrational :func:`vibrational_S0`. The sum
    is the dimensionless population-weighted Boltzmann average; the most
    favourable environments dominate but all contribute (replacing the old
    single-representative collapse).

    Parameters
    ----------
    dh_sol_by_env : dict
        Output of :func:`build_dh_sol_by_env`.
    S0 : float
        Solubility pre-exponential [mol·m⁻³·Pa^(−½)].
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        S(T) in mol·m⁻³·Pa^(−½).
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    if not dh_sol_by_env:
        return 0.0
    kT = _KB_EV * T_K
    boltz = sum(d['w_env'] * np.exp(-d['dH_sol_eV'] / kT)
                for d in dh_sol_by_env.values())
    _S = float(S0 * boltz)
    print(f'[S by env] S(T={T_K:.0f}K)={_S:.3e}  (Σw·e^(−ΔH/kT)={boltz:.3e}, '
          f'{len(dh_sol_by_env)} envs)')
    return _S


def langmuir_theta(dH_sol_eV: float, T_K: float, K0: float = 1.0,
                   P_Pa: float = 1.0, P_ref_Pa: float = 1.0) -> float:
    """Equilibrium site occupancy under dissociative (Sieverts) adsorption.

    .. math::

        \\theta = \\frac{K\\sqrt{P/P_{ref}}}{1 + K\\sqrt{P/P_{ref}}},
        \\qquad K = K_0\\,e^{-\\Delta H_{sol}/k_BT}

    Bounded on [0, 1] by construction, unlike the bare Boltzmann factor. In the
    dilute limit (θ ≪ 1) this reduces to ``θ ≈ K√(P/P_ref)``, which is what
    :func:`solubility_by_environment` assumes.
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    x = K0 * np.exp(-dH_sol_eV / (_KB_EV * T_K)) * np.sqrt(P_Pa / P_ref_Pa)
    return float(x / (1.0 + x))


def solubility_by_environment_saturating(
    dh_sol_by_env: dict, S0: float, rho_site: float, T_K: float,
    P_Pa: float = 1.0, P_ref_Pa: float = 1.0,
) -> dict:
    """Occupancy-limited counterpart of :func:`solubility_by_environment`.

    ``solubility_by_environment`` uses an unbounded ``exp(−ΔH/kT)``, so a single
    exothermic environment can drive S past the point where every site is
    already full — physically impossible, and observed here: Hastelloy N 7's
    geometric route reaches 1.3e11 against a site density of 1.5e5.

    The fix is not to change the enthalpies but to stop assuming θ ≪ 1. The
    route's own prefactor supplies the entropic constant, ``K0 = S0/ρ_site``, so
    the two forms agree exactly wherever the dilute limit is valid and diverge
    only where it is not.

    Parameters
    ----------
    dh_sol_by_env : dict
        Output of :func:`build_dh_sol_by_env`.
    S0 : float
        The route's dilute pre-exponential [mol·m⁻³·Pa^(−½)] — geometric,
        vibrational, or any other.
    rho_site : float
        Total site density available to H [mol·m⁻³], i.e.
        :func:`lattice_site_S0`. This is the ceiling on the dissolved
        concentration.
    T_K, P_Pa, P_ref_Pa : float
        Temperature [K] and pressures [Pa].

    Returns
    -------
    dict
        ``{'S', 'S_dilute', 'C', 'theta_mean', 'theta_max', 'regime',
        'saturation_ratio'}``. ``regime`` uses the same vocabulary and
        thresholds as :func:`classify_sieverts_regime`.
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    if rho_site <= 0:
        raise ValueError(f'rho_site must be positive; got {rho_site}.')
    if not dh_sol_by_env:
        return {'S': 0.0, 'S_dilute': 0.0, 'C': 0.0, 'theta_mean': 0.0,
                'theta_max': 0.0, 'regime': 'insufficient_data',
                'saturation_ratio': 0.0}

    K0     = S0 / rho_site
    sqrtP  = np.sqrt(P_Pa / P_ref_Pa)
    thetas = {k: langmuir_theta(d['dH_sol_eV'], T_K, K0, P_Pa, P_ref_Pa)
              for k, d in dh_sol_by_env.items()}

    C = rho_site * sum(dh_sol_by_env[k]['w_env'] * t for k, t in thetas.items())
    S = C / sqrtP

    S_dilute   = solubility_by_environment(dh_sol_by_env, S0, T_K)
    theta_max  = max(thetas.values())
    theta_mean = sum(dh_sol_by_env[k]['w_env'] * t for k, t in thetas.items())

    if theta_max >= 0.85:                       # sat_theta
        regime = 'saturated_only'
    elif theta_max >= 0.4:                      # dilute_theta_max
        regime = 'partially_saturated'
    else:
        regime = 'sieverts_compatible'

    print(f'[S sat] S(T={T_K:.0f}K)={S:.3e} vs dilute {S_dilute:.3e}  '
          f'(θ_max={theta_max:.3f}, {regime})')
    return {
        'S': float(S), 'S_dilute': float(S_dilute), 'C': float(C),
        'theta_mean': float(theta_mean), 'theta_max': float(theta_max),
        'regime': regime,
        'saturation_ratio': float(S_dilute / S) if S > 0 else float('inf'),
    }


def solubility_env_rel_err(dh_sol_by_env: dict, T_K: float,
                           S0_rel_err: float = 0.0) -> float:
    """Fractional error σ_S/S of the per-environment Boltzmann solubility.

    For ``S = S0·B`` with ``B(T) = Σ_env w_env·exp(−ΔH_env/kT)`` and per-env
    absolute errors ``dH_sol_err_eV`` (see :func:`build_dh_sol_by_env`):

        f_env = w_env·exp(−ΔH_env/kT) / B          (fractional Boltzmann weight)
        σ_B/B = (1/k_BT)·√( Σ_env f_env²·σ_(ΔH_env)² )
        σ_S/S = √( (σ_S0/S0)² + (σ_B/B)² )

    Because Σ f_env² ≤ 1, the Boltzmann average reduces the effective ΔH error
    relative to a single environment (single-env limit: σ_B/B → σ_ΔH/kT).

    Parameters
    ----------
    dh_sol_by_env : dict
        Output of :func:`build_dh_sol_by_env` (must carry ``dH_sol_err_eV``).
    T_K : float
        Temperature [K].
    S0_rel_err : float
        Fractional error on the S₀ prefactor. Default 0.0.

    Returns
    -------
    float
        Fractional error σ_S/S (dimensionless).
    """
    if T_K <= 0 or not dh_sol_by_env:
        return float(S0_rel_err)
    kT = _KB_EV * T_K
    contribs = {e: d['w_env'] * np.exp(-d['dH_sol_eV'] / kT)
                for e, d in dh_sol_by_env.items()}
    B = sum(contribs.values())
    if B <= 0:
        return float(S0_rel_err)
    var = sum((contribs[e] / B * d.get('dH_sol_err_eV', 0.0)) ** 2
              for e, d in dh_sol_by_env.items())
    sigma_B_over_B = float(np.sqrt(var) / kT)
    return float(np.sqrt(S0_rel_err ** 2 + sigma_B_over_B ** 2))


def fit_solubility_from_kmc(sweep_result: dict) -> dict:
    """Option 3 — Empirical S(T) from a KMC pressure sweep.

    Applies Sieverts' law  C₀ = S · √P  point-wise to each converged pressure
    in a ``sweep_pressure`` result.  The mean of S over all converged points is
    the best empirical estimate at the sweep temperature.

    To extract the Arrhenius pre-exponential S₀ and ΔH_sol, call this function
    at multiple temperatures and fit ln(S) vs 1/T externally.

    Parameters
    ----------
    sweep_result : dict
        Output of :func:`sweep_pressure`.  Must contain ``'P_vals'``,
        ``'C0_vals'``, and optionally ``'converged'``.

    Returns
    -------
    dict
        ``{'S_vals': list, 'P_vals': list, 'S_mean': float, 'S_std': float,
           'n_converged': int}``
        ``S_vals[i]`` is ``C0[i] / √P[i]``  in mol m⁻³ Pa^(−½).
        Non-converged or zero-pressure points are excluded from statistics
        but appear as ``None`` in ``S_vals``.
    """
    P_vals       = sweep_result['P_vals']
    C0_vals      = sweep_result['C0_vals']
    converged    = sweep_result.get('converged', [True] * len(P_vals))
    sqrt_P_saved = sweep_result.get('sqrt_P_vals')

    S_vals: list = []
    valid:  list[float] = []

    for i, (P, C0, conv) in enumerate(zip(P_vals, C0_vals, converged)):
        if conv and P > 0:
            sqrtP = sqrt_P_saved[i] if sqrt_P_saved else np.sqrt(float(P))
            s = float(C0) / sqrtP
            S_vals.append(s)
            valid.append(s)
        else:
            S_vals.append(None)

    S_mean = float(np.mean(valid))              if valid            else 0.0
    S_std  = float(np.std(valid, ddof=1))       if len(valid) > 1  else 0.0

    print(f'[S opt3] S_mean={S_mean:.3e}  S_std={S_std:.3e}  n_converged={len(valid)}')
    return {
        'S_vals':      S_vals,
        'P_vals':      list(P_vals),
        'S_mean':      S_mean,
        'S_std':       S_std,
        'n_converged': len(valid),
    }


def sieverts_solubility(dH_sol_eV: float, S0: float, T_K: float) -> float:
    """Arrhenius Sieverts solubility at temperature T.

    .. math::

        S(T) = S_0 \\cdot \\exp\\!\\left(-\\frac{\\Delta H_{\\text{sol}}}{k_B T}\\right)

    Parameters
    ----------
    dH_sol_eV : float
        Solution enthalpy per H atom [eV].
        ΔH_sol = ΔH_diss / 2 + ΔH_entry  (sign convention: positive = endothermic).
    S0 : float
        Solubility pre-exponential [mol m⁻³ Pa^(−½)].
        Use :func:`lattice_site_S0` for Option 1, or extract from
        :func:`solubility_from_rates` / :func:`fit_solubility_from_kmc`.
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        S(T) in mol m⁻³ Pa^(−½).
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    _S = S0 * np.exp(-dH_sol_eV / (_KB_EV * T_K))
    print(f'[Sieverts] S({T_K:.0f}K)={_S:.3e}  (S0={S0:.3e}  dH={dH_sol_eV:.3f} eV)')
    return _S


def permeability(D_m2s: float, S_m3_pasqrt: float) -> float:
    """Richardson-Sieverts permeability Φ = D × S.

    .. math::

        \\Phi = D \\cdot S
        \\quad [\\text{mol}\\cdot\\text{m}^{-1}\\cdot\\text{s}^{-1}\\cdot\\text{Pa}^{-1/2}]

    Parameters
    ----------
    D_m2s : float
        Bulk diffusivity [m²/s].
    S_m3_pasqrt : float
        Sieverts solubility [mol m⁻³ Pa^(−½)].

    Returns
    -------
    float
        Permeability Φ in mol·m⁻¹·s⁻¹·Pa^(−½).
    """
    _Phi = D_m2s * S_m3_pasqrt
    print(f'[Phi] Φ={_Phi:.3e} mol·m⁻¹·s⁻¹·Pa⁻⁰·⁵')
    return _Phi


def fit_arrhenius(T_K_arr, y_arr, y_err_arr=None) -> dict:
    """Fit ``y(T) = A·exp(−Ea / kB T)`` by (optionally weighted) regression of ln y vs 1/T.

    Reusable for the solubility (Ea = ΔH_sol) and permeability (Ea = E_Φ)
    Arrhenius outputs. Points with non-positive T or y are dropped. The R² of
    the ln-linear fit is the curvature/fit-quality flag the plan calls for: a
    per-environment S(T) is a *sum* of Arrhenius terms, so R² < 1 flags the
    physically-real curvature (the dominant environment shifting with T) rather
    than hiding it behind a forced straight line.

    Parameters
    ----------
    T_K_arr : iterable of float
        Temperatures [K].
    y_arr : iterable of float
        Quantity at each temperature (e.g. S or Φ).

    Returns
    -------
    dict
        ``{'prefactor', 'Ea_eV', 'r2', 'n_points', 'Ea_err_eV',
        'prefactor_rel_err'}``. When per-point absolute errors ``y_err_arr`` are
        supplied the fit is inverse-variance weighted in log space
        (wᵢ = (yᵢ/σ_yᵢ)²) and ``Ea_err_eV`` (absolute σ on Ea) and
        ``prefactor_rel_err`` (σ_ln of the prefactor) are propagated; otherwise
        those two are NaN. ``prefactor``/``Ea_eV`` are NaN with < 2 valid points.
    """
    T = np.asarray(list(T_K_arr), dtype=float)
    y = np.asarray(list(y_arr),   dtype=float)
    ye = np.asarray(list(y_err_arr), dtype=float) if y_err_arr is not None else None
    mask = (T > 0) & (y > 0)
    if ye is not None:
        mask = mask & np.isfinite(ye)
    T, y = T[mask], y[mask]
    if ye is not None:
        ye = ye[mask]
    _nan = float('nan')
    if len(T) < 2:
        return {'prefactor': _nan, 'Ea_eV': _nan, 'r2': _nan,
                'n_points': int(len(T)), 'Ea_err_eV': _nan, 'prefactor_rel_err': _nan}
    x  = 1.0 / T
    ly = np.log(y)
    # inverse-variance weights in log space: wᵢ = 1/σ_(ln y)² = (y/σ_y)²
    if ye is not None and np.all(ye > 0):
        w = (y / ye) ** 2
    else:
        w = np.ones_like(x)
    sw  = w.sum(); sx = (w * x).sum(); sy = (w * ly).sum()
    sxx = (w * x * x).sum(); sxy = (w * x * ly).sum()
    denom = sw * sxx - sx ** 2
    slope = (sw * sxy - sx * sy) / denom
    inter = (sy - slope * sx) / sw
    pred   = slope * x + inter
    ss_res = float((w * (ly - pred) ** 2).sum())
    ss_tot = float((w * (ly - sy / sw) ** 2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    Ea = -float(slope) * _KB_EV
    A  = float(np.exp(inter))
    # parameter standard errors (need >2 points for a residual-variance estimate)
    if len(T) > 2:
        sigma2    = ss_res / (len(T) - 2)
        Ea_err    = _KB_EV * float(np.sqrt(sigma2 * sw / denom))    # absolute σ on Ea
        A_rel_err = float(np.sqrt(sigma2 * sxx / denom))            # σ_ln A (fractional)
    else:
        Ea_err = A_rel_err = _nan
    print(f'[Arrhenius fit] A={A:.3e}  Ea={Ea:.4f}±{Ea_err:.4f} eV  R²={r2:.4f}  (n={len(T)})')
    return {'prefactor': A, 'Ea_eV': Ea, 'r2': float(r2), 'n_points': int(len(T)),
            'Ea_err_eV': Ea_err, 'prefactor_rel_err': A_rel_err}


def permeability_arrhenius(D0_m2s: float, E_D_eV: float,
                           S0: float, dH_sol_eV: float,
                           D0_rel_err: float = 0.0, E_D_err_eV: float = 0.0,
                           S0_rel_err: float = 0.0, dH_sol_err_eV: float = 0.0) -> dict:
    """Permeability Arrhenius parameters from the diffusivity and solubility fits.

    Since Φ = D·S with D = D₀·exp(−E_D/kT) and S = S₀·exp(−ΔH_sol/kT):

    .. math::

        \\Phi_0 = D_0 \\cdot S_0, \\qquad E_\\Phi = E_D + \\Delta H_{\\text{sol}}

    the textbook result that the permeation activation energy is the diffusion
    barrier plus the solution enthalpy. No separate fit is required; a direct
    ``ln Φ`` vs 1/T fit (via :func:`fit_arrhenius`) is the cross-check.

    Parameters
    ----------
    D0_m2s : float
        Diffusivity pre-exponential [m²/s].
    E_D_eV : float
        Diffusion activation energy [eV].
    S0 : float
        Solubility pre-exponential [mol·m⁻³·Pa^(−½)] (geometric or vibrational).
    dH_sol_eV : float
        Solution enthalpy [eV].

    Error propagation (§ audits/error_propagation_plan.md): Φ0 is a product, so
    fractional errors add in quadrature; E_Φ is a sum, so absolute errors do::

        Φ0_rel_err = √( D0_rel_err² + S0_rel_err² )          (fractional)
        E_phi_err  = √( E_D_err²    + dH_sol_err² )           (absolute, eV)

    ``Phi0_factor = exp(Φ0_rel_err)`` is the multiplicative (×/÷) 1σ band for the
    log-normal Φ0 (report Φ0 ∈ [Φ0/factor, Φ0·factor], never a symmetric ±).

    Returns
    -------
    dict
        ``{'Phi0', 'E_phi_eV', 'Phi0_rel_err', 'Phi0_factor', 'E_phi_err_eV'}``.
    """
    Phi0  = float(D0_m2s * S0)
    E_phi = float(E_D_eV + dH_sol_eV)
    Phi0_rel_err = float(np.sqrt(D0_rel_err ** 2 + S0_rel_err ** 2))
    E_phi_err    = float(np.sqrt(E_D_err_eV ** 2 + dH_sol_err_eV ** 2))
    return {'Phi0': Phi0, 'E_phi_eV': E_phi,
            'Phi0_rel_err': Phi0_rel_err, 'Phi0_factor': float(np.exp(Phi0_rel_err)),
            'E_phi_err_eV': E_phi_err}


def richardson_flux(
    Phi: float,
    P_high_Pa: float,
    P_low_Pa: float,
    L_m: float,
) -> float:
    """Permeation flux from the Richardson-Sieverts equation.

    .. math::

        J = \\frac{\\Phi}{L} \\left(\\sqrt{P_{\\text{high}}} - \\sqrt{P_{\\text{low}}}\\right)

    Parameters
    ----------
    Phi : float
        Permeability [mol·m⁻¹·s⁻¹·Pa^(−½)].  From :func:`permeability`.
    P_high_Pa : float
        Feed-side H₂ pressure [Pa].
    P_low_Pa : float
        Permeate-side H₂ pressure [Pa].  Use 0 for a fully swept permeate.
    L_m : float
        Membrane thickness [m].

    Returns
    -------
    float
        Flux J in mol m⁻² s⁻¹.
    """
    if L_m <= 0:
        raise ValueError(f'Membrane thickness must be positive; got L_m={L_m}.')
    _J = Phi * (np.sqrt(P_high_Pa) - np.sqrt(max(P_low_Pa, 0.0))) / L_m
    print(f'[Richardson] J={_J:.3e} mol·m⁻²·s⁻¹  (P_high={P_high_Pa:.2e} Pa  L={L_m:.3e} m)')
    return _J


# ---------------------------------------------------------------------------
# Section 5 — Per-(stem, n_H) diffusivity resolution (Part 2 ↔ Part 3 handoff)
# ---------------------------------------------------------------------------

def resolve_nh_diffusivity(work_dir: str, stem: str, n_h: int) -> dict:
    """
    Load Part 3's per-(stem, n_H) diffusivity fit for use in Part 2.

    No placeholder is ever substituted: a missing file, or a fit with a
    NaN/missing D0 or Ea, means this concentration is not ready — the
    caller should skip it entirely rather than fabricate a value. This
    is the decision logic behind permeation_run.py's per-n_H loop
    (generated by :func:`models.permeation_workflow.generate_permeation_scripts`),
    extracted here so it can be unit tested directly.

    Parameters
    ----------
    work_dir : str
        The shared root directory Parts 2 and 3 both write ``results/`` under.
    stem : str
        Material stem, e.g. ``'ni_bulk_test'``.
    n_h : int
        H concentration — number of H mol in the Part-3 MD box.

    Returns
    -------
    dict
        ``nh_dir`` : str — ``results/{stem}_{n_h}H`` directory.
        ``diff_file`` : str — path to ``diffusivity_arrhenius.json``.
        ``ready`` : bool — True iff a valid (non-NaN) D0/Ea was found.
        ``D0_m2s`` : float or None.
        ``E_D_eV`` : float or None.
        ``message`` : str or None — human-readable reason, set iff not ready.
        ``dilute_note`` : str or None — set iff ready and ``n_h > 1``
            (Sieverts'/Richardson's formulas assume dilute H; n_H > 1 is
            not the dilute limit, so H-H interactions were present in the
            MD box that produced this fit).
    """
    nh_dir = os.path.join(work_dir, 'results', f'{stem}_{n_h}H')
    diff_file = os.path.join(nh_dir, 'diffusivity_arrhenius.json')

    if not os.path.exists(diff_file):
        return dict(
            nh_dir=nh_dir, diff_file=diff_file, ready=False,
            D0_m2s=None, E_D_eV=None, D0_err_m2s=None, E_D_err_eV=None,
            dilute_note=None,
            message=(f'{diff_file} not found — Part 3 has not produced a '
                     f'diffusivity fit for n_H={n_h}. Skipping this '
                     f'concentration entirely (no permeability computed, '
                     f'nothing fabricated).'),
        )

    with open(diff_file) as f:
        diff_fit = json.load(f)
    D0, Ea = diff_fit.get('D0_m2s'), diff_fit.get('E_D_eV')
    if D0 is None or Ea is None or D0 != D0 or Ea != Ea:   # NaN-safe
        return dict(
            nh_dir=nh_dir, diff_file=diff_file, ready=False,
            D0_m2s=None, E_D_eV=None, D0_err_m2s=None, E_D_err_eV=None,
            dilute_note=None,
            message=(f'{diff_file} has no valid D0/Ea (NaN or missing — '
                      f'Part 3 likely could not fit an Arrhenius relation '
                      f'for n_H={n_h}, e.g. fewer than 2 valid '
                      f'temperatures). Skipping this concentration '
                      f'entirely.'),
        )

    dilute_note = None
    if n_h > 1:
        dilute_note = (
            f"Sieverts' law and Richardson's permeation formula assume dilute "
            f"dissolved H. n_H={n_h} is not the dilute limit (n_H=1), so H-H "
            f"interactions were present in this MD box and may make the "
            f"derived solubility/permeability below an approximation. For a "
            f"rigorous non-dilute treatment, use the raw KMC sweep data in "
            f"permeation_sweep_T<T>K.json under {nh_dir} directly "
            f"(flux J(P), coverage θ(P), concentration C0(P) vs pressure — "
            f"no √P scaling assumed)."
        )

    return dict(nh_dir=nh_dir, diff_file=diff_file, ready=True,
                D0_m2s=D0, E_D_eV=Ea,
                D0_err_m2s=diff_fit.get('D0_err_m2s'),
                E_D_err_eV=diff_fit.get('E_D_err_eV'),
                dilute_note=dilute_note, message=None)

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
   d. ``permeability``        — Φ = D × S  [atoms·m⁻¹·s⁻¹·Pa^(−½)]
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

import numpy as np

from models.kmc import make_grid, run_kmc_to_steady_state


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
        H concentration at the high-pressure surface [atoms/m³].
    L_m : float
        Membrane thickness [m].
    C_low_m3 : float
        H concentration at the low-pressure surface [atoms/m³].
        Default 0 (permeate side assumed H-free).

    Returns
    -------
    float
        Flux J [atoms/(m²·s)].
    """
    if L_m <= 0:
        raise ValueError(f'Membrane thickness must be positive; got L_m={L_m}.')
    return D_m2s * (C0_m3 - C_low_m3) / L_m


# ---------------------------------------------------------------------------
# Section 2 — Pressure sweep
# ---------------------------------------------------------------------------

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

    Returns
    -------
    dict
        ``{'P_vals': list, 'J_vals': list, 'C0_vals': list,
           'sqrt_P_vals': list, 'converged': list}``
    """
    kw = kmc_kwargs or {}
    P_out:       list[float] = []
    J_out:       list[float] = []
    C0_out:      list[float] = []
    sqrtP_out:   list[float] = []
    conv_out:    list[bool]  = []

    for P in P_vals_Pa:
        grid = make_grid(nx, ny, composition=composition, seed=seed)
        ss   = run_kmc_to_steady_state(grid, rate_dict, P, T_K, D_m2s, a0_m, **kw)
        C0   = ss['C0']
        J    = fick_flux(D_m2s, C0, L_m)

        P_out.append(float(P))
        J_out.append(float(J))
        C0_out.append(float(C0))
        sqrtP_out.append(float(np.sqrt(P)))
        conv_out.append(bool(ss['converged']))

        print(
            f'  P={P:.2e} Pa | C0={C0:.3e} atoms/m³ | '
            f'J={J:.3e} atoms/m²/s | converged={ss["converged"]}'
        )

    return {
        'P_vals':     P_out,
        'J_vals':     J_out,
        'C0_vals':    C0_out,
        'sqrt_P_vals': sqrtP_out,
        'converged':  conv_out,
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
        Corresponding fluxes [atoms/(m²·s)].
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
            ax.set_ylabel('J  [atoms m⁻² s⁻¹]')
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


# ---------------------------------------------------------------------------
# Section 4 — Richardson-Sieverts permeability
# ---------------------------------------------------------------------------

# Physical constants used throughout this section
_KB_EV   = 8.617333262e-5    # eV / K
_KB_J    = 1.380649e-23      # J / K
_M_H2_KG = 2.0 * 1.6735575e-27  # kg  (H₂ molecule)


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

        S_0 = \\frac{4}{a_0^3}  \\quad [\\text{atoms m}^{-3}\\,\\text{Pa}^{-1/2}]

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
        S₀ in atoms m⁻³ Pa^(−½).
    """
    return 4.0 / (a0_m ** 3)


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
        S(T) in atoms m⁻³ Pa^(−½).

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

    rho_oct  = 4.0 / (a0_m ** 3)           # oct-site density [m⁻³]
    A_site   = (a0_m / np.sqrt(2.0)) ** 2  # dissociation site area [m²]

    # Denominator: k_des × √(2π m_H2 k_B T)  [s⁻¹ × kg·m/s = N]
    denom = k_des_s1 * np.sqrt(2.0 * np.pi * _M_H2_KG * _KB_J * T_K)

    S = rho_oct * (k_entry_s1 / k_exit_s1) * np.sqrt(k_diss * A_site / denom)
    return S


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
        ``S_vals[i]`` is ``C0[i] / √P[i]``  in atoms m⁻³ Pa^(−½).
        Non-converged or zero-pressure points are excluded from statistics
        but appear as ``None`` in ``S_vals``.
    """
    P_vals    = sweep_result['P_vals']
    C0_vals   = sweep_result['C0_vals']
    converged = sweep_result.get('converged', [True] * len(P_vals))

    S_vals: list = []
    valid:  list[float] = []

    for P, C0, conv in zip(P_vals, C0_vals, converged):
        if conv and P > 0:
            s = float(C0) / np.sqrt(float(P))
            S_vals.append(s)
            valid.append(s)
        else:
            S_vals.append(None)

    S_mean = float(np.mean(valid))              if valid            else 0.0
    S_std  = float(np.std(valid, ddof=1))       if len(valid) > 1  else 0.0

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
        Solubility pre-exponential [atoms m⁻³ Pa^(−½)].
        Use :func:`lattice_site_S0` for Option 1, or extract from
        :func:`solubility_from_rates` / :func:`fit_solubility_from_kmc`.
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        S(T) in atoms m⁻³ Pa^(−½).
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    return S0 * np.exp(-dH_sol_eV / (_KB_EV * T_K))


def permeability(D_m2s: float, S_m3_pasqrt: float) -> float:
    """Richardson-Sieverts permeability Φ = D × S.

    .. math::

        \\Phi = D \\cdot S
        \\quad [\\text{atoms}\\cdot\\text{m}^{-1}\\cdot\\text{s}^{-1}\\cdot\\text{Pa}^{-1/2}]

    Parameters
    ----------
    D_m2s : float
        Bulk diffusivity [m²/s].
    S_m3_pasqrt : float
        Sieverts solubility [atoms m⁻³ Pa^(−½)].

    Returns
    -------
    float
        Permeability Φ in atoms·m⁻¹·s⁻¹·Pa^(−½).
    """
    return D_m2s * S_m3_pasqrt


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
        Permeability [atoms·m⁻¹·s⁻¹·Pa^(−½)].  From :func:`permeability`.
    P_high_Pa : float
        Feed-side H₂ pressure [Pa].
    P_low_Pa : float
        Permeate-side H₂ pressure [Pa].  Use 0 for a fully swept permeate.
    L_m : float
        Membrane thickness [m].

    Returns
    -------
    float
        Flux J in atoms m⁻² s⁻¹.
    """
    if L_m <= 0:
        raise ValueError(f'Membrane thickness must be positive; got L_m={L_m}.')
    return Phi * (np.sqrt(P_high_Pa) - np.sqrt(max(P_low_Pa, 0.0))) / L_m

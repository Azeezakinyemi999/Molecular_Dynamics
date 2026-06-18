"""
models/energetics.py
=====================
Energy calculation and NEB post-processing for Hastelloy N surface science.

Sections
--------
1. Core energy formulas          — calc_binding_energy, calc_dissociation_barrier,
                                   calc_reaction_energy
2. Chemical potential / pressure — mu_h_to_pressure, pressure_to_mu_h
3. Site ranking                  — rank_sites, best_n_sites
4. NEB post-processing           — summarise_neb, plot_mep
"""

from __future__ import annotations

import os

import numpy as np

# Boltzmann constant in eV/K  (same value as in diffusivity_post_processing)
KB_EV = 8.617333e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Core energy formulas
# ═══════════════════════════════════════════════════════════════════════════════

def calc_binding_energy(
    e_with_h: float,
    e_clean: float,
    e_h2_gas: float,
    n_h2_ref: float = 0.5,
) -> float:
    """Compute the H binding / adsorption / solution energy.

    General formula:

    .. math::

        E_{bind} = E(\\text{slab+H}) - E(\\text{clean slab})
                   - n \\cdot E(\\text{H}_2)

    The ``n_h2_ref`` parameter selects the physical context:

    * ``n_h2_ref=1.0`` — H₂ molecular adsorption energy (NB05)
    * ``n_h2_ref=0.5`` — H-atom adsorption or subsurface solution energy
      (NB06, NB08); equivalent to referencing against half the H₂ molecule

    Parameters
    ----------
    e_with_h : float
        Total energy of the slab with H (or H₂) present, in eV.
    e_clean : float
        Total energy of the clean slab, in eV.
    e_h2_gas : float
        Total energy of a gas-phase H₂ molecule, in eV.
    n_h2_ref : float, optional
        Number of H₂ molecules used as the reference.  Default ``0.5``.

    Returns
    -------
    float
        Binding energy in eV.  Negative = exothermic / stable.

    Notes
    -----
    Source: NB05 cell 29 (H₂, ``n_h2_ref=1.0``);
            NB06 cell 15 (H*, ``n_h2_ref=0.5``);
            NB08 cell 19 (H_sub, ``n_h2_ref=0.5``).
    """
    return e_with_h - e_clean - n_h2_ref * e_h2_gas


def calc_dissociation_barrier(e_ts: float, e_is: float) -> float:
    """Compute the forward dissociation barrier from a NEB transition state.

    .. math::

        E_a = E_{TS} - E_{IS}

    Parameters
    ----------
    e_ts : float
        Energy of the transition-state image (highest point on MEP), eV.
    e_is : float
        Energy of the initial state, eV.

    Returns
    -------
    float
        Forward barrier in eV.  Always ≥ 0 for a true saddle point.

    Notes
    -----
    Source: NB06 cell 32 ``E_a_final``;
            NB09 cell 23 ``E_abs_final``.
    """
    return float(e_ts - e_is)


def calc_reaction_energy(e_fs: float, e_is: float) -> float:
    """Compute the reaction energy (thermodynamic driving force).

    .. math::

        \\Delta E = E_{FS} - E_{IS}

    Parameters
    ----------
    e_fs : float
        Energy of the final state, eV.
    e_is : float
        Energy of the initial state, eV.

    Returns
    -------
    float
        Reaction energy in eV.  Negative = exothermic.

    Notes
    -----
    Source: NB06 cell 32 ``dE_final``;
            NB09 cell 23 ``dE_final``.
    """
    return float(e_fs - e_is)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Chemical potential ↔ H₂ pressure  (NB07)
# ═══════════════════════════════════════════════════════════════════════════════

def mu_h_to_pressure(
    mu_h: float,
    e_h2_gas: float,
    T: float,
) -> float:
    """Convert H chemical potential to equivalent H₂ partial pressure.

    Inverts the ideal-gas expression:

    .. math::

        \\mu_H = \\tfrac{1}{2}E_{H_2} + \\tfrac{1}{2}k_BT\\ln(P/P_0)
        \\implies P/P_0 = \\exp\\!\\left(
            \\frac{2(\\mu_H - \\tfrac{1}{2}E_{H_2})}{k_B T}
        \\right)

    Parameters
    ----------
    mu_h : float
        H chemical potential in eV.
    e_h2_gas : float
        DFT/ML energy of a gas-phase H₂ molecule in eV.
    T : float
        Temperature in K.

    Returns
    -------
    float
        Dimensionless pressure ratio P/P₀.

    Notes
    -----
    Source: NB07 cell 13.
    """
    return float(np.exp(2.0 * (mu_h - 0.5 * e_h2_gas) / (KB_EV * T)))


def pressure_to_mu_h(
    pressure: float,
    e_h2_gas: float,
    T: float,
) -> float:
    """Convert H₂ partial pressure to H chemical potential.

    .. math::

        \\mu_H = \\tfrac{1}{2}E_{H_2}
               + \\tfrac{1}{2}k_BT\\ln(P/P_0)

    Parameters
    ----------
    pressure : float
        Dimensionless pressure ratio P/P₀.
    e_h2_gas : float
        DFT/ML energy of gas-phase H₂ in eV.
    T : float
        Temperature in K.

    Returns
    -------
    float
        H chemical potential in eV.

    Notes
    -----
    Source: NB07 cell 13 (inverse of ``mu_h_to_pressure``).
    """
    return float(0.5 * e_h2_gas + 0.5 * KB_EV * T * np.log(pressure))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Site ranking  (NB05, NB06)
# ═══════════════════════════════════════════════════════════════════════════════

def rank_sites(
    energy_dict: dict[str, float],
) -> list[tuple[str, float]]:
    """Sort adsorption sites by binding energy, most stable first.

    Parameters
    ----------
    energy_dict : dict
        ``{site_label: binding_energy_eV}`` — e.g.
        ``{'top_Ni': -0.45, 'bridge': -0.61, 'fcc': -0.73}``.

    Returns
    -------
    list of (str, float)
        Site–energy pairs sorted ascending (most negative = most stable first).

    Notes
    -----
    Source: NB05 cell 29 ranking logic;
            NB06 cell 15 ``h_eads`` ranking.
    """
    return sorted(energy_dict.items(), key=lambda x: x[1])


def best_n_sites(
    energy_dict: dict[str, float],
    n: int = 2,
    min_separation: float | None = None,
    site_coords: dict[str, np.ndarray] | None = None,
) -> list[tuple[str, float]]:
    """Return the n most stable sites, optionally enforcing a minimum separation.

    Used in NB06 to identify the IS/FS pair for the NEB calculation: the
    two most stable H* adsorption sites that are far enough apart to
    represent dissociated H atoms.

    Parameters
    ----------
    energy_dict : dict
        ``{site_label: binding_energy_eV}``.
    n : int, optional
        Number of sites to return.  Default ``2``.
    min_separation : float, optional
        Minimum distance in Å between any two selected sites.  If ``None``
        no spatial filter is applied.
    site_coords : dict, optional
        ``{site_label: array([x, y])}`` — required when
        ``min_separation`` is set.

    Returns
    -------
    list of (str, float)
        Up to ``n`` site–energy pairs satisfying the separation constraint,
        sorted by energy (most stable first).

    Raises
    ------
    ValueError
        If ``min_separation`` is set but ``site_coords`` is not provided.

    Notes
    -----
    Source: NB06 cell 20 — IS/FS selection logic (H-H > 2.5 Å enforced).
    """
    if min_separation is not None and site_coords is None:
        raise ValueError('site_coords is required when min_separation is set.')

    ranked = rank_sites(energy_dict)

    if min_separation is None or site_coords is None:
        _sel = ranked[:n]
        for _i, (_lbl, _e) in enumerate(_sel):
            print(f'  [site {_i+1}] {_lbl:<18s}  E_bind={_e:.4f} eV')
        print(f'[best_sites] {len(_sel)}/{n} selected')
        return _sel

    selected: list[tuple[str, float]] = []
    for label, energy in ranked:
        if label not in site_coords:
            continue
        pos = np.asarray(site_coords[label], dtype=float)
        too_close = any(
            np.linalg.norm(pos - np.asarray(site_coords[s], dtype=float)) < min_separation
            for s, _ in selected
        )
        if not too_close:
            selected.append((label, energy))
        if len(selected) == n:
            break

    for _i, (_lbl, _e) in enumerate(selected):
        print(f'  [site {_i+1}] {_lbl:<18s}  E_bind={_e:.4f} eV')
    print(f'[best_sites] {len(selected)}/{n} selected  (sep≥{min_separation:.2f} Å)')
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — NEB post-processing  (NB06, NB09)
# ═══════════════════════════════════════════════════════════════════════════════

def summarise_neb(
    barrier_file: str,
    path_file: str,
) -> dict:
    """Load NEB barrier and path files and return a unified results dict.

    Parameters
    ----------
    barrier_file : str
        Path to ``neb_barrier.txt`` written by :func:`models.ase_neb.write_neb_results`.
    path_file : str
        Path to ``neb_path.dat`` written by :func:`models.ase_neb.write_neb_results`.

    Returns
    -------
    dict with keys
        ``Ea``         – forward barrier in eV
        ``E_des``      – desorption barrier (reverse) in eV
        ``delta_E``    – reaction energy (FS − IS) in eV
        ``ts_index``   – index of the transition-state image
        ``converged``  – bool
        ``fmax_final`` – final max force in eV/Å
        ``rxn_coord``  – ndarray, reaction coordinate (normalised arc length)
        ``energies``   – ndarray, relative energies along MEP in eV

    Notes
    -----
    Source: NB06 cell 28; NB09 cell 19.
    Calls: ``models.parsers.parse_barrier_file``,
           ``models.parsers.parse_neb_path``.
    """
    from models.parsers import parse_barrier_file, parse_neb_path

    barrier = parse_barrier_file(barrier_file) if os.path.exists(barrier_file) else {}
    path    = parse_neb_path(path_file)         if os.path.exists(path_file)    else None

    rxn_coord = path[0] if path is not None else None
    energies  = path[1] if path is not None else None

    _Ea   = barrier.get('E_abs',    float('nan'))
    _Ed   = barrier.get('E_des',    float('nan'))
    _conv = barrier.get('converged', None)
    print(f'[summarise_neb] Ea={_Ea:.3f} eV  E_des={_Ed:.3f} eV  converged={_conv}')
    return {
        'Ea':         barrier.get('E_abs'),
        'E_des':      barrier.get('E_des'),
        'delta_E':    barrier.get('delta_E'),
        'ts_index':   barrier.get('ts_index'),
        'converged':  barrier.get('converged'),
        'fmax_final': barrier.get('fmax_final'),
        'rxn_coord':  rxn_coord,
        'energies':   energies,
    }


def plot_mep(
    rxn_coord: np.ndarray,
    energies: np.ndarray,
    Ea: float | None = None,
    delta_E: float | None = None,
    outfile: str | None = None,
    title: str = 'Minimum Energy Path',
) -> 'matplotlib.figure.Figure':
    """Plot the NEB minimum energy path with IS/TS/FS annotations.

    Parameters
    ----------
    rxn_coord : ndarray
        Reaction coordinate values (normalised arc length, 0→1).
    energies : ndarray
        Relative energies in eV along the MEP (IS = 0 by convention).
    Ea : float, optional
        Forward barrier in eV — drawn as an annotation arrow.
    delta_E : float, optional
        Reaction energy in eV — drawn as an annotation.
    outfile : str, optional
        If given, save the figure to this path.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Source: NB06 cell 30; NB09 cell 21.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(rxn_coord, energies, 'o-', color='tab:blue', lw=2, ms=6, zorder=5)

    # Mark IS, TS, FS
    ts_idx = int(np.argmax(energies))
    ax.scatter([rxn_coord[0]],   [energies[0]],   color='green',  s=80, zorder=6, label='IS')
    ax.scatter([rxn_coord[-1]],  [energies[-1]],  color='red',    s=80, zorder=6, label='FS')
    ax.scatter([rxn_coord[ts_idx]], [energies[ts_idx]], color='orange', s=100,
               marker='^', zorder=6, label=f'TS (image {ts_idx})')

    # Annotations
    if Ea is not None:
        ax.annotate(
            f'$E_a$ = {Ea:.3f} eV',
            xy=(rxn_coord[ts_idx], energies[ts_idx]),
            xytext=(rxn_coord[ts_idx] + 0.05, energies[ts_idx] + 0.02),
            fontsize=9, color='orange',
        )
    if delta_E is not None:
        ax.annotate(
            f'$\\Delta E$ = {delta_E:.3f} eV',
            xy=(rxn_coord[-1], energies[-1]),
            xytext=(rxn_coord[-1] - 0.15, energies[-1] + 0.02),
            fontsize=9, color='red',
        )

    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.set_xlabel('Reaction coordinate')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if outfile:
        os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved: {outfile}')

    return fig

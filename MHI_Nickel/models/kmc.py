"""
models/kmc.py
=============
Rejection-free BKL kinetic Monte Carlo on a 2D dual-layer alloy surface.

Models H permeation through a Hastelloy N slab:

  Gas H₂  ←→  H* surface  ←→  H subsurface  →  bulk (drain)

The surface layer is a 2D alloy grid (Ni/Mo/Cr/Fe).  Each grid point has a
corresponding subsurface-1 octahedral site directly beneath it (one-to-one,
from the topology built by ``subsurface_graph.py``).

Algorithm
---------
BKL (Bortz-Kalos-Lebowitz):
1. Build the complete event list from the current grid state.
2. Sum all rates → Q.
3. dt = −ln(u₁) / Q,  u₁ ~ U(0,1).
4. Select event i by binary search on the cumulative rate array (u₂ ~ U(0,Q)).
5. Execute the selected event (mutate grid in-place).
6. Repeat.

Random state
------------
``kmc_step`` draws from NumPy's global random state.  Set ``np.random.seed(n)``
before calling ``run_kmc`` / ``run_kmc_to_steady_state`` for reproducibility.

Rate dict format
----------------
The ``rate_dict`` argument consumed by ``build_event_list`` must follow this
structure (all rates in s⁻¹ unless noted):

.. code-block:: python

    rate_dict = {
        # H₂ → 2H* sticking probability (dimensionless Boltzmann factor)
        # Multiply by gas_strike_rate(P,T) inside build_event_list.
        # Derive as: exp(-Ea_diss_zpe / (kB * T)) from tst_rates output.
        'k_diss':      {('Ni', 'Ni'): float, ('Mo', 'Ni'): float, ...},

        # 2H* → H₂ desorption rate [s⁻¹]  (full TST: ν × exp(-Ea/kT))
        'k_des':       {('Ni', 'Ni'): float, ...},

        # H* surface diffusion rate [s⁻¹]  (optional; omit or leave empty)
        'k_surf_diff': {('Ni', 'Ni'): float, ...},

        # H* → H subsurface entry [s⁻¹]  (Hop A k_forward from build_rate_dict)
        'k_entry':     {'Ni': float, 'Mo': float, 'Cr': float, 'Fe': float},

        # H subsurface → H* surface exit [s⁻¹]  (Hop A k_reverse)
        'k_exit':      {'Ni': float, 'Mo': float, 'Cr': float, 'Fe': float},
    }

k_drain is not stored here — it is computed from ``drain_rate(D_m2s, a0_m)``
inside ``build_event_list``.
"""

import math

import numpy as np


# ── Physical constants ─────────────────────────────────────────────────────────
_M_H2_KG = 2.0 * 1.6735575e-27   # kg, molecular mass of H₂
_KB_J    = 1.380649e-23           # J/K


# ---------------------------------------------------------------------------
# Section 1 — Physics helpers
# ---------------------------------------------------------------------------

def gas_strike_rate(P_Pa: float, T_K: float, m_H2_kg: float, site_area_m2: float) -> float:
    """Hertz-Knudsen collision rate of H₂ on a single surface site [s⁻¹].

    .. math::

        R_{\\text{strike}} = \\frac{P}{\\sqrt{2\\pi m k_B T}} \\times A_{\\text{site}}

    Parameters
    ----------
    P_Pa : float
        H₂ partial pressure [Pa].
    T_K : float
        Temperature [K].
    m_H2_kg : float
        Molecular mass of H₂ [kg].  Use ``2 × 1.674e-27`` for H₂.
    site_area_m2 : float
        Area of one surface site [m²].  For FCC(111) use ``(a₀/√2)²``.

    Returns
    -------
    float
        Collision rate per site [s⁻¹].
    """
    return P_Pa * site_area_m2 / math.sqrt(2.0 * math.pi * m_H2_kg * _KB_J * T_K)


def drain_rate(D_m2s: float, a0_m: float) -> float:
    """Rate of H leaving the subsurface layer into the bulk [s⁻¹].

    Derived from 1-D Fick's law applied to a single hop of length
    dx = a₀/√2  (FCC oct–oct nearest-neighbour distance):

    .. math::

        k_{\\text{drain}} = D / dx^2, \\quad dx = a_0 / \\sqrt{2}

    D is taken directly from LAMMPS MSD calculations (no bulk NEB required).

    Parameters
    ----------
    D_m2s : float
        Bulk diffusivity [m²/s].
    a0_m : float
        FCC lattice constant [m].

    Returns
    -------
    float
        Drain rate constant [s⁻¹].
    """
    dx = a0_m / math.sqrt(2.0)
    return D_m2s / (dx * dx)


# ---------------------------------------------------------------------------
# Section 2 — Grid construction and queries
# ---------------------------------------------------------------------------

def make_grid(nx: int, ny: int, composition: dict | None = None, seed: int = 42) -> dict:
    """Create a fresh KMC grid state dict.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    composition : dict, optional
        Element probabilities, e.g. ``{'Ni':0.71,'Mo':0.16,'Cr':0.07,'Fe':0.06}``.
        Defaults to Hastelloy N composition.
    seed : int
        Seed for the element assignment RNG.

    Returns
    -------
    dict
        Keys: ``surface_elem`` (ndarray[nx,ny], dtype str),
        ``surface_occ`` (ndarray[nx,ny], int8, 0/1),
        ``subsurface_occ`` (ndarray[nx,ny], int8, 0/1),
        ``nx``, ``ny``.
    """
    if composition is None:
        composition = {'Ni': 0.71, 'Mo': 0.16, 'Cr': 0.07, 'Fe': 0.06}
    elems = list(composition.keys())
    probs = [composition[e] for e in elems]
    rng   = np.random.default_rng(seed)
    choices = rng.choice(elems, size=(nx, ny), p=probs)
    return {
        'surface_elem':   choices,
        'surface_occ':    np.zeros((nx, ny), dtype=np.int8),
        'subsurface_occ': np.zeros((nx, ny), dtype=np.int8),
        'nx': nx,
        'ny': ny,
    }


def grid_neighbors(i: int, j: int, nx: int, ny: int) -> list[tuple[int, int]]:
    """4-connected periodic neighbours of site (i, j)."""
    return [
        ((i - 1) % nx, j),
        ((i + 1) % nx, j),
        (i, (j - 1) % ny),
        (i, (j + 1) % ny),
    ]


def element_pair(grid: dict, i1: int, j1: int, i2: int, j2: int) -> tuple[str, str]:
    """Sorted (elem1, elem2) for symmetric rate dict lookup."""
    e1 = grid['surface_elem'][i1, j1]
    e2 = grid['surface_elem'][i2, j2]
    return (e1, e2) if e1 <= e2 else (e2, e1)


def surface_coverage(grid: dict) -> float:
    """θ = occupied surface sites / total surface sites."""
    return float(grid['surface_occ'].sum()) / (grid['nx'] * grid['ny'])


def subsurface_population(grid: dict) -> int:
    """Total H atoms in the subsurface layer."""
    return int(grid['subsurface_occ'].sum())


def subsurface_concentration(grid: dict, a0_m: float) -> float:
    """Subsurface H concentration C₀ [atoms/m³].

    Volume of the simulated subsurface layer:
    V = nx × a₀ × ny × a₀ × (a₀/√2)
    """
    N_sub = subsurface_population(grid)
    nx, ny = grid['nx'], grid['ny']
    vol    = nx * ny * (a0_m ** 3) / math.sqrt(2.0)
    return N_sub / vol if vol > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Section 3 — Event list builder
# ---------------------------------------------------------------------------

def build_event_list(
    grid: dict,
    rate_dict: dict,
    P_Pa: float,
    T_K: float,
    D_m2s: float,
    a0_m: float,
) -> list[dict]:
    """Enumerate all active events for the current grid state.

    Each event is a dict ``{'kind': str, 'sites': list, 'rate': float}``.

    Event kinds
    -----------
    ``'adsorb'``    : H₂ → 2H* on adjacent empty site pair (sites = 2 tuples)
    ``'desorb'``    : 2H* → H₂ from adjacent occupied pair (sites = 2 tuples)
    ``'surf_diff'`` : H* hops to adjacent empty site       (sites = [src, dst])
    ``'enter'``     : H* surface → H subsurface            (sites = 1 tuple)
    ``'exit'``      : H subsurface → H* surface            (sites = 1 tuple)
    ``'drain'``     : H subsurface → bulk                  (sites = 1 tuple)

    Parameters
    ----------
    grid : dict
        Grid state from :func:`make_grid`.
    rate_dict : dict
        Rate constants (see module docstring for format).
    P_Pa : float
        H₂ partial pressure [Pa].
    T_K : float
        Temperature [K].
    D_m2s : float
        Bulk diffusivity [m²/s] for drain rate.
    a0_m : float
        FCC lattice constant [m].

    Returns
    -------
    list of dict
        Complete event catalogue for the current grid state.
    """
    events: list[dict] = []

    nx      = grid['nx']
    ny      = grid['ny']
    s_occ   = grid['surface_occ']
    sub_occ = grid['subsurface_occ']

    # Precomputed scalars
    site_area  = (a0_m / math.sqrt(2.0)) ** 2
    R_str      = gas_strike_rate(P_Pa, T_K, _M_H2_KG, site_area)
    k_drain_v  = drain_rate(D_m2s, a0_m)

    k_diss = rate_dict.get('k_diss',      {})
    k_des  = rate_dict.get('k_des',       {})
    k_diff = rate_dict.get('k_surf_diff', {})
    k_ent  = rate_dict.get('k_entry',     {})
    k_ext  = rate_dict.get('k_exit',      {})

    for i in range(nx):
        for j in range(ny):
            occ_ij  = s_occ[i, j]
            sub_ij  = sub_occ[i, j]
            elem_ij = grid['surface_elem'][i, j]

            # ── Single-site events ─────────────────────────────────────────

            # H* → subsurface entry (surface occupied, subsurface empty)
            if occ_ij and not sub_ij:
                r = k_ent.get(elem_ij, 0.0)
                if r > 0.0:
                    events.append({'kind': 'enter', 'sites': [(i, j)], 'rate': r})

            # H subsurface → surface exit (subsurface occupied, surface empty)
            if sub_ij and not occ_ij:
                r = k_ext.get(elem_ij, 0.0)
                if r > 0.0:
                    events.append({'kind': 'exit', 'sites': [(i, j)], 'rate': r})

            # H subsurface → bulk drain (subsurface occupied, unconditional)
            if sub_ij and k_drain_v > 0.0:
                events.append({'kind': 'drain', 'sites': [(i, j)], 'rate': k_drain_v})

            # ── Pairwise events ────────────────────────────────────────────
            for (i2, j2) in grid_neighbors(i, j, nx, ny):
                occ_2  = s_occ[i2, j2]
                pair   = element_pair(grid, i, j, i2, j2)

                # Adsorption / desorption — symmetric, enumerate once per pair
                if (i2, j2) > (i, j):
                    if not occ_ij and not occ_2:
                        # H₂ adsorbs on two empty adjacent sites
                        sticking = k_diss.get(pair, 0.0)
                        r = R_str * sticking
                        if r > 0.0:
                            events.append({
                                'kind': 'adsorb',
                                'sites': [(i, j), (i2, j2)],
                                'rate': r,
                            })
                    elif occ_ij and occ_2:
                        # 2H* desorbs from two occupied adjacent sites
                        r = k_des.get(pair, 0.0)
                        if r > 0.0:
                            events.append({
                                'kind': 'desorb',
                                'sites': [(i, j), (i2, j2)],
                                'rate': r,
                            })

                # Surface diffusion — directional: src=(i,j) must be occupied
                if occ_ij and not occ_2:
                    r = k_diff.get(pair, 0.0)
                    if r > 0.0:
                        events.append({
                            'kind': 'surf_diff',
                            'sites': [(i, j), (i2, j2)],
                            'rate': r,
                        })

    return events


# ---------------------------------------------------------------------------
# Section 4 — Event execution (private)
# ---------------------------------------------------------------------------

def _execute_event(grid: dict, event: dict) -> None:
    """Mutate grid in-place according to event kind."""
    kind    = event['kind']
    sites   = event['sites']
    s_occ   = grid['surface_occ']
    sub_occ = grid['subsurface_occ']

    if kind == 'adsorb':
        (i1, j1), (i2, j2) = sites
        s_occ[i1, j1] = 1
        s_occ[i2, j2] = 1

    elif kind == 'desorb':
        (i1, j1), (i2, j2) = sites
        s_occ[i1, j1] = 0
        s_occ[i2, j2] = 0

    elif kind == 'surf_diff':
        (si, sj), (di, dj) = sites
        s_occ[si, sj] = 0
        s_occ[di, dj] = 1

    elif kind == 'enter':
        (i, j) = sites[0]
        s_occ[i, j]   = 0
        sub_occ[i, j] = 1

    elif kind == 'exit':
        (i, j) = sites[0]
        sub_occ[i, j] = 0
        s_occ[i, j]   = 1

    elif kind == 'drain':
        (i, j) = sites[0]
        sub_occ[i, j] = 0


# ---------------------------------------------------------------------------
# Section 5 — BKL step
# ---------------------------------------------------------------------------

def kmc_step(grid: dict, events: list[dict]) -> float:
    """Execute one BKL step; mutate grid in-place.  Return elapsed time dt [s].

    Uses NumPy's global random state — call ``np.random.seed(n)`` before
    ``run_kmc`` for reproducibility.

    Parameters
    ----------
    grid : dict
        Current grid state (mutated in-place).
    events : list of dict
        Event catalogue from :func:`build_event_list`.

    Returns
    -------
    float
        Time increment dt [s].  Returns 0.0 if event list is empty.
    """
    if not events:
        return 0.0

    rates      = np.array([e['rate'] for e in events], dtype=np.float64)
    Q          = rates.sum()
    if Q == 0.0:
        return 0.0

    dt         = -math.log(np.random.random()) / Q
    cumulative = np.cumsum(rates)
    idx        = int(np.searchsorted(cumulative, np.random.random() * Q))
    idx        = min(idx, len(events) - 1)

    _execute_event(grid, events[idx])
    return dt


# ---------------------------------------------------------------------------
# Section 6 — Run helpers
# ---------------------------------------------------------------------------

def run_kmc(
    grid: dict,
    rate_dict: dict,
    P_Pa: float,
    T_K: float,
    D_m2s: float,
    a0_m: float,
    n_steps: int,
) -> dict:
    """Run exactly *n_steps* BKL steps; return trajectory arrays.

    Parameters
    ----------
    grid : dict
        Grid state (mutated in-place during the run).
    rate_dict : dict
        Rate constants (see module docstring).
    P_Pa : float
        H₂ partial pressure [Pa].
    T_K : float
        Temperature [K].
    D_m2s : float
        Bulk diffusivity [m²/s].
    a0_m : float
        FCC lattice constant [m].
    n_steps : int
        Number of KMC steps to execute.

    Returns
    -------
    dict
        ``{'t_arr': ndarray, 'theta_arr': ndarray, 'n_sub_arr': ndarray}``
        Length n_steps + 1 (includes initial state at index 0).
    """
    t       = 0.0
    t_arr   = np.empty(n_steps + 1)
    th_arr  = np.empty(n_steps + 1)
    ns_arr  = np.empty(n_steps + 1, dtype=np.int32)

    t_arr[0]  = 0.0
    th_arr[0] = surface_coverage(grid)
    ns_arr[0] = subsurface_population(grid)

    for step in range(n_steps):
        events       = build_event_list(grid, rate_dict, P_Pa, T_K, D_m2s, a0_m)
        t           += kmc_step(grid, events)
        t_arr[step + 1]  = t
        th_arr[step + 1] = surface_coverage(grid)
        ns_arr[step + 1] = subsurface_population(grid)

    return {'t_arr': t_arr, 'theta_arr': th_arr, 'n_sub_arr': ns_arr}


def run_kmc_to_steady_state(
    grid: dict,
    rate_dict: dict,
    P_Pa: float,
    T_K: float,
    D_m2s: float,
    a0_m: float,
    window: int = 5000,
    rtol: float = 0.02,
    max_steps: int = 5_000_000,
) -> dict:
    """Run until surface coverage and subsurface population reach steady state.

    Convergence criterion: the rolling mean of θ and N_sub over the last
    *window* steps changes by less than *rtol* relative to the preceding
    window.

    Parameters
    ----------
    grid : dict
        Grid state (mutated in-place).
    rate_dict, P_Pa, T_K, D_m2s, a0_m : see :func:`run_kmc`.
    window : int
        Number of steps in each rolling-average window.  Default 5000.
    rtol : float
        Relative tolerance for convergence.  Default 0.02 (2 %).
    max_steps : int
        Hard cap on total steps.  Returns last state if never converges.

    Returns
    -------
    dict
        ``{'t_total': float, 'theta_ss': float, 'C0': float, 'n_steps': int,
           'converged': bool}``
    """
    t         = 0.0
    step      = 0
    th_hist: list[float] = []
    ns_hist: list[float] = []
    converged = False

    while step < max_steps:
        events  = build_event_list(grid, rate_dict, P_Pa, T_K, D_m2s, a0_m)
        t      += kmc_step(grid, events)
        step   += 1
        th_hist.append(surface_coverage(grid))
        ns_hist.append(float(subsurface_population(grid)))

        if step >= 2 * window:
            th_now  = float(np.mean(th_hist[-window:]))
            th_prev = float(np.mean(th_hist[-2 * window:-window]))
            ns_now  = float(np.mean(ns_hist[-window:]))
            ns_prev = float(np.mean(ns_hist[-2 * window:-window]))

            th_ok = abs(th_now - th_prev) <= rtol * (th_prev + 1e-12)
            ns_ok = abs(ns_now - ns_prev) <= rtol * (ns_prev + 1e-12)

            if th_ok and ns_ok:
                converged = True
                break

    theta_ss = float(np.mean(th_hist[-window:])) if len(th_hist) >= window else float(np.mean(th_hist))
    C0       = subsurface_concentration(grid, a0_m)

    return {
        't_total':  t,
        'theta_ss': theta_ss,
        'C0':       C0,
        'n_steps':  step,
        'converged': converged,
    }

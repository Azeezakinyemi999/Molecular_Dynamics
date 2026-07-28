"""
models/kmc.py
=============
Rejection-free BKL kinetic Monte Carlo on a 2D three-layer alloy slab.

Models H permeation through a Hastelloy N slab as a connected chain:

  Gas H₂  ⇄  H* surface  ⇄  H sub1 (oct)  ⇄  H sub2 (oct)  →  bulk (drain)

Each lateral grid point (i, j) is one surface unit cell with a sub1 octahedral
site directly beneath it and a sub2 octahedral site beneath that (one-to-one,
from the topology built by ``subsurface_graph.py``). A single H occupies one
layer at a time; every inter-layer move is column-local (same (i, j)).

This is the two-layer engine: the surface→sub1 (Hop A) and sub1→sub2 (Hop B)
entry/exit rates are resolved per **octahedral-site environment** (the
coordination fingerprint ``composition_label``, e.g. ``'Ni3MoCrFe_oct'``),
not collapsed to one rate per element. The irreversible drain to bulk happens
from **sub2**, the deepest explicit layer.

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
``make_grid`` uses its own seeded ``default_rng`` for the lattice realisation.

Rate dict format
----------------
The ``rate_dict`` argument consumed by ``build_event_list`` (all rates s⁻¹
unless noted):

.. code-block:: python

    rate_dict = {
        # Surface pair rates — keyed by a sorted element pair (unchanged).
        'k_diss':      {('Ni', 'Ni'): float, ('Mo', 'Ni'): float, ...},  # dimensionless sticking
        'k_des':       {('Ni', 'Ni'): float, ...},                       # 2H* → H₂ desorption
        'k_surf_diff': {('Ni', 'Ni'): float, ...},                       # optional H* surface hop

        # Inter-layer rates — keyed by OCTAHEDRAL-SITE ENVIRONMENT
        # (composition_label). Surface⇄sub1 uses the sub1 env; sub1⇄sub2 uses
        # the sub2 env. A grid cell whose env has no entry falls back to the
        # mean over that dict's known environments (never a silent 0.0).
        'k_entry':      {'Ni6_oct': float, 'Ni5Mo_oct': float, ...},  # Hop A fwd (surf→sub1)
        'k_exit':       {'Ni6_oct': float, ...},                      # Hop A rev (sub1→surf)
        'k_hopB_entry': {'Ni6_oct': float, ...},                      # Hop B fwd (sub1→sub2)
        'k_hopB_exit':  {'Ni6_oct': float, ...},                      # Hop B rev (sub2→sub1)
    }

Back-compatibility: if an inter-layer dict is keyed by bare element symbols
(the pre-two-layer convention) and the grid's env labels default to the
surface element (the degenerate ``make_grid`` case), the lookups still resolve.

k_drain is not stored here — it is computed from ``drain_rate(D_m2s, a0_m)``
inside ``build_event_list`` and applied to occupied sub2 sites.
"""

from __future__ import annotations

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
    """Rate of H leaving the sub2 layer into the bulk [s⁻¹].

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


def _rate_lookup(group: dict, key, fallback_mean: float) -> float:
    """Look up ``group[key]``, falling back to a precomputed mean, never 0.0-on-miss.

    ``group`` is one rate-class dict (e.g. ``k_entry``). If ``key`` (an env
    label) is present, its rate is returned; otherwise ``fallback_mean`` (the
    mean over that dict's known environments, precomputed once per
    ``build_event_list`` call) is returned. An empty ``group`` yields 0.0 —
    the only way this returns 0.0, and only when the rate class is genuinely
    absent, so a key/schema mismatch never silently produces an inert site.
    """
    if not group:
        return 0.0
    v = group.get(key)
    if v is not None:
        return float(v)
    return float(fallback_mean)


def _mean_of(group: dict) -> float:
    """Mean over a rate-class dict's values (0.0 if empty)."""
    if not group:
        return 0.0
    return float(sum(group.values()) / len(group))


# ---------------------------------------------------------------------------
# Section 2 — Grid construction and queries
# ---------------------------------------------------------------------------

def make_grid(
    nx: int,
    ny: int,
    composition: dict | None = None,
    seed: int = 42,
    sub1_env_composition: dict | None = None,
    sub2_env_composition: dict | None = None,
) -> dict:
    """Create a fresh three-layer KMC grid state dict.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    composition : dict, optional
        Surface element probabilities, e.g.
        ``{'Ni':0.71,'Mo':0.16,'Cr':0.07,'Fe':0.06}``.
        Defaults to Hastelloy N composition.
    seed : int
        Seed for the lattice realisation RNG (element + env draws).
    sub1_env_composition : dict, optional
        Oct-site environment probabilities for the sub1 layer, e.g.
        ``{'Ni6_oct':0.6,'Ni5Mo_oct':0.3,...}``. When ``None`` (default), the
        sub1 environment of each cell defaults to that cell's surface element —
        the degenerate case where env-keyed rate dicts reduce to element-keyed.
    sub2_env_composition : dict, optional
        Oct-site environment probabilities for the sub2 layer. Same default
        behaviour as ``sub1_env_composition``.

    Returns
    -------
    dict
        Keys: ``surface_elem`` (ndarray[nx,ny] str), ``surface_occ``,
        ``sub1_occ``, ``sub2_occ`` (ndarray[nx,ny] int8, 0/1),
        ``sub1_env``, ``sub2_env`` (ndarray[nx,ny] object, env labels),
        ``nx``, ``ny``.
    """
    if composition is None:
        composition = {'Ni': 0.71, 'Mo': 0.16, 'Cr': 0.07, 'Fe': 0.06}
    elems = list(composition.keys())
    probs = np.array([composition[e] for e in elems], dtype=float)
    probs = probs / probs.sum()
    rng   = np.random.default_rng(seed)
    surface_elem = rng.choice(elems, size=(nx, ny), p=probs)

    def _draw_env(env_comp):
        # object dtype avoids fixed-width truncation of long env labels
        # (e.g. 'Ni3MoCrFe_oct') — the exact silent-corruption class this
        # whole change is meant to eliminate.
        if not env_comp:
            return surface_elem.astype(object)
        labels = list(env_comp.keys())
        p = np.array([env_comp[k] for k in labels], dtype=float)
        p = p / p.sum()
        return rng.choice(labels, size=(nx, ny), p=p).astype(object)

    sub1_env = _draw_env(sub1_env_composition)
    sub2_env = _draw_env(sub2_env_composition)

    return {
        'surface_elem': surface_elem,
        'surface_occ':  np.zeros((nx, ny), dtype=np.int8),
        'sub1_occ':     np.zeros((nx, ny), dtype=np.int8),
        'sub2_occ':     np.zeros((nx, ny), dtype=np.int8),
        'sub1_env':     sub1_env,
        'sub2_env':     sub2_env,
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
    """Sorted (elem1, elem2) for symmetric surface-pair rate dict lookup."""
    e1 = grid['surface_elem'][i1, j1]
    e2 = grid['surface_elem'][i2, j2]
    return (e1, e2) if e1 <= e2 else (e2, e1)


def surface_coverage(grid: dict) -> float:
    """θ = occupied surface sites / total surface sites."""
    return float(grid['surface_occ'].sum()) / (grid['nx'] * grid['ny'])


def sub1_population(grid: dict) -> int:
    """Total H atoms in the sub1 layer."""
    return int(grid['sub1_occ'].sum())


def sub2_population(grid: dict) -> int:
    """Total H atoms in the sub2 layer."""
    return int(grid['sub2_occ'].sum())


def subsurface_population(grid: dict) -> int:
    """Total H atoms across both subsurface layers (sub1 + sub2)."""
    return sub1_population(grid) + sub2_population(grid)


def subsurface_concentration(grid: dict, a0_m: float, layer: str = 'sub2') -> float:
    """H concentration C₀ [atoms/m³] in a subsurface layer ('sub1' or 'sub2').

    Per-layer oct-site volume: V = nx × ny × a₀³ / √2. ``layer='sub1'`` is the
    first subsurface (dissolved-reference) layer; ``layer='sub2'`` (default) is
    the deeper layer that hands off to the bulk via the drain. Both are reported
    as diagnostics; the headline solubility is derived from the surface coverage
    via detailed balance, not from these rare-event occupancies.
    """
    n      = sub1_population(grid) if layer == 'sub1' else sub2_population(grid)
    nx, ny = grid['nx'], grid['ny']
    vol    = nx * ny * (a0_m ** 3) / math.sqrt(2.0)
    return n / vol if vol > 0.0 else 0.0


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
    """Enumerate all active events for the current three-layer grid state.

    Each event is a dict ``{'kind': str, 'sites': list, 'rate': float}``.

    Event kinds
    -----------
    ``'adsorb'``     : H₂ → 2H* on adjacent empty surface pair (2 tuples)
    ``'desorb'``     : 2H* → H₂ from adjacent occupied surface pair (2 tuples)
    ``'surf_diff'``  : H* hops to adjacent empty surface site (src, dst)
    ``'enter'``      : H* surface → sub1        (1 tuple; rate by sub1 env)
    ``'exit'``       : H sub1 → surface         (1 tuple; rate by sub1 env)
    ``'hopB_enter'`` : H sub1 → sub2            (1 tuple; rate by sub2 env)
    ``'hopB_exit'``  : H sub2 → sub1            (1 tuple; rate by sub2 env)
    ``'drain'``      : H sub2 → bulk            (1 tuple)

    Parameters
    ----------
    grid : dict
        Grid state from :func:`make_grid`.
    rate_dict : dict
        Rate constants (see module docstring for format).
    P_Pa, T_K, D_m2s, a0_m : float
        Pressure, temperature, bulk diffusivity, lattice constant.

    Returns
    -------
    list of dict
        Complete event catalogue for the current grid state.
    """
    events: list[dict] = []

    nx       = grid['nx']
    ny       = grid['ny']
    s_occ    = grid['surface_occ']
    s1_occ   = grid['sub1_occ']
    s2_occ   = grid['sub2_occ']
    sub1_env = grid['sub1_env']
    sub2_env = grid['sub2_env']

    # Precomputed scalars
    site_area  = (a0_m / math.sqrt(2.0)) ** 2
    R_str      = gas_strike_rate(P_Pa, T_K, _M_H2_KG, site_area)
    k_drain_v  = drain_rate(D_m2s, a0_m)

    k_diss  = rate_dict.get('k_diss',       {})
    k_des   = rate_dict.get('k_des',        {})
    k_diff  = rate_dict.get('k_surf_diff',  {})
    k_ent   = rate_dict.get('k_entry',      {})
    k_ext   = rate_dict.get('k_exit',       {})
    k_hbent = rate_dict.get('k_hopB_entry', {})
    k_hbext = rate_dict.get('k_hopB_exit',  {})

    # Fallback means precomputed once (outside the i/j loops — build_event_list
    # runs every KMC step, so recomputing per cell would dominate cost).
    # Surface-pair rates (k_diss/k_des/k_surf_diff) get the same treatment as
    # the inter-layer rates so a key/schema mismatch (e.g. a mislabelled
    # dissociation pair) can never silently zero adsorption on a known element.
    _m_ent   = _mean_of(k_ent)
    _m_ext   = _mean_of(k_ext)
    _m_hbent = _mean_of(k_hbent)
    _m_hbext = _mean_of(k_hbext)
    _m_diss  = _mean_of(k_diss)
    _m_des   = _mean_of(k_des)
    _m_diff  = _mean_of(k_diff)

    for i in range(nx):
        for j in range(ny):
            occ_s  = s_occ[i, j]
            occ_1  = s1_occ[i, j]
            occ_2  = s2_occ[i, j]
            env1   = sub1_env[i, j]
            env2   = sub2_env[i, j]

            # ── Inter-layer single-site events (column-local) ───────────────

            # Hop A forward: surface → sub1  (surface occupied, sub1 empty)
            if occ_s and not occ_1:
                r = _rate_lookup(k_ent, env1, _m_ent)
                if r > 0.0:
                    events.append({'kind': 'enter', 'sites': [(i, j)], 'rate': r})

            # Hop A reverse: sub1 → surface  (sub1 occupied, surface empty)
            if occ_1 and not occ_s:
                r = _rate_lookup(k_ext, env1, _m_ext)
                if r > 0.0:
                    events.append({'kind': 'exit', 'sites': [(i, j)], 'rate': r})

            # Hop B forward: sub1 → sub2  (sub1 occupied, sub2 empty)
            if occ_1 and not occ_2:
                r = _rate_lookup(k_hbent, env2, _m_hbent)
                if r > 0.0:
                    events.append({'kind': 'hopB_enter', 'sites': [(i, j)], 'rate': r})

            # Hop B reverse: sub2 → sub1  (sub2 occupied, sub1 empty)
            if occ_2 and not occ_1:
                r = _rate_lookup(k_hbext, env2, _m_hbext)
                if r > 0.0:
                    events.append({'kind': 'hopB_exit', 'sites': [(i, j)], 'rate': r})

            # Drain: sub2 → bulk  (sub2 occupied, unconditional)
            if occ_2 and k_drain_v > 0.0:
                events.append({'kind': 'drain', 'sites': [(i, j)], 'rate': k_drain_v})

            # ── Pairwise surface events ─────────────────────────────────────
            for (i2, j2) in grid_neighbors(i, j, nx, ny):
                occ_s2 = s_occ[i2, j2]
                pair   = element_pair(grid, i, j, i2, j2)

                # Adsorption / desorption — symmetric, enumerate once per pair
                if (i2, j2) > (i, j):
                    if not occ_s and not occ_s2:
                        sticking = _rate_lookup(k_diss, pair, _m_diss)
                        r = R_str * sticking
                        if r > 0.0:
                            events.append({
                                'kind': 'adsorb',
                                'sites': [(i, j), (i2, j2)],
                                'rate': r,
                            })
                    elif occ_s and occ_s2:
                        r = _rate_lookup(k_des, pair, _m_des)
                        if r > 0.0:
                            events.append({
                                'kind': 'desorb',
                                'sites': [(i, j), (i2, j2)],
                                'rate': r,
                            })

                # Surface diffusion — directional: src=(i,j) must be occupied
                if occ_s and not occ_s2:
                    r = _rate_lookup(k_diff, pair, _m_diff)
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
    s1_occ  = grid['sub1_occ']
    s2_occ  = grid['sub2_occ']

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

    elif kind == 'enter':               # surface → sub1
        (i, j) = sites[0]
        s_occ[i, j]  = 0
        s1_occ[i, j] = 1

    elif kind == 'exit':                # sub1 → surface
        (i, j) = sites[0]
        s1_occ[i, j] = 0
        s_occ[i, j]  = 1

    elif kind == 'hopB_enter':          # sub1 → sub2
        (i, j) = sites[0]
        s1_occ[i, j] = 0
        s2_occ[i, j] = 1

    elif kind == 'hopB_exit':           # sub2 → sub1
        (i, j) = sites[0]
        s2_occ[i, j] = 0
        s1_occ[i, j] = 1

    elif kind == 'drain':               # sub2 → bulk
        (i, j) = sites[0]
        s2_occ[i, j] = 0


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
    rate_dict, P_Pa, T_K, D_m2s, a0_m : see module docstring / :func:`build_event_list`.
    n_steps : int
        Number of KMC steps to execute.

    Returns
    -------
    dict
        ``{'t_arr': ndarray, 'theta_arr': ndarray, 'n_sub_arr': ndarray}``
        Length n_steps + 1 (includes initial state at index 0).
        ``n_sub_arr`` is the total subsurface population (sub1 + sub2).
    """
    t       = 0.0
    t_arr   = np.empty(n_steps + 1)
    th_arr  = np.empty(n_steps + 1)
    ns_arr  = np.empty(n_steps + 1, dtype=np.int32)

    t_arr[0]  = 0.0
    th_arr[0] = surface_coverage(grid)
    ns_arr[0] = subsurface_population(grid)

    _print_every = max(1, n_steps // 10)
    for step in range(n_steps):
        events       = build_event_list(grid, rate_dict, P_Pa, T_K, D_m2s, a0_m)
        t           += kmc_step(grid, events)
        t_arr[step + 1]  = t
        th_arr[step + 1] = surface_coverage(grid)
        ns_arr[step + 1] = subsurface_population(grid)
        if (step + 1) % _print_every == 0:
            print(f'  [KMC] step {step+1:>8d}/{n_steps}  t={t:.3e} s  θ={th_arr[step+1]:.4f}  n_sub={int(ns_arr[step+1])}')

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

    Convergence criterion: the rolling mean of θ and total N_sub (sub1 + sub2)
    over the last *window* steps changes by less than *rtol* relative to the
    preceding window.

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
        ``{'t_total': float, 'theta_ss': float, 'C0': float, 'C0_sub1': float,
           'C0_sub2': float, 'n_steps': int, 'converged': bool}``. ``C0`` aliases
           ``C0_sub2`` (back-compat); both subsurface layers are reported as
           diagnostics.
    """
    t         = 0.0
    step      = 0
    th_hist: list[float] = []
    ns_hist: list[float] = []
    ns1_hist: list[float] = []
    ns2_hist: list[float] = []
    converged = False

    while step < max_steps:
        events  = build_event_list(grid, rate_dict, P_Pa, T_K, D_m2s, a0_m)
        t      += kmc_step(grid, events)
        step   += 1
        th_hist.append(surface_coverage(grid))
        ns_hist.append(float(subsurface_population(grid)))
        ns1_hist.append(float(sub1_population(grid)))
        ns2_hist.append(float(sub2_population(grid)))

        if step >= 2 * window:
            th_now  = float(np.mean(th_hist[-window:]))
            th_prev = float(np.mean(th_hist[-2 * window:-window]))
            ns_now  = float(np.mean(ns_hist[-window:]))
            ns_prev = float(np.mean(ns_hist[-2 * window:-window]))

            th_ok = abs(th_now - th_prev) <= rtol * (th_prev + 1e-12)
            ns_ok = abs(ns_now - ns_prev) <= rtol * (ns_prev + 1e-12)

            if (step // window) % 50 == 0:
                print(f'  [KMC] step={step:>7d}  t={t:.3e} s  θ={th_now:.4f}  n_sub={ns_now:.1f}  Δθ={abs(th_now-th_prev):.4f}')

            if th_ok and ns_ok:
                converged = True
                break

    theta_ss = float(np.mean(th_hist[-window:])) if len(th_hist) >= window else float(np.mean(th_hist))
    # C0_sub1/C0_sub2 are STEADY-STATE (time-averaged) concentrations over the
    # same window as theta_ss -- not single snapshots. Both subsurface layers
    # hold few atoms, so these are noise-limited DIAGNOSTICS; the headline
    # solubility is derived from theta_ss via detailed balance, not from these.
    _win = lambda h: (h[-window:] if len(h) >= window else h)
    _n1 = _win(ns1_hist); _n2 = _win(ns2_hist)
    _N1 = float(np.mean(_n1)) if _n1 else 0.0
    _N2 = float(np.mean(_n2)) if _n2 else 0.0
    _vol = grid['nx'] * grid['ny'] * (a0_m ** 3) / math.sqrt(2.0)
    C0_sub1 = _N1 / _vol if _vol > 0.0 else 0.0
    C0_sub2 = _N2 / _vol if _vol > 0.0 else 0.0

    print(f'[KMC] converged={converged}  steps={step}  t={t:.3e} s  θ={theta_ss:.4f}  '
          f'C0(sub1)={C0_sub1:.3e}  C0(sub2)={C0_sub2:.3e} atoms/m³')
    return {
        't_total':  t,
        'theta_ss': theta_ss,
        'C0':        C0_sub2,   # back-compat alias for sub2 (deepest explicit layer)
        'C0_sub1':   C0_sub1,
        'C0_sub2':   C0_sub2,
        'n_steps':  step,
        'converged': converged,
    }

#!/usr/bin/env python3
"""
models/plots/solubility_plot.py
===============================
Sieverts solubility in Arrhenius form, for the three energy/rate routes that do
not depend on KMC.

Routes plotted
--------------
``geometric``         S₀ from the FCC oct-site density, ``4/a0³/N_A``
``vibrational``       S₀ from dissolved-H partition functions
``detailed_balance``  no S₀ — ``k_entry/k_exit`` per environment instead

All three share one enthalpy distribution: ``dH_sol_by_env.json`` is consumed by
every route, so they differ only in how the prefactor is obtained. The KMC
routes (``kmc``, ``kmc_theta``) and ``solubility_arrhenius_kmc.json`` are
deliberately excluded — see ``permeation_workflow.py``'s ``solubility_headline``.

Site-saturation guard
---------------------
Solubility at 1 Pa cannot exceed the density of sites available to hold H. The
oct-site density ``4/a0³/N_A`` is read from ``lattice_params_vs_T.json`` and
drawn on every figure; any route crossing it is flagged here and in the console.
This is not decoration — it is currently breached by every Ni-based system,
because ``solubility_by_environment`` uses an unbounded ``exp(−ΔH/kT)`` with no
Langmuir cap, so a single exothermic environment can dominate without limit.

Figures
-------
  * ``<outdir>/<material>_solubility_arrhenius.png``   routes overlaid, per material
  * ``<outdir>/solubility_arrhenius_all_materials.png``  one panel per route

Solubility is an equilibrium property derived from NEB energetics, not from the
MD, so it does not vary with H loading: Al's 1H/3H/5H files are byte-identical.
One figure is emitted per material, and the console says when concentrations
agree or differ.

Typical usage
-------------
::

    python models/plots/solubility_plot.py                 # every material
    python models/plots/solubility_plot.py --pattern 'Al*'
    python models/plots/solubility_plot.py --routes geometric vibrational
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from dataclasses import dataclass

import numpy as np

from diffusivity_plot import (            # noqa: E402
    DEFAULT_RESULTS_DIR, Run, discover_runs, pretty_material,
    _add_temperature_axis, _save,
)
from models.diffusivity_post_processing import KB_EV                  # noqa: E402
from models.permeation import solubility_by_environment_saturating    # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

N_A       = 6.02214076e23
S_UNITS   = 'mol H m⁻³ Pa⁻⁰·⁵'

NON_KMC_ROUTES = ('geometric', 'vibrational', 'detailed_balance')

# Same threshold classify_sieverts_regime uses for "still dilute". Below it the
# Boltzmann and Langmuir forms agree and only one curve is worth drawing.
DILUTE_THETA_MAX = 0.4

# marker, linestyle, colour, is_headline — matching the project's own map in
# permeation_workflow.py so figures from either place read the same.
ROUTE_STYLE = {
    'geometric':        ('o', '-',  'steelblue', True),
    'vibrational':      ('s', '-',  'coral',     True),
    'detailed_balance': ('^', '--', 'seagreen',  False),
}

ROUTE_LABEL = {
    'geometric':        'geometric  ($S_0 = 4/a_0^3/N_A$)',
    'vibrational':      'vibrational  ($S_0$ from partition functions)',
    'detailed_balance': 'detailed balance  ($k_{entry}/k_{exit}$)',
}


# ---------------------------------------------------------------------------
# Section 2 — Loading
# ---------------------------------------------------------------------------

@dataclass
class RouteFit:
    """One solubility route's Arrhenius fit for one material."""

    name:       str
    T:          np.ndarray      # K
    S:          np.ndarray      # mol H m^-3 Pa^-0.5
    S0:         float
    dH:         float           # eV
    dH_err:     float | None
    S0_rel_err: float | None
    r2:         float | None

    def curve(self, T: np.ndarray) -> np.ndarray:
        """S(T) = S0 · exp(−ΔH/kT) — the fit as recorded."""
        return self.S0 * np.exp(-self.dH / (KB_EV * np.asarray(T, float)))


def load_solubility(run: Run, routes: tuple[str, ...]) -> dict[str, RouteFit]:
    """Read the non-KMC routes out of ``solubility_arrhenius.json``."""
    path = os.path.join(run.path, 'solubility_arrhenius.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f'  ! {run.name}: unreadable solubility file ({exc})')
        return {}

    out: dict[str, RouteFit] = {}
    for name in routes:
        r = (raw.get('routes') or {}).get(name)
        if not r or not r.get('available'):
            continue
        T = np.asarray(r.get('T_K_arr', []), float)
        S = np.asarray(r.get('S_arr', []), float)
        if T.size == 0 or T.size != S.size or r.get('S0') is None:
            print(f'  ! {run.name}/{name}: malformed, skipped')
            continue
        good = S > 0
        if not good.any():
            print(f'  ! {run.name}/{name}: no positive S, skipped')
            continue
        order = np.argsort(T[good])
        out[name] = RouteFit(
            name=name, T=T[good][order], S=S[good][order],
            S0=float(r['S0']), dH=float(r.get('dH_sol_eV', 0.0)),
            dH_err=r.get('dH_sol_err_eV'), S0_rel_err=r.get('S0_rel_err'),
            r2=r.get('r2'),
        )
    return out


def site_density(results_dir: str, stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(T, oct-site density) in mol m⁻³ from ``lattice_params_vs_T.json``.

    This is the ceiling on S at 1 Pa: one H per octahedral site is full
    occupancy, and Sieverts' law has no meaning beyond it.
    """
    path = os.path.join(results_dir, stem, 'lattice_params_vs_T.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            raw = json.load(fh)
        T  = np.asarray(raw['temperatures'], float)
        a0 = np.asarray(raw['a0_m'], float)
    except (OSError, ValueError, KeyError):
        return None
    if T.size == 0 or T.size != a0.size:
        return None
    return T, 4.0 / a0 ** 3 / N_A


def load_environments(results_dir: str, stem: str) -> dict | None:
    """Per-environment ΔH_sol for a material, or None if not recorded."""
    path = os.path.join(results_dir, stem, 'dH_sol_by_env.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            env = json.load(fh)
    except (OSError, ValueError):
        return None
    return env or None


def saturating_curve(fit: RouteFit, env: dict | None,
                     ceiling: tuple[np.ndarray, np.ndarray] | None,
                     T: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Occupancy-limited S(T) recomputed from ΔH_sol per environment.

    Derived here rather than read from the payload so it works on the results
    already on disk — every existing ``solubility_arrhenius.json`` predates the
    saturating route being recorded, and rerunning the pipeline is not needed
    just to see how far the dilute limit has been pushed.
    """
    if env is None or ceiling is None:
        return None
    T_c, n_c = ceiling
    S_out, th_out = [], []
    # Both helpers log a line per call; a 200-point curve would flood the run.
    with contextlib.redirect_stdout(io.StringIO()):
        for t in T:
            rho = float(np.interp(t, T_c, n_c))
            try:
                r = solubility_by_environment_saturating(env, fit.S0, rho,
                                                         float(t))
            except (ValueError, ZeroDivisionError):
                return None
            S_out.append(r['S'])
            th_out.append(r['theta_max'])
    return np.asarray(S_out), np.asarray(th_out)


def breaches(fit: RouteFit, ceiling: tuple[np.ndarray, np.ndarray] | None
             ) -> float | None:
    """Worst factor by which this route exceeds the site-density ceiling."""
    if ceiling is None:
        return None
    T_c, n_c = ceiling
    worst = 0.0
    for T, S in zip(fit.T, fit.S):
        n = float(np.interp(T, T_c, n_c))
        if n > 0 and S / n > worst:
            worst = S / n
    return worst if worst > 1.0 else None


# ---------------------------------------------------------------------------
# Section 3 — Grouping by material
# ---------------------------------------------------------------------------

def _signature(fits: dict[str, RouteFit]) -> tuple:
    """Fingerprint used to tell whether two concentrations really differ."""
    return tuple((n, round(f.S0, 12), round(f.dH, 12)) for n, f in sorted(fits.items()))


def group_by_material(runs: list[Run], routes: tuple[str, ...]
                      ) -> dict[str, tuple[Run, dict[str, RouteFit]]]:
    """One representative run per material, reporting whether loadings agree."""
    by_stem: dict[str, list[tuple[Run, dict[str, RouteFit]]]] = {}
    for run in runs:
        fits = load_solubility(run, routes)
        if fits:
            by_stem.setdefault(run.stem, []).append((run, fits))

    chosen: dict[str, tuple[Run, dict[str, RouteFit]]] = {}
    for stem, entries in by_stem.items():
        entries.sort(key=lambda e: e[0].n_H)
        sigs = {_signature(f) for _r, f in entries}
        loads = ', '.join(f'{r.n_H}H' for r, _f in entries)
        if len(sigs) == 1 and len(entries) > 1:
            print(f'  · {pretty_material(stem)}: {loads} are identical '
                  f'(solubility comes from energetics, not the MD) — '
                  f'plotting one curve set')
        elif len(entries) > 1:
            print(f'  ! {pretty_material(stem)}: {loads} DIFFER — '
                  f'plotting {entries[0][0].n_H}H only; rerun per loading if '
                  f'that is unexpected')
        chosen[stem] = entries[0]
    return chosen


# ---------------------------------------------------------------------------
# Section 4 — Per-material figure
# ---------------------------------------------------------------------------

def _legend_label(fit: RouteFit, headline: bool) -> str:
    """Route name plus its fitted parameters, all in the legend.

    Kept in the legend rather than a separate caption: a caption block sized
    from the row count collides with the legend as soon as three routes are
    present, and the legend already reserves space for itself.
    """
    txt = f'{ROUTE_LABEL[fit.name]}'
    if not headline:
        txt += '  [diagnostic]'
    txt += f'\n     $S_0$={fit.S0:.3e}'
    if fit.S0_rel_err:
        txt += f' ±{fit.S0_rel_err * 100:.1f}%'
    txt += f',  Δ$H$={fit.dH:+.4f}'
    if fit.dH_err is not None:
        txt += f'±{fit.dH_err:.4f}'
    txt += ' eV'
    if fit.r2 is not None:
        txt += f',  $r^2$={fit.r2:.5f}'
    return txt


def plot_material(stem: str, fits: dict[str, RouteFit],
                  ceiling: tuple[np.ndarray, np.ndarray] | None,
                  outfile: str, env: dict | None = None):
    """log₁₀(S) vs 1000/T for every non-KMC route of one material."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    T_lo = min(f.T.min() for f in fits.values())
    T_hi = max(f.T.max() for f in fits.values())
    T_fine = np.linspace(T_lo * 0.95, T_hi * 1.05, 200)

    flagged: list[str] = []
    saturating_shown = False
    for name, fit in sorted(fits.items(), key=lambda kv: list(ROUTE_STYLE).index(kv[0])):
        marker, ls, colour, headline = ROUTE_STYLE[name]
        ax.plot(1000.0 / fit.T, np.log10(fit.S), marker, color=colour, ms=7,
                mfc=colour if headline else 'none', mew=1.5, zorder=5)
        ax.plot(1000.0 / T_fine, np.log10(fit.curve(T_fine)), ls, color=colour,
                lw=1.6, alpha=0.9, zorder=4,
                label=_legend_label(fit, headline))
        if breaches(fit, ceiling):
            flagged.append(name)

        # Occupancy-limited counterpart, recomputed from ΔH per environment.
        # Drawn only once θ leaves the dilute window — comparing curves
        # numerically would always differ slightly, because the plotted line is
        # the fitted Arrhenius law while this is the per-environment sum.
        sat = saturating_curve(fit, env, ceiling, T_fine)
        if sat is not None:
            S_sat, theta = sat
            ok = (S_sat > 0) & (theta >= DILUTE_THETA_MAX)
            if ok.any():
                ax.plot(1000.0 / T_fine[ok], np.log10(S_sat[ok]), ':',
                        color=colour, lw=2.0, alpha=0.95, zorder=6)
                saturating_shown = True

    if ceiling is not None:
        T_c, n_c = ceiling
        ax.plot(1000.0 / T_c, np.log10(n_c), 'k--', lw=1.4, zorder=3,
                label='oct-site saturation  ($4/a_0^3/N_A$)')
        ax.fill_between(1000.0 / T_c, np.log10(n_c), ax.get_ylim()[1],
                        color='red', alpha=0.06, zorder=0)
    if saturating_shown:
        ax.plot([], [], ':', color='0.35', lw=2.0,
                label='dotted: occupancy-limited (Langmuir θ, same colour)')

    _add_temperature_axis(ax)
    ax.set_xlabel('1000 / T  (K⁻¹)')
    ax.set_ylabel(f'log₁₀( S / {S_UNITS} )')
    title = f'H solubility (Sieverts) — {pretty_material(stem)}'
    if flagged:
        title += '\n⚠ ' + ', '.join(flagged) + ' exceed site saturation'
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

    # Each route contributes two legend lines (name, then its parameters), plus
    # one for the saturation line.
    rows     = 2 * len(fits) + (1 if ceiling is not None else 0) \
               + (1 if saturating_shown else 0)
    fig_h    = 5.4 + rows * 0.21
    fig.set_size_inches(7.6, fig_h)
    reserved = min(0.45, (rows * 0.22 + 0.35) / fig_h)
    fig.tight_layout(rect=(0, reserved, 1, 1))
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, reserved), fontsize=8,
               frameon=True, borderaxespad=0.0, labelspacing=0.7)
    _save(fig, outfile)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 5 — Cross-material figure
# ---------------------------------------------------------------------------

def plot_all_materials(chosen: dict[str, tuple[Run, dict[str, RouteFit]]],
                       ceilings: dict[str, tuple[np.ndarray, np.ndarray] | None],
                       routes: tuple[str, ...], outfile: str):
    """One panel per route, every material overlaid."""
    import matplotlib.pyplot as plt

    present = [r for r in routes
               if any(r in fits for _run, fits in chosen.values())]
    if not present or len(chosen) < 2:
        return None

    stems  = sorted(chosen)
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(stems)))
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.9),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, route in zip(axes, present):
        for stem, colour in zip(stems, colours):
            fits = chosen[stem][1]
            if route not in fits:
                continue
            fit = fits[route]
            T_fine = np.linspace(fit.T.min() * 0.95, fit.T.max() * 1.05, 200)
            ax.plot(1000.0 / fit.T, np.log10(fit.S), 'o', color=colour, ms=6,
                    zorder=5)
            ax.plot(1000.0 / T_fine, np.log10(fit.curve(T_fine)), '-',
                    color=colour, lw=1.5, zorder=4,
                    label=f'{pretty_material(stem)}  Δ$H$={fit.dH:+.3f} eV')
            ceil = ceilings.get(stem)
            if ceil is not None:
                ax.plot(1000.0 / ceil[0], np.log10(ceil[1]), '--',
                        color=colour, lw=1.0, alpha=0.55, zorder=3)
        ax.set_title(f'{route}\n(dashed = that material’s site saturation)',
                     fontsize=9.5)
        ax.set_xlabel('1000 / T  (K⁻¹)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc='best')

    axes[0].set_ylabel(f'log₁₀( S / {S_UNITS} )')
    fig.suptitle('H solubility by route and material — '
                 'non-KMC routes only', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, outfile)
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Section 6 — Console report
# ---------------------------------------------------------------------------

def report(chosen: dict[str, tuple[Run, dict[str, RouteFit]]],
           ceilings: dict[str, tuple[np.ndarray, np.ndarray] | None],
           envs: dict[str, dict | None]) -> list[str]:
    """Print S0/ΔH/r² per route, with the saturation verdict and peak θ."""
    print(f'\n  S in {S_UNITS};  θ_max is the peak per-environment occupancy '
          f'at P_ref = 1 Pa')
    hdr = (f"  {'material':<18s}{'route':<18s}{'S0':>12s}{'dH (eV)':>10s}"
           f"{'S_max':>12s}{'θ_max':>8s}{'S saturating':>14s}  verdict")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    problems: list[str] = []
    for stem in sorted(chosen):
        _run, fits = chosen[stem]
        ceil, env = ceilings.get(stem), envs.get(stem)
        for name in ROUTE_STYLE:
            fit = fits.get(name)
            if fit is None:
                print(f'  {pretty_material(stem):<18s}{name:<18s}'
                      f'{"-":>12s}{"-":>10s}{"-":>12s}{"-":>8s}{"-":>14s}  absent')
                continue
            over = breaches(fit, ceil)
            verdict = ('no ceiling data' if ceil is None
                       else f'OVER x{over:.0e}' if over else 'ok')
            if over:
                problems.append(f'{pretty_material(stem)}/{name} '
                                f'(x{over:.0e} over saturation)')

            sat = saturating_curve(fit, env, ceil, fit.T)
            th  = f'{sat[1].max():.3f}' if sat is not None else '-'
            s_s = f'{sat[0].max():.3e}' if sat is not None else '-'
            print(f'  {pretty_material(stem):<18s}{name:<18s}{fit.S0:>12.3e}'
                  f'{fit.dH:>+10.4f}{max(fit.S):>12.3e}{th:>8s}{s_s:>14s}'
                  f'  {verdict}')
    return problems


# ---------------------------------------------------------------------------
# Section 7 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='Plot Sieverts solubility in Arrhenius form (non-KMC routes).')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                    help='directory holding the run folders '
                         '(default: calculation/results)')
    ap.add_argument('--pattern', default='*',
                    help="glob for run folders, e.g. 'Al*' (default: '*')")
    ap.add_argument('--outdir', default=None,
                    help='where figures go (default: <results-dir>/plots)')
    ap.add_argument('--routes', nargs='+', default=list(NON_KMC_ROUTES),
                    choices=list(NON_KMC_ROUTES),
                    help='which non-KMC routes to plot '
                         '(default: all three). KMC routes are never plotted.')
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))
    routes      = tuple(args.routes)

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    runs = discover_runs(results_dir, args.pattern)
    if not runs:
        print(f"No H-loaded run directories matched '{args.pattern}'")
        return 1

    print(f'\nGrouping {len(runs)} run(s) by material …')
    chosen = group_by_material(runs, routes)
    if not chosen:
        print('No run has a solubility_arrhenius.json with a non-KMC route.')
        return 1

    ceilings = {stem: site_density(results_dir, stem) for stem in chosen}
    missing  = [pretty_material(s) for s, c in ceilings.items() if c is None]
    if missing:
        print(f'  ! no lattice_params_vs_T.json for {", ".join(missing)} — '
              f'saturation ceiling unavailable there')

    envs     = {stem: load_environments(results_dir, stem) for stem in chosen}
    problems = report(chosen, ceilings, envs)

    print('\nPer-material plots:')
    for stem, (_run, fits) in sorted(chosen.items()):
        material = pretty_material(stem).replace(' ', '_')
        plot_material(stem, fits, ceilings.get(stem),
                      os.path.join(outdir, f'{material}_solubility_arrhenius.png'),
                      env=envs.get(stem))

    print('\nCross-material plot:')
    if plot_all_materials(chosen, ceilings, routes,
                          os.path.join(outdir,
                                       'solubility_arrhenius_all_materials.png')
                          ) is None:
        print('  · needs at least 2 materials — skipped')

    if problems:
        print(f'\n{len(problems)} route(s) exceed octahedral-site saturation:')
        for p in problems:
            print(f'  · {p}')
        print('S cannot exceed one H per site at 1 Pa. Two known causes:\n'
              '  (a) files generated before commit ed5bb11 (2026-07-28) carry an\n'
              '      S0 a factor of Avogadro too large — rerun Phase 6;\n'
              '  (b) solubility_by_environment uses an unbounded exp(-dH/kT) with\n'
              '      no Langmuir cap, so one exothermic environment can dominate\n'
              '      without limit.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

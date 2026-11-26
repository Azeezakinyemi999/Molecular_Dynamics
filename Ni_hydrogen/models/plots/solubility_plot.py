#!/usr/bin/env python3
"""
models/plots/solubility_plot.py
===============================
Sieverts solubility in Arrhenius form, for the three energy/rate routes.

Routes plotted
--------------
``geometric``         S₀ from the FCC oct-site density, ``4/a0³/N_A``
``vibrational``       S₀ from dissolved-H partition functions
``detailed_balance``  no S₀ — ``k_entry/k_exit`` per environment instead

All three share one enthalpy distribution: ``dH_sol_by_env.json`` is consumed by
every route, so they differ only in how the prefactor is obtained. ``geometric``
and ``vibrational`` are the headline routes; ``detailed_balance`` is a rate-based
cross-check — see ``permeation_workflow.py``'s ``solubility_headline``.

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

import figstyle as fs                     # noqa: E402
from diffusivity_plot import (            # noqa: E402
    DEFAULT_RESULTS_DIR, S_LIT_FILE, Run, discover_runs, pretty_material,
    _add_temperature_axis, _save, _tex, draw_literature, literature_curves, literature_for,
    literature_notes, literature_window, load_aliases, load_solubility_literature, name_literature,
)
from models.diffusivity_post_processing import KB_EV                  # noqa: E402
from models.permeation import solubility_by_environment_saturating    # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

N_A       = 6.02214076e23
S_UNITS   = 'mol H m⁻³ Pa⁻⁰·⁵'
# The literature quotes Ks per mol of H2; this work per mol of H atoms, so a literature law is doubled.
H_PER_H2    = 2.0
LIT_DECADES = 10.0          # literature laws further than this from the MD values are named, not drawn

ROUTES = ('geometric', 'vibrational', 'detailed_balance')

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
    """Read the solubility routes out of ``solubility_arrhenius.json``."""
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

ROUTE_WORDS = {'geometric': r'geometric ($S_0 = 4/a_0^3 N_A$)',
               'vibrational': r'vibrational ($S_0$ from partition functions)',
               'detailed_balance': r'detailed balance ($k_\mathrm{entry}/k_\mathrm{exit}$, a diagnostic)'}
MODULE = 'plots/solubility_plot.py'
SAT_LINE = dict(color='black', lw=2.4, ls=(0, (6, 2, 1, 2)))


def plot_material(stem: str, fits: dict[str, RouteFit],
                  ceiling: tuple[np.ndarray, np.ndarray] | None,
                  outfile: str, env: dict | None = None, run=None, lit=None):
    """S against 1000/T for every route of one material, with octahedral-site saturation: one figure
    written as PDF, PNG and caption. ``lit``: the material's published laws
    (``load_solubility_literature``), drawn as grey dashed lines converted to mol H and named by
    their first author, as on the diffusivity plots."""
    from matplotlib.lines import Line2D
    from permeability_plot import _route_line, _loading_words
    fig, ax = fs.new_figure()
    ys, xs, items, handles, dotted = [], [], [], [], []
    order = sorted(fits, key=lambda n: list(fs.ROUTE_STYLE).index(n))
    for name in order:
        fit = fits[name]
        T_fit = np.linspace(fit.T.min(), fit.T.max(), 120)
        handles.append(_route_line(ax, name, fit.T, fit.S, T_fit, fit.curve(T_fit)))
        ys += [fit.S, fit.curve(T_fit)]
        xs.append(1000.0 / fit.T)
        i = int(np.argmin(fit.T))
        items.append((1000.0 / fit.T[i], fit.S[i], fs.ROUTE_STYLE[name][0], fs.ROUTE_STYLE[name][1]))
        sat = saturating_curve(fit, env, ceiling, T_fit)
        if sat is not None:
            S_sat, theta = sat
            ok = (S_sat > 0) & (theta >= DILUTE_THETA_MAX)
            if ok.any():
                ax.plot(1000.0 / T_fit[ok], S_sat[ok], color=fs.ROUTE_STYLE[name][1], ls=(0, (1, 1.5)),
                        lw=fs.CURVE_PT, zorder=6)
                dotted.append(name)
                ys.append(S_sat[ok])
    if ceiling is not None:
        T_c, n_c = ceiling
        ax.plot(1000.0 / T_c, n_c, zorder=3, **SAT_LINE)
        ys.append(n_c)
        i = int(np.argmin(T_c))
        items.append((1000.0 / T_c[i], n_c[i], 'Site saturation', 'black'))
        handles.append(Line2D([], [], **SAT_LINE))
    T_all = np.concatenate([fits[n].T for n in fits])
    lit_curves, left_out = [], []
    if lit:
        lo, hi, lit_curves, left_out = literature_window(np.concatenate([np.asarray(v, float) for v in ys]),
                                                         literature_curves(lit, scale=H_PER_H2), LIT_DECADES)
        ys += [y[(np.log10(y) >= lo) & (np.log10(y) <= hi)] for _e, _T, y in lit_curves]
    fs.inverse_temperature_axis(ax, np.concatenate([T_all] + [T for _e, T, _y in lit_curves]))
    fs.set_log_y(ax, np.concatenate([np.asarray(v, float) for v in ys]))
    if ceiling is not None:
        ax.fill_between(1000.0 / T_c, n_c, ax.get_ylim()[1], color='#EDEDED', zorder=0, lw=0)
    ax.set_ylabel(r'$S$ (mol H m$^{-3}$ Pa$^{-1/2}$)')
    keylabels = [f"{fs.ROUTE_STYLE[n][0]}\n$\\Delta H_\\mathrm{{sol}}$ = {fs.energy(fits[n].dH, fits[n].dH_err)}, "
                 f"$S_0$ = {fs.sci(fits[n].S0)}" for n in order]
    keycols = [fs.ROUTE_STYLE[n][1] for n in order]
    if ceiling is not None:
        keylabels.append('Octahedral-site saturation')
        keycols.append('black')
    if lit_curves:
        handles.append(draw_literature(ax, lit_curves))
        keylabels.append(f'Literature ({len(lit_curves)})')
        keycols.append('black')
    fs.key_below(fig, handles, keylabels, ncol=1, colours=keycols)
    named = name_literature(fig, ax, lit_curves)
    keyed = True
    mat = fs.material_styles([stem], pretty_material)[stem][0]
    parts = []
    for name in order:
        fit = fits[name]
        txt = f'{ROUTE_WORDS[name]}: $S_0$ = {fs.tex_num(fit.S0, 3)}'
        if fit.S0_rel_err:
            txt += f' ($\\pm${100 * fit.S0_rel_err:.0f}\\%)'
        txt += f', $\\Delta H_\\mathrm{{sol}}$ = {fs.ev(fit.dH)}'
        if fit.dH_err is not None:
            txt += f' $\\pm$ {fit.dH_err:.3f}'
        txt += '~eV'
        if fit.r2 is not None:
            txt += f' ($R^2$ = {fit.r2:.4f})'
        parts.append(txt)
    sat_txt = ''
    if ceiling is not None:
        over = [f"{ROUTE_WORDS[n].split(' (')[0]} (up to {v:.3g} times)" for n, f in fits.items()
                if (v := breaches(f, ceiling))]
        sat_txt = (r' The dash-dot line is octahedral-site saturation, $4/a_0^3 N_A$ from the lattice constant at each '
                   rf'temperature ({fs.tex_num(n_c.max(), 3)} to {fs.tex_num(n_c.min(), 3)}~mol~m$^{{-3}}$); a solubility '
                   r'in the shaded region above it would put more than one H on every octahedral site at 1~Pa. '
                   + (f'The {fs.tex_join(over)} route exceeds it.' if over else 'No route exceeds it.'))
    dot_txt = (f" Dotted lines are the occupancy-limited solubility of the "
               f"{fs.tex_join(ROUTE_WORDS[n].split(' (')[0] for n in dotted)} route, drawn where the site occupancy "
               f"reaches {DILUTE_THETA_MAX}." if dotted else '')
    text = (rf'Hydrogen solubility of {mat} (Sieverts) against inverse temperature, {T_all.min():.0f} to '
            rf'{T_all.max():.0f}~K{_loading_words(run)}. Points are the per-temperature results; lines are Arrhenius '
            r'fits, $S = S_0\,e^{-\Delta H_\mathrm{sol}/k_BT}$, for each route: ' + '; '.join(parts) + '.' + sat_txt + dot_txt)
    drawn = [e for e, _T, _y in lit_curves]
    if drawn:
        words = '; '.join(f'{_tex(e.label)}: $S_0$ = {fs.tex_num(H_PER_H2 * e.pre, 3)}~mol~H~m$^{{-3}}$~Pa$^{{-1/2}}$, '
                          f'$\\Delta H_\\mathrm{{sol}}$ = {fs.ev(e.E_eV)}~eV, {e.T_K[0]:.0f}--{e.T_K[1]:.0f}~K' for e in drawn)
        text += (f' Grey dashed lines are {len(drawn)} published law{"s" if len(drawn) != 1 else ""}, converted from mol H$_2$ to mol H ($\\times 2$) and '
                 f'each drawn only over its own temperature range{" and named by its first author" if named else ""}: {words}.')
    text += literature_notes(drawn, left_out, LIT_DECADES) if (drawn or left_out) else ''
    fs.finish(fig, ax, fs.panel_stem(outfile, ''), MODULE, text, np.concatenate(xs),
              np.concatenate([fits[n].S for n in fits]), keyed=keyed)


# ---------------------------------------------------------------------------
# Section 5 — Cross-material figure
# ---------------------------------------------------------------------------

def plot_all_materials(chosen: dict[str, tuple[Run, dict[str, RouteFit]]],
                       ceilings: dict[str, tuple[np.ndarray, np.ndarray] | None],
                       routes: tuple[str, ...], outfile: str):
    """One figure per route, every material overlaid with its own saturation line (dash-dot, same
    colour); written as <outfile>_<route>.*."""
    from matplotlib.lines import Line2D
    present = [r for r in routes if any(r in fits for _run, fits in chosen.values())]
    if not present or len(chosen) < 2:
        return None
    styles = fs.material_styles(sorted(chosen), pretty_material)
    for route in present:
        fig, ax = fs.new_figure()
        ys, items, handles, words, temps, keylabels, keycols = [], [], [], [], [], [], []
        for stem, (label, colour, marker, ls) in styles.items():
            fits = chosen[stem][1]
            if route not in fits:
                continue
            fit = fits[route]
            T_fit = np.linspace(fit.T.min(), fit.T.max(), 120)
            ax.plot(1000.0 / T_fit, fit.curve(T_fit), color=colour, ls=ls, lw=fs.CURVE_PT, zorder=4)
            ax.plot(1000.0 / fit.T, fit.S, ls='none', marker=marker, ms=10, color=colour, zorder=5)
            ys += [fit.S, fit.curve(T_fit)]
            temps.append(fit.T)
            ceil = ceilings.get(stem)
            if ceil is not None:
                ax.plot(1000.0 / ceil[0], ceil[1], color=colour, lw=2.4, ls=(0, (6, 2, 1, 2)), zorder=3)
                ys.append(ceil[1])
            i = int(np.argmin(fit.T))
            items.append((1000.0 / fit.T[i], fit.S[i], label, colour))
            handles.append(Line2D([], [], color=colour, ls=ls, lw=fs.CURVE_PT, marker=marker, ms=9))
            words.append(f'{label} {fs.ev(fit.dH)}')
            keylabels.append(f"{label}\n$\\Delta H_\\mathrm{{sol}}$ = {fs.energy(fit.dH, fit.dH_err)}, $S_0$ = {fs.sci(fit.S0)}")
            keycols.append(colour)
        fs.inverse_temperature_axis(ax, np.concatenate(temps))
        fs.set_log_y(ax, np.concatenate([np.asarray(v, float) for v in ys]))
        ax.set_ylabel(r'$S$ (mol H m$^{-3}$ Pa$^{-1/2}$)')
        fs.key_below(fig, handles, keylabels, ncol=1, colours=keycols)
        keyed = True
        text = (rf'Hydrogen solubility of each material by the {ROUTE_WORDS[route]} route against inverse temperature; '
                r'points are the per-temperature results and lines the Arrhenius fits. Dash-dot lines in the same '
                r'colours are each material\'s octahedral-site saturation. $\Delta H_\mathrm{sol}$ (eV): '
                + '; '.join(words) + '.').replace("\\'", "'")
        fs.finish(fig, ax, fs.panel_stem(outfile, route), MODULE, text, keyed=keyed)
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
        description='Plot Sieverts solubility in Arrhenius form.')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                    help='directory holding the run folders '
                         '(default: calculation/results)')
    ap.add_argument('--pattern', default='*',
                    help="glob for run folders, e.g. 'Al*' (default: '*')")
    ap.add_argument('--include-1h', action='store_true',
                    help='keep single-H runs, which are excluded by default '
                         '(one walker, no ensemble average)')
    ap.add_argument('--outdir', default=None,
                    help='where figures go (default: <results-dir>/plots)')
    ap.add_argument('--routes', nargs='+', default=list(ROUTES),
                    choices=list(ROUTES),
                    help='which routes to plot (default: all three).')
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))
    routes      = tuple(args.routes)

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    runs = discover_runs(results_dir, args.pattern,
                         include_single_h=args.include_1h)
    if not runs:
        print(f"No H-loaded run directories matched '{args.pattern}'")
        return 1

    print(f'\nGrouping {len(runs)} run(s) by material …')
    chosen = group_by_material(runs, routes)
    if not chosen:
        print('No run has a solubility_arrhenius.json with a usable route.')
        return 1

    ceilings = {stem: site_density(results_dir, stem) for stem in chosen}
    missing  = [pretty_material(s) for s, c in ceilings.items() if c is None]
    if missing:
        print(f'  ! no lattice_params_vs_T.json for {", ".join(missing)} — '
              f'saturation ceiling unavailable there')

    envs     = {stem: load_environments(results_dir, stem) for stem in chosen}
    problems = report(chosen, ceilings, envs)

    print('\nLoading literature data …')
    lit_all, aliases = load_solubility_literature(), load_aliases(S_LIT_FILE)

    print('\nPer-material plots:')
    for stem, (_run, fits) in sorted(chosen.items()):
        material = pretty_material(stem).replace(' ', '_')
        plot_material(stem, fits, ceilings.get(stem),
                      os.path.join(outdir, f'{material}_solubility_arrhenius.png'),
                      env=envs.get(stem), run=_run, lit=literature_for(stem, lit_all, aliases))

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

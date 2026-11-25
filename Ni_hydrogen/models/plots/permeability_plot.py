#!/usr/bin/env python3
"""
models/plots/permeability_plot.py
================================
Richardson–Sieverts permeability Φ(T) in Arrhenius form — the pipeline's
headline deliverable and the quantity permeation experiments actually measure.

``Φ = D(T) · S(T)``, so ``Φ0 = D0·S0`` and ``E_Φ = E_D + ΔH_sol``. Three routes
are plotted (``geometric``, ``vibrational``, ``detailed_balance``), matching
``solubility_plot.py``.

Two guards, because Φ inherits everything wrong with its factors
----------------------------------------------------------------
1. **Negative E_Φ.** ``E_Φ = E_D + ΔH_sol`` reproduces to four decimals, so an
   exothermic ΔH_sol drags E_Φ below zero and Φ then *rises* as the metal cools.
   That is impossible for diffusion-limited permeation. Hastelloy N 7 currently
   sits at E_Φ = −0.567 eV.
2. **A saturated solubility.** Φ0 carries S0, so a solubility above octahedral
   saturation propagates straight into Φ0 — Ni's Φ0 = 1.9e22 is its Avogadro
   factor, not a permeability. The check from ``solubility_plot`` is reused.

Points come from ``permeability_T<T>K.json`` (per-temperature ``Phi``), lines
from the fitted ``Phi0``/``E_phi_eV`` in ``permeability_arrhenius.json``. The
two files disagree on route naming — ``option1``/``option2`` per temperature
versus ``geometric``/``vibrational`` in the fit — which is mapped below.

Each per-temperature file also carries a ``sieverts_regime`` field. It is
currently always ``None``: deciding whether Sieverts' law applies at all needs a
coverage isotherm, which no thermodynamic route can supply.

Figures
-------
  * ``<outdir>/<material>_permeability_arrhenius.png``       routes overlaid
  * ``<outdir>/permeability_arrhenius_all_materials.png``    one panel per route

Typical usage
-------------
::

    python models/plots/permeability_plot.py
    python models/plots/permeability_plot.py --pattern 'Al*'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from diffusivity_plot import (            # noqa: E402
    DEFAULT_RESULTS_DIR, Run, discover_runs, pretty_material,
)
import figstyle as fs                     # noqa: E402
from models.diffusivity_post_processing import KB_EV      # noqa: E402
from solubility_plot import (             # noqa: E402
    ROUTES, ROUTE_STYLE, breaches, load_solubility, site_density,
)


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

PHI_UNITS = 'mol H m⁻¹ s⁻¹ Pa⁻⁰·⁵'

# permeability_T<T>K.json names the energy-based routes option1/option2; the
# Arrhenius fit names them geometric/vibrational.
PER_T_KEY = {'geometric': 'option1', 'vibrational': 'option2',
             'detailed_balance': 'detailed_balance'}

ROUTE_LABEL = {
    'geometric':        'geometric',
    'vibrational':      'vibrational',
    'detailed_balance': 'detailed balance',
}

_T_FILE = re.compile(r'permeability_T(\d+(?:\.\d+)?)K\.json$')


# ---------------------------------------------------------------------------
# Section 2 — Loading
# ---------------------------------------------------------------------------

@dataclass
class PhiFit:
    """One route's Φ(T) points and its Arrhenius fit."""

    name:     str
    T:        np.ndarray          # K
    Phi:      np.ndarray          # mol H m^-1 s^-1 Pa^-0.5
    Phi0:     float
    E_phi:    float               # eV
    E_phi_err: float | None
    Phi0_factor: float | None     # ×/÷ 1-sigma band
    r2_S:     float | None

    def curve(self, T: np.ndarray) -> np.ndarray:
        return self.Phi0 * np.exp(-self.E_phi / (KB_EV * np.asarray(T, float)))

    @property
    def inverted(self) -> bool:
        """True when Φ rises as the metal cools — not physically possible."""
        return self.E_phi < 0.0


def load_per_temperature(run: Run) -> tuple[dict[str, dict[float, float]], dict]:
    """(route -> {T: Φ}, T -> sieverts_regime) from the per-temperature files."""
    phi: dict[str, dict[float, float]] = {r: {} for r in ROUTES}
    regimes: dict[float, dict] = {}

    for path in sorted(glob.glob(os.path.join(run.path, 'permeability_T*K.json'))):
        m = _T_FILE.search(os.path.basename(path))
        if not m:
            continue
        try:
            with open(path) as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            continue
        T = float(raw.get('T_K', m.group(1)))
        if raw.get('sieverts_regime'):
            regimes[T] = raw['sieverts_regime']
        for route in ROUTES:
            blk = raw.get(PER_T_KEY[route])
            if isinstance(blk, dict) and blk.get('Phi') not in (None, 0):
                phi[route][T] = float(blk['Phi'])
    return phi, regimes


def load_permeability(run: Run) -> tuple[dict[str, PhiFit], dict]:
    """Fitted Φ0/E_Φ joined to the per-temperature Φ points."""
    path = os.path.join(run.path, 'permeability_arrhenius.json')
    if not os.path.isfile(path):
        return {}, {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f'  ! {run.name}: unreadable permeability file ({exc})')
        return {}, {}

    per_T, regimes = load_per_temperature(run)
    out: dict[str, PhiFit] = {}
    for name in ROUTES:
        r = (raw.get('routes') or {}).get(name)
        if not r or not r.get('available') or r.get('Phi0') is None:
            continue
        pts = per_T.get(name, {})
        T   = np.asarray(sorted(pts), float)
        Phi = np.asarray([pts[t] for t in sorted(pts)], float)
        out[name] = PhiFit(
            name=name, T=T, Phi=Phi,
            Phi0=float(r['Phi0']), E_phi=float(r.get('E_phi_eV', 0.0)),
            E_phi_err=r.get('E_phi_err_eV'),
            Phi0_factor=r.get('Phi0_factor'), r2_S=r.get('r2_S'),
        )
    return out, regimes


# ---------------------------------------------------------------------------
# Section 3 — Grouping
# ---------------------------------------------------------------------------

def group_by_material(runs: list[Run]) -> dict[str, tuple[Run, dict[str, PhiFit], dict]]:
    """One representative run per material — the *highest* H loading.

    Unlike solubility, Φ does vary with loading, because ΔH_sol is fixed by the
    energetics while E_D is not. The variation is therefore entirely the
    diffusivity's, and the 1H runs are exactly the ones whose MSD never
    converged (a single H is a single random walker). Taking the lowest loading
    would propagate that: Hastelloy N 7 reads E_D = −0.0004 eV at 1H against
    +0.267 and +0.339 eV at 3H and 5H. The best-sampled loading is used instead
    and the full spread is printed.
    """
    by_stem: dict[str, list[tuple[Run, dict[str, PhiFit], dict]]] = {}
    for run in runs:
        fits, regimes = load_permeability(run)
        if fits:
            by_stem.setdefault(run.stem, []).append((run, fits, regimes))

    chosen: dict[str, tuple[Run, dict[str, PhiFit], dict]] = {}
    for stem, entries in by_stem.items():
        entries.sort(key=lambda e: e[0].n_H)
        if len(entries) > 1:
            spread = ',  '.join(
                f'{r.n_H}H: $E_Φ$={f["geometric"].E_phi:+.4f}'.replace('$', '')
                for r, f, _g in entries if 'geometric' in f)
            print(f'  · {pretty_material(stem)}: Φ varies with loading '
                  f'(E_D does, ΔH_sol does not) — {spread}')
            print(f'    using {entries[-1][0].n_H}H (best sampled; 1H MSD is '
                  f'not converged)')
        chosen[stem] = entries[-1]
    return chosen


# ---------------------------------------------------------------------------
# Section 4 — Per-material figure
# ---------------------------------------------------------------------------

ROUTE_WORDS = {'geometric': r'geometric ($S_0 = 4/a_0^3 N_A$)',
               'vibrational': r'vibrational ($S_0$ from partition functions)',
               'detailed_balance': r'detailed balance ($k_\mathrm{entry}/k_\mathrm{exit}$, a diagnostic)'}
MODULE = 'plots/permeability_plot.py'


def _route_line(ax, name, T, y, T_fit, y_fit):
    """Points and fit of one route in its house style; returns the key handle."""
    from matplotlib.lines import Line2D
    _label_, colour, marker, filled, ls = fs.ROUTE_STYLE[name]
    ax.plot(1000.0 / T_fit, y_fit, color=colour, ls=ls, lw=fs.CURVE_PT, zorder=4)
    if np.size(T):
        ax.plot(1000.0 / T, y, ls='none', marker=marker, ms=11, mfc=colour if filled else 'white',
                mec=colour, mew=2.2, zorder=5)
    return Line2D([], [], color=colour, ls=ls, lw=fs.CURVE_PT, marker=marker, ms=10,
                  mfc=colour if filled else 'white', mec=colour, mew=2.0)


def _loading_words(run) -> str:
    if run is None:
        return ''
    from diffusivity_plot import resolve_host_atoms
    if not run.n_host:
        resolve_host_atoms([run], None)
    pct = f', {run.at_pct:.2f} at.\\%' if run.at_pct else ''
    return f' from the run with {run.n_H} H in the supercell{pct}'


def plot_material(stem: str, fits: dict[str, PhiFit], outfile: str,
                  sol_flags: dict[str, float | None] | None = None, run=None):
    """Φ against 1000/T for every route of one material: one figure, written as PDF, PNG and caption."""
    fig, ax = fs.new_figure()
    ys, xs, items, handles = [], [], [], []
    T_all = np.concatenate([f.T for f in fits.values() if f.T.size]) if any(f.T.size for f in fits.values()) \
        else np.array([400.0, 800.0])
    for name in fs.ROUTE_STYLE:
        fit = fits.get(name)
        if fit is None:
            continue
        T = fit.T if fit.T.size else T_all
        T_fit = np.linspace(T.min(), T.max(), 120)
        handles.append(_route_line(ax, name, fit.T, fit.Phi, T_fit, fit.curve(T_fit)))
        ys += [fit.Phi, fit.curve(T_fit)]
        if fit.T.size:
            xs.append(1000.0 / fit.T)
            i = int(np.argmin(fit.T))
            items.append((1000.0 / fit.T[i], fit.Phi[i], fs.ROUTE_STYLE[name][0], fs.ROUTE_STYLE[name][1]))
        else:
            items.append((1000.0 / T_fit[0], fit.curve(T_fit[:1])[0], fs.ROUTE_STYLE[name][0], fs.ROUTE_STYLE[name][1]))
    fs.inverse_temperature_axis(ax, T_all)
    fs.set_log_y(ax, np.concatenate([np.asarray(v, float) for v in ys]))
    ax.set_ylabel(r'$\Phi$ (mol H m$^{-1}$ s$^{-1}$ Pa$^{-1/2}$)')
    names = [n for n in fs.ROUTE_STYLE if n in fits]
    fs.key_below(fig, handles, [f"{fs.ROUTE_STYLE[n][0]}\n$E_\\Phi$ = {fs.energy(fits[n].E_phi, fits[n].E_phi_err)}, "
                                f"$\\Phi_0$ = {fs.sci(fits[n].Phi0)}" for n in names],
                 ncol=1, colours=[fs.ROUTE_STYLE[n][1] for n in names])
    keyed = True
    mat = fs.material_styles([stem], pretty_material)[stem][0]
    parts = []
    for name in fs.ROUTE_STYLE:
        fit = fits.get(name)
        if fit is None:
            continue
        txt = f'{ROUTE_WORDS[name]}: $\\Phi_0$ = {fs.tex_num(fit.Phi0, 3)}'
        if fit.Phi0_factor:
            txt += f' ($\\times/\\div$ {fit.Phi0_factor:.2f})'
        txt += f', $E_\\Phi$ = {fs.ev(fit.E_phi)}'
        if fit.E_phi_err is not None:
            txt += f' $\\pm$ {fit.E_phi_err:.3f}'
        txt += '~eV' + (' (negative: $\\Phi$ rises as the metal cools, not possible for diffusion-limited '
                        'permeation)' if fit.inverted else '')
        parts.append(txt)
    over = [f"{ROUTE_WORDS[n].split(' (')[0]} (up to {v:.3g} times)" for n, v in (sol_flags or {}).items() if v]
    warn = (f' The solubility of the {fs.tex_join(over)} route exceeds octahedral-site saturation, so its '
            r'$\Phi_0$ is not a usable permeability.' if over else '')
    name_ = os.path.basename(fs.panel_stem(outfile, ''))
    text = (rf'Hydrogen permeability of {mat}, $\Phi = D\,S$ (Richardson--Sieverts), against inverse temperature, '
            rf'{T_all.min():.0f} to {T_all.max():.0f}~K{_loading_words(run)}. Points are the per-temperature results; '
            r'lines are Arrhenius fits, $\Phi = \Phi_0\,e^{-E_\Phi/k_BT}$, for each solubility route: '
            + '; '.join(parts) + '.' + warn)
    fs.finish(fig, ax, fs.panel_stem(outfile, ''), MODULE, text,
              np.concatenate(xs) if xs else None,
              np.concatenate([fits[n].Phi for n in fits if fits[n].T.size]) if xs else None, keyed=keyed)


# ---------------------------------------------------------------------------
# Section 5 — Cross-material figure
# ---------------------------------------------------------------------------

def plot_all_materials(chosen: dict, outfile: str):
    """One figure per route, every material overlaid (written as <outfile>_<route>.*)."""
    from matplotlib.lines import Line2D
    present = [r for r in ROUTES if any(r in f for _run, f, _g in chosen.values())]
    if not present or len(chosen) < 2:
        return None
    styles = fs.material_styles(sorted(chosen), pretty_material)
    for route in present:
        fig, ax = fs.new_figure()
        ys, xs, items, handles, words, temps, keylabels, keycols = [], [], [], [], [], [], [], []
        for stem, (label, colour, marker, ls) in styles.items():
            fit = chosen[stem][1].get(route)
            if fit is None:
                continue
            T = fit.T if fit.T.size else np.array([400.0, 800.0])
            T_fit = np.linspace(T.min(), T.max(), 120)
            ax.plot(1000.0 / T_fit, fit.curve(T_fit), color=colour, ls=ls, lw=fs.CURVE_PT, zorder=4)
            if fit.T.size:
                ax.plot(1000.0 / fit.T, fit.Phi, ls='none', marker=marker, ms=10, color=colour, zorder=5)
                xs.append(1000.0 / fit.T)
            ys += [fit.curve(T_fit)] + ([fit.Phi] if fit.T.size else [])
            temps.append(T)
            i = int(np.argmin(T_fit))
            items.append((1000.0 / T_fit[i], fit.curve(T_fit[i:i + 1])[0], label, colour))
            handles.append(Line2D([], [], color=colour, ls=ls, lw=fs.CURVE_PT, marker=marker, ms=9))
            words.append(f'{label} {fs.ev(fit.E_phi)}' + (' (negative)' if fit.inverted else ''))
            keylabels.append(f"{label}\n$E_\\Phi$ = {fs.energy(fit.E_phi, fit.E_phi_err)}, $\\Phi_0$ = {fs.sci(fit.Phi0)}")
            keycols.append(colour)
        fs.inverse_temperature_axis(ax, np.concatenate(temps))
        fs.set_log_y(ax, np.concatenate([np.asarray(v, float) for v in ys]))
        ax.set_ylabel(r'$\Phi$ (mol H m$^{-1}$ s$^{-1}$ Pa$^{-1/2}$)')
        fs.key_below(fig, handles, keylabels, ncol=1, colours=keycols)
        keyed = True
        stem_out = fs.panel_stem(outfile, route)
        text = (rf'Hydrogen permeability of each material by the {ROUTE_WORDS[route]} route against inverse temperature; '
                "points are the per-temperature results of each material's best-sampled H loading and lines its "
                rf'Arrhenius fit. $E_\Phi$ (eV): {"; ".join(words)}.')
        fs.finish(fig, ax, stem_out, MODULE, text, keyed=keyed)
    return outfile


# ---------------------------------------------------------------------------
# Section 6 — Console report
# ---------------------------------------------------------------------------

def report(chosen: dict, sol_flags: dict[str, dict[str, float | None]]) -> list[str]:
    """Print Φ0/E_Φ per route with the inversion and saturation verdicts."""
    print(f'\n  Φ in {PHI_UNITS}')
    hdr = (f"  {'material':<18s}{'route':<18s}{'Phi0':>12s}{'E_phi (eV)':>12s}"
           f"{'r2_S':>9s}  verdict")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    problems: list[str] = []
    for stem in sorted(chosen):
        _run, fits, _reg = chosen[stem]
        for name in ROUTE_STYLE:
            fit = fits.get(name)
            if fit is None:
                print(f'  {pretty_material(stem):<18s}{name:<18s}'
                      f'{"-":>12s}{"-":>12s}{"-":>9s}  absent')
                continue
            notes = []
            if fit.inverted:
                notes.append('E_Φ<0: Φ rises on cooling')
            if sol_flags.get(stem, {}).get(name):
                notes.append('S over saturation → Φ0 unusable')
            verdict = '; '.join(notes) or 'ok'
            if notes:
                problems.append(f'{pretty_material(stem)}/{name}: {verdict}')
            r2 = f'{fit.r2_S:.5f}' if fit.r2_S is not None else '-'
            print(f'  {pretty_material(stem):<18s}{name:<18s}{fit.Phi0:>12.3e}'
                  f'{fit.E_phi:>+12.4f}{r2:>9s}  {verdict}')

    for stem in sorted(chosen):
        regimes = chosen[stem][2]
        if not regimes:
            continue
        bits = ', '.join(f'{T:.0f}K {r.get("regime")}'
                         f' (θ_max={r.get("theta_max", float("nan")):.2f})'
                         for T, r in sorted(regimes.items()))
        print(f'  · {pretty_material(stem)} Sieverts regime: {bits}')
    return problems


# ---------------------------------------------------------------------------
# Section 7 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='Plot Richardson–Sieverts permeability in Arrhenius form.')
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
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    runs = discover_runs(results_dir, args.pattern,
                         include_single_h=args.include_1h)
    if not runs:
        print(f"No H-loaded run directories matched '{args.pattern}'")
        return 1

    print(f'\nGrouping {len(runs)} run(s) by material …')
    chosen = group_by_material(runs)
    if not chosen:
        print('No run has a permeability_arrhenius.json with a usable route.')
        return 1

    # Φ0 carries S0, so a saturated solubility invalidates Φ0 too.
    sol_flags: dict[str, dict[str, float | None]] = {}
    for stem, (run, _fits, _reg) in chosen.items():
        ceil = site_density(results_dir, stem)
        sol  = load_solubility(run, ROUTES)
        sol_flags[stem] = {n: breaches(f, ceil) for n, f in sol.items()}

    problems = report(chosen, sol_flags)

    print('\nPer-material plots:')
    for stem, (run, fits, _reg) in sorted(chosen.items()):
        material = pretty_material(stem).replace(' ', '_')
        plot_material(stem, fits,
                      os.path.join(outdir,
                                   f'{material}_permeability_arrhenius.png'),
                      sol_flags=sol_flags.get(stem), run=run)

    print('\nCross-material plot:')
    if plot_all_materials(chosen,
                          os.path.join(outdir,
                                       'permeability_arrhenius_all_materials.png')
                          ) is None:
        print('  · needs at least 2 materials — skipped')

    if problems:
        print(f'\n{len(problems)} route(s) are not usable as a permeability:')
        for p in problems:
            print(f'  · {p}')
        print('Φ = D·S, so Φ0 = D0·S0 and E_Φ = E_D + ΔH_sol. Neither factor is\n'
              'independent: fix the solubility (rerun Phase 6 for pre-2026-07-28\n'
              'files; address site saturation) and Φ follows.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

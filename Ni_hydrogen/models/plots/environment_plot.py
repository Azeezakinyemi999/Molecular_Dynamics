#!/usr/bin/env python3
"""
models/plots/environment_plot.py
===============================
Distributions over interstitial environments and sites — the spread that every
population-averaged number downstream is hiding.

Three figures per material:

1. ``<mat>_env_dH_sol.png``      ΔH_sol per environment, and each environment's
                                 share of the total solubility.
2. ``<mat>_site_barriers.png``   hopA / hopB barriers per site, raw and
                                 ZPE-corrected, sorted by height.
3. ``<mat>_site_rates.png``      forward / reverse TST rates per site, at every
                                 temperature available.

Why figure 1 matters
--------------------
``solubility_by_environment`` sums ``w_env·exp(−ΔH_sol/kT)``, which is
exponential in ΔH, so the most exothermic environment dominates however few
sites it represents. For Hastelloy N 7, three tetrahedral environments holding
~10 % of the population contribute essentially 100 % of S; for Ni a single
``Ni6_oct`` (w = 0.20) supplies 99.2 %. The right-hand panel makes that
explicit rather than leaving it in a table.

Data sources
------------
``results/<material>/dH_sol_by_env.json``   per-environment ΔH_sol + weight
``results/<material>/rate_dict_T<T>K.json`` per-site barriers, rates, ν, ΔE

Note these live in the **0-H material directory**, not the per-loading run
directories — they are properties of the lattice, not of an MD run.

Typical usage
-------------
::

    python models/plots/environment_plot.py
    python models/plots/environment_plot.py --pattern 'Hastelloy*'
    python models/plots/environment_plot.py --temperature 600
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
    DEFAULT_RESULTS_DIR, pretty_material, _save,
)
from models.diffusivity_post_processing import KB_EV      # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

# An interstitial hop in a metal is well under this; anything above it points at
# an unconverged NEB rather than a real barrier.
BARRIER_SANITY_EV = 3.0

STAGE_COLOUR = {'hopa': 'tab:blue', 'hopb': 'tab:orange'}
STAGE_LABEL  = {'hopa': 'hopA  (surface → sub1)', 'hopb': 'hopB  (sub1 → sub2)'}

_T_FILE  = re.compile(r'rate_dict_T(\d+(?:\.\d+)?)K\.json$')
_SITE_RE = re.compile(r'^(hopa|hopb)_(\S+)$')


# ---------------------------------------------------------------------------
# Section 2 — Loading
# ---------------------------------------------------------------------------

@dataclass
class SiteRate:
    """One site's barrier and rate data at one temperature."""

    stage:   str
    site:    str
    T:       float
    Ea_raw:  float | None
    Ea_zpe:  float | None
    k_fwd:   float | None
    k_rev:   float | None
    nu:      float | None
    delta_e: float | None


def discover_materials(results_dir: str, pattern: str) -> list[str]:
    """Material directories (0-H) holding per-environment or per-site data.

    Skips names containing a space — ``Ni_supercell copy`` is a stray duplicate
    that would otherwise be plotted as a distinct material.
    """
    found: list[str] = []
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        if not os.path.isdir(path):
            continue
        stem = os.path.basename(path)
        if ' ' in stem:
            print(f'  · {stem}: skipped, name contains a space '
                  f'(looks like a stray copy)')
            continue
        has_env  = os.path.isfile(os.path.join(path, 'dH_sol_by_env.json'))
        has_rate = bool(glob.glob(os.path.join(path, 'rate_dict_T*K.json')))
        if has_env or has_rate:
            found.append(stem)
    return found


def load_env(results_dir: str, stem: str) -> dict | None:
    path = os.path.join(results_dir, stem, 'dH_sol_by_env.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh) or None
    except (OSError, ValueError):
        return None


def load_rates(results_dir: str, stem: str) -> list[SiteRate]:
    """Every per-site rate entry across all available temperatures."""
    out: list[SiteRate] = []
    for path in sorted(glob.glob(os.path.join(results_dir, stem,
                                              'rate_dict_T*K.json'))):
        m = _T_FILE.search(os.path.basename(path))
        if not m:
            continue
        try:
            with open(path) as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            continue
        for key, v in raw.items():
            sm = _SITE_RE.match(key)
            if not sm or not isinstance(v, dict):
                continue
            out.append(SiteRate(
                stage=sm.group(1), site=sm.group(2),
                T=float(v.get('T_K', m.group(1))),
                Ea_raw=v.get('Ea_raw'), Ea_zpe=v.get('Ea_zpe'),
                k_fwd=v.get('k_forward'), k_rev=v.get('k_reverse'),
                nu=v.get('nu'), delta_e=v.get('delta_e'),
            ))
    return out


# ---------------------------------------------------------------------------
# Section 3 — Figure 1: ΔH_sol per environment + share of S
# ---------------------------------------------------------------------------

MODULE = 'plots/environment_plot.py'
STAGE_WORDS = {'hopa': 'hop A (surface to the first subsurface site)',
               'hopb': 'hop B (first to second subsurface site)'}


def _env_name(key: str) -> str:
    """'Ni4_tet' -> 'Ni$_4$ tetrahedral'."""
    comp, _, kind = key.rpartition('_')
    kind = {'tet': 'tetrahedral', 'oct': 'octahedral'}.get(kind, kind)
    if not comp:
        return key
    comp = re.sub(r'(\d+)', r'$_{\1}$', comp)
    return f'{comp} {kind}'


def plot_env_dH(stem: str, env: dict, T_K: float, outfile: str):
    """Two figures: the solution enthalpy of each environment (<outfile>.*) and each environment's
    share of the solubility at T_K (<outfile>_share.*), each as PDF, PNG and caption."""
    import figstyle as fs
    names = sorted(env, key=lambda k: env[k]['dH_sol_eV'])
    dH = np.array([env[k]['dH_sol_eV'] for k in names])
    dH_err = np.array([env[k].get('dH_sol_err_eV') or 0.0 for k in names])
    w = np.array([env[k]['w_env'] for k in names])
    boltz = w * np.exp(-dH / (KB_EV * T_K))
    share = boltz / boltz.sum() * 100.0 if boltz.sum() > 0 else np.zeros_like(w)
    mat = fs.material_styles([stem], pretty_material)[stem][0]
    rows = '; '.join(f"{_env_name(k)}: {env[k].get('n_sites', '?')} sites, weight $w$ = {env[k]['w_env']:.3f}, "
                     f"$\\Delta H_\\mathrm{{sol}}$ = {fs.ev(env[k]['dH_sol_eV'])}"
                     + (f" $\\pm$ {env[k]['dH_sol_err_eV']:.3f}" if env[k].get('dH_sol_err_eV') else '')
                     + f'~eV, {share[i]:.1f}\\% of $S$' for i, k in enumerate(names))
    y = np.arange(len(names))
    height = max(128.0, 40.0 + 22.0 * len(names))          # mm: room for many environments
    for which in ('dH', 'share'):
        fig, ax = fs.new_figure()
        fig.set_figheight(height * fs.MM)
        if which == 'dH':
            colours = [fs.OKABE['vermilion'] if v < 0 else fs.OKABE['blue'] for v in dH]
            ax.barh(y, dH, height=0.6, color=colours, edgecolor='black', linewidth=1.5, zorder=3)
            ax.errorbar(dH, y, xerr=dH_err, ls='none', color='black', capsize=6, elinewidth=2.2, capthick=2.2, zorder=4)
            for yi, v, e in zip(y, dH, dH_err):
                fs.text_label(ax, v + (e if v >= 0 else -e), yi, f'  {v:.3f} eV  ', ha='left' if v >= 0 else 'right',
                              va='center', backing=False)
            lo, hi = min(0.0, (dH - dH_err).min()), max(0.0, (dH + dH_err).max())
            rng = (hi - lo) or 1.0
            ax.set_xlim(lo - 0.08 * rng - (0.35 * rng if lo < 0 else 0), hi + 0.45 * rng)
            ax.axvline(0.0, color='black', lw=2.2, zorder=2)
            ax.set_xlabel(r'$\Delta H_\mathrm{sol}$ (eV)')
            text = (rf'Solution enthalpy of H in each interstitial environment of {mat}, from the subsurface NEB end states; '
                    r'error bars are the standard errors over the sites of each environment. Blue bars are endothermic, '
                    rf'vermillion exothermic. {rows}.')
            stem_out = fs.panel_stem(outfile, '')
        else:
            ax.barh(y, share, height=0.6, color=fs.OKABE['green'], edgecolor='black', linewidth=1.5, zorder=3)
            for yi, sv, wi in zip(y, share, w):
                fs.text_label(ax, sv, yi, f'  {sv:.1f}% (w = {wi:.3f})', ha='left', va='center', backing=False)
            ax.set_xlim(0, 160)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_xlabel(f'Share of the solubility at {T_K:.0f} K (%)')
            text = (rf'Share of the H solubility of {mat} carried by each interstitial environment at {T_K:.0f}~K, '
                    r'$w\,e^{-\Delta H_\mathrm{sol}/k_BT}$ normalised over the environments, where $w$ is the fraction of '
                    rf'interstitial sites in the environment. {rows}.')
            stem_out = fs.panel_stem(outfile, 'share')
        ax.set_yticks(y, [_env_name(k) for k in names])
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.tick_params(axis='y', which='both', left=False, right=False)
        fs.finish(fig, ax, stem_out, MODULE, text)


def _by_site(rates: list[SiteRate], stage: str) -> list[SiteRate]:
    out: dict[str, SiteRate] = {}
    for r in rates:
        if r.stage == stage and r.Ea_zpe is not None:
            out.setdefault(r.site, r)                   # barriers do not depend on T: one entry per site
    return sorted(out.values(), key=lambda r: r.Ea_zpe)


def plot_site_barriers(stem: str, rates: list[SiteRate], outfile: str):
    """Hop A and hop B barrier per site, with and without ZPE, sorted: one figure per hop
    (<outfile>_hopa.*, <outfile>_hopb.*), each as PDF, PNG and caption."""
    import figstyle as fs
    from matplotlib.lines import Line2D
    stages = [s for s in ('hopa', 'hopb') if any(r.stage == s and r.Ea_zpe is not None for r in rates)]
    if not stages:
        return None
    mat = fs.material_styles([stem], pretty_material)[stem][0]
    for stage in stages:
        items = _by_site(rates, stage)
        x = np.arange(len(items))
        zpe = np.array([r.Ea_zpe for r in items])
        raw = np.array([r.Ea_raw if r.Ea_raw is not None else np.nan for r in items])
        fig, ax = fs.new_figure()
        ax.axhline(zpe.mean(), zorder=1, **fs.REF_LINE)
        if np.isfinite(raw).any():
            ax.plot(x, raw, ls='none', marker='s', ms=11, mfc='white', mec='black', mew=2.0, zorder=4)
        ax.plot(x, zpe, ls='none', marker='o', ms=12, color='black', zorder=5)
        ax.set_xlim(-0.6, len(items) - 0.4)
        if len(items) <= 20:
            ax.set_xticks(x, [r.site for r in items], rotation=45, ha='right', rotation_mode='anchor')
        else:
            ax.set_xlabel('Site, sorted by barrier')
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        fs.set_linear_y(ax, np.r_[zpe, raw[np.isfinite(raw)]], frac=0.12)
        ax.set_ylabel(r'$E_a$ (eV)')
        handles = [Line2D([], [], ls='none', marker='o', ms=11, color='black'),
                   Line2D([], [], ls='none', marker='s', ms=10, mfc='white', mec='black', mew=2.0),
                   Line2D([], [], **fs.REF_LINE)]
        fs.key_below(fig, handles, ['With ZPE', 'Without ZPE', f'Mean, {zpe.mean():.3f} eV'], ncol=3)
        d = raw - zpe
        esc = lambda t: t.replace('_', r'\_')                                   # noqa: E731
        text = (rf'Barrier of {STAGE_WORDS[stage]} in {mat} at each of the {len(items)} sites, sorted, with the zero-point '
                rf'energy correction (filled circles) and without it (open squares). With ZPE the barriers run from '
                rf'{zpe.min():.3f} ({esc(items[0].site)}) to {zpe.max():.3f}~eV ({esc(items[-1].site)}), a spread of '
                rf'{zpe.max() - zpe.min():.3f}~eV; the dashed line is their mean, {zpe.mean():.3f}~eV.'
                + (rf' The ZPE correction changes them by {np.nanmin(d):+.3f} to {np.nanmax(d):+.3f}~eV.'.replace('-', '$-$')
                   if np.isfinite(d).any() else ''))
        fs.finish(fig, ax, fs.panel_stem(outfile, stage), MODULE, text, x, zpe, keyed=True)
    return outfile


def plot_site_rates(stem: str, rates: list[SiteRate], outfile: str):
    """Forward and reverse TST rate per site, one line per temperature: one figure per hop
    (<outfile>_hopa.*, <outfile>_hopb.*), each as PDF, PNG and caption."""
    import figstyle as fs
    from matplotlib.lines import Line2D
    stages = [s for s in ('hopa', 'hopb') if any(r.stage == s for r in rates)]
    if not stages:
        return None
    mat = fs.material_styles([stem], pretty_material)[stem][0]
    for stage in stages:
        order = [r.site for r in _by_site(rates, stage)] or sorted({r.site for r in rates if r.stage == stage})
        idx = {s: i for i, s in enumerate(order)}
        temps = sorted({r.T for r in rates if r.stage == stage})
        colours = fs.temp_styles(temps)
        fig, ax = fs.new_figure()
        ys, xs = [], []
        for T in temps:
            c = colours[T]
            pts = sorted((idx[r.site], r.k_fwd, r.k_rev) for r in rates if r.stage == stage and r.T == T and r.site in idx)
            xf = [q[0] for q in pts if q[1] and q[1] > 0]
            fw = [q[1] for q in pts if q[1] and q[1] > 0]
            xr = [q[0] for q in pts if q[2] and q[2] > 0]
            rv = [q[2] for q in pts if q[2] and q[2] > 0]
            if xf:
                ax.plot(xf, fw, color=c, lw=fs.CURVE_PT, marker='o', ms=10, zorder=4)
            if xr:
                ax.plot(xr, rv, color=c, lw=fs.CURVE_PT, ls=(0, (4, 2)), marker='s', ms=9, mfc='white', mec=c, mew=2.0,
                        zorder=3)
            ys += [fw, rv]
            xs += [xf, xr]
        ax.set_xlim(-0.6, len(order) - 0.4)
        if len(order) <= 20:
            ax.set_xticks(range(len(order)), order, rotation=45, ha='right', rotation_mode='anchor')
        else:
            ax.set_xlabel('Site, sorted by barrier')
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ally = np.concatenate([np.asarray(v, float) for v in ys if len(v)])
        fs.set_log_y(ax, ally)
        ax.set_ylabel(r'$k$ (s$^{-1}$)')
        handles = [Line2D([], [], color=colours[T], lw=fs.CURVE_PT) for T in temps]
        labels = [f'{T:.0f} K' for T in temps]
        handles += [Line2D([], [], color='black', lw=fs.CURVE_PT, marker='o', ms=9),
                    Line2D([], [], color='black', lw=fs.CURVE_PT, ls=(0, (4, 2)), marker='s', ms=8, mfc='white', mew=2.0)]
        labels += ['Forward', 'Reverse']
        fs.key_below(fig, handles, labels, ncol=4)

        def span(T, attr):
            v = [getattr(r, attr) for r in rates if r.stage == stage and r.T == T and getattr(r, attr)
                 and getattr(r, attr) > 0]
            return f'{fs.tex_num(min(v))} to {fs.tex_num(max(v))}~s$^{{-1}}$' if v else 'not available'
        text = (rf'Transition-state-theory rate of {STAGE_WORDS[stage]} in {mat} at each site, sites sorted by barrier, at '
                rf"{fs.tex_join([f'{T:.0f}' for T in temps])}~K: forward (solid, filled circles) and reverse (dashed, "
                rf"open squares). Forward rates run {span(temps[0], 'k_fwd')} at {temps[0]:.0f}~K and "
                rf"{span(temps[-1], 'k_fwd')} at {temps[-1]:.0f}~K; reverse rates {span(temps[0], 'k_rev')} and "
                rf"{span(temps[-1], 'k_rev')}.")
        fs.finish(fig, ax, fs.panel_stem(outfile, stage), MODULE, text,
                  np.concatenate([np.asarray(v, float) for v in xs if len(v)]), ally, keyed=True)
    return outfile


# ---------------------------------------------------------------------------
# Section 6 — Console report
# ---------------------------------------------------------------------------

def report(stem: str, env: dict | None, rates: list[SiteRate], T_K: float):
    """Summarise the spread that population averaging hides."""
    if env:
        names = sorted(env, key=lambda k: env[k]['dH_sol_eV'])
        dH = np.array([env[k]['dH_sol_eV'] for k in names])
        w  = np.array([env[k]['w_env'] for k in names])
        b  = w * np.exp(-dH / (KB_EV * T_K))
        share = b / b.sum() * 100.0 if b.sum() > 0 else np.zeros_like(w)
        print(f'  environments: {len(names)}   ΔH_sol {dH.min():+.4f} … '
              f'{dH.max():+.4f} eV  (spread {dH.ptp():.3f} eV = '
              f'{dH.ptp() / (KB_EV * T_K):.0f} kT at {T_K:.0f} K)')
        order = np.argsort(share)[::-1]
        cum = 0.0
        for i in order[:4]:
            cum += share[i]
            print(f'    {names[i]:<16s} ΔH={dH[i]:+.4f}  w={w[i]:.3f}  '
                  f'→ {share[i]:5.1f}% of S   (cumulative {cum:5.1f}%)')
        n90 = int(np.searchsorted(np.cumsum(share[order]), 90.0) + 1)
        print(f'    {n90} of {len(names)} environment(s) carry 90% of S, '
              f'holding w = {w[order][:n90].sum():.3f} of the sites')

    suspect: list[str] = []
    for stage in ('hopa', 'hopb'):
        uniq = {r.site: r for r in rates
                if r.stage == stage and r.Ea_zpe is not None}
        if not uniq:
            continue
        a = np.array([r.Ea_zpe for r in uniq.values()])
        print(f'  {stage}: {len(a)} sites   E_a {a.min():.3f} … {a.max():.3f} eV '
              f'(mean {a.mean():.3f}, spread {a.ptp():.3f})')

        for site, r in uniq.items():
            if r.Ea_zpe < 0.0:
                # A negative barrier gives a rate above the attempt frequency.
                # Usually ZPE applied to an all-but-barrierless hop rather than
                # a broken NEB — the raw barrier tells which.
                kind = ('ZPE artifact on a near-zero barrier'
                        if (r.Ea_raw or 0.0) >= 0.0 else 'raw barrier negative')
                suspect.append(f'{stage}_{site}: E_a={r.Ea_zpe:+.4f} eV '
                               f'(raw {r.Ea_raw:+.4f}) — {kind}')
            elif r.Ea_zpe > BARRIER_SANITY_EV:
                suspect.append(f'{stage}_{site}: E_a={r.Ea_zpe:.3f} eV '
                               f'(raw {r.Ea_raw:.3f}, ΔE={r.delta_e:+.3f}) — '
                               f'far above a plausible interstitial hop; '
                               f'likely an unconverged NEB')
    if suspect:
        print(f'  ! {len(suspect)} site(s) with an unusable barrier:')
        for s in suspect:
            print(f'      {s}')
        print('    These enter the TST rate table and the per-environment '
              'averages as-is.')


# ---------------------------------------------------------------------------
# Section 7 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='Plot per-environment and per-site distributions.')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                    help='directory holding the material folders '
                         '(default: calculation/results)')
    ap.add_argument('--pattern', default='*_supercell',
                    help="glob for material folders (default: '*_supercell')")
    ap.add_argument('--outdir', default=None,
                    help='where figures go (default: <results-dir>/plots)')
    ap.add_argument('--temperature', type=float, default=400.0,
                    help='temperature for the solubility-share panel, in K '
                         '(default: 400)')
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    materials = discover_materials(results_dir, args.pattern)
    if not materials:
        print(f"No material folder matching '{args.pattern}' has "
              f'dH_sol_by_env.json or rate_dict_T*K.json')
        return 1

    for stem in materials:
        env   = load_env(results_dir, stem)
        rates = load_rates(results_dir, stem)
        print(f'\n{pretty_material(stem)}  ({stem})')
        report(stem, env, rates, args.temperature)

        material = pretty_material(stem).replace(' ', '_')
        if env:
            plot_env_dH(stem, env, args.temperature,
                        os.path.join(outdir, f'{material}_env_dH_sol.png'))
        else:
            print('  · no dH_sol_by_env.json — enthalpy figure skipped')

        if rates:
            plot_site_barriers(stem, rates,
                               os.path.join(outdir,
                                            f'{material}_site_barriers.png'))
            plot_site_rates(stem, rates,
                            os.path.join(outdir, f'{material}_site_rates.png'))
        else:
            print('  · no rate_dict_T*K.json — barrier/rate figures skipped')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

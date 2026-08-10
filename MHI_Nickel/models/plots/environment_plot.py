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

def plot_env_dH(stem: str, env: dict, T_K: float, outfile: str):
    """ΔH_sol per environment beside each environment's share of S."""
    import matplotlib.pyplot as plt

    names = sorted(env, key=lambda k: env[k]['dH_sol_eV'])
    dH    = np.array([env[k]['dH_sol_eV'] for k in names])
    w     = np.array([env[k]['w_env'] for k in names])

    boltz = w * np.exp(-dH / (KB_EV * T_K))
    share = boltz / boltz.sum() * 100.0 if boltz.sum() > 0 else np.zeros_like(w)

    h = max(3.2, 0.26 * len(names) + 1.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, h), sharey=True)
    y = np.arange(len(names))

    ax1.barh(y, dH, color=np.where(dH < 0, 'tab:red', 'tab:blue'), alpha=0.85)
    ax1.axvline(0.0, color='k', lw=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=7.5)
    ax1.set_xlabel('$\\Delta H_{sol}$  (eV)')
    ax1.set_title('per-environment solution enthalpy\n'
                  '(red = exothermic, drives the Boltzmann sum)', fontsize=9.5)
    ax1.grid(True, axis='x', alpha=0.3)

    ax2.barh(y, share, color='tab:green', alpha=0.85)
    ax2.set_xlabel(f'share of total S at {T_K:.0f} K  (%)')
    ax2.set_title('contribution to solubility\n'
                  '$w_{env}\\,e^{-\\Delta H/kT}$, normalised', fontsize=9.5)
    ax2.grid(True, axis='x', alpha=0.3)

    # Population weight alongside the share makes the mismatch legible.
    for yi, (wi, si) in enumerate(zip(w, share)):
        if si >= 1.0:
            ax2.text(si, yi, f'  w={wi:.3f}', va='center', fontsize=7,
                     color='0.3')

    top = np.argsort(share)[::-1][:3]
    note = ',  '.join(f'{names[i]}: {share[i]:.1f}% of S from w={w[i]:.3f}'
                      for i in top if share[i] > 0)
    fig.suptitle(f'H interstitial environments — {pretty_material(stem)}  '
                 f'({len(names)} environments)', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.text(0.5, 0.012, note, ha='center', fontsize=8, color='0.3')
    _save(fig, outfile)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 4 — Figure 2: per-site barriers
# ---------------------------------------------------------------------------

def plot_site_barriers(stem: str, rates: list[SiteRate], outfile: str):
    """hopA / hopB barriers per site, raw vs ZPE-corrected, sorted."""
    import matplotlib.pyplot as plt

    stages = [s for s in ('hopa', 'hopb')
              if any(r.stage == s and r.Ea_zpe is not None for r in rates)]
    if not stages:
        return None

    fig, axes = plt.subplots(1, len(stages), figsize=(5.6 * len(stages), 4.6),
                             squeeze=False)
    axes = axes[0]

    for ax, stage in zip(axes, stages):
        # Barriers are T-independent; take one entry per site.
        by_site: dict[str, SiteRate] = {}
        for r in rates:
            if r.stage == stage and r.Ea_zpe is not None:
                by_site.setdefault(r.site, r)
        items = sorted(by_site.values(), key=lambda r: r.Ea_zpe)
        x     = np.arange(len(items))
        zpe   = np.array([r.Ea_zpe for r in items])
        raw   = np.array([r.Ea_raw if r.Ea_raw is not None else np.nan
                          for r in items])

        ax.plot(x, zpe, 'o-', color=STAGE_COLOUR[stage], ms=4, lw=1.2,
                label='$E_a$ with ZPE')
        if np.isfinite(raw).any():
            ax.plot(x, raw, 's', color='0.45', ms=3.5, alpha=0.8,
                    label='$E_a$ raw')
        ax.axhline(zpe.mean(), color=STAGE_COLOUR[stage], ls='--', lw=1.0,
                   alpha=0.7, label=f'mean {zpe.mean():.3f} eV')

        ax.set_title(f'{STAGE_LABEL[stage]}\n{len(items)} sites, '
                     f'spread {zpe.max() - zpe.min():.3f} eV', fontsize=9.5)
        ax.set_xlabel('site, sorted by barrier')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc='upper left')
        if len(items) <= 14:
            ax.set_xticks(x)
            ax.set_xticklabels([r.site for r in items], rotation=60,
                               fontsize=7, ha='right')

    axes[0].set_ylabel('barrier  (eV)')
    fig.suptitle(f'Per-site hop barriers — {pretty_material(stem)}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, outfile)
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Section 5 — Figure 3: per-site rates
# ---------------------------------------------------------------------------

def plot_site_rates(stem: str, rates: list[SiteRate], outfile: str):
    """Forward and reverse TST rates per site, one series per temperature."""
    import matplotlib.pyplot as plt

    stages = [s for s in ('hopa', 'hopb') if any(r.stage == s for r in rates)]
    if not stages:
        return None
    temps = sorted({r.T for r in rates})

    fig, axes = plt.subplots(1, len(stages), figsize=(5.9 * len(stages), 4.7),
                             squeeze=False, sharey=True)
    axes = axes[0]
    colours = plt.cm.plasma(np.linspace(0.05, 0.72, len(temps)))

    for ax, stage in zip(axes, stages):
        # Sort sites once, by barrier, so the x-order is comparable across T.
        order = sorted({r.site for r in rates if r.stage == stage},
                       key=lambda s: next((x.Ea_zpe or np.inf) for x in rates
                                          if x.stage == stage and x.site == s))
        idx = {s: i for i, s in enumerate(order)}

        for T, colour in zip(temps, colours):
            pts = [(idx[r.site], r.k_fwd, r.k_rev) for r in rates
                   if r.stage == stage and r.T == T and r.site in idx]
            if not pts:
                continue
            pts.sort()
            xs = [p[0] for p in pts]
            fw = [p[1] for p in pts if p[1] and p[1] > 0]
            xf = [p[0] for p in pts if p[1] and p[1] > 0]
            rv = [p[2] for p in pts if p[2] and p[2] > 0]
            xr = [p[0] for p in pts if p[2] and p[2] > 0]
            if xf:
                ax.plot(xf, np.log10(fw), 'o-', color=colour, ms=4, lw=1.1,
                        label=f'{T:.0f} K forward')
            if xr:
                ax.plot(xr, np.log10(rv), 's--', color=colour, ms=3.5, lw=1.0,
                        alpha=0.65, label=f'{T:.0f} K reverse')

        ax.set_title(f'{STAGE_LABEL[stage]} — {len(order)} sites', fontsize=9.5)
        ax.set_xlabel('site, sorted by barrier')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='best')

    axes[0].set_ylabel('log₁₀( k / s⁻¹ )')
    fig.suptitle(f'Per-site TST rates — {pretty_material(stem)}  '
                 '(reverse dashed; forward solid)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, outfile)
    plt.close(fig)
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
        print('    These enter the KMC rate table and the per-environment '
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

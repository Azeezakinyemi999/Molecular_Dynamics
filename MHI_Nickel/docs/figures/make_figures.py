#!/usr/bin/env python3
"""
make_figures.py — regenerate every figure in the documentation.

All figures are built from the fictitious system in ``example/toy.py``. Nothing
here reads a real calculation, so the documentation can be rebuilt on any
machine with no run data at all.

Usage
-----
    ~/anaconda3/envs/mace_env/bin/python docs/figures/make_figures.py

Writes PNGs beside this script and prints one line per figure.
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'example'))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from toy import TOY, KB_EV                                   # noqa: E402

OUT = _HERE
plt.rcParams.update({'figure.dpi': 140, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.spines.top': False, 'axes.spines.right': False})
_written: list[str] = []


def _save(fig, name: str):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    _written.append(name)
    print(f'  wrote {name}')


# ── F1. MSD and the fit window ───────────────────────────────────────────────
def fig_msd_window():
    """Why the slope is taken over an interior window, not the whole trace."""
    t = np.linspace(0, 100, 400)
    # schematic: ballistic start, linear middle, noisy tail
    rng = np.random.default_rng(0)
    msd = TOY.msd_slope_A2_ps * t
    msd[:40] = TOY.msd_slope_A2_ps * t[:40] ** 2 / t[39]          # ballistic
    tail = t > 80
    msd[tail] += rng.normal(0, 18, tail.sum()).cumsum() * 0.35     # thin statistics

    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.plot(t, msd, lw=1.4, color='#1f4e79', label='MSD')
    lo, hi = 0.2 * t[-1], 0.8 * t[-1]
    ax.axvspan(lo, hi, color='#1f4e79', alpha=0.10, label='fit window (0.2–0.8)')
    m = (t >= lo) & (t <= hi)
    c = np.polyfit(t[m], msd[m], 1)
    ax.plot(t, np.polyval(c, t), '--', lw=1.1, color='#b03a2e',
            label=f'slope → D = slope/6')
    ax.annotate('ballistic:\nMSD ∝ t²', xy=(6, msd[25]), xytext=(11, 330),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.annotate('few time origins\nremain', xy=(93, msd[-20]), xytext=(55, 120),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xlabel('lag time τ  [ps]'); ax.set_ylabel('MSD  [Å²]')
    ax.set_title('F1 · Both ends of an MSD trace fail, for opposite reasons')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _save(fig, 'f1_msd_fit_window.png')


# ── F2. the Boltzmann collapse over environments ─────────────────────────────
def fig_environment_collapse():
    """Why a minority environment can carry almost all of the solubility."""
    env = TOY.dh_sol_by_env()
    names = sorted(env, key=lambda k: env[k]['dH_sol_eV'])
    T = np.linspace(300, 1200, 200)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    w = [env[n]['w_env'] for n in names]
    ax.bar(names, w, color=['#1f4e79', '#8aa8c4'], width=0.55)
    for i, n in enumerate(names):
        ax.text(i, w[i] + 0.02, f"ΔH = {env[n]['dH_sol_eV']:+.2f} eV",
                ha='center', fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_ylabel('site weight $w_e$')
    ax.set_title('what the sites are')

    ax = axes[1]
    contrib = {n: env[n]['w_env'] * np.exp(-env[n]['dH_sol_eV'] / (KB_EV * T))
               for n in names}
    tot = sum(contrib.values())
    ax.stackplot(T, [100 * contrib[n] / tot for n in names],
                 labels=names, colors=['#1f4e79', '#8aa8c4'], alpha=0.9)
    ax.set_xlabel('temperature  [K]'); ax.set_ylabel('share of $S$  [%]')
    ax.set_ylim(0, 100); ax.set_title('what carries the solubility')
    ax.legend(frameon=False, fontsize=8, loc='center right')
    fig.suptitle('F2 · Weight and influence are not the same thing',
                 y=1.05, fontsize=10)
    fig.subplots_adjust(top=0.80)
    _save(fig, 'f2_environment_collapse.png')


# ── F3. quasi-harmonic floor: raise versus discard ───────────────────────────
def fig_qho_floor():
    """Why low modes are raised to a floor rather than discarded."""
    floor = 100.0
    modes = np.array(TOY.freqs_is_cm1, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for i, m_ in enumerate(modes):
        raised = max(m_, floor)
        ax.plot([i, i], [m_, raised], color='#b03a2e', lw=1.0, zorder=1)
        ax.scatter(i, m_, s=26, color='#1f4e79', zorder=3,
                   label='as computed' if i == 0 else None)
        if raised != m_:
            ax.scatter(i, raised, s=26, marker='^', color='#b03a2e', zorder=3,
                       label='raised to floor' if m_ == modes.min() else None)
    ax.axhline(floor, ls='--', lw=1.0, color='#444',
               label=f'floor = {floor:.0f} cm⁻¹')
    ax.set_xticks(range(len(modes)))
    ax.set_xlabel('mode index'); ax.set_ylabel('wavenumber  [cm⁻¹]')
    ax.set_title('F3 · Raising preserves the mode count; discarding would not')
    ax.legend(frameon=False, fontsize=8)
    _save(fig, 'f3_quasiharmonic_floor.png')


# ── F4. the two-point Arrhenius trap ─────────────────────────────────────────
def fig_two_point_trap():
    """Why R² = 1 from a two-point fit is arithmetic, not evidence."""
    # Exactly-Arrhenius data would fit perfectly at any point count, hiding the
    # very contrast this figure exists to show. Realistic scatter is added so
    # the three-point fit has residuals, while the two-point fit cannot.
    T3 = np.array(TOY.temperatures_K)
    D3 = np.array([TOY.d_at(t) for t in T3]) * np.array([1.18, 0.84, 1.10])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)

    for ax, keep, title in (
            (axes[0], slice(None), 'three points · 1 DOF'),
            (axes[1], slice(1, None), 'two points · 0 DOF')):
        x, y = 1000.0 / T3[keep], np.log(D3[keep])
        ax.scatter(1000.0 / T3, np.log(D3), s=30, facecolors='none',
                   edgecolors='#999', label='all data')
        ax.scatter(x, y, s=34, color='#1f4e79', label='used in fit', zorder=3)
        c = np.polyfit(x, y, 1)
        xx = np.linspace((1000.0 / T3).min() * 0.97, (1000.0 / T3).max() * 1.03, 20)
        ax.plot(xx, np.polyval(c, xx), '--', lw=1.1, color='#b03a2e')
        yh = np.polyval(c, x)
        ss = float(((y - yh) ** 2).sum()); st = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss / st if st else float('nan')
        ax.set_title(f'{title}\nR² = {r2:.6f}', fontsize=9,
                     color='#b03a2e' if len(x) < 3 else 'black')
        ax.set_xlabel('1000/T  [K⁻¹]')
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    axes[0].set_ylabel('ln D')
    fig.suptitle('F4 · A perfect R² from two points is arithmetic, not evidence',
                 y=1.06, fontsize=10)
    fig.subplots_adjust(top=0.78)
    _save(fig, 'f4_two_point_trap.png')


# ── F5. sign conventions on one energy profile ───────────────────────────────
def fig_sign_conventions():
    """Ea, E_des and dE defined on a single profile, with the ZPE shift."""
    x = np.linspace(0, 1, 400)
    e = 0.40 * np.sin(np.pi * x) ** 2 - 0.15 * x          # barrier + exothermic drop
    i_ts = int(np.argmax(e))
    E_IS, E_TS, E_FS = e[0], e[i_ts], e[-1]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(x, e, lw=1.8, color='#1f4e79')
    ax.plot(x, e + 0.045, lw=1.0, ls=':', color='#b03a2e')
    for xi, yi, lab in ((0, E_IS, 'IS'), (x[i_ts], E_TS, 'TS'), (1, E_FS, 'FS')):
        ax.scatter(xi, yi, s=32, color='#1f4e79', zorder=4)
        ax.annotate(lab, (xi, yi), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontweight='bold')
    def arrow(xa, y0, y1, text, dx):
        ax.annotate('', (xa, y1), (xa, y0),
                    arrowprops=dict(arrowstyle='<->', lw=1.1, color='#444'))
        ax.text(xa + dx, (y0 + y1) / 2, text, fontsize=9, va='center')
    arrow(0.16, E_IS, E_TS, r'$E_a=E_{TS}-E_{IS}$', 0.02)
    arrow(0.84, E_FS, E_TS, r'$E_{des}=E_{TS}-E_{FS}$', -0.33)
    ax.annotate('', (1.02, E_IS), (1.02, E_FS),
                arrowprops=dict(arrowstyle='<->', lw=1.1, color='#2e7d32'))
    ax.text(1.04, (E_IS + E_FS) / 2, r'$\Delta E$', color='#2e7d32', fontsize=9,
            va='center')
    ax.axhline(E_IS, lw=0.6, color='#bbb'); ax.axhline(E_FS, lw=0.6, color='#bbb')
    # lower-left is the only region of this profile with free space
    ax.text(0.06, E_FS + 0.015, 'dotted: the same profile once\nzero-point '
            'energy is added', fontsize=7.5, color='#b03a2e', ha='left',
            va='bottom')
    ax.set_xlim(-0.04, 1.22); ax.set_xticks([])
    ax.set_ylabel('energy  [eV]')
    ax.set_title('F5 · Each barrier is measured from the state it leaves',
                 pad=10)
    ax.grid(False)
    _save(fig, 'f5_sign_conventions.png')


# ── F6. the union mobile set ─────────────────────────────────────────────────
def fig_union_mobile_set():
    """Why IS and TS must displace the same atoms."""
    rng = np.random.default_rng(3)
    lat = np.array([[i, j] for i in range(5) for j in range(4)], float)
    lat[:, 0] += 0.5 * (lat[:, 1] % 2)
    lat += rng.normal(0, 0.03, lat.shape)

    cases = [((1.9, 1.45), (2.4, 1.45), 'initial state\nmolecule intact'),
             ((1.15, 1.45), (3.2, 1.45), 'transition state\natoms separated')]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1),
                             gridspec_kw={'width_ratios': [1, 1, 0.9]})
    sets = []
    for ax, (h1, h2, title) in zip(axes[:2], cases):
        near = set()
        for h in (h1, h2):
            d = np.linalg.norm(lat - np.array(h), axis=1)
            near |= set(np.argsort(d)[:6].tolist())
        sets.append(near)
        ax.scatter(lat[:, 0], lat[:, 1], s=120, color='#dfe6ec',
                   edgecolors='#aab7c4', zorder=2)
        sel = sorted(near)
        ax.scatter(lat[sel, 0], lat[sel, 1], s=120, color='#8aa8c4',
                   edgecolors='#1f4e79', lw=1.4, zorder=3)
        for h in (h1, h2):
            ax.scatter(*h, s=70, color='#b03a2e', zorder=5)
        ax.set_title(f'{title}\n{len(near)} metals displaced', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(-0.7, 4.7); ax.set_ylim(-0.7, 3.7)

    ax = axes[2]
    union = sets[0] | sets[1]
    ax.scatter(lat[:, 0], lat[:, 1], s=120, color='#dfe6ec',
               edgecolors='#aab7c4', zorder=2)
    ax.scatter(lat[sorted(union), 0], lat[sorted(union), 1], s=120,
               color='#2e7d32', edgecolors='#1b5e20', lw=1.4, alpha=0.85, zorder=3)
    ax.set_title(f'union · {len(union)} metals\nused for BOTH states', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.7, 4.7); ax.set_ylim(-0.7, 3.7)
    fig.suptitle('F6 · Per-state neighbours give different subsystems; '
                 'the union gives one', y=1.12, fontsize=10)
    fig.subplots_adjust(top=0.74)
    _save(fig, 'f6_union_mobile_set.png')


# ── F7. two-phase band ───────────────────────────────────────────────────────
def fig_two_phase_band():
    """Why the band is relaxed before the climbing image is enabled."""
    x = np.linspace(0, 1, 400)
    mep = 0.45 * np.sin(np.pi * x) ** 2 - 0.12 * x
    n = 9
    xi = np.linspace(0, 1, n + 2)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    titles = ['initial chain\n(interpolated)',
              'phase 1 · relaxed onto the path\n(loose tolerance)',
              'phase 2 · climbing image\n(full tolerance)']
    offsets = [0.19 * np.sin(np.pi * xi) + 0.05, 0.028 * np.sin(2 * np.pi * xi),
               np.zeros_like(xi)]
    for k, (ax, t, off) in enumerate(zip(axes, titles, offsets)):
        ax.plot(x, mep, lw=1.4, color='#1f4e79', zorder=1,
                label='true path' if k == 0 else None)
        y = np.interp(xi, x, mep) + off
        ax.scatter(xi, y, s=30, color='#8aa8c4', edgecolors='#1f4e79', zorder=3)
        if k == 2:
            # the climbing image sits at the true maximum of the path, not at
            # whichever interpolation node happens to be nearest it
            xm = float(x[int(np.argmax(mep))])
            ax.scatter(xm, mep.max(), s=110, marker='^', color='#b03a2e',
                       zorder=6, edgecolors='white', lw=0.8)
            ax.annotate('climbing image\ndriven uphill to the saddle',
                        (xm, mep.max()), textcoords='offset points',
                        xytext=(-6, -52), fontsize=8, ha='center',
                        arrowprops=dict(arrowstyle='->', lw=0.8))
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.grid(False)
        ax.set_xlabel('reaction coordinate')
    axes[0].set_ylabel('energy  [eV]')
    fig.suptitle('F7 · The climbing image is released only once the band '
                 'is on the path', y=1.08, fontsize=10)
    fig.subplots_adjust(top=0.76)
    _save(fig, 'f7_two_phase_band.png')


# ── F8. freeze-cutoff snapping ───────────────────────────────────────────────
def fig_freeze_snapping():
    """Why a raw thickness/3 cutoff is unsafe when the layer count is 1 mod 3."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, N in zip(axes, (12, 13)):
        z = np.arange(N, dtype=float)
        raw = z.min() + (z.max() - z.min()) / 3.0
        gaps = [(z[i] + z[i + 1]) / 2 for i in range(N - 1)]
        snapped = min(gaps, key=lambda g: abs(g - raw))
        for zi in z:
            on = abs(zi - raw) < 1e-9
            ax.plot([0, 1], [zi, zi], lw=3,
                    color='#b03a2e' if on else '#8aa8c4',
                    solid_capstyle='butt', zorder=2)
        ax.axhline(raw, ls='--', lw=1.2, color='#b03a2e', label='raw thickness/3')
        ax.axhline(snapped, ls='-', lw=1.2, color='#2e7d32', label='snapped to gap')
        hit = any(abs(z - raw) < 1e-9)
        ax.set_title(f'{N} layers   ({N} mod 3 = {N % 3})\n'
                     + ('cutoff lands ON a plane' if hit else 'cutoff falls in a gap'),
                     fontsize=9, color='#b03a2e' if hit else 'black')
        ax.set_xticks([]); ax.grid(False); ax.set_xlim(-0.1, 1.1)
    axes[0].set_ylabel('layer index (bottom = 0)')
    axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('F8 · Snapping keeps the cutoff off the atomic planes',
                 y=1.06, fontsize=10)
    fig.subplots_adjust(top=0.78)
    _save(fig, 'f8_freeze_snapping.png')


# ── F9. lateral-only replication ─────────────────────────────────────────────
def fig_lateral_replication():
    """Replicating through the thickness would invent interstitials in vacuum."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    rng = np.random.default_rng(1)
    xs = np.tile(np.arange(4, dtype=float), 4)
    zs = np.repeat(np.arange(4, dtype=float), 4)
    xs += rng.normal(0, 0.02, xs.size)

    for ax, mode in zip(axes, ('lateral only · correct', 'also in z · wrong')):
        for dx in (-4, 0, 4):
            ax.scatter(xs + dx, zs, s=48, color='#dfe6ec' if dx else '#8aa8c4',
                       edgecolors='#aab7c4' if dx else '#1f4e79', zorder=2)
        ax.axhspan(3.4, 7.2, color='#f2f6fa', zorder=0)
        ax.text(0.2, 5.3, 'vacuum', fontsize=9, color='#888')
        if 'wrong' in mode:
            for dx in (-4, 0, 4):
                ax.scatter(xs + dx, zs + 5.0, s=48, color='#f4dede',
                           edgecolors='#b03a2e', zorder=2)
            ax.scatter([1.5, 2.5], [4.4, 4.4], marker='x', s=70, lw=2,
                       color='#b03a2e', zorder=5)
            ax.annotate('interstitials that\ndo not exist', (2.0, 4.4),
                        textcoords='offset points', xytext=(14, -26), fontsize=8,
                        color='#b03a2e',
                        arrowprops=dict(arrowstyle='->', lw=0.8, color='#b03a2e'))
        ax.axvspan(-0.5, 3.5, color='#1f4e79', alpha=0.06, zorder=1)
        ax.set_title(mode, fontsize=9, color='#b03a2e' if 'wrong' in mode else 'black')
        ax.set_xlim(-4.8, 8.0); ax.set_ylim(-0.8, 8.0)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel('in-plane →'); ax.set_ylabel('through thickness →')
    fig.suptitle('F9 · The slab is periodic in plane and finite through its '
                 'thickness', y=1.06, fontsize=10)
    fig.subplots_adjust(top=0.80)
    _save(fig, 'f9_lateral_replication.png')


# ── F10. the solution-enthalpy ladder ────────────────────────────────────────
def fig_dh_sol_ladder():
    """What dH_sol counts, and where diffusivity takes over."""
    t = TOY
    env = t.dh_sol_by_env()
    best = min(env, key=lambda k: env[k]['dH_sol_eV'])
    half = 0.5 * t.dh_diss_ev
    entry = env[best]['dH_hopA_eV']
    levels = [('½ H₂\n(gas)', 0.0), ('adsorbed\natom', half),
              ('first\nsubsurface', half + entry),
              ('deeper\nlayers', half + entry + 0.03)]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, (lab, y) in enumerate(levels):
        c = '#aab7c4' if i == 3 else '#1f4e79'
        ax.plot([i - 0.32, i + 0.32], [y, y], lw=3, color=c, solid_capstyle='round')
        # the top level would collide with the title, so its label goes below
        dy, va = (-0.055, 'top') if i == 3 else (0.045, 'bottom')
        ax.text(i, y + dy, lab, ha='center', va=va, fontsize=8.5,
                color='#888' if i == 3 else 'black')
    for i in range(3):
        ax.plot([i + 0.32, i + 1 - 0.32], [levels[i][1], levels[i + 1][1]],
                ls=':', lw=1.0, color='#aaa')
    ax.annotate('', (0.5, 0.0), (0.5, half),
                arrowprops=dict(arrowstyle='<->', lw=1.2, color='#2e7d32'))
    ax.text(0.56, half / 2, r'$\frac{1}{2}\Delta H_{diss}$', color='#2e7d32', fontsize=9)
    ax.annotate('', (1.5, half), (1.5, half + entry),
                arrowprops=dict(arrowstyle='<->', lw=1.2, color='#2e7d32'))
    ax.text(1.56, half + entry / 2, r'$\Delta H_{entry}(e)$', color='#2e7d32',
            fontsize=9)
    ax.annotate('', (-0.3, 0.0), (-0.3, half + entry),
                arrowprops=dict(arrowstyle='<->', lw=1.6, color='#b03a2e'))
    ax.text(-0.42, (half + entry) / 2, r'$\Delta H_{sol}$', color='#b03a2e',
            fontsize=10, rotation=90, va='center', ha='center')
    ax.axvline(2.5, ls='--', lw=1.1, color='#888')
    ax.text(2.52, 0.08, 'beyond here is transport,\ncarried by $D$ — not $S$',
            fontsize=8, color='#666')
    ax.set_xlim(-0.75, 3.6); ax.set_xticks([]); ax.grid(False)
    ax.set_ylim(-0.08, max(y for _, y in levels) + 0.16)
    ax.set_ylabel('energy  [eV]')
    ax.set_title('F10 · Solubility counts getting in, not moving through',
                 pad=12)
    _save(fig, 'f10_dh_sol_ladder.png')


if __name__ == '__main__':
    fig_msd_window()
    fig_environment_collapse()
    fig_qho_floor()
    fig_two_point_trap()
    fig_sign_conventions()
    fig_union_mobile_set()
    fig_two_phase_band()
    fig_freeze_snapping()
    fig_lateral_replication()
    fig_dh_sol_ladder()
    print(f'\n  {len(_written)} figure(s) written to {OUT}')

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
    fig.suptitle('F2 · Weight and influence are not the same thing', y=1.02)
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
                 y=1.03)
    _save(fig, 'f4_two_point_trap.png')


if __name__ == '__main__':
    fig_msd_window()
    fig_environment_collapse()
    fig_qho_floor()
    fig_two_point_trap()
    print(f'\n  {len(_written)} figure(s) written to {OUT}')

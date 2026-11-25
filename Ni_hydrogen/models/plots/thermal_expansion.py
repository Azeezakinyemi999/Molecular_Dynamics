#!/usr/bin/env python3
"""
models/plots/thermal_expansion.py
================================
Lattice constant a₀(T) and the linear thermal-expansion coefficient, from the
NPT equilibration that precedes every diffusivity and permeability run.

a₀(T) is not just a structural curiosity here — it sets the octahedral-site
density ``4/a₀³/N_A``, which is both the geometric solubility prefactor and the
saturation ceiling every solubility route is checked against. A wrong a₀
propagates into S, into Φ = D·S, and into the ceiling used to judge them.

Two figures
-----------
  * ``<outdir>/thermal_expansion.png``
    a₀ vs T for every material, plus the fractional expansion a₀(T)/a₀(T_min)
    so materials with different absolute a₀ can be compared on one axis.
  * ``<outdir>/site_density_vs_T.png``
    the derived oct-site density, i.e. the saturation ceiling used by
    ``solubility_plot.py`` and ``permeability_plot.py``.

The linear expansion coefficient is fitted as
``α = (1/a₀) · da₀/dT`` from a straight-line fit to a₀(T), reported in
10⁻⁶ K⁻¹ for comparison with handbook values.

Typical usage
-------------
::

    python models/plots/thermal_expansion.py
    python models/plots/thermal_expansion.py --pattern 'Ni*'
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from diffusivity_plot import (            # noqa: E402
    DEFAULT_RESULTS_DIR, pretty_material, _save,
)


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

N_A = 6.02214076e23


# ---------------------------------------------------------------------------
# Section 2 — Loading
# ---------------------------------------------------------------------------

def load_lattice(results_dir: str, pattern: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """material -> (T in K, a0 in m), skipping stray duplicate directories."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, pattern,
                                              'lattice_params_vs_T.json'))):
        stem = os.path.basename(os.path.dirname(path))
        if ' ' in stem:
            print(f'  · {stem}: skipped, name contains a space '
                  f'(looks like a stray copy)')
            continue
        try:
            with open(path) as fh:
                raw = json.load(fh)
            T  = np.asarray(raw['temperatures'], float)
            a0 = np.asarray(raw['a0_m'], float)
        except (OSError, ValueError, KeyError) as exc:
            print(f'  ! {stem}: unreadable lattice file ({exc})')
            continue
        if T.size < 2 or T.size != a0.size:
            print(f'  ! {stem}: needs at least 2 matching (T, a0) points')
            continue
        order = np.argsort(T)
        out[stem] = (T[order], a0[order])
    return out


def expansion_coefficient(T: np.ndarray, a0: np.ndarray) -> tuple[float, float]:
    """(α in K⁻¹, slope stderr in K⁻¹) from a straight-line fit to a₀(T).

    α = (1/a₀)·da₀/dT, referenced to a₀ at the lowest temperature so the value
    is comparable with handbook coefficients quoted near room temperature.
    """
    from scipy.stats import linregress
    res = linregress(T, a0)
    return float(res.slope / a0[0]), float(res.stderr / a0[0])


# ---------------------------------------------------------------------------
# Section 3 — Figures
# ---------------------------------------------------------------------------

def _lattice_figure(data, which, outstem):
    import figstyle as fs
    from matplotlib.lines import Line2D
    styles = fs.material_styles(sorted(data), pretty_material)
    fig, ax = fs.new_figure()
    xs, ys, items, handles, words, keylabels, keycols = [], [], [], [], [], [], []
    for stem, (label, c, m, ls) in styles.items():
        T, a0 = data[stem]
        if which == 'a0':
            y = a0 * 1e10
            alpha, err = expansion_coefficient(T, a0)
            words.append(f'{label} {alpha * 1e6:.1f} $\\pm$ {err * 1e6:.1f}')
            keylabels.append(f'{label}\n' + r'$\alpha$ = ' + f'{alpha * 1e6:.1f} \u00b1 {err * 1e6:.1f}' + r' $\times10^{-6}$ K$^{-1}$')
        elif which == 'frac':
            y = a0 / a0[0]
            words.append(f'{label} {100 * (y[-1] - 1):.2f}\\% at {T[-1]:.0f}~K')
        else:
            y = 4.0 / a0 ** 3 / N_A
            words.append(f'{label} {fs.tex_num(y[0], 4)} to {fs.tex_num(y[-1], 4)}')
        ax.plot(T, y, color=c, ls=ls, lw=fs.CURVE_PT, marker=m, ms=10, zorder=4)
        xs.append(T)
        ys.append(y)
        items.append((T[-1], y[-1], label, c))
        handles.append(Line2D([], [], color=c, ls=ls, lw=fs.CURVE_PT, marker=m, ms=9))
        keycols.append(c)
    T_all = np.concatenate(xs)
    fs.temperature_axis(ax, T_all)
    fs.set_linear_y(ax, np.concatenate(ys), include=[1.0] if which == 'frac' else [])
    if which == 'frac':
        ax.axhline(1.0, zorder=1, **fs.REF_LINE)
    if which == 'rho':
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_ylabel({'a0': r'$a_0$ ($\mathrm{\AA}$)', 'frac': r'$a_0(T)\,/\,a_0(T_\mathrm{min})$',
                   'rho': r'Octahedral sites (mol m$^{-3}$)'}[which])
    if which == 'a0':
        fs.key_below(fig, handles, keylabels, ncol=1 if len(keylabels) < 3 else 2, colours=keycols)
        keyed = True
    else:
        keyed = fs.label_or_key(fig, ax, items, handles, ncol=2)
    if which == 'a0':
        text = (r'Lattice constant $a_0$ of each host against temperature, from NPT equilibration of the supercell. '
                r'Linear thermal expansion coefficients $\alpha = (1/a_0)\,\mathrm{d}a_0/\mathrm{d}T$, from a straight-line '
                rf"fit referenced to the lowest temperature, in $10^{{-6}}$~K$^{{-1}}$: {'; '.join(words)}.")
    elif which == 'frac':
        text = (r'Fractional thermal expansion of each host, $a_0(T)/a_0(T_\mathrm{min})$, comparable across materials '
                rf"whatever their lattice constant: {'; '.join(words)}.")
    else:
        text = (r'Density of octahedral interstitial sites, $4/a_0^3 N_A$, of each host against temperature: the '
                r'geometric solubility pre-factor $S_0$ and the ceiling every solubility route is checked against '
                rf"(mol~m$^{{-3}}$, lowest to highest temperature): {'; '.join(words)}.")
    fs.finish(fig, ax, outstem, 'plots/thermal_expansion.py', text, T_all, np.concatenate(ys), keyed=keyed)


def plot_expansion(data: dict[str, tuple[np.ndarray, np.ndarray]], outfile: str):
    """Two figures: a0 against T (<outfile>_lattice_constant.*) and the fractional expansion
    (<outfile>_fractional.*), each as PDF, PNG and caption."""
    import figstyle as fs
    _lattice_figure(data, 'a0', fs.panel_stem(outfile, 'lattice_constant'))
    _lattice_figure(data, 'frac', fs.panel_stem(outfile, 'fractional'))


def plot_site_density(data: dict[str, tuple[np.ndarray, np.ndarray]], outfile: str):
    """The octahedral-site density, the saturation ceiling used elsewhere: one figure, PDF, PNG and caption."""
    import figstyle as fs
    _lattice_figure(data, 'rho', fs.panel_stem(outfile, ''))


# ---------------------------------------------------------------------------
# Section 4 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='Plot lattice thermal expansion and the derived site density.')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                    help='directory holding the material folders '
                         '(default: calculation/results)')
    ap.add_argument('--pattern', default='*_supercell',
                    help="glob for material folders (default: '*_supercell')")
    ap.add_argument('--outdir', default=None,
                    help='where figures go (default: <results-dir>/plots)')
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    data = load_lattice(results_dir, args.pattern)
    if not data:
        print(f"No lattice_params_vs_T.json under '{args.pattern}' "
              f'in {results_dir}')
        return 1

    print(f'\n  {"material":<20s}{"T range (K)":>14s}{"a0 range (Å)":>20s}'
          f'{"α (10⁻⁶/K)":>16s}   {"oct sites (mol/m³)":<24s}')
    print('  ' + '-' * 96)
    for stem, (T, a0) in sorted(data.items()):
        alpha, err = expansion_coefficient(T, a0)
        rho = 4.0 / a0 ** 3 / N_A
        print(f'  {pretty_material(stem):<20s}'
              f'{f"{T.min():.0f}–{T.max():.0f}":>14s}'
              f'{f"{a0.min() * 1e10:.4f}–{a0.max() * 1e10:.4f}":>20s}'
              f'{f"{alpha * 1e6:.1f} ± {err * 1e6:.1f}":>16s}'
              f'   {f"{rho.max():.3e} → {rho.min():.3e}":<24s}')

    print('\nPlots:')
    plot_expansion(data, os.path.join(outdir, 'thermal_expansion.png'))
    plot_site_density(data, os.path.join(outdir, 'site_density_vs_T.png'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

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

def plot_expansion(data: dict[str, tuple[np.ndarray, np.ndarray]], outfile: str):
    """a₀ vs T (absolute) beside the fractional expansion (normalised)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(data)))

    for (stem, (T, a0)), colour in zip(sorted(data.items()), colours):
        alpha, err = expansion_coefficient(T, a0)
        ax1.plot(T, a0 * 1e10, 'o-', color=colour, ms=5, lw=1.4,
                 label=f'{pretty_material(stem)}  '
                       f'α={alpha * 1e6:.1f}±{err * 1e6:.1f}×10⁻⁶ K⁻¹')
        ax2.plot(T, a0 / a0[0], 'o-', color=colour, ms=5, lw=1.4,
                 label=pretty_material(stem))

    ax1.set_xlabel('Temperature  (K)')
    ax1.set_ylabel('$a_0$  (Å)')
    ax1.set_title('lattice constant from NPT equilibration', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')

    ax2.axhline(1.0, color='k', lw=0.8, ls='--')
    ax2.set_xlabel('Temperature  (K)')
    ax2.set_ylabel('$a_0(T)\\,/\\,a_0(T_{min})$')
    ax2.set_title('fractional expansion — comparable across materials',
                  fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')

    fig.suptitle('Thermal expansion of the host lattice', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, outfile)
    plt.close(fig)


def plot_site_density(data: dict[str, tuple[np.ndarray, np.ndarray]], outfile: str):
    """The derived oct-site density — the saturation ceiling used elsewhere."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(data)))

    for (stem, (T, a0)), colour in zip(sorted(data.items()), colours):
        rho = 4.0 / a0 ** 3 / N_A
        ax.plot(T, rho, 'o-', color=colour, ms=5, lw=1.4,
                label=f'{pretty_material(stem)}  '
                      f'{rho[0]:.3e} → {rho[-1]:.3e}')

    ax.set_xlabel('Temperature  (K)')
    ax.set_ylabel('oct-site density  (mol H m⁻³)')
    ax.set_title('Octahedral-site density $4/a_0^3/N_A$\n'
                 'the geometric $S_0$, and the saturation ceiling every '
                 'solubility route is judged against', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', title='low T → high T')
    fig.tight_layout()
    _save(fig, outfile)
    plt.close(fig)


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

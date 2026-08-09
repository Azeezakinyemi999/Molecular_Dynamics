#!/usr/bin/env python3
"""
models/plots/arrhenius_params_plot.py
=====================================
Fitted Arrhenius parameters against H concentration — with and without the
runs whose MSD never converged.

Answers "does H loading change transport?" by plotting the fitted activation
energy and pre-exponential against at.% H. Because a single under-sampled
loading can invent a trend on its own, the figure is drawn twice side by side:
every run in the left column, and only the runs whose MSD slope converged in
the right one. If the trend survives the cut it is worth talking about; if it
does not, it was an artefact.

Which runs get cut is decided by data, not by hand: msd_plot's half-vs-half
slope test flags them (``--exclude`` adds more by name if needed).

Produces, per material matched by ``--pattern``:

    <outdir>/<material>_arrhenius_params_vs_concentration.png

Typical usage
-------------
::

    python models/plots/arrhenius_params_plot.py                    # 'Al*'
    python models/plots/arrhenius_params_plot.py --exclude 1H 3H    # extra cuts
    python models/plots/arrhenius_params_plot.py --no-auto-exclude  # keep all
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from diffusivity_plot import (            # noqa: E402
    DEFAULT_RESULTS_DIR, Run, discover_runs, load_arrhenius, pretty_material,
    resolve_host_atoms, _save,
)
from msd_plot import DEFAULT_FIT_WINDOW, load_msd_curves     # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

FIGNAME = '{material}_arrhenius_params_vs_concentration.png'

# A log-scale error bar cannot reach zero. 1H's σ(D₀) is ~3× D₀ itself, so the
# lower whisker is clipped to just above the axis and reported as clipped.
LOG_ERR_FLOOR = 0.99


# ---------------------------------------------------------------------------
# Section 2 — Assembling the parameter series
# ---------------------------------------------------------------------------

def unconverged(runs: list[Run],
                fit_window: tuple[float, float]) -> dict[str, list[float]]:
    """Run name → temperatures whose MSD slope failed msd_plot's ratio test.

    Runs with no MSD files on disk cannot be judged and are left in.
    """
    flagged: dict[str, list[float]] = {}
    for run in runs:
        curves = load_msd_curves(run, fit_window)
        bad = [c.T for c in curves if not c.converged]
        if bad:
            flagged[run.name] = bad
    return flagged


def series(runs: list[Run]) -> dict[str, np.ndarray]:
    """Pull (at.%, E_a, σE_a, D₀, σD₀) out of a list of runs, sorted by loading."""
    runs = sorted(runs, key=lambda r: r.n_H)
    return {
        'x':      np.array([r.at_pct if r.at_pct is not None else r.n_H
                            for r in runs], dtype=float),
        'n_H':    np.array([r.n_H for r in runs], dtype=int),
        'Ea':     np.array([r.data['E_D'] for r in runs], dtype=float),
        'Ea_err': np.array([r.data['E_D_err'] or 0.0 for r in runs], dtype=float),
        'D0':     np.array([r.data['D0'] for r in runs], dtype=float),
        'D0_err': np.array([r.data['D0_err'] or 0.0 for r in runs], dtype=float),
    }


def _trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Ordinary least-squares slope and its standard error, or None if < 3 pts.

    Deliberately unweighted: a weighted fit would down-weight a badly
    constrained run into irrelevance, which would hide the very difference
    the two columns exist to show.
    """
    if x.size < 3:
        return None
    from scipy.stats import linregress
    res = linregress(x, y)
    return float(res.slope), float(res.stderr)


# ---------------------------------------------------------------------------
# Section 3 — The figure
# ---------------------------------------------------------------------------

def _panel(ax, x, y, y_err, n_H, *, log: bool, color: str,
           trend_units: str) -> bool:
    """Draw one parameter-vs-concentration panel. True if an error bar clipped."""
    clipped = False

    if log:
        lo = np.minimum(y_err, y * LOG_ERR_FLOOR)
        clipped = bool(np.any(y_err > y * LOG_ERR_FLOOR))
        err = np.vstack([lo, y_err])
        ax.set_yscale('log')
    else:
        err = y_err

    ax.errorbar(x, y, yerr=err, fmt='o-', color=color, ms=7, lw=1.0,
                capsize=4, alpha=0.9, zorder=5)

    for xi, yi, n in zip(x, y, n_H):
        ax.annotate(f'{n}H', xy=(xi, yi), xytext=(6, -11),
                    textcoords='offset points', fontsize=7.5, color='0.35')

    fit_y = np.log10(y) if log else y
    tr = _trend(x, fit_y)
    if tr is not None:
        slope, stderr = tr
        xs = np.linspace(x.min(), x.max(), 50)
        icept = np.mean(fit_y) - slope * np.mean(x)
        ys = slope * xs + icept
        ax.plot(xs, 10 ** ys if log else ys, '--', color='0.35', lw=1.2,
                zorder=3)
        ax.set_title(f'slope = {slope:+.4f} ± {stderr:.4f} {trend_units}',
                     fontsize=8.5, color='0.3', loc='right')

    ax.grid(True, alpha=0.3)
    return clipped


def _row_limits(values: np.ndarray, log: bool,
                margin: float = 0.22) -> tuple[float, float]:
    """Axis limits set by the fitted values, not by their error bars.

    A run whose σ dwarfs its own value would otherwise dictate the scale for
    the whole row and flatten every other point into a line. Its bar is left
    to run off the panel instead, and the caption says so.
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    if log:
        lo, hi = np.log10(lo), np.log10(hi)
    pad = margin * (hi - lo) if hi > lo else margin * abs(hi or 1.0)
    lo, hi = lo - pad, hi + pad
    return (10 ** lo, 10 ** hi) if log else (lo, hi)


def plot_params(stem: str, columns: list[tuple[str, list[Run]]],
                outfile: str, full_errorbars: bool = False):
    """Two rows (E_a, D₀) × one column per run selection, shared y per row."""
    import matplotlib.pyplot as plt

    n_col = len(columns)
    fig, axes = plt.subplots(2, n_col, figsize=(5.4 * n_col, 7.6),
                             sharey='row', squeeze=False)

    clipped = False
    rows: list[list[np.ndarray]] = [[], []]
    for col, (title, runs) in enumerate(columns):
        s = series(runs)
        rows[0].append(s['Ea'])
        rows[1].append(s['D0'])

        clipped |= _panel(axes[0][col], s['x'], s['Ea'], s['Ea_err'], s['n_H'],
                          log=False, color='tab:blue',
                          trend_units='eV / at.%')
        clipped |= _panel(axes[1][col], s['x'], s['D0'], s['D0_err'], s['n_H'],
                          log=True, color='tab:red',
                          trend_units='dec / at.%')

        axes[0][col].set_xlabel('')
        axes[1][col].set_xlabel('H concentration  (at.%)')
        axes[0][col].text(0.5, 1.13, f'{title}  (n = {len(runs)})',
                          transform=axes[0][col].transAxes, ha='center',
                          fontsize=11, fontweight='bold')

    overflow = False
    if not full_errorbars:
        for row, (values, is_log) in enumerate(zip(rows, (False, True))):
            lo, hi = _row_limits(np.concatenate(values), is_log)
            axes[row][0].set_ylim(lo, hi)       # sharey propagates across the row
            for col, (_title, runs) in enumerate(columns):
                s = series(runs)
                key, err = ('Ea', 'Ea_err') if row == 0 else ('D0', 'D0_err')
                if np.any(s[key] + s[err] > hi) or np.any(s[key] - s[err] < lo):
                    overflow = True

    axes[0][0].set_ylabel('$E_a$  (eV)')
    axes[1][0].set_ylabel('$D_0$  (m² s⁻¹)')

    caption = ('Dashed grey: unweighted least-squares trend. '
               'Point labels give the H count per supercell.')
    if overflow:
        caption += ('\nAxes are scaled to the fitted values; error bars larger '
                    'than the panel run off it — see the printed table for σ.')
    if clipped:
        caption += ('\nA lower error bar exceeded its own value and was '
                    'clipped to stay on the log axis.')
    fig.text(0.5, 0.012, caption, ha='center', fontsize=8, color='0.35')

    fig.suptitle(f'Arrhenius parameters vs H concentration — '
                 f'{pretty_material(stem)}', fontsize=13)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    _save(fig, outfile)
    return fig


# ---------------------------------------------------------------------------
# Section 4 — Console report
# ---------------------------------------------------------------------------

def report(runs: list[Run], flagged: dict[str, list[float]]) -> None:
    header = (f"  {'run':<18s}{'at.% H':>8s}{'E_a (eV)':>11s}{'σE_a':>10s}"
              f"{'σ/E_a':>8s}{'D0 (m²/s)':>12s}{'σ/D0':>9s}  status")
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for run in sorted(runs, key=lambda r: r.n_H):
        d = run.data
        ea, ea_e = d['E_D'], d['E_D_err'] or 0.0
        d0, d0_e = d['D0'], d['D0_err'] or 0.0
        bad = flagged.get(run.name)
        status = ('MSD unconverged at ' +
                  ', '.join(f'{t:.0f} K' for t in bad)) if bad else 'ok'
        pct = run.at_pct if run.at_pct is not None else float('nan')
        print(f'  {run.name:<18s}{pct:>8.2f}{ea:>11.4f}{ea_e:>10.4f}'
              f'{ea_e / ea:>8.1%}{d0:>12.3e}{d0_e / d0:>9.1%}  {status}')


def monotonicity_report(runs: list[Run], cut: set[str]) -> list[str]:
    """Two independent monotonicity checks. Returns anything that failed.

    *Across temperature* — D must rise with T for a thermally activated
    process. A run that fails this has a broken fit, not a physical result, and
    nothing downstream of it is worth reading.

    *Across concentration at fixed T* — whether D actually tracks H loading.
    Fitted E_a and D₀ can climb perfectly monotonically while D does not,
    because the two compensate in D = D₀·exp(−E_a/kT). This is the check that
    decides whether a concentration claim is supportable; the parameter plot
    on its own cannot answer it.
    """
    problems: list[str] = []

    print('  across temperature  (D must rise with T):')
    for run in sorted(runs, key=lambda r: r.n_H):
        T, D = run.data['T'], run.data['D']
        rising = bool(np.all(np.diff(D) > 0))
        trace  = '  '.join(f'{t:.0f}K {d:.3e}' for t, d in zip(T, D))
        print(f'    {run.n_H:>3d}H  {trace}   '
              f'{"ok" if rising else "NOT RISING  <-- fit is not usable"}')
        if not rising:
            problems.append(f'{run.name}: D does not rise monotonically with T')

    kept = sorted((r for r in runs if r.name not in cut), key=lambda r: r.n_H)
    if len(kept) < 2:
        return problems

    dropped = sorted(r.n_H for r in runs if r.name in cut)
    note = f'  (excluding {", ".join(f"{n}H" for n in dropped)})' if dropped else ''
    print(f'  across concentration (fixed T){note}:')

    by_T = [{float(t): float(d) for t, d in zip(r.data['T'], r.data['D'])}
            for r in kept]
    shared = sorted(set.intersection(*(set(m) for m in by_T)))
    if not shared:
        print('    no temperature is common to every run — check skipped')
        return problems

    for T in shared:
        vals = [m[T] for m in by_T]
        parts = [f'{kept[0].n_H}H {vals[0]:.3e}']
        for run, prev, cur in zip(kept[1:], vals, vals[1:]):
            parts.append(f'{"<" if cur > prev else ">"} {run.n_H}H {cur:.3e}')
        rising  = all(b > a for a, b in zip(vals, vals[1:]))
        falling = all(b < a for a, b in zip(vals, vals[1:]))
        verdict = 'monotonic' if (rising or falling) else 'NOT monotonic'
        print(f'    {T:>6.0f} K  {" ".join(parts)}   {verdict}')
        if not (rising or falling):
            problems.append(f'{T:.0f} K: D is not monotonic in H concentration')

    print('    note: sigma on D is the regression error of one MSD curve, not '
          'run-to-run\n          scatter — do not read it as a confidence '
          'interval on D.')
    return problems


# ---------------------------------------------------------------------------
# Section 5 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='Plot fitted E_a and D0 against H concentration.')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                    help='directory holding the run folders '
                         '(default: calculation/results)')
    ap.add_argument('--pattern', default='Al*',
                    help="glob for run folders, e.g. 'Ni*' (default: 'Al*')")
    ap.add_argument('--host-atoms', type=int, default=None,
                    help='host supercell atom count; skips on-disk detection')
    ap.add_argument('--outdir', default=None,
                    help='where plots go (default: <results-dir>/plots)')
    ap.add_argument('--exclude', nargs='*', default=[], metavar='SUBSTR',
                    help="extra runs to cut from the right column, matched as "
                         "substrings of the folder name, e.g. --exclude 1H")
    ap.add_argument('--no-auto-exclude', action='store_true',
                    help='do not cut runs on the MSD convergence test')
    ap.add_argument('--full-errorbars', action='store_true',
                    help='scale the axes to fit every error bar, even when one '
                         'run\'s σ flattens all the others')
    ap.add_argument('--fit-window', type=float, nargs=2, metavar=('LO', 'HI'),
                    default=list(DEFAULT_FIT_WINDOW),
                    help='MSD fit window used for the convergence test '
                         '(default: 0.2 0.8)')
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    runs = discover_runs(results_dir, args.pattern)
    ready = []
    for run in runs:
        data = load_arrhenius(run)
        if data and data['E_D'] is not None and data['D0'] is not None:
            run.data = data
            ready.append(run)

    if not ready:
        print(f"No run matching '{args.pattern}' has a fitted E_a and D0.")
        return 1

    print(f'Found {len(ready)} run(s) with Arrhenius fits matching '
          f"'{args.pattern}'")

    print('\nResolving host supercell sizes …')
    resolve_host_atoms(ready, args.host_atoms)

    flagged: dict[str, list[float]] = {}
    if not args.no_auto_exclude:
        print('\nChecking MSD convergence …')
        flagged = unconverged(ready, tuple(args.fit_window))

    cut = {r.name for r in ready
           if r.name in flagged
           or any(sub in r.name for sub in args.exclude)}

    print('\nFitted parameters:')
    report(ready, flagged)

    print('\nMonotonicity checks:')
    problems = monotonicity_report(ready, cut)

    by_stem: dict[str, list[Run]] = {}
    for run in ready:
        by_stem.setdefault(run.stem, []).append(run)

    print('\nPlots:')
    for stem, group in sorted(by_stem.items()):
        kept = [r for r in group if r.name not in cut]
        dropped = [r for r in group if r.name in cut]

        columns = [('All concentrations', group)]
        if dropped and len(kept) >= 2:
            names = ', '.join(sorted(f'{r.n_H}H' for r in dropped))
            columns.append((f'Excluding {names}', kept))
        elif dropped:
            print(f'  · {pretty_material(stem)}: cutting '
                  f'{len(dropped)} run(s) would leave only {len(kept)} — '
                  f'second column omitted')
        else:
            print(f'  · {pretty_material(stem)}: nothing flagged — '
                  f'single column only')

        material = pretty_material(stem).replace(' ', '_')
        plot_params(stem, columns,
                    os.path.join(outdir, FIGNAME.format(material=material)),
                    full_errorbars=args.full_errorbars)

    if problems:
        print(f'\n{len(problems)} monotonicity issue(s):')
        for p in problems:
            print(f'  · {p}')
        print('A concentration trend in the fitted E_a and D_0 does not carry '
              'over to D\nitself when D is non-monotonic — check the trend '
              'against D before claiming it.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

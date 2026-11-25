#!/usr/bin/env python3
"""
models/plots/msd_plot.py
========================
Mean-squared-displacement diagnostics behind the H diffusivity numbers.

Every D in ``diffusivity_table.txt`` is the slope of a straight line fitted to
MSD(t) over the middle 20–80 % of a trajectory. That fit is only meaningful if
the MSD is actually linear there, and a single H atom is a single random walker
with no ensemble averaging to smooth it. These plots make the difference
visible instead of leaving it buried in an R².

Produces, per material matched by ``--pattern``:

  * one two-panel figure per H concentration
        -> <run>/analysis/msd_vs_time.png
    Left  : MSD vs t, all temperatures, fit window shaded, fitted lines dashed.
    Right : the same on log–log, against a slope-1 Fickian reference.

  * one three-panel figure per material
        -> <outdir>/<material>_msd_all_concentrations.png
    One panel per temperature, every concentration overlaid, so an
    under-sampled loading stands out against its neighbours.

Also prints a convergence table: the fit window is split in half and the two
slopes compared. A converged MSD gives a ratio near 1; a ratio far from it
means the trajectory is too short (or too thinly populated) for the quoted D.

Typical usage
-------------
::

    python models/plots/msd_plot.py                       # pattern 'Al*'
    python models/plots/msd_plot.py --pattern 'Ni*'
    python models/plots/msd_plot.py --fit-window 0.3 0.9  # non-default fit
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import numpy as np

# diffusivity_plot lives alongside this script and puts the project root on
# sys.path when imported, which is what makes ``models`` importable below.
from diffusivity_plot import (            # noqa: E402
    DEFAULT_RESULTS_DIR, Run, discover_runs, pretty_material,
    resolve_host_atoms, _save,
)
from models.parsers import parse_diffusivity_file        # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

ANG2_PS_TO_M2S   = 1e-8            # Å² ps⁻¹ → m² s⁻¹  (matches the pipeline)
EINSTEIN_3D      = 6.0             # MSD = 6·D·t in three dimensions

DEFAULT_FIT_WINDOW = (0.2, 0.8)    # models.diffusivity_post_processing default
CONVERGED_BAND     = (0.75, 1.25)  # acceptable half-vs-half slope ratio

PER_RUN_FIGNAME  = 'msd_vs_time.png'
_MSD_RE          = re.compile(r'msd_(\d+(?:\.\d+)?)K\.txt$')


# ---------------------------------------------------------------------------
# Section 2 — Loading MSD curves
# ---------------------------------------------------------------------------

@dataclass
class MSDCurve:
    """One MSD(t) trace and everything derived from it."""

    T:         float
    t:         np.ndarray      # ps
    msd:       np.ndarray      # Å²
    t_win:     tuple[float, float]
    slope:     float           # Å² ps⁻¹ over the fit window
    intercept: float
    s_first:   float           # slope over the first half of the window
    s_second:  float           # slope over the second half
    D_table:   float | None    # m² s⁻¹ as recorded by the pipeline
    R2_table:  float | None

    @property
    def D_fit(self) -> float:
        """Diffusivity implied by the slope drawn on the plot."""
        return self.slope / EINSTEIN_3D * ANG2_PS_TO_M2S

    @property
    def ratio(self) -> float:
        """Second-half slope over first-half slope; ~1 when converged."""
        if self.s_first == 0:
            return float('nan')
        return self.s_second / self.s_first

    @property
    def converged(self) -> bool:
        return CONVERGED_BAND[0] <= self.ratio <= CONVERGED_BAND[1]


def find_msd_files(run: Run) -> dict[float, str]:
    """Map temperature → ``analysis/msd_<T>K.txt`` for one run."""
    found: dict[float, str] = {}
    for path in glob.glob(os.path.join(run.path, 'analysis', 'msd_*K.txt')):
        m = _MSD_RE.search(os.path.basename(path))
        if m:
            found[float(m.group(1))] = path
    return dict(sorted(found.items()))


def _recorded_D(run: Run) -> dict[float, tuple[float, float]]:
    """Per-temperature (D, R²) from the pipeline's diffusivity table.

    Returns an empty map when the table is missing — the plots still work,
    they just lose the cross-check against the recorded numbers.
    """
    table = os.path.join(run.path, 'analysis', 'diffusivity_table.txt')
    if not os.path.isfile(table):
        return {}
    try:
        T, D, _D_err, R2 = parse_diffusivity_file(table)
    except (OSError, ValueError) as exc:
        print(f'  ! {run.name}: could not read diffusivity_table.txt ({exc})')
        return {}
    return {float(t): (float(d), float(r)) for t, d, r in zip(T, D, R2)}


def _slope(t: np.ndarray, msd: np.ndarray, lo: float, hi: float) -> float:
    """Least-squares slope of MSD(t) over the time interval [lo, hi]."""
    mask = (t >= lo) & (t <= hi)
    if mask.sum() < 2:
        return float('nan')
    return float(np.polyfit(t[mask], msd[mask], 1)[0])


def load_msd_curves(run: Run,
                    fit_window: tuple[float, float]) -> list[MSDCurve]:
    """Read every MSD file for `run` and fit it the way the pipeline does.

    The fit window is a fraction of the *time span*, matching
    ``models.diffusivity_post_processing.fit_diffusivity``. Slope and intercept
    come from a local least-squares fit so the line can be drawn; D and R² are
    read back from the pipeline's own table and cross-checked against it.
    """
    recorded = _recorded_D(run)
    curves: list[MSDCurve] = []

    for T, path in find_msd_files(run).items():
        try:
            t, msd = np.loadtxt(path, unpack=True)
        except (OSError, ValueError) as exc:
            print(f'  ! {run.name} {T:.0f} K: unreadable MSD file ({exc})')
            continue
        if t.size < 4:
            print(f'  ! {run.name} {T:.0f} K: only {t.size} MSD points, skipping')
            continue

        span  = t[-1] - t[0]
        t_lo  = t[0] + fit_window[0] * span
        t_hi  = t[0] + fit_window[1] * span
        mask  = (t >= t_lo) & (t <= t_hi)
        if mask.sum() < 2:
            print(f'  ! {run.name} {T:.0f} K: fit window holds < 2 points')
            continue

        slope, intercept = np.polyfit(t[mask], msd[mask], 1)
        mid = 0.5 * (t_lo + t_hi)

        D_tab, R2_tab = recorded.get(T, (None, None))
        curve = MSDCurve(
            T=T, t=t, msd=msd, t_win=(t_lo, t_hi),
            slope=float(slope), intercept=float(intercept),
            s_first=_slope(t, msd, t_lo, mid),
            s_second=_slope(t, msd, mid, t_hi),
            D_table=D_tab, R2_table=R2_tab,
        )

        # A stale table would otherwise silently disagree with the drawn line.
        if D_tab and D_tab > 0:
            drift = abs(curve.D_fit - D_tab) / D_tab
            if drift > 0.02:
                print(f'  ! {run.name} {T:.0f} K: refitted D differs from the '
                      f'recorded value by {drift:.1%} '
                      f'({curve.D_fit:.3e} vs {D_tab:.3e} m² s⁻¹)')

        curves.append(curve)

    return curves


# ---------------------------------------------------------------------------
# Section 3 — Convergence report
# ---------------------------------------------------------------------------

def convergence_table(runs: list[Run],
                      curves_by_run: dict[str, list[MSDCurve]]) -> list[str]:
    """Print the half-vs-half slope check; return the unconverged run labels."""
    header = (f"  {'run':<14s}{'T (K)':>7s}{'t_max':>8s}{'MSD_end':>10s}"
              f"{'slope 1st':>11s}{'slope 2nd':>11s}{'ratio':>7s}  status")
    print(header)
    print('  ' + '-' * (len(header) - 2))

    suspect: list[str] = []
    for run in runs:
        for c in curves_by_run.get(run.name, []):
            flag = '' if c.converged else '  <-- not converged'
            if not c.converged:
                suspect.append(f'{run.name} @ {c.T:.0f} K')
            print(f'  {run.name:<14s}{c.T:>7.0f}{c.t[-1]:>8.1f}'
                  f'{c.msd[-1]:>10.1f}{c.s_first:>11.4f}{c.s_second:>11.4f}'
                  f'{c.ratio:>7.2f}{flag}')
    return suspect


# ---------------------------------------------------------------------------
# Section 4 — Per-concentration figure
# ---------------------------------------------------------------------------

def _shade_fit_window(ax, curves: list[MSDCurve]) -> bool:
    """Shade the fit window; returns False if runs disagree on where it is.

    Trajectory lengths are not always identical between temperatures, and the
    window is a *fraction* of each span — so the absolute interval can differ.
    Only the common overlap is shaded, and the caller is told when that
    happened so it can say so rather than implying a single shared window.
    """
    lo = max(c.t_win[0] for c in curves)
    hi = min(c.t_win[1] for c in curves)
    uniform = (max(abs(c.t_win[0] - curves[0].t_win[0]) for c in curves) < 1e-6
               and max(abs(c.t_win[1] - curves[0].t_win[1]) for c in curves) < 1e-6)
    if hi > lo:
        ax.axvspan(lo, hi, color='0.9', alpha=0.5, zorder=0,
                   label='fit window' if uniform else 'fit window (overlap)')
    return uniform


def plot_msd_run(run: Run, curves: list[MSDCurve], outfile: str):
    """MSD against time for one H loading, every temperature: two figures, linear with the fit
    window and fitted lines (<outfile>_linear.*) and log-log with a slope-1 guide for the diffusive
    regime (<outfile>_loglog.*), each as PDF, PNG and caption. Returns None."""
    import figstyle as fs
    from matplotlib.lines import Line2D
    from diffusivity_plot import _loading_label
    if not curves:
        return None
    temps = [c.T for c in curves]
    colours = fs.temp_styles(temps)
    mat = fs.material_styles([run.stem], pretty_material)[run.stem][0]
    handles = [Line2D([], [], color=colours[c.T], lw=fs.CURVE_PT) for c in curves]
    labels = [f'{c.T:.0f} K\n$D$ = {fs.sci(c.D_table if c.D_table else c.D_fit)} m$^2$ s$^{{-1}}$, ratio {c.ratio:.2f}'
              + ('' if c.converged else ', not converged') for c in curves]
    words = '; '.join(f'{c.T:.0f}~K $D$ = {fs.tex_num(c.D_table if c.D_table else c.D_fit, 3)}~m$^2$~s$^{{-1}}$'
                      + (f' ($R^2$ = {c.R2_table:.4f})' if c.R2_table is not None else '')
                      + f', slope ratio {c.ratio:.2f}' for c in curves)
    notconv = [f'{c.T:.0f}' for c in curves if not c.converged]
    conv_txt = (f' The {fs.tex_join(notconv)}~K trace{"s have" if len(notconv) > 1 else " has"} not converged (slope ratio '
                f'outside {CONVERGED_BAND[0]}--{CONVERGED_BAND[1]}).' if notconv else
                f' Every trace has converged (slope ratio within {CONVERGED_BAND[0]}--{CONVERGED_BAND[1]}).')
    label = _loading_label(run).replace('%', r'\%')
    for mode in ('linear', 'loglog'):
        fig, ax = fs.new_figure()
        xs, ys = [], []
        if mode == 'linear':
            lo = max(c.t_win[0] for c in curves)
            hi = min(c.t_win[1] for c in curves)
            if hi > lo:
                ax.axvspan(lo, hi, color='#EDEDED', zorder=0, lw=0)
        for c in curves:
            col = colours[c.T]
            if mode == 'linear':
                ax.plot(c.t, c.msd, color=col, lw=fs.CURVE_PT, zorder=3)
                tw = np.array(c.t_win)
                ax.plot(tw, c.slope * tw + c.intercept, color='black', lw=2.4, ls=(0, (5, 3)), zorder=5)
                xs.append(c.t)
                ys.append(c.msd)
            else:
                ok = (c.t > 0) & (c.msd > 0)
                ax.plot(c.t[ok], c.msd[ok], color=col, lw=fs.CURVE_PT, zorder=3)
                xs.append(c.t[ok])
                ys.append(c.msd[ok])
        extra_h, extra_l = [], []
        if mode == 'linear':
            ax.set_xlim(0, max(c.t[-1] for c in curves) * 1.02)
            fs.set_linear_y(ax, np.concatenate(ys), include=[0.0])
            extra_h = [Line2D([], [], color='black', lw=2.4, ls=(0, (5, 3)))]
            extra_l = ['Fit over the window']
        else:
            all_t, all_m = np.concatenate(xs), np.concatenate(ys)
            amp = float(np.median(all_m / all_t)) * 0.12
            t_g = np.array([all_t.min(), all_t.max()])
            ax.plot(t_g, amp * t_g, zorder=2, **fs.REF_LINE)
            ax.set_xscale('log')
            lo_t, hi_t = all_t.min() / 1.3, all_t.max() * 1.3
            ax.set_xlim(lo_t, hi_t)
            fs.log_ticks(ax.xaxis, lo_t, hi_t)
            fs.set_log_y(ax, np.r_[all_m, amp * t_g])
            extra_h = [Line2D([], [], **fs.REF_LINE)]
            extra_l = ['Slope 1 (Fickian)']
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel(r'MSD ($\mathrm{\AA}^2$)')
        fs.key_below(fig, handles + extra_h, labels + extra_l, ncol=1,
                     colours=[colours[c.T] for c in curves] + ['black'] * len(extra_l))
        if mode == 'linear':
            text = (rf'Mean-squared displacement of H in bulk {mat} against time for the {label} loading, one line per '
                    rf'temperature. The shaded band is the common fit window and the dashed lines the least-squares fits '
                    rf'over each window, $D$ = slope/6: {words}.{conv_txt}')
        else:
            text = (rf'The same mean-squared displacement of H in bulk {mat} ({label}) on log--log axes: a diffusive '
                    r'(Fickian) trace runs parallel to the dashed slope-1 guide; a flatter early part is the ballistic '
                    rf'and caged regime before diffusion sets in.{conv_txt}')
        fs.finish(fig, ax, fs.panel_stem(outfile, mode), 'plots/msd_plot.py', text,
                  np.concatenate(xs), np.concatenate(ys), keyed=True)
    return None


# ---------------------------------------------------------------------------
# Section 5 — Per-material figure, one panel per temperature
# ---------------------------------------------------------------------------

def plot_msd_material(stem: str, runs: list[Run],
                      curves_by_run: dict[str, list[MSDCurve]], outfile: str):
    """MSD against time, every H loading overlaid: one figure per temperature (written as
    <outfile>_<T>K.pdf, .png and _caption.tex). Each figure keeps its own y-scale, since MSD grows by
    an order of magnitude from the coldest temperature to the hottest; the comparison that matters is
    between loadings at one temperature."""
    import figstyle as fs
    from matplotlib.lines import Line2D
    from diffusivity_plot import _loading_label
    runs = sorted(runs, key=lambda r: r.n_H)
    styles = fs.loading_styles([r.n_H for r in runs])
    temps = sorted({c.T for r in runs for c in curves_by_run.get(r.name, [])})
    mat = fs.material_styles([stem], pretty_material)[stem][0]
    for T in temps:
        fig, ax = fs.new_figure()
        items, handles, xs, ys, notconv, words, shown, keylabels, keycols = [], [], [], [], [], [], [], [], []
        for run in runs:
            c = next((c for c in curves_by_run.get(run.name, []) if c.T == T), None)
            if c is None:
                continue
            col = styles[run.n_H][0]
            ax.plot(c.t, c.msd, color=col, lw=fs.CURVE_PT, zorder=4)
            if not c.converged:
                ax.plot([c.t[-1]], [c.msd[-1]], ls='none', marker='X', ms=15, color=col, mec='black', mew=1.2, zorder=6)
                notconv.append(f'{run.n_H}H')
            shown.append(c)
            xs.append(c.t)
            ys.append(c.msd)
            items.append((c.t[-1], c.msd[-1], _loading_label(run), col))
            handles.append(Line2D([], [], color=col, lw=fs.CURVE_PT))
            keylabels.append(f'{_loading_label(run)}\n$D$ = {fs.sci(c.D_table if c.D_table else c.D_fit)} m$^2$ s$^{{-1}}$'
                             + ('' if c.converged else ', not converged'))
            keycols.append(col)
            words.append(f'{run.n_H}H $D$ = {fs.tex_num(c.D_table if c.D_table else c.D_fit, 3)}~m$^2$~s$^{{-1}}$, '
                         f'slope ratio {c.ratio:.2f}')
        lo = max(c.t_win[0] for c in shown)
        hi = min(c.t_win[1] for c in shown)
        if hi > lo:
            ax.axvspan(lo, hi, color='#EDEDED', zorder=0, lw=0)
        ax.set_xlim(0, max(c.t[-1] for c in shown) * 1.02)
        fs.set_linear_y(ax, np.concatenate(ys), include=[0.0])
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel(r'MSD ($\mathrm{\AA}^2$)')
        fs.key_below(fig, handles, keylabels, ncol=1, colours=keycols)
        keyed = True
        win = shown[0].t_win
        span = shown[0].t[-1] - shown[0].t[0]
        f0, f1 = (win[0] - shown[0].t[0]) / span, (win[1] - shown[0].t[0]) / span
        text = (rf'Mean-squared displacement of H in bulk {mat} against time at {T:.0f}~K, one line per H loading of '
                rf'the supercell. The shaded band is the fit window ({100 * f0:.0f}--{100 * f1:.0f}\% of each trajectory) '
                rf'over which $D$ = slope/6 is taken: {"; ".join(words)}. The slope ratio compares the second and first '
                rf'halves of the window; a trace is taken as converged between {CONVERGED_BAND[0]} and {CONVERGED_BAND[1]}'
                + (f', so the {fs.tex_join(notconv)} trace{"s" if len(notconv) > 1 else ""} (crossed end) '
                   f'{"have" if len(notconv) > 1 else "has"} not converged.' if notconv else ', which every trace here meets.'))
        fs.finish(fig, ax, fs.panel_stem(outfile, f'{T:.0f}K'), 'plots/msd_plot.py', text,
                  np.concatenate(xs), np.concatenate(ys), keyed=keyed)
    return None


# ---------------------------------------------------------------------------
# Section 6 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='MSD diagnostics for the H diffusivity pipeline.')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                    help='directory holding the run folders '
                         '(default: calculation/results)')
    ap.add_argument('--pattern', default='Al*',
                    help="glob for run folders, e.g. 'Ni*' or '*' (default: 'Al*')")
    ap.add_argument('--include-1h', action='store_true',
                    help='keep single-H runs, which are excluded by default '
                         '(one walker, no ensemble average)')
    ap.add_argument('--host-atoms', type=int, default=None,
                    help='host supercell atom count; skips on-disk detection')
    ap.add_argument('--outdir', default=None,
                    help='where per-material plots go (default: <results-dir>/plots)')
    ap.add_argument('--fit-window', type=float, nargs=2, metavar=('LO', 'HI'),
                    default=list(DEFAULT_FIT_WINDOW),
                    help='fraction of the time span used for the linear fit '
                         '(default: 0.2 0.8)')
    args = ap.parse_args(argv)

    lo, hi = args.fit_window
    if not 0.0 <= lo < hi <= 1.0:
        print(f'Invalid --fit-window {lo} {hi}: need 0 <= LO < HI <= 1')
        return 1

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    runs = discover_runs(results_dir, args.pattern,
                         include_single_h=args.include_1h)
    if not runs:
        print(f"No H-loaded run directories matched '{args.pattern}' "
              f'in {results_dir}')
        return 1

    print(f'Found {len(runs)} H-loaded run(s) matching '
          f"'{args.pattern}' in {results_dir}")

    print('\nLoading MSD curves …')
    curves_by_run: dict[str, list[MSDCurve]] = {}
    ready, skipped = [], []
    for run in runs:
        curves = load_msd_curves(run, (lo, hi))
        if curves:
            curves_by_run[run.name] = curves
            ready.append(run)
        else:
            skipped.append(run)

    if not ready:
        print('\nNo run has MSD files (analysis/msd_<T>K.txt) — nothing to plot.')
        return 1

    print('\nResolving host supercell sizes …')
    resolve_host_atoms(ready, args.host_atoms)

    print(f'\nConvergence check (fit window {lo:g}–{hi:g} of the time span):')
    suspect = convergence_table(ready, curves_by_run)

    print('\nPer-concentration plots:')
    for run in ready:
        plot_msd_run(run, curves_by_run[run.name],
                     os.path.join(run.path, 'analysis', PER_RUN_FIGNAME))

    by_stem: dict[str, list[Run]] = {}
    for run in ready:
        by_stem.setdefault(run.stem, []).append(run)

    print('\nPer-material plots:')
    for stem, group in sorted(by_stem.items()):
        material = pretty_material(stem).replace(' ', '_')
        plot_msd_material(stem, group, curves_by_run,
                          os.path.join(outdir,
                                       f'{material}_msd_all_concentrations.png'))

    if skipped:
        print(f'\nSkipped {len(skipped)} run(s) with no MSD files:')
        for run in skipped:
            print(f'  · {run.name}')

    if suspect:
        print(f'\n{len(suspect)} trace(s) have an unconverged MSD slope '
              f'(ratio outside {CONVERGED_BAND[0]}–{CONVERGED_BAND[1]}):')
        for label in suspect:
            print(f'  · {label}')
        print('The D quoted for these is not trustworthy — lengthen the run or '
              'raise the H count to average over more walkers.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

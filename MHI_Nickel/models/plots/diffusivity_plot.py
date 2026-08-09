#!/usr/bin/env python3
"""
models/plots/diffusivity_plot.py
================================
Arrhenius plots of H diffusivity vs 1000/T, resolved by H concentration.

Reads the ``diffusivity_arrhenius.json`` written by Phase 3 of the diffusivity
pipeline in each ``calculation/results/<material>_supercell_<N>H/`` directory
and produces:

  * one plot per H concentration
        -> <run>/analysis/diffusivity_vs_invT.png
  * one overlay of every concentration of a given material
        -> <outdir>/<material>_diffusivity_all_concentrations.png

H loading is reported as at.% H, computed from the host supercell size. The
host atom count is resolved per run by the cascade in ``host_atom_count`` and,
failing that, inferred from sibling runs of the same material (a run whose
trajectories have been cleaned up still gets a correct label this way).

Runs with no ``diffusivity_arrhenius.json`` are skipped and listed at the end,
so this becomes a no-op for pending concentrations rather than an error.

Runnable from anywhere — every default path is resolved relative to this file,
not the working directory.

Typical usage
-------------
::

    python models/plots/diffusivity_plot.py                    # 'Al*'
    python models/plots/diffusivity_plot.py --pattern 'Ni*'
    python models/plots/diffusivity_plot.py --pattern '*' --outdir figures
    python models/plots/diffusivity_plot.py --host-atoms 500    # skip the cascade
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_HERE         = os.path.dirname(os.path.abspath(__file__))       # …/models/plots
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))          # …/MHI_Nickel
sys.path.insert(0, _PROJECT_ROOT)

from models.diffusivity_post_processing import arrhenius_D   # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

# Results are written by the pipeline under calculation/, two levels up from
# here — the sibling plot scripts import this rather than recomputing it, so
# moving this package only needs the two paths above to change.
DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'calculation', 'results')
DEFAULT_LIT_FILE    = os.path.join(_HERE, 'literature_diffusivity.json')
PER_RUN_FIGNAME     = 'diffusivity_vs_invT.png'   # NOT arrhenius.png — that
                                                  # name belongs to the pipeline
FIG_DPI             = 150

# Literature curves are drawn dashed in a warm palette so they never read as
# this work's data, which is solid + markers in viridis.
LIT_COLORS     = ('tab:red', 'tab:orange', 'tab:brown', 'tab:purple',
                  'tab:pink', 'dimgray', 'olive', 'teal')
LIT_LINESTYLES = ('--', '-.', ':')

# basename -> (material stem, H count).  '_10H' suffix optional; its absence
# means the pristine 0-H supercell, which carries no diffusivity by definition.
_RUN_RE = re.compile(r'^(?P<stem>.+?)(?:_(?P<nh>\d+)H)?$')


# ---------------------------------------------------------------------------
# Section 2 — Run discovery
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One material + H-loading result directory."""

    path:     str
    stem:     str                       # e.g. 'Al_supercell'
    n_H:      int
    data:     dict = field(default_factory=dict)
    n_host:   int | None = None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def at_pct(self) -> float | None:
        """H concentration in atomic percent, or None if host size unknown."""
        if not self.n_host:
            return None
        return 100.0 * self.n_H / (self.n_host + self.n_H)

    @property
    def label(self) -> str:
        """Legend label — at.% where resolvable, raw H count otherwise."""
        pct = self.at_pct
        if pct is None:
            return f'{self.n_H} H'
        return f'{pct:.2f} at.% H'


def pretty_material(stem: str) -> str:
    """'Hastelloy_N_7_supercell' -> 'Hastelloy N 7'."""
    return stem.replace('_supercell', '').replace('_', ' ') or stem


def discover_runs(results_dir: str, pattern: str) -> list[Run]:
    """Glob `pattern` under `results_dir` and parse material + H count.

    Directories with no ``_<N>H`` suffix hold the pristine host and are
    dropped — there is no hydrogen in them to diffuse.
    """
    runs: list[Run] = []
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        if not os.path.isdir(path):
            continue
        m = _RUN_RE.match(os.path.basename(path))
        if not m or m.group('nh') is None:
            continue
        runs.append(Run(path=path, stem=m.group('stem'), n_H=int(m.group('nh'))))
    return sorted(runs, key=lambda r: (r.stem, r.n_H))


# ---------------------------------------------------------------------------
# Section 3 — Loading the Arrhenius JSON
# ---------------------------------------------------------------------------

def load_arrhenius(run: Run) -> dict | None:
    """Read ``diffusivity_arrhenius.json``; return None if absent or unusable.

    Non-positive diffusivities are dropped (log₁₀ is undefined there) and
    points are sorted by temperature so the connecting geometry is sane.
    """
    jf = os.path.join(run.path, 'diffusivity_arrhenius.json')
    if not os.path.isfile(jf):
        return None

    with open(jf) as fh:
        raw = json.load(fh)

    T = np.asarray(raw.get('T_K_arr', []), dtype=float)
    D = np.asarray(raw.get('D_arr', []), dtype=float)
    if T.size == 0 or T.size != D.size:
        print(f'  ! {run.name}: malformed T_K_arr/D_arr, skipping')
        return None

    D_err = np.asarray(raw.get('D_err_arr') or np.zeros_like(D), dtype=float)
    if D_err.size != D.size:
        D_err = np.zeros_like(D)

    good = D > 0
    if not good.any():
        print(f'  ! {run.name}: no positive diffusivities, skipping')
        return None
    if not good.all():
        print(f'  ! {run.name}: dropped {int((~good).sum())} non-positive D value(s)')

    order = np.argsort(T[good])
    return {
        'T':      T[good][order],
        'D':      D[good][order],
        'D_err':  D_err[good][order],
        'E_D':    raw.get('E_D_eV'),
        'E_D_err': raw.get('E_D_err_eV'),
        'D0':     raw.get('D0_m2s'),
        'D0_err': raw.get('D0_err'),
        'R2':     raw.get('R2_fit'),
    }


# ---------------------------------------------------------------------------
# Section 4 — Host supercell size
# ---------------------------------------------------------------------------

def _atoms_from_lammps_data(path: str) -> int | None:
    """Pull `N atoms` out of a LAMMPS data-file header."""
    try:
        with open(path) as fh:
            for line in fh:
                m = re.match(r'\s*(\d+)\s+atoms\b', line)
                if m:
                    return int(m.group(1))
                if line.lstrip().startswith('Atoms'):
                    break
    except OSError:
        pass
    return None


def _atoms_from_dump(path: str) -> int | None:
    """Pull the atom count out of a LAMMPS dump header."""
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith('ITEM: NUMBER OF ATOMS'):
                    return int(next(fh).strip())
                if line.startswith('ITEM: ATOMS'):
                    break
    except (OSError, StopIteration, ValueError):
        pass
    return None


def host_atom_count(run: Run) -> int | None:
    """Host (non-H) atom count for `run`, or None if nothing on disk says.

    Every source below describes the *charged* cell, so the H count comes
    back off the total.
    """
    patterns = (
        os.path.join(run.path, 'structures', '*', f'bulk_{run.n_H}H_initial.lammps'),
        os.path.join(run.path, 'structures', '*', 'bulk_min_h.lammps'),
    )
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            total = _atoms_from_lammps_data(path)
            if total and total > run.n_H:
                return total - run.n_H

    for path in sorted(glob.glob(os.path.join(run.path, 'results', '*', '*.dump'))):
        total = _atoms_from_dump(path)
        if total and total > run.n_H:
            return total - run.n_H

    return None


def resolve_host_atoms(runs: list[Run], override: int | None) -> None:
    """Fill in ``run.n_host`` in place, inferring across a material's runs.

    A run whose trajectories and structures have been cleaned up has nothing
    locally to measure, so it borrows the modal host count from its siblings —
    the supercell is the same one in every loading of a given material.
    """
    if override:
        for run in runs:
            run.n_host = override
        return

    for run in runs:
        run.n_host = host_atom_count(run)

    by_stem: dict[str, Counter] = {}
    for run in runs:
        if run.n_host:
            by_stem.setdefault(run.stem, Counter())[run.n_host] += 1

    for run in runs:
        if run.n_host:
            continue
        counts = by_stem.get(run.stem)
        if counts:
            run.n_host = counts.most_common(1)[0][0]
            print(f'  · {run.name}: host size not on disk, '
                  f'inferred {run.n_host} atoms from sibling runs')
        else:
            print(f'  ! {run.name}: host size unknown, labelling by raw H count')


# ---------------------------------------------------------------------------
# Section 5 — Literature reference data
# ---------------------------------------------------------------------------

# Papers quote E_a in eV or kJ/mol about equally often, and D_0 in m²/s or
# cm²/s. Each entry declares its own units so values can be transcribed
# exactly as printed in the source — no hand arithmetic, no transcription slips.

EV_PER_KJMOL = 1.0 / 96.48533212          # eV per kJ mol⁻¹

_EA_TO_EV = {
    'ev':        1.0,
    'mev':       1e-3,
    'kj/mol':    EV_PER_KJMOL,
    'kjmol-1':   EV_PER_KJMOL,
    'j/mol':     EV_PER_KJMOL * 1e-3,
    'jmol-1':    EV_PER_KJMOL * 1e-3,
    'kcal/mol':  4.184 * EV_PER_KJMOL,
    'kcalmol-1': 4.184 * EV_PER_KJMOL,
}

_D0_TO_M2S = {
    'm2/s':   1.0,
    'm2s-1':  1.0,
    'cm2/s':  1e-4,
    'cm2s-1': 1e-4,
    'mm2/s':  1e-6,
    'mm2s-1': 1e-6,
}


@dataclass
class LitEntry:
    """One literature Arrhenius law, valid only over ``T_K``."""

    label:   str
    D0_m2s:  float
    Ea_eV:   float
    T_K:     tuple[float, float]
    note:    str = ''

    @property
    def legend(self) -> str:
        T_C = (self.T_K[0] - 273.15, self.T_K[1] - 273.15)
        return (f'{self.label}   —   $E_a$ = {self.Ea_eV:.3f} eV,   '
                f'{T_C[0]:.0f}–{T_C[1]:.0f} °C')


def _norm_units(raw: str) -> str:
    """'cm² s⁻¹' -> 'cm2s-1', 'kJ mol^-1' -> 'kjmol-1'."""
    s = str(raw).strip().lower()
    for a, b in (('²', '2'), ('^', ''), ('⁻', '-'), ('¹', '1'),
                 ('·', ''), (' ', ''), ('per', '/')):
        s = s.replace(a, b)
    return s


def _convert(value: float, units: str, table: dict[str, float], what: str) -> float:
    key = _norm_units(units)
    if key not in table:
        raise ValueError(
            f'unknown {what} units {units!r}; supported: {sorted(table)}')
    return float(value) * table[key]


def _parse_lit_entry(raw: dict, material: str) -> LitEntry:
    """Validate one JSON entry and convert it to eV / m² s⁻¹ / K."""
    label = str(raw.get('label') or '').strip()
    if not label:
        raise ValueError('entry is missing "label"')
    for key in ('D0', 'Ea'):
        if raw.get(key) is None:
            raise ValueError(f'{label}: missing "{key}"')

    D0 = _convert(raw['D0'], raw.get('D0_units', 'm2/s'), _D0_TO_M2S, 'D0')
    Ea = _convert(raw['Ea'], raw.get('Ea_units', 'eV'), _EA_TO_EV, 'Ea')
    if D0 <= 0:
        raise ValueError(f'{label}: D0 must be positive, got {D0}')

    # Temperature validity window: °C by default, K accepted as an alternative.
    if raw.get('T_C') is not None:
        span = [float(v) + 273.15 for v in raw['T_C']]
    elif raw.get('T_K') is not None:
        span = [float(v) for v in raw['T_K']]
    else:
        raise ValueError(f'{label}: needs "T_C" (preferred) or "T_K"')
    if len(span) != 2:
        raise ValueError(f'{label}: temperature range needs exactly 2 values')
    T_K = (min(span), max(span))
    if T_K[0] <= 0:
        raise ValueError(f'{label}: temperature range reaches absolute zero')

    return LitEntry(label=label, D0_m2s=D0, Ea_eV=Ea, T_K=T_K,
                    note=str(raw.get('note') or ''))


def load_literature(path: str) -> dict[str, list[LitEntry]]:
    """Load the literature JSON: ``{material: [entry, …]}``.

    Top-level keys beginning with ``_`` are documentation (schema notes,
    worked examples) and are ignored. A malformed entry is reported and
    dropped rather than killing the whole run.
    """
    if not os.path.isfile(path):
        print(f'  ! no literature file at {path} — comparison plots skipped')
        return {}

    with open(path) as fh:
        raw = json.load(fh)

    out: dict[str, list[LitEntry]] = {}
    for material, entries in raw.items():
        if material.startswith('_') or not isinstance(entries, list):
            continue
        parsed: list[LitEntry] = []
        for item in entries:
            try:
                parsed.append(_parse_lit_entry(item, material))
            except (ValueError, TypeError, KeyError) as exc:
                print(f'  ! literature[{material}]: {exc} — entry dropped')
        if parsed:
            out[material] = sorted(parsed, key=lambda e: e.label)
            print(f'  · {material}: {len(parsed)} literature source(s)')
    return out


def literature_for(stem: str, lit: dict[str, list[LitEntry]]) -> list[LitEntry]:
    """Match a run stem to a literature key, case- and separator-insensitively.

    'Al_supercell' matches a key of 'Al', 'al', or 'Al_supercell'.
    """
    pretty = pretty_material(stem)
    candidates = (pretty, stem, pretty.replace(' ', '_'), pretty.replace(' ', ''))
    lowered = {k.lower().replace(' ', '_'): v for k, v in lit.items()}
    for cand in candidates:
        hit = lowered.get(cand.lower().replace(' ', '_'))
        if hit:
            return hit
    return []


# ---------------------------------------------------------------------------
# Section 6 — Shared axis furniture
# ---------------------------------------------------------------------------

def _log10_yerr(D: np.ndarray, D_err: np.ndarray) -> np.ndarray | None:
    """Propagate σ_D into log₁₀ units; None when there are no error bars."""
    if not np.any(D_err > 0):
        return None
    return np.abs(D_err / (D * np.log(10.0)))


def _fit_curve(T_lo: float, T_hi: float, E_D: float, D0: float):
    """Smooth Arrhenius line over a slightly widened temperature range."""
    T_fine = np.linspace(T_lo * 0.9, T_hi * 1.1, 200)
    return 1000.0 / T_fine, np.log10(arrhenius_D(T_fine, E_D, D0))


def _temperature_ticks(T_lo: float, T_hi: float) -> np.ndarray:
    """Round temperature ticks spanning [T_lo, T_hi], at most ~8 of them.

    The step adapts to the span so a narrow MD-only window and a wide
    literature window both come out readable.
    """
    for step in (50, 100, 200, 250, 500, 1000):
        ticks = np.arange(np.ceil(T_lo / step) * step, T_hi + 1e-9, step)
        if 2 <= ticks.size <= 8:
            return ticks
    return np.linspace(T_lo, T_hi, 5)


def _thin_ticks(ticks: np.ndarray, x_lo: float, x_hi: float,
                min_gap_frac: float = 0.06) -> np.ndarray:
    """Drop temperature ticks that would collide once mapped onto 1000/T.

    Even steps in T bunch up at the hot end of a 1000/T axis, so a tick list
    that looks fine in Kelvin can render as '1000900'. Scanning from the cold
    end, where the spacing is widest, confines the thinning to the crowded end.
    """
    min_gap = min_gap_frac * abs(x_hi - x_lo)
    kept: list[float] = []
    for t in sorted(ticks):                 # ascending T == descending 1000/T
        if not kept or abs(1000.0 / t - 1000.0 / kept[-1]) >= min_gap:
            kept.append(float(t))
    return np.array(kept)


def _add_temperature_axis(ax) -> None:
    """Mirror the 1000/T axis with plain temperatures along the top.

    Ticks are derived from the axis limits actually in force, so this stays
    correct however wide the plotted temperature range turns out to be.
    """
    ax2 = ax.twiny()
    x_lo, x_hi = ax.get_xlim()
    ax2.set_xlim(x_lo, x_hi)

    # x = 1000/T, so the left edge is the *hottest* point.
    T_hi = 1000.0 / max(x_lo, 1e-9)
    T_lo = 1000.0 / max(x_hi, 1e-9)

    ticks = _thin_ticks(_temperature_ticks(T_lo, T_hi), x_lo, x_hi)
    ax2.set_xticks(1000.0 / ticks)
    ax2.set_xticklabels([f'{t:.0f}' for t in ticks])
    ax2.set_xlabel('Temperature (K)')


def _y_limits(md_y: np.ndarray, lit_y: list[np.ndarray],
              max_decades: float) -> tuple[float, float]:
    """Choose log₁₀(D) limits that keep this work's data legible.

    Published H diffusivities scatter over many decades, and a single wildly
    different source would otherwise compress the MD points into a flat
    smear. The window covers everything when it fits inside `max_decades`,
    and otherwise stays centred on the MD data.
    """
    pad = 0.35
    md_lo, md_hi = float(np.min(md_y)), float(np.max(md_y))

    lo = min([md_lo] + [float(np.min(y)) for y in lit_y])
    hi = max([md_hi] + [float(np.max(y)) for y in lit_y])
    if (hi - lo) <= max_decades:
        return lo - pad, hi + pad

    slack = max((max_decades - (md_hi - md_lo)) / 2.0, 1.0)
    return md_lo - slack, md_hi + slack


def _save(fig, outfile: str) -> None:
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches='tight')
    print(f'  Saved: {outfile}')


# ---------------------------------------------------------------------------
# Section 7 — Per-concentration plot
# ---------------------------------------------------------------------------

def plot_single(run: Run, outfile: str):
    """log₁₀(D) vs 1000/T for one H concentration, with its Arrhenius fit."""
    import matplotlib.pyplot as plt

    d = run.data
    T, D = d['T'], d['D']

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.errorbar(1000.0 / T, np.log10(D), yerr=_log10_yerr(D, d['D_err']),
                fmt='o', color='tab:blue', capsize=4, zorder=5, label='MD data')

    if d['E_D'] is not None and d['D0'] is not None:
        x_fine, y_fine = _fit_curve(T.min(), T.max(), d['E_D'], d['D0'])
        ax.plot(x_fine, y_fine, 'k-', lw=1.5, label='Arrhenius fit')

        err = f' ± {d["E_D_err"]:.3f}' if d['E_D_err'] is not None else ''
        note = (f'$E_a$ = {d["E_D"]:.3f}{err} eV\n'
                f'$D_0$ = {d["D0"]:.2e} m² s⁻¹')
        if d['R2'] is not None:
            note += f'\n$R^2$ = {d["R2"]:.3f}'
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8,
                va='bottom', ha='left',
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))

    _add_temperature_axis(ax)

    ax.set_xlabel('1000 / T  (K⁻¹)')
    ax.set_ylabel('log₁₀( D / m² s⁻¹ )')
    ax.set_title(f'H Diffusivity in {pretty_material(run.stem)} — {run.label}')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, outfile)
    return fig


# ---------------------------------------------------------------------------
# Section 8 — Combined all-concentrations plot
# ---------------------------------------------------------------------------

def _fit_legend_label(run: Run) -> str:
    """'0.20 at.% H — E_a = 0.210 eV, D_0 = 1.21e-06 m² s⁻¹'."""
    d = run.data
    if d['E_D'] is None or d['D0'] is None:
        return run.label
    return (f'{run.label}   —   $E_a$ = {d["E_D"]:.3f} eV,   '
            f'$D_0$ = {d["D0"]:.2e} m² s⁻¹')


def plot_combined(stem: str, runs: list[Run], outfile: str):
    """Overlay every H concentration of one material on a single axis."""
    import matplotlib.pyplot as plt

    runs = sorted(runs, key=lambda r: r.n_H)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(runs)))

    T_lo = min(r.data['T'].min() for r in runs)
    T_hi = max(r.data['T'].max() for r in runs)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for run, color in zip(runs, colors):
        d = run.data
        T, D = d['T'], d['D']

        ax.errorbar(1000.0 / T, np.log10(D), yerr=_log10_yerr(D, d['D_err']),
                    fmt='o', color=color, capsize=3, ms=6, zorder=5,
                    label=_fit_legend_label(run))

        if d['E_D'] is not None and d['D0'] is not None:
            x_fine, y_fine = _fit_curve(T_lo, T_hi, d['E_D'], d['D0'])
            ax.plot(x_fine, y_fine, '-', color=color, lw=1.4, alpha=0.9,
                    zorder=4)

    _add_temperature_axis(ax)

    ax.set_xlabel('1000 / T  (K⁻¹)')
    ax.set_ylabel('log₁₀( D / m² s⁻¹ )')
    ax.set_title(f'H Diffusivity in {pretty_material(stem)} — all H concentrations')
    ax.grid(True, alpha=0.3)

    # Legend below the axes: the E_a / D_0 entries are far too wide to sit
    # inside the frame without covering the data.
    ax.legend(fontsize=8, title='H loading — Arrhenius fit parameters',
              title_fontsize=9, loc='upper center',
              bbox_to_anchor=(0.5, -0.13), ncol=1 if len(runs) <= 3 else 2,
              frameon=True, borderaxespad=0.0, handletextpad=0.6,
              columnspacing=1.6)

    fig.tight_layout()
    _save(fig, outfile)
    return fig


# ---------------------------------------------------------------------------
# Section 9 — This work vs literature
# ---------------------------------------------------------------------------

def plot_vs_literature(stem: str, runs: list[Run], lit: list[LitEntry],
                       outfile: str, max_decades: float = 10.0):
    """Overlay this work's MD data on published Arrhenius laws.

    Each literature line is drawn strictly across its own stated validity
    window — a study measured over 300–600 °C is never extrapolated to meet
    the MD points. The axis spans the union of everything plotted, so where a
    source does and does not overlap this work is visible directly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs   = sorted(runs, key=lambda r: r.n_H)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(runs)))

    T_md_lo = min(r.data['T'].min() for r in runs)
    T_md_hi = max(r.data['T'].max() for r in runs)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # ── this work ────────────────────────────────────────────────────────────
    md_handles, md_labels, md_y = [], [], []
    for run, color in zip(runs, colors):
        d = run.data
        T, D = d['T'], d['D']

        ax.errorbar(1000.0 / T, np.log10(D), yerr=_log10_yerr(D, d['D_err']),
                    fmt='o', color=color, capsize=3, ms=6, zorder=6)
        md_y.append(np.log10(D))

        if d['E_D'] is not None and d['D0'] is not None:
            x_fine, y_fine = _fit_curve(T_md_lo, T_md_hi, d['E_D'], d['D0'])
            ax.plot(x_fine, y_fine, '-', color=color, lw=1.4, alpha=0.9, zorder=5)
            md_y.append(y_fine)
            label = f'{run.label}   —   $E_a$ = {d["E_D"]:.3f} eV'
        else:
            label = run.label

        md_handles.append(Line2D([], [], color=color, marker='o', ls='-',
                                 lw=1.4, ms=6))
        md_labels.append(label)

    # ── literature ───────────────────────────────────────────────────────────
    # Curves are built before anything is drawn so the y-window can be fixed
    # first; a source lying entirely outside it is left off the canvas rather
    # than stretching the axes, but still appears in the legend as off scale.
    curves = []
    for entry in lit:
        T_fine = np.linspace(entry.T_K[0], entry.T_K[1], 200)
        curves.append((entry, 1000.0 / T_fine,
                       np.log10(arrhenius_D(T_fine, entry.Ea_eV, entry.D0_m2s))))

    y_lo, y_hi = _y_limits(np.concatenate(md_y), [c[2] for c in curves],
                           max_decades)

    lit_handles, lit_labels, off_scale = [], [], []
    for i, (entry, x, y) in enumerate(curves):
        color = LIT_COLORS[i % len(LIT_COLORS)]
        style = LIT_LINESTYLES[i % len(LIT_LINESTYLES)]
        label = entry.legend

        if np.any((y >= y_lo) & (y <= y_hi)):
            ax.plot(x, y, ls=style, color=color, lw=1.8, alpha=0.95, zorder=3)
            alpha = 1.0
        else:
            off_scale.append(entry.label)
            label += '   (off scale)'
            alpha = 0.35

        lit_handles.append(Line2D([], [], color=color, ls=style, lw=1.8,
                                  alpha=alpha))
        lit_labels.append(label)

    if off_scale:
        print(f'  ! {pretty_material(stem)}: {len(off_scale)} source(s) fall '
              f'outside a {max_decades:.0f}-decade window and are not drawn: '
              f'{", ".join(off_scale)}')
        print('    (raise --max-decades to include them, or check their units)')

    ax.set_ylim(y_lo, y_hi)
    _add_temperature_axis(ax)

    ax.set_xlabel('1000 / T  (K⁻¹)')
    ax.set_ylabel('log₁₀( D / m² s⁻¹ )')
    ax.set_title(f'H Diffusivity in {pretty_material(stem)} — '
                 f'this work vs literature')
    ax.grid(True, alpha=0.3)

    # Two legends side by side beneath the plot: this work on the left,
    # literature on the right. They are anchored to the *figure* and given
    # reserved space, rather than offset in axes fractions — an axes-relative
    # offset scales with axes height while the x-label's offset is fixed in
    # points, so a long literature list slides the legend up over the label.
    rows   = max(len(md_labels), len(lit_labels)) + 1        # + title row
    fig_h  = 5.5 + max(0, rows - 5) * 0.19
    fig.set_size_inches(7.5, fig_h)
    reserved = min(0.55, (rows * 0.185 + 0.25) / fig_h)
    fig.tight_layout(rect=(0, reserved, 1, 1))

    # Anchored to the top of the reserved band so the two boxes align there
    # regardless of how many rows each one carries.
    fig.legend(md_handles, md_labels, title='This work (MD)',
               loc='upper left', bbox_to_anchor=(0.01, reserved),
               fontsize=7.5, title_fontsize=8.5, frameon=True,
               borderaxespad=0.0, handletextpad=0.6)
    fig.legend(lit_handles, lit_labels, title='Literature',
               loc='upper right', bbox_to_anchor=(0.99, reserved),
               fontsize=7.5, title_fontsize=8.5, frameon=True,
               borderaxespad=0.0, handletextpad=0.6)

    _save(fig, outfile)
    return fig


# ---------------------------------------------------------------------------
# Section 10 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='Plot H diffusivity vs 1000/T per H concentration.')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                    help='directory holding the run folders '
                         '(default: calculation/results)')
    ap.add_argument('--pattern', default='Al*',
                    help="glob for run folders, e.g. 'Ni*' or '*' "
                         "(default: 'Al*')")
    ap.add_argument('--host-atoms', type=int, default=None,
                    help='host supercell atom count; skips on-disk detection')
    ap.add_argument('--outdir', default=None,
                    help='where combined plots go (default: <results-dir>/plots)')
    ap.add_argument('--literature', default=DEFAULT_LIT_FILE,
                    help='literature JSON for the comparison plot '
                         '(default: calculation/literature_diffusivity.json)')
    ap.add_argument('--max-decades', type=float, default=10.0,
                    help='widest log₁₀(D) window on the comparison plot; '
                         'sources outside it are listed but not drawn '
                         '(default: 10)')
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    outdir      = os.path.abspath(args.outdir or os.path.join(results_dir, 'plots'))

    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        return 1

    runs = discover_runs(results_dir, args.pattern)
    if not runs:
        print(f"No H-loaded run directories matched '{args.pattern}' "
              f'in {results_dir}')
        return 1

    print(f'Found {len(runs)} H-loaded run(s) matching '
          f"'{args.pattern}' in {results_dir}")

    ready, skipped = [], []
    for run in runs:
        data = load_arrhenius(run)
        if data is None:
            skipped.append(run)
        else:
            run.data = data
            ready.append(run)

    if not ready:
        print('\nNo run has a diffusivity_arrhenius.json yet — nothing to plot.')
        _report_skipped(skipped)
        return 1

    print('\nResolving host supercell sizes …')
    resolve_host_atoms(ready, args.host_atoms)

    print('\nPer-concentration plots:')
    for run in ready:
        plot_single(run, os.path.join(run.path, 'analysis', PER_RUN_FIGNAME))

    by_stem: dict[str, list[Run]] = {}
    for run in ready:
        by_stem.setdefault(run.stem, []).append(run)

    print('\nCombined plots:')
    for stem, group in sorted(by_stem.items()):
        if len(group) < 2:
            print(f'  · {stem}: only one concentration with data '
                  f'({group[0].label}) — combined plot skipped')
            continue
        material = pretty_material(stem).replace(' ', '_')
        plot_combined(stem, group,
                      os.path.join(outdir,
                                   f'{material}_diffusivity_all_concentrations.png'))

    print('\nLoading literature data …')
    lit = load_literature(os.path.abspath(args.literature))

    print('\nLiterature comparison plots:')
    for stem, group in sorted(by_stem.items()):
        entries = literature_for(stem, lit)
        if not entries:
            print(f'  · {pretty_material(stem)}: no literature entries — '
                  f'add them to {os.path.basename(args.literature)} to enable')
            continue
        material = pretty_material(stem).replace(' ', '_')
        plot_vs_literature(stem, group, entries,
                           os.path.join(outdir,
                                        f'{material}_diffusivity_vs_literature.png'),
                           max_decades=args.max_decades)

    _report_skipped(skipped)
    return 0


def _report_skipped(skipped: list[Run]) -> None:
    if not skipped:
        return
    print(f'\nSkipped {len(skipped)} run(s) with no diffusivity_arrhenius.json:')
    for run in skipped:
        print(f'  · {run.name}')
    print('These will be picked up automatically once Phase 3 finishes.')


if __name__ == '__main__':
    raise SystemExit(main())

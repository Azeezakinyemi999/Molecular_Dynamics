#!/usr/bin/env python3
"""
models/plots/neb_mep_plot.py
============================
Minimum-energy-path plots for H arriving at, and sinking below, the surface.

Three NEB stages feed this:

  surface   H₂ dissociative adsorption   calculation/neb/<mat>/neb/<IS>__<FS1>+<FS2>/
  hopa      surface → sub1               calculation/neb_subsurface/<mat>/hopa/<site>/
  hopb      sub1    → sub2               calculation/neb_subsurface/<mat>/hopb/<site>/

Three sets of figures
---------------------
1. One MEP per NEB run          -> <run_dir>/mep.png
   ΔE along the path, the reported barrier marked, and the largest *uphill
   excursion* marked separately (see "Two barriers" below).

2. Per-stage overlay            -> <outdir>/<mat>_neb_mep_overlay_<stage>.png
   Two panels: (a) one common zero, so real differences between initial states
   stay visible; (b) every curve zeroed on its own IS, so barriers compare
   directly. Panel (a) carries a second axis in formation energy when the
   reference energies are available.

3. Chained entry pathway        -> <outdir>/<mat>_neb_mep_full_pathway.png
   dissociation → hopa → hopb as one continuous profile, built by chaining
   *relative* energies so no absolute energy is ever compared across cells of
   different stoichiometry.

Chaining, and the assumption it needs
-------------------------------------
The dissociation cell holds two H (from H₂); the hop cells hold one. Chaining
ΔE values therefore equals the true energy change only if moving one H down
costs the same whether or not a second H is adsorbed. That is testable rather
than assumed: with ``E_clean`` and ``E(H₂)`` this module compares the formation
energy of the adsorbed pair against the sum of the two isolated adsorbates and
prints the residual H–H interaction. It is NOT uniformly small: Ni comes out
at −28 to −35 meV, about 3 % of its hopA barrier, but Al reaches −274 meV,
some 27 % of its own. The earlier claim that the chaining error is "bounded and
small" was generalised from Ni alone, and an ``n_H`` guard had been preventing
the check from ever running on Al. Use ``--mep-reference formation`` to see the
gap on the figure instead of assuming it away.

Note also that a chain follows **one** H. Both adsorbed H can enter, and
distinct surface sites sometimes map to the *same* sub1/sub2 destination — the
console flags those, because two H cannot occupy one site and the independent
hop picture fails there outright.

Two barriers
------------
``neb_barrier.txt`` reports ``E_abs = max(ΔE from IS)``. When a path dips below
its own initial state — common for dissociation, which is exothermic — that
number is 0 even though the system must still climb out of the well it fell
into. Both are reported: the file's value, and ``max ascent``, the largest rise
between any earlier point and a later one, which is the barrier that actually
governs the rate.

Typical usage
-------------
::

    python models/plots/neb_mep_plot.py                  # every material
    python models/plots/neb_mep_plot.py --pattern 'Ni*'
    python models/plots/neb_mep_plot.py --no-individual  # overlays only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from diffusivity_plot import _PROJECT_ROOT, pretty_material, _save  # noqa: E402
from models.parsers import parse_neb_path                            # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 — Constants
# ---------------------------------------------------------------------------

DEFAULT_CALC_DIR = os.path.join(_PROJECT_ROOT, 'calculation')

STAGES = {
    'surface': 'H₂ dissociation on surface',
    'hopa':    'surface → sub1',
    'hopb':    'sub1 → sub2',
}

STAGE_NH = {'surface': 2, 'hopa': 1, 'hopb': 1}   # fallback if no structure file

MAX_DISTINCT_DEFAULT = 12
HIGHLIGHT_DEFAULT    = 8

_SITE_NUM  = re.compile(r'(\d+)')
_H_MASS    = 1.008
_STRUCT_CANDIDATES = ('neb_initial.lammps', 'sub1_fs_relaxed.lammps',
                      'sub2_fs_relaxed.lammps', 'neb_final_relaxed.lammps',
                      'min_fs.lammps')


# ---------------------------------------------------------------------------
# Section 2 — Reference energies
# ---------------------------------------------------------------------------

def load_references(calc_dir: str, material: str) -> tuple[float | None, float | None]:
    """(E_clean, E_H2_gas) for `material`, either None if not recorded.

    Neither is persisted in a results file — ``E_clean`` is only printed while
    the NEB workflow runs, so it is recovered from ``neb_run_<mat>.log``, and
    ``E_H2_GAS`` is a literal in the matching run script.
    """
    e_clean = e_h2 = None

    # Preferred source: the slab relaxation, which is a persisted artifact that
    # exists for every material. The last PotEng of its final thermo block is
    # the relaxed clean-slab energy. The NEB run log, used as a fallback below,
    # is transient and is absent for several metals.
    relax = os.path.join(calc_dir, 'slabs', material, 'phase2_relax',
                         'surface_relax.log')
    if os.path.isfile(relax):
        last = None
        for blk in open(relax, errors='ignore').read().split('Step')[1:]:
            hdr, *rest = blk.splitlines()
            cols = ('Step ' + hdr).split()
            if 'PotEng' not in cols:
                continue
            i = cols.index('PotEng')
            for ln in rest:
                parts = ln.split()
                if len(parts) == len(cols) and parts[0].isdigit():
                    try:
                        last = float(parts[i])
                    except ValueError:
                        pass
        e_clean = last

    if e_clean is None:
        log = os.path.join(calc_dir, f'neb_run_{material}.log')
        if os.path.isfile(log):
            hits = re.findall(r'E_CLEAN\s*[:=]\s*(-?\d+\.?\d*)',
                              open(log).read())
            if hits:
                e_clean = float(hits[-1])

    script = os.path.join(calc_dir, f'neb_run_{material}.py')
    if os.path.isfile(script):
        m = re.search(r'^E_H2_GAS\s*=\s*(-?\d+\.?\d*)', open(script).read(), re.M)
        if m:
            e_h2 = float(m.group(1))

    return e_clean, e_h2


def formation_energy(E: np.ndarray | float, n_H: int,
                     e_clean: float, e_h2: float) -> np.ndarray | float:
    """E − E_clean − (n_H/2)·E(H₂): energy referenced to H₂ gas + clean slab."""
    return E - e_clean - (n_H / 2.0) * e_h2


# ---------------------------------------------------------------------------
# Section 3 — Parsing a NEB run
# ---------------------------------------------------------------------------

def max_ascent(E: np.ndarray) -> tuple[float, int, int]:
    """Largest uphill excursion along the path, with its (from, to) indices.

    This is the barrier that governs the rate. ``max(ΔE from IS)`` misses it
    whenever the path first drops below its initial state.
    """
    lo = 0
    best, pair = 0.0, (0, 0)
    for j in range(len(E)):
        if E[j] < E[lo]:
            lo = j
        if E[j] - E[lo] > best:
            best, pair = float(E[j] - E[lo]), (lo, j)
    return best, pair[0], pair[1]


def count_h(run_dir: str) -> int | None:
    """H atoms in a run's cell, read from whichever structure file is present."""
    paths = [os.path.join(run_dir, n) for n in _STRUCT_CANDIDATES]
    paths += sorted(glob.glob(os.path.join(run_dir, '*.lammps')))
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            text = open(path).read()
            masses = text.split('Masses', 1)[1].split('Atoms', 1)[0]
            htypes = {t for t, m in re.findall(r'^\s*(\d+)\s+([\d.eE+-]+)',
                                               masses, re.M)
                      if abs(float(m) - _H_MASS) < 0.05}
            if not htypes:
                continue
            rows = [ln.split() for ln in text.split('Atoms', 1)[1].splitlines()
                    if ln.strip()]
            return sum(1 for p in rows
                       if len(p) >= 5 and p[0].isdigit() and p[1] in htypes)
        except (OSError, IndexError, ValueError):
            continue
    return None


def parse_neb_barrier(path: str) -> dict:
    """Read ``neb_barrier.txt``; missing fields are simply absent from the dict."""
    out: dict = {}
    if not os.path.isfile(path):
        return out
    text = open(path).read()

    for key in ('IS', 'FS'):
        m = re.search(rf'^\s*{key}\s*[:=]\s*(\S+)', text, re.M)
        if m:
            out[key] = m.group(1)
    for key in ('E_IS', 'E_FS', 'E_abs', 'E_des', 'delta_E', 'fmax_final'):
        m = re.search(rf'^\s*{key}\s*[:=]\s*(-?[\d.]+)', text, re.M)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    m = re.search(r'^\s*Converged\s*[:=]\s*(\w+)', text, re.M)
    if m:
        out['converged'] = m.group(1).lower() == 'true'
    return out


@dataclass
class NEBRun:
    """One NEB calculation: its path, its barrier metadata and its cell."""

    material: str
    stage:    str
    label:    str
    run_dir:  str
    frac:     np.ndarray
    E_abs:    np.ndarray
    dE:       np.ndarray
    meta:     dict = field(default_factory=dict)
    n_H:      int | None = None

    # ---- identity -------------------------------------------------------
    @property
    def is_site(self) -> str:
        """Initial-state site: 's_14__s_0+s_72' -> 's_14';  's_0' -> 's_0'."""
        return self.label.split('__', 1)[0]

    @property
    def fs_sites(self) -> list[str]:
        """Final-state sites. Surface runs deposit H at two of them."""
        if '__' not in self.label:
            return []
        return self.label.split('__', 1)[1].split('+')

    @property
    def key_site(self) -> str:
        """Site used for colour identity — the IS for hops, likewise surface."""
        return self.is_site

    # ---- energetics -----------------------------------------------------
    @property
    def Ea_reported(self) -> float:
        val = self.meta.get('E_abs')
        return float(val) if val is not None else float(np.max(self.dE))

    @property
    def Ea_ascent(self) -> float:
        return max_ascent(self.dE)[0]

    @property
    def delta_E(self) -> float:
        val = self.meta.get('delta_E')
        return float(val) if val is not None else float(self.dE[-1])

    @property
    def converged(self) -> bool | None:
        return self.meta.get('converged')

    @property
    def transition(self) -> str:
        IS, FS = self.meta.get('IS'), self.meta.get('FS')
        return f'{IS} → {FS}' if IS and FS else self.label


def _load_run(material: str, stage: str, run_dir: str) -> NEBRun | None:
    path_file = os.path.join(run_dir, 'neb_path.dat')
    if not os.path.isfile(path_file):
        return None
    try:
        frac, E_abs, dE = parse_neb_path(path_file)
    except (OSError, ValueError) as exc:
        print(f'  ! {material}/{stage}/{os.path.basename(run_dir)}: '
              f'unreadable path ({exc})')
        return None
    if len(frac) < 3:
        print(f'  ! {material}/{stage}/{os.path.basename(run_dir)}: '
              f'only {len(frac)} images, skipping')
        return None

    return NEBRun(
        material=material, stage=stage, label=os.path.basename(run_dir),
        run_dir=run_dir,
        frac=np.asarray(frac, float), E_abs=np.asarray(E_abs, float),
        dE=np.asarray(dE, float),
        meta=parse_neb_barrier(os.path.join(run_dir, 'neb_barrier.txt')),
        n_H=count_h(run_dir),
    )


# ---------------------------------------------------------------------------
# Section 4 — Discovery
# ---------------------------------------------------------------------------

def _site_order(site: str) -> tuple[int, str]:
    m = _SITE_NUM.search(site)
    return (int(m.group(1)) if m else 1 << 30, site)


def discover(calc_dir: str, pattern: str) -> dict[str, dict[str, list[NEBRun]]]:
    """material -> stage -> [NEBRun] for every material matching `pattern`."""
    found: dict[str, dict[str, list[NEBRun]]] = {}

    for mat_dir in sorted(glob.glob(os.path.join(calc_dir, 'neb', pattern))):
        if not os.path.isdir(mat_dir):
            continue
        material = os.path.basename(mat_dir)
        runs = [r for d in sorted(glob.glob(os.path.join(mat_dir, 'neb', '*')))
                if os.path.isdir(d) and (r := _load_run(material, 'surface', d))]
        if runs:
            found.setdefault(material, {})['surface'] = runs

    for mat_dir in sorted(glob.glob(os.path.join(calc_dir, 'neb_subsurface',
                                                pattern))):
        if not os.path.isdir(mat_dir):
            continue
        material = os.path.basename(mat_dir)
        for stage in ('hopa', 'hopb'):
            runs = [r for d in sorted(glob.glob(os.path.join(mat_dir, stage, '*')))
                    if os.path.isdir(d) and (r := _load_run(material, stage, d))]
            if runs:
                found.setdefault(material, {})[stage] = runs

    for stages in found.values():
        for runs in stages.values():
            runs.sort(key=lambda r: _site_order(r.key_site))
    return found


def surface_backlog(calc_dir: str, material: str, n_local: int) -> int | None:
    """Ranked surface transitions with no ``neb_path.dat`` downloaded yet."""
    ranked = os.path.join(calc_dir, 'neb', material, 'ranked_barriers.json')
    if not os.path.isfile(ranked):
        return None
    try:
        return max(0, len(json.load(open(ranked))) - n_local)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Section 5 — Styling
# ---------------------------------------------------------------------------

def site_colours(stages: dict[str, list[NEBRun]]) -> dict[str, tuple]:
    """One colour per site, shared across every stage and every figure."""
    import matplotlib.pyplot as plt

    sites = sorted({r.key_site for runs in stages.values() for r in runs},
                   key=_site_order)
    n     = 10 if len(sites) <= 10 else 20
    cmap  = plt.cm.tab10 if len(sites) <= 10 else plt.cm.tab20
    return {s: cmap(i % n) for i, s in enumerate(sites)}


def _mark_barriers(ax, run: NEBRun, y: np.ndarray, colour) -> None:
    """Mark the reported maximum and the largest uphill excursion."""
    top = int(np.argmax(run.dE))
    ax.plot(run.frac[top], y[top], 'o', color=colour, ms=5, zorder=6)

    _, i, j = max_ascent(run.dE)
    if j != top or i != 0:
        ax.annotate('', xy=(run.frac[j], y[j]), xytext=(run.frac[i], y[i]),
                    arrowprops=dict(arrowstyle='<->', color=colour, lw=1.0,
                                    alpha=0.75, shrinkA=0, shrinkB=0),
                    zorder=6)


# ---------------------------------------------------------------------------
# Section 6 — Figure set 1: one MEP per run
# ---------------------------------------------------------------------------

def plot_single_run(run: NEBRun, colour, outfile: str):
    """ΔE along one NEB path, with both barrier measures marked: one figure, PDF, PNG and caption.
    The reported maximum is an open circle; where the largest uphill climb differs from it (a path that
    first drops below its initial state), that climb is a double arrow. colour is kept for callers."""
    import figstyle as fs
    from matplotlib.lines import Line2D
    fig, ax = fs.new_figure()
    ax.plot(run.frac, run.dE, color='black', lw=fs.OUTLINE_PT, marker='o', ms=9, zorder=4)
    ax.axhline(0.0, zorder=1, **fs.REF_LINE)
    top = int(np.argmax(run.dE))
    ax.plot([run.frac[top]], [run.dE[top]], ls='none', marker='o', ms=18, mfc='none', mec=fs.OKABE['vermilion'],
            mew=3.0, zorder=6)
    handles = [Line2D([], [], ls='none', marker='o', ms=15, mfc='none', mec=fs.OKABE['vermilion'], mew=3.0)]
    labels = [f'Reported maximum, $E_a$ = {fs.energy(run.Ea_reported)}']
    colours = [fs.OKABE['vermilion']]
    asc, i, j = max_ascent(run.dE)
    if asc > run.Ea_reported + 1e-6:
        ax.annotate('', xy=(run.frac[j], run.dE[j]), xytext=(run.frac[i], run.dE[i]),
                    arrowprops=dict(arrowstyle='<->', color=fs.OKABE['blue'], lw=3.0, shrinkA=0, shrinkB=0), zorder=6)
        handles.append(Line2D([], [], color=fs.OKABE['blue'], lw=3.0))
        labels.append(f'Largest uphill climb, {fs.energy(asc)}')
        colours.append(fs.OKABE['blue'])
    path_facts = [f'$\\Delta E$ = {fs.energy(run.delta_E)}']
    if run.meta.get('E_des') is not None:
        path_facts.append(f'$E_\\mathrm{{des}}$ = {fs.energy(run.meta["E_des"])}')
    if run.converged is not None:
        path_facts.append('converged' if run.converged else 'not converged')
    handles.append(Line2D([], [], ls='none'))
    labels.append(', '.join(path_facts))
    colours.append('black')
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, 1, 6))
    fs.set_linear_y(ax, run.dE, include=[0.0], frac=0.12)
    ax.set_xlabel('Reaction coordinate')
    ax.set_ylabel('ΔE from initial state (eV)')
    fs.key_below(fig, handles, labels, ncol=1, colours=colours)
    mat = fs.material_styles([run.material], pretty_material)[run.material][0]
    esc = lambda t: str(t).replace('_', r'\_').replace('→', r'$\rightarrow$')     # noqa: E731
    facts = []
    if run.n_H is not None:
        facts.append(f'{run.n_H}-H cell')
    facts.append(f'$\\Delta E$ = {fs.ev(run.delta_E)}~eV')
    if run.meta.get('E_des') is not None:
        facts.append(f'$E_\\mathrm{{des}}$ = {fs.ev(run.meta["E_des"])}~eV')
    if run.meta.get('fmax_final') is not None:
        facts.append(f'final $f_\\mathrm{{max}}$ = {run.meta["fmax_final"]:.4f}~eV/\\AA')
    conv = ('' if run.converged is None else (' The band converged.' if run.converged else
                                              ' The band did not converge: treat the barrier as provisional.'))
    text = (rf'Nudged-elastic-band minimum-energy path on {mat}, {STAGES[run.stage].replace("H₂", "H$_2$")} '
            rf'({esc(run.transition)}; run {esc(run.label)}), energy from the initial state against the reaction '
            rf'coordinate, {len(run.frac)} images. The open circle marks the reported barrier, the highest point above '
            rf'the initial state, {fs.ev(run.Ea_reported)}~eV'
            + (rf'; the arrow the largest uphill climb, {fs.ev(asc)}~eV, the barrier that governs the rate when the path '
               r'first drops below its initial state' if asc > run.Ea_reported + 1e-6 else '')
            + f'. {"; ".join(facts)}.{conv}').replace('→', r'$\rightarrow$')
    fs.finish(fig, ax, fs.panel_stem(outfile, ''), MODULE, text, run.frac, run.dE, keyed=True)


# ---------------------------------------------------------------------------
# Section 7 — Figure set 2: per-stage overlay, two referencing schemes
# ---------------------------------------------------------------------------

STAGE_WORDS = {'surface': r'dissociation of H$_2$ on the surface',
               'hopa': 'the hop of H from the surface to the first subsurface site (hop A)',
               'hopb': 'the hop of H from the first to the second subsurface site (hop B)'}
MODULE = 'plots/neb_mep_plot.py'
GREY_LW = 2.4


def _site_name(run: NEBRun) -> str:
    """'s_14 → s_0 + s_72' for a surface path, 's_72' for a hop."""
    if run.fs_sites:
        return f"{run.is_site} → {' + '.join(run.fs_sites)}"
    return run.is_site


def _tex_site(text: str) -> str:
    """Site names for LaTeX text: underscores escaped, arrows as math."""
    return text.replace('_', r'\_').replace('→', r'$\rightarrow$')


def plot_stage_overlay(material: str, stage: str, runs: list[NEBRun],
                       colours: dict[str, tuple], outfile: str,
                       refs: tuple[float | None, float | None] = (None, None),
                       max_distinct: int = MAX_DISTINCT_DEFAULT,
                       n_highlight: int = HIGHLIGHT_DEFAULT,
                       backlog: int | None = None):
    """Every path of one stage, twice: from one common zero, the stage's lowest initial state, so
    differences between initial states stay visible (<outfile>_common_zero.*), and from each path's
    own initial state, so barriers compare directly (<outfile>_own_zero.*). Every path is grey; the
    lowest-barrier path (the largest uphill climb, as report() ranks them) is black and named.
    colours, max_distinct and n_highlight are kept for callers; the house style needs no per-site colour."""
    import figstyle as fs
    best = min(runs, key=lambda r: r.Ea_ascent)
    E_ref = min(r.E_abs[0] for r in runs)
    asc = np.array([r.Ea_ascent for r in runs])
    n_conv = sum(1 for r in runs if r.converged)
    mat = fs.material_styles([material], pretty_material)[material][0]
    e_clean, e_h2 = refs
    n_H = runs[0].n_H if runs[0].n_H is not None else STAGE_NH.get(stage)
    for mode in ('common', 'own'):
        fig, ax = fs.new_figure()
        ys = []
        for run in runs:
            y = run.E_abs - E_ref if mode == 'common' else run.dE
            ys.append(y)
            if run is not best:
                ax.plot(run.frac, y, color=fs.GREY, lw=GREY_LW, zorder=2)
        yb = best.E_abs - E_ref if mode == 'common' else best.dE
        ax.plot(best.frac, yb, color='black', lw=fs.OUTLINE_PT, zorder=5)
        _, _i, j = max_ascent(best.dE)
        ax.plot([best.frac[j]], [yb[j]], ls='none', marker='o', ms=13, color='black', zorder=6)
        ax.axhline(0.0, zorder=1, **fs.REF_LINE)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks(np.linspace(0, 1, 6))
        fs.set_linear_y(ax, np.concatenate(ys), include=[0.0], frac=0.1)
        ax.set_xlabel('Reaction coordinate')
        ax.set_ylabel('E − E(lowest initial state) (eV)' if mode == 'common' else 'ΔE from own initial state (eV)')
        ha = 'left' if best.frac[j] < 0.6 else 'right'
        fs.text_label(ax, best.frac[j] + (0.03 if ha == 'left' else -0.03), yb[j],
                      f'{_site_name(best)}\n{best.Ea_ascent:.2f} eV', ha=ha, va='bottom')
        form_txt = ''
        if mode == 'common' and e_clean is not None and e_h2 is not None and n_H:
            shift = formation_energy(E_ref, n_H, e_clean, e_h2)
            form_txt = (rf' The zero, the lowest initial state, has a formation energy of {fs.ev(shift, 2)}~eV for the '
                        rf'{n_H}-H cell relative to H$_2$ gas and the clean slab; add it to read formation energies.')
        zero = ('one common zero, the lowest initial state of the stage, so differences between initial states stay '
                'visible' if mode == 'common' else 'each path from its own initial state, so the barriers compare directly')
        text = (rf'Nudged-elastic-band minimum-energy paths for {STAGE_WORDS[stage]} on {mat}, {len(runs)} paths, plotted '
                rf'from {zero}. The barrier of a path is the largest uphill climb along it (a path that first drops below '
                rf'its initial state must climb out again): {asc.min():.2f} to {asc.max():.2f}~eV over the {len(runs)} '
                rf'paths. The black path has the lowest, {best.Ea_ascent:.2f}~eV ({_tex_site(_site_name(best))}; circle at '
                rf'its top); all others are grey. {n_conv} of {len(runs)} bands converged'
                + (f'; the barriers of the other {len(runs) - n_conv} are provisional.' if n_conv < len(runs) else '.')
                + form_txt + (f' A further {backlog} ranked transitions have no path on disk yet.' if backlog else ''))
        fs.finish(fig, ax, fs.panel_stem(outfile, f'{mode}_zero'), MODULE, text)


# ---------------------------------------------------------------------------
# Section 8 — Figure set 3: chained entry pathway
# ---------------------------------------------------------------------------

@dataclass
class Chain:
    """One H followed from H₂ through dissociation and both hops."""

    surface: NEBRun
    hop_site: str
    hopa: NEBRun
    hopb: NEBRun | None

    @property
    def label(self) -> str:
        fs = '+'.join(self.surface.fs_sites)
        tail = f'{self.surface.is_site}→{{{fs}}}, following {self.hop_site}'
        parts = [f'diss {self.surface.Ea_ascent:.2f}',
                 f'hopA {self.hopa.Ea_reported:.2f}']
        if self.hopb is not None:
            parts.append(f'hopB {self.hopb.Ea_reported:.2f}')
        return f'{tail}   ({", ".join(parts)} eV)'

    @property
    def spectator(self) -> str | None:
        """The other H's site — the one that stays adsorbed while this one moves."""
        others = [s for s in self.surface.fs_sites if s != self.hop_site]
        return others[0] if len(others) == 1 else None

    def profile_formation(self, e_clean: float, e_h2: float,
                          hopa_map: dict, per_h: bool = False):
        """Each stage at its true formation energy — nothing shifted to meet.

        Dissociation is a 2-H cell; a hop is a 1-H cell. They are put on one
        axis by noting that during a hop the system still holds the other H,
        adsorbed, so the hop is lifted by that spectator's own formation
        energy::

            y_hop = E_form(E_hop, 1) + E_form(E_spectator_adsorbed, 1)

        Both stages then describe a 2-H system, so the only discontinuity left
        is the H–H interaction — the quantity :func:`interaction_residual`
        measures and the chained profile assumes away.

        Returns ``(segments, gaps)``: a list of ``(x, y)`` arrays drawn without
        joining, and the energy step at each seam in eV. ``None`` if the
        spectator has no Hop A run, since its energy would be unknown.
        """
        spec = self.spectator
        if spec is None or spec not in hopa_map:
            return None, None

        def F(E, n):
            return formation_energy(E, n, e_clean, e_h2)

        n_surf = self.surface.n_H or STAGE_NH['surface']
        spec_E = F(hopa_map[spec].E_abs[0], 1)

        segs = [(self.surface.frac, F(self.surface.E_abs, n_surf))]
        ya = F(self.hopa.E_abs, self.hopa.n_H or STAGE_NH['hopa']) + spec_E
        segs.append((1.0 + self.hopa.frac, ya))
        if self.hopb is not None:
            yb = F(self.hopb.E_abs, self.hopb.n_H or STAGE_NH['hopb']) + spec_E
            segs.append((2.0 + self.hopb.frac, yb))

        if per_h:
            segs = [(x, y / 2.0) for x, y in segs]
        gaps = [float(segs[i + 1][1][0] - segs[i][1][-1])
                for i in range(len(segs) - 1)]
        return segs, gaps

    def profile(self) -> tuple[np.ndarray, np.ndarray]:
        """x in stage units (0-1 diss, 1-2 hopA, 2-3 hopB); y chained ΔE."""
        xs = [self.surface.frac]
        ys = [self.surface.dE]
        offset = self.surface.dE[-1]

        xs.append(1.0 + self.hopa.frac)
        ys.append(offset + self.hopa.dE)
        offset += self.hopa.dE[-1]

        if self.hopb is not None:
            xs.append(2.0 + self.hopb.frac)
            ys.append(offset + self.hopb.dE)
        return np.concatenate(xs), np.concatenate(ys)


def build_chains(stages: dict[str, list[NEBRun]]) -> list[Chain]:
    """Link each surface run to the hops starting from its final-state sites."""
    hopa = {r.key_site: r for r in stages.get('hopa', [])}
    hopb = {r.key_site: r for r in stages.get('hopb', [])}

    chains: list[Chain] = []
    for surf in stages.get('surface', []):
        for site in surf.fs_sites:
            if site in hopa:
                chains.append(Chain(surface=surf, hop_site=site,
                                    hopa=hopa[site], hopb=hopb.get(site)))
    return chains


def interaction_residual(surf: NEBRun, hopa: dict[str, NEBRun],
                         e_clean: float, e_h2: float) -> float | None:
    """H–H interaction: E_form(pair) − Σ E_form(isolated), in eV.

    Bounds the error the chaining incurs by splicing a 2-H stage onto a 1-H
    one. Needs a hopa run for *both* final-state sites of the surface run.
    """
    sites = surf.fs_sites
    if len(sites) != 2 or not all(s in hopa for s in sites):
        return None
    # A run whose structure files were cleaned up reports n_H=None; a surface
    # run always holds 2 H, so fall back to that rather than silently
    # declining to measure the chaining error (which is how Al went unchecked).
    if (surf.n_H or STAGE_NH['surface']) != 2:
        return None
    pair = formation_energy(surf.E_abs[-1], 2, e_clean, e_h2)
    iso  = sum(formation_energy(hopa[s].E_abs[0], 1, e_clean, e_h2)
               for s in sites)
    return float(pair - iso)


def _chain_barrier(c: Chain) -> float:
    """The highest single-stage barrier along a chain: the one that limits its rate."""
    parts = [c.surface.Ea_ascent, c.hopa.Ea_reported] + ([c.hopb.Ea_reported] if c.hopb is not None else [])
    return max(parts)


def _mev(v: float) -> str:
    m = round(1000 * v)
    return '0' if m == 0 else f'{m:+d}'.replace('-', '$-$')


def plot_full_pathway(material: str, stages: dict[str, list[NEBRun]],
                      colours: dict[str, tuple], outfile: str,
                      refs: tuple[float | None, float | None] = (None, None),
                      reference: str = 'chained'):
    """Dissociation -> hop A -> hop B on one axis, one curve per H followed: one figure, PDF, PNG and
    caption. Every chain is grey; the chain whose highest single-stage barrier is lowest is black and
    named. reference: 'chained' (each stage starts where the previous ended, assuming no H-H
    interaction at the seam), 'formation' (each stage at its true formation energy, seams drawn as
    dotted risers) or 'formation-per-h' (the same per H); the formation modes fall back to chained
    without E_clean and E(H2), or for a chain whose spectator H has no hop A run."""
    import figstyle as fs
    chains = build_chains(stages)
    if not chains:
        return None
    best = min(chains, key=_chain_barrier)
    e_clean, e_h2 = refs
    hopa_map = {r.key_site: r for r in stages.get('hopa', [])}
    want_form = reference in ('formation', 'formation-per-h')
    use_form = want_form and e_clean is not None and e_h2 is not None
    fig, ax = fs.new_figure()
    ys, gaps_all, n_form = [], [], 0
    for c in chains:
        colour, lw, z = ('black', fs.OUTLINE_PT, 5) if c is best else (fs.GREY, GREY_LW, 2)
        segs = None
        if use_form:
            segs, gaps = c.profile_formation(e_clean, e_h2, hopa_map, per_h=(reference == 'formation-per-h'))
            if segs is not None:
                gaps_all.extend(gaps)
                n_form += 1
        if segs is None:
            x, y = c.profile()
            ax.plot(x, y, color=colour, lw=lw, zorder=z)
            ys.append(y)
        else:
            for k, (xs_, ys_) in enumerate(segs):
                ax.plot(xs_, ys_, color=colour, lw=lw, zorder=z)
                ys.append(ys_)
                if k + 1 < len(segs):
                    xv = float(segs[k + 1][0][0])
                    ax.plot([xv, xv], [ys_[-1], segs[k + 1][1][0]], color=colour, lw=2.4, ls=(0, (1, 1.5)), zorder=z)
    for xv in (1.0, 2.0):
        ax.axvline(xv, color='black', lw=2.4, ls=(0, (1, 2)), zorder=1)
    ax.axhline(0.0, zorder=1, **fs.REF_LINE)
    ax.set_xlim(-0.05, 3.05)
    ax.set_xticks([0, 1, 2, 3], [r'H$_2$ ads.', '2H ads.', 'sub1', 'sub2'])
    ax.set_xlabel('Reaction coordinate')
    fs.set_linear_y(ax, np.concatenate(ys), include=[0.0], frac=0.1)
    per = ' per H' if reference == 'formation-per-h' else ''
    ax.set_ylabel(f'$E_\\mathrm{{form}}${per} of the 2-H system (eV)' if (use_form and n_form) else 'Chained ΔE (eV)')
    fs.text_label(ax, 0.03, 0.05, f'{_site_name(best.surface)}, then {best.hop_site}', ha='left', va='bottom',
                  coords='axes')
    parts = [f'dissociation {best.surface.Ea_ascent:.2f}', f'hop A {best.hopa.Ea_reported:.2f}']
    if best.hopb is not None:
        parts.append(f'hop B {best.hopb.Ea_reported:.2f}')
    mat = fs.material_styles([material], pretty_material)[material][0]
    if use_form and n_form:
        how = (r'each stage at its true formation energy, nothing shifted to meet; dotted risers are the gaps between '
               r'stages, the H--H interaction' + (f' (largest {_mev(max(gaps_all, key=abs))}~meV)' if gaps_all else '')
               + (f'; {len(chains) - n_form} of {len(chains)} chains have no hop A run for the spectator H and are '
                  'chained instead' if n_form < len(chains) else ''))
    else:
        how = 'chaining the energies of each stage relative to its own initial state'
        if want_form:
            how += ' (the formation reference was asked for, but the reference energies are not available)'
    res_txt = ''
    if e_clean is not None and e_h2 is not None and not (use_form and n_form):
        res = [r for sf in stages.get('surface', []) if (r := interaction_residual(sf, hopa_map, e_clean, e_h2)) is not None]
        if res:
            res_txt = (r' Chaining a two-H stage onto one-H stages assumes the two H do not interact; the measured H--H '
                       rf'interaction is {_mev(min(res))} to {_mev(max(res))}~meV (largest magnitude '
                       rf'{_mev(max(res, key=abs))}~meV).')
    text = (rf'Entry path of one H into {mat}: H$_2$ adsorbed $\rightarrow$ dissociated (2H adsorbed) $\rightarrow$ first '
            rf'subsurface site (sub1) $\rightarrow$ second (sub2), {how}. Each of the {len(chains)} curves follows one of '
            rf"the two dissociated H atoms from one of {len(stages.get('surface', []))} dissociation paths into the hops "
            rf'starting at its site. The black curve has the lowest highest barrier of any chain '
            rf'({_tex_site(_site_name(best.surface))}, then {_tex_site(best.hop_site)}: {", ".join(parts)}~eV); all others '
            rf'are grey.{res_txt}')
    fs.finish(fig, ax, fs.panel_stem(outfile, ''), MODULE, text)
    return outfile


# ---------------------------------------------------------------------------
# Section 9 — Console report
# ---------------------------------------------------------------------------

def report(material: str, stages: dict[str, list[NEBRun]],
           refs: tuple[float | None, float | None]) -> list[str]:
    """Print barriers and consistency checks; return unconverged run labels."""
    problems: list[str] = []

    for stage in STAGES:
        runs = stages.get(stage)
        if not runs:
            continue
        print(f'  {stage:8s} ({STAGES[stage]}) — {len(runs)} path(s)')
        print(f'    {"run":<22s}{"transition":<26s}{"nH":>3s}'
              f'{"E_a rep":>9s}{"ascent":>8s}{"ΔE":>9s}')
        for run in runs:
            flag = '' if run.converged is not False else '  <-- NOT converged'
            if run.converged is False:
                problems.append(f'{material}/{stage}/{run.label}')
            gap = '  *' if run.Ea_ascent > run.Ea_reported + 1e-6 else ''
            print(f'    {run.label:<22s}{run.transition:<26s}'
                  f'{run.n_H if run.n_H is not None else "?":>3}'
                  f'{run.Ea_reported:>9.3f}{run.Ea_ascent:>8.3f}'
                  f'{run.delta_E:>+9.3f}{gap}{flag}')

    understated = [f'{s}/{r.label}' for s, rs in stages.items() for r in rs
                   if r.Ea_ascent > r.Ea_reported + 1e-6]
    if understated:
        print(f'    * {len(understated)} path(s) dip below their own IS, so the '
              f'reported E_a understates the\n      real barrier — compare the '
              f'"ascent" column.')

    # sub1 / sub2 destinations that two different sites both map onto
    for stage, field_ in (('hopa', 'sub1'), ('hopb', 'sub2')):
        dest: dict[str, list[str]] = {}
        for run in stages.get(stage, []):
            fs = run.meta.get('FS')
            if fs:
                dest.setdefault(fs, []).append(run.key_site)
        shared = {k: v for k, v in dest.items() if len(v) > 1}
        for target, sites in sorted(shared.items()):
            print(f'  ! {stage}: {", ".join(sites)} all map to {target} — '
                  f'two H cannot occupy it, so independent hops fail there')

    e_clean, e_h2 = refs
    hopa_map = {r.key_site: r for r in stages.get('hopa', [])}
    if e_clean is not None and e_h2 is not None:
        for surf in stages.get('surface', []):
            res = interaction_residual(surf, hopa_map, e_clean, e_h2)
            if res is not None:
                print(f'  · H–H interaction at {"+".join(surf.fs_sites)}: '
                      f'{res * 1000:+.0f} meV '
                      f'(bounds the chaining error for that pair)')
    else:
        print('  · no E_clean / E_H2 for this material — formation-energy axis '
              'and the H–H check are skipped')
    return problems


# ---------------------------------------------------------------------------
# Section 10 — Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='NEB minimum-energy-path plots for surface and subsurface H.')
    ap.add_argument('--calc-dir', default=DEFAULT_CALC_DIR,
                    help='holds neb/, neb_subsurface/ and the neb_run_* logs '
                         '(default: calculation)')
    ap.add_argument('--pattern', default='*',
                    help="glob for material folders, e.g. 'Ni*' (default: '*')")
    ap.add_argument('--outdir', default=None,
                    help='overlay/pathway figures '
                         '(default: <calc-dir>/results/plots)')
    ap.add_argument('--mep-reference', default='chained',
                    choices=('chained', 'formation', 'formation-per-h'),
                    help="how the stages are placed on the full-pathway "
                         "figure: 'chained' shifts each to meet the last "
                         "(assumes no H-H interaction at the seam); "
                         "'formation' draws each at its true formation energy "
                         "and shows the gap instead (default: chained)")
    ap.add_argument('--no-individual', action='store_true',
                    help='skip the per-run MEP figures')
    ap.add_argument('--max-individual', type=int, default=60,
                    help='skip per-run figures for a material with more runs '
                         'than this — Hastelloy N 7 alone has ~1000 surface '
                         'paths, which takes minutes to render (default: 60)')
    ap.add_argument('--max-distinct', type=int, default=MAX_DISTINCT_DEFAULT,
                    help='per-site colours up to this many paths in a stage '
                         f'(default: {MAX_DISTINCT_DEFAULT})')
    ap.add_argument('--highlight', type=int, default=HIGHLIGHT_DEFAULT,
                    help='paths highlighted once a stage is crowded '
                         f'(default: {HIGHLIGHT_DEFAULT})')
    args = ap.parse_args(argv)

    calc_dir = os.path.abspath(args.calc_dir)
    mep_reference = args.mep_reference
    outdir   = os.path.abspath(args.outdir or
                               os.path.join(calc_dir, 'results', 'plots'))

    if not os.path.isdir(calc_dir):
        print(f'Calculation directory not found: {calc_dir}')
        return 1

    found = discover(calc_dir, args.pattern)
    if not found:
        print(f"No NEB paths matched '{args.pattern}' under "
              f'{calc_dir}/neb and {calc_dir}/neb_subsurface')
        return 1

    problems: list[str] = []
    for material, stages in sorted(found.items()):
        total = sum(len(v) for v in stages.values())
        print(f'\n{pretty_material(material)}  ({material}) — {total} path(s), '
              f'stages: {", ".join(s for s in STAGES if s in stages)}')

        refs    = load_references(calc_dir, material)
        colours = site_colours(stages)
        backlog = surface_backlog(calc_dir, material,
                                  len(stages.get('surface', [])))

        problems += report(material, stages, refs)
        if backlog:
            print(f'  ! surface: {backlog} ranked transition(s) have no '
                  f'neb_path.dat locally')

        if args.no_individual:
            pass
        elif total > args.max_individual:
            print(f'  · {total} runs exceeds --max-individual '
                  f'({args.max_individual}) — per-run figures skipped; '
                  f'raise the limit to force them')
        else:
            n = 0
            for runs in stages.values():
                for run in runs:
                    plot_single_run(run, colours[run.key_site],
                                    os.path.join(run.run_dir, 'mep.png'))
                    n += 1
            print(f'  {n} individual MEP figure(s) written beside their data')

        stem = pretty_material(material).replace(' ', '_')
        for stage, runs in stages.items():
            plot_stage_overlay(
                material, stage, runs, colours,
                os.path.join(outdir, f'{stem}_neb_mep_overlay_{stage}.png'),
                refs=refs, max_distinct=args.max_distinct,
                n_highlight=args.highlight,
                backlog=backlog if stage == 'surface' else None)

        if plot_full_pathway(
                material, stages, colours,
                os.path.join(outdir, f'{stem}_neb_mep_full_pathway.png'),
                refs=refs, reference=mep_reference) is None:
            have = ', '.join(s for s in STAGES if s in stages)
            print(f'  · no surface run links to an available hopa '
                  f'(stages present: {have}) — pathway plot skipped')

    if problems:
        print(f'\n{len(problems)} NEB run(s) did not converge:')
        for label in problems:
            print(f'  · {label}')
        print('Their saddles may be unresolved — treat those barriers as '
              'provisional.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

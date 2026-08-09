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
prints the residual H–H interaction. For Ni it is −28 meV, i.e. 3 % of the
0.86 eV hopA barrier, so the chaining error is bounded and small.

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

    log = os.path.join(calc_dir, f'neb_run_{material}.log')
    if os.path.isfile(log):
        hits = re.findall(r'E_CLEAN\s*[:=]\s*(-?\d+\.?\d*)', open(log).read())
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
    """ΔE along one NEB path, with both barrier measures annotated."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    ax.plot(run.frac, run.dE, '-o', color=colour, lw=1.6, ms=4.5, zorder=4)
    ax.axhline(0.0, color='k', lw=0.8, ls='--', zorder=2)

    top = int(np.argmax(run.dE))
    ax.plot(run.frac[top], run.dE[top], 'o', color='k', ms=7, mfc='none',
            mew=1.4, zorder=6, label=f'reported $E_a$ = {run.Ea_reported:.3f} eV')

    asc, i, j = max_ascent(run.dE)
    if asc > run.Ea_reported + 1e-6:
        ax.annotate('', xy=(run.frac[j], run.dE[j]),
                    xytext=(run.frac[i], run.dE[i]),
                    arrowprops=dict(arrowstyle='<->', color='tab:red', lw=1.4),
                    zorder=6)
        ax.plot([], [], '-', color='tab:red', lw=1.4,
                label=f'max ascent = {asc:.3f} eV')

    note = [f'{run.transition}']
    if run.n_H is not None:
        note.append(f'cell: {run.n_H} H')
    note.append(f'ΔE = {run.delta_E:+.3f} eV')
    if run.meta.get('E_des') is not None:
        note.append(f'$E_{{des}}$ = {run.meta["E_des"]:.3f} eV')
    if run.meta.get('fmax_final') is not None:
        note.append(f'$f_{{max}}$ = {run.meta["fmax_final"]:.4f} eV/Å')
    if run.converged is not None:
        note.append('converged' if run.converged else 'NOT CONVERGED')
    # Keep the note and the legend out of the curve: a net-downhill path leaves
    # the top-left free, a net-uphill one leaves the bottom-right free.
    downhill = run.dE[-1] < run.dE[0]
    nx, ny, va, ha = (0.02, 0.98, 'top', 'left') if downhill else \
                     (0.98, 0.02, 'bottom', 'right')
    ax.text(nx, ny, '\n'.join(note), transform=ax.transAxes, fontsize=7.5,
            va=va, ha=ha, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))

    ax.set_xlabel('Reaction coordinate')
    ax.set_ylabel('ΔE from initial state  (eV)')
    ax.set_title(f'{pretty_material(run.material)} — {run.stage}  '
                 f'({STAGES[run.stage]})\n{run.label}', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right' if downhill else 'upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, outfile)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 7 — Figure set 2: per-stage overlay, two referencing schemes
# ---------------------------------------------------------------------------

def plot_stage_overlay(material: str, stage: str, runs: list[NEBRun],
                       colours: dict[str, tuple], outfile: str,
                       refs: tuple[float | None, float | None] = (None, None),
                       max_distinct: int = MAX_DISTINCT_DEFAULT,
                       n_highlight: int = HIGHLIGHT_DEFAULT,
                       backlog: int | None = None):
    """(a) one common zero, preserving IS differences; (b) each curve on its own."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    crowded = len(runs) > max_distinct
    keep    = ({r.label for r in sorted(runs, key=lambda r: r.Ea_ascent)[:n_highlight]}
               if crowded else {r.label for r in runs})

    E_ref = min(r.E_abs[0] for r in runs)      # lowest initial state in the stage
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.4, 4.9))

    handles: dict[str, Line2D] = {}
    for run in runs:
        colour = colours[run.key_site]
        faint  = run.label not in keep
        style  = dict(color='0.8', lw=0.6, alpha=0.5, zorder=1) if faint else \
                 dict(color=colour, lw=1.5, zorder=4)

        ax_a.plot(run.frac, run.E_abs - E_ref, '-', **style)
        ax_b.plot(run.frac, run.dE, '-', **style)
        if not faint:
            _mark_barriers(ax_a, run, run.E_abs - E_ref, colour)
            _mark_barriers(ax_b, run, run.dE, colour)
            handles.setdefault(run.key_site,
                               Line2D([], [], color=colour, lw=1.5, marker='o',
                                      ms=4))

    for ax in (ax_a, ax_b):
        ax.axhline(0.0, color='k', lw=0.8, ls='--', zorder=2)
        ax.set_xlabel('Reaction coordinate')
        ax.grid(True, alpha=0.3)

    ax_a.set_ylabel('E − E(lowest initial state)  (eV)')
    ax_a.set_title('(a) common zero — initial-state differences preserved',
                   fontsize=9.5)
    ax_b.set_ylabel('ΔE from own initial state  (eV)')
    ax_b.set_title('(b) each curve zeroed on its own IS — barriers comparable',
                   fontsize=9.5)

    # Panel (a) is a constant offset from formation energy within a stage, so a
    # second axis gives it a physical zero for free.
    e_clean, e_h2 = refs
    n_H = runs[0].n_H if runs[0].n_H is not None else STAGE_NH.get(stage)
    if e_clean is not None and e_h2 is not None and n_H:
        shift = formation_energy(E_ref, n_H, e_clean, e_h2)
        ax2 = ax_a.twinx()
        lo, hi = ax_a.get_ylim()
        ax2.set_ylim(lo + shift, hi + shift)
        ax2.set_ylabel(f'$E_{{form}}$ ({n_H} H, rel. H₂ gas + clean slab)  (eV)',
                       fontsize=8.5)

    order = sorted(handles, key=_site_order)
    rows  = (len(order) + 5) // 6
    title = (f'NEB MEP overlay — {pretty_material(material)}, {stage}  '
             f'({STAGES[stage]}) — {len(runs)} path(s)')
    if crowded:
        title += f', {n_highlight} lowest highlighted'
    if backlog:
        title += f'\n{backlog} further ranked transition(s) not downloaded'

    fig_h    = 4.9 + 0.75 + rows * 0.16
    fig.set_size_inches(11.4, fig_h)
    reserved = min(0.36, (0.5 + rows * 0.16) / fig_h)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, reserved, 1, 0.93))
    fig.legend([handles[s] for s in order], order, title='Site',
               loc='upper center', bbox_to_anchor=(0.5, reserved),
               ncol=min(len(order), 6), fontsize=8, title_fontsize=8.5,
               frameon=True, borderaxespad=0.0)
    fig.text(0.5, 0.008,
             'Circles mark max(ΔE from IS); a red/coloured arrow marks the '
             'largest uphill excursion where the two differ.',
             ha='center', fontsize=7.5, color='0.4')
    _save(fig, outfile)
    plt.close(fig)


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
    if surf.n_H != 2:
        return None
    pair = formation_energy(surf.E_abs[-1], 2, e_clean, e_h2)
    iso  = sum(formation_energy(hopa[s].E_abs[0], 1, e_clean, e_h2)
               for s in sites)
    return float(pair - iso)


def plot_full_pathway(material: str, stages: dict[str, list[NEBRun]],
                      colours: dict[str, tuple], outfile: str,
                      refs: tuple[float | None, float | None] = (None, None)):
    """Chain dissociation → hopa → hopb into one continuous profile."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    chains = build_chains(stages)
    if not chains:
        return None

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    handles, labels = [], []

    # Colour says which H is followed; linestyle says which dissociation event
    # produced it. Two chains can otherwise share a colour — the same H reached
    # from two different surface runs — and become indistinguishable.
    surf_order = sorted({c.surface.label for c in chains})
    dashes = ('-', '--', '-.', ':')

    for chain in chains:
        colour = colours[chain.hop_site]
        style  = dashes[surf_order.index(chain.surface.label) % len(dashes)]
        x, y = chain.profile()
        ax.plot(x, y, style, color=colour, lw=1.6, zorder=4)

        # Mark each stage's own saddle on the chained curve.
        n = 0
        for run, x0 in ((chain.surface, 0.0), (chain.hopa, 1.0),
                        (chain.hopb, 2.0)):
            if run is None:
                continue
            seg = slice(n, n + len(run.frac))
            top = n + int(np.argmax(y[seg]))
            ax.plot(x[top], y[top], 'o', color=colour, ms=5, zorder=6)
            n += len(run.frac)

        handles.append(Line2D([], [], color=colour, lw=1.6, ls=style,
                              marker='o', ms=5))
        labels.append(chain.label)

    ax.axhline(0.0, color='k', lw=0.8, ls='--', zorder=2)
    for xv in (1.0, 2.0):
        ax.axvline(xv, color='0.55', lw=1.0, ls=':', zorder=2)

    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xticklabels(['H₂ + slab', '2H adsorbed', 'sub1', 'sub2'])
    ax.set_xlabel('Reaction coordinate   (dissociation → hopA → hopB)')
    ax.set_ylabel('E relative to H₂ + slab  (eV)')
    ax.set_title(f'H entry pathway for one H — {pretty_material(material)}')
    ax.grid(True, alpha=0.3)

    caption = ('Chained from relative energies; each curve follows ONE of the '
               'two dissociated H atoms.')
    e_clean, e_h2 = refs
    hopa_map = {r.key_site: r for r in stages.get('hopa', [])}
    if e_clean is not None and e_h2 is not None:
        residuals = [r for surf in stages.get('surface', [])
                     if (r := interaction_residual(surf, hopa_map, e_clean,
                                                   e_h2)) is not None]
        if residuals:
            worst = max(residuals, key=abs)
            caption += (f'\nSplicing a 2-H stage onto a 1-H stage assumes '
                        f'H–H non-interaction; measured here at '
                        f'{worst * 1000:+.0f} meV.')

    rows     = len(labels) + 1
    fig_h    = 5.4 + max(0, rows - 4) * 0.19
    fig.set_size_inches(9.0, fig_h)
    reserved = min(0.5, (rows * 0.185 + 0.42) / fig_h)
    fig.tight_layout(rect=(0, reserved, 1, 1))
    fig.legend(handles, labels, title='Dissociation → which H is followed',
               loc='upper center', bbox_to_anchor=(0.5, reserved),
               ncol=1 if len(labels) <= 5 else 2, fontsize=7.5,
               title_fontsize=8.5, frameon=True, borderaxespad=0.0)
    fig.text(0.5, 0.008, caption, ha='center', fontsize=7.5, color='0.4')

    _save(fig, outfile)
    plt.close(fig)
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
                refs=refs) is None:
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

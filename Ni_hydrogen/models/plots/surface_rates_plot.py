#!/usr/bin/env python3
"""
models/plots/surface_rates_plot.py
==================================
Published H2 surface rate constants in Arrhenius form, one figure per material:

  * dissociation   k_d = k_d0 exp(-E_d/RT), with J = k_d P      (literature_dissociation.json)
  * recombination  k_r = k_r0 exp(-E_r/RT), with J = k_r C^2    (literature_recombination.json)

This work does not compute k_d or k_r, so these figures hold the literature only: every law is a
grey dashed line drawn over its own temperature range and named beside it by its first author
and the surface condition it was measured on ("Perujo (oxidized)*"). A law stored without a
temperature range, or as a single value, is listed in the caption but not drawn.

Figures
-------
  * ``<outdir>/<material>_dissociation_literature.{pdf,png}`` and ``_caption.tex``
  * ``<outdir>/<material>_recombination_literature.{pdf,png}`` and ``_caption.tex``

Typical usage
-------------
::

    python models/plots/surface_rates_plot.py
    python models/plots/surface_rates_plot.py --materials '316L SS' 'Inconel 625'
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from diffusivity_plot import (            # noqa: E402
    DEFAULT_RESULTS_DIR, KD_LIT_FILE, KR_LIT_FILE, _first_author, _tex, draw_literature,
    literature_curves, literature_notes, load_dissociation_literature, load_recombination_literature,
    name_literature,
)
import figstyle as fs                     # noqa: E402

MODULE = 'plots/surface_rates_plot.py'
EV_TO_KJ = 96.485332                      # kJ/mol per eV

#: per quantity: the loader, its file, the pre-factor and energy fields, and the words for figure and caption
QUANTITIES = {
    'dissociation': dict(
        load=load_dissociation_literature, path=KD_LIT_FILE, pre_key='kd0', e_key='Ed',
        ylabel=r'$k_d$ (mol H$_2$ m$^{-2}$ s$^{-1}$ Pa$^{-1}$)',
        pre=r'$k_{d0}$', E=r'$E_d$', unit=r'mol~H$_2$~m$^{-2}$~s$^{-1}$~Pa$^{-1}$',
        what=r'H$_2$ dissociation rate constant, $k_d = k_{d0}\exp(-E_d/RT)$ with flux $J = k_d P$,'),
    'recombination': dict(
        load=load_recombination_literature, path=KR_LIT_FILE, pre_key='kr0', e_key='Er',
        ylabel=r'$k_r$ (m$^4$ mol$^{-1}$ s$^{-1}$)',
        pre=r'$k_{r0}$', E=r'$E_r$', unit=r'm$^4$~mol$^{-1}$~s$^{-1}$',
        what=r'H recombination rate constant, $k_r = k_{r0}\exp(-E_r/RT)$ with flux $J = k_r C^2$,'),
}

#: a part of a label's brackets that names where it was found ('INL 2020', 'via Tanabe'), not the surface
_SOURCE_TAG = re.compile(r'(19|20)\d\d|^via\b', re.I)


def surface_name(label: str) -> str:
    """'Perujo et al., 1996 (oxidized; INL 2020)*' -> 'Perujo (oxidized)*'; 'Guo et al., 2025†' ->
    'Guo†'. A surface rate depends on the surface as much as on the lab, so the condition stays."""
    name = _first_author(label)
    mark = name[-1] if name.endswith(('†', '*')) else ''
    name = name[:-1] if mark else name
    inside = re.search(r'\(([^)]*)\)', label)
    parts = [p.strip() for p in inside.group(1).split(';')] if inside else []
    condition = ', '.join(p for p in parts if p and not _SOURCE_TAG.search(p))
    return f'{name} ({condition}){mark}' if condition else name + mark


def stored_only(path: str, pre_key: str) -> dict[str, list[str]]:
    """Per material, the labels of laws kept in the file but not drawn: no temperature range, or a
    single value with no law. Read from the raw JSON, because the loader drops them."""
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    out: dict[str, list[str]] = {}
    for material, entries in raw.items():
        if material.startswith('_') or not isinstance(entries, list):
            continue
        for e in entries:
            no_range = e.get('T_C') is None and e.get('T_K') is None
            if no_range or e.get(pre_key) is None:
                out.setdefault(material, []).append(str(e.get('label', '')).strip())
    return out


def axis_temperatures(T) -> np.ndarray:
    """The temperatures the x-axis spans: the laws' own, widened to the nearest 100 K when fewer than
    two round hundreds fall inside them (the top axis labels only those). The lines keep their ranges."""
    T = np.asarray(T, float)
    lo, hi = T.min(), T.max()
    if np.floor(hi / 100) - np.ceil(lo / 100) + 1 >= 2:
        return T
    return np.append(T, [np.floor(lo / 100) * 100, np.ceil(hi / 100) * 100])


def _kj(e_ev: float) -> str:
    return f'{e_ev * EV_TO_KJ:.4g}'


def plot_material(material: str, entries: list, not_drawn: list[str], quantity: str, outdir: str):
    """One material's published laws for one quantity: grey dashed lines over their own ranges,
    named beside them when there are few enough, the full list in the caption."""
    q = QUANTITIES[quantity]
    curves = literature_curves(entries)
    fig, ax = fs.new_figure()
    draw_literature(ax, curves)
    fs.inverse_temperature_axis(ax, axis_temperatures(np.concatenate([T for _e, T, _y in curves])), min_gap=0.1)
    fs.set_log_y(ax, np.concatenate([y for _e, _T, y in curves]), pad_decades=0.35)
    ax.set_ylabel(q['ylabel'])
    named = name_literature(fig, ax, curves, namer=surface_name)
    laws = '; '.join(f'{_tex(e.label)}: {q["pre"]} = {fs.tex_num(e.pre, 3)}~{q["unit"]}, '
                     f'{q["E"]} = {_kj(e.E_eV)}~kJ~mol$^{{-1}}$, {e.T_K[0]:.0f}--{e.T_K[1]:.0f}~K'
                     for e in entries)
    n = len(entries)
    text = (rf'Published {q["what"]} for {_tex(material)}: {n} law{"s" if n != 1 else ""} (grey dashed lines, '
            rf'each drawn only over its own temperature range'
            rf'{", named by first author and surface condition" if named else ""}): {laws}.'
            rf' This work does not compute this rate constant; the figure collects the published values.'
            + literature_notes(entries, [], 10.0)
            + (f' Stored without a temperature range or a law, so not drawn: {fs.tex_join([_tex(s) for s in not_drawn])}.'
               if not_drawn else ''))
    stem = os.path.join(outdir, f'{material.replace(" ", "_")}_{quantity}_literature')
    fs.finish(fig, ax, stem, MODULE, text)
    if not named:
        print(f'  · {material}: names left to the caption (more than the plot can hold, or no free spot)')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Plot published H2 dissociation and recombination rate constants.')
    ap.add_argument('--outdir', default=os.path.join(DEFAULT_RESULTS_DIR, 'plots'),
                    help='where figures go (default: calculation/results/plots)')
    ap.add_argument('--materials', nargs='*', default=None,
                    help="material keys as in the JSON files, e.g. '316L SS' (default: every one with a drawable law)")
    ap.add_argument('--quantity', choices=sorted(QUANTITIES), nargs='*', default=sorted(QUANTITIES),
                    help='which rate constant(s) to plot (default: both)')
    args = ap.parse_args(argv)
    outdir = os.path.abspath(args.outdir)

    wrote = 0
    for quantity in args.quantity:
        q = QUANTITIES[quantity]
        print(f'\nLoading {os.path.basename(q["path"])} …')
        lit = q['load'](q['path'])
        stored = stored_only(q['path'], q['pre_key'])
        print(f'\n{quantity.capitalize()} plots:')
        for material in args.materials or sorted(set(lit) | set(stored)):
            entries = lit.get(material, [])
            if not entries:
                why = 'no law with a temperature range' if material in stored else 'no entry'
                print(f'  · {material}: {why}, so no figure')
                continue
            plot_material(material, entries, stored.get(material, []), quantity, outdir)
            wrote += 1
    print(f'\n{wrote} figure(s) written to {outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

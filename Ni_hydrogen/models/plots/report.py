#!/usr/bin/env python3
"""
report.py — one PDF per material, built from its export folder.

Nothing is computed and nothing touches the cluster: the report reads the CSV
tables, figures, manifest.json and notes.json that export_results.py wrote to
``calculation/results/export/<stem>/``, so it can be rebuilt at any time from a
finished or partly finished material. Every part with exported results gets a
section: its goal, where in docs/ to read more, what was computed, key numbers,
its tables, its figures, and how to read them. The fixed text is in
report_text.py; the LaTeX building blocks are in latex_report.py.

postprocess.py runs it last, after the export; or run it yourself:

    python models/plots/report.py --stem Ni_supercell
    python models/plots/report.py --stem Ni_supercell --tex-only     # no LaTeX here
    python models/plots/report.py --export-dir ~/thesis/ni

The PDF lands in ``<export folder>/report/<stem>_report.pdf``. Without LaTeX
(latexmk + pdflatex) the .tex is kept and the log says how to build it.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import latex_report as L          # noqa: E402
import report_text as RT          # noqa: E402

DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'calculation', 'results')

SECTION_ORDER = ['dissociation', 'subsurface', 'rates', 'solubility', 'diffusivity', 'permeability']

# The files of each section (glob patterns in the export folder): a section is
# written for every one with at least one of them.
SECTION_OUTPUTS = {
    'dissociation': ['part1_dissociation.csv', 'figures/material/*neb_mep_overlay_surface.png'],
    'subsurface': ['part2_hop_barriers.csv', 'figures/material/*neb_mep_overlay_hop[ab].png',
                   'figures/material/*neb_mep_full_pathway.png'],
    'rates': ['part2_tst_rates.csv', 'figures/material/*site_barriers.png', 'figures/material/*site_rates.png'],
    'solubility': ['part2_solubility_by_env.csv', 'figures/material/*env_dH_sol.png',
                   'figures/material/*solubility_arrhenius.png'],
    'diffusivity': ['part3_diffusivity_fits.csv', 'part3_diffusivity_vs_T.csv', 'figures/*H/msd_vs_time.png',
                    'figures/*H/diffusivity_vs_invT.png', 'figures/*H/arrhenius.png',
                    'figures/material/*msd_all_concentrations.png', 'figures/material/*diffusivity_all_concentrations.png',
                    'figures/material/*arrhenius_params_vs_concentration.png',
                    'figures/material/*diffusivity_vs_literature*.png', 'figures/material/*thermal_expansion.png'],
    'permeability': ['overall_permeability.csv', 'figures/material/*permeability_arrhenius.png',
                     'figures/*H/permeation_summary.png'],
}
ROUTE_ORDER = {'geometric': 0, 'vibrational': 1, 'detailed_balance': 2}
PREFACTOR_RANGE = (1e12, 1e14)        # s-1, the sound range of docs/03-stages/08 "Checks and failure modes"
BARRIERLESS_EV = 1e-3                 # a band maximum this close to its start: docs/03-stages/06 "Checks and failure modes"


# ── values ───────────────────────────────────────────────────────────────────
def num(x):
    """A CSV cell as a float, or None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def flag(x):
    """A CSV cell as True/False, or None."""
    if isinstance(x, bool):
        return x
    t = str(x).strip().lower() if x is not None else ''
    return True if t == 'true' else False if t == 'false' else None


def ev(v):
    return L.tex_fixed(v, 3)


def ev_pm(v, err):
    return L.tex_pm(v, err, ev)


def pct(v):
    return '--' if v is None else f'${100 * v:.0f}\\,\\%$' if v >= 0.01 else f'${100 * v:.2g}\\,\\%$'


def n_h(v):
    """A loading as '3H'."""
    f = num(v)
    return f'{int(f)}H' if f is not None and f.is_integer() else f'{v}H'


def span(values, fmt=ev):
    vals = [v for v in values if v is not None]
    if not vals:
        return '--'
    lo, hi = min(vals), max(vals)
    return fmt(lo) if lo == hi else f'{fmt(lo)} to {fmt(hi)}'


def labels_list(labels, n=5):
    labels = list(labels)
    shown = ', '.join(L.tex_file(x) for x in labels[:n])
    return shown + (f' and {len(labels) - n} more' if len(labels) > n else '')


# ── the export folder ────────────────────────────────────────────────────────
def parse_summary_md(path):
    """The facts of an export made before manifest.json existed, from its SUMMARY.md."""
    facts, notes = {}, []
    if not os.path.isfile(path):
        return facts, notes
    with open(path, errors='replace') as fh:
        text = fh.read()
    m = re.search(r'Source: `([^`]*)`\s*·\s*stem: `([^`]*)`', text)
    if m:
        facts['source_calculation_dir'], facts['stem'] = m.group(1), m.group(2)
    m = re.search(r'Loadings included: (.*)', text)
    if m:
        facts['loadings'] = [int(x) for x in re.findall(r'(\d+)H', m.group(1))]
    facts['single_h_excluded'] = 'Single-H runs are excluded' in text
    part = text.split('## Read these before quoting any number', 1)
    if len(part) == 2:
        notes = [m.group(1).strip() for m in re.finditer(r'^\d+\.\s+(.*)$', part[1], re.M)]
    facts['exported_at'] = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec='seconds')
    facts['from_summary_md'] = True
    return facts, notes


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def git(*args):
    try:
        return subprocess.run(['git', '-C', _PROJECT_ROOT, *args], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def pipeline_config():
    """The potential and paths in models/config.py of this checkout (loaded on its own, so
    the models package is not imported)."""
    path = os.path.join(_PROJECT_ROOT, 'models', 'config.py')
    try:
        spec = importlib.util.spec_from_file_location('_report_pipeline_config', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:                           # noqa: BLE001 -- a fact, not a requirement
        return {}
    return {k: getattr(mod, k, None) for k in ('MACE_MODEL_ASE', 'MACE_MODEL_LAMMPS', 'MACE_HEAD', 'LAMMPS_CMD')}


class MaterialReport(L.LatexReport):
    """The report of one material's export folder."""

    def __init__(self, export_dir, log=print, docs_root=_PROJECT_ROOT, web_base='auto'):
        super().__init__(os.path.abspath(export_dir), docs_root=docs_root,
                         web_base=L.git_web_base(docs_root) if web_base == 'auto' else web_base, log=log)
        facts, notes = parse_summary_md(os.path.join(self.out, 'SUMMARY.md'))
        manifest = load_json(os.path.join(self.out, 'manifest.json'))
        if isinstance(manifest, dict):
            facts = {**facts, **manifest, 'from_summary_md': False}
        js = load_json(os.path.join(self.out, 'notes.json'))
        if isinstance(js, list):
            notes = [str(n) for n in js]
        self.facts, self.notes = facts, notes
        self.stem = facts.get('stem') or os.path.basename(self.out.rstrip('/'))
        self.material = facts.get('material') or self.stem.replace('_supercell', '').replace('_', ' ')
        self.built = []

    def rows(self, name):
        path = os.path.join(self.out, name)
        if not os.path.isfile(path):
            return []
        self.used.add(name)
        with open(path, newline='') as fh:
            return list(csv.DictReader(fh))

    def find(self, pattern):
        return sorted(os.path.relpath(p, self.out) for p in glob.glob(os.path.join(self.out, pattern)))

    def material_fig(self, suffix):
        found = [f for f in self.find('figures/material/*' + suffix)
                 if not os.path.basename(f).endswith('_all_materials.png')]
        return found[0] if found else None

    def loading_figs(self, name):
        """[(loading label, file)] of one per-loading figure, by loading."""
        found = []
        for f in self.find(f'figures/*H/{name}'):
            sub = f.split(os.sep)[1]
            m = re.match(r'0*(\d+)H$', sub)
            found.append((int(m.group(1)) if m else 0, sub, f))
        return [(sub, f) for _, sub, f in sorted(found)]

    def outputs(self, name):
        found = set()
        for pattern in SECTION_OUTPUTS.get(name, []):
            found.update(self.find(pattern))
        return sorted(found)


# ── title page and run summary ───────────────────────────────────────────────
def title_page(rep):
    f = rep.facts
    src = f.get('source_calculation_dir')
    commit = git('rev-parse', '--short', 'HEAD')
    dirty = bool(git('status', '--porcelain', '--', '.'))
    cfg = pipeline_config()
    facts = [('material', L.tex_escape(rep.material)), ('stem', L.tex_file(rep.stem)),
             ('export folder', L.tex_path(rep.out))]
    if src:
        facts.append(('pipeline results', L.tex_path(src) + ('' if os.path.isdir(src) else ' (not on this machine)')))
    if f.get('exported_at'):
        facts.append(('exported', L.tex_escape(f['exported_at'].replace('T', ' '))))
    loads = f.get('loadings')
    if loads is not None:
        facts.append(('loadings', ', '.join(f'{n}H' for n in loads) or 'none'))
    if cfg.get('MACE_HEAD') or cfg.get('MACE_MODEL_ASE'):
        model = os.path.basename(str(cfg.get('MACE_MODEL_ASE') or ''))
        facts.append(('potential (this checkout)', L.tex_file(model) + (f', head {L.tex_file(cfg["MACE_HEAD"])}'
                                                                        if cfg.get('MACE_HEAD') else '')))
    facts += [('report built', datetime.datetime.now().strftime('%Y-%m-%d %H:%M')),
              ('code', ('commit ' + L.tex_escape(commit) + (' (uncommitted changes)' if dirty else '')) if commit else 'not a git checkout'),
              ('documentation', 'linked on GitHub at that commit' if rep.web_base else r'\texttt{docs/} in the project folder')]
    title = L.tex_escape(rep.material) + ': hydrogen permeation report'
    footer = (r'Built from the export folder; nothing was computed to make it. Rebuild with '
              r'\texttt{python models/plots/report.py --stem ' + L.tex_escape(rep.stem) + r'}.')
    return title, L.title_page(title, r'stem \texttt{' + L.tex_escape(rep.stem) + '}', facts, footer)


def generated_scripts(rep):
    """The generated run scripts of this material in the pipeline's calculation folder, and
    whether each is older than the last commit to models/ (it then holds older code)."""
    src = rep.facts.get('source_calculation_dir')
    if not src or not os.path.isdir(src):
        return None
    last = num(git('log', '-1', '--format=%ct', '--', 'models'))
    rows = []
    for p in sorted(set(glob.glob(os.path.join(src, f'*_run_{rep.stem}.py'))
                        + glob.glob(os.path.join(src, f'*_run_{rep.stem}_*.py')))):
        t = os.path.getmtime(p)
        rows.append({'script': os.path.basename(p),
                     'modified': datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M'),
                     'older': None if last is None else t < last})
    return rows


def run_summary(rep):
    f = rep.facts
    out = [r'\section{Run summary}', r'\label{sec:summary}',
           r'\paragraph{Where to read more.}', rep.pointers_block('Run summary', RT.SUMMARY['docs'])]

    # what was run
    fits = {num(r['n_H']): r for r in rep.rows('part3_diffusivity_fits.csv')}
    perm = {}
    for r in rep.rows('overall_permeability.csv'):
        perm.setdefault(num(r['n_H']), []).append(r['route'])
    loads = sorted(set(fits) | set(perm) | {float(n) for n in (f.get('loadings') or [])})
    grid = [{'load': n_h(n), 'D': 'yes' if n in fits else 'no',
             'pts': num(fits[n].get('n_positive_D_points')) if n in fits else None,
             'routes': ', '.join(sorted(perm.get(n, []), key=lambda x: ROUTE_ORDER.get(x, 9))) or 'none'} for n in loads]
    p1, hops = rep.rows('part1_dissociation.csv'), rep.rows('part2_hop_barriers.csv')
    env, tst = rep.rows('part2_solubility_by_env.csv'), rep.rows('part2_tst_rates.csv')
    facts = [f'Part 1: {len(p1)} surface dissociation pathways.' if p1 else 'Part 1: no dissociation pathways exported.',
             (f"Part 2: {sum(r['hop'] == 'hopa' for r in hops)} Hop A and {sum(r['hop'] == 'hopb' for r in hops)} "
              f"Hop B pathways; {len(env)} subsurface environments; rate constants at "
              + (', '.join(f'{t:g}' for t in sorted({num(r['T_K']) for r in tst if num(r['T_K']) is not None})) or 'no')
              + ' K.') if hops or env or tst else 'Part 2: nothing exported.',
             (f'Part 3 and permeability, by loading: see the table.' if grid else 'Part 3: nothing exported.')]
    if f.get('single_h_excluded'):
        facts.append(r'Single-H runs are left out (one walker, no ensemble average); '
                     r'\texttt{export\_results.py --include-1h} keeps them.')
    out += [r'\subsection{What was run}', L.bullets(facts)]
    if grid:
        out.append(rep.table(grid, [('load', 'loading', 'l', L.tex_escape), ('D', r'diffusivity\\fit', 'l', None),
                                    ('pts', r'$D$ points\\in the fit', 'r', None),
                                    ('routes', r'permeability\\routes', 'l', L.tex_escape)],
                             'What each loading has. A loading without a diffusivity fit has no permeability: no '
                             'placeholder is ever substituted.'))

    # caveats
    out += [r'\subsection{Read these before quoting any number}',
            (L.enumerated([L.tex_md(n) for n in rep.notes]) if rep.notes else 'The export recorded no caveats.')
            + ('' if not f.get('from_summary_md') else
               r'\par\emph{From SUMMARY.md: this export predates manifest.json and notes.json; re-running '
               r'\texttt{export\_results.py} writes them.}')]

    # generated scripts
    scripts = generated_scripts(rep)
    out.append(r'\subsection{Generated run scripts}')
    msg = (r'The pipeline runs scripts \emph{generated} from \texttt{models/}: editing \texttt{models/} changes nothing '
           r'until the scripts are regenerated (README, ``How to approach this codebase''). ')
    if scripts is None:
        out.append(msg + r'The pipeline\textquotesingle s calculation folder is not on this machine, so the scripts that '
                   r'produced these results could not be checked.')
    elif not scripts:
        out.append(msg + 'No generated script of this material was found in the calculation folder.')
    else:
        dirty = bool(git('status', '--porcelain', '--', 'models'))
        out.append(msg + 'A script older than the last commit to \\texttt{models/} may hold older code; regenerate it '
                   'with \\texttt{calculation/tools/regenerate\\_*\\_scripts.py}.'
                   + (r' \textbf{\texttt{models/} has uncommitted changes}, which no generated script holds yet.' if dirty else ''))
        out.append(rep.table(scripts, [('script', 'script', 'l', L.tex_file), ('modified', 'modified', 'l', L.tex_escape),
                                       ('older', r'older than the last\\commit to models/', 'l', L.tex_flag)],
                             'Generated run scripts of this material.'))

    # sections
    rows = [{'name': RT.SECTIONS[s]['title'], 'n': len(rep.outputs(s)),
             'ref': (r'\ref{sec:' + s + '}') if s in rep.built else '--'} for s in SECTION_ORDER]
    out += [r'\subsection{Sections}',
            rep.table(rows, [('name', 'part', 'l', lambda s: s), ('n', r'files\\found', 'r', None),
                             ('ref', 'section', 'l', lambda s: s)],
                      'The parts of the workflow, in the order of the stages in \\texttt{docs/03-stages/}. A section is '
                      'shown for each part with exported files.')]
    out += [r'\subsection{Conventions used in this report}', RT.CONVENTIONS]
    return '\n\n'.join(out)


# ── sections ─────────────────────────────────────────────────────────────────
def section_dissociation(rep):
    rows = rep.rows('part1_dissociation.csv')
    parts = []
    if rows:
        rows = sorted(rows, key=lambda r: (num(r['Ea_eV']) is None, num(r['Ea_eV']) or 0))
        conv = [r for r in rows if flag(r['neb_converged']) is True]
        sound = [r for r in rows if flag(r['is_minimum']) and flag(r['ts_saddle'])]
        best = (conv or rows)[0]
        zpe = Counter(r['zpe_source'] or 'none' for r in rows)
        keys = [f'{len(rows)} pathways; {len(conv)} bands converged'
                + (f', {len(rows) - len(conv)} did not (their barriers are provisional).' if len(conv) < len(rows) else '.'),
                f"Lowest {'converged ' if conv else ''}barrier: {L.tex_file(best['label'])}, $E_a$ = {ev(num(best['Ea_eV']))} eV "
                f"({ev(num(best['Ea_zpe_eV']))} eV with the zero-point correction). Converged barriers span "
                f"{span(num(r['Ea_eV']) for r in conv)} eV.",
                f"Reaction energy $\\Delta E$ {span(num(r['delta_E_eV']) for r in rows)} eV; desorption barrier "
                f"$E_\\mathrm{{des}}$ {span(num(r['E_des_eV']) for r in rows)} eV.",
                f'{len(sound)} of {len(rows)} pathways have both a relaxed initial state and a resolved saddle, so only '
                f'{"those carry" if len(sound) != 1 else "that one carries"} a trustworthy prefactor'
                + (f': {labels_list(r["label"] for r in sound)}.' if sound else '.'),
                'Zero-point source: ' + ', '.join(f'{L.tex_file(k)} for {v}' for k, v in zpe.most_common())
                + ('; a fallback makes the reverse quantities provisional.' if any(k != 'FS' for k in zpe) else '.')]
        flat = [r for r in rows if num(r['Ea_eV']) is not None and num(r['Ea_eV']) <= BARRIERLESS_EV]
        if flat:
            keys.insert(2, f'{len(flat)} pathway{"s have" if len(flat) != 1 else " has"} no barrier ($E_a \\le$ '
                        f'{BARRIERLESS_EV:g} eV): {labels_list(r["label"] for r in flat)}. The saddle sits at an endpoint, so '
                        'the reaction is barrierless as posed or the endpoints are wrong; check that the final state is a '
                        'distinct minimum (Stage 6, ``Checks and failure modes\'\').')
        parts += [('What was computed', f'A nudged elastic band for each of the {len(rows)} enumerated dissociation '
                   'pathways of H$_2$ on the relaxed slab, then vibrations at its initial, transition and final states. '
                   'The table is sorted by barrier.'),
                  ('Key numbers', L.bullets(keys)),
                  ('All pathways', rep.table(rows, [
                      ('label', 'pathway', 'l', L.tex_escape), ('Ea_eV', r'$E_a$\\(eV)', 'r', ev),
                      ('Ea_zpe_eV', r'$E_{a,\mathrm{ZPE}}$\\(eV)', 'r', ev), ('E_des_eV', r'$E_\mathrm{des}$\\(eV)', 'r', ev),
                      ('delta_E_eV', r'$\Delta E$\\(eV)', 'r', ev), ('neb_converged', 'conv.', 'l', L.tex_flag),
                      ('fmax_final', r'$f_\mathrm{max}$\\(eV/\AA)', 'r', None), ('nu_fwd_s-1', r'$\nu_\mathrm{fwd}$\\(s$^{-1}$)', 'r', None),
                      ('nu_rev_s-1', r'$\nu_\mathrm{rev}$\\(s$^{-1}$)', 'r', None), ('is_minimum', r'IS\\min.', 'l', L.tex_flag),
                      ('ts_saddle', r'TS\\saddle', 'l', L.tex_flag), ('zpe_source', r'ZPE\\source', 'l', L.tex_escape)],
                      'Surface dissociation pathways, lowest barrier first.', 'part1_dissociation.csv'))]
    parts.append(('figure', 'mep_overlay_surface', rep.material_fig('neb_mep_overlay_surface.png')))
    if rep.find('figures/pathways/dissociation/*/*.png'):
        parts.append((None, r'The band of each pathway on its own is in Appendix~\ref{app:pathways}.'))
    return rep.section('dissociation', RT.SECTIONS['dissociation'], parts)


def section_subsurface(rep):
    rows = rep.rows('part2_hop_barriers.csv')
    parts = []
    if rows:
        rows = sorted(rows, key=lambda r: (r['hop'], num(r['Ea_zpe_eV']) is None, num(r['Ea_zpe_eV']) or 0))
        keys = []
        for hop, name in (('hopa', 'Hop A'), ('hopb', 'Hop B')):
            hr = [r for r in rows if r['hop'] == hop]
            if not hr:
                keys.append(f'{name}: no pathways exported.')
                continue
            envs = Counter(r['env'] or 'unlabelled' for r in hr)
            b = hr[0]
            keys.append(f"{name}: {len(hr)} pathway{'s' if len(hr) != 1 else ''} ("
                        + ', '.join(f'{v} into {L.tex_escape(k)}' for k, v in envs.most_common())
                        + f"); lowest $E_{{a,\\mathrm{{ZPE}}}}$ = {ev(num(b['Ea_zpe_eV']))} eV ({L.tex_file(b['label'])}, "
                        f"{L.tex_escape(b['env'] or 'unlabelled')}); forward barriers span {span(num(r['Ea_zpe_eV']) for r in hr)} eV, "
                        f"reverse {span(num(r['Ed_zpe_eV']) for r in hr)} eV.")
        fb = [r for r in rows if (r['zpe_source'] or '') != 'FS']
        if fb:
            keys.append(f'{len(fb)} pathway(s) use a fallback zero-point source; their reverse quantities are provisional: '
                        f'{labels_list(r["label"] for r in fb)}.')
        parts += [('What was computed', 'A nudged elastic band for each entry hop (surface to first subsurface site, Hop A) '
                   'and each deeper hop (first to second, Hop B), then vibrations at both ends and the saddle.'),
                  ('Key numbers', L.bullets(keys)),
                  ('All subsurface pathways', rep.table(rows, [
                      ('hop', 'hop', 'l', L.tex_escape), ('label', 'pathway', 'l', L.tex_escape),
                      ('env', 'environment', 'l', L.tex_escape), ('Ea_raw_eV', r'$E_a$\\(eV)', 'r', ev),
                      ('Ea_zpe_eV', r'$E_{a,\mathrm{ZPE}}$\\(eV)', 'r', ev), ('Ed_raw_eV', r'$E_d$\\(eV)', 'r', ev),
                      ('Ed_zpe_eV', r'$E_{d,\mathrm{ZPE}}$\\(eV)', 'r', ev), ('nu_fwd_s-1', r'$\nu_\mathrm{fwd}$\\(s$^{-1}$)', 'r', None),
                      ('nu_rev_s-1', r'$\nu_\mathrm{rev}$\\(s$^{-1}$)', 'r', None), ('zpe_source', r'ZPE\\source', 'l', L.tex_escape)],
                      'Hop A and Hop B pathways, lowest corrected barrier first within each hop.', 'part2_hop_barriers.csv',
                      group_by='hop'))]
    parts += [('figure', 'mep_full_pathway', rep.material_fig('neb_mep_full_pathway.png')),
              ('figure', 'mep_overlay_hopa', rep.material_fig('neb_mep_overlay_hopa.png')),
              ('figure', 'mep_overlay_hopb', rep.material_fig('neb_mep_overlay_hopb.png'))]
    return rep.section('subsurface', RT.SECTIONS['subsurface'], parts)


def section_rates(rep):
    rows = rep.rows('part2_tst_rates.csv')
    parts = []
    if rows:
        Ts = sorted({num(r['T_K']) for r in rows if num(r['T_K']) is not None})
        sites = list(dict.fromkeys(r['site'] for r in sorted(rows, key=lambda r: r['site'])))
        per_site = {r['site']: r for r in rows}
        lo, hi = PREFACTOR_RANGE
        odd = [s for s, r in per_site.items() if any(v is not None and not lo <= v <= hi
                                                     for v in (num(r['nu_fwd_s-1']), num(r['nu_rev_s-1'])))]
        keys = [f'{len(sites)} pathways $\\times$ {len(Ts)} temperatures ({", ".join(f"{t:g}" for t in Ts)} K).']
        for T in (Ts[0], Ts[-1]) if len(Ts) > 1 else Ts[:1]:
            a = [r for r in rows if num(r['T_K']) == T and r['site'].startswith('hopa') and num(r['k_forward_s-1']) is not None]
            if a:
                fast = max(a, key=lambda r: num(r['k_forward_s-1']))
                keys.append(f"At {T:g} K the fastest Hop A entry is {L.tex_file(fast['site'])}, $k_\\mathrm{{fwd}}$ = "
                            f"{L.tex_num(num(fast['k_forward_s-1']))} s$^{{-1}}$ (back out: {L.tex_num(num(fast['k_reverse_s-1']))} s$^{{-1}}$).")
        keys.append(f'{len(odd)} of {len(per_site)} pathways have a prefactor outside '
                    f'$10^{{12}}$--$10^{{14}}$\\,s$^{{-1}}$' + (f': {labels_list(odd)}; see the checks in Stage 8.' if odd else '.'))
        site_order = {s: i for i, s in enumerate(sites)}
        rows = sorted(rows, key=lambda r: (site_order[r['site']], num(r['T_K']) or 0))
        head = lambda T: f'{float(T):g} K'          # noqa: E731
        parts += [('What was computed', 'Forward and reverse rate constants of each hop at each temperature, from its '
                   'zero-point-corrected barriers and attempt frequencies (harmonic transition-state theory).'),
                  ('Key numbers', L.bullets(keys)),
                  ('Forward rate constants', rep.pivot(rows, 'site', 'T_K', 'k_forward_s-1',
                                                       r'Forward rate constant $k_\mathrm{fwd}$ (s$^{-1}$): one row per pathway, one '
                                                       'column per temperature.', 'pathway', 'part2_tst_rates.csv', col_head=head)),
                  ('Reverse rate constants', rep.pivot(rows, 'site', 'T_K', 'k_reverse_s-1',
                                                       r'Reverse rate constant $k_\mathrm{rev}$ (s$^{-1}$).', 'pathway',
                                                       'part2_tst_rates.csv', col_head=head))]
    parts += [('figure', 'site_barriers', rep.material_fig('site_barriers.png')),
              ('figure', 'site_rates', rep.material_fig('site_rates.png'))]
    return rep.section('rates', RT.SECTIONS['rates'], parts)


def section_solubility(rep):
    rows = rep.rows('part2_solubility_by_env.csv')
    parts = []
    if rows:
        rows = sorted(rows, key=lambda r: (num(r['dH_sol_eV']) is None, num(r['dH_sol_eV']) or 0))
        low = rows[0]
        big = max(rows, key=lambda r: num(r['weight']) or 0)
        keys = [f"{len(rows)} environment{'s' if len(rows) != 1 else ''}, "
                f"{int(sum(num(r['n_sites']) or 0 for r in rows))} sites in all.",
                f"Lowest solution enthalpy: {L.tex_escape(low['environment'])}, $\\Delta H_\\mathrm{{sol}}$ = "
                f"{ev_pm(num(low['dH_sol_eV']), num(low['dH_sol_err_eV']))} eV, weight {L.tex_num(num(low['weight']))}."]
        if big is not low:
            keys.append(f"The largest weight is {L.tex_escape(big['environment'])} ({L.tex_num(num(big['weight']))}), "
                        f"$\\Delta H_\\mathrm{{sol}}$ = {ev_pm(num(big['dH_sol_eV']), num(big['dH_sol_err_eV']))} eV: the most "
                        'favourable environment is not the most common one, so the weight and the Boltzmann factor pull '
                        'apart (the solubility figure shows by how much).')
        for r in rows:                  # each enthalpy with its error, in one cell
            r['_sol'] = ev_pm(num(r['dH_sol_eV']), num(r['dH_sol_err_eV']))
            r['_hop'] = ev_pm(num(r['dH_hopA_eV']), num(r['dH_hopA_err_eV']))
        parts += [('What was computed', 'The solution enthalpy of H in each first-layer subsurface environment, from the '
                   'dissociation and entry-hop energies, and the fraction of entry pathways that reach it. $S(T)$ follows '
                   'by route; the routes are carried to the permeability section.'),
                  ('Key numbers', L.bullets(keys)),
                  ('Environments', rep.table(rows, [
                      ('environment', 'environment', 'l', L.tex_escape),
                      ('_sol', r'$\Delta H_\mathrm{sol}$\\(eV)', 'r', lambda s: s),
                      ('_hop', r'$\Delta H_\mathrm{hopA}$\\(eV)', 'r', lambda s: s), ('weight', 'weight', 'r', None),
                      ('n_sites', 'sites', 'r', None)],
                      'Subsurface environments, lowest solution enthalpy first.', 'part2_solubility_by_env.csv'))]
    parts += [('figure', 'env_dH_sol', rep.material_fig('env_dH_sol.png')),
              ('figure', 'solubility_arrhenius', rep.material_fig('solubility_arrhenius.png'))]
    return rep.section('solubility', RT.SECTIONS['solubility'], parts)


def section_diffusivity(rep):
    fits = sorted(rep.rows('part3_diffusivity_fits.csv'), key=lambda r: num(r['n_H']) or 0)
    per_T = sorted(rep.rows('part3_diffusivity_vs_T.csv'), key=lambda r: (num(r['n_H']) or 0, num(r['T_K']) or 0))
    parts = []
    if fits or per_T:
        keys = []
        for r in fits:
            pts = num(r['n_positive_D_points'])
            keys.append(f"{n_h(r['n_H'])} ({L.tex_num(num(r['at_pct_H']))} at.\\,\\%): "
                        f"$D_0$ = {L.tex_pm(num(r['D0_m2s']), num(r['D0_err_m2s']))} m$^2$/s, "
                        f"$E_D$ = {ev_pm(num(r['E_D_eV']), num(r['E_D_err_eV']))} eV, $R^2$ = {L.tex_num(num(r['R2_fit']), 4)}, "
                        f"{int(pts) if pts is not None else '--'} points"
                        + (' --- too few for the fit to be evidence.' if pts is not None and pts < 3 else '.'))
        good = [r for r in fits if num(r['D0_m2s']) and num(r['E_D_eV']) is not None]
        if len(good) >= 3:
            by_d0 = [r['n_H'] for r in sorted(good, key=lambda r: num(r['D0_m2s']))]
            by_e = [r['n_H'] for r in sorted(good, key=lambda r: num(r['E_D_eV']))]
            if by_d0 == by_e:
                keys.append(r'Across loadings $D_0$ and $E_D$ rise and fall together: compensation, not faster or slower '
                            r'transport. Compare $D$ itself (table below).')
        Ts = sorted({num(r['T_K']) for r in per_T if num(r['T_K']) is not None})
        if Ts:
            top = [r for r in per_T if num(r['T_K']) == Ts[-1]]
            keys.append(f'At {Ts[-1]:g} K, $D$ = {span((num(r["D_m2s"]) for r in top), L.tex_num)} m$^2$/s across the loadings.')
        for r in per_T:
            d, e = num(r['D_m2s']), num(r['D_err_m2s'])
            r['_D'] = L.tex_pm(d, e)
            r['_rel'] = e / d if d and e is not None else None
        parts += [('What was computed', 'Constant-volume molecular dynamics at each temperature and loading; $D$ from the '
                   'slope of the mean-squared displacement over an interior window, then an Arrhenius fit over temperature '
                   'for each loading.'),
                  ('Key numbers', L.bullets(keys)),
                  ('Arrhenius fits', rep.table(fits, [
                      ('n_H', '$n$H', 'r', None), ('at_pct_H', r'at.\,\%', 'r', None),
                      ('D0_m2s', r'$D_0$\\(m$^2$/s)', 'r', None), ('D0_err_m2s', r'$\sigma_{D_0}$\\(m$^2$/s)', 'r', None),
                      ('E_D_eV', r'$E_D$\\(eV)', 'r', ev), ('E_D_err_eV', r'$\sigma_{E_D}$\\(eV)', 'r', ev),
                      ('R2_fit', '$R^2$', 'r', lambda v: L.tex_num(v, 4)), ('n_positive_D_points', 'points', 'r', None)],
                      'Diffusivity fits, one per loading.', 'part3_diffusivity_fits.csv')),
                  ('Diffusivity at each temperature', rep.table(per_T, [
                      ('n_H', '$n$H', 'r', None), ('T_K', '$T$ (K)', 'r', None), ('_D', r'$D \pm \sigma_D$\\(m$^2$/s)', 'r', lambda s: s),
                      ('_rel', r'relative\\error', 'r', pct)],
                      '$D(T)$ for each loading.', 'part3_diffusivity_vs_T.csv', group_by='n_H'))]
    for key, name, per_row in (('msd_vs_time', 'msd_vs_time.png', 1),      # one panel per temperature: full width
                               ('diffusivity_vs_invT', 'diffusivity_vs_invT.png', 3), ('arrhenius', 'arrhenius.png', 3)):
        figs = rep.loading_figs(name)
        if figs:
            parts.append(('grid', key, [f for _, f in figs], [L.tex_escape(s) for s, _ in figs], per_row))
    for key in ('msd_all_concentrations', 'diffusivity_all_concentrations', 'arrhenius_params_vs_concentration',
                'diffusivity_vs_literature', 'thermal_expansion'):
        parts.append(('figure', key, rep.material_fig(key + '.png')))
    return rep.section('diffusivity', RT.SECTIONS['diffusivity'], parts)


def section_permeability(rep):
    rows = sorted(rep.rows('overall_permeability.csv'),
                  key=lambda r: (num(r['n_H']) or 0, ROUTE_ORDER.get(r['route'], 9)))
    parts = []
    if rows:
        keys = []
        for n in sorted({num(r['n_H']) for r in rows}):
            head = [r for r in rows if num(r['n_H']) == n and r['status'] == 'headline']
            if head:
                keys.append(f'{n_h(n)}: ' + '; '.join(
                    f"{L.tex_escape(r['route'])} $\\Phi_0$ = {L.tex_num(num(r['Phi0_molH_m-1_s-1_Pa-0.5']))} "
                    f"($\\pm${pct(num(r['Phi0_rel_err']))}), $E_\\Phi$ = {ev_pm(num(r['E_phi_eV']), num(r['E_phi_err_eV']))} eV"
                    for r in head) + '.')
            g = next((num(r['Phi0_molH_m-1_s-1_Pa-0.5']) for r in head if r['route'] == 'geometric'), None)
            v = next((num(r['Phi0_molH_m-1_s-1_Pa-0.5']) for r in head if r['route'] == 'vibrational'), None)
            if g and v:
                keys[-1] += f' The geometric $\\Phi_0$ is {L.tex_num(g / v)} times the vibrational: quote both, as a bracket.'
        for route in ('geometric', 'vibrational'):
            es = [num(r['E_phi_eV']) for r in rows if r['route'] == route]
            if len(es) > 1:
                keys.append(f'Spread over loadings, {route}: $E_\\Phi$ = {span(es)} eV. The loadings are different '
                            'systems; quote the spread with the chosen one.')
        if any(r['status'] != 'headline' for r in rows):
            keys.append(r'\texttt{detailed\_balance} rows are a diagnostic cross-check, not a result.')
        for r in rows:
            r['_E'] = ev_pm(num(r['E_phi_eV']), num(r['E_phi_err_eV']))
        parts += [('What was computed', r'$\Phi = D\,S$ at each temperature for each loading and route, written as '
                   r'$\Phi_0$ and $E_\Phi$: $E_\Phi = E_D + \Delta H_\mathrm{sol}$, computed, not fitted.'),
                  ('Key numbers', L.bullets(keys)),
                  ('Permeability', rep.table(rows, [
                      ('n_H', '$n$H', 'r', None), ('route', 'route', 'l', L.tex_escape), ('status', 'status', 'l', L.tex_escape),
                      ('Phi0_molH_m-1_s-1_Pa-0.5', r'$\Phi_0$ (mol\,H\\m$^{-1}$\,s$^{-1}$\,Pa$^{-1/2}$)', 'r', None),
                      ('Phi0_rel_err', r'$\Phi_0$\\rel.\ error', 'r', lambda v: pct(num(v))),
                      ('_E', r'$E_\Phi$\\(eV)', 'r', lambda s: s), ('r2_S', r'$R^2_S$', 'r', lambda v: L.tex_num(v, 4))],
                      'Permeability per loading and route; headline routes first.', 'overall_permeability.csv', group_by='n_H'))]
    parts.append(('figure', 'permeability_arrhenius', rep.material_fig('permeability_arrhenius.png')))
    figs = rep.loading_figs('permeation_summary.png')
    if figs:
        parts.append(('grid', 'permeation_summary', [f for _, f in figs], [L.tex_escape(s) for s, _ in figs], 1))
    return rep.section('permeability', RT.SECTIONS['permeability'], parts)


SECTION_BUILDERS = {
    'dissociation': section_dissociation, 'subsurface': section_subsurface, 'rates': section_rates,
    'solubility': section_solubility, 'diffusivity': section_diffusivity, 'permeability': section_permeability,
}


# ── appendices ───────────────────────────────────────────────────────────────
def appendix_pathways(rep, per_figure=9):
    t = RT.PATHWAYS
    out = []
    for fam, fam_name in t['families'].items():
        blocks = []
        for verdict, v_name in t['verdicts'].items():
            files = rep.find(f'figures/pathways/{fam}/{verdict}/*.png')
            for i in range(0, len(files), per_figure):
                chunk = files[i:i + per_figure]
                caps = [L.tex_escape(os.path.splitext(os.path.basename(f))[0]) for f in chunk]
                part = f' ({i + 1}--{i + len(chunk)} of {len(files)})' if len(files) > per_figure else ''
                tex, _ = rep.figure_grid(chunk, caps, f'{fam_name}: {v_name}{part}. Folder: '
                                         + L.tex_file(f'figures/pathways/{fam}/{verdict}/') + '.')
                blocks.append(tex)
        if blocks:
            out += [r'\subsection{' + fam_name + '}', *blocks]
    if not out:
        return ''
    return '\n\n'.join([r'\section{' + t['title'] + '}', r'\label{app:pathways}',
                        r'\paragraph{Where to read more.}', rep.pointers_block(t['title'], t['docs']),
                        'One panel per pathway, labelled with its folder name. Each should show a single maximum.', *out])


def appendix_files(rep):
    owner = {}
    for s in SECTION_ORDER:
        for f in rep.outputs(s):
            owner[f] = s
    rows = []
    for entry in sorted(os.listdir(rep.out)):
        path = os.path.join(rep.out, entry)
        if entry == 'report' or entry.startswith('.'):
            continue
        if os.path.isdir(path):
            for sub, _, files in sorted(os.walk(path)):
                pngs = [os.path.relpath(os.path.join(sub, f), rep.out) for f in files]
                if pngs:
                    rel = os.path.relpath(sub, rep.out) + '/'
                    secs = sorted({owner[p] for p in pngs if p in owner}, key=SECTION_ORDER.index)
                    if rel.startswith('figures/pathways/'):
                        secs = ['pathways']
                    rows.append({'file': rel, 'kind': 'figures', 'size': f'{len(pngs)} files',
                                 'section': ', '.join(secs), 'shown': f'{sum(p in rep.used for p in pngs)} of {len(pngs)}'})
            continue
        size, kind = '', {'.csv': 'table', '.json': 'facts', '.md': 'summary', '.png': 'figure'}.get(
            os.path.splitext(entry)[1].lower(), 'other')
        if kind == 'table':
            with open(path, newline='') as fh:
                size = f'{max(0, sum(1 for _ in fh) - 1)} rows'
        rows.append({'file': entry, 'kind': kind, 'size': size, 'section': owner.get(entry, ''),
                     'shown': 'yes' if entry in rep.used else 'no'})
    for r in rows:
        r['section'] = ', '.join((r'\ref{sec:' + s + '}') if s in rep.built else
                                 (r'\ref{app:pathways}' if s == 'pathways' else '--')
                                 for s in r['section'].split(', ')) if r['section'] else '--'
    return '\n\n'.join([r'\section{Exported files}', r'\label{app:files}',
                        'Every file of the export folder, ' + L.tex_path(rep.out) + '.',
                        rep.table(rows, [('file', 'file', 'P6.0', L.tex_escape), ('kind', 'kind', 'l', None),
                                         ('size', 'size', 'l', L.tex_escape), ('section', 'section', 'P3.0', lambda s: s),
                                         ('shown', 'shown', 'l', L.tex_escape)], 'Contents of the export folder.', full=True)])


def appendix_docs(rep):
    """Every documentation page this report points to, with the headings and sections."""
    pages = {}
    for title, path, heading in rep.pointers:
        p = pages.setdefault(path, {'headings': [], 'sections': []})
        if heading and heading not in p['headings']:
            p['headings'].append(heading)
        if title not in p['sections']:
            p['sections'].append(title)
    rows = [{'page': rep.pointer_link(path), 'headings': '; '.join(L.tex_heading(h) for h in p['headings']) or '(whole page)',
             'sections': '; '.join(p['sections'])} for path, p in sorted(pages.items())]
    out = [r'\section{Documentation map}', r'\label{app:docs}',
           r'Where each part of this report is explained. Paths are from the project folder'
           + (r'; each links to the page on GitHub at the commit the report was built from.' if rep.web_base else '.'),
           rep.table(rows, [('page', 'page', 'P5.2', lambda s: s), ('headings', 'sections of the page used', 'P6.2', lambda s: s),
                            ('sections', 'pointed to from', 'P4.6', lambda s: s)], 'Documentation pages.', full=True)]
    if rep.broken:
        out.append(r'\textbf{Pointers that did not resolve} (a page or heading was renamed; update report\_text.py):'
                   + L.bullets([L.tex_escape(b) for b in sorted(set(rep.broken))]))
    return '\n\n'.join(out)


# ── build ────────────────────────────────────────────────────────────────────
def write_tex(rep):
    title, title_tex = title_page(rep)
    rep.built = [s for s in SECTION_ORDER if rep.outputs(s)]
    sections = [SECTION_BUILDERS[s](rep) for s in rep.built]
    summary = run_summary(rep)                       # after the sections: it lists them
    appendices = [a for a in (appendix_pathways(rep), appendix_files(rep)) if a]
    appendices.append(appendix_docs(rep))            # last: it lists every pointer shown
    parts = [L.preamble(title), r'\begin{document}', title_tex, r'\tableofcontents', r'\clearpage', summary,
             *[r'\clearpage' + '\n' + s for s in sections],
             r'\clearpage', r'\appendix', *[r'\clearpage' + '\n' + a for a in appendices], r'\end{document}']
    os.makedirs(rep.dir, exist_ok=True)
    path = os.path.join(rep.dir, f'{rep.stem}_report.tex')
    with open(path, 'w') as fh:
        fh.write('\n\n'.join(parts) + '\n')
    return path


def build(export_dir, latex=True, latexmk=None, log=print, web_base='auto'):
    """Write <export_dir>/report/<stem>_report.tex and build the PDF from it.
    Returns the PDF path, or the .tex path when the PDF is not built."""
    if not os.path.isdir(export_dir):
        raise L.ReportError(f'no export folder at {export_dir}: run export_results.py (or postprocess.py) first')
    rep = MaterialReport(export_dir, log=log, web_base=web_base)
    tex = write_tex(rep)
    exe = L.latexmk_path(latexmk) if latex else None
    if exe is None:
        log(f'report: wrote {tex}' + ('' if not latex else
                                       f'; LaTeX (latexmk) not found. Build it with: cd {rep.dir} && latexmk -pdf '
                                       f'{os.path.basename(tex)}'))
        return tex
    pdf = L.build_pdf(tex, exe, log)
    log(f'report: {pdf}')
    return pdf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--stem', help="one material, e.g. 'Ni_supercell'")
    g.add_argument('--export-dir', help='an export folder, e.g. calculation/results/export/Ni_supercell')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    ap.add_argument('--tex-only', action='store_true', help='write the .tex but do not run LaTeX')
    ap.add_argument('--latexmk', default=None, help='path of latexmk (default: on PATH or /Library/TeX/texbin)')
    a = ap.parse_args(argv)
    export_dir = a.export_dir or os.path.join(a.results_dir, 'export', a.stem)
    try:
        build(export_dir, latex=not a.tex_only, latexmk=a.latexmk)
    except L.ReportError as exc:
        print(f'report: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

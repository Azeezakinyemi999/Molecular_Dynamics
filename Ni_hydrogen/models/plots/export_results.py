#!/usr/bin/env python3
"""
export_results.py
=================
Collect one material's finished results into a self-contained folder: CSV
tables for Part 1, Part 2, Part 3 and the overall permeability, a Markdown
summary, and copies of every figure that belongs to that material.

This writes nothing back into the pipeline — it only reads the artifacts the
workflow has already produced and reshapes them for a thesis or a paper. A
missing artifact is reported in the summary rather than silently omitted, so
the export always states what is absent.

Single-H runs are excluded by default, matching ``discover_runs`` — see its
docstring for why. Pass ``--include-1h`` to keep them.

Usage
-----
    python models/plots/export_results.py --stem Al_supercell
    python models/plots/export_results.py --stem Ni_supercell --outdir ~/thesis/ni
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _HERE)

from diffusivity_plot import (                      # noqa: E402
    DEFAULT_RESULTS_DIR, Run, discover_runs, pretty_material,
    resolve_host_atoms,
)

CALC_DIR = os.path.join(_PROJECT_ROOT, 'calculation')

# detailed_balance is a rate-based cross-check, not a reported solubility.
# The permeability JSON says so itself in `solubility_headline`; keep that
# distinction visible in every table rather than listing three equal routes.
HEADLINE_ROUTES   = ('geometric', 'vibrational')
DIAGNOSTIC_ROUTES = ('detailed_balance',)


# ── small helpers ────────────────────────────────────────────────────────────
def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_csv(path, rows, header):
    if not rows:
        return False
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return True


def _g(d, *keys, default=''):
    """First present key, else default."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


# ── Part 1: surface H2 dissociation ──────────────────────────────────────────
def part1(stem, outdir, notes):
    ranked = _load(os.path.join(CALC_DIR, 'neb', stem, 'ranked_barriers.json'))
    vib    = _load(os.path.join(CALC_DIR, 'neb', stem, 'diss_vib_rates.json')) or {}
    if not ranked:
        notes.append('Part 1: no ranked_barriers.json — surface NEB has not run.')
        return None

    by_label = {v.get('label', k): v for k, v in vib.items()}
    rows = []
    for e in ranked:
        lab = e.get('label')
        v   = by_label.get(lab, {})
        rows.append([
            lab, e.get('is_site'), e.get('fs_site1'), e.get('fs_site2'),
            e.get('Ea'), e.get('E_des'), e.get('delta_E'),
            e.get('converged'), e.get('fmax_final'),
            _g(v, 'Ea_zpe'), _g(v, 'Ed_zpe'), _g(v, 'nu'), _g(v, 'nu_reverse'),
            _g(v, 'is_minimum'), _g(v, 'ts_saddle'), _g(v, 'zpe_source'),
        ])
    header = ['label', 'is_site', 'fs_site1', 'fs_site2',
              'Ea_eV', 'E_des_eV', 'delta_E_eV', 'neb_converged', 'fmax_final',
              'Ea_zpe_eV', 'Ed_zpe_eV', 'nu_fwd_s-1', 'nu_rev_s-1',
              'is_minimum', 'ts_saddle', 'zpe_source']
    _write_csv(os.path.join(outdir, 'part1_dissociation.csv'), rows, header)

    n_unconv = sum(1 for e in ranked if e.get('converged') is False)
    n_sound  = sum(1 for v in vib.values()
                   if v.get('is_minimum') and v.get('ts_saddle'))
    if n_unconv:
        notes.append(f'Part 1: {n_unconv} of {len(ranked)} NEB pathways did not '
                     f'converge — their barriers are provisional.')
    if vib:
        notes.append(f'Part 1: {n_sound} of {len(vib)} pathways have both a '
                     f'relaxed IS and a resolved saddle (is_minimum & ts_saddle); '
                     f'only those carry a trustworthy prefactor.')
    if len(ranked) != len(vib):
        notes.append(f'Part 1: {len(ranked)} NEB pathways but {len(vib)} with '
                     f'vibrations — the difference has no prefactor.')
    return {'n_pathways': len(ranked), 'n_sound': n_sound}


# ── Part 2: subsurface hops, TST rates, solubility ───────────────────────────
def part2(stem, outdir, notes):
    out = {}
    # 2a. Hop A / Hop B barriers
    rows = []
    for hop in ('hopa', 'hopb'):
        d = _load(os.path.join(CALC_DIR, 'neb_subsurface', stem, hop,
                               f'{hop}_vib_rates.json')) or {}
        for k, v in sorted(d.items()):
            rows.append([hop, v.get('label', k), _g(v, 'env'),
                         _g(v, 'sub1_env'), _g(v, 'sub2_env'),
                         _g(v, 'Ea_raw'), _g(v, 'Ea_zpe'),
                         _g(v, 'Ed_raw'), _g(v, 'Ed_zpe'),
                         _g(v, 'nu'), _g(v, 'nu_reverse'), _g(v, 'zpe_source')])
    if rows:
        _write_csv(os.path.join(outdir, 'part2_hop_barriers.csv'), rows,
                   ['hop', 'label', 'env', 'sub1_env', 'sub2_env',
                    'Ea_raw_eV', 'Ea_zpe_eV', 'Ed_raw_eV', 'Ed_zpe_eV',
                    'nu_fwd_s-1', 'nu_rev_s-1', 'zpe_source'])
        n_a = sum(1 for r in rows if r[0] == 'hopa')
        n_b = len(rows) - n_a
        out['n_hopa'], out['n_hopb'] = n_a, n_b
        if n_b <= 1:
            notes.append(f'Part 2: Hop B rests on {n_b} site — any Hop B '
                         f'barrier quoted for this material is a single '
                         f'measurement, not an average.')
        # Say *why* a hop lost sites: an unconverged NEB is dropped before
        # vibrations, so the rate table is silently narrower than the NEB set.
        for hop in ('hopa', 'hopb'):
            ranked = _load(os.path.join(CALC_DIR, 'neb_subsurface', stem, hop,
                                        f'{hop}_ranked.json'))
            if not isinstance(ranked, list):
                continue
            bad = [e.get('label') for e in ranked if e.get('converged') is False]
            if bad:
                notes.append(
                    f'Part 2: {hop.upper()} NEB did not converge for '
                    f'{", ".join(str(b) for b in bad)} — dropped before '
                    f'vibrations, so {len(ranked)} NEB pathway(s) became '
                    f'{len(ranked) - len(bad)} in the rate table.')
    else:
        notes.append('Part 2: no hop vibration rates found.')

    # 2b. per-temperature TST rates
    rows = []
    for jf in sorted(glob.glob(os.path.join(CALC_DIR, 'results', stem,
                                            'rate_dict_T*K.json'))):
        d = _load(jf) or {}
        for k, v in sorted(d.items()):
            rows.append([_g(v, 'T_K'), k, _g(v, 'Ea_zpe'), _g(v, 'Ed_zpe'),
                         _g(v, 'delta_e'), _g(v, 'nu'), _g(v, 'nu_reverse'),
                         _g(v, 'k_forward'), _g(v, 'k_reverse')])
    rows.sort(key=lambda r: (r[0] if isinstance(r[0], (int, float)) else 0, r[1]))
    if rows:
        _write_csv(os.path.join(outdir, 'part2_tst_rates.csv'), rows,
                   ['T_K', 'site', 'Ea_zpe_eV', 'Ed_zpe_eV', 'delta_E_eV',
                    'nu_fwd_s-1', 'nu_rev_s-1', 'k_forward_s-1', 'k_reverse_s-1'])

    # 2c. per-environment solubility
    env = _load(os.path.join(CALC_DIR, 'results', stem, 'dH_sol_by_env.json'))
    if env:
        rows = [[k, _g(v, 'dH_sol_eV'), _g(v, 'dH_sol_err_eV'),
                 _g(v, 'dH_hopA_eV'), _g(v, 'dH_hopA_err_eV'),
                 _g(v, 'w_env'), _g(v, 'n_sites')]
                for k, v in sorted(env.items(),
                                   key=lambda kv: _g(kv[1], 'dH_sol_eV', default=0))]
        _write_csv(os.path.join(outdir, 'part2_solubility_by_env.csv'), rows,
                   ['environment', 'dH_sol_eV', 'dH_sol_err_eV',
                    'dH_hopA_eV', 'dH_hopA_err_eV', 'weight', 'n_sites'])
        out['n_env'] = len(env)
    return out


# ── Part 3: bulk diffusivity ─────────────────────────────────────────────────
def part3(runs, outdir, notes):
    resolve_host_atoms(runs, None)       # fills run.n_host in place
    per_T, fits = [], []
    for run in runs:
        d = _load(os.path.join(run.path, 'diffusivity_arrhenius.json'))
        if not d:
            notes.append(f'Part 3: {run.name} has no diffusivity_arrhenius.json.')
            continue
        pct = run.at_pct if run.at_pct is not None else ''
        T  = _g(d, 'T_K_arr', 'temperatures_K', default=[])
        D  = _g(d, 'D_arr', 'D_per_T_m2s', default=[])
        E  = _g(d, 'D_err_arr', default=[])
        for i, t in enumerate(T):
            per_T.append([run.n_H, pct, t,
                          D[i] if i < len(D) else '',
                          E[i] if i < len(E) else ''])
        fits.append([run.n_H, pct, _g(d, 'D0_m2s'), _g(d, 'D0_err'),
                     _g(d, 'E_D_eV'), _g(d, 'E_D_err_eV'),
                     _g(d, 'R2_fit', 'R2'), len([x for x in D if x and x > 0])])
    _write_csv(os.path.join(outdir, 'part3_diffusivity_vs_T.csv'), per_T,
               ['n_H', 'at_pct_H', 'T_K', 'D_m2s', 'D_err_m2s'])
    _write_csv(os.path.join(outdir, 'part3_diffusivity_fits.csv'), fits,
               ['n_H', 'at_pct_H', 'D0_m2s', 'D0_err_m2s',
                'E_D_eV', 'E_D_err_eV', 'R2_fit', 'n_positive_D_points'])
    for f in fits:
        if f[7] and f[7] < 3:
            notes.append(f'Part 3: n_H={f[0]} fit uses only {f[7]} positive D '
                         f'point(s) — a {f[7]}-point Arrhenius fit has '
                         f'{max(0, f[7] - 2)} degrees of freedom, so its R² is '
                         f'arithmetic, not evidence.')
    return {'n_loadings': len(fits)}


# ── Overall: permeability ────────────────────────────────────────────────────
def overall(runs, outdir, notes):
    rows, caveats = [], set()
    for run in runs:
        d = _load(os.path.join(run.path, 'permeability_arrhenius.json'))
        if not d:
            notes.append(f'Overall: {run.name} has no permeability_arrhenius.json.')
            continue
        for name, r in (d.get('routes') or {}).items():
            if not r or not r.get('available'):
                continue
            rows.append([run.n_H, name,
                         'headline' if name in HEADLINE_ROUTES else 'diagnostic',
                         _g(r, 'Phi0'), _g(r, 'E_phi_eV'),
                         _g(r, 'Phi0_rel_err'), _g(r, 'E_phi_err_eV'),
                         _g(r, 'r2_S')])
        for jf in sorted(glob.glob(os.path.join(run.path, 'permeability_T*K.json'))):
            p = _load(jf) or {}
            if p.get('dilute_limit_caveat'):
                # Identical text per loading apart from the n_H it names, so
                # keep one copy and list the loadings it applies to.
                caveats.add(run.n_H)
    _write_csv(os.path.join(outdir, 'overall_permeability.csv'), rows,
               ['n_H', 'route', 'status', 'Phi0_molH_m-1_s-1_Pa-0.5',
                'E_phi_eV', 'Phi0_rel_err', 'E_phi_err_eV', 'r2_S'])
    if rows:
        notes.append('Overall: `detailed_balance` is a rate-based cross-check, '
                     'not a reported solubility — the permeability JSON labels '
                     'it so itself. Quote geometric/vibrational.')
    if caveats:
        loads = ', '.join(f'{n}H' for n in sorted(caveats))
        notes.append(
            f"Overall: Sieverts' law and Richardson's permeation formula assume "
            f"dilute dissolved H. None of the exported loadings ({loads}) is the "
            f"dilute limit (n_H=1), so H-H interactions were present in the MD "
            f"box and the derived solubility and permeability are approximations. "
            f"Note the tension with excluding 1H: the dilute limit is exactly the "
            f"loading whose MSD does not converge, so there is no loading here "
            f"that is both dilute and well sampled.")
    return {'n_rows': len(rows)}


# ── figures ──────────────────────────────────────────────────────────────────
def figures(stem, runs, outdir):
    mat = pretty_material(stem).replace(' ', '_')
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    seen, pats = [], [
        os.path.join(CALC_DIR, 'results', 'plots', '**', f'{mat}_*.png'),
        os.path.join(CALC_DIR, 'results', 'plots', '**', f'{stem}_*.png'),
    ] + [os.path.join(r.path, 'analysis', '*.png') for r in runs]
    for pat in pats:
        for src in sorted(glob.glob(pat, recursive=True)):
            base = os.path.basename(src)
            if os.sep + 'analysis' + os.sep in src:
                run = os.path.basename(os.path.dirname(os.path.dirname(src)))
                base = f'{run}__{base}'
            dst = os.path.join(figdir, base)
            if dst in seen:
                continue
            shutil.copy2(src, dst)
            seen.append(dst)
    return [os.path.basename(x) for x in seen]


# ── summary ──────────────────────────────────────────────────────────────────
def write_summary(stem, outdir, counts, figs, notes, runs, excluded_1h):
    lines = [f'# {pretty_material(stem)} — exported results', '',
             f'Source: `{CALC_DIR}`  ·  stem: `{stem}`',
             f'Loadings included: ' +
             (', '.join(f'{r.n_H}H' for r in runs) or 'none'), '']
    if excluded_1h:
        lines += ['> Single-H runs are excluded by default (one walker, no '
                  'ensemble average). Re-run with `--include-1h` to keep them.', '']
    lines += ['## Tables', '']
    for fn, what in (
            ('part1_dissociation.csv',        'Part 1 — H₂ dissociation pathways on the surface'),
            ('part2_hop_barriers.csv',        'Part 2 — Hop A/B subsurface barriers and prefactors'),
            ('part2_tst_rates.csv',           'Part 2 — TST forward/reverse rates per temperature'),
            ('part2_solubility_by_env.csv',   'Part 2 — per-environment ΔH_sol and site weights'),
            ('part3_diffusivity_vs_T.csv',    'Part 3 — D(T) per loading'),
            ('part3_diffusivity_fits.csv',    'Part 3 — Arrhenius fits D₀, E_D'),
            ('overall_permeability.csv',      'Overall — Φ₀ and E_Φ per route'),
    ):
        mark = 'x' if os.path.exists(os.path.join(outdir, fn)) else ' '
        lines.append(f'- [{mark}] `{fn}` — {what}')
    lines += ['', f'## Figures ({len(figs)})', '']
    lines += [f'- `figures/{f}`' for f in figs]
    if notes:
        lines += ['', '## Read these before quoting any number', '']
        lines += [f'{i}. {n}' for i, n in enumerate(notes, 1)]
    lines.append('')
    p = os.path.join(outdir, 'SUMMARY.md')
    with open(p, 'w') as fh:
        fh.write('\n'.join(lines))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stem', required=True, help="e.g. 'Al_supercell'")
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    ap.add_argument('--outdir', default=None,
                    help='default: <results-dir>/export/<stem>')
    ap.add_argument('--include-1h', action='store_true',
                    help='keep single-H runs, excluded by default')
    args = ap.parse_args(argv)

    outdir = args.outdir or os.path.join(args.results_dir, 'export', args.stem)
    os.makedirs(outdir, exist_ok=True)

    runs = discover_runs(args.results_dir, f'{args.stem}_*H',
                         include_single_h=args.include_1h)
    runs.sort(key=lambda r: r.n_H)

    notes: list[str] = []
    counts = {}
    print(f'\n{pretty_material(args.stem)} → {outdir}\n')
    counts['part1']   = part1(args.stem, outdir, notes)
    counts['part2']   = part2(args.stem, outdir, notes)
    counts['part3']   = part3(runs, outdir, notes)
    counts['overall'] = overall(runs, outdir, notes)
    figs = figures(args.stem, runs, outdir)
    summary = write_summary(args.stem, outdir, counts, figs, notes, runs,
                            excluded_1h=not args.include_1h)

    for fn in sorted(os.listdir(outdir)):
        if fn.endswith('.csv'):
            n = sum(1 for _ in open(os.path.join(outdir, fn))) - 1
            print(f'  {fn:<34} {n:>4} row(s)')
    print(f'  {"figures/":<34} {len(figs):>4} figure(s)')
    print(f'\n  summary: {summary}')
    if notes:
        print(f'  {len(notes)} caveat(s) recorded in the summary.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

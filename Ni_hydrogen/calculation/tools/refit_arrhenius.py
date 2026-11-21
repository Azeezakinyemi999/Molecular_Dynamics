#!/usr/bin/env python3
"""
refit_arrhenius.py
==================
TEMPORARY (issue #18). Delete this file once every results dir has been refit.

Refits every ``<results-dir>/*/diffusivity_arrhenius.json`` with the ordinary
least-squares ``fit_arrhenius`` that replaced the inverse-variance weighted fit
in #18. Files written before #18 keep the weighted E_D and D0 until refit.

Two JSON layouts are handled:

- pipeline dirs (``{stem}_{n}H``): refit from ``analysis/diffusivity_table.txt``
  through ``run_arrhenius_pipeline``, the same call the generated diffusivity
  scripts make, so the point mask, the fit and ``arrhenius.png`` are identical
  to a fresh Part 3 run. Keys rewritten: ``D0_m2s``, ``E_D_eV``, ``D0_err``,
  ``E_D_err_eV``, ``R2_fit``.
- legacy ``{stem}_01H`` dirs (written by the deleted build_*_from_old.py
  converters; no analysis/ dir): refit from the file's own ``temperatures_K``
  and ``D_per_T_m2s`` over finite D > 0, the rule run_arrhenius_pipeline uses.
  Keys rewritten: ``D0_m2s``, ``E_D_eV``, ``D0_err_m2s``, ``E_D_err_eV``, ``R2``.

Every other key, and the key order, is kept.

Usage
-----
    python calculation/tools/refit_arrhenius.py            # dry run: print before/after, change nothing
    python calculation/tools/refit_arrhenius.py --write    # apply
    python calculation/tools/refit_arrhenius.py --results-dir /path/to/results --write

``--results-dir`` may be given more than once; the default is
``calculation/results``.

With ``--write``, each refit dir also loses its ``permeability_T*K.done``
markers, so the next Part 2 run recomputes Φ(T) and J(T) from the new D0 and
E_D instead of skipping those temperatures. Nothing else is deleted. Then
regenerate and re-run Part 2 (regenerate_permeation_scripts.py,
wrap_permeation_runs_west.py) and the exports.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')   # headless cluster nodes

# Project root = the directory containing models/ (same walk-up as the other tools).
_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, 'models')):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

import numpy as np                                                    # noqa: E402

from models.diffusivity_post_processing import (                      # noqa: E402
    fit_arrhenius,
    run_arrhenius_pipeline,
)

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(WORK_DIR, 'results')


def _refit(run_dir, doc, write):
    """Refit one dir. Returns (Ea, Ea_err, D0, D0_err, R2, n_points) or None."""
    table = os.path.join(run_dir, 'analysis', 'diffusivity_table.txt')
    if os.path.exists(table):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = os.path.join(run_dir, 'analysis') if write else tmp
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                arr = run_arrhenius_pipeline(table, outdir=outdir)
            for line in log.getvalue().splitlines():
                if line.startswith('WARNING'):
                    print(f'    {line}')
            if arr['fig'] is not None:
                import matplotlib.pyplot as plt
                plt.close(arr['fig'])
        if not np.isfinite(arr['Ea']):
            return None
        D = np.asarray(arr['D_arr'], dtype=float)
        n = int((np.isfinite(D) & (D > 0)).sum())
        return arr['Ea'], arr['Ea_err'], arr['D0'], arr['D0_err'], arr['R2'], n

    # legacy _01H layout: no table, the D(T) series lives in the JSON itself
    T = np.array([float(t) for t in doc['temperatures_K']])
    D = np.array([float(doc['D_per_T_m2s'][str(int(t))]) for t in T])
    ok = np.isfinite(D) & (D > 0)
    if ok.sum() < 2:
        return None
    return (*fit_arrhenius(T[ok], D[ok]), int(ok.sum()))


def main():
    ap = argparse.ArgumentParser(
        description='Refit diffusivity_arrhenius.json files by ordinary least squares (#18).')
    ap.add_argument('--results-dir', action='append', dest='results_dirs',
                    help=f'results directory to scan (repeatable; default {DEFAULT_RESULTS_DIR})')
    ap.add_argument('--write', action='store_true',
                    help='rewrite the JSON, redraw arrhenius.png, remove permeability_T*K.done')
    args = ap.parse_args()

    dirs = args.results_dirs or [DEFAULT_RESULTS_DIR]
    files = sorted(f for d in dirs
                   for f in glob.glob(os.path.join(d, '*', 'diffusivity_arrhenius.json')))
    if not files:
        print(f'No diffusivity_arrhenius.json under: {dirs}')
        sys.exit(1)

    print(f'{"WRITING" if args.write else "DRY RUN (nothing written)"}: '
          f'{len(files)} file(s) under {dirs}\n')
    print(f'{"dir":34s} {"n":>2}  {"E_D weighted → OLS (eV)":>24}  {"ΔE_D":>7}  '
          f'{"D0 weighted → OLS (m²/s)":>25}  {"R²":>6}')

    n_written, skipped, removed = 0, [], []
    for path in files:
        run_dir = os.path.dirname(path)
        name = os.path.basename(run_dir)
        with open(path) as f:
            doc = json.load(f)
        k_d0_err = 'D0_err' if 'D0_err' in doc else 'D0_err_m2s'
        k_r2 = 'R2_fit' if 'R2_fit' in doc else 'R2'

        res = _refit(run_dir, doc, args.write)
        if res is None:
            skipped.append(name)
            print(f'{name:34s}  < 2 valid points, left unchanged')
            continue
        Ea, Ea_err, D0, D0_err, R2, n = res
        E_old, D0_old = doc['E_D_eV'], doc['D0_m2s']
        print(f'{name:34s} {n:>2}  {E_old:10.4f} → {Ea:10.4f}  {Ea - E_old:+7.4f}  '
              f'{D0_old:11.3e} → {D0:11.3e}  {R2:6.4f}')

        if args.write:
            doc.update({'D0_m2s': D0, 'E_D_eV': Ea, k_d0_err: D0_err,
                        'E_D_err_eV': Ea_err, k_r2: R2})
            with open(path, 'w') as f:
                json.dump(doc, f, indent=2)
            n_written += 1
            for marker in sorted(glob.glob(os.path.join(run_dir, 'permeability_T*K.done'))):
                os.remove(marker)
                removed.append(marker)

    print()
    if skipped:
        print(f'Skipped (fewer than 2 valid points): {skipped}')
    if args.write:
        print(f'Rewrote {n_written} file(s); removed {len(removed)} permeability_T*K.done marker(s):')
        for marker in removed:
            print(f'  {marker}')
        print('\nNext: regenerate the permeation scripts, re-run Part 2 for every stem, '
              'then the exports.')
    else:
        print('Dry run only. Re-run with --write to apply.')


if __name__ == '__main__':
    main()

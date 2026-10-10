#!/usr/bin/env python3
"""
postprocess.py — everything that happens after Parts 1, 2 and 3 have run.

One entry point, used two ways:

* the pipeline calls :func:`postprocess` when a metal finishes, so analysis is
  never something you have to remember;
* you call the CLI yourself, as often as you like.

Both take the same path through the same code.

Nothing here touches the cluster or writes into the pipeline. It reads run
artifacts and produces figures, tables and a summary, so it is safe to repeat
after any change -- which is the point: when a correction lands, re-running
this is how the analysis catches up.

Design
------
Each stage is gated on **its own** inputs, not on the previous stage having
succeeded. A metal with only Part 3 gets its diffusivity figures and a CSV
export; the permeation figures are skipped with the reason stated. "Part 2 is
not done yet" is a normal state, not a failure, and does not affect the exit
code.

A plot script that raises is caught, reported and stepped over. Analysis must
never destroy a run that already produced the physics.

Usage
-----
    python models/plots/postprocess.py --stem Ni_supercell
    python models/plots/postprocess.py --all
    python models/plots/postprocess.py --all --include-1h
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusivity_plot import DEFAULT_RESULTS_DIR, discover_runs  # noqa: E402

# Each stage: (label, module, pattern kind, what it needs).
# 'run'      -> pattern is <stem>_*H     (one figure set per H loading)
# 'material' -> pattern is <stem>        (loading-independent quantities)
# 'none'     -> no pattern at all
DIFFUSIVITY_STAGES = [
    ('diffusivity_plot',      'run',      '--outdir'),
    ('msd_plot',              'run',      '--outdir'),
    ('arrhenius_params_plot', 'run',      '--outdir'),
    ('thermal_expansion',     'none',     '--outdir'),
]
PERMEATION_STAGES = [
    ('permeability_plot', 'runH',     '--outdir'),
    ('solubility_plot',   'runH',     '--outdir'),
    ('environment_plot',  'material', '--outdir'),
    ('neb_mep_plot',      'material', '--outdir'),
]


# ── readiness ───────────────────────────────────────────────────────────────
def _has(results_dir, stem, name, loading=True, include_1h=False):
    """Does a usable artifact exist?

    For loading-keyed artifacts this asks discover_runs, NOT a raw glob, so the
    gate applies exactly the exclusions the plot scripts will apply. A material
    whose only run is validation-grade or single-H globs as present but is
    discarded downstream -- gating on the glob made those look READY and then
    fail, when they are simply not ready.
    """
    if not loading:
        return bool(glob.glob(os.path.join(results_dir, stem, name)))
    import io as _io
    from contextlib import redirect_stdout as _rs
    with _rs(_io.StringIO()):                      # discover_runs narrates
        runs = discover_runs(results_dir, f'{stem}_*H', include_single_h=include_1h)
    return any(os.path.isfile(os.path.join(r.path, name)) for r in runs)


def readiness(stem, results_dir=DEFAULT_RESULTS_DIR, calc_dir=None,
              include_1h=False):
    """What exists for this material, and therefore what can be produced."""
    calc_dir = calc_dir or os.path.dirname(results_dir.rstrip('/'))
    return {
        'diffusivity': _has(results_dir, stem, 'diffusivity_arrhenius.json',
                            include_1h=include_1h),
        'permeability': _has(results_dir, stem, 'permeability_arrhenius.json',
                             include_1h=include_1h),
        'part1': os.path.isfile(os.path.join(calc_dir, 'neb', stem,
                                             'ranked_barriers.json')),
        'environments': _has(results_dir, stem, 'dH_sol_by_env.json',
                             loading=False),
    }


def completeness(stem, results_dir=DEFAULT_RESULTS_DIR, calc_dir=None):
    """How far through the pipeline this material is, 0-4. Used to order --all."""
    return sum(bool(v) for v in readiness(stem, results_dir, calc_dir).values())


def discover_stems(results_dir=DEFAULT_RESULTS_DIR):
    """Every material with any result at all, most complete first."""
    stems = set()
    for p in glob.glob(os.path.join(results_dir, '*')):
        if not os.path.isdir(p):
            continue
        b = os.path.basename(p)
        if b in ('plots', 'export') or ' ' in b or b.startswith('_'):
            continue
        stems.add(b.split('_supercell')[0] + '_supercell'
                  if '_supercell' in b else b)
    return sorted(stems, key=lambda s: (-completeness(s, results_dir), s))


# ── running one stage ───────────────────────────────────────────────────────
def _run(module_name, argv, verbose):
    """Call a plot script's main() in process. Never raises."""
    try:
        mod = __import__(module_name)
    except Exception as exc:                                  # import failure
        return False, f'could not import ({exc})'
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = mod.main(argv)
    except SystemExit as exc:                     # argparse or an explicit exit
        rc = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        return False, f'{type(exc).__name__}: {exc}'
    out = buf.getvalue()
    if verbose and out:
        print('\n'.join('        ' + l for l in out.rstrip().splitlines()))
    return (rc in (0, None)), ('' if rc in (0, None) else f'exit {rc}')


def _argv(kind, stem, outdir, include_1h):
    a = []
    if kind == 'run':
        a += ['--pattern', f'{stem}_*']
    elif kind == 'runH':
        a += ['--pattern', f'{stem}_*H']
    elif kind == 'material':
        a += ['--pattern', stem]
    a += ['--outdir', outdir]
    if include_1h and kind in ('run', 'runH'):
        a += ['--include-1h']
    return a


# ── the entry point ─────────────────────────────────────────────────────────
def postprocess(stem, results_dir=DEFAULT_RESULTS_DIR, outdir=None,
                include_1h=False, calc_dir=None, verbose=False):
    """Plot, then export, for one material. Returns a report dict.

    Ordering is enforced here rather than left to whoever runs it: the export
    COPIES figures, so plotting has to happen first or the export quietly
    produces a folder with none.
    """
    outdir = outdir or os.path.join(results_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)
    rep = {'stem': stem, 'ran': [], 'skipped': [], 'failed': [],
           'ready': readiness(stem, results_dir, calc_dir, include_1h)}

    def do(stages, gate, why):
        for name, kind, _ in stages:
            if not gate:
                rep['skipped'].append((name, why)); continue
            ok, err = _run(name, _argv(kind, stem, outdir, include_1h), verbose)
            (rep['ran'] if ok else rep['failed']).append(
                name if ok else (name, err))

    do(DIFFUSIVITY_STAGES, rep['ready']['diffusivity'],
       'no usable diffusivity fit (Part 3 incomplete, or only '
       'validation-grade / single-H runs exist)')
    do(PERMEATION_STAGES, rep['ready']['permeability'],
       'no permeability yet (Part 2 incomplete)')

    # export last, and only if there is something for it to collect
    if any(rep['ready'][k] for k in ('diffusivity', 'permeability', 'part1')):
        ok, err = _run('export_results',
                       ['--stem', stem, '--results-dir', results_dir]
                       + (['--include-1h'] if include_1h else []), verbose)
        (rep['ran'] if ok else rep['failed']).append(
            'export_results' if ok else ('export_results', err))
    else:
        rep['skipped'].append(('export_results',
                               'nothing produced yet for this material'))
    return rep


def print_report(rep, indent='  '):
    r = rep['ready']
    state = ' '.join(f'{k}={"y" if v else "-"}' for k, v in r.items())
    print(f'{indent}{rep["stem"]}   [{state}]')
    if rep['ran']:
        print(f'{indent}  ran     : {", ".join(rep["ran"])}')
    for name, why in rep['skipped']:
        print(f'{indent}  skipped : {name} - {why}')
    for name, err in rep['failed']:
        print(f'{indent}  FAILED  : {name} - {err}')
    if not rep['ran'] and not rep['failed']:
        print(f'{indent}  nothing to do yet')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--stem', help="one material, e.g. 'Ni_supercell'")
    g.add_argument('--all', action='store_true',
                   help='every material with results, most complete first')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--include-1h', action='store_true',
                    help='keep single-H runs, excluded by default')
    ap.add_argument('--verbose', action='store_true',
                    help='show each script output and full tracebacks')
    a = ap.parse_args(argv)

    stems = (discover_stems(a.results_dir) if a.all else [a.stem])
    print(f'\npost-processing {len(stems)} material(s)\n')
    reps = []
    for st in stems:
        rep = postprocess(st, a.results_dir, a.outdir, a.include_1h,
                          verbose=a.verbose)
        print_report(rep); print()
        reps.append(rep)

    n_fail = sum(len(r['failed']) for r in reps)
    n_ran = sum(len(r['ran']) for r in reps)
    print(f'  {n_ran} step(s) ran, {n_fail} failed, '
          f'{sum(len(r["skipped"]) for r in reps)} skipped (not ready)\n')
    # not-ready is normal; only a real failure is an error
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())

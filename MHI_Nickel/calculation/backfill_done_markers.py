#!/usr/bin/env python3
"""
calculation/backfill_done_markers.py
=====================================
One-off, idempotent script: walks the already-populated calculation/ tree
and writes `.done` markers wherever the corresponding real output already
exists but the marker doesn't -- so already-completed cluster work isn't
mistaken for "not done" the first time the new checkpoint mechanism
(models/checkpoint.py) ships.

Only ever CREATES .done files -- never deletes or modifies anything else.
Safe to re-run any number of times.

Usage
-----
    python calculation/backfill_done_markers.py [--dry-run]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.checkpoint import mark_done

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def _backfill(pattern, marker_name, dry_run, marker_dir_fn=None, exclude_substr=None):
    """
    For every path matching `pattern` (a real output file), write a
    `marker_name` .done file in the directory `marker_dir_fn` maps it to
    (default: the same directory the real output lives in). Matches whose
    basename contains `exclude_substr` are skipped -- e.g. bulk_min_*.lammps
    also matches the NPT-derived bulk_min_{stem}_npt_final_{T}K.lammps /
    _npt_after_heat.lammps files, which are a different step's output.
    """
    n_written = 0
    n_already = 0
    for real_output in sorted(glob.glob(pattern, recursive=True)):
        if exclude_substr and exclude_substr in os.path.basename(real_output):
            continue
        marker_dir = marker_dir_fn(real_output) if marker_dir_fn else os.path.dirname(real_output)
        marker_path = os.path.join(marker_dir, marker_name)
        if os.path.exists(marker_path):
            n_already += 1
            continue
        n_written += 1
        if dry_run:
            print(f'  [dry-run] would write: {marker_path}')
        else:
            mark_done(marker_path)
    n_total = n_written + n_already
    print(f'{marker_name:24s}  pattern={pattern}')
    print(f'  {n_written} written, {n_already} already present, {n_total} total matches')
    return n_written


def _backfill_by_suffix(pattern, prefix, marker_prefix, dry_run):
    """
    Like _backfill, but for the "one file per temperature" family
    ({prefix}T{T}K.json -> {marker_prefix}T{T}K.done, same directory) --
    the marker name itself varies per match, so it can't be a fixed string.
    """
    n_written = 0
    n_already = 0
    for real_output in sorted(glob.glob(pattern)):
        base = os.path.basename(real_output)
        t_part = base[len(prefix):-len('.json')]   # '{T}K'
        marker_path = os.path.join(os.path.dirname(real_output), f'{marker_prefix}{t_part}.done')
        if os.path.exists(marker_path):
            n_already += 1
            continue
        n_written += 1
        if dry_run:
            print(f'  [dry-run] would write: {marker_path}')
        else:
            mark_done(marker_path)
    n_total = n_written + n_already
    print(f'{marker_prefix + "{T}K.done":24s}  pattern={pattern}')
    print(f'  {n_written} written, {n_already} already present, {n_total} total matches')
    return n_written


def _phase1a_marker_dir(real_output):
    # real_output = calculation/structures/bulk_min_{stem}.lammps
    # marker lives in calculation/results/{stem}/phase1a.done -- a
    # different directory than the real output, since Phase 1a's output
    # is shared (n_H-independent) and lives under structures/, not results/.
    stem = os.path.basename(real_output)[len('bulk_min_'):-len('.lammps')]
    return os.path.join(WORK_DIR, 'results', stem)


def _npt_backfill(dry_run):
    # real_output = calculation/structures/bulk_min_{stem}_npt_final_{T}K.lammps
    # marker lives in calculation/results/{stem}/npt_{T}K.done -- same
    # cross-directory situation as Phase 1a above.
    pattern = os.path.join(WORK_DIR, 'structures', 'bulk_min_*_npt_final_*K.lammps')
    n_written = 0
    n_already = 0
    for real_output in sorted(glob.glob(pattern)):
        base = os.path.basename(real_output)[len('bulk_min_'):-len('.lammps')]
        stem, sep, t_part = base.partition('_npt_final_')
        if not sep or not t_part.endswith('K'):
            continue
        marker_path = os.path.join(WORK_DIR, 'results', stem, f'npt_{t_part}.done')
        if os.path.exists(marker_path):
            n_already += 1
            continue
        n_written += 1
        if dry_run:
            print(f'  [dry-run] would write: {marker_path}')
        else:
            mark_done(marker_path)
    n_total = n_written + n_already
    print(f'{"npt_{T}K.done":24s}  pattern={pattern}')
    print(f'  {n_written} written, {n_already} already present, {n_total} total matches')
    return n_written


def main(dry_run):
    total = 0

    print('=== NEB workflow (Section A, Phase E) ===')
    total += _backfill(
        os.path.join(WORK_DIR, 'slabs/*/phase1_slab/slab_*.lammps'),
        'slab.done', dry_run,
    )
    total += _backfill(
        os.path.join(WORK_DIR, 'slabs/*/phase2_relax/relaxed_slab.lammps'),
        'relax.done', dry_run,
    )
    total += _backfill(
        os.path.join(WORK_DIR, 'slabs/*/phase3_sites/surface_sites.json'),
        'sites.done', dry_run,
    )
    total += _backfill(
        os.path.join(WORK_DIR, 'neb/*/vibrations_diss/*/vib_frequencies.json'),
        'vib.done', dry_run,
    )

    print('\n=== Permeation workflow (Phase 3 vibrations) ===')
    total += _backfill(
        os.path.join(WORK_DIR, '*/vibrations/*/vib_frequencies.json'),
        'vib.done', dry_run,
    )

    print('\n=== Diffusivity workflow (Phase 1a, 1b-A, Phase 3) ===')
    total += _backfill(
        os.path.join(WORK_DIR, 'structures/bulk_min_*.lammps'),
        'phase1a.done', dry_run, _phase1a_marker_dir,
        exclude_substr='_npt_',
    )
    total += _npt_backfill(dry_run)
    total += _backfill(
        os.path.join(WORK_DIR, 'results/*H/diffusivity_arrhenius.json'),
        'phase3.done', dry_run,
        lambda p: os.path.join(os.path.dirname(p), 'analysis'),
    )

    print('\n=== Diffusivity workflow (Phase 1b-B, co-located) ===')
    total += _backfill(
        os.path.join(WORK_DIR, 'results/*H/structures/*K/bulk_min_h.lammps'),
        'minh.done', dry_run,
    )

    print('\n=== Permeation workflow (Phase 4/5/6, solubility -- co-located) ===')
    total += _backfill_by_suffix(
        os.path.join(WORK_DIR, 'results/*/rate_dict_T*K.json'),
        'rate_dict_T', 'rate_T', dry_run,
    )
    total += _backfill_by_suffix(
        os.path.join(WORK_DIR, 'results/*H/permeation_sweep_T*K.json'),
        'permeation_sweep_T', 'sweep_T', dry_run,
    )
    total += _backfill_by_suffix(
        os.path.join(WORK_DIR, 'results/*H/permeability_T*K.json'),
        'permeability_T', 'permeability_T', dry_run,
    )
    total += _backfill(
        os.path.join(WORK_DIR, 'results/*H/solubility_arrhenius_kmc.json'),
        'solubility.done', dry_run,
    )

    print(f'\n=== Done. {total} marker(s) {"would be " if dry_run else ""}written total. ===')
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                         help='Preview what would be written without creating any files.')
    args = parser.parse_args()
    main(dry_run=args.dry_run)

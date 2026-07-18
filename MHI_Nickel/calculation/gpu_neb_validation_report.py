#!/usr/bin/env python3
"""
calculation/gpu_neb_validation_report.py
=========================================
Parses the real CPU+mh-1 baseline log plus whichever of the three
gpu_neb_validation_test.py variants have produced output so far, and prints
a single side-by-side comparison: device, model, seconds/step, starting
fmax, fmax at a matched step count, and total elapsed wall time.

Safe to run at any point -- before any variant has started (reports "not
started yet"), while jobs are still mid-run (reads whatever's been written
so far), or after they finish (also reports convergence from neb_barrier.txt
if present).

Read-only: never writes into any of the log files it inspects.

Usage
-----
    python calculation/gpu_neb_validation_report.py
"""
import datetime as _dt
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import BASE_DIR

WORK_DIR = os.path.join(BASE_DIR, 'calculation')
PAIR_DIR = os.path.join(
    WORK_DIR, 'neb', 'Hastelloy_N_1234_supercell', 'neb', 's_145__s_73+s_144')
TEST_DIR = os.path.join(WORK_DIR, 'gpu_neb_validation_test')

_STEP_RE = re.compile(
    r'^(?:FIRE|MDMin):\s*(\d+)\s+(\d{2}:\d{2}:\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')

# name, description, log path, model, device
_VARIANTS = [
    ('baseline (existing)', 'device=cpu   model=mh-1 (current)',
     os.path.join(PAIR_DIR, 'neb_phase1.log')),
    ('gpu_mh1', 'device=cuda  model=mh-1 (current, real code path)',
     os.path.join(TEST_DIR, 'gpu_mh1', 'neb_phase1.log')),
    ('cpu_mp0b2', 'device=cpu   model=mp-0b2-medium (old)',
     os.path.join(TEST_DIR, 'cpu_mp0b2', 'neb_phase1.log')),
    ('gpu_mp0b2', 'device=cuda  model=mp-0b2-medium (old)',
     os.path.join(TEST_DIR, 'gpu_mp0b2', 'neb_phase1.log')),
]


def _parse_log(path):
    """Return the most recent leg's step rows as [(step, seconds_of_day, energy, fmax), ...].

    A log can contain more than one leg concatenated (a from-scratch
    restart resets the step counter to 0) -- only the last leg is kept,
    since that reflects current progress.
    """
    if not os.path.exists(path):
        return None
    legs = [[]]
    with open(path) as f:
        for line in f:
            m = _STEP_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            h, mi, s = (int(x) for x in m.group(2).split(':'))
            secs = h * 3600 + mi * 60 + s
            energy = float(m.group(3))
            fmax = float(m.group(4))
            if legs[-1] and step <= legs[-1][-1][0]:
                legs.append([])
            legs[-1].append((step, secs, energy, fmax))
    last_leg = legs[-1]
    return last_leg if last_leg else None


def _elapsed_seconds(rows):
    """Sum of positive step-to-step deltas, tolerating a single midnight rollover."""
    total = 0.0
    for (_, t0, _, _), (_, t1, _, _) in zip(rows, rows[1:]):
        d = t1 - t0
        if d < 0:
            d += 24 * 3600
        total += d
    return total


def _fmax_at_step(rows, target_step):
    """fmax of the row at or immediately after target_step, or None."""
    for step, _, _, fmax in rows:
        if step >= target_step:
            return step, fmax
    return None


def main():
    print(f'Real pair: {PAIR_DIR}')
    print()

    parsed = {}
    for name, desc, path in _VARIANTS:
        rows = _parse_log(path)
        parsed[name] = rows
        if rows is None:
            print(f'{name:22s} {desc:42s}  -- not started yet ({path})')
            continue

        n_steps = rows[-1][0] + 1
        elapsed = _elapsed_seconds(rows)
        sec_per_step = elapsed / max(1, len(rows) - 1) if len(rows) > 1 else float('nan')
        start_fmax = rows[0][3]
        latest_fmax = rows[-1][3]

        print(f'{name:22s} {desc}')
        print(f'  steps logged (this leg) : {n_steps}')
        print(f'  elapsed (this leg)      : {elapsed/60:.1f} min')
        print(f'  seconds/step            : {sec_per_step:.2f}')
        print(f'  starting fmax           : {start_fmax:.2f} eV/A')
        print(f'  latest fmax             : {latest_fmax:.2f} eV/A')

        barrier_file = os.path.join(os.path.dirname(path), 'neb_barrier.txt')
        if os.path.exists(barrier_file):
            with open(barrier_file) as f:
                content = f.read()
            conv = 'Converged   : True' in content
            print(f'  neb_barrier.txt         : present  (converged={conv})')
        print()

    # Matched-step-count comparison -- fairest single number to eyeball.
    available = {name: rows for name, rows in parsed.items() if rows}
    if len(available) >= 2:
        common_step = min(rows[-1][0] for rows in available.values())
        common_step = min(common_step, 100)  # cap so one long-running leg doesn't dominate
        print(f'--- fmax at step {common_step} (fairest apples-to-apples point) ---')
        for name, _, _ in _VARIANTS:
            rows = parsed.get(name)
            if not rows:
                continue
            hit = _fmax_at_step(rows, common_step)
            if hit:
                step, fmax = hit
                print(f'  {name:22s} step {step:4d}  fmax={fmax:.2f} eV/A')
    else:
        print('Fewer than 2 variants have started -- nothing to compare yet.')


if __name__ == '__main__':
    main()

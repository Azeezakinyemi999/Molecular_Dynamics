"""
tests/pipeline/diagnose.py
============================
Post-mortem diagnostic tool for a completed (or partially completed)
tests/pipeline/work/ smoke-test run.

Automates the manual investigation process used to root-cause Bugs
17-20 this session (reading barrier files, reconstructing rate-dict
calls locally, grepping logs for tracebacks) so the next failure can be
triaged in seconds instead of a multi-step manual session.

Usage
-----
    python tests/pipeline/diagnose.py [work_dir]

    work_dir defaults to tests/pipeline/work (relative to repo root).

Checks performed
-----------------
1. Traceback scan       — every *.out/*.log file, any Python traceback.
2. NEB barrier sanity    — flags |E_abs|/|E_des|/|delta_E| > 20 eV as
                           physically implausible (the Bug 18 signature:
                           a hardcoded/missing reference energy gives
                           barriers offset by the whole slab's energy).
3. Vibration completeness — for every Hop A/B job, checks whether both
                           its IS and TS vib_frequencies.json actually
                           exist (the Bug 19/20 signature: NEB "done"
                           doesn't guarantee vibration inputs landed).
4. Rate-dict emptiness   — flags any rate_dict_T*K.json with 0 entries.
5. Phase-6 readiness     — did DH_DISS_EV/DH_ENTRY_EV resolve?
6. Diffusivity fit health — classifies each diffusivity_arrhenius.json
                           as real/graceful-NaN/smoke-test-fallback, so
                           a NaN fit isn't mistaken for a bug.
7. Summary correlation   — reads summary.txt and prints each FAIL line
                           next to whatever evidence 1-6 turned up.

This is a read-only diagnostic; it never modifies the work directory.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_BARRIER_ENERGY_THRESHOLD_EV = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Traceback scan
# ─────────────────────────────────────────────────────────────────────────────

def find_tracebacks(work_dir: Path) -> list[tuple[str, str]]:
    """Return (file, exception_line) for every Python traceback found in
    any *.out or *.log file under work_dir."""
    findings = []
    paths = sorted(work_dir.rglob('*.out')) + sorted(work_dir.rglob('*.log'))
    for path in paths:
        try:
            lines = path.read_text(errors='replace').splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(lines):
            if line.strip() != 'Traceback (most recent call last):':
                continue
            exc_line = None
            for j in range(i + 1, min(i + 80, len(lines))):
                cand = lines[j]
                if cand and not cand[0].isspace():
                    exc_line = cand.strip()
                    break
            findings.append((str(path), exc_line or '(exception line not found)'))
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEB barrier sanity
# ─────────────────────────────────────────────────────────────────────────────

def check_neb_barriers(
    work_dir: Path, threshold_ev: float = _BARRIER_ENERGY_THRESHOLD_EV
) -> list[tuple[str, dict]]:
    """Flag any neb_barrier.txt whose E_abs/E_des/delta_E exceeds
    threshold_ev in magnitude -- the Bug 18 signature (hardcoded/missing
    reference energy gives barriers offset by ~100+ eV)."""
    issues = []
    for path in sorted(work_dir.rglob('neb_barrier.txt')):
        try:
            text = path.read_text()
        except OSError:
            continue
        vals = {}
        for key in ('E_abs', 'E_des', 'delta_E'):
            m = re.search(rf'{key}\s*=\s*(-?[\d.]+)', text)
            if m:
                vals[key] = float(m.group(1))
        bad = {k: v for k, v in vals.items() if abs(v) > threshold_ev}
        if bad:
            issues.append((str(path), bad))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vibration completeness (Hop A / Hop B)
# ─────────────────────────────────────────────────────────────────────────────

def check_vibration_completeness(work_dir: Path) -> list[str]:
    """For every hopa_jobs.json / hopb_jobs.json found, check whether the
    corresponding {hop}_{sid}_IS and {hop}_{sid}_TS vib_frequencies.json
    files actually exist. Returns a list of human-readable problem
    descriptions (empty list = all good, or no Hop A/B jobs found).

    Paths stored inside the jobs JSON (barrier_file, job_dir, ...) are
    absolute paths written at submit time on the machine that ran the
    job (e.g. the cluster) -- they do NOT point at wherever this work_dir
    was rsync'd/copied to for local diagnosis, so they can't be trusted
    directly. Every path used here is instead re-derived from jobs_path's
    own (real, local) location plus each job's sid, mirroring the layout
    generate_permeation_scripts actually writes:
    .../neb_subsurface/<stem>/hopX/<sid>/... and
    .../vibrations/<stem>/<hop>_<sid>_<IS|TS>/...
    """
    problems: list[str] = []
    for jobs_path in sorted(work_dir.rglob('hop[ab]_jobs.json')):
        hop = 'hopa' if 'hopa' in jobs_path.name else 'hopb'
        try:
            jobs = json.loads(jobs_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f'{jobs_path}: could not read jobs file ({exc})')
            continue

        hop_dir = jobs_path.parent               # .../neb_subsurface/<stem>/hopX/
        stem_dir = hop_dir.parent                 # .../neb_subsurface/<stem>/
        diffusivity_dir = stem_dir.parent.parent  # .../diffusivity/
        vib_dir = diffusivity_dir / 'vibrations' / stem_dir.name

        for job in jobs:
            sid = job.get('sid', '?')
            local_barrier = hop_dir / str(sid) / 'neb_barrier.txt'
            if not local_barrier.exists():
                problems.append(
                    f'{hop}_{sid}: neb_barrier.txt missing at {local_barrier} '
                    f'— NEB never completed for this job'
                )
                continue
            for suffix in ('IS', 'TS'):
                vib_json = vib_dir / f'{hop}_{sid}_{suffix}' / 'vib_frequencies.json'
                if not vib_json.exists():
                    problems.append(
                        f'{hop}_{sid}_{suffix}: vib_frequencies.json missing '
                        f'({vib_json}) — check the matching vib_*.out for a '
                        f'traceback (e.g. "Expected exactly 1 H atom, found 0")'
                    )
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rate-dict emptiness
# ─────────────────────────────────────────────────────────────────────────────

def check_rate_dicts(work_dir: Path) -> list[tuple[str, str]]:
    """Flag any rate_dict_T*K.json with 0 entries."""
    issues = []
    for path in sorted(work_dir.rglob('rate_dict_T*K.json')):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            issues.append((str(path), f'unreadable: {exc}'))
            continue
        if not data:
            issues.append((str(path), 'empty — 0 rate constants produced'))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# 5. Phase-6 readiness (DH_DISS_EV / DH_ENTRY_EV)
# ─────────────────────────────────────────────────────────────────────────────

def check_phase6_readiness(work_dir: Path) -> list[str]:
    """Grep permeation logs for the DH_DISS_EV/DH_ENTRY_EV resolution
    warning. Returns a list of the exact warning lines found (empty
    list means Phase 6 was never blocked by this, or never ran)."""
    hits = []
    for path in sorted(work_dir.rglob('*.out')):
        try:
            text = path.read_text(errors='replace')
        except OSError:
            continue
        if 'DH_DISS_EV or DH_ENTRY_EV could not be determined' in text:
            hits.append(str(path))
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# 6. Diffusivity fit health
# ─────────────────────────────────────────────────────────────────────────────

def check_diffusivity_fits(work_dir: Path) -> list[tuple[str, str]]:
    """Classify each diffusivity_arrhenius.json as real / graceful-NaN /
    smoke-test-fallback / unreadable."""
    results = []
    for path in sorted(work_dir.rglob('diffusivity_arrhenius.json')):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            results.append((str(path), f'unreadable: {exc}'))
            continue
        d0 = data.get('D0_m2s')
        ea = data.get('E_D_eV')
        is_nan = d0 is None or ea is None or d0 != d0 or ea != ea
        if data.get('SMOKE_TEST_SYNTHETIC_FALLBACK'):
            results.append((str(path), f'synthetic fallback injected (D0={d0:.3e}, Ea={ea:.3f} eV) '
                                        f'— real fit was NaN at smoke scale, not a bug'))
        elif is_nan:
            results.append((str(path), 'real fit is NaN — expected at smoke scale if '
                                        '<2 valid D(T) points; not a bug unless unexpected'))
        else:
            results.append((str(path), f'valid fit: D0={d0:.3e} m²/s, Ea={ea:.4f} eV'))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary correlation
# ─────────────────────────────────────────────────────────────────────────────

def parse_summary(work_dir: Path) -> list[tuple[str, str]]:
    """Parse summary.txt into (status, label) tuples."""
    summary_path = work_dir / 'summary.txt'
    if not summary_path.exists():
        return []
    entries = []
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if line.startswith('PASS ') or line.startswith('FAIL '):
            status, label = line.split(None, 1)
            entries.append((status, label.strip()))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnosis(work_dir: Path) -> dict:
    """Run all checks and return a structured results dict."""
    return {
        'summary': parse_summary(work_dir),
        'tracebacks': find_tracebacks(work_dir),
        'bad_barriers': check_neb_barriers(work_dir),
        'vib_problems': check_vibration_completeness(work_dir),
        'empty_rate_dicts': check_rate_dicts(work_dir),
        'phase6_blocked_in': check_phase6_readiness(work_dir),
        'diffusivity_fits': check_diffusivity_fits(work_dir),
    }


def print_report(results: dict) -> None:
    summary = results['summary']
    n_pass = sum(1 for s, _ in summary if s == 'PASS')
    n_fail = sum(1 for s, _ in summary if s == 'FAIL')

    print('=' * 70)
    print('  PIPELINE SMOKE TEST DIAGNOSIS')
    print('=' * 70)

    if summary:
        print(f'\nsummary.txt: {n_pass} passed, {n_fail} failed')
        if n_fail:
            print('Failed checks:')
            for _, label in summary:
                if (('FAIL', label)) in summary:
                    pass
            for status, label in summary:
                if status == 'FAIL':
                    print(f'  - {label}')
    else:
        print('\nNo summary.txt found (run may still be in progress).')

    print(f'\n[1] Tracebacks found: {len(results["tracebacks"])}')
    for path, exc in results['tracebacks']:
        print(f'  {path}\n      {exc}')

    print(f'\n[2] Implausible NEB barriers (>{_BARRIER_ENERGY_THRESHOLD_EV} eV): '
          f'{len(results["bad_barriers"])}')
    for path, bad in results['bad_barriers']:
        print(f'  {path}: {bad}')

    print(f'\n[3] Vibration completeness problems: {len(results["vib_problems"])}')
    for problem in results['vib_problems']:
        print(f'  {problem}')

    print(f'\n[4] Empty rate dicts: {len(results["empty_rate_dicts"])}')
    for path, msg in results['empty_rate_dicts']:
        print(f'  {path}: {msg}')

    print(f'\n[5] Phase 6 (permeability) blocked in: {len(results["phase6_blocked_in"])} log(s)')
    for path in results['phase6_blocked_in']:
        print(f'  {path}')

    print(f'\n[6] Diffusivity fits: {len(results["diffusivity_fits"])}')
    for path, msg in results['diffusivity_fits']:
        print(f'  {path}: {msg}')

    print('\n' + '=' * 70)
    if not summary:
        print('  Nothing to correlate yet (no summary.txt).')
    elif n_fail == 0 and not results['tracebacks'] and not results['bad_barriers'] \
            and not results['vib_problems'] and not results['empty_rate_dicts']:
        print('  Clean run: no failures, no tracebacks, no anomalies detected.')
    else:
        print('  See findings [1]-[6] above for root-cause evidence on the '
              'FAILed checks listed at the top.')
    print('=' * 70)


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        work_dir = Path(argv[1])
    else:
        work_dir = Path(__file__).parent / 'work'

    if not work_dir.exists():
        print(f'Work directory not found: {work_dir}', file=sys.stderr)
        return 1

    results = run_diagnosis(work_dir)
    print_report(results)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

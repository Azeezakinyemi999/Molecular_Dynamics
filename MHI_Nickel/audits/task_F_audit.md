# Task F Audit — Checkpoint guards in all three workflow generators

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `models/neb_workflow.py`, `models/permeation_workflow.py`
(F1 guards in `models/diffusivity_workflow.py` were committed with Task B)

---

## 1. Goal

The pipeline orchestrator (`pipeline_run.py`) runs as a long-lived Python process on the `west`
partition.  If it is killed and restarted, all three run scripts previously re-submitted every
job unconditionally — even for phases whose outputs already existed.  This task adds
existence-check guards so each phase is silently skipped when its sentinel output file is present.

---

## 2. Mechanism

### F1 — `models/diffusivity_workflow.py` (committed with Task B)

| Phase | Guard file | Skip message |
| --- | --- | --- |
| 1a bare-min | `min_bare_out` (`bulk_min.lammps`) | `[1a] Already exists: … — skipping` |
| 1b NPT per-T | `npt_final_paths[T]` | `[1b] NPT {T}K already done — skipping` |
| 1b bulk+H min per-T | `min_h_out_T` | `[1b] bulk+H min {T}K already done — skipping` |
| 2 NVT per-T | `msd_file` | `[2] NVT {T}K already done — skipping` |

### F2 — `models/neb_workflow.py` (this commit)

Wraps the entire Phases A-D block (orchestrate + Phase D array submissions) in:

```python
_ranked_f = os.path.join(NEB_DIR, 'ranked_barriers.json')
if not os.path.exists(_ranked_f):
    result = orchestrate_full_neb_workflow(...)
    ...
    wait_for_jobs({'neb_array': neb_jid})
    print('  All NEB calculations done.')
else:
    print(f'  Phases A-D already complete: {_ranked_f} — skipping')
```

`ranked_barriers.json` is the sentinel: it is written by `orchestrate_full_neb_workflow` only
after all NEB calculations finish and barriers are ranked (line 1895 of `neb_workflow.py`).

Phase E (vibrational frequencies) already has its own independent guard
(`if not os.path.exists(_ranked_f_e)`) and is not affected.

### F3 — `models/permeation_workflow.py` (this commit)

Three guards added:

**Hop A** — guard file: `SUB_NEB_DIR/hopa/hopa_jobs.json`

Written by `orchestrate_hopa_neb` (line 377-378 of `neb_subsurface.py`).  The `else` branch
loads `hopa_jobs` from JSON so downstream code (`orchestrate_hopb_neb`, `collect_is_ts_paths`)
still receives the list.

**Hop B** — guard file: `SUB_NEB_DIR/hopb/hopb_jobs.json`

Written by `orchestrate_hopb_neb` (line 641-642 of `neb_subsurface.py`).  The `else` branch
loads `hopb_jobs` from JSON identically.

**KMC sweep per-T** — guard file: `RESULTS_DIR/permeation_sweep_T{T}K.json`

Early-`continue` at the top of the `for _T in TEMPERATURES` loop before the expensive rate-dict
load.  The `_out` path variable is now defined once at loop entry (guard check) rather than once
near the end (write).  Duplicate assignment at the bottom of the loop body was removed.

---

## 3. Impact

| File | Effect |
| --- | --- |
| `neb_workflow.py` generated `neb_run.py` | Phases A-D skipped on re-run if `ranked_barriers.json` exists |
| `permeation_workflow.py` generated `permeation_run.py` | Hop A, Hop B, KMC per-T all idempotent |
| `diffusivity_workflow.py` generated `diffusivity_run.py` | Already guarded (Task B) |
| Simulation physics | None — guards are pure orchestration logic |

---

## 4. Verification results

### F-V1 — Syntax parse

Both modified files parse cleanly with `ast.parse` ✓

### F-V2 — neb_workflow.py guard in place

`_ranked_f = os.path.join(NEB_DIR, 'ranked_barriers.json')` at line 2267 ✓
`if not os.path.exists(_ranked_f):` guard present ✓
`else: print('Phases A-D already complete…')` present ✓
Phase E guard is independent (`_ranked_f_e` at line 2328) — not broken ✓

### F-V3 — permeation_workflow.py Hop A guard

`_hopa_jobs_json = os.path.join(SUB_NEB_DIR, 'hopa', 'hopa_jobs.json')` at line 108 ✓
`if not os.path.exists(_hopa_jobs_json):` present ✓
`else: hopa_jobs = json.load(...)` loads the list for downstream use ✓
Guard file path matches `orchestrate_hopa_neb` write path (`str(hopa_dir / 'hopa_jobs.json')`) ✓

### F-V4 — permeation_workflow.py Hop B guard

`_hopb_jobs_json = os.path.join(SUB_NEB_DIR, 'hopb', 'hopb_jobs.json')` at line 145 ✓
`if not os.path.exists(_hopb_jobs_json):` present ✓
`else: hopb_jobs = json.load(...)` loads the list ✓
Guard file path matches `orchestrate_hopb_neb` write path ✓

### F-V5 — KMC sweep guard

`_out = os.path.join(...)` defined at top of loop body ✓
`if os.path.exists(_out): continue` present ✓
Duplicate `_out = ...` at end of loop body removed ✓

### F-V6 — F1 guards in diffusivity_workflow.py confirmed

All four phases guarded: Phase 1a, 1b NPT, 1b bulk+H min, Phase 2 NVT ✓

---

## 5. Issues found

None.

---

## 6. Status

**VERIFIED.** `models/neb_workflow.py` and `models/permeation_workflow.py` committed in this
task.  F1 (`models/diffusivity_workflow.py`) was already committed with Task B.

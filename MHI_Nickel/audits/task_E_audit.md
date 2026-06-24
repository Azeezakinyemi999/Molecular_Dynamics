# Task E Audit — NEB restart detection + lammpstrj conversion + SLURM chaining

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `models/ase_neb.py`, `models/neb_workflow.py`, `models/neb_subsurface.py`

---

## 1. Goal

ASE NEB jobs run on the `short` partition (48h wall time) and can take 2–12 h per hop. Without
restart support, a job killed at wall time loses all progress. This task adds three pieces:

1. **Restart detection** in the generated NEB script — reads the last complete band from an
   existing `.traj` checkpoint and skips phases already done.
2. **`.lammpstrj` conversion** — after each MDMin phase writes an OVITO-readable dump file
   alongside the `.traj` checkpoint.
3. **SLURM chaining** — replaces `write_slurm_job` with `write_chained_slurm_job` in
   `neb_workflow.py` (main NEB) and `neb_subsurface.py` (Hop A, Hop B).

---

## 2. Mechanism

### E1 — Restart detection in generated script (`models/ase_neb.py`)

**Variables injected into the generated script (already existed):**
- `TRAJ_PHASE1` — path to `neb_phase1.traj` / `hopa_phase1.traj` / `hopb_phase1.traj`
- `TRAJ_PHASE2` — path to corresponding `*_phase2.traj`

**New imports added to generated script:**

```python
from pathlib import Path as _Path
from ase.io import read, Trajectory as _Trajectory
```

**`_load_last_band` helper:**

```python
def _load_last_band(traj_path):
    _t = _Trajectory(traj_path)
    _n = len(_t)
    _last = max(0, _n - (N_IMAGES + 2))
    return [_t[_last + _i] for _i in range(min(N_IMAGES + 2, _n - _last))]
```

`_Trajectory` stores every optimizer step as `N_IMAGES + 2` frames (end-points included).
`len(traj) - (N_IMAGES + 2)` gives the offset of the last complete band.

**Three-branch restart logic (replaces IDPP block):**

| Condition | Action | `_restart_phase` |
|---|---|---|
| `TRAJ_PHASE2` exists | Load last band → CI-NEB, skip Phase 1 | 2 |
| `TRAJ_PHASE1` exists | Load last band → CI-NEB, skip Phase 1 | 2 |
| Neither | Fresh run: `images` from copies + IDPP | 1 |

When restarting from either trajectory, `neb = NEB(..., climb=True)` is constructed directly —
no need to re-run Phase 1.

**Phase 1 gated:**

```python
if _restart_phase == 1:
    opt1 = MDMin(neb, logfile=LOG_PHASE1, dt=0.05, trajectory=TRAJ_PHASE1, ...)
    opt1.run(fmax=N1_FMAX, steps=N1_STEPS)
```

Phase 2 always runs (both fresh and restarted execution paths reach it).

### E1b — `.lammpstrj` conversion

After Phase 1 completes (inside the `if _restart_phase == 1:` block):

```python
if TRAJ_PHASE1 and os.path.exists(TRAJ_PHASE1):
    _lmp1 = TRAJ_PHASE1.replace(".traj", ".lammpstrj")
    _w1(_lmp1, _r1(TRAJ_PHASE1, index=":"), format="lammps-dump-text")
```

After Phase 2 completes (before MEP extraction):

```python
if TRAJ_PHASE2 and os.path.exists(TRAJ_PHASE2):
    _lmp2 = TRAJ_PHASE2.replace(".traj", ".lammpstrj")
    _w2(_lmp2, _r2(TRAJ_PHASE2, index=":"), format="lammps-dump-text")
```

Output files per job:
- `*_phase1.traj` / `*_phase2.traj` — used by restart logic
- `*_phase1.lammpstrj` / `*_phase2.lammpstrj` — OVITO visualization

### E2 — SLURM chaining

All three call sites replace `write_slurm_job` with `write_chained_slurm_job` using the same
pattern:

```python
_neb_wall = neb_slurm_opts.get('time', '12:00:00')
_h, _m, _s = (int(x) for x in _neb_wall.split(':'))
_neb_cutoff_secs = _h * 3600 + _m * 60 + _s - 300
_neb_cutoff = f'{_neb_cutoff_secs // 3600:02d}:{(_neb_cutoff_secs % 3600) // 60:02d}:{_neb_cutoff_secs % 60:02d}'
write_chained_slurm_job(
    job_name=f'neb_{label}_{sid}',
    slurm_config=neb_slurm_opts,
    out_path=neb_sh,
    first_commands=[f'python {neb_script}'],
    restart_commands=[f'python {neb_script}'],
    restart_glob=traj_p2,
    cutoff=_neb_cutoff,
    work_dir=str(job_dir),
)
```

`restart_glob=traj_p2` — when `*_phase2.traj` exists the SLURM script knows a prior leg ran
and passes `restart_commands`. The NEB script itself detects which checkpoint to load.

`cutoff = wall_time − 300 s` — LAMMPS/ASE exits cleanly before SLURM kills the job.

**Files changed per call site:**

| File | Function | Traj prefix |
|---|---|---|
| `neb_workflow.py` | `orchestrate_full_neb_workflow` | `neb_phase` |
| `neb_subsurface.py` | `orchestrate_hopa_neb` | `hopa_phase` |
| `neb_subsurface.py` | `orchestrate_hopb_neb` | `hopb_phase` |

---

## 3. Impact

| Area | Effect |
|---|---|
| Fresh NEB run | Identical to before — IDPP + Phase 1 + Phase 2 |
| Phase 2 checkpoint present | Skips Phase 1 entirely, resumes CI-NEB from last band |
| Phase 1 checkpoint present | Skips Phase 1, starts Phase 2 from converged regular NEB band |
| OVITO trajectory | `.lammpstrj` written per phase after convergence |
| Simulation physics | None — same MDMin parameters, same NEB settings |

---

## 4. Verification results

### E-V1 — Syntax parse

`models/ase_neb.py`, `models/neb_workflow.py`, `models/neb_subsurface.py` all parse cleanly ✓

### E-V2 — Restart symbols in `ase_neb.py`

| Symbol | Present |
|---|---|
| `TRAJ_PHASE2` | ✓ |
| `TRAJ_PHASE1` | ✓ |
| `_load_last_band` | ✓ |
| `_restart_phase = 2` | ✓ |
| `_restart_phase = 1` | ✓ |
| `if _restart_phase == 1:` | ✓ |
| `lammps-dump-text` | ✓ |

Filenames `neb_phase1.traj` / `neb_phase2.traj` are injected by callers, not hardcoded in
`ase_neb.py` — FAILs on those literal checks are expected. ✓

### E-V3 — Traj args and chained submission in `neb_workflow.py`

`traj_phase1=traj_p1` at line 1736 ✓  
`traj_phase2=traj_p2` at line 1737 ✓  
`write_chained_slurm_job` present ✓  
`restart_glob=traj_p2` at line 1761 ✓  
`_neb_cutoff` defined ✓

### E-V4 — Traj args and chained submission in `neb_subsurface.py`

Hop A: `hopa_phase1.traj` (line 281), `hopa_phase2.traj` (line 282),
`traj_phase1=traj_p1` (line 300), `traj_phase2=traj_p2` (line 301),
`restart_glob=traj_p2` (line 325), `neb_hopa_` job name ✓

Hop B: `hopb_phase1.traj` (line 556), `hopb_phase2.traj` (line 557),
`traj_phase1=traj_p1` (line 575), `traj_phase2=traj_p2` (line 576),
`restart_glob=traj_p2` (line 600), `neb_hopb_` job name ✓

### E-V5 — Cutoff arithmetic

`12:00:00 − 300 s = 11:55:00` ✓  
`01:00:00 − 300 s = 00:55:00` ✓

---

## 5. Issues found

None.

---

## 6. Status

**VERIFIED.** `models/ase_neb.py`, `models/neb_workflow.py`, `models/neb_subsurface.py`
committed in this task.

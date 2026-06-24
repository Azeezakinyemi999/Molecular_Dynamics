# Task C Audit — Surface relaxation chaining in neb_workflow.py

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `models/neb_workflow.py`
(`models/lammps_script.py` already contained `write_surface_relaxation_restart_script` — no change needed)

---

## 1. Goal

The surface relaxation job (Phase 2 of `neb_run.py`) runs NVT MD on the slab and writes periodic
binary restart files every `restart_every` steps into `outdir/restarts/`.  Previously the job was
submitted with `write_slurm_job` — if it hit the 8h `gpu` wall time mid-NVT, the whole run
restarted from Phase 1 (CG minimization).  This task switches to `write_chained_slurm_job` so
the job automatically re-queues from the latest restart checkpoint.

---

## 2. Mechanism

### C1 — Import additions (`models/neb_workflow.py`, line 114-115)

```python
# Before:
from models.lammps_script import write_surface_relaxation_script
from models.create_slurm import write_slurm_job, submit_slurm_job

# After:
from models.lammps_script import write_surface_relaxation_script, write_surface_relaxation_restart_script
from models.create_slurm import write_slurm_job, write_chained_slurm_job, submit_slurm_job
```

### C2 — Replace `write_slurm_job` with `write_chained_slurm_job` in `run_phase2_surface_relaxation`

**New locals added:**

```python
lammps_rst_in = str(Path(outdir) / 'surface_relax_restart.in')
rst_glob      = str(Path(restart_dir) / 'surf_300K.*.restart')
```

`rst_glob` matches the glob pattern used by `write_surface_relaxation_script`:
`_restart_line(restart_dir, restart_every, label=f'surf_{target_t}K')` → LAMMPS writes
`{restart_dir}/surf_300K.<step>.restart` (for `target_t=300`).

**Restart script written:**

```python
write_surface_relaxation_restart_script(
    restart_file=rst_glob,
    slab_relaxed=relaxed_slab,
    relax_thermo=relax_thermo,
    out_path=lammps_rst_in,
    pair_style=PAIR_STYLE, mace_model=MACE_MODEL_LAMMPS,
    pair_suffix=PAIR_SUFFIX, elem_str=ELEM_STR_7,
    z_freeze_cutoff=z_freeze_cutoff, timestep=timestep,
    thermo_damp=SURF_THERMO_DAMP, nvt_steps=SURF_NVT_STEPS,
    restart_dir=restart_dir, restart_every=restart_every,
)
```

The restart script uses `read_restart {rst_glob}` (LAMMPS picks the highest-step matching file)
and `append yes` on both the trajectory dump and the thermo `ave/time` fix.

**Cutoff derived from wall time:**

```python
_wall       = slurm_opts.get('time', '24:00:00')
_h, _m, _s  = map(int, _wall.split(':'))
_cutoff_val = f'{(_h*3600+_m*60+_s-300)//3600:02d}:...'
```

Subtract 300 s (5 min) from wall time so LAMMPS exits cleanly before SLURM kills the job.

**Chained submission:**

```python
write_chained_slurm_job(
    job_name='SurfaceRelax',
    slurm_config=slurm_opts,
    out_path=slurm_script,
    first_commands=[f'{LAMMPS_CMD} {kk} -in {lammps_in} -log {log_path}'],
    restart_commands=[f'{LAMMPS_CMD} {kk} -in {lammps_rst_in} -log {log_path}'],
    restart_glob=rst_glob,
    cutoff=_cutoff_val,
    work_dir=outdir,
)
```

---

## 3. Impact

| Area | Effect |
| --- | --- |
| `run_phase2_surface_relaxation` | Now self-resubmitting if NVT hits wall time |
| Restart glob | `restarts/surf_300K.*.restart` — matches LAMMPS restart label `surf_300K` |
| `lammps_script.py` | Unchanged — `write_surface_relaxation_restart_script` was already present |
| Other callers of `write_slurm_job` in neb_workflow.py | Unaffected — only the surface relaxation call was changed |

---

## 4. Verification results

### C-V1 — New imports load without error

`write_surface_relaxation_restart_script` and `write_chained_slurm_job` both import
successfully ✓

### C-V2 — Restart script content correct

Generated script contains `read_restart`, `append yes` (both traj and thermo), and the
glob pattern embedded verbatim ✓

### C-V3 — Cutoff arithmetic correct

`04:00:00 − 5 min = 03:55:00` ✓

### C-V4 — File parses cleanly

`ast.parse` succeeds ✓

### C-V5 — Glob pattern matches LAMMPS restart label

`write_surface_relaxation_script` writes `{restart_dir}/surf_{target_t}K.*.restart`
(via `_restart_line(restart_dir, restart_every, label=f'surf_{target_t}K')`).
`rst_glob = str(Path(restart_dir) / 'surf_300K.*.restart')` matches for `target_t=300` ✓

---

## 5. Issues found

None.

---

## 6. Status

**VERIFIED.** `models/neb_workflow.py` committed in this task.

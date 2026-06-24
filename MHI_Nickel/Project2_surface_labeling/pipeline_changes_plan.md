# Plan: Thermostat fix + partition changes + restart guarantee

---

## Wall time limits

| Partition | Max wall time |
|---|---|
| `gpu` | 8 hours |
| `multigpu` | 24 hours |
| `short` | 48 hours |
| `west` | 30 days |

---

## Restart status of every job

| Job | Partition | Est. runtime | Margin | Current restart capability |
|---|---|---|---|---|
| Bare bulk minimization | `gpu` | ~5 min | 96× | None — too fast to matter |
| NPT (all T) | `gpu` | ~2.6 h | 3× | Binary restart after heating stage, but **no chaining** |
| Bulk+H minimization | `gpu` | ~5 min | 96× | None — too fast to matter |
| Surface relaxation | `gpu` | ~1 h | 8× | Binary restart files written every 10k steps, but **no chaining** |
| NVT (chained) | `multigpu` | ~84 h | ✓ | **Fully chained** — `write_chained_slurm_job` globs latest `.restart` and resumes |
| NEB ASE | `short` | ~2–12 h | 4–24× | Trajectory checkpoint exists, **no chaining** — Task E |
| NEB vibrations | `short` | ~1–4 h | 12–48× | `write_slurm_job` only — not needed (large margin) |
| Pipeline orchestrator | `west` | days | 30d | Python loop — restartable by re-running |

---

## Implementation priority

| Priority | Task | File(s) | Reason |
|---|---|---|---|
| P1 | G | `models/structure.py`, `models/diffusivity_workflow.py` | Scientific correctness — `a0(T)` feeds all downstream calculations |
| P1 | D | `calculation/pipeline.ipynb` | Config only — prerequisite before regenerating any run scripts |
| P2 | A3 | `models/lammps_script.py` | Prerequisite for B |
| P2 | B | `models/diffusivity_workflow.py` | Moves NPT/min to `gpu`, adds NPT chaining — needed before running diffusivity pipeline |
| P3 | F | `models/diffusivity_workflow.py`, `models/neb_workflow.py`, `models/permeation_workflow.py` | Checkpoint guards — must be in place before submitting full pipeline on `west` |
| P3 | C | `models/neb_workflow.py` | Surface relaxation chaining — 8× margin but "every job" requirement |
| P4 | E | `models/ase_neb.py`, `models/neb_workflow.py`, `models/neb_subsurface.py` | NEB restart — 4–24× margin on `short`, large safety buffer |

---

## What needs chaining added

**NPT** and **surface relaxation** both have binary restart files on disk, but no automatic re-submission if they hit wall time. Given their margins (3× and 8×), hitting the wall is unlikely — but "every single one" means we need chaining.

### NPT restart approach

NPT writes two restart files:

- `*_after_heat.restart` — after Stage 1 (heating)
- `*_final.restart` — after Stage 2 (production)

If Stage 2 hits wall time, a restart script can read `*_after_heat.restart` and re-run Stage 2 only. This mirrors exactly how NVT chaining works. We need:

1. A `write_npt_restart_script()` function in `lammps_script.py` (reads restart, runs Stage 2 production only)
2. Use `write_chained_slurm_job` for NPT submissions instead of `write_slurm_job`

### Surface relaxation restart approach

Surface relaxation writes binary restart files into `restarts/` every 10k steps. If it hits wall time during Phase 3 (NVT), a restart reads the latest `.restart` and continues from that step. This also mirrors NVT chaining. We need:

1. A `write_surface_relaxation_restart_script()` function in `lammps_script.py`
2. Use `write_chained_slurm_job` in `run_phase2_surface_relaxation` instead of `write_slurm_job`

### NEB ASE restart approach

The NEB script already supports `traj_phase1` and `traj_phase2` parameters — MDMin writes every optimization step (all N_images+2 atoms per step) to those trajectory files. The restart infrastructure is partially in place; it just needs wiring up.

Restart logic (added to the generated script in `write_ase_neb_script`):

- If `neb_phase2.traj` exists → Phase 2 hit wall time; load last band state from it, skip Phase 1, continue Phase 2 from where it stopped
- Elif `neb_phase1.traj` exists → Phase 1 finished, job died before Phase 2 started; load last band state from Phase 1 traj, go straight to Phase 2
- Else → fresh run, IDPP interpolation from scratch

For SLURM chaining, both `first_commands` and `restart_commands` are `python run_neb.py` — same script, because the script detects the checkpoint internally. The `restart_glob` points to `neb_phase2.traj`.

### NEB vibrations

Run on `short` (48h). Expected runtime 1–4 h gives a 12–48× margin. Treat as low priority for now.

---

## Task A — `models/lammps_script.py`

### ~~A1: Thermostat damping fix~~ ✓ DONE (commit `a5687df`)

All 6 `* 1000` multipliers removed. No action needed.

### ~~A2: Velocity initialization fix~~ ✓ DONE (commit `a5687df`)

`velocity all create 10.0 12345` is in place. No action needed.

### A3: Add `write_npt_restart_script()` (new function)

New function that reads `*_after_heat.restart` and runs Stage 2 (NPT production) only — mirrors `write_nvt_bulk_restart_script`. Needed for NPT chaining in Task B.

---

## Task B — `models/diffusivity_workflow.py`

### B1: Add `short_gpu_partition` + `short_gpu_time` parameters to function signature

Add after `gpu_time`:

```python
    short_gpu_partition,
    short_gpu_time,
```

### B2: Inject into `_header` template (after `GPU_TIME` line)

```python
SHORT_GPU_PARTITION = {short_gpu_partition!r}
SHORT_GPU_TIME      = {short_gpu_time!r}
```

### B3: Split `GPU_SLURM_CFG` in `_body` (line 134)

Replace:

```python
GPU_SLURM_CFG = dict(SLURM_DEFAULTS, partition='multigpu', time=NVT_WALL_TIME)
```

With:

```python
GPU_SLURM_CFG       = dict(SLURM_DEFAULTS, partition=GPU_PARTITION,      time=NVT_WALL_TIME)
SHORT_GPU_SLURM_CFG = dict(SLURM_DEFAULTS, partition=SHORT_GPU_PARTITION, time=SHORT_GPU_TIME)
```

### B4: Switch Phase 1a (bare minimization) → `SHORT_GPU_SLURM_CFG` (line ~180)

### B5: Switch Phase 1b NPT → `write_chained_slurm_job` with `SHORT_GPU_SLURM_CFG` (line ~220)

Replace `write_slurm_job` with `write_chained_slurm_job` for the NPT submission, passing the `_after_heat.restart` glob as the restart file. Requires `write_npt_restart_script` from Task A3.

### B6: Switch Phase 1b bulk+H minimization → `SHORT_GPU_SLURM_CFG` (line ~278)

Phase 2 NVT stays as `GPU_SLURM_CFG` — do not touch.

---

## Task C — `models/neb_workflow.py`

### C1: Switch `run_phase2_surface_relaxation` to `write_chained_slurm_job`

Line 189 sets up the SLURM submission. Replace `write_slurm_job` with `write_chained_slurm_job`, using the `restarts/*.restart` glob as the restart file. The restart script reads the latest binary restart and continues the NVT phase.

This requires a `write_surface_relaxation_restart_script()` in `lammps_script.py` (similar to `write_nvt_bulk_restart_script`).

---

## Task D — `calculation/pipeline.ipynb` Cell 5

### D1: Add short GPU variables

```python
SHORT_GPU_PARTITION = 'gpu'
SHORT_GPU_TIME      = '04:00:00'
```

### D2: Move NEB GPU to `gpu` partition

```python
NEB_GPU_SLURM = dict(SLURM_DEFAULTS, partition='gpu', time='04:00:00')
```

### D3: Pass new vars to `generate_diffusivity_scripts`

```python
    short_gpu_partition = SHORT_GPU_PARTITION,
    short_gpu_time      = SHORT_GPU_TIME,
```

---

## Task E — NEB automatic restart

Files: `models/ase_neb.py`, `models/neb_workflow.py`, `models/neb_subsurface.py`

### E1: Add restart detection to generated NEB script (`models/ase_neb.py`)

In `write_ase_neb_script`, inject a checkpoint block near the top of the generated script (before IDPP interpolation):

```python
# --- restart detection ---
if Path(TRAJ_PHASE2).exists():
    # Phase 2 hit wall time — reload last band state and skip Phase 1
    traj = Trajectory(TRAJ_PHASE2)
    last_frame = len(traj) - (N_IMAGES + 2)  # last complete band
    images = [traj[last_frame + i] for i in range(N_IMAGES + 2)]
    _restart_phase = 2
elif Path(TRAJ_PHASE1).exists():
    # Phase 1 finished, Phase 2 never started
    traj = Trajectory(TRAJ_PHASE1)
    last_frame = len(traj) - (N_IMAGES + 2)
    images = [traj[last_frame + i] for i in range(N_IMAGES + 2)]
    _restart_phase = 2
else:
    # Fresh run — IDPP interpolation
    images = [initial.copy() for _ in range(N_IMAGES + 2)]
    images[0] = initial; images[-1] = final
    neb_tmp = NEB(images); neb_tmp.interpolate('idpp')
    _restart_phase = 1
```

Then gate Phase 1 MDMin on `_restart_phase == 1`.

Always pass `traj_phase1` and `traj_phase2` in `run_neb_pipeline` (currently optional/None).

### E1b: Convert `.traj` → `.lammpstrj` after each MDMin phase

LAMMPS jobs already write `.dump` (LAMMPS format, readable in OVITO). For ASE NEB, add a conversion step after each phase completes — the workflow still uses `.traj` for restart, but the `.lammpstrj` file is available for OVITO visualization:

```python
# after Phase 1 MDMin completes:
from ase.io import read as _read, write as _write
_frames1 = _read(TRAJ_PHASE1, index=':')
_write(TRAJ_PHASE1.replace('.traj', '.lammpstrj'), _frames1,
       format='lammps-dump-text')

# after Phase 2 MDMin completes:
_frames2 = _read(TRAJ_PHASE2, index=':')
_write(TRAJ_PHASE2.replace('.traj', '.lammpstrj'), _frames2,
       format='lammps-dump-text')
```

This goes into the generated script body in `write_ase_neb_script`. Output files:
- `neb_phase1.traj` / `neb_phase2.traj` — used by restart logic
- `neb_phase1.lammpstrj` / `neb_phase2.lammpstrj` — used for OVITO visualization

### E2: Switch NEB SLURM submission to `write_chained_slurm_job`

In `models/neb_workflow.py` (line ~1718) and `models/neb_subsurface.py`, replace `write_slurm_job` with `write_chained_slurm_job` for the ASE NEB job:

```python
write_chained_slurm_job(
    job_name         = ...,
    slurm_opts       = NEB_GPU_SLURM,
    work_dir         = outdir,
    first_commands   = [f'python {neb_script}'],
    restart_commands = [f'python {neb_script}'],  # same — script is self-detecting
    restart_glob     = str(Path(outdir) / 'neb_phase2.traj'),
    cutoff           = ...,
)
```

---

## Task F — Checkpoint guards in generated run scripts

**Problem**: Re-running the pipeline orchestrator after a sudden cluster failure restarts everything from scratch — all three run scripts (`neb_run.py`, `diffusivity_run.py`, `permeation_run.py`) submit jobs unconditionally, with no check for existing outputs. The pattern already used correctly in Cell 6 of `pipeline.ipynb` for the pre-pipeline step must be applied throughout.

**Where the fix lives**: The guards go into the `_body` templates inside the three workflow generator files, not in the generated scripts themselves (which are regenerated each time from the notebook).

Files: `models/diffusivity_workflow.py`, `models/neb_workflow.py`, `models/permeation_workflow.py`

### F1: `models/diffusivity_workflow.py` — wrap each phase with existence check

**Phase 1a** — guard on `min_bare_out`:

```python
if not os.path.exists(min_bare_out):
    write_minimization_script(...)
    write_slurm_job(...)
    jid = submit_slurm_job(min_bare_sh)
    wait_for_jobs({'min_bare': jid})
else:
    print(f'  [1a] Already exists: {min_bare_out} — skipping')
```

**Phase 1b NPT** — guard on `npt_final_paths[T]` for each T:

```python
if not os.path.exists(npt_final_paths[T]):
    write_npt_script(...); write_slurm_job(...); submit + wait
else:
    print(f'  [1b] NPT {T}K already done — skipping')
```

*(collect `a0_T` from the existing file via `get_lattice_parameter` when skipping)*

**Phase 1b bulk+H min** — guard on `T_to_bulk_h[T]`:

```python
if not os.path.exists(T_to_bulk_h[T]):
    # insert H, write min script, submit + wait
else:
    print(f'  [1b] bulk+H min {T}K already done — skipping')
```

**Phase 2 NVT** — guard on the MSD output file:

```python
if not os.path.exists(msd_file):
    write_nvt_bulk_script(...); write_chained_slurm_job(...); submit + wait
else:
    print(f'  [2] NVT {T}K already done — skipping')
```

### F2: `models/neb_workflow.py` / `neb_run.py`

Guard the entire `orchestrate_full_neb_workflow` + Phase D submission on whether `ranked_barriers.json` already exists:

```python
_ranked_f = os.path.join(NEB_DIR, 'ranked_barriers.json')
if not os.path.exists(_ranked_f):
    result = orchestrate_full_neb_workflow(...)
    fsmin_jid = submit_slurm_job(result['fsmin_array_script'])
    wait_for_jobs({'fsmin_array': fsmin_jid})
    neb_jid = submit_slurm_job(result['neb_array_script'])
    wait_for_jobs({'neb_array': neb_jid})
else:
    print(f'  Phase A-D already complete: {_ranked_f} — skipping')
```

Phase E (diss vibrations) already checks `if not os.path.exists(_ranked_f_e)` — no change needed there.

### F3: `models/permeation_workflow.py` / `permeation_run.py`

Guard Hop A NEB on existing output; guard Hop B similarly. Guard each KMC sweep on `permeation_sweep_T{T}K.json`:

```python
_hopa_done = os.path.join(SUB_NEB_DIR, 'hopa', 'hopa_results.json')
if not os.path.exists(_hopa_done):
    hopa_out = orchestrate_hopa_neb(...); submit + wait
else:
    print('  Hop A already done — skipping')

# same pattern for Hop B

for _T in TEMPERATURES:
    _sweep_f = os.path.join(RESULTS_DIR, f'permeation_sweep_T{int(_T)}K.json')
    if not os.path.exists(_sweep_f):
        # run KMC sweep
    else:
        print(f'  KMC sweep T={_T}K already done — skipping')
```

---

## Task G — Average lattice parameter over last N NPT frames

**Problem**: `get_lattice_parameter(npt_final_paths[T])` reads a single LAMMPS structure file — the last snapshot. One frame can be a thermal fluctuation. The NPT run already writes box dimensions every `NPT_DUMP_EVERY=100` steps to `npt_boxdims_{T}K.dat`. Averaging the last N of those readings gives a thermally-averaged, more stable `a0(T)`.

### G1: Add `get_lattice_parameter_from_dump(dump_file, n_last)` to `models/structure.py`

The dump file has columns like:

```
# Step Lx Ly Lz ...
1000   3.5240  3.5240  3.5240
...
```

New function reads the file, takes the last `n_last` rows, averages `Lx`, and returns `a0` in Å:

```python
def get_lattice_parameter_from_dump(dump_file, n_last=50):
    rows = []
    with open(dump_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.split()
            rows.append(float(cols[1]))   # Lx column
    lx_mean = sum(rows[-n_last:]) / min(n_last, len(rows))
    return lx_mean  # already in Å for units metal
```

`n_last=50` → last 5000 steps of production (at 100-step dump interval) — tune as needed.

### G2: Update `diffusivity_workflow.py` to use the dump file

In the Phase 1b body template, replace:

```python
a0_T = get_lattice_parameter(npt_final_paths[T])
```

With:

```python
a0_T = get_lattice_parameter_from_dump(npt_dump, n_last=50)
```

Where `npt_dump` is already defined in the Phase 1b loop as `os.path.join(dirs['structures'], f'npt_boxdims_{T}K.dat')`.

Also update the import line in the template from `get_lattice_parameter` to `get_lattice_parameter_from_dump`.

---

## After all changes: regenerate scripts

1. Re-run Cell 8 → new `neb_run.py`
2. Re-run Cell with `generate_diffusivity_scripts` → new `diffusivity_run.py`
3. Re-run Cell with `generate_permeation_scripts` → new `permeation_run.py`

---

## Final partition map

| Job | Partition | Wall time | Restart mechanism |
|---|---|---|---|
| Bare bulk minimization | `gpu` | 04:00:00 | Not needed (~5 min) |
| NPT (all T) | `gpu` | 04:00:00 | Chained via `_after_heat.restart` |
| Bulk+H minimization | `gpu` | 04:00:00 | Not needed (~5 min) |
| Surface relaxation | `gpu` | 04:00:00 | Chained via latest `restarts/*.restart` |
| NVT (chained) | `multigpu` | 24:00:00 | Already chained ✓ |
| NEB ASE | `short` | 12:00:00 | Chained via `neb_phase2.traj` checkpoint |
| NEB vibrations | `short` | 06:00:00 | Not needed (12–48× margin) |
| Pipeline orchestrator | `west` | 30 days | Checkpoint guards (Task F) + Python re-run |

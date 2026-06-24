# Task B Audit — Partition split + NPT chaining in diffusivity_workflow.py

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `models/diffusivity_workflow.py`

---

## 1. Goal

The original `generate_diffusivity_scripts()` used a single `GPU_SLURM_CFG = multigpu/24h` for
all GPU jobs — including bare minimization (~5 min) and NPT (~2.6h) which do not need the 24h
multigpu partition. Two additional problems existed:
1. NPT had no restart capability — if a job hit the wall time after heating but before production
   completed, the whole run had to restart from scratch.
2. `get_lattice_parameter` (single-frame snapshot) was still used; Task G's thermal-average
   function was not yet wired into the template.

---

## 2. Mechanism

### B1 — New signature parameters

```python
short_gpu_partition='gpu',
short_gpu_time='04:00:00',
short_gpu_cutoff=None,   # auto-derived as short_gpu_time − 5 min if None
```

### B2 — New header variables in generated script

```python
SHORT_GPU_PARTITION  = 'gpu'
SHORT_GPU_TIME       = '04:00:00'
SHORT_GPU_CUTOFF     = '03:55:00'
```

### B3 — SHORT_GPU_SLURM_CFG in generated body

```python
GPU_SLURM_CFG       = dict(SLURM_DEFAULTS, partition=GPU_PARTITION,      time=NVT_WALL_TIME)
SHORT_GPU_SLURM_CFG = dict(SLURM_DEFAULTS, partition=SHORT_GPU_PARTITION, time=SHORT_GPU_TIME)
```

### B4 — Phase 1a (bare min) → `SHORT_GPU_SLURM_CFG`

Wrapped in `if not os.path.exists(min_bare_out)` checkpoint guard.

### B5 — Phase 1b NPT → `write_chained_slurm_job` + `SHORT_GPU_SLURM_CFG`

Both `write_npt_script` (fresh run) and `write_npt_restart_script` (Stage 2 restart) are
generated. `write_chained_slurm_job` wires them together with `restart_glob=npt_after_heat_rst`.
Wrapped in `if not os.path.exists(npt_final_paths[T])` checkpoint guard.

### B6 — Phase 1b bulk+H min → `SHORT_GPU_SLURM_CFG`

Wrapped in `if not os.path.exists(min_h_out_T)` checkpoint guard.

### B7 — Phase 2 NVT — unchanged

`GPU_SLURM_CFG = multigpu/24h` preserved. NVT is the only job that genuinely needs 24h.

### B8 — Task G import fix

```python
# Before:
from models.structure import get_lattice_parameter, insert_hydrogen
# After:
from models.structure import get_lattice_parameter_from_dump, insert_hydrogen
```

Call site updated:
```python
# Before:
a0_T = get_lattice_parameter(npt_final_paths[T])
# After:
_npt_dump_T = os.path.join(dirs['structures'], f'npt_boxdims_{T}K.dat')
a0_T = get_lattice_parameter_from_dump(_npt_dump_T, n_last=50)
```

---

## 3. Impact

| Area | Effect |
| --- | --- |
| `generate_diffusivity_scripts()` signature | 3 new optional params with defaults — backward compatible |
| Generated `diffusivity_run.py` | SHORT_GPU vars baked in; Phase 1a/1b-min on `gpu`; NPT chained |
| `pipeline.ipynb` Cell 4 | No longer raises `TypeError` — the `short_gpu_partition/time` args are now accepted |
| `a0(T)` extraction | Now uses thermal average over last 50 NPT box-dim frames (Task G fix) |
| Phase 2 NVT | Unchanged — still `multigpu/24h` |
| Checkpoint guards | Phases 1a, 1b NPT, 1b bulk+H min all skip if output already exists |

---

## 4. Verification results

### B-V1 — New params in signature

`short_gpu_partition`, `short_gpu_time`, `short_gpu_cutoff` all present ✓

### B-V2 — SHORT_GPU vars in generated header

`SHORT_GPU_PARTITION`, `SHORT_GPU_TIME`, `SHORT_GPU_CUTOFF` all present in generated script ✓

### B-V3 — SHORT_GPU_SLURM_CFG used

`SHORT_GPU_SLURM_CFG` present and applied to Phase 1a/1b-min ✓

### B-V4 — NPT uses `write_chained_slurm_job`

`write_chained_slurm_job` present in generated script ✓

### B-V5 — Phase 2 NVT preserved

`GPU_SLURM_CFG` and `partition=GPU_PARTITION` (multigpu) still present for NVT ✓

### B-V6 — `get_lattice_parameter_from_dump` wired in

`get_lattice_parameter_from_dump` imported ✓ — old `get_lattice_parameter` not imported ✓

### B-V7 — `write_npt_restart_script` imported in generated body

`write_npt_restart_script` present in generated script imports ✓

### B-V8 — Notebook Cell 4 no longer raises TypeError

`generate_diffusivity_scripts()` accepts `short_gpu_partition` and `short_gpu_time` without
error ✓ — Issue 1 from cross-impact audit resolved.

---

## 5. Issues found

None.

---

## 6. Additional checks

| Check | Result |
| --- | --- |
| `short_gpu_cutoff` auto-derivation: `04:00:00 − 5 min = 03:55:00` | ✓ Verified in stash diff |
| `insert_hydrogen` still imported (not touched) | ✓ |
| `write_nvt_bulk_restart_script` still imported (NVT chaining unchanged) | ✓ |
| Checkpoint guards use correct output path for each phase | ✓ `min_bare_out`, `npt_final_paths[T]`, `min_h_out_T` |

---

## 7. Status

**VERIFIED.** `models/diffusivity_workflow.py` committed in this task.
Also closes Task G's integration gap: `get_lattice_parameter_from_dump` now live in the
generated template.

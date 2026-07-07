# Task D Audit — Short-GPU partition config in pipeline.ipynb

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `calculation/pipeline.ipynb`

---

## 1. Goal

The original config used a single `GPU_SLURM_CFG` pointing to `multigpu` (24h) for all GPU jobs.
Short jobs — bare minimization (~5 min), NPT (~2.6h), NEB FS-min, surface relaxation — do not
need 24h multigpu slots and unnecessarily occupy that partition. The `gpu` partition (8h wall
time, faster queue) is the correct home for them. `multigpu` should be reserved exclusively for
the long NVT MD jobs that genuinely require up to 24h.

---

## 2. Mechanism

Three changes in `pipeline.ipynb` Cell 2 (the configuration hub):

### D1 — Add short-GPU variables

```python
SHORT_GPU_PARTITION = 'gpu'        # NPT + minimization (< 8 h)
SHORT_GPU_TIME      = '04:00:00'
```

### D2 — Switch NEB_GPU_SLURM to `gpu`

```python
# Before:
NEB_GPU_SLURM = dict(SLURM_DEFAULTS, partition='multigpu', time='04:00:00')
# After:
NEB_GPU_SLURM = dict(SLURM_DEFAULTS, partition='gpu',      time='04:00:00')
```

This affects Part 1 Phase A (surface relaxation) and Phase C (FS minimization array).

### D3 — Pass new vars to `generate_diffusivity_scripts()`

```python
generate_diffusivity_scripts(
    ...
    short_gpu_partition = SHORT_GPU_PARTITION,
    short_gpu_time      = SHORT_GPU_TIME,
    ...
)
```

`diffusivity_workflow.py` (Task B stash) accepts these and injects them into `diffusivity_run.py`
as `SHORT_GPU_PARTITION`, `SHORT_GPU_TIME`, and `SHORT_GPU_CUTOFF` (auto-derived as
`short_gpu_time − 5 min` to give LAMMPS time to write restart before SLURM kills the job).

---

## 3. Impact

| Area | Effect |
| --- | --- |
| `pipeline.ipynb` Cell 2 | 2 new variables added; `NEB_GPU_SLURM` partition changed |
| `neb_run.py` (generated, Part 1) | Surface relaxation and FS-min jobs use `gpu` (8h) instead of `multigpu` (24h) |
| `diffusivity_run.py` (generated, Part 3) | `SHORT_GPU_PARTITION/TIME/CUTOFF` baked in; Task B uses them for bare-min and NPT |
| `multigpu` usage | Reserved for NVT MD only — the one job that legitimately needs 24h |
| Runtime logic | None — config-only change, no simulation physics affected |

---

## 4. Verification results

### V1 — SHORT_GPU variables defined with correct values

- `SHORT_GPU_PARTITION = 'gpu'` present in Cell 2 ✓
- `SHORT_GPU_TIME = '04:00:00'` present in Cell 2 ✓

### V2 — NEB_GPU_SLURM uses `gpu` not `multigpu`

```python
NEB_GPU_SLURM = dict(SLURM_DEFAULTS, partition='gpu', time='04:00:00')
```
Confirmed — `multigpu` is gone from this line ✓

### V3 — `generate_diffusivity_scripts()` call passes both new vars

```python
short_gpu_partition = SHORT_GPU_PARTITION,
short_gpu_time      = SHORT_GPU_TIME,
```
Both present in the Cell 4 call ✓

### V4 — `diffusivity_workflow.py` (Task B stash) accepts the parameters

Stash diff confirms signature additions:

```python
def generate_diffusivity_scripts(
    ...
    short_gpu_partition='gpu',
    short_gpu_time='04:00:00',
    short_gpu_cutoff=None,
    ...
)
```
✓ Wired end-to-end — notebook → generator → generated script

### V5 — `GPU_PARTITION='multigpu'` (NVT long runs) unchanged

```python
GPU_PARTITION = 'multigpu'   # NVT long runs
```
NVT partition correctly preserved at `multigpu` ✓

---

## 5. Issues found

None. All three changes match the plan exactly.

---

## 6. Additional checks

| Check | Result |
| --- | --- |
| `short_gpu_cutoff` auto-derivation: `04:00:00 − 5 min = 03:55:00` | ✓ Confirmed in `diffusivity_workflow.py` stash diff |
| Config print block updated to show both GPU vars | ✓ Added `print(f'  SHORT_GPU_PARTITION : {SHORT_GPU_PARTITION}  (NPT + min)')` |
| `PERM_GPU_SLURM` still `multigpu` (Part 2 Hop A/B, vibrations) | Out of Task D scope — Part 2 short GPU jobs are ~2–4h and would also benefit from `gpu`. Flag for future task. |

---

## 7. Status

**VERIFIED.** `calculation/pipeline.ipynb` committed in this task.
`diffusivity_workflow.py` integration committed with Task B.

---

## 8. Follow-up — 2026-07-06 (branch `fix_redundant_min_npt_run`)

The flagged item in Section 6 (`PERM_GPU_SLURM` still `multigpu`) has been resolved, and the
partition scheme has since been rebalanced by job *category* rather than by pipeline *part* —
"short GPU job" turned out to be too coarse a bucket once NPT (a real chained MD run) and
one-shot CG minimisations were both being routed through it.

**Current scheme (see `Project2_surface_labeling/PIPELINE_GUIDE.md` Section 3 and
`Project2_surface_labeling/multiscale_permeation_plan.md` Section 2b–2e for full detail):**

- `NEB_GPU_SLURM` → `gpu`/8h, but **narrowed in scope**: it now covers *only* slab surface
  relaxation (Section A of `neb_workflow.py`) — the four-phase chained MD run
  (min → heat → NVT → quench). It no longer covers Section C's FS-minimisation.
- A new `MIN_SLURM` config (`sharing`/1h) covers every quick one-shot CG minimisation across
  the whole pipeline: H₂ reference energy, H₂*/H* adsorption energy (Section B), and FS-min
  before NEB (Section C) — previously lumped in with `NEB_GPU_SLURM`.
- `PERM_GPU_SLURM` (this section's flagged TODO) is now `sharing`/1h, same reasoning as
  `MIN_SLURM` — Hop A/B FS-minimisation is a one-shot CG minimisation, not real dynamics.
- Diffusivity's `SHORT_GPU_PARTITION` (bare-min + bulk+H-min) stayed on the "quick minimisation"
  category (now `sharing`/1h); NPT was split out onto its own `gpu`/8h config, since it is real
  chained MD like slab surface relaxation, not a one-shot minimisation — see
  `audits/task_B_audit.md` Section 8.

**Why this reclassification, not just a name change:** the original rule ("short jobs → `gpu`,
long NVT → `multigpu`") conflated two different things that happen to both take a few hours:
real molecular dynamics (surface relaxation, NPT) and one-shot energy minimisation (H₂
reference, adsorption energies, FS-min, bare-bulk min). The former needs GPU wall time
commensurate with actually integrating a trajectory; the latter converges in minutes to an
hour regardless of GPU speed. Splitting by category rather than by duration keeps `gpu`
reserved for jobs that need it and frees `sharing` capacity for the much larger number of
quick minimisations the multi-metal pipeline now runs (11 structures × several minimisation
types each).

# Task A3 Audit — `write_npt_restart_script()` in lammps_script.py

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `models/lammps_script.py`

---

## 1. Goal

`write_chained_slurm_job` (used for NVT) requires two LAMMPS inputs: a fresh-run script and a
restart script. NPT had no restart script — only a fresh-run script existed. If an NPT job hit the
8h `gpu` wall time after the heating stage, there was no way to resume Stage 2 (production) from
the `*_after_heat.restart` checkpoint. Task A3 adds that missing function.

---

## 2. Mechanism

New function `write_npt_restart_script()` in `models/lammps_script.py`. It mirrors
`write_npt_script` Stage 2 exactly with three differences:

1. `read_restart <restart_file>` replaces `read_data` — thermostat state and velocities are
   restored from the checkpoint; no `velocity all create` command needed.
2. `fix boxdump` uses `append yes` — box dimensions are appended to the existing
   `npt_boxdims_{T}K.dat` rather than overwriting it.
3. Only Stage 2 runs — the `heat_steps` parameter is absent; Stage 1 temperature ramp is omitted.

The function is called from `diffusivity_workflow.py` (Task B) to write the restart input
alongside the fresh-run input, so `write_chained_slurm_job` has both scripts wired up.

---

## 3. Impact

| Area | Effect |
| --- | --- |
| `models/lammps_script.py` | Additive — new function, nothing changed |
| `diffusivity_workflow.py` (Task B) | Imports and calls the new function for each NPT temperature |
| NPT restart behavior | If NPT hits wall time after heating, job chain detects `*_after_heat.restart` and runs Stage 2 continuation |
| Existing callers | None — `write_npt_script` is unchanged |

---

## 4. Verification results

### A3-V1 — Function importable

`from models.lammps_script import write_npt_restart_script` succeeds ✓

### A3-V2 — All expected parameters present

Parameters: `restart_file`, `npt_dump`, `out_path`, `pair_style`, `mace_model`, `pair_suffix`,
`elem_str`, `target_t`, `timestep`, `thermo_damp`, `baro_damp`, `npt_steps`, `dump_every` — all
present ✓

### A3-V3 — Generated script is correct Stage 2

Generated script called with a fake restart path and verified:

- `read_restart` present ✓
- `append yes` in box-dim dump fix ✓
- No `velocity all create` (Stage 2 does not reinitialise velocities) ✓
- NPT fix present ✓

### A3-V4 — Stage 1 correctly excluded

- `heat_steps` parameter absent from signature ✓
- `restart_file` parameter present ✓

---

## 5. Issues found

None.

---

## 6. Status

**VERIFIED.** `models/lammps_script.py` committed with this task.

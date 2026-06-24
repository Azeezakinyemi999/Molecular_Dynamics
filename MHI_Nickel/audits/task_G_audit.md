# Task G Audit — Thermally-averaged lattice parameter from NPT dump

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `models/structure.py`, `models/diffusivity_workflow.py` (import line)

---

## 1. Goal

The previous code called `get_lattice_parameter(npt_final_paths[T])`, which reads a
single LAMMPS structure snapshot — one thermal frame that could be an outlier.

The NPT run already writes box dimensions every `NPT_DUMP_EVERY` steps to
`npt_boxdims_{T}K.dat` via `fix ave/time`. Averaging the last N Lx values from
that file gives a thermally-stable `a0(T)`.

`a0(T)` is downstream-critical: it is used to construct the supercell for hydrogen
insertion in Phase 1b. A single-frame outlier (e.g., ±0.01 Å fluctuation) would
produce a mismatched H insertion lattice at every temperature.

---

## 2. Mechanism

### `models/structure.py` — new function `get_lattice_parameter_from_dump`

```
get_lattice_parameter_from_dump(dump_file, n_last=50, supercell_reps=(5,5,5))
```

- Opens `npt_boxdims_{T}K.dat`
- Skips lines starting with `#` or blank
- Parses column 1 (Lx) from each data row
- Takes the last `n_last` values, averages them
- Returns `lx_mean / supercell_reps[0]` in Å

`n_last=50` at `dump_every=100` steps = last 5000 NPT production steps averaged.

**Why divide by `supercell_reps[0]`:** Lx in the dump is the full supercell box
length (5 × a0 for a 5×5×5 supercell), not the primitive cell parameter.

### `models/diffusivity_workflow.py` — import fix

Removed dead import of `get_lattice_parameter` (old single-frame function).
Kept `get_lattice_parameter_from_dump`. The call site at the `a0` extraction step
reconstructs the dump path as:

```python
_npt_dump_T = os.path.join(dirs['structures'], f'npt_boxdims_{T}K.dat')
a0_T = get_lattice_parameter_from_dump(_npt_dump_T, n_last=50)
```

---

## 3. Impact

| Area | Effect |
|---|---|
| `structure.py` | Additive — new function, nothing removed |
| Generated `diffusivity_run.py` | Imports new function; call site uses dump path |
| `a0(T)` value | More stable than single-frame; typically within ~0.001 Å of single frame for converged NPT |
| H insertion supercell | Built from averaged `a0(T)` → correct lattice spacing |
| `supercell_reps` consistency | Both `write_npt_script` and `get_lattice_parameter_from_dump` default to 5 — consistent |

---

## 4. Verification results

### V1 — Correct average with `n_last=2`

Synthetic dump with Lx values `[17.60, 17.65, 17.70]`, `n_last=2`:
- Expected: `(17.65 + 17.70) / 2 / 5 = 3.535000 Å`
- Got: `3.535000 Å` ✓

### V2 — Edge case: fewer rows than `n_last`

Same file, `n_last=50` (only 3 rows available):
- Expected: `(17.60 + 17.65 + 17.70) / 3 / 5 = 3.530000 Å`
- Got: `3.530000 Å` ✓

### V3 — Template import

`diffusivity_workflow.py` body template imports `get_lattice_parameter_from_dump`,
not the old `get_lattice_parameter` ✓

Dead import of `get_lattice_parameter` removed ✓

### V4 — Call site

Call uses `_npt_dump_T` (dump path), not `npt_final_paths[T]` (structure file) ✓

### V5 — Path scope

`_npt_dump_T` is reconstructed in the H-insertion loop (separate from the NPT
submission loop) using the same formula — paths are identical ✓

---

## 5. Issues found and fixed

| Issue | Fix |
|---|---|
| Dead import of `get_lattice_parameter` left in generated template | Removed from import line in `diffusivity_workflow.py` |

---

## 6. Status

**VERIFIED.** `structure.py` committed in this task. `diffusivity_workflow.py`
integration (import fix + call site) will be committed together with Tasks B and F,
which own the majority of changes to that file.

# Task H Audit — Surface preparation quench + graph robustness

**Branch:** `feature/multiscale-permeation-pipeline`
**Date:** 2026-06-24
**Files changed:** `models/lammps_script.py`, `models/surface_graph.py`, `models/subsurface_graph.py`

---

## 1. Goal

Three related improvements driven by the same root problem: downstream pipeline
steps (site enumeration, H/H₂ placement, layer assignment) were operating on the
raw last NVT frame — a thermally-displaced, non-minimum-energy snapshot. This
caused:

1. The NEB endpoint slab geometry to be a finite-temperature snapshot rather than
   a 0K local minimum, inconsistent with NEB being a static calculation
2. `surface_graph.py` site enumeration to use a z-threshold cutoff that broke
   whenever one atom was thermally elevated, causing real top-layer atoms to be
   excluded or second-layer atoms to be included
3. `subsurface_graph.py` Voronoi tessellation to run on raw thermal positions,
   shifting interstitial site locations and causing spurious oct/tet sites

---

## 2. Changes

### H1 — Phase 4 quench in `write_surface_relaxation_script` (`lammps_script.py`)

#### Before

The 3-stage surface relaxation protocol:
1. CG minimize (crash prevention — removes atomic clashes from as-cut slab)
2. NVE + velocity rescale (heat to 300K)
3. NVT at 300K (surface reconstruction, relaxation, rumpling)

The final `write_data {slab_relaxed}` came at the end of Phase 3, writing the raw
last NVT frame. All downstream code (`run_phase3_site_enumeration`,
`add_adsorbate`, NEB endpoint construction) consumed this thermally-displaced
structure.

#### After

A 4th stage added after NVT:

4. **CG minimize the NVT snapshot** (anneal-and-quench protocol)

Phase 3 now writes to `{stem}_phase3_nvt.lammps` (new intermediate name).
Phase 4 CG minimize writes to `{slab_relaxed}` — the same filename all
downstream callers already expect. Zero changes to any caller.

**Key distinction between Phase 1 and Phase 4:**

| | Phase 1 | Phase 4 |
|---|---|---|
| Purpose | Make slab MD-stable (crash prevention) | Get proper 0K reference for NEB |
| Input | Raw as-cut slab | NVT-annealed slab |
| Output basin | Nearest local minimum to as-cut geometry | Bottom of the thermally-explored basin |
| Science | Engineering necessity | Scientifically motivated |

The structural information from the NVT (surface reconstruction, element
segregation, interlayer spacing) is preserved — Phase 4 only removes the random
thermal noise on top of it. This is the standard **anneal-and-quench** protocol
in computational surface science.

**New output files per run:**
- `{stem}_phase3_nvt.lammps` — raw NVT last frame (intermediate, kept for reference)
- `{stem}_phase4_quench_traj.lammpstrj` — quench minimization trajectory
- `{stem}_phase4_quench.restart` — post-quench binary restart
- `{slab_relaxed}` — quenched structure (same name as before, now better quality)

**Results block extended:**

```
z_top_before_Ang       : ...   (before Phase 1 minimize)
z_top_after_min_Ang    : ...   (after Phase 1 minimize)
surface_contraction    : ...   (Phase 1 delta)
z_top_after_nvt_Ang    : ...   (after Phase 3 NVT)
z_top_after_quench_Ang : ...   (after Phase 4 quench)   ← new
pe_final_eV            : ...   (after Phase 4 quench)
```

### H2 — Phase 4 quench in `write_surface_relaxation_restart_script` (`lammps_script.py`)

The restart script (used when the NVT job hits SLURM wall time and chains) was
updated identically: Phase 3 NVT continuation now writes to `_phase3_nvt.lammps`,
and Phase 4 quench follows before writing to `{slab_relaxed}`.

New parameters added to the restart function signature (matching fresh script
defaults): `etol=0.0`, `ftol=1e-6`, `maxiter=10000`, `maxeval=100000`.

### H3 — Rank-based top-layer selection in `surface_graph.py`

#### Before

```python
z_max     = pos[:, 2].max()            # single-atom maximum
top_mask  = pos[:, 2] > (z_max - 1.8) # z-threshold
z_top3    = z_max - 5.0
top3_mask = pos[:, 2] > z_top3        # z-threshold
```

**Failure modes:**

- One thermally-elevated outlier atom inflates `z_max`
- Cutoff `z_max - 1.8` shifts upward → real top-layer atoms that are thermally
  depressed fall below the cutoff and are excluded from the graph
- Or: mean-based z_max lowers the cutoff → second-layer atoms get included

Test confirmed: with outlier+2.0 Å + std=0.10 Å, naive approach **missed 29 out
of 30** top-layer atoms. With outlier+1.0 Å, a mean-based fix contaminated **3
second-layer atoms** into the top-layer group.

#### After

```python
_atoms_per_layer = max(1, len(pos) // 12)
_sorted_by_z     = np.argsort(pos[:, 2])

# Top 1 layer — always exactly atoms_per_layer atoms
_top1_idx        = _sorted_by_z[-_atoms_per_layer:]
top_mask         = np.zeros(len(pos), dtype=bool)
top_mask[_top1_idx] = True

# Top 3 layers for ACAT — always exactly 3 × atoms_per_layer atoms
_top3_idx = _sorted_by_z[-(3 * _atoms_per_layer):]
top3_mask = np.zeros(len(pos), dtype=bool)
top3_mask[_top3_idx] = True

# z_max for graph metadata: mean z of top layer (robust estimate)
z_max = float(np.mean(pos[_top1_idx, 2]))
```

This is the same rank-based philosophy already used by `subsurface_graph.py`
(`_identify_layers`). Now both modules are consistent.

**Guarantee:** always selects exactly `atoms_per_layer` top-layer atoms regardless
of any thermal displacement, outlier atoms, or surface reconstruction — as long as
thermal displacements are small compared to layer spacing (2.08 Å >> 0.3 Å at
500K), which is always true for this system.

**Default path updated:**
```python
# Before (stale reference to old notebook structure):
_DEFAULT_SLAB_PATH = 'structures/notebook05-adsorption-energy/7/clean_slab_reminimized.lammps'

# After (current pipeline output):
_DEFAULT_SLAB_PATH = 'calculation/phase2_relax/relaxed_slab.lammps'
```

The `reminimized` in the old default was evidence that this function was always
intended to receive a quenched slab — the pipeline just never produced one until H1.

### H4 — Z-layer smoothing before Voronoi in `subsurface_graph.py`

#### Before

```python
vor = Voronoi(replicated_positions)   # raw thermal positions
# ...
diffs = replicated_positions - v       # raw positions for proximity filter
```

**Failure mode:** thermal displacements of 0.2–0.3 Å shift Voronoi vertices,
creating spurious interstitial sites, merging real distinct sites across the
0.75 Å clustering tolerance, or misclassifying oct sites as tet.

#### After

```python
# Project each atom's z to its layer mean (x, y unchanged)
_layer_map, _layer_z = _identify_layers(metal_positions)
smoothed_positions = metal_positions.copy()
for _i in range(len(smoothed_positions)):
    smoothed_positions[_i, 2] = _layer_z[_layer_map[_i]]

vor = Voronoi(replicated_smoothed)     # clean Voronoi on smoothed positions
# ...
diffs = replicated_raw - v             # raw positions for proximity filter
```

**Why only z:** x, y displacements are random with no systematic bias — averaging
them would destroy local chemical environment information. z displacements have a
clear layer structure: each atom's expected z is its layer mean. Projecting to
that mean removes thermal noise without losing chemical diversity.

**Why keep raw positions for proximity filter (Step 4):** the `min_dist` filter
discards Voronoi vertices that are too close to any atom. Using real (raw) atom
positions here ensures we don't accidentally keep a vertex that sits inside a
thermally-displaced atom's excluded zone.

Test confirmed: raw z-std per layer of 0.19 Å (300K) and 0.28 Å (500K) reduced
to exactly 0.000 Å after smoothing.

---

## 3. Impact

| Area | Before | After |
|---|---|---|
| NEB endpoint geometry | Last NVT frame (thermally displaced, forces ≠ 0) | Quenched local minimum (forces ≈ 0) |
| Adsorption site positions | From distorted surface | From clean minimized surface |
| E_clean reference energy | From thermally-elevated state | From true basin minimum |
| H/H₂ placement height | Ill-defined on corrugated surface | Well-defined on flat minimum |
| Layer/freeze assignment | z-distribution overlap between layers | Clean layer separation |
| `surface_graph` top-layer selection | z-threshold (fragile) | Rank-based (robust) |
| `surface_graph` ACAT input | z-threshold top-3 (fragile) | Rank-based top-3 (robust) |
| `subsurface_graph` Voronoi | Raw thermal positions | z-smoothed positions |
| Callers of all three functions | Unchanged | Unchanged |

---

## 4. Verification results

### H-V1 — Syntax

`models/lammps_script.py`, `models/surface_graph.py`, `models/subsurface_graph.py`
all parse cleanly ✓

### H-V2 — Phase 4 script generation (fresh)

| Check | Result |
|---|---|
| Phase 1/2/3/4 all present | ✓ |
| Phase 3 writes `_phase3_nvt.lammps` (intermediate) | ✓ |
| Phase 4 writes `_phase4_quench.restart` and `_phase4_quench_traj.lammpstrj` | ✓ |
| `z_top_after_nvt` captured between Phase 3 and 4 | ✓ |
| `z_top_after_quench_Ang` in results block | ✓ |
| Final `write_data` = `relaxed_slab.lammps` (unchanged name) | ✓ |
| `unfix freeze` comes after Phase 4 | ✓ |

### H-V3 — Phase 4 script generation (restart)

Same checks as H-V2 plus: NVT trajectory uses `append yes` ✓

### H-V4 — Rank-based selection (7 scenarios)

| Scenario | Top atoms selected | Missed | Contaminated |
|---|---|---|---|
| Clean slab | 30 | 0 | 0 |
| 300K noise (std=0.20 Å) | 30 | 0 | 0 |
| 500K noise (std=0.30 Å) | 30 | 0 | 0 |
| outlier+0.5 Å + std=0.20 | 30 | 0 | 0 |
| outlier+1.0 Å + depression-1.0 Å + std=0.20 | 30 | 0 | 0 |
| outlier+2.0 Å + depression-0.2 Å + std=0.10 | 30 | 0 | 0 |
| outlier+3.5 Å + std=0.20 (extreme) | 30 | 0 | 0 |

Previous z-threshold approach missed 29/30 atoms in the `outlier+2.0` scenario ✓

### H-V5 — Z-smoothing

- 300K (std=0.20 Å): raw z-std = 0.1893 Å → smoothed = 0.000000 Å ✓
- 500K (std=0.30 Å): raw z-std = 0.2839 Å → smoothed = 0.000000 Å ✓

---

## 5. Issues found

None.

---

## 6. Status

**VERIFIED.** `models/lammps_script.py`, `models/surface_graph.py`,
`models/subsurface_graph.py` ready to commit.

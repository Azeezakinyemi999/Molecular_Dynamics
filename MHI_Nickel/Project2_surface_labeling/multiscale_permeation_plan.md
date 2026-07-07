# pipeline.ipynb — Complete Workflow Reference

## Broad Overview

`pipeline.ipynb` is the single entry point for the entire multiscale H permeation simulation campaign, run across **every material in `INPUT_STRUCTURES` in one pass** — not just one metal. It does not run any simulation itself — instead, it generates, per structure, a `neb_run_{stem}.py` and `permeation_run_{stem}.py`, plus one shared `diffusivity_run.py` and one shared `pipeline_run.py`/`pipeline_run.sh`, that when executed on the cluster carry out all computation end-to-end.

The pipeline is built around three physically distinct problems. Each must be solved independently before the final permeability can be computed:

- **Part 1 (NEB)**: What is the energy landscape for H on each material's surface? Specifically: what is the barrier for an H₂ molecule to dissociate into two H* atoms, how strongly do those H* atoms adsorb to the surface, and how much energy does it cost for an H* atom to hop between neighboring surface sites?
- **Part 3 (Diffusivity MD)**: Once H is inside the bulk metal lattice, how fast does it diffuse? What is D(T) — the H diffusivity as a function of temperature — and what Arrhenius parameters D₀ and E_D describe it?
- **Part 2 (Permeation KMC)**: Given the surface energetics from Part 1 and the bulk diffusivity from Part 3, what is the steady-state H permeability Φ(T) of the membrane at each temperature?

**Why Parts 1 and 3 run in parallel:** They share no inputs with each other. Part 1 only needs each material's bulk supercell to build a surface slab. Part 3 only needs the bulk structures to run bulk MD. Neither reads anything the other produces. `pipeline_run.py` therefore launches every metal's `neb_run_{stem}.py` and the shared `diffusivity_run.py` simultaneously.

**Why Part 2 must wait for both:** Part 2 needs the surface dissociation barriers and rate constants from Part 1 (to set up the KMC surface reaction events) and the bulk diffusivity Arrhenius parameters from Part 3 (to set the bulk H transport rate). If either is missing, Part 2 cannot run. `pipeline_run.py` enforces this by blocking on every launched process before submitting any Part 2 script.

**Why Part 2 runs sequentially across metals (not in parallel like Parts 1/3):** Each `permeation_run_{stem}.py` submits its own SLURM arrays for Hop A/Hop B NEB and can saturate the `short`/`sharing` partitions on its own. Part 2 is also far cheaper per metal than Parts 1/3, so running metals one at a time costs little wall-clock time while avoiding partition oversubscription.

```
  pipeline.ipynb
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Cell 2 : shared config + classify_metal() → METAL_CONFIGS (one per      │
  │          structure in INPUT_STRUCTURES) + shared E_H2_GAS (cached once) │
  │ Cell 3 : write_neb_run_script()        → neb_run_{stem}.py  (per metal) │
  │ Cell 4 : generate_diffusivity_scripts() → diffusivity_run.py (shared)   │
  │ Cell 5 : generate_permeation_scripts()  → permeation_run_{stem}.py (per │
  │          metal)                                                         │
  │ Cell 6 : pre-pipeline bulk_min_{stem}.lammps scripts (Part A, manual    │
  │          sbatch) + generate_pipeline_scripts()/_sh() → pipeline_run.py  │
  │          + pipeline_run.sh (Part B)                                     │
  └─────────────────────────────────────────────────────────────────────────┘
                            │
                sbatch pipeline_run.sh
                            │
            ┌───────────────┴────────────────────────────┐
            ▼                                             ▼
   neb_run_{stem}.py  (× N metals, parallel)        diffusivity_run.py
   (Part 1 — NEB)                                   (Part 3 — MD, shared
   surface barriers                                  across all metals × n_H)
            │                                             │
            └───────────────────┬─────────────────────────┘
                                 ▼  (wait for every launched process)
                    permeation_run_{stem}.py  (× N metals, one at a time)
                    (Part 2 — KMC)
                    TST rates → KMC → Φ(T)  per metal, per n_H
                                 │
                                 ▼
                Final permeability Φ(T) = Φ₀ exp(−E_Φ / k_B T)  per metal
```

The notebook is designed so that **Cell 2 is the only cell you need to edit** before a new run. All downstream scripts are regenerated from those variables when you re-run Cells 3–6.

---

## Section 1 — Cell 1: Imports & Setup

Cell 1 adds the repository root to `sys.path` so that the `models/` package is importable, then imports everything the notebook needs from six modules. None of these imports trigger any computation — they just load Python functions into memory.

### `models/config.py`

The central constants file. Every hardware path, element table, SLURM default, and simulation physics constant lives here. The reason these live in a single file rather than in the notebooks themselves is so that any change (e.g., updating the conda environment path or adding a new element type) propagates everywhere automatically when the notebooks are re-run.

Imported names and their roles:

| Imported name | What it is | Where it is used |
|--------------|-----------|-----------------|
| `MACE_MODEL_ASE` | Path to the `.model` file for MACE-MH-1 (used by ASE/Python) | Part 1 Phase E vibrations; Part 2 Phases 3 vibrations |
| `MACE_MODEL_LAMMPS` | Path to the `.pt` file for MACE-MH-1 (used by LAMMPS) | All generated LAMMPS scripts (pair_coeff) |
| `LAMMPS_CMD` | Full path to the compiled LAMMPS binary | Written into every generated SLURM `.sh` script |
| `KOKKOS_FLAGS` | List of CLI flags to enable Kokkos GPU acceleration (`-k on g 1 -sf kk ...`) | Appended to the LAMMPS call in every GPU job |
| `PAIR_STYLE`, `PAIR_SUFFIX` | LAMMPS pair_style keyword and suffix for MACE-MH-1 | Written into every generated LAMMPS input file |
| `ELEM_STR_7`, `E2T_7`, `MASSES_7` | 7-metal element string/type-map/mass-table (`Al B C Cr Fe Mo Ni H`) | Alloy and pure metal structures |
| `ELEM_STR_10`, `E2T_10`, `MASSES_10` | 10-type table adding O (`Al B C Cr Fe Mo Ni O H`) | Oxide structures only |
| `SLURM_DEFAULTS` | Dict with ntasks, cpus_per_task, GPU type, conda env path, CUDA version, LD_LIBRARY_PATH | Base SLURM config that Part-specific configs update with partition/time |
| `BASE_DIR` | Root project directory on the cluster (`/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel`, Northeastern Explorer) | Used to construct all output paths — `WORK_DIR = os.path.join(BASE_DIR, 'calculation')` |
| `N_REPLICAS`, `SPRING_CONST`, `NEB_FTOL` | Default NEB images (18), spring constant (1.0 eV/Å²), force tolerance (0.05 eV/Å) | Fallback defaults if Cell 2 variables are not overridden |

### `models/create_slurm.py`

Three functions that abstract all SLURM interaction:

- `write_slurm_job(script_path, lammps_input, slurm_cfg)` — writes a complete `.sh` SLURM script. It handles the `#SBATCH` header lines, module loads, conda activation, LD_LIBRARY_PATH setup, and the LAMMPS command with KOKKOS flags. The caller only needs to provide paths and the SLURM config dict.
- `submit_slurm_job(sh_path, dry_run=False)` — calls `sbatch sh_path`, captures stdout to extract the job ID (e.g., `Submitted batch job 12345`), and returns the integer job ID. If `dry_run=True`, it prints the command and returns a fake ID without actually submitting.
- `wait_for_jobs(job_ids, poll_interval=60)` — polls `squeue -j {ids}` every 60 seconds. When a job ID no longer appears in `squeue` output, it has finished (either completed or failed). The function returns when ALL provided job IDs have disappeared from the queue.

### `models/lammps_script.py`

Low-level LAMMPS input file writers. These functions write the actual text of LAMMPS `.lammps` scripts. The key function used by Cell 6 is `write_minimization_script()`, which generates a CG energy minimisation script for each material's bare bulk supercell.

### `models/neb_workflow.py`

Contains `write_neb_run_script()` — the function that generates one `neb_run_{stem}.py` orchestrator per metal. It takes all Part 1 configuration values plus that metal's `elem_str`/`e2t`/`masses`/`metal_type`/`slab_seed` as arguments and writes them as constants into the header of the generated script. Also contains `calculate_ref_adsorbate_energy()`, called once in Cell 2 (not per metal) to produce the shared `E_H2_GAS` reference.

### `models/diffusivity_workflow.py`

Contains `generate_diffusivity_scripts()` — generates a single `diffusivity_run.py` with all Part 3 configuration baked in, including a `metal_table` dict keyed by structure stem (so the one script can handle every structure's element table), and `generate_orchestrator_sh()` for its SLURM wrapper.

### `models/permeation_workflow.py`

Contains `generate_permeation_scripts()` — generates one `permeation_run_{stem}.py` per metal with that metal's Part 2 configuration baked in, and `generate_permeation_sh()` for its SLURM wrapper.

### `models/pipeline_workflow.py`

Contains `generate_pipeline_scripts()` — generates `pipeline_run.py` (the master multi-metal orchestrator that launches every metal's NEB script plus the shared diffusivity script in parallel, waits for all of them, then runs every metal's permeation script sequentially) and `generate_pipeline_sh()` — generates `pipeline_run.sh` (the SLURM script that runs `pipeline_run.py` itself as a long-running job on the cluster).

### `models/structure.py`

Contains `is_pure_bcc_structure()`, imported directly into `pipeline.ipynb` Cell 2 to build the surface-step skip list (see Section 2a below) — pure BCC structures are excluded from Parts 1/2 because the surface/subsurface site-finding code is untested for BCC geometry (GitHub #6).

---

## Section 2 — Cell 2: Configuration Hub

Cell 2 is the single place where all simulation parameters are set. Every variable defined here is injected directly into the generated scripts' header blocks as top-level Python constants, so changing a value here and re-running Cells 3–6 updates every downstream script at once — no searching through multiple files.

### 2a — Shared Settings

These variables are used across more than one pipeline part.

| Variable | Value | Purpose |
|----------|-------|---------|
| `WORK_DIR` | `os.path.join(BASE_DIR, 'calculation')` | Root directory on the Explorer cluster. All generated scripts use this as the base path for all inputs and outputs. |
| `TEMPERATURES` | `[400, 600, 800]` K | The three temperatures at which the full pipeline is evaluated. Part 3 runs NPT and NVT MD at each temperature. Part 2 computes TST rate constants and runs KMC at each temperature. The final Φ(T) is an Arrhenius fit across these points. |
| `NEB_RUN_PY`, `DIFFUSIVITY_RUN_PY`, `PERMEATION_RUN_PY`, `PIPELINE_RUN_PY`, `PIPELINE_RUN_SH` | `{WORK_DIR}/calculation/*.py`/`.sh` | Kept for individual partial-run notebooks (`neb_calculation.ipynb`, etc.); the multi-metal pipeline generates per-stem names instead (`neb_run_{stem}.py`, `permeation_run_{stem}.py`) via Cells 3 and 5. |
| `dry_run` | `True` | Gates every `submit_slurm_job()`/`calculate_ref_adsorbate_energy()` call in Cell 2 onward: scripts are written but not submitted until this is flipped to `False`. |

### 2a′ — Metal Classification and Multi-Metal Config

Before any script generation, Cell 2 classifies every structure in `INPUT_STRUCTURES`:

```python
def classify_metal(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if 'oxide' in stem:
        return 'oxide'
    if any(k in stem for k in ('hastelloy', 'bestsqs', 'sqs', 'alloy')):
        return 'alloy'
    return 'pure'
```

This builds `METAL_CONFIGS`, a list of one dict per structure: `struct_path`, `stem`, `type` (`'alloy'`/`'pure'`/`'oxide'`), and that type's element table (`elem_str`/`e2t`/`masses` — the 7-element `ELEM_STR_7` table for alloy/pure, the 10-element `ELEM_STR_10` table with O for oxide). The current `INPUT_STRUCTURES` list has 11 entries: six Hastelloy N alloy variants (seeds 7, 42, 111, 1234, 12345, plus `bestsqs3`), three pure metals (Al, Fe, Ni), and two oxides (Cr₂O₃, NiO).

**Surface-step skip list:** Parts 1 (surface NEB) and 2 (permeation) are validated for FCC(111) only. A `skip_surface` flag is computed per structure:

```python
SKIP_OXIDE_STEMS = {
    'Ni_oxide_supercell',   # GitHub #5: NiO(111) primitive-cell Miller indices
                            # give the polar Tasker-III termination
}
```

- `Ni_oxide_supercell` is skipped (`skip_reason = 'polar oxide termination (GitHub #5)'`).
- Any pure structure where `is_pure_bcc_structure(struct_path)` is true is skipped (`skip_reason = 'BCC surface/subsurface untested (GitHub #6)'`).
- Skipped structures still flow through Part 3 (bulk diffusivity) unaffected — Cells 3 and 5 simply exclude them from the NEB/permeation script-generation loops.

**Shared H₂ reference energy:** `E_H2_GAS` is computed **once** and reused by every metal, rather than each metal computing its own copy (previously each of the N metals running in parallel via `pipeline_run.py` independently submitted its own redundant ref-H₂ job):

```python
E_H2_GAS = calculate_ref_adsorbate_energy(
    adsorbate='H2', outdir=os.path.join(WORK_DIR, 'adsorption', 'ref_energies'),
    ..., slurm_opts=MIN_SLURM, dry_run=dry_run,
)
```

With `dry_run=True`: if `adsorption/ref_energies/h2_ref_energy.json` already exists, `E_H2_GAS` is read from it immediately. If not, an `h2_ref.sh` script is written (but never submitted, even with `dry_run=False`) and the function returns `None` — submit `h2_ref.sh` yourself with `sbatch`, wait for it, then re-run Cell 2; it detects the completed log and populates the cache without resubmitting. Every metal's `neb_run_{stem}.py` reads this same shared cache at real run time and raises `RuntimeError` with clear instructions if it isn't there yet.

### 2b — Part 1 (NEB) Settings

**Input files:** each metal's `neb_run_{stem}.py` reads `structures/bulk_min_{stem}.lammps` — the energy-minimised supercell from which the surface slab is carved. Cell 6 Part A creates any missing ones.

**Surface geometry:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `MILLER` | `(1, 1, 1)` | Miller index of the surface facet to expose. The (1,1,1) facet of an FCC alloy is the close-packed surface — the most stable and most relevant for H₂ dissociation studies. |
| `LAYERS` | `12` | Number of atomic layers in the slab for alloy/pure structures. **Ignored for oxides** — one repeat unit of an oxide primitive cell contains several atomic planes, so `build_slab()` auto-derives the repeat count to match a ~22 Å target thickness instead (see `audits/oxide_support_plan.md`). |
| `VACUUM` | `15.0` Å | Vacuum gap added above the topmost surface atom. 15 Å is sufficient to prevent an H₂ molecule placed at 2.5 Å above the surface from interacting with its periodic image through the vacuum. |
| `LAT_REPEAT` | `(5, 6)` | Repeats the slab unit cell 5 times in x and 6 times in y for alloy/pure structures. A large lateral cell is needed so that H* pairs at `SEP_MAX` separation do not interact with their periodic images. |
| `Z_FREEZE_CUTOFF` | `None` | `None` = auto: freeze the bottom 1/3 of slab thickness. Only the top ~2/3 of layers are free to move during the four-phase relaxation (see Phase A below). |

**H₂ dissociation search parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `SEP_MIN` | `2.5` Å | Minimum H–H separation allowed for H* pairs scanned in Phase B. Pairs closer than this are physically unrealistic (H–H repulsion becomes very large). |
| `SEP_MAX` | `5.0` Å | Maximum H–H separation for H* pairs. Pairs farther apart than this are unlikely to be directly connected by a single NEB transition — they would require multiple hops. |
| `GRAPH_DIST_MIN` | `2` | Minimum surface-graph hop distance between an H₂* site and a candidate H* pair for the pair to be considered reachable in one dissociation event. |
| `PROX_CUTOFF` | `5.0` Å | Sites separated by less than this are considered equivalent and merged during site enumeration. |

**NEB physics parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `N_IMAGES` | `18` (`N_REPLICAS`) | Number of intermediate images placed between IS and FS along the NEB chain. |
| `SPRING_K` | `1.0` eV/Å² (`SPRING_CONST`) | Spring constant coupling adjacent NEB images. |
| `NEB_FTOL_VAL` | `0.05` eV/Å (`NEB_FTOL`) | CI-NEB convergence criterion: the maximum force component perpendicular to the NEB chain on any image must fall below this value. |
| `SURF_TIMESTEP_PS` | `0.0005` ps | Timestep (0.5 fs) for the surface relaxation MD in Phase A. |
| `H_HEIGHT` | `1.5` Å | Height above a surface site at which an H atom is initially placed before minimisation. |

**SLURM resource configurations — split by job category, not by pipeline part:**

| Variable | Partition | Wall time | Used for |
|----------|-----------|-----------|-------|
| `NEB_GPU_SLURM` | `gpu` | 8 h | Section A only — slab surface relaxation's four-phase chained MD run (min → heat → NVT → quench). This is real, long-running dynamics, so it gets its own GPU wall-time budget separate from the quick minimisations below. |
| `MIN_SLURM` | `sharing` | 1 h | Every quick one-shot CG minimisation shared across the whole pipeline: the shared H₂ reference energy, H₂*/H* adsorption energy (Section B), and FS-min before NEB (Section C). |
| `NEB_CPU_SLURM` | `short` | 12 h, 16 CPUs | No GPU. NEB in LAMMPS uses the `neb` command which runs MPI-parallel across images — each MPI rank owns one image. GPU acceleration is not supported for multi-replica NEB in LAMMPS, so this runs on CPUs only. |
| `NEB_VIB_SLURM` | `short` | 6 h, 8 CPUs | Vibrational-frequency (Hessian) jobs for Phase E — also CPU-only. |

### 2c — Part 3 (Diffusivity MD) Settings

**Input structures:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `INPUT_STRUCTURES` | 11 `.lammps` file paths | The same list used to build `METAL_CONFIGS` in 2a′ — six Hastelloy N alloy variants, three pure metals, two oxides. Diffusivity has no surface-step skip list: every structure runs through Part 3 regardless of `skip_surface`. |
| `N_H_VALUES` | `[1, 3, 5, 10]` | Number of H atoms inserted into each supercell. Running multiple concentrations lets you check whether D depends on concentration (dilute-limit D(T) should be concentration-independent). |
| `DIFF_TEMPS` | `= TEMPERATURES` | Diffusivity shares the same temperature grid as Parts 1 and 2 rather than defining its own. |

**NVT MD parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `TIMESTEP_PS` | `0.0005` ps | NVT MD timestep (0.5 fs). |
| `TAU_T_PS` | `0.1` ps | Nosé-Hoover thermostat coupling time. Loose enough not to artificially suppress H diffusion events (which occur on ps–ns timescales). |
| `N_EQUIL_STEPS` | `2,000,000` | NVT equilibration steps = 1 ns at 0.5 fs/step. Discarded entirely before MSD collection begins. |
| `N_PROD_STEPS` | `5,000,000` | NVT production steps = 2.5 ns. The MSD is measured over these 2.5 ns. |
| `THERMO_EVERY`, `DUMP_EVERY` | `1000` | Print/dump frequency (0.5 ps) for monitoring and trajectory visualization. |
| `RESTART_EVERY` | `100,000` | Binary restart file every 50 ps — enables automatic job chaining if a SLURM job hits its wall time. |

**NPT lattice equilibration parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `NPT_HEAT_STEPS` | `100,000` | Steps for the NPT heating ramp from 10 K to the target temperature (50 ps). |
| `NPT_PROD_STEPS` | `500,000` | NPT production steps at constant T and P = 0 (250 ps) — long enough for box dimensions to converge well past the ~1–10 ps lattice-expansion timescale. |
| `NPT_BARO_DAMP` | `1.0` ps | Barostat coupling time for the Parrinello-Rahman barostat. |
| `NPT_DUMP_EVERY` | `1000` | Write box dimensions (Lx, Ly, Lz) every 1000 steps (0.5 ps) during NPT production — 500 data points per NPT run, thermally averaged (last 50 frames by default) via `get_lattice_parameter_from_dump()` rather than trusting a single snapshot (`audits/task_G_audit.md`). |

**Bulk minimisation tolerances:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `MIN_ETOL` | `0.0` eV | Energy convergence criterion (disabled) — only the force criterion governs convergence. |
| `MIN_FTOL` | `1e-8` eV/Å | Extremely tight force convergence, appropriate for producing a reference structure reused by many downstream simulations. |
| `MIN_MAXITER` / `MIN_MAXEVAL` | `50,000` / `500,000` | Safety caps on CG iterations / force evaluations. |

**SLURM — split by real-MD vs quick-minimisation, same rule as Part 1:**

| Variable | Partition | Wall time | Used for |
|----------|-----------|-----------|----------|
| `GPU_PARTITION` (`GPU_TIME`) | `multigpu` | 24 h | NVT production (Phase 2) — the longest single job type in the whole pipeline. |
| `NPT_GPU_PARTITION` | `gpu` | 8 h | NPT lattice equilibration (Phase 1b) — real chained MD, same category as slab surface relaxation, kept on its own budget separate from the quick minimisations below. |
| `SHORT_GPU_PARTITION` | `sharing` | 1 h | Bare-bulk minimisation (Phase 1a) and bulk+H re-minimisation (Phase 1b) only — one-shot CG minimisations. |

**Phase 1a and Phase 1b-NPT run once per structure, not once per `(structure, n_H)` pair:** bare-bulk minimisation and NPT lattice equilibration do not depend on H concentration, so both are hoisted out of the `N_H_VALUES` loop into a shared per-structure block with its own failure isolation — previously they were redundantly recomputed for every `n_H` value. See "Four phases of `diffusivity_run.py`" in Section 4 and the follow-up note in `audits/task_B_audit.md`.

### 2d — Part 2 (Permeation KMC) Settings

**Input paths — produced by Parts 1 and 3, per metal:**

| Variable | What it points to | Why Part 2 needs it |
|----------|------------------|---------------------|
| `relaxed_slab_path` | `slabs/{stem}/phase2_relax/relaxed_slab.lammps` from Part 1 | Provides the surface geometry for building the Hop A NEB initial state |
| `surface_sites_json` | `slabs/{stem}/phase3_sites/surface_sites.json` from Part 1 | Provides site coordinates so Part 2 knows where to place H* for the Hop A IS |
| `phase2_h_dir` | `adsorption/{stem}/phase2_h/results` from Part 1 | Directory of relaxed H* structures; Part 2 picks the lowest-energy one as the Hop A NEB IS |
| `sub_neb_dir`, `vib_dir`, `results_dir` | `neb_subsurface/{stem}/`, `vibrations/{stem}/`, `results/{stem}/` | Per-metal internal directories for Hop A/B NEB inputs, vibration files, and rate/KMC/permeability output |

**Thermodynamic variables — set to `None` at notebook time, auto-filled at runtime:**

| Variable | Initial value | What it becomes at runtime |
|----------|--------------|--------------------------|
| `DH_DISS_EV` | `None` | ΔH_diss (eV), read from the lowest-barrier entry in that metal's `neb/ranked_barriers.json` |
| `DH_ENTRY_EV` | `None` | ΔH_entry (eV), read from that metal's converged Hop A NEB result |

**Diffusivity variables — loaded per `(stem, n_H)`, not once per metal:**

Each `permeation_run_{stem}.py` loads its own diffusivity fit from `results/{stem}_{n_h}H/diffusivity_arrhenius.json` at runtime, for every `n_H` in `N_H_VALUES` — there is no shared placeholder. If a given `(stem, n_H)`'s fit file is missing or invalid, that H-concentration is **skipped entirely** rather than substituted with a fake D₀/E_D. `permeation_status.json` records which `n_H` were skipped and why; the script exits non-zero only if zero H-concentrations produced a permeability result.

**Pressure sweep and KMC grid:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `P_VALS_PA` | `list(np.logspace(-5, 6, 40))` — 40 log-spaced points, 1×10⁻⁵ to 1×10⁶ Pa | Spans ultra-high vacuum to 10 bar, covering the full experimental range for H permeation measurements. |
| `NX`, `NY` | `40`, `40` | 1600 surface sites and 1600 subsurface sites (3200 total) — large enough to capture cooperative coverage effects and avoid finite-size artifacts. |
| `SEED` | `42` | Integer random seed for the BKL KMC random number generator, and for the alloy composition assignment on the KMC grid. |
| `KMC_MAX_STEPS` | `500,000` | Maximum KMC events per (T, P) point; the simulation stops early if steady state is reached first. |

**Membrane geometry:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `L_M` | `1×10⁻³` m | Membrane thickness (1 mm). |
| `A0_M` | `3.52×10⁻¹⁰` m | Fallback lattice parameter (3.52 Å, literature Ni FCC) used only if `lattice_params_vs_T.json` is not available. |

**SLURM — same category rule again:**

| Variable | Partition | Wall time | Used for |
|----------|-----------|-----------|----------|
| `PERM_GPU_SLURM` | `sharing` | 1 h | Hop A/B FS-minimisation — one-shot CG minimisations. |
| `PERM_NEB_SLURM` | `short` | 12 h, 16 CPUs | Hop A/B CI-NEB arrays — same reasoning as Part 1's `NEB_CPU_SLURM`. |
| `PERM_VIB_SLURM` | `short` | 6 h, 8 CPUs | Hop A/B vibrational-frequency jobs. |

### 2e — Pipeline SLURM

`PIPE_*` variables define the SLURM job that runs `pipeline_run.py` itself — the master orchestrator that waits for every metal's Part 1 and the shared Part 3 to complete, then runs every metal's Part 2:
- **Partition**: `west` — a long-running CPU partition (30-day maximum) since `pipeline_run.py` does no computation, only orchestration
- **CPUs**: 4 — enough for Python and occasional file parsing
- **RAM**: 16 GB
- **Wall time**: 30 days — this job sits alive for however long the combined campaign across all metals takes

---

## Section 3 — Cell 3: Part 1 — NEB Surface Barrier Workflow (per metal)

Cell 3 loops over `METAL_CONFIGS`, skipping any structure with `skip_surface=True`, and calls `write_neb_run_script()` once per remaining metal. Each call writes a complete, self-contained `neb_run_{stem}.py` with that metal's configuration baked into its header, including `metal_type` and a per-metal `slab_seed` derived from the numeric suffix in the stem (`Hastelloy_N_42_supercell` → seed 42; `bestsqs3` → seed 3; falls back to 7 if no digit is found in the stem).

### What `write_neb_run_script()` does at notebook execution time

It constructs the text of `neb_run_{stem}.py` as a Python f-string, substituting all configuration variables (including `metal_type`, `slab_seed`, that metal's `elem_str`/`e2t`/`masses`, and `min_slurm_cfg`) into the script header. The script body calls `orchestrate_full_neb_workflow(...)` with those header constants as arguments. Nothing is actually executed in the notebook at this point.

### Phase A — Slab Preparation

**Goal**: Produce a well-relaxed surface slab that represents the physical surface at the simulation temperature, and identify where on that surface H atoms (and H₂ molecules) will preferentially sit.

1. **Build slab from bulk**: `build_slab()` reads `structures/bulk_min_{stem}.lammps` and routes on `metal_type`:
   - **alloy**: Ni-FCC geometry template; element symbols randomly shuffled from the bulk composition fractions, seeded by `slab_seed`.
   - **pure**: correct FCC or BCC geometry from the crystal-structure map; every site is the single element, no shuffle.
   - **oxide**: primitive cell extracted via spglib from the minimised bulk supercell; stoichiometry preserved exactly; `LAYERS` is ignored — the repeat count is auto-derived to match `oxide_target_thickness` (default 22 Å, matching the metal slabs' thickness).

   The `surface()` cleave adds a 15 Å vacuum region and repeats laterally by `LAT_REPEAT` (alloy/pure only). This produces a slab with periodic boundary conditions in x and y, with a vacuum gap in z.

2. **Freeze bottom layers**: Atoms with z below `Z_FREEZE_CUTOFF` (auto = bottom 1/3 of slab thickness if `None`) are assigned to a LAMMPS `freeze` group, excluded from force integration. The frozen layers serve as a rigid bulk reference.

3. **Four-phase surface relaxation** (real chained MD, submitted on `gpu`/8 h): unlike a single NVT run, the slab goes through:
   - **Phase 1 — CG minimisation**: initial relaxation to remove large forces from the freshly-cut surface.
   - **Phase 2 — Thermal anneal**: short NVT ramp (~5 ps) to a representative surface temperature.
   - **Phase 3 — NVT production**: longer NVT hold (~50 ps) at constant T with `SURF_TIMESTEP_PS` (0.5 fs) timestep, allowing surface atoms to settle into their thermally relaxed positions.
   - **Phase 4 — Quench**: a second CG minimisation on the final NVT frame, removing residual thermal noise before the structure is used as input to Phase B. This quench step was added after review found the original 3-phase protocol (min → heat → NVT) left detectably non-zero forces in the "relaxed" slab.

   This whole sequence runs as one chained SLURM job (`write_chained_slurm_job`, restart-file based) so it survives past the 8 h wall time if needed.

4. **Surface site enumeration**: For alloy/pure structures, ACAT identifies all symmetry-inequivalent hollow, bridge, and atop sites. For oxide structures (`metal_type='oxide'`), a purpose-built enumerator instead produces ontop sites (every top-plane atom) and M–O bridge midpoints, since ACAT cannot describe oxide surfaces. Sites separated by less than `PROX_CUTOFF` are merged. The result maps each site label to its fractional coordinates in the slab cell.

5. **Outputs**:
   - `slabs/{stem}/phase2_relax/relaxed_slab.lammps` — relaxed slab (used by Phase B and by Part 2 Hop A NEB)
   - `slabs/{stem}/phase3_sites/surface_sites.json` — site labels mapped to fractional coordinates (used by Phase B and Part 2)

### Phase B — H₂* Adsorption and H* Pair Scanning

**Goal**: Generate the structures needed as initial states (IS) and final states (FS) for all NEB calculations in Phase C, using the `sharing`/1 h partition for every one-shot minimisation in this phase.

**Two distinct energy references are used here, and it is critical to understand which is which:**

> **E_H2_GAS**: The MACE total energy of a free H₂ molecule in vacuum, computed once (Cell 2, Section 2a′) and shared by every metal. A **fixed thermodynamic reference** used only for computing adsorption energies and reaction enthalpies relative to the gas phase. It is **not** a structural input to any NEB calculation.

> **H₂\* structure**: An H₂ molecule placed above a surface site and CG-minimised (on `sharing`). This gives the energy of H₂ molecularly adsorbed on the surface. This structure IS the NEB initial state for dissociation NEB calculations.

**Step 1 — H₂\* minimisation (produces NEB IS for dissociation NEB):** For each surface site, an H₂ molecule is placed at height `H_HEIGHT` above the site and CG-minimised together with the slab.

**Step 2 — H\* pair scanning (produces NEB FS for dissociation NEB, and IS/FS for surface diffusion NEB):** For each surface site, the H₂ molecule from Step 1 is split into two separate H atoms. Every symmetry-distinct pair of surface sites where the H–H distance satisfies `SEP_MIN` ≤ d(H,H) ≤ `SEP_MAX` (2.5–5.0 Å) is selected as a candidate dissociated H* pair and CG-minimised.

**Step 3 — Adsorption energy calculation and ranking:**

```
E_ads(site_i) = E(slab + H* at site_i) − E(bare slab) − ½ × E_H2_GAS
```

A negative E_ads means the H* state is lower in energy than H in the gas phase. The most negative E_ads site is the thermodynamically preferred H adsorption site.

**Outputs**:
- `adsorption/{stem}/phase2_h/results/h_atom_{site}_relaxed.lammps` — one LAMMPS data file per H* site (used by Part 2 Hop A NEB as the IS)
- `adsorption/{stem}/adsorption_energies.json` — E_ads for every H* site, sorted from most stable to least stable

### Phase C — NEB Job Generation and Submission

For each NEB transition (dissociation H₂*→2H*, or surface diffusion H*→H*):
1. An FS-minimisation script is written and submitted on `sharing`/1 h.
2. A CI-NEB script is written; all CI-NEB jobs are collected into a single CPU SLURM array (`neb_array.sh`) and submitted on `short`/12 h, 16 CPUs per array element.

### Phase D — Barrier Parsing

For each completed NEB, the LAMMPS log is parsed for the highest-energy image (TS). Forward barrier ΔE_fwd = E(TS) − E(IS); reverse barrier ΔE_rev = E(TS) − E(FS); reaction energy ΔH = E(FS) − E(IS). All transitions are sorted by ΔE_fwd and written to `neb/{stem}/ranked_barriers.json`. This step runs locally (no SLURM) once the arrays finish.

### Phase E — Vibrational Frequencies and TST Rate Constants

Submitted on `short`/6 h, 8 CPUs per job. For each NEB transition, ASE's `Vibrations` class (MACE as the force engine) computes normal-mode frequencies at IS and TS. The Vineyard prefactor and ZPE correction combine into ZPE-corrected TST rate constants at each `T` in `TEMPERATURES` = [400, 600, 800] K. See Section 8 for the full rate-assembly derivation (unchanged physics, still current).

**Outputs**:
- `neb/{stem}/diss_vib_rates.json` — k_diss(T) and k_des(T) at each temperature, plus ν_Vineyard, ΔE_ZPE
- `neb/{stem}/diff_vib_rates.json` — k_diff(T) at each temperature for surface diffusion transitions
- `vibrations/{stem}/` — ASE vibration files

---

## Section 4 — Cell 4: Part 3 — Bulk H Diffusivity Workflow (shared script, all metals)

Cell 4 calls `generate_diffusivity_scripts()` once, producing a single `diffusivity_run.py` that internally loops over every `(structure, n_H)` combination in `INPUT_STRUCTURES × N_H_VALUES` (44 combinations currently: 11 structures × 4 concentrations). A `metal_table` dict, keyed by structure stem, carries each structure's `elem_str`/`e2t`/`masses` so the one script can handle alloy, pure, and oxide inputs alike.

### Phase 1a — Bare Bulk Minimisation (once per structure)

**Goal**: Produce the lowest-energy T=0 K structure of each bare bulk supercell — the starting point for all NPT runs.

This phase, along with Phase 1b's NPT step, runs **once per structure**, hoisted out of the `N_H_VALUES` loop into a shared block that precedes it — since neither depends on H concentration, recomputing them per `n_H` (as an earlier version of this script did) wasted GPU time four-fold. The shared block has its own try/except: if it fails, the failure is recorded once and **all** `n_H` values for that structure are skipped (rather than crashing the whole run), via a module-level `_FAILURES` list written to `diffusivity_failures.json` at the end.

1. `write_minimization_script()` generates a LAMMPS CG minimisation input (`MIN_ETOL`=0.0, `MIN_FTOL`=1e-8 eV/Å, `MIN_MAXITER`=50,000, `MIN_MAXEVAL`=500,000).
2. Submitted on `sharing`/1 h (`SHORT_GPU_PARTITION`); `wait_for_jobs()` blocks until LAMMPS reports convergence.
3. **Output**: `structures/bulk_min_{struct_stem}.lammps` — reused directly by Phase 1b, and it is the same path Part 1's `neb_run_{stem}.py` reads as `BULK_MIN_PATH`.

### Phase 1b — NPT Lattice Equilibration (once per structure, per temperature)

**Goal**: Find the thermally-equilibrated lattice parameter a₀(T) at each of the three target temperatures — necessary because the lattice expands with temperature, and using the 0 K lattice parameter at 800 K would impose unphysical compressive strain.

All temperature runs are submitted as independent, chained SLURM GPU jobs on `gpu`/8 h (`NPT_GPU_PARTITION`) — its own partition/time budget, separate from the quick minimisations, since NPT is real chained MD like the surface relaxation in Part 1.

**NPT MD sequence for each temperature:**

1. **Heating stage**: linear ramp from 10 K to T_target over `NPT_HEAT_STEPS` = 100,000 steps = 50 ps. Parrinello-Rahman barostat maintains P = 0 bar throughout (`NPT_BARO_DAMP` = 1.0 ps).
2. **NPT production stage**: held at T_target, P = 0 for `NPT_PROD_STEPS` = 500,000 steps = 250 ps. Every `NPT_DUMP_EVERY` = 1000 steps (0.5 ps), box dimensions are appended to `npt_boxdims_{T}K.dat` — 500 snapshots over 250 ps.
3. **a₀(T) extraction**: `get_lattice_parameter_from_dump()` averages the last 50 Lx values (by default) from `npt_boxdims_{T}K.dat`, dividing by the supercell repeat count — more stable than trusting a single frame (`audits/task_G_audit.md`).
4. **H atom insertion** (per `(structure, n_H, T)` — this part IS repeated per `n_H`): `n_H` hydrogen atoms are placed at octahedral interstitial sites of the equilibrated cell, maximising the minimum H–H distance. A CG minimisation (`MIN_FTOL` = 1e-8 eV/Å) finds the bulk+H ground state at the T-corrected cell dimensions, submitted on `sharing`/1 h.

**Shared outputs**:
- `results/{struct_stem}/lattice_params_vs_T.json` — a₀(T) per structure, written once and reused across every `n_H` for that structure.
- `structures/{T}K/bulk_min_h_{struct_stem}.lammps` (per `n_H`) — the T-corrected bulk cell with n_H hydrogen atoms.

### Phase 2 — NVT MD (per `(structure, n_H, T)`, submitted on `multigpu`/24 h)

**Goal**: Run long NVT MD trajectories to measure the H mean-square displacement (MSD), from which D(T) = ⟨|r(t)|²⟩ / 6t is extracted.

Each job uses SLURM job-chaining (`write_chained_slurm_job`): if LAMMPS hits its wall time rather than completing normally, a continuation job automatically restarts from the last checkpoint.

1. **Equilibration (excluded from MSD)**: NVT Nosé-Hoover (`TAU_T_PS` = 0.1 ps) for `N_EQUIL_STEPS` = 2,000,000 steps = 1 ns. No MSD data is written during equilibration.
2. **Production (MSD collected)**: `N_PROD_STEPS` = 5,000,000 steps = 2.5 ns. MSD printed to `msd_{T}K.dat` every `DUMP_EVERY` = 1000 steps (0.5 ps).
3. **Restart checkpoints**: binary restart every `RESTART_EVERY` = 100,000 steps (50 ps) enables the chaining described above.

### Phase 3 — Post-processing: D(T) and Arrhenius Fit

1. **MSD data parsing**: identifies the diffusive (Einstein) regime via the local slope of log(MSD) vs. log(t).
2. **Einstein relation**: D(T) = ⟨|r(t)|²⟩ / (6t), weighted least-squares fit over the diffusive regime.
3. **Arrhenius fit**: D(T) = D₀ exp(−E_D / k_B T) across all temperatures in `TEMPERATURES`.
4. **Output**: `results/{struct_stem}_{n_h}H/diffusivity_arrhenius.json` — one file per `(structure, n_H)` pair: D₀, E_D, D(T) at each temperature, R² for both fits.

Part 2 reads this file per `(stem, n_H)` at runtime (see Section 2d) — a missing or invalid fit skips that H-concentration rather than substituting a placeholder.

---

## Section 5 — Cell 5: Part 2 — Permeation Workflow (per metal)

Cell 5 loops over `METAL_CONFIGS`, skipping structures with `skip_surface=True`, and calls `generate_permeation_scripts()` once per remaining metal, producing `permeation_run_{stem}.py`. This script runs the complete six-phase permeation calculation for that metal, across every `n_H` in `N_H_VALUES`.

### What `generate_permeation_scripts()` does at notebook execution time

Like the other generators, it writes a Python script with header constants baked in, including `metal_type` and that metal's element table. `DH_DISS_EV` and `DH_ENTRY_EV` are written as `None` placeholders; at runtime, before any NEB or KMC, the script loads the JSON output files from that metal's own Part 1 run and replaces these with actual computed numbers. D₀/E_D, by contrast, are loaded **fresh per `n_H`** inside the main loop (see Section 2d) — not patched once at the top of the script.

### Phase 1 — Hop A NEB (H* surface → H subsurface-1)

**Goal**: Compute the barrier for H to move from its preferred surface adsorption site into the first subsurface interstitial site — the surface-to-bulk entry step.

1. **IS construction**: the lowest-energy single-H* structure from that metal's `adsorption/{stem}/phase2_h/results/` directory.
2. **FS construction**: the first subsurface interstitial site (built via `build_subsurface_graph`, `metal_type`-aware — octahedral/tetrahedral classification for alloy/pure via rank-based layers, generic "interstitial" classification for oxide via gap-based layers), using the T-appropriate a₀ from `lattice_params_vs_T.json`.
3. **NEB run**: 18-image CI-NEB, submitted as a CPU array on `short`/12 h (FS-min on `sharing`/1 h first).
4. **Energy extraction**: ΔE_entry, ΔE_exit, ΔH_entry (stored, auto-fills `DH_ENTRY_EV`).
5. **Output**: `neb_subsurface/{stem}/hopa/ranked_hopa.json`. Guarded by an existence check on `hopa_jobs.json` so a restarted run skips Hop A if already submitted (`audits/task_F_audit.md`).

### Phase 2 — Hop B NEB (H subsurface-1 → subsurface-2)

**Goal**: Compute the bulk migration barrier ΔE_mig between adjacent interstitial sites — a consistency check against Part 3's E_D.

The sub1↔sub2 connectivity this phase depends on was previously missing from `subsurface_graph.py` for every metal (a dead-path bug where Hop B silently found zero neighbors and skipped); it is now built explicitly via periodic xy proximity between sub1 and sub2 layers (`audits/oxide_support_plan.md`). Same NEB/FS-min partition split as Hop A.

**Output**: `neb_subsurface/{stem}/hopb/ranked_hopb.json`, guarded the same way as Hop A.

### Phase 3 — Vibrational Frequencies for Hop A and Hop B

Submitted on `short`/6 h, 8 CPUs. Same ZPE/Vineyard procedure as Part 1 Phase E, applied to the Hop A and Hop B IS/TS structures. `DH_DISS_EV` is also auto-extracted here from that metal's own `neb/{stem}/ranked_barriers.json`.

**Outputs**: `vibrations/{stem}/hopa_is_vib.json`, `hopa_ts_vib.json`, `hopb_is_vib.json`, `hopb_ts_vib.json`; partial update to `results/{stem}/rate_dict_T{T}K.json`.

### Phase 4 — TST Rate Constants: Complete Assembly

Assembles all six elementary rate constants per temperature (see Section 8 for the full formula table — unchanged physics). The bulk drainage rate `k_drain(T) = D(T) / dx²` now uses D₀/E_D loaded **per `n_H`** from `results/{stem}_{n_h}H/diffusivity_arrhenius.json`, so `rate_dict_T{T}K.json` is effectively computed once per `(stem, n_H, T)` inside the main loop, not once per `(stem, T)`.

**Output**: `results/{stem}/rate_dict_T{T}K.json` per `n_H` iteration (path includes the `n_H`-specific results subdirectory in practice — see the generated script for the exact join).

### Phase 5 — KMC Pressure Sweep

Guarded per-temperature by an existence check on `permeation_sweep_T{T}K.json` (`audits/task_F_audit.md`). Runs the BKL KMC algorithm on the 40×40 dual-layer grid (see Section 8/original event catalog below — physics unchanged) at each of the 40 `P_VALS_PA` points.

**Output**: `results/{stem}/permeation_sweep_T{T}K.json`.

### Phase 6 — Permeability Calculation

Three cross-validated Φ(T) routes (geometric, TST detailed-balance, KMC-empirical), Arrhenius fit, Richardson-Sieverts flux — unchanged physics, see Section 8.

**Outputs**: `results/{stem}/permeability_T{T}K.json`, `results/{stem}/solubility_arrhenius_kmc.json`.

### Success tracking across `n_H` values

`permeation_run_{stem}.py` maintains a `_PERM_STATUS` dict across the `n_H` loop (`stem`, `n_h_requested`, `phase6_ready`, `n_h_skipped`, `permeability_written`), written to `results/{stem}/permeation_status.json` at the end. The script exits non-zero only if **zero** H-concentrations produced a permeability result — a single missing diffusivity fit for one `n_H` no longer silently aborts the whole metal's Part 2 run, nor does it get treated as a false "success" if it was the only `n_H` attempted.

---

## Section 6 — Cell 6: Pre-Pipeline Minimisation + Pipeline Orchestration

### Part A — Pre-pipeline bulk minimisation (per metal, manual submission)

Part 1 needs `structures/bulk_min_{stem}.lammps` to exist before `neb_run_{stem}.py` can run (it reads this file to build the surface slab in Phase A); Part 3 generates its own copy internally and does not depend on this pre-step.

For each metal in `METAL_CONFIGS`, if `bulk_min_{stem}.lammps` does not already exist, Cell 6 Part A writes (but does **not** submit) a CG minimisation script and its SLURM wrapper (`sharing`/1 h). It prints the list of `sbatch` commands you need to run yourself, and waits for none of them — this is a deliberate change from an earlier version of this cell that submitted and blocked on the minimisation directly inside the notebook. Submitting scripts only, rather than submitting-and-waiting, keeps the notebook non-blocking when there are 11 structures to minimise instead of one, and lets you run all 11 pre-pipeline minimisations in parallel on the cluster rather than serially inside the notebook process.

Wait for all printed `sbatch` commands to finish before running Part B with `dry_run=False`.

### Part B — `generate_pipeline_scripts()` → `pipeline_run.py`

`metals_list` is built by filtering `METAL_CONFIGS` down to structures with `skip_surface=False`, pairing each stem with its already-generated `neb_run_{stem}.py` and `permeation_run_{stem}.py` paths (from Cells 3 and 5). This function writes the master Python orchestrator. When `pipeline_run.py` runs on the cluster (inside its own SLURM job), it executes:

1. **Launch every metal's Part 1 in parallel**: `subprocess.Popen(['python', neb_run_py])` for each metal in `metals_list`.
2. **Launch the shared Part 3 in parallel**: `subprocess.Popen(['python', DIFFUSIVITY_RUN_PY])` simultaneously — no dependency on any Part 1 script.
3. **Wait for everything launched in steps 1–2**: `proc.wait()` on every process; each non-zero return code is recorded but does not stop the others.
4. **Run every metal's Part 2 sequentially**: `subprocess.run(['python', permeation_run_py])` one metal at a time (not `Popen`) — see the "Why Part 2 runs sequentially" note in the Broad Overview. A non-zero return code from any metal's permeation run causes `pipeline_run.py` to exit 1 for that metal without necessarily blocking the rest, depending on the generated script's exact error handling — check the generated script for the current behavior if this matters for your run.
5. **Final summary**: prints a completion summary once all metals' Part 2 scripts have run.

Each `permeation_run_{stem}.py` is itself responsible for reading its own metal's `ranked_barriers.json`/`diss_vib_rates.json` (Part 1 output) and the shared `diffusivity_arrhenius.json` (Part 3 output, per `n_H`) — there is no separate "patch step" in `pipeline_run.py` itself; the auto-extraction described in Section 2d/5 happens inside each permeation script at its own startup.

### `generate_pipeline_sh()` → `pipeline_run.sh`

A minimal SLURM submission script for the `west` partition, 4 CPUs, 16 GB RAM, 30-day wall time:
```
#SBATCH --partition=west
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30-00:00:00        # 30 days

conda activate <mace_env path from SLURM_DEFAULTS>
python pipeline_run.py
```
No GPU is requested — `pipeline_run.py` itself runs no MACE or LAMMPS calculations, only submits and monitors child processes/jobs.

### Dry-run mode

```python
job_id = submit_slurm_job(PIPELINE_RUN_SH, dry_run=True)
```

With `dry_run=True` (the Cell 2 default): prints the `sbatch pipeline_run.sh` command to stdout but does NOT submit. Set `dry_run=False` in Cell 2 (which also gates every SLURM submission upstream, including the shared `E_H2_GAS` computation) once every Part A pre-pipeline minimisation has finished, then re-run Cells 2–6 and submit for real.

---

## Section 7 — Full Data Flow Diagram (single metal; repeats per structure in `METAL_CONFIGS`)

The complete end-to-end dependency map showing every file produced and consumed, for one metal's run through Parts 1–2 plus its share of Part 3. Arrows show data flow direction.

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                   pipeline.ipynb — Cell 2 (configuration)              │
  │  WORK_DIR, TEMPERATURES=[400,600,800], METAL_CONFIGS (classify_metal), │
  │  E_H2_GAS (shared, cached), MILLER, LAYERS, VACUUM, LAT_REPEAT,        │
  │  SEP_MIN/MAX, N_IMAGES, SPRING_K, NEB_FTOL, Z_FREEZE_CUTOFF,           │
  │  SURF_TIMESTEP_PS, INPUT_STRUCTURES, N_H_VALUES=[1,3,5,10],           │
  │  NPT/NVT params, MIN params, P_VALS_PA, NX/NY=40/40, KMC_MAX_STEPS...  │
  └────────────┬──────────────────────┬──────────────────────┬────────────┘
               │ Cell 3 (per metal)   │ Cell 4 (shared)      │ Cell 5 (per metal)
               ▼                      ▼                      ▼
  write_neb_run_script()  generate_diffusivity_scripts()  generate_permeation_scripts()
               │                      │                      │
               ▼                      ▼                      ▼
    neb_run_{stem}.py           diffusivity_run.py     permeation_run_{stem}.py
    (Part 1)                    (Part 3 — all metals)   (Part 2 — waits for that
               │                      │                  metal's Part 1 + shared Part 3)
               │                      │                      ▲
  ─────────── PART 1 CLUSTER EXECUTION (gpu / sharing / short) ──────────  │
               │                      │                      │
    Phase A (gpu, 8h)                 │                      │
  ┌────────────────────────────┐      │                      │
  │Build slab (metal_type)     │      │                      │
  │Freeze btm; 4-phase relax:  │      │                      │
  │  min→anneal→NVT→quench     │      │                      │
  │Site enum (ACAT/oxide enum) │      │                      │
  └────────────────────────────┘      │                      │
       ▼                              │                      │
  slabs/{stem}/phase2_relax/relaxed_slab.lammps ──────────────►(Hop A IS geometry)
  slabs/{stem}/phase3_sites/surface_sites.json  ───────────────►(surface site coords)
       │                              │                      │
    Phase B (sharing, 1h)             │                      │
  ┌───────────────────────────┐       │                      │
  │H₂* min → NEB IS (diss)   │       │                      │
  │H* pair → NEB FS (diss)   │       │                      │
  │H* alone → IS/FS (diff)   │       │                      │
  │E_ads = E(H*)-E(slab)      │       │                      │
  │  - ½×E_H2_GAS (shared)    │       │                      │
  └───────────────────────────┘       │                      │
       ▼                              │                      │
  adsorption/{stem}/phase2_h/results/h_atom_*_relaxed.lammps ──►(Hop A IS)
  adsorption/{stem}/adsorption_energies.json (ranking)         │
       │                              │                      │
    Phase C (sharing FS-min + short CI-NEB array) / Phase D (local, barrier parsing)
  ┌────────────────────────────┐      │                      │
  │Type 1: diss NEB, Type 2: surf diff NEB → ΔE_diss,ΔE_des,ΔH,ΔE_diff│
  └────────────────────────────┘      │                      │
       ▼                              │                      │
  neb/{stem}/ranked_barriers.json ─────────────────────────────►(ΔH_diss auto-fill)
       │                              │                      │
    Phase E (short, 6h)                │                      │
  ┌────────────────────────────┐      │                      │
  │Vibrations at IS and TS → ν_Vineyard, ZPE → k_diss(T), k_des(T), k_surf_diff(T)│
  └────────────────────────────┘      │                      │
       ▼                              │                      │
  neb/{stem}/diss_vib_rates.json ──────────────────────────────►(k_diss, k_des)
  neb/{stem}/diff_vib_rates.json ──────────────────────────────►(k_surf_diff)
                                      │                      │
  ─────────── PART 3 CLUSTER EXECUTION (sharing / gpu / multigpu, all metals × n_H) ──
                                      │                      │
                Phase 1a (sharing, 1h, once per structure)   │
                        ┌─────────────┤                      │
                        │Bare CG min  │                      │
                        └─────────────┘                      │
                              ▼                              │
                         structures/bulk_min_{stem}.lammps    │
                              │                              │
                Phase 1b (gpu 8h NPT once/structure; sharing 1h bulk+H min per n_H)
                        ┌────────────────────┐              │
                        │NPT: heat 10K→T(50ps)│              │
                        │ prod T,P=0 (250ps)  │              │
                        │ dump Lx every 0.5ps │              │
                        │ a₀(T)=avg(last 50)  │              │
                        │ insert n_H H atoms  │              │
                        │ CG min bulk+H       │              │
                        └────────────────────┘              │
                              ▼                              │
                         results/{stem}/lattice_params_vs_T.json ►(a₀(T) for rate calc)
                         structures/{T}K/bulk_min_h_{stem}.lammps│
                              │                              │
                Phase 2 (multigpu, 24h, per structure × n_H × T)
                        ┌────────────────────┐              │
                        │NVT: 1ns equil (no MSD) + 2.5ns prod + MSD│
                        │restart every 50ps, auto job chaining│
                        └────────────────────┘              │
                              ▼                              │
                         msd_{T}K.dat, nvt_{T}K.dump          │
                              │                              │
                           Phase 3 (local Python)             │
                        ┌────────────────────┐              │
                        │MSD → D(T); Arrhenius → D₀,E_D│      │
                        └────────────────────┘              │
                              ▼                              │
                         results/{stem}_{n_h}H/diffusivity_arrhenius.json ►(D₀,E_D per n_H)
                                                             │
  ─────────── each permeation_run_{stem}.py auto-extracts its own metal's ─────
  DH_DISS_EV/DH_ENTRY_EV (from its Part 1 output) + loads D₀/E_D per n_H at startup│
                                                             │
  ─────────── PART 2 CLUSTER EXECUTION (sharing / short, per metal, per n_H) ──
                                                             │
                                              Phase 1 — Hop A NEB
                                         ┌───────────────────────┐
                                         │IS: h_atom lowest-E    │
                                         │FS: H at nearest sub1  │
                                         │NEB → ΔE_entry,ΔE_exit,ΔH_entry│
                                         └───────────────────────┘
                                              ▼
                                         neb_subsurface/{stem}/hopa/ranked_hopa.json
                                              │
                                              Phase 2 — Hop B NEB
                                         ┌───────────────────────┐
                                         │IS: H at sub1, FS: H at sub2 (via fixed sub1↔sub2 edges)│
                                         │NEB → ΔE_mig, ΔH_mig  │
                                         └───────────────────────┘
                                              ▼
                                         neb_subsurface/{stem}/hopb/ranked_hopb.json
                                              │
                                              Phase 3 — Vibrations → k_entry,k_exit,k_mig
                                              Phase 4 — TST rates (all 6 + k_drain=D(T)/dx² per n_H)
                                              ▼
                                         results/{stem}/rate_dict_T{T}K.json
                                              │
                                              Phase 5 — KMC sweep (guarded per-T)
                                         ┌───────────────────────┐
                                         │40×40 dual-layer grid, BKL KMC, 40 pressures → C₀(P,T)│
                                         └───────────────────────┘
                                              ▼
                                         results/{stem}/permeation_sweep_T{T}K.json
                                              │
                                              Phase 6 — permeability (Sieverts fit, Φ_A/B/C, Arrhenius)
                                              ▼
                                         results/{stem}/permeability_T{T}K.json
                                         results/{stem}/solubility_arrhenius_kmc.json
                                         results/{stem}/permeation_status.json  (success tracking, all n_H)
                                              ▼
                                    Φ(T) = Φ₀ exp(−E_Φ / k_B T)   [per metal, per n_H]
                                    E_Φ = E_D (Part 3, per n_H) + ΔH_sol (Parts 1+2 NEBs, per metal)
                                    J = Φ(T) × √P / L  [mol/m²/s]
```

### Key file locations (all relative to `WORK_DIR`)

| File | Produced by | Consumed by | What it contains |
|------|------------|-------------|-----------------|
| `slabs/{stem}/phase2_relax/relaxed_slab.lammps` | Part 1, Phase A | Part 2 Phase 1 (Hop A IS slab) | Relaxed slab for this metal |
| `slabs/{stem}/phase3_sites/surface_sites.json` | Part 1, Phase A | Part 1 Phase B, Part 2 Phase 1 | Site labels → fractional coordinates |
| `adsorption/{stem}/phase2_h/results/h_atom_*_relaxed.lammps` | Part 1, Phase B | Part 1 Phase C (diss NEB FS, diff NEB IS/FS), Part 2 Phase 1 (Hop A IS) | Individual H* at each surface site |
| `adsorption/{stem}/adsorption_energies.json` | Part 1, Phase B | Part 2 Phase 1 (selects lowest-E H* site) | E_ads(site) sorted by stability |
| `neb/{stem}/ranked_barriers.json` | Part 1, Phase D | Part 2 Phase 3 (ΔH_diss auto-fill) | All NEB barriers sorted by ΔE_fwd, for this metal |
| `neb/{stem}/diss_vib_rates.json` | Part 1, Phase E | Part 2 Phase 4 (k_diss, k_des) | k_diss(T), k_des(T), k_diff(T) with Vineyard ν and ZPE |
| `results/{stem}/lattice_params_vs_T.json` | Part 3, Phase 1b (once per structure) | Part 2 Phase 4 (a₀(T)) | a₀(T) in Å at each of the 3 temperatures |
| `results/{stem}_{n_h}H/diffusivity_arrhenius.json` | Part 3, Phase 3 (per `n_H`) | Part 2 Phase 4 (D₀, E_D, per `n_H`) | D₀ (m²/s), E_D (eV), D(T) at 3 T, per H concentration |
| `neb_subsurface/{stem}/hopa/ranked_hopa.json` | Part 2, Phase 1 | Part 2 Phase 3 (ΔE_entry, ΔE_exit, ΔH_entry) | Hop A NEB barriers and reaction energy |
| `neb_subsurface/{stem}/hopb/ranked_hopb.json` | Part 2, Phase 2 | Part 2 Phase 3 (ΔE_mig) | Hop B NEB barrier |
| `results/{stem}/rate_dict_T{T}K.json` | Part 2, Phase 4 | Part 2 Phase 5 (KMC event rates) | All 6 rate constants + k_drain at each T, per `n_H` |
| `results/{stem}/permeation_sweep_T{T}K.json` | Part 2, Phase 5 | Part 2 Phase 6 (Sieverts fit) | C₀(P,T) from KMC at each T and 40 P values |
| `results/{stem}/permeability_T{T}K.json` | Part 2, Phase 6 | Final result | Φ from options A, B, C; Φ₀; E_Φ, per `n_H` |
| `results/{stem}/permeation_status.json` | Part 2, end of run | Diagnostics | Which `n_H` succeeded/were skipped, and why |
| `diffusivity_failures.json` | Part 3, end of run | Diagnostics | Which `(structure, n_H)` combinations failed |

---

## Section 8 — ZPE Corrections, Temperature Dependence, and KMC Rate Assembly

This section is pure physics/rate-theory and is unaffected by the multi-metal/partition changes described above — the formulas below apply identically per metal, per `n_H`, per temperature.

### ZPE corrections: what they capture and what they do not

Phase E (Part 1) and Phase 3 (Part 2) both compute harmonic vibrational frequencies at the IS and TS of every NEB transition using MACE as the force engine. These frequencies feed into two corrections:

**Zero-point energy correction to the barrier:**

```text
ΔE_ZPE = ½ℏ × (∑ωᵢ^IS − ∑ωᵢ^TS_real)
Ea_eff  = Ea_NEB + ΔE_ZPE
```

H is a light atom (mass 1.008 amu). Its vibrational frequencies are high (H–H stretch ~3800 cm⁻¹, H–metal stretch ~800–1200 cm⁻¹), which means large zero-point energies. The ZPE difference between IS (H fully coordinated in its adsorption site) and TS (H at the saddle point, partially uncoordinated) is typically −0.05 to −0.15 eV for H on transition metals — a downward correction that can change rates by a factor of 2–5 at 500 K. ZPE corrections are the dominant quantum correction for light-atom diffusion and cannot be neglected.

**Vineyard prefactor:**

```text
ν_Vineyard = c × (∏ᵢ νᵢ^IS) / (∏ᵢ νᵢ^TS_real)
```

The imaginary TS mode is excluded from the denominator product (it is not a vibration — it is the instability direction). The Vineyard prefactor replaces the assumed default of 10¹³ s⁻¹ with a physically-derived attempt frequency. For H on metals, ν_Vineyard is typically 10¹²–10¹³ s⁻¹.

**What ZPE corrections do NOT capture:**

ZPE corrections are evaluated at 0 K harmonic geometry. They do not capture:

- Finite-temperature entropic contributions to the free-energy barrier (ΔS‡ terms in free-energy TST)
- Anharmonic corrections to vibrational frequencies at high T
- How the saddle-point geometry shifts as the lattice thermally expands across `TEMPERATURES`

The geometric effect of thermal expansion on the barrier is small for metals in this temperature range (typically < 0.02 eV over 400 K). By contrast, ZPE shifts the barrier by 0.05–0.15 eV. ZPE corrections therefore dominate the quantum correction budget, and the residual error from using 0 K NEB geometries on a thermally-relaxed slab is negligible relative to the ZPE correction already applied.

**Conclusion on temperature-dependent slabs:** Using temperature-dependent lattice parameters for the NEB slab would be a half-measure (adsorption IS/FS structures would still be 0 K CG-minimised). The ZPE corrections already capture the most important physical effect. Temperature-dependent slabs are not worth the compute cost for this system.

---

### How `diss_vib_rates.json` is used in the KMC temperature sweep

`build_rate_dict()` in `models/tst_rates.py` computes rates at a specific `T_K` and stores them alongside the temperature-independent quantities (`Ea_zpe`, `Ed_zpe`, `nu`). This value only affects reference numbers printed to screen — it does **not** affect what `neb_run_{stem}.py` writes to `diss_vib_rates.json`.

What `diss_vib_rates.json` contains (written by `neb_run_{stem}.py` Phase E):

```json
{
  "label__elem1+elem2": {
    "pair":   ["Ni", "Mo"],
    "Ea_zpe": 0.312,     ← ZPE-corrected forward barrier (eV) — temperature-independent
    "Ed_zpe": 0.189,     ← ZPE-corrected reverse barrier (eV) — temperature-independent
    "Ea_raw": 0.358,     ← raw NEB barrier (eV)
    "Ed_raw": 0.235,     ← raw NEB reverse barrier (eV)
    "nu":     8.3e12,    ← Vineyard prefactor (s⁻¹) — temperature-independent
    "label":  "..."
  }
}
```

`permeation_run_{stem}.py` reads `Ea_zpe`, `Ed_zpe`, and `nu` from this file and recomputes rates fresh at every temperature in the sweep:

```python
for _T in TEMPERATURES:          # [400, 600, 800] K
    _kBT = KB_EV * _T
    for _pkey, _dv in _diss_vib.items():
        _k_diss[_pkey] = np.exp(-_dv['Ea_zpe'] / _kBT)           # sticking factor
        _k_des[_pkey]  = _dv['nu'] * np.exp(-_dv['Ed_zpe'] / _kBT)  # TST rate (s⁻¹)
```

The reference `T_K` used inside Phase E therefore has no effect on the temperature sweep — it is a documentation/print artifact.

---

### Why k_diss has no Vineyard prefactor

`k_diss` in the rate dict is dimensionally different from `k_des`. The KMC `build_event_list()` function (`models/kmc.py`) documents this explicitly:

```python
# H₂ → 2H* sticking probability (dimensionless Boltzmann factor)
# Multiply by gas_strike_rate(P,T) inside build_event_list.
# Derive as: exp(-Ea_diss_zpe / (kB * T)) from tst_rates output.
'k_diss':  {('Ni', 'Ni'): float, ...}
```

The actual adsorption rate in the KMC is:

```text
R_adsorb = R_strike × k_diss = [Hertz-Knudsen collision rate] × exp(−Ea_ZPE / k_B T)
```

where `R_strike = P / √(2π m_H₂ k_B T) × A_site` is computed inside `build_event_list` from the gas pressure, temperature, and surface site area. The attempt frequency for the forward (adsorption) process comes from gas-phase kinetic theory, not from the Vineyard formula. `k_diss` is therefore a dimensionless thermal activation factor, not a full TST rate constant.

`k_des`, by contrast, is the rate for a surface process (2H* → H₂) that has no gas-phase component. It uses the full TST expression including the Vineyard prefactor:

```text
k_des = ν_Vineyard × exp(−Ed_ZPE / k_B T)   [s⁻¹]
```

This asymmetry — gas kinetic theory for adsorption, harmonic TST for desorption — is the standard treatment for dissociative chemisorption kinetics. It is internally consistent with detailed balance because both rates are derived from the same NEB barrier: at equilibrium, `R_strike × k_diss = k_des` recovers the correct Sieverts equilibrium constant.

---

### The BKL rejection-free KMC algorithm and event catalog

At each KMC step: enumerate all possible events across the 40×40 grid; compute the rate Rₙ for each; advance time by Δt = −ln(u₁)/R_total; select an event with probability Rₙ/R_total; execute it. "Rejection-free" means every step executes exactly one physical event, unlike Metropolis MC which may reject proposed events — essential for simulating rare, low-pressure adsorption events efficiently.

| Event | Rate | Condition |
|-------|------|-----------|
| H₂ adsorption + dissociation at pair (i,j) | R_strike × k_diss(T) | Both sites empty, adjacent |
| H* recombination + desorption | k_des(T) | Both sites occupied, adjacent |
| H* surface hop | k_surf_diff(T) | Occupied → empty neighbor |
| H* entry to subsurface | k_entry(T) | Surface occupied, subsurface empty |
| H_sub exit to surface | k_exit(T) | Subsurface occupied, surface empty |
| H_sub bulk drainage | k_drain(T) = D(T)/dx² | Subsurface occupied |

The drainage event is the model's representation of bulk permeation: once an H_sub atom drains from the subsurface layer, it has effectively entered the bulk membrane and will diffuse through to the other side — valid in the steady-state regime where downstream bulk concentration is approximately zero.

---

### Richardson-Sieverts permeability: three cross-validated routes

All three routes use Φ = D(T) × S(T), where D(T) is taken from Part 3 for the relevant `n_H`. They differ only in how S(T) is computed:

- **Option A — geometric lattice site density**: `S_A(T) = (site density) × exp(−ΔH_sol/k_BT)`, using only geometric and thermodynamic inputs, no KMC output.
- **Option B — TST detailed balance**: `S_B(T) = k_entry(T) × R_strike / k_exit(T)`, using the TST rate constants, not the KMC output — an independent check.
- **Option C — KMC empirical**: `S_C(T) = C₀(P,T)/√P` from the Phase 5 Sieverts fit — the most physically complete but most expensive.

If A, B, and C agree to within ~20%, the Sieverts regime holds and the permeability is robust. Each Φ(T) series is fit to `Φ(T) = Φ₀ exp(−E_Φ/k_BT)`, where `E_Φ = E_D + ΔH_sol` combines the bulk diffusion barrier (Part 3, per `n_H`) and the solution enthalpy (Parts 1+2 NEBs, per metal). The Richardson-Sieverts flux `J = (Φ(T)/L) × (√P_high − √P_low)` is the quantity measured in experimental permeation studies.

---

### Summary: temperature dependence of the complete rate assembly

| Rate constant | Source | Temperature enters via | Notes |
| --- | --- | --- | --- |
| `k_diss(T)` | Part 1 Phase E, ZPE-corrected Ea | Boltzmann factor exp(−Ea_ZPE/kBT) | No ν prefactor — uses gas Hertz-Knudsen flux |
| `k_des(T)` | Part 1 Phase E, ZPE-corrected Ed | ν_Vineyard × exp(−Ed_ZPE/kBT) | Full TST rate (s⁻¹) |
| `k_entry(T)` | Part 2 Phase 3, Hop A ZPE-corrected | ν_A × exp(−ΔE_entry_ZPE/kBT) | Full TST rate (s⁻¹) |
| `k_exit(T)` | Part 2 Phase 3, Hop A ZPE-corrected | ν_A × exp(−ΔE_exit_ZPE/kBT) | Full TST rate (s⁻¹) |
| `k_mig(T)` | Part 2 Phase 3, Hop B ZPE-corrected | ν_B × exp(−ΔE_mig_ZPE/kBT) | Full TST rate (s⁻¹) |
| `k_surf_diff(T)` | Part 1 Phase E, ZPE-corrected | ν_diff × exp(−ΔE_diff_ZPE/kBT) | Full TST rate (s⁻¹) |
| `k_drain(T)` | Part 3 Phase 3 Arrhenius D(T), per `n_H` | D(T) = D₀ exp(−E_D/kBT), then D/dx² | Not from NEB — from MD MSD |

All ZPE-corrected barriers and Vineyard prefactors are stored as temperature-independent constants in JSON files. The KMC recomputes the actual rate constants at each temperature in the sweep from these stored values. No rate pre-computed at a specific temperature is reused at a different temperature.

---

*Document originally generated from `pipeline.ipynb` source — 2026-06-22. Revised 2026-07-06 to reflect the multi-metal (`METAL_CONFIGS`/`classify_metal`) architecture, the current `gpu`/`sharing`/`short`/`multigpu`/`west` partition split, hoisted bare-bulk-min/NPT (once per structure), the shared `E_H2_GAS` cache, per-`(stem, n_H)` diffusivity loading with skip-on-missing, and the Hop B `sub1↔sub2` connectivity fix — see `Molecular_Dynamics/MHI_Nickel/audits/` for the individual task audits underlying each change.*

# pipeline.ipynb — Complete Workflow Reference

## Broad Overview

`pipeline.ipynb` is the single entry point for the entire multiscale H permeation simulation campaign. It does not run any simulation itself — instead, it generates three orchestrator Python scripts and one master SLURM wrapper that, when executed on the cluster, carry out all computation end-to-end.

The pipeline is built around three physically distinct problems. Each must be solved independently before the final permeability can be computed:

- **Part 1 (NEB)**: What is the energy landscape for H on the Hastelloy N (1,1,1) surface? Specifically: what is the barrier for an H₂ molecule to dissociate into two H* atoms, how strongly do those H* atoms adsorb to the surface, and how much energy does it cost for an H* atom to hop between neighboring surface sites?
- **Part 3 (Diffusivity MD)**: Once H is inside the bulk metal lattice, how fast does it diffuse? What is D(T) — the H diffusivity as a function of temperature — and what Arrhenius parameters D₀ and E_D describe it?
- **Part 2 (Permeation KMC)**: Given the surface energetics from Part 1 and the bulk diffusivity from Part 3, what is the steady-state H permeability Φ(T) of the membrane at each temperature?

**Why Parts 1 and 3 run in parallel:** They share no inputs with each other. Part 1 only needs the bulk Hastelloy N structure to build a surface slab. Part 3 only needs the bulk structure to run bulk MD. Neither reads anything the other produces. `pipeline_run.py` therefore submits them simultaneously.

**Why Part 2 must wait for both:** Part 2 needs the surface dissociation barriers and rate constants from Part 1 (to set up the KMC surface reaction events) and the bulk diffusivity Arrhenius parameters from Part 3 (to set the bulk H transport rate). If either is missing, Part 2 cannot run. `pipeline_run.py` enforces this by blocking on both job IDs before submitting Part 2.

```
  pipeline.ipynb
  ┌─────────────────────────────────────────────────────────────────┐
  │ Cell 2 : set all configuration variables (one place)            │
  │ Cell 3 : write_neb_run_script()     → neb_run.py               │
  │ Cell 4 : generate_diffusivity_scripts() → diffusivity_run.py   │
  │ Cell 5 : generate_permeation_scripts()  → permeation_run.py    │
  │ Cell 6 : generate_pipeline_scripts()    → pipeline_run.py      │
  │           generate_pipeline_sh()        → pipeline_run.sh      │
  └─────────────────────────────────────────────────────────────────┘
                            │
                sbatch pipeline_run.sh
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
       neb_run.py                  diffusivity_run.py
       (Part 1 — NEB)              (Part 3 — MD)
       surface barriers            bulk H diffusivity
            │                               │
            └───────────────┬───────────────┘
                            ▼  (wait_for_jobs: BOTH must finish)
                   permeation_run.py
                   (Part 2 — KMC)
                   TST rates → KMC → Φ(T)
                            │
                            ▼
                Final permeability Φ(T) = Φ₀ exp(−E_Φ / k_B T)
```

The notebook is designed so that **Cell 2 is the only cell you need to edit** before a new run. All downstream scripts are regenerated from those variables when you re-run Cells 3–6.

---

## Section 1 — Cell 1: Imports and Setup

Cell 1 adds the repository root to `sys.path` so that the `models/` package is importable, then imports everything the notebook needs from six modules. None of these imports trigger any computation — they just load Python functions into memory.

### `models/config.py`

The central constants file. Every hardware path, element table, SLURM default, and simulation physics constant lives here. The reason these live in a single file rather than in the notebooks themselves is so that any change (e.g., updating the conda environment path or adding a new element type) propagates everywhere automatically when the notebooks are re-run.

Imported names and their roles:

| Imported name | What it is | Where it is used |
|--------------|-----------|-----------------|
| `MACE_MODEL_ASE` | Path to the `.model` file for MACE-MH-1 (used by ASE/Python) | Part 1 Phase E vibrations; Part 2 Phases 1–3 vibrations |
| `MACE_MODEL_LAMMPS` | Path to the `.pt` file for MACE-MH-1 (used by LAMMPS) | All generated LAMMPS scripts (pair_coeff) |
| `LAMMPS_CMD` | Full path to the compiled LAMMPS binary | Written into every generated SLURM `.sh` script |
| `KOKKOS_FLAGS` | List of CLI flags to enable Kokkos GPU acceleration (`-k on g 1 -sf kk ...`) | Appended to the LAMMPS call in every GPU job |
| `PAIR_STYLE`, `PAIR_SUFFIX` | LAMMPS pair_style keyword and suffix for MACE-MH-1 | Written into every generated LAMMPS input file |
| `ELEM_STR_7` | Element string `'Al B C Cr Fe Mo Ni H'` for pair_coeff | Written into every LAMMPS input that uses the 8-element table |
| `SLURM_DEFAULTS` | Dict with ntasks, cpus_per_task, GPU type, conda env path, CUDA version, LD_LIBRARY_PATH | Base SLURM config that Part-specific configs update with partition/time |
| `BASE_DIR` | Root project directory on the cluster | Used to construct all output paths |
| `N_REPLICAS`, `SPRING_CONST`, `NEB_FTOL` | Default NEB images (18), spring constant (1.0 eV/Å²), force tolerance (0.05 eV/Å) | Fallback defaults if Cell 2 variables are not overridden |

### `models/create_slurm.py`

Three functions that abstract all SLURM interaction:

- `write_slurm_job(script_path, lammps_input, slurm_cfg)` — writes a complete `.sh` SLURM script. It handles the `#SBATCH` header lines, module loads, conda activation, LD_LIBRARY_PATH setup, and the LAMMPS command with KOKKOS flags. The caller only needs to provide paths and the SLURM config dict.
- `submit_slurm_job(sh_path, dry_run=False)` — calls `sbatch sh_path`, captures stdout to extract the job ID (e.g., `Submitted batch job 12345`), and returns the integer job ID. If `dry_run=True`, it prints the command and returns a fake ID without actually submitting.
- `wait_for_jobs(job_ids, poll_interval=60)` — polls `squeue -j {ids}` every 60 seconds. When a job ID no longer appears in `squeue` output, it has finished (either completed or failed). The function returns when ALL provided job IDs have disappeared from the queue.

### `models/lammps_script.py`

Low-level LAMMPS input file writers. These functions write the actual text of LAMMPS `.lammps` scripts. The key function used by Cell 6 is `write_minimization_script()`, which generates a CG energy minimisation script for the bare bulk supercell.

### `models/neb_workflow.py`

Contains `write_neb_run_script()` — the one function that generates the entire `neb_run.py` orchestrator. It takes all Part 1 configuration values as arguments and writes them as constants into the header of the generated script.

### `models/diffusivity_workflow.py`

Contains `generate_diffusivity_scripts()` — generates `diffusivity_run.py` with all Part 3 configuration baked in, and `generate_orchestrator_sh()` for its SLURM wrapper.

### `models/permeation_workflow.py`

Contains `generate_permeation_scripts()` — generates `permeation_run.py` with all Part 2 configuration baked in, and `generate_permeation_sh()` for its SLURM wrapper.

### `models/pipeline_workflow.py`

Contains `generate_pipeline_scripts()` — generates `pipeline_run.py` (the master Python orchestrator that submits Parts 1, 3, and 2 in order) and `generate_pipeline_sh()` — generates `pipeline_run.sh` (the SLURM script that runs `pipeline_run.py` itself as a long-running job on the cluster).

---

## Section 2 — Cell 2: Configuration Hub

Cell 2 is the single place where all simulation parameters are set. Every variable defined here is injected directly into the generated scripts' header blocks as top-level Python constants, so changing a value here and re-running Cells 3–6 updates every downstream script at once — no searching through multiple files.

### 2a — Shared Settings

These variables are used across more than one pipeline part.

| Variable | Value | Purpose |
|----------|-------|---------|
| `WORK_DIR` | `/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel` | Root directory on Discovery cluster. All generated scripts use this as the base path for all inputs and outputs. |
| `TEMPERATURES` | `[500, 600, 700, 800, 900]` K | The five temperatures at which the full pipeline is evaluated. Part 3 runs NPT and NVT MD at each temperature. Part 2 computes TST rate constants and runs KMC at each temperature. The final Φ(T) is an Arrhenius fit across these five points. |
| `NEB_RUN_PY` | `{WORK_DIR}/calculation/neb_run.py` | Where Cell 3 writes the Part 1 orchestrator script. |
| `DIFFUSIVITY_RUN_PY` | `{WORK_DIR}/calculation/diffusivity_run.py` | Where Cell 4 writes the Part 3 orchestrator script. |
| `PERMEATION_RUN_PY` | `{WORK_DIR}/calculation/permeation_run.py` | Where Cell 5 writes the Part 2 orchestrator script. |
| `PIPELINE_RUN_PY` | `{WORK_DIR}/calculation/pipeline_run.py` | Where Cell 6 writes the master pipeline orchestrator. |
| `PIPELINE_RUN_SH` | `{WORK_DIR}/calculation/pipeline_run.sh` | Where Cell 6 writes the master SLURM wrapper. |

### 2b — Part 1 (NEB) Settings

**Input files:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `BULK_MIN_PATH` | path to `bulk_min.lammps` | The energy-minimised 5×5×5 Hastelloy N bulk supercell from which the surface slab is carved. Must exist before Part 1 can run — Cell 6 creates it if absent. |
| `E_H2_GAS` | `-6.790499` eV | The MACE total energy of a single H₂ molecule computed in vacuum with no surface present. **This is NOT the energy of H₂ adsorbed on the surface (H₂*).** It is a fixed reference number used exclusively for computing thermodynamic quantities: adsorption energies E_ads and the overall dissociation reaction energy ΔH_diss referenced to the gas phase. It is NOT the NEB initial state energy — the NEB IS is the H₂* molecularly adsorbed structure computed in Phase B. |

**Surface geometry:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `MILLER` | `(1, 1, 1)` | Miller index of the surface facet to expose. The (1,1,1) facet of an FCC alloy is the close-packed surface — the most stable and most relevant for H₂ dissociation studies. |
| `LAYERS` | `12` | Number of atomic layers in the slab. 12 layers (~24 Å thick) is thick enough that the bottom of the slab behaves as bulk, so freezing the bottom layers is a valid constraint. |
| `VACUUM` | `15.0` Å | Vacuum gap added above the topmost surface atom. 15 Å is sufficient to prevent an H₂ molecule placed at 2.5 Å above the surface from interacting with its periodic image through the vacuum. |
| `LAT_REPEAT` | `(5, 6)` | Repeats the slab unit cell 5 times in x and 6 times in y, producing a ~720-atom supercell. A large lateral cell is needed so that H* pairs at `SEP_MAX` (6.0 Å) separation do not interact with their periodic images. |
| `Z_FREEZE_CUTOFF` | `22.115` Å | Atoms with z-coordinate below this value are held fixed during surface relaxation. Only the top ~3 layers (above this cutoff) are free to move. The frozen bottom layers mimic a semi-infinite bulk — they provide the correct restoring forces without allowing the whole slab to drift or tilt. |

**H₂ dissociation search parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `SEP_MIN` | `2.5` Å | Minimum H–H separation allowed for H* pairs scanned in Phase B. Pairs closer than this are physically unrealistic (H–H repulsion becomes very large). |
| `SEP_MAX` | `6.0` Å | Maximum H–H separation for H* pairs. Pairs farther apart than this are unlikely to be directly connected by a single NEB transition — they would require multiple hops. This bounds the NEB transition set to nearest-neighbor and next-nearest-neighbor events. |

**NEB physics parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `N_IMAGES` | `18` | Number of intermediate images placed between IS and FS along the NEB chain. More images give a better-resolved minimum energy path and more accurate TS geometry. 18 is a good compromise between resolution and computational cost for H hops of ~2–6 Å. |
| `SPRING_K` | `1.0` eV/Å² | Spring constant coupling adjacent NEB images. If too weak, images cluster near the IS and FS and leave the TS region poorly sampled. If too strong, images are dragged off the true MEP. 1.0 eV/Å² is standard for surface reactions with MACE potentials. |
| `NEB_FTOL_VAL` | `0.05` eV/Å | CI-NEB convergence criterion: the maximum force component perpendicular to the NEB chain on any image must fall below this value. 0.05 eV/Å gives TS energies converged to ~1 meV for H on metal surfaces. |
| `SURF_TIMESTEP_PS` | `0.0005` ps | Timestep (0.5 fs) for the NVT surface relaxation MD in Phase A. 0.5 fs is the standard safe timestep for MACE/LAMMPS with H present — H has the smallest mass and fastest vibrations (~3000 cm⁻¹ = period ~11 fs). |

**SLURM resource configurations:**

| Variable | Partition | Wall time | Notes |
|----------|-----------|-----------|-------|
| `GPU_SLURM_CFG` | multigpu | 4 h | A100 GPU + 8 CPUs. Used for surface relaxation NVT (Phase A) and all FS minimisation jobs (Phase C). MACE on GPU is ~10–50× faster than CPU for these tasks. |
| `NEB_SLURM_CFG` | short | 12 h, 16 CPUs | No GPU. NEB in LAMMPS uses the `neb` command which runs MPI-parallel across images — each MPI rank owns one image. GPU acceleration is not supported for multi-replica NEB in LAMMPS, so this runs on CPUs only with 16 MPI ranks (one per image, plus 2 extras for CI-NEB head images). |

### 2c — Part 3 (Diffusivity MD) Settings

**Input structures:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `INPUT_STRUCTURES` | list of 9 `.lammps` file paths | The nine Hastelloy N supercell variants: base SQS composition plus Al-rich, Fe-rich, Ni-rich, Mo-rich variants, and `bestsqs3` ordering. Running all nine gives D(T) statistics across compositional disorder. |
| `N_H_VALUES` | `[1, 3, 5, 7, 10]` | Number of H atoms inserted into each supercell. The 5×5×5 supercell has ~875 metal atoms, so n_H = 1,3,5,7,10 corresponds to H concentrations of ~0.001–0.011 at.fraction. Running multiple concentrations lets you check whether D depends on concentration (dilute-limit D(T) should be concentration-independent). |

**NVT MD parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `TIMESTEP_PS` | `0.0005` ps | NVT MD timestep (0.5 fs). Same reasoning as for surface relaxation — H vibrational period ~11 fs, so 0.5 fs gives ~22 steps per fastest vibration. |
| `TAU_T_PS` | `0.1` ps | Nosé-Hoover thermostat coupling time. Controls how tightly the thermostat corrects the kinetic energy back to the target temperature. 0.1 ps is loose enough not to artificially suppress H diffusion events (which occur on ps–ns timescales) while still maintaining temperature control. |
| `N_EQUIL_STEPS` | `2,000,000` | NVT equilibration steps = 1 ns at 0.5 fs/step. The first 1 ns is discarded entirely — it allows H atoms to thermalize from their initial (minimised, T=0) positions into a realistic room-temperature configuration before MSD collection begins. |
| `N_PROD_STEPS` | `5,000,000` | NVT production steps = 2.5 ns. The MSD is measured over these 2.5 ns. For H in metals at 500–900 K, a hop rate of 10⁸–10¹⁰ s⁻¹ means ~250–25000 hops per H atom occur in 2.5 ns, giving good statistics for D extraction. |
| `THERMO_EVERY` | `1000` | Print temperature, energy, pressure to the LAMMPS log every 1000 steps (0.5 ps). Used for monitoring thermal equilibration. |
| `DUMP_EVERY` | `1000` | Write full atom coordinates to the `.dump` file every 1000 steps (0.5 ps). Used for trajectory visualization and post-hoc analysis. |
| `RESTART_EVERY` | `100,000` | Write binary restart file every 100,000 steps (50 ps). These checkpoints enable automatic job chaining: if a SLURM job hits its wall time limit mid-run, the chain script reads the latest restart and continues from there rather than starting over. |

**NPT lattice equilibration parameters:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `NPT_HEAT_STEPS` | `20,000` | Steps for the NPT heating ramp from 10 K to the target temperature (10 ps). Ramping gently avoids thermal shock (large atomic displacements in the first few steps) that can crash the MACE potential. |
| `NPT_PROD_STEPS` | `200,000` | NPT production steps at constant T and P = 0 (100 ps). 100 ps is more than sufficient for the box dimensions to converge to their equilibrium values at each temperature — the lattice expansion timescale is ~1–10 ps. |
| `NPT_BARO_DAMP` | `1.0` ps | Barostat coupling time for the Parrinello-Rahman barostat. Controls how aggressively the box dimensions are adjusted to maintain zero pressure. 1.0 ps allows slow, smooth box relaxation without causing oscillations. |
| `NPT_DUMP_EVERY` | `100` | Write box dimensions (Lx, Ly, Lz) every 100 steps (0.05 ps) during NPT production. These values are time-averaged afterward to extract a₀(T) = ⟨Lx⟩ / supercell_reps. Writing every 100 steps gives 2000 data points per NPT run — enough for a well-converged average. |

**Bulk minimisation tolerances:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `MIN_ETOL` | `0.0` eV | Energy convergence criterion (disabled). Setting this to 0 means only the force criterion matters — the minimiser does not stop early based on energy change alone. |
| `MIN_FTOL` | `1e-8` eV/Å | Force convergence criterion. All force components on all atoms must fall below 1e-8 eV/Å. This is an extremely tight criterion (~1000× tighter than typical NEB tolerances) appropriate for producing a reference minimum-energy bulk structure that will be used as the starting point for many downstream simulations. |
| `MIN_MAXITER` | `50,000` | Maximum CG line-search iterations. This is a safety cap — a well-behaved bulk minimisation typically converges in a few hundred iterations. |
| `MIN_MAXEVAL` | `500,000` | Maximum total force evaluations across all CG steps. Another safety cap. |

**SLURM:** `GPU_SLURM_CFG` — multigpu partition, 24 h wall time, A100 GPU. The 24 h wall time covers the longest combined NPT + NVT run in the matrix. Job chaining handles cases where production NVT exceeds the wall time.

### 2d — Part 2 (Permeation KMC) Settings

**Input paths — all produced by Parts 1 and 3:**

| Variable | What it points to | Why Part 2 needs it |
|----------|------------------|---------------------|
| `RELAXED_SLAB_PATH` | `slabs/slab_relaxed.lammps` from Part 1 Phase A | Provides the Hastelloy N (1,1,1) surface geometry for building the Hop A NEB initial state (H* on the surface) |
| `SURFACE_SITES_JSON` | `slabs/surface_sites.json` from Part 1 Phase A | Provides the fractional coordinates of all symmetry-distinct surface sites so Part 2 knows where to place H* for the Hop A IS |
| `PHASE2_H_DIR` | `adsorption/h_atom/` from Part 1 Phase B | Directory of all relaxed H* structures. Part 2 picks the lowest-energy H* structure from this directory as the Hop A NEB IS. |
| `SUB_NEB_DIR` | Part 2 internal directory | Where Part 2 writes the Hop A and Hop B NEB input scripts and output files |
| `VIB_DIR` | Part 2 internal directory | Where Part 2 writes ASE vibration input/output files for Hop A and Hop B IS and TS structures |

**Thermodynamic variables — set to None at notebook time, auto-filled at runtime:**

| Variable | Initial value | What it becomes at runtime | Where it comes from |
|----------|--------------|--------------------------|---------------------|
| `DH_DISS_EV` | `None` | ΔH_diss (eV): the H₂(g) → 2H* reaction energy. Negative = exothermic (H* more stable than gas). | Lowest-barrier entry in `neb/ranked_barriers.json` from Part 1. Automatically read by `permeation_run.py` at startup. |
| `DH_ENTRY_EV` | `None` | ΔH_entry (eV): the H*(surface) → H_sub(subsurface-1) reaction energy = E(H_sub) − E(H*). Positive = endothermic (H* prefers surface). | Extracted from Hop A NEB result (Part 2 Phase 1). Filled in after Phase 1 completes. |

**Diffusivity variables — auto-loaded from Part 3 output:**

| Variable | Fallback | What it becomes | Where it comes from |
|----------|----------|----------------|---------------------|
| `D0_M2S` | placeholder (0.0) | Pre-exponential factor D₀ in m²/s from the Arrhenius fit D(T) = D₀ exp(−E_D/k_BT) | `diffusivity_arrhenius.json` written by Part 3 Phase 3. Read at `permeation_run.py` startup. |
| `E_D_EV` | placeholder (0.0) | Activation energy E_D in eV for bulk H diffusion | Same file as D₀. |

**Pressure sweep:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `P_VALS_PA` | 40 log-spaced points from 1×10⁻⁵ to 1×10⁶ Pa | The gas-phase H₂ pressure range over which the KMC simulation is run. This spans from ultra-high vacuum (10⁻⁵ Pa) to 10 bar (10⁶ Pa), covering the full experimental range for H permeation measurements. The 40 points are needed to trace out the C₀(P) curve and verify Sieverts-law behavior (C₀ ∝ √P). |

**KMC grid:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `NX`, `NY` | `40`, `40` | Dimensions of the 2D surface grid in units of lattice sites. 40×40 = 1600 surface sites and 1600 subsurface sites (3200 total). Large enough to capture cooperative coverage effects and avoid finite-size artifacts in the H* occupation statistics, small enough to run millions of KMC steps quickly. |
| `KMC_SEED` | `42` | Integer random seed for the BKL KMC random number generator. Fixed seed ensures reproducibility — re-running with the same inputs gives identical KMC trajectories. |
| `KMC_MAX_STEPS` | `500,000` | Maximum number of KMC events per (T, P) point. The simulation stops early if steady state is reached before this limit. 500,000 events is typically ~10³–10⁵ surface residence times, enough to equilibrate coverage at all pressures. |

**Membrane geometry:**

| Variable | Value | Meaning |
|----------|-------|---------|
| `L_M` | `1×10⁻³` m | Membrane thickness (1 mm). This is the physical distance H must traverse through the bulk of the membrane. It enters the Richardson-Sieverts flux equation: J = Φ(T) × (√P_high − √P_low) / L. Thicker membranes → lower flux. |
| `A0_M` | `3.52×10⁻¹⁰` m | Fallback lattice parameter (3.52 Å) used only if `lattice_params_vs_T.json` is not available. This is the literature value for Ni FCC. In normal pipeline execution this fallback is never used — the NPT-computed a₀(T) values are always loaded instead. |

### 2e — Pipeline SLURM

`PIPELINE_SLURM_CFG` defines the SLURM job that runs `pipeline_run.py` itself — the master orchestrator that waits for Parts 1, 3, and 2 to complete:
- **Partition**: `west` — a long-running CPU partition (not multigpu) since `pipeline_run.py` does no computation, only orchestration
- **CPUs**: 4 — enough for Python and occasional file parsing
- **RAM**: 16 GB
- **Wall time**: 30 days — this job sits alive for however long the combined Part 1 + Part 3 + Part 2 runs take. Parts 1 and 3 alone can take multiple days each.

---

## Section 3 — Cell 3: Part 1 — NEB Surface Barrier Workflow

Cell 3 calls `write_neb_run_script()` from `models/neb_workflow.py`, passing all Part 1 configuration variables as arguments. The function writes a complete, self-contained Python script to `NEB_RUN_PY`. This script contains all configuration constants baked into its header, followed by a single function call that runs the entire 5-phase NEB surface workflow when executed on the cluster.

### What `write_neb_run_script()` does at notebook execution time

It constructs the text of `neb_run.py` as a Python f-string, substituting all configuration variables (Z_FREEZE_CUTOFF, SURF_TIMESTEP_PS, N_IMAGES, SPRING_K, etc.) into the script header. The script body calls `orchestrate_full_neb_workflow(...)` with those header constants as arguments. All SLURM configs are embedded in the script so the script can submit and monitor its own child SLURM jobs autonomously when run on the cluster. Nothing is actually executed in the notebook at this point.

### Phase A — Slab Preparation

**Goal**: Produce a well-relaxed (1,1,1) Hastelloy N surface slab that represents the physical surface at the simulation temperature, and identify where on that surface H atoms (and H₂ molecules) will preferentially sit.

1. **Build slab from bulk**: ASE reads `BULK_MIN_PATH` (the energy-minimised bulk Hastelloy N supercell). The `surface()` function cleaves along the (1,1,1) plane, stacks 12 layers, adds a 15 Å vacuum region above, and repeats laterally by (5, 6). This produces a slab with ~720 atoms and periodic boundary conditions in x and y, with a vacuum gap in z. The 15 Å vacuum ensures that H₂ placed at 2.5 Å above the surface (Phase B) does not interact with its periodic image across the vacuum.

2. **Freeze bottom layers**: All atoms with z < `Z_FREEZE_CUTOFF` (22.115 Å) are assigned to a LAMMPS `freeze` group. These atoms are excluded from all force integration during NVT MD. Their z-positions define the freeze boundary: with 12 layers of ~2 Å spacing, the top ~3 layers (above 22.115 Å) are free and the bottom ~9 layers are frozen. The frozen layers serve as a rigid bulk reference — they provide the correct lateral lattice parameter and restoring forces without allowing the slab to drift or rotate.

3. **Surface relaxation via NVT MD**: LAMMPS is submitted as a SLURM GPU job (`GPU_SLURM_CFG`). The NVT protocol is:
   - **Heating**: Ramp from 0 K to 300 K (a representative surface relaxation temperature) over `SURF_HEAT_STEPS` (10,000) steps × 0.5 fs/step = 5 ps. The ramp avoids giving atoms large initial velocities that could displace them far from their starting positions.
   - **NVT production**: Run at constant T for `SURF_NVT_STEPS` (100,000) steps × 0.5 fs/step = 50 ps with Nosé-Hoover thermostat coupling time `SURF_THERMO_DAMP` (0.05 ps). This gives the surface atoms time to settle into their thermally relaxed positions. The final frame is saved as `slab_relaxed.lammps`.
   - A CG energy minimisation (`SURF_FTOL` = 1e-6 eV/Å) is also run on the final frame to remove any residual forces before the structure is used as input to Phase B.

4. **Surface site enumeration**: Using the relaxed slab, ASE's symmetry analysis identifies all symmetry-inequivalent adsorption sites:
   - **Hollow sites**: centers of triangles formed by three nearest-neighbor surface atoms (FCC and HCP hollows)
   - **Bridge sites**: midpoints between two nearest-neighbor surface atoms
   - **Atop sites**: directly above each surface atom
   Sites separated by less than `prox_cutoff` are considered equivalent and merged. The result maps each site label (e.g., `'hollow_0'`, `'bridge_1'`) to its fractional coordinates in the slab cell.

5. **Outputs**:
   - `slabs/slab_relaxed.lammps` — relaxed slab LAMMPS data file (used by Phase B and by Part 2 Hop A NEB)
   - `slabs/surface_sites.json` — site labels mapped to fractional coordinates (used by Phase B and Part 2)

### Phase B — H₂* Adsorption and H* Pair Scanning

**Goal**: Generate the structures needed as initial states (IS) and final states (FS) for all NEB calculations in Phase C. This phase produces both the intact H₂* molecularly adsorbed structures (NEB IS for dissociation NEB) and the dissociated 2H* pair structures (NEB FS for dissociation NEB), as well as isolated H* structures for each site (NEB IS/FS for surface diffusion NEB).

**Two distinct energy references are used here, and it is critical to understand which is which:**

> **E_H2_GAS** (`-6.790499` eV): The MACE total energy of a free H₂ molecule in vacuum. This is a **fixed thermodynamic reference** used only for computing adsorption energies and reaction enthalpies relative to the gas phase. It is the energy of a reactant at the start of the overall process H₂(g) → 2H*(ads). It is **not** a structural input to any NEB calculation.

> **H₂\* structure**: An H₂ molecule placed above a surface site and CG-minimised. This gives the energy of H₂ molecularly adsorbed on the surface. The H–H bond is still intact. This structure IS the NEB initial state for dissociation NEB calculations. The energy of this structure is what LAMMPS uses as the starting point of the NEB chain.

**Step 1 — H₂* minimisation (produces NEB IS for dissociation NEB):**

For each surface site identified in Phase A, an H₂ molecule is placed with its center of mass at height `H2_HEIGHT` (2.5 Å) above the site, with the H–H bond parallel to the surface at its experimental length `H2_BOND` (0.741 Å). The full slab + H₂ system (frozen bottom layers, free top layers + H₂) is CG-minimised using `ADS_MIN_FTOL` (1e-6 eV/Å). The resulting structure — call it `H₂*` — has the H₂ molecule settling into its lowest-energy orientation and height above that site, with the surface atoms beneath it also relaxed. This is the NEB initial state: the state before dissociation occurs, with H₂ still intact.

**Step 2 — H* pair scanning (produces NEB FS for dissociation NEB, and IS/FS for surface diffusion NEB):**

For each surface site, the H₂ molecule from Step 1 is split into two separate H atoms. Every symmetry-distinct pair of surface sites (site_i, site_j) where the H–H distance satisfies `SEP_MIN` ≤ d(H,H) ≤ `SEP_MAX` (2.5–6.0 Å) is selected as a candidate dissociated H* pair. For each candidate pair:
- One H atom is placed at site_i and one at site_j, both at height `H2_HEIGHT` above the surface.
- The full slab + 2H system is CG-minimised.
- The resulting structure, where both H atoms are individually relaxed into their preferred surface sites, is the NEB final state: the state after complete H₂ dissociation.

Each individually-minimised single-H structure (one H at one site) is also saved. These become the IS and FS for surface diffusion NEB calculations: a single H* hopping from site_i to adjacent site_j.

**Step 3 — Adsorption energy calculation and ranking:**

For each minimised single-H* structure, the adsorption energy is computed:

```
E_ads(site_i) = E(slab + H* at site_i) − E(bare slab) − ½ × E_H2_GAS
```

This formula uses `E_H2_GAS` (free H₂ in vacuum) as the reference. The ½ factor comes from treating one H* as one-half of an H₂ molecule. A negative E_ads means the H* state is lower in energy than H in the gas phase — H adsorption is exothermic. The most negative E_ads site is the thermodynamically preferred H adsorption site.

Note the distinction between the NEB barrier (kinetic) and ΔH_diss (thermodynamic):
- **NEB barrier ΔE_diss**: the energy difference between the TS (highest-energy NEB image) and the IS (H₂* structure). This is measured entirely from the H₂* IS, not from H₂(g).
- **ΔH_diss**: the energy of the FS (2H* at sites i,j) minus the energy of H₂(g). This uses E_H2_GAS and tells you the thermodynamic driving force for the overall gas-phase dissociation. Computed as: ΔH_diss = E(slab + 2H*) − E(bare slab) − E_H2_GAS.

Both quantities are needed: ΔE_diss (kinetic barrier) controls the rate of dissociation via TST; ΔH_diss (thermodynamic) feeds into the Sieverts equilibrium constant and ultimately into the permeability calculation.

**Outputs**:
- `adsorption/h_atom/h_atom_{site}_relaxed.lammps` — one LAMMPS data file per H* site (used by Part 2 Hop A NEB as the IS)
- `adsorption/h2star/h2star_{site}_relaxed.lammps` — one LAMMPS data file per H₂* site (used by Phase C as the dissociation NEB IS)
- `adsorption/adsorption_energies.json` — E_ads for every H* site, sorted from most stable to least stable

### Phase C — NEB Job Generation

**Goal**: Take the structures from Phase B and write all LAMMPS NEB input files. Two types of NEB transitions are generated.

**Type 1 — Dissociation NEB (H₂* → 2H*):**

For each (H₂* site, H* pair) combination where the two H* atoms in the pair are within `graph_dist_min` hops of the H₂* site:
- **IS**: The `h2star_{site}_relaxed.lammps` structure from Phase B Step 1. H₂ is intact on the surface.
- **FS**: The `h_atom_{site_i}_h_atom_{site_j}_relaxed.lammps` pair structure from Phase B Step 2. Two H atoms are separately adsorbed at sites i and j.
- The NEB path goes from the intact H₂* molecule through a transition state where the H–H bond is partially broken, to the two separated H* atoms. The TS is the point of maximum energy along this path — the H–H bond is breaking but neither H has fully settled into its final surface site.
- The forward barrier ΔE_diss = E(TS) − E(H₂* IS) gives the kinetic barrier for H₂ to dissociate.
- The reverse barrier ΔE_des = E(TS) − E(2H* FS) gives the kinetic barrier for two H* to recombine and desorb as H₂.

**Type 2 — Surface diffusion NEB (H* site_i → H* site_j):**

For each pair of surface sites (site_i, site_j) within `graph_dist_min` hops where an H atom needs to hop across the surface:
- **IS**: The `h_atom_{site_i}_relaxed.lammps` structure — single H* at site_i.
- **FS**: The `h_atom_{site_j}_relaxed.lammps` structure — single H* at adjacent site_j.
- The NEB path goes from H* at site_i through a bridge-site TS to H* at site_j.
- The barrier ΔE_diff = E(TS) − E(H* at site_i) is the surface diffusion hop barrier. This feeds into k_surf_diff(T) in Part 2's KMC.

**Script generation:**

For each NEB transition (either type), Phase C writes:
1. A LAMMPS FS-minimisation script — runs a CG minimisation of the FS structure using the GPU. This is required because LAMMPS NEB needs a separately-minimised FS file to initialise the FS end of the chain.
2. A LAMMPS CI-NEB script — runs the climbing-image NEB from IS to FS, using the FS structure from step 1. Parameters: `N_IMAGES` (18) intermediate images, spring constant `SPRING_K` (1.0 eV/Å²), force convergence `NEB_FTOL_VAL` (0.05 eV/Å).

All FS-minimisation jobs are collected into a single GPU SLURM array script `fsmin_array.sh` (one array element per NEB transition). All CI-NEB jobs are collected into a CPU SLURM array script `neb_array.sh`.

**Outputs**:
- `neb/neb_IS_{i}/` — directory per NEB, containing the IS structure and NEB LAMMPS input
- `neb/neb_FS_{i}/` — directory per NEB, containing the FS structure and FS minimisation input
- `slurm_scripts/fsmin_array.sh` — GPU SLURM array for all FS minimisations
- `slurm_scripts/neb_array.sh` — CPU SLURM array for all CI-NEB runs

### Phase D — SLURM Submission and Barrier Parsing

1. **Submit FS minimisations**: `submit_slurm_job(fsmin_array.sh)` submits the GPU array (multigpu partition, 4 h). Returns a job ID. `wait_for_jobs({fsmin_id})` blocks until every array element has completed and every FS structure is minimised.

2. **Submit CI-NEB runs**: `submit_slurm_job(neb_array.sh)` submits the CPU array (short partition, 16 CPUs, 12 h). Returns a job ID. `wait_for_jobs({neb_id})` blocks until every CI-NEB has converged.

3. **Parse NEB log files**: For each completed NEB, the LAMMPS log file is read. The final-iteration NEB output reports the energy of each image. The highest-energy image is the TS. The forward barrier is ΔE_fwd = E(TS) − E(IS). The reverse barrier is ΔE_rev = E(TS) − E(FS). The reaction energy is ΔH = E(FS) − E(IS).

   For dissociation NEB: ΔE_fwd is the kinetic dissociation barrier from H₂*; ΔE_rev is the recombination barrier; ΔH is E(2H* FS) − E(H₂* IS). Note: ΔH here is referenced to H₂* on the surface, not to H₂(g). To get ΔH_diss relative to gas phase (the Sieverts thermodynamic quantity), you add the H₂* binding energy: ΔH_diss(gas ref) = ΔH(NEB) + E_ads(H₂*).

4. All transitions are sorted by ΔE_fwd and written to `ranked_barriers.json`.

**Outputs**:
- `neb/ranked_barriers.json` — list of all NEB transitions sorted by forward barrier, each entry containing: transition type (dissociation or surface diffusion), IS label, FS label, ΔE_fwd (eV), ΔE_rev (eV), ΔH (eV), TS image index, TS geometry file path

### Phase E — Vibrational Frequencies and TST Rate Constants

**Goal**: Convert the static NEB barriers into temperature-dependent TST rate constants. The NEB gives a 0 K barrier height; Phase E adds two quantum corrections (ZPE) and computes the attempt frequency (Vineyard prefactor) needed for absolute rate constants at each temperature.

**Why vibrational frequencies are needed**: Transition state theory gives the rate constant as:

```
k(T) = ν_attempt × exp(−ΔE_eff / k_B T)
```

where `ν_attempt` is the attempt frequency (how often the system tries to cross the barrier) and `ΔE_eff` is the effective barrier height including quantum zero-point energy. Both require knowledge of vibrational modes at the IS and TS.

1. **Frozen-phonon frequency calculation**: For each NEB transition, ASE's `Vibrations` class is used with MACE as the force engine. Atoms in the IS structure are displaced by ±0.01 Å along each Cartesian direction; the forces are computed by MACE; the dynamical matrix is assembled and diagonalised to give normal mode frequencies. This is repeated for the TS structure (the highest-energy NEB image). The TS has exactly one imaginary frequency (negative eigenvalue) — this is the unstable mode along the reaction coordinate (H–H bond stretching for dissociation NEB, or H lateral displacement for diffusion NEB). All other TS modes are real.

2. **Vineyard prefactor**: The attempt frequency is computed as the ratio of normal mode products:
   ```
   ν_Vineyard = (∏ᵢ νᵢ^IS) / (∏ᵢ νᵢ^TS_real)
   ```
   where the numerator is the product of all real frequencies at the IS, and the denominator is the product of all real frequencies at the TS (excluding the imaginary reaction-coordinate mode). The imaginary mode must be excluded from the TS denominator because it is not a vibration — it is the instability direction. Including it would make the denominator imaginary, which has no physical meaning in TST.

3. **Zero-point energy (ZPE) correction**: H is a light atom (mass = 1.008 amu). Its high-frequency vibrations (H–H stretch ~3800 cm⁻¹, H–metal stretch ~800–1200 cm⁻¹) have large ZPE contributions: E_ZPE = ½ℏω. The ZPE correction to the effective barrier is:
   ```
   ΔE_ZPE = ½ℏ × (∑ωᵢ^IS − ∑ωᵢ^TS_real)
   ```
   This is the difference in zero-point energy between IS and TS. If the IS has higher-frequency modes than the TS (common for H), ΔE_ZPE is positive and the effective barrier is higher than the classical NEB barrier. For H on metals, ZPE corrections are typically +0.05 to +0.15 eV — significant enough to affect rates by a factor of 2–5 at 500 K.

4. **Final rate constants at each temperature**:
   ```
   k_diss(T) = ν_Vineyard,diss × exp(−(ΔE_fwd + ΔE_ZPE,diss) / k_B T)
   k_des(T)  = ν_Vineyard,des  × exp(−(ΔE_rev + ΔE_ZPE,des)  / k_B T)
   k_diff(T) = ν_Vineyard,diff × exp(−(ΔE_diff + ΔE_ZPE,diff) / k_B T)
   ```
   These are computed at each temperature in `TEMPERATURES` = [500, 600, 700, 800, 900] K.

5. **Outputs**:
   - `neb/diss_vib_rates.json` — k_diss(T) and k_des(T) at each temperature, plus ν_Vineyard, ΔE_ZPE for each dissociation NEB transition
   - `neb/diff_vib_rates.json` — k_diff(T) at each temperature for each surface diffusion NEB transition
   - `vibrations/` — directory of ASE vibration files (`.json` per mode per structure)

---

## Section 4 — Cell 4: Part 3 — Bulk H Diffusivity Workflow

Cell 4 calls `generate_diffusivity_scripts()` from `models/diffusivity_workflow.py`. This function writes `diffusivity_run.py` (and its SLURM wrapper) with all Part 3 configuration baked into its header as Python constants. When `diffusivity_run.py` is executed on the cluster, it iterates over every combination in `INPUT_STRUCTURES × N_H_VALUES` and runs three sequential phases for each.

### What `generate_diffusivity_scripts()` does at notebook execution time

It constructs `diffusivity_run.py` as an f-string, substituting all Cell 2 diffusivity parameters into the header. Like `neb_run.py`, this script is self-contained: it writes its own LAMMPS inputs, submits its own SLURM jobs, and waits for them. Nothing is pre-generated in the notebook — all directory creation, script writing, and job submission happen at runtime on the cluster.

### Phase 1a — Bare Bulk Minimisation

**Goal**: Produce the lowest-energy T=0 K structure of the bare Hastelloy N bulk supercell. This is the starting point for all NPT runs. Starting from a properly minimised structure prevents the initial NPT frames from being contaminated by large forces from non-equilibrium atomic positions.

1. `write_minimization_script()` from `models/lammps_script.py` generates a LAMMPS CG minimisation input. The minimisation uses:
   - `MIN_ETOL` = 0.0 (energy criterion disabled — force criterion alone governs convergence)
   - `MIN_FTOL` = 1e-8 eV/Å (extremely tight — all forces essentially zero)
   - `MIN_MAXITER` = 50,000 (generous iteration cap)
   - `MIN_MAXEVAL` = 500,000 (generous force-evaluation cap)
2. A SLURM GPU job is created and submitted. `wait_for_jobs()` blocks until LAMMPS reports convergence.
3. **Output**: `structures/bulk_min.lammps` — the ground-state bulk structure at T=0 K, P=0. This file is the input to Phase 1b.

### Phase 1b — NPT Lattice Equilibration (5 parallel SLURM jobs)

**Goal**: Find the thermally-equilibrated lattice parameter a₀(T) at each of the five target temperatures. This is necessary because the FCC Hastelloy N lattice expands with temperature — using the 0 K lattice parameter for 900 K NVT MD would impose ~0.5% compressive strain on the simulation box, creating unphysical residual stresses that would distort H diffusion rates.

All five temperature runs are submitted as independent SLURM GPU jobs simultaneously.

**NPT MD sequence for each temperature:**

1. **Read structure**: LAMMPS reads `bulk_min.lammps`. Atomic velocities are initialised from a Maxwell-Boltzmann distribution at the target temperature.

2. **Heating stage**: NPT fix is applied with a linear temperature ramp from 10 K to T_target over `NPT_HEAT_STEPS` = 20,000 steps = 10 ps. The Parrinello-Rahman barostat maintains P = 0 bar in all three box directions throughout, allowing the box to expand freely as it heats. Barostat coupling time is `NPT_BARO_DAMP` = 1.0 ps. Starting from 10 K (not 0 K) avoids zero-velocity stagnation in velocity rescaling.

3. **NPT production stage**: Temperature is held at T_target and pressure at 0 bar for `NPT_PROD_STEPS` = 200,000 steps = 100 ps. Every `NPT_DUMP_EVERY` = 100 steps (0.05 ps), the current box dimensions Lx, Ly, Lz are appended to `npt_boxdims_{T}K.dat`. This generates 2000 box-dimension snapshots over 100 ps.

4. **a₀(T) extraction**: The post-processor reads `npt_boxdims_{T}K.dat` and computes ⟨Lx⟩, ⟨Ly⟩, ⟨Lz⟩ over the last 80% of the production period (discarding the first 20 ps as additional equilibration). For a 5×5×5 supercell, a₀(T) = ⟨Lx⟩ / 5. The average is taken from all three dimensions and checked for consistency (for a cubic crystal all three should agree within ~0.001 Å).

5. **H atom insertion**: `n_H` hydrogen atoms are placed at octahedral interstitial sites of the equilibrated cell (octahedral sites = face-centered positions of the FCC lattice, coordinates ½½0, ½0½, 0½½ in fractional coordinates). The sites are chosen to maximise the minimum H–H distance, avoiding unphysical H clustering. A CG minimisation (`MIN_FTOL` = 1e-8 eV/Å) finds the ground state of the bulk+H structure at the T-corrected cell dimensions.

6. **Output per temperature**: `structures/{T}K/bulk_min_h.lammps` — the T-corrected bulk cell with n_H hydrogen atoms, energy minimised at 0 K in the T-corrected box.

**Shared outputs**:
- `results/lattice_params_vs_T.json` — dict mapping temperature (K) to a₀(T) (Å), written after all five NPT jobs complete. Part 2 reads this file to get the correct lattice parameter for each temperature's rate constant calculation.

### Phase 2 — NVT MD (5 parallel SLURM jobs)

**Goal**: Run long NVT molecular dynamics trajectories to measure the mean-square displacement (MSD) of H atoms in the bulk, from which the diffusivity D(T) = ⟨|r(t)|²⟩ / 6t is extracted.

All five temperature runs are submitted simultaneously. Each job uses SLURM job-chaining: a wrapper script submits LAMMPS and, if LAMMPS exits due to hitting the wall time (rather than completing normally), automatically submits a continuation job that restarts from the last checkpoint.

**NVT MD sequence for each temperature:**

1. **Read structure**: LAMMPS reads `structures/{T}K/bulk_min_h.lammps`. Initial velocities are drawn from a Maxwell-Boltzmann distribution at T.

2. **Equilibration (excluded from MSD)**:  NVT Nosé-Hoover thermostat with coupling time `TAU_T_PS` = 0.1 ps is applied for `N_EQUIL_STEPS` = 2,000,000 steps = 1 ns. During this phase, H atoms diffuse out of their initial interstitial positions and start exploring the lattice. The 1 ns equilibration ensures H has visited many sites before MSD collection begins, so the measured MSD reflects steady-state diffusion rather than initial relaxation. **No MSD data is written during equilibration.** The LAMMPS `compute msd` and `fix print` commands are only activated after equilibration.

3. **Production (MSD collected)**:  NVT runs for `N_PROD_STEPS` = 5,000,000 steps = 2.5 ns. The LAMMPS `compute msd` command tracks ⟨|r(t) − r(0)|²⟩ for all H atoms, where r(0) is each H atom's position at the start of the production phase. MSD values are printed to `msd_{T}K.dat` every `DUMP_EVERY` = 1000 steps (0.5 ps). Full atom positions are written to `nvt_{T}K.dump` every 1000 steps for trajectory visualization.

4. **Restart checkpoints**: A binary restart file is written every `RESTART_EVERY` = 100,000 steps (50 ps) to `checkpoints/nvt_{T}K.{step}.restart`. If the SLURM job hits its wall time, the chain script detects the incomplete run (checks whether the expected final step was written), finds the most recent restart file, and submits a new LAMMPS job with `read_restart` instead of `read_data`. The MSD counter continues from where it left off.

**Outputs per temperature**:
- `results/{T}K/msd_{T}K.dat` — MSD vs. time: columns are (time in ps, MSD in Å²)
- `results/{T}K/nvt_{T}K.dump` — full trajectory for visualization
- `results/{T}K/nvt_{T}K.out` — LAMMPS thermo log (temperature, energy, pressure vs. time)

### Phase 3 — Post-processing: D(T) and Arrhenius Fit

**Goal**: Extract the scalar diffusivity D(T) at each temperature from the MSD data, then fit the five D(T) values to an Arrhenius equation.

1. **MSD data parsing**: Each `msd_{T}K.dat` file provides ⟨|r(t)|²⟩ vs. t. The curve has three regimes:
   - **Ballistic regime** (t < ~0.1 ps): MSD ∝ t² — atoms move in straight lines before their first collision. This regime is excluded from the D fit.
   - **Sub-diffusive crossover** (t ~ 0.1–10 ps): MSD deviates from both t² and t — this is the regime where atoms are caged by their neighbors before making their first hop. Also excluded.
   - **Diffusive (Einstein) regime** (t > ~10 ps for H at 500 K): MSD ∝ t — the mean-square displacement grows linearly with time. This is the regime used for D extraction. The crossover time to the diffusive regime decreases at higher temperature (faster hops).
   The linear regime is identified by computing the local slope of log(MSD) vs. log(t) — where this slope equals 1.0 ± 0.05, the data is in the diffusive regime.

2. **Einstein relation**: In the diffusive regime, D(T) = ⟨|r(t)|²⟩ / (6t). The factor 6 comes from 3D diffusion: ⟨x²⟩ + ⟨y²⟩ + ⟨z²⟩ = 6Dt. A weighted least-squares fit of MSD vs. t over the diffusive regime gives the slope 6D(T). The R² of this fit is reported — R² < 0.99 indicates insufficient statistics (run longer) or non-Fickian behavior.

3. **Arrhenius fit**: D(T) = D₀ exp(−E_D / k_B T). Taking logs: ln(D) = ln(D₀) − E_D / (k_B T). A linear regression of ln(D) vs. 1/T across all five temperatures gives:
   - Slope = −E_D / k_B → E_D (eV)
   - Intercept = ln(D₀) → D₀ (m²/s)
   The R² of this fit indicates how well Arrhenius behavior holds across the 500–900 K range. If R² < 0.99, there may be a change in diffusion mechanism (e.g., quantum tunneling contribution at low T, or vacancy-assisted diffusion at high T).

4. **Outputs**:
   - `results/diffusivity_arrhenius.json` — for each (structure, n_H) combination: D₀ (m²/s), E_D (eV), D(T) at each temperature (m²/s), R² for both MSD linear fit and Arrhenius fit
   - `results/analysis/diffusivity_table.txt` — human-readable table comparing D(T) across all structures and n_H values

Part 2 reads `diffusivity_arrhenius.json` at runtime and uses D₀ and E_D to compute D(T) at each KMC temperature.

---

## Section 5 — Cell 5: Part 2 — Permeation Workflow

Cell 5 calls `generate_permeation_scripts()` from `models/permeation_workflow.py`, producing `permeation_run.py`. This script runs the complete six-phase permeation calculation. It is the integrator: it takes surface energetics from Part 1 and bulk diffusivity from Part 3, connects them through TST rate theory and KMC simulation, and produces the final macroscopic permeability Φ(T).

### What `generate_permeation_scripts()` does at notebook execution time

Like the other generators, it writes a Python script with header constants baked in. The critical difference from Parts 1 and 3 is that several key constants — `DH_DISS_EV`, `DH_ENTRY_EV`, `D0_M2S`, `E_D_EV`, `A0_M` — are written into the header as `None` or placeholder values. At the very start of `permeation_run.py` execution on the cluster, before running any NEB or KMC, it loads the JSON output files from Parts 1 and 3 and replaces these `None` values with the actual computed numbers. This auto-fill is why the pipeline requires Parts 1 and 3 to complete before Part 2 starts.

### Phase 1 — Hop A NEB (H* surface → H subsurface-1)

**Goal**: Compute the energy barrier for a hydrogen atom to move from its preferred surface adsorption site (H*) into the first subsurface octahedral interstitial site (H_sub-1). This is the surface-to-bulk entry step. Without this barrier, you cannot compute how fast H enters the metal from the surface layer.

This NEB is structurally distinct from the Part 1 dissociation NEB: the IS is a single H atom on the surface (not H₂), and the FS is a single H atom in the first bulk layer below the surface (not on the surface). The slab geometry used is the same relaxed slab from Part 1 Phase A.

1. **IS construction**: The lowest-energy single-H* structure from Part 1's `adsorption/h_atom/` directory is selected (the site with the most negative E_ads). This file is read from `PHASE2_H_DIR`. The IS is: H* at the most stable surface adsorption site, with all slab atoms in their relaxed positions.

2. **FS construction**: The first subsurface octahedral interstitial site is identified as the nearest oct-site in the first atomic layer below the surface. Using the T-appropriate a₀ (from `lattice_params_vs_T.json`), an H atom is placed at this site and the full slab + H_sub system is CG-minimised. The FS is: H atom in the first subsurface octahedral site, with the nearby surface atoms relaxed around it.

3. **NEB run**: 18-image CI-NEB is run between the H* IS and H_sub-1 FS. The NEB path describes H moving from the surface-binding site, through the surface layer, into the first bulk octahedral site. The NEB is submitted as a CPU SLURM array job, identical in setup to the Part 1 NEB runs.

4. **Energy extraction**: The NEB output gives:
   - ΔE_entry = E(TS) − E(H* IS): the kinetic barrier for H to enter the bulk from the surface. For most transition metals this is 0.1–0.5 eV.
   - ΔE_exit = E(TS) − E(H_sub-1 FS): the kinetic barrier for H to exit the bulk back to the surface.
   - ΔH_entry = E(H_sub-1 FS) − E(H* IS): the thermodynamic reaction energy for the surface → subsurface hop. Positive means H is more stable on the surface (endothermic entry). This `ΔH_entry` is stored and auto-fills `DH_ENTRY_EV` in Phase 4.

5. **Output**: `neb_subsurface/hopa/ranked_hopa.json` — ΔE_entry, ΔE_exit, ΔH_entry, TS image geometry path

### Phase 2 — Hop B NEB (H subsurface-1 → subsurface-2)

**Goal**: Compute the energy barrier for H to hop between two adjacent octahedral interstitial sites inside the bulk lattice. This bulk migration barrier ΔE_mig characterises how easily H diffuses through the bulk, independent of the surface. It serves as a consistency check against the Arrhenius activation energy E_D from Part 3: they should agree within ~0.05 eV. If they differ significantly, there may be an error in the Part 3 NVT MD or the NEB IS/FS are not representative of the dominant diffusion mechanism.

1. **IS/FS construction**: Two nearest-neighbor octahedral sites in the second and third subsurface layers (away from the surface, where bulk behavior is expected) are selected. Using the T-appropriate a₀, H is placed at each site and individually minimised with the slab frozen. The IS is H at oct-site A; the FS is H at nearest oct-site B. In FCC, adjacent oct-sites are connected by a hop distance of a₀/√2 passing through a tetrahedral interstitial TS.

2. **NEB run**: 18-image CI-NEB between IS and FS. The TS is the tetrahedral interstitial site between the two oct-sites — the saddlepoint where H has maximum energy.

3. **Energy extraction**:
   - ΔE_mig = E(TS) − E(IS): the kinetic barrier for bulk oct-to-oct migration
   - ΔH_mig = E(FS) − E(IS): should be ~0 eV for a symmetric hop in a pure FCC lattice; non-zero in Hastelloy N due to compositional disorder

4. **Output**: `neb_subsurface/hopb/ranked_hopb.json` — ΔE_mig, ΔH_mig, TS image geometry path

### Phase 3 — Vibrational Frequencies for Hop A and Hop B

**Goal**: Apply ZPE corrections and compute Vineyard prefactors for the surface entry/exit rate constants (from Hop A) and the bulk migration rate constant (from Hop B). The physical reasoning is identical to Part 1 Phase E — H is light, so its ZPE contributions are large and cannot be neglected.

1. **ASE Vibrations for Hop A IS (H* on surface)**: Finite-difference Hessian computed at the H* IS structure using MACE. Gives all normal mode frequencies ωᵢ^HopA_IS, including H–surface stretch, surface diffusion modes, and the slab phonon modes of nearby metal atoms.

2. **ASE Vibrations for Hop A TS (H at surface–subsurface boundary)**: Hessian at the TS structure identified from the Hop A NEB (the highest-energy image). One imaginary frequency is found — this is H moving in the direction normal to the surface (the entry/exit reaction coordinate). All other modes are real.

3. **Hop A rate constants**:
   ```
   ν_Vineyard,A = (∏ωᵢ^HopA_IS) / (∏ωᵢ^HopA_TS_real)
   ΔE_ZPE,A     = ½ℏ(∑ωᵢ^HopA_IS − ∑ωᵢ^HopA_TS_real)
   k_entry(T)   = ν_Vineyard,A × exp(−(ΔE_entry + ΔE_ZPE,A) / k_B T)
   k_exit(T)    = ν_Vineyard,A × exp(−(ΔE_exit  + ΔE_ZPE,A,rev) / k_B T)
   ```

4. **ASE Vibrations for Hop B IS and TS**: Same procedure for the bulk oct-to-oct hop. The Hop B TS is the tetrahedral interstitial site; its imaginary mode is H moving along the oct–oct vector.

5. **Hop B rate constant**:
   ```
   k_mig(T) = ν_Vineyard,B × exp(−(ΔE_mig + ΔE_ZPE,B) / k_B T)
   ```

6. **ΔH_diss auto-extraction**: At this point, `DH_DISS_EV` is filled in from Part 1's `neb/ranked_barriers.json` — specifically the ΔH_diss value from the lowest-barrier dissociation NEB (most exothermic H₂ → 2H* reaction).

7. **Outputs**:
   - `vibrations/hopa_is_vib.json`, `vibrations/hopa_ts_vib.json` — frequency files for Hop A
   - `vibrations/hopb_is_vib.json`, `vibrations/hopb_ts_vib.json` — frequency files for Hop B
   - Partial update to `results/rate_dict_T{T}K.json` with k_entry(T), k_exit(T), k_mig(T) at each T

### Phase 4 — TST Rate Constants: Complete Assembly

**Goal**: Assemble all six elementary rate constants at each temperature into a complete rate dictionary that the KMC simulation (Phase 5) will consume.

At each temperature T in `TEMPERATURES`:

| Rate constant | Source NEB | Formula | Physical event |
|--------------|-----------|---------|---------------|
| `k_diss(T)` | Part 1 Phase E (dissociation NEB, H₂* → 2H*) | ν_diss × exp(−(ΔE_diss + ΔE_ZPE,diss) / k_BT) | Gas-phase H₂ adsorbs and dissociates into 2H* on the surface |
| `k_des(T)` | Part 1 Phase E (reverse of dissociation NEB, 2H* → H₂*) | ν_des × exp(−(ΔE_des + ΔE_ZPE,des) / k_BT) | Two adjacent H* atoms recombine and desorb as H₂ gas |
| `k_entry(T)` | Part 2 Phase 3 (Hop A forward, H* → H_sub) | ν_A × exp(−(ΔE_entry + ΔE_ZPE,A) / k_BT) | Surface H* crosses the surface barrier and enters the first subsurface site |
| `k_exit(T)` | Part 2 Phase 3 (Hop A reverse, H_sub → H*) | ν_A × exp(−(ΔE_exit + ΔE_ZPE,A,rev) / k_BT) | Subsurface H_sub crosses back up to the surface site |
| `k_mig(T)` | Part 2 Phase 3 (Hop B, H_sub → H_sub') | ν_B × exp(−(ΔE_mig + ΔE_ZPE,B) / k_BT) | Subsurface H hops to adjacent oct-site in the bulk |
| `k_surf_diff(T)` | Part 1 Phase E (surface diffusion NEB, H* → H*) | ν_diff × exp(−(ΔE_diff + ΔE_ZPE,diff) / k_BT) | H* hops between adjacent surface sites |

In addition, the **bulk drainage rate** is computed from Part 3's diffusivity output:

```
k_drain(T) = D(T) / dx²
```

where D(T) = D₀ × exp(−E_D / k_BT) (Arrhenius from Part 3), D₀ and E_D are auto-loaded from `diffusivity_arrhenius.json`, and dx = a₀(T) / √2 is the nearest-neighbor oct-to-oct hop distance in the FCC lattice (loaded from `lattice_params_vs_T.json`). This is NOT a TST rate from NEB — it is computed directly from the MD diffusivity. It represents how fast H drains from the subsurface layer into the bulk membrane, averaged over all bulk migration hops.

**Output**: `results/rate_dict_T{T}K.json` for each T — complete dict with k_diss, k_des, k_entry, k_exit, k_mig, k_surf_diff, k_drain and all their Vineyard prefactors and ZPE corrections.

### Phase 5 — KMC Pressure Sweep

**Goal**: For each (temperature, pressure) pair, run a stochastic kinetic Monte Carlo simulation on a model of the Hastelloy N surface to determine the steady-state H coverage C₀ under those conditions. C₀(P,T) is needed to compute the permeability via Sieverts law.

**Physical model — the dual-layer alloy grid:**

The simulation domain is a 40×40 (`NX × NY`) grid with periodic boundary conditions in x and y. Each grid point (i,j) represents one metal surface site. The grid has two layers:

- **Surface layer** `S[i,j]`: Each site has (1) a fixed element type assigned once at initialization from the Hastelloy N composition (71% Ni, 16% Mo, 7% Cr, 6% Fe, drawn randomly), and (2) a binary occupancy: 0 (empty) or 1 (occupied by H*). Element type affects k_diss and k_des via site-specific barriers if compositional disorder is included; otherwise all sites use the average rate constants.
- **Subsurface layer** `B[i,j]`: Each site directly below surface site (i,j) in the first bulk layer. Binary occupancy only: 0 (empty) or 1 (occupied by H_sub). No element type assignment.

This dual-layer model captures: surface adsorption/desorption, surface diffusion, and the surface↔subsurface exchange. Bulk transport deeper than the subsurface layer is captured by k_drain (a coarse-grained sink rate), not by explicitly tracking every H atom through the 1 mm membrane.

**The BKL rejection-free KMC algorithm:**

At each KMC step:
1. Enumerate all possible events across the entire grid (all occupied surface sites, all occupied subsurface sites, all empty surface site pairs adjacent to an occupied surface site).
2. Compute the rate Rₙ for each possible event n (see event list below).
3. Compute R_total = ΣRₙ.
4. Advance simulation time by Δt = −ln(u₁) / R_total, where u₁ is a uniform random number in (0,1). This correctly samples the exponential waiting-time distribution.
5. Select event n with probability Rₙ / R_total using a second uniform random number u₂.
6. Execute event n (update the grid).
7. Return to step 1.

"Rejection-free" means every KMC step executes exactly one physical event — unlike Metropolis MC which may propose events that are rejected. This is essential for simulating rare events (low-pressure adsorption) efficiently.

**Complete event catalog:**

| Event | Rate | Condition |
|-------|------|-----------|
| H₂ adsorption + dissociation at pair (i,j) | R_strike × k_diss(T), where R_strike = P × A_site / √(2πm_H₂ k_B T) | Both S[i,j] = 0 and S[i',j'] = 0 for adjacent pair (i',j') |
| H* recombination + desorption at pair (i,j)+(i',j') | k_des(T) | S[i,j] = 1 and S[i',j'] = 1, (i',j') adjacent to (i,j) |
| H* surface hop from (i,j) to (i',j') | k_surf_diff(T) | S[i,j] = 1 and S[i',j'] = 0 |
| H* entry from (i,j) to B[i,j] | k_entry(T) | S[i,j] = 1 and B[i,j] = 0 |
| H_sub exit from B[i,j] to (i,j) | k_exit(T) | B[i,j] = 1 and S[i,j] = 0 |
| H_sub bulk drainage from B[i,j] | k_drain(T) | B[i,j] = 1 |

R_strike is the rate at which H₂ molecules from the gas phase strike a single surface site. It comes from the Hertz-Knudsen formula: R_strike = P / √(2π m_H₂ k_B T). A_site is the surface area per site (≈ a₀²/√2 for FCC(111)).

The drainage event is the model's representation of bulk permeation: when an H_sub atom drains from the subsurface layer at rate k_drain, it has effectively entered the bulk membrane and will diffuse through to the other side. This is valid in the steady-state regime where the bulk concentration is approximately zero at the downstream face (vacuum permeate side). The drainage rate k_drain = D(T)/dx² represents the rate at which a random walker with diffusivity D moves one hop distance dx.

**Steady state and data recording:**

The simulation runs until the surface coverage θ = (ΣS[i,j]) / (NX × NY) stops changing — specifically, until the running average ⟨θ⟩ over the last 50,000 steps changes by less than 0.1% per 10,000 steps. At steady state, C₀(P,T) = θ × (4/a₀³) converts the dimensionless coverage to a hydrogen concentration in H/m³ (using 4 surface sites per a₀² area for FCC(111)).

**Output**: `results/permeation_sweep_T{T}K.json` — at each temperature, a dict mapping pressure (Pa) to steady-state C₀ (H/m³) and estimated permeation flux J = C₀ × D(T) / L (mol/m²/s/√Pa).

### Phase 6 — Permeability Calculation

**Goal**: Extract the macroscopic Richardson-Sieverts permeability Φ(T) from the KMC results and provide three independent cross-validation estimates.

**Step 1 — Verify Sieverts law:**

The KMC output C₀(P,T) at each temperature is fit to the Sieverts-law form:
```
C₀ = S(T) × √P
```

S(T) is the Sieverts solubility constant (units: H/m³/Pa^½). A good fit (R² > 0.99) means the system is in the Henry's-law / Sieverts regime: H₂ dissociation is in fast equilibrium with H* adsorption/desorption, and the equilibrium H* coverage is proportional to √P_H₂. This is the prerequisite for using the Richardson-Sieverts permeability framework. If R² < 0.99, the surface is kinetically limited (adsorption or desorption is the bottleneck) and the full KMC flux must be used directly rather than the Φ(T) formalism.

**Step 2 — Compute permeability via three independent routes:**

All three routes use Φ = D(T) × S(T), where D(T) is taken from Part 3. They differ only in how S(T) is computed:

**Option A — Geometric lattice site density:**
```
S_A(T) = (4/a₀³) × exp(−ΔH_sol / k_B T)
```
4/a₀³ is the density of octahedral interstitial sites in an FCC lattice (4 oct-sites per cubic unit cell of side a₀). ΔH_sol is the solution enthalpy: ΔH_sol = ΔH_entry + ΔH_diss/2, combining the surface entry energy (from Hop A NEB) and the gas-phase dissociation energy (from Part 1 Part D). This option assumes ideal Sieverts behavior and uses only geometric and thermodynamic inputs — no KMC output.

**Option B — TST detailed balance:**
```
S_B(T) = k_entry(T) × R_strike / k_exit(T)
```
At thermodynamic equilibrium, the rate of H entering the subsurface equals the rate of H exiting: k_entry × θ_surface = k_exit × θ_subsurface. Combined with the Sieverts law for the surface (θ_surface = k_diss × R_strike / k_des, from equilibrium at the surface), this gives S_B(T). This option uses the TST rate constants from Phases 3 and 4 — it does not use the KMC output, making it an independent check.

**Option C — KMC empirical:**
```
S_C(T) = C₀(P,T) / √P   [from Phase 5 Sieverts fit]
```
This is the KMC-measured S(T), directly from the simulation. It implicitly includes all surface kinetics, coverage effects, and compositional disorder — it is the most physically complete but also the most computationally expensive.

**Cross-validation**: If A, B, and C agree to within ~20%, the Sieverts regime holds and the permeability is robust. If C disagrees with B while A and B agree, the surface is kinetically limited (coverage is not at equilibrium). If all three disagree, there may be an error in one of the NEB calculations or the MD diffusivity.

**Step 3 — Arrhenius fit:**

Each Φ(T) series (from options A, B, C) is fit to:
```
Φ(T) = Φ₀ × exp(−E_Φ / k_B T)
```
The activation energy E_Φ = E_D + ΔH_sol is the fundamental materials property for permeability — it combines the bulk diffusion barrier E_D (from Part 3) and the solution enthalpy ΔH_sol (from Parts 1 and 2 NEBs). Φ₀ = D₀ × S₀ is the pre-exponential permeability.

**Step 4 — Richardson-Sieverts flux:**

The permeation flux through a membrane of thickness L at high-pressure P_high on one side and P_low ≈ 0 on the other:
```
J = (Φ(T) / L) × (√P_high − √P_low)  [mol H₂ / m² / s]
```
With P_low → 0 (vacuum permeate): J = Φ(T) × √P / L. This is the quantity measured in experimental permeation studies — the comparison target for validating the computed Φ(T).

**Outputs**:
- `results/permeability_T{T}K.json` — Φ (all three options) at each T, with Φ₀ and E_Φ from Arrhenius fit
- `results/solubility_arrhenius_kmc.json` — Arrhenius fit to S(T) from KMC: S₀ and ΔH_sol

---

## Section 6 — Cell 6: Pipeline Orchestration

Cell 6 ties everything together. It generates the master orchestrator scripts and optionally submits the entire pipeline as a single long-running SLURM job.

### Pre-step: Bare Bulk Minimisation (conditional, runs at notebook execution time)

Part 1 requires `BULK_MIN_PATH` (the energy-minimised bulk supercell) to exist on the cluster before `neb_run.py` can run, because `neb_run.py` reads this file to build the surface slab in Phase A. Cell 6 checks whether `BULK_MIN_PATH` exists:

- **If it does not exist**: `write_minimization_script()` generates a LAMMPS CG minimisation input; `write_slurm_job()` wraps it in a SLURM GPU script; `submit_slurm_job()` submits it immediately (this is NOT dry-run — it runs now, in the notebook); `wait_for_jobs()` blocks notebook execution until the minimisation finishes. Only after this completes does Cell 6 continue to generate the pipeline scripts.
- **If it already exists**: This step is skipped entirely — no job is submitted.

This pre-step runs at **notebook execution time**, not at pipeline submission time. The rationale: `pipeline_run.py` cannot do the minimisation itself because it needs `BULK_MIN_PATH` to exist before `neb_run.py` is submitted, and `neb_run.py` is the first thing `pipeline_run.py` submits.

### `generate_pipeline_scripts()` → `pipeline_run.py`

This function writes the master Python orchestrator. When `pipeline_run.py` runs on the cluster (inside its own SLURM job), it executes the following logic:

1. **Submit Part 1**: Calls `subprocess.run(['python', NEB_RUN_PY])` inside a SLURM wrapper (or equivalently, submits `neb_run.py` as a SLURM job). The returned job ID is recorded as `part1_id`.

2. **Submit Part 3**: Calls `subprocess.run(['python', DIFFUSIVITY_RUN_PY])` simultaneously (no dependency on Part 1). The returned job ID is recorded as `part3_id`. Parts 1 and 3 now run in parallel on the cluster — each has its own SLURM jobs, each manages its own child jobs.

3. **Wait for both**: `wait_for_jobs({part1_id, part3_id})` polls `squeue` every 60 seconds. This call blocks for however many days Parts 1 and 3 take to complete. `pipeline_run.py` does nothing but poll during this period.

4. **Auto-transfer data from Parts 1 and 3 to Part 2**: Once both jobs complete, `pipeline_run.py`:
   - Reads `neb/ranked_barriers.json` from Part 1 → extracts `ΔH_diss` (reaction energy of the lowest-barrier dissociation NEB) → writes this value into `permeation_run.py`'s `DH_DISS_EV` header constant, replacing the `None` placeholder.
   - Reads `neb/diss_vib_rates.json` from Part 1 → verifies k_diss(T) and k_des(T) are present for all five temperatures.
   - Reads `diffusivity_arrhenius.json` from Part 3 → extracts `D0_M2S` and `E_D_EV` → writes these into `permeation_run.py`'s header, replacing placeholders.
   - Reads `lattice_params_vs_T.json` from Part 3 → extracts a₀(T) for each T → writes into `permeation_run.py`'s header.
   This "patch" is implemented as a Python `re.sub()` call that finds and replaces the `None` / placeholder lines in the `permeation_run.py` header with the actual numerical values. After patching, `permeation_run.py` is a complete, self-contained script with no missing values.

5. **Submit Part 2**: Submits the now-patched `permeation_run.py` as a SLURM job. The returned job ID is recorded as `part2_id`.

6. **Wait for Part 2**: `wait_for_jobs({part2_id})` blocks until Part 2 completes.

7. **Final summary**: Reads `permeability_T{T}K.json` files and prints a summary table of Φ(T) from all three routes, Φ₀, and E_Φ.

### `generate_pipeline_sh()` → `pipeline_run.sh`

A minimal SLURM submission script for the `west` partition:
```
#SBATCH --partition=west
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30-00:00:00        # 30 days

conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
python pipeline_run.py
```

No GPU is requested because `pipeline_run.py` itself runs no MACE or LAMMPS calculations — it only submits and monitors child jobs. The 30-day wall time is the outer envelope of the entire campaign.

### Dry-run mode

In Cell 6, the final submission is gated by a flag:
```python
submit_slurm_job(PIPELINE_RUN_SH, dry_run=DRY_RUN)
```

With `DRY_RUN = True` (default): prints the `sbatch pipeline_run.sh` command to stdout but does NOT submit. You can inspect all generated scripts before committing.

With `DRY_RUN = False`: submits immediately. The job ID is printed. The entire multiscale campaign is now running.

---

## Section 7 — Full Data Flow Diagram

The complete end-to-end dependency map showing every file produced and consumed across all phases of all three pipeline parts. Arrows show data flow direction.

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                   pipeline.ipynb — Cell 2 (configuration)              │
  │  WORK_DIR, TEMPERATURES, BULK_MIN_PATH, E_H2_GAS, MILLER, LAYERS,    │
  │  VACUUM, LAT_REPEAT, SEP_MIN/MAX, N_IMAGES, SPRING_K, NEB_FTOL,      │
  │  Z_FREEZE_CUTOFF, SURF_TIMESTEP_PS, INPUT_STRUCTURES, N_H_VALUES,    │
  │  NPT/NVT params, MIN params, P_VALS_PA, NX/NY, KMC_MAX_STEPS, ...   │
  └────────────┬──────────────────────┬──────────────────────┬────────────┘
               │ Cell 3               │ Cell 4               │ Cell 5
               ▼                      ▼                      ▼
  write_neb_run_script()  generate_diffusivity_scripts()  generate_permeation_scripts()
               │                      │                      │
               ▼                      ▼                      ▼
          neb_run.py           diffusivity_run.py        permeation_run.py
          (Part 1)              (Part 3)                  (Part 2 — waits for
               │                      │                   Parts 1 and 3)
               │                      │                      ▲
  ─────────── PART 1 CLUSTER EXECUTION ────────────────────  │
               │                      │                      │
    Phase A    │                      │                      │
  ┌────────────┤                      │                      │
  │Build slab  │                      │                      │
  │Freeze btm  │                      │                      │
  │NVT relax   │                      │                      │
  │Site enum   │                      │                      │
  └────────────┘                      │                      │
       ▼                              │                      │
  slab_relaxed.lammps ───────────────────────────────────────►(Hop A IS geometry)
  surface_sites.json  ───────────────────────────────────────►(surface site coords)
       │                              │                      │
    Phase B                           │                      │
  ┌───────────────────────────┐       │                      │
  │H₂* min  →  NEB IS (diss) │       │                      │
  │H* pair  →  NEB FS (diss) │       │                      │
  │H* alone →  IS/FS (diff)  │       │                      │
  │E_ads = E(H*)-E(slab)      │       │                      │
  │        - ½×E_H2_GAS       │       │                      │
  │  (thermodynamic ref only) │       │                      │
  └───────────────────────────┘       │                      │
       ▼                              │                      │
  h2star_*_relaxed.lammps  (NEB IS)  │                      │
  h_atom_*_relaxed.lammps  (NEB FS & IS/FS) ────────────────►(Hop A IS)
  adsorption_energies.json (ranking) │                      │
       │                              │                      │
    Phase C/D                         │                      │
  ┌────────────────────────────┐      │                      │
  │Type 1: diss NEB            │      │                      │
  │  IS = h2star_*_relaxed     │      │                      │
  │  FS = h_atom pair          │      │                      │
  │  → ΔE_diss, ΔE_des, ΔH    │      │                      │
  │Type 2: surf diff NEB       │      │                      │
  │  IS = h_atom at site_i     │      │                      │
  │  FS = h_atom at site_j     │      │                      │
  │  → ΔE_diff                 │      │                      │
  └────────────────────────────┘      │                      │
       ▼                              │                      │
  ranked_barriers.json ──────────────────────────────────────►(ΔH_diss auto-fill)
       │                              │                      │
    Phase E                           │                      │
  ┌────────────────────────────┐      │                      │
  │Vibrations at IS and TS     │      │                      │
  │ → Vineyard ν, ZPE ΔE_ZPE   │      │                      │
  │ → k_diss(T), k_des(T),     │      │                      │
  │   k_surf_diff(T)  at 5T    │      │                      │
  └────────────────────────────┘      │                      │
       ▼                              │                      │
  diss_vib_rates.json  ──────────────────────────────────────►(k_diss, k_des)
  diff_vib_rates.json  ──────────────────────────────────────►(k_surf_diff)
                                      │                      │
  ─────────── PART 3 CLUSTER EXECUTION ────────────────────  │
                                      │                      │
                           Phase 1a   │                      │
                        ┌─────────────┤                      │
                        │Bare CG min  │                      │
                        └─────────────┘                      │
                              ▼                              │
                         bulk_min.lammps                     │
                              │                              │
                           Phase 1b (×5 T, parallel)        │
                        ┌────────────────────┐              │
                        │NPT at each T:       │              │
                        │  heat 10K→T (10ps)  │              │
                        │  prod at T,P=0(100ps│              │
                        │  dump Lx every 50fs │              │
                        │  a₀(T) = ⟨Lx⟩/5    │              │
                        │  insert n_H H atoms  │              │
                        │  CG min bulk+H      │              │
                        └────────────────────┘              │
                              ▼                              │
                         lattice_params_vs_T.json ───────────►(a₀(T) for rate calc)
                         bulk_min_h_{T}K.lammps              │
                              │                              │
                           Phase 2 (×5 T, parallel)         │
                        ┌────────────────────┐              │
                        │NVT MD at each T:    │              │
                        │  1ns equil (no MSD) │              │
                        │  2.5ns prod + MSD   │              │
                        │  restart every 50ps │              │
                        │  auto job chaining  │              │
                        └────────────────────┘              │
                              ▼                              │
                         msd_{T}K.dat                        │
                         nvt_{T}K.dump                       │
                              │                              │
                           Phase 3                           │
                        ┌────────────────────┐              │
                        │MSD → lin regime     │              │
                        │D(T) = slope/6t      │              │
                        │ln(D) vs 1/T → D₀,E_D│              │
                        └────────────────────┘              │
                              ▼                              │
                         diffusivity_arrhenius.json ─────────►(D₀, E_D auto-fill)
                                                             │
  ─────────── pipeline_run.py patches permeation_run.py ──  │
  (fills DH_DISS_EV, D0_M2S, E_D_EV, A0_M from JSONs above)│
                                                             │
  ─────────── PART 2 CLUSTER EXECUTION ────────────────────  │
                                                             │
                                              Phase 1 — Hop A NEB
                                         ┌───────────────────────┐
                                         │IS: h_atom lowest-E    │
                                         │HS: H at oct_sub-1     │
                                         │NEB → ΔE_entry,ΔE_exit │
                                         │      ΔH_entry         │
                                         └───────────────────────┘
                                              ▼
                                         ranked_hopa.json
                                              │
                                              Phase 2 — Hop B NEB
                                         ┌───────────────────────┐
                                         │IS: H at oct_A (bulk)  │
                                         │FS: H at oct_B (bulk)  │
                                         │NEB → ΔE_mig, ΔH_mig  │
                                         └───────────────────────┘
                                              ▼
                                         ranked_hopb.json
                                              │
                                              Phase 3 — Vibrations
                                         ┌───────────────────────┐
                                         │Hop A IS, TS → ν_A,ZPE │
                                         │Hop B IS, TS → ν_B,ZPE │
                                         │→ k_entry, k_exit,k_mig│
                                         └───────────────────────┘
                                              │
                                              Phase 4 — TST rates
                                         ┌───────────────────────┐
                                         │All 6 rate constants   │
                                         │at each of 5 T values  │
                                         │+ k_drain = D(T)/dx²   │
                                         └───────────────────────┘
                                              ▼
                                         rate_dict_T{T}K.json
                                              │
                                              Phase 5 — KMC sweep
                                         ┌───────────────────────┐
                                         │40×40 dual-layer grid  │
                                         │BKL KMC, 40 pressures  │
                                         │→ C₀(P,T) at steady st.│
                                         └───────────────────────┘
                                              ▼
                                         permeation_sweep_T{T}K.json
                                              │
                                              Phase 6 — permeability
                                         ┌───────────────────────┐
                                         │Sieverts fit C₀ = S√P  │
                                         │Φ_A = D × S_geom       │
                                         │Φ_B = D × S_TST        │
                                         │Φ_C = D × S_KMC        │
                                         │Arrhenius: Φ₀, E_Φ     │
                                         └───────────────────────┘
                                              ▼
                                         permeability_T{T}K.json
                                         solubility_arrhenius_kmc.json
                                              ▼
                                    Φ(T) = Φ₀ exp(−E_Φ / k_B T)
                                    E_Φ = E_D (Part 3) + ΔH_sol (Parts 1+2 NEBs)
                                    J = Φ(T) × √P / L  [mol/m²/s]
```

### Key file locations (all relative to `WORK_DIR/calculation/`)

| File | Produced by | Consumed by | What it contains |
|------|------------|-------------|-----------------|
| `slabs/slab_relaxed.lammps` | Part 1, Phase A | Part 2 Phase 1 (Hop A IS slab) | Relaxed 720-atom (1,1,1) slab |
| `slabs/surface_sites.json` | Part 1, Phase A | Part 1 Phase B, Part 2 Phase 1 | Site labels → fractional coordinates |
| `adsorption/h2star/h2star_*_relaxed.lammps` | Part 1, Phase B step 1 | Part 1 Phase C (dissociation NEB IS) | H₂ intact, molecularly adsorbed on each site |
| `adsorption/h_atom/h_atom_*_relaxed.lammps` | Part 1, Phase B step 2 | Part 1 Phase C (diss NEB FS, diff NEB IS/FS), Part 2 Phase 1 (Hop A IS) | Individual H* at each surface site |
| `adsorption/adsorption_energies.json` | Part 1, Phase B step 3 | Part 2 Phase 1 (selects lowest-E H* site) | E_ads(site) sorted by stability |
| `neb/ranked_barriers.json` | Part 1, Phase D | Part 2 Phase 3 (ΔH_diss), pipeline_run.py (auto-fill DH_DISS_EV) | All NEB barriers sorted by ΔE_fwd |
| `neb/diss_vib_rates.json` | Part 1, Phase E | Part 2 Phase 4 (k_diss, k_des at 5 T) | k_diss(T), k_des(T), k_diff(T) with Vineyard ν and ZPE |
| `results/lattice_params_vs_T.json` | Part 3, Phase 1b | Part 2 Phase 4 (a₀(T)), pipeline_run.py (auto-fill A0_M) | a₀(T) in Å at each of 5 temperatures |
| `results/diffusivity_arrhenius.json` | Part 3, Phase 3 | Part 2 Phase 4 (D₀, E_D), pipeline_run.py (auto-fill D0_M2S, E_D_EV) | D₀ (m²/s), E_D (eV), D(T) at 5 T |
| `neb_subsurface/hopa/ranked_hopa.json` | Part 2, Phase 1 | Part 2 Phase 3 (ΔE_entry, ΔE_exit, ΔH_entry) | Hop A NEB barriers and reaction energy |
| `neb_subsurface/hopb/ranked_hopb.json` | Part 2, Phase 2 | Part 2 Phase 3 (ΔE_mig) | Hop B NEB barrier |
| `results/rate_dict_T{T}K.json` | Part 2, Phase 4 | Part 2 Phase 5 (KMC event rates) | All 6 rate constants + k_drain at each T |
| `results/permeation_sweep_T{T}K.json` | Part 2, Phase 5 | Part 2 Phase 6 (Sieverts fit) | C₀(P,T) from KMC at each T and 40 P values |
| `results/permeability_T{T}K.json` | Part 2, Phase 6 | Final result | Φ from options A, B, C; Φ₀; E_Φ |

---

*Document generated from `pipeline.ipynb` source — 2026-06-22*

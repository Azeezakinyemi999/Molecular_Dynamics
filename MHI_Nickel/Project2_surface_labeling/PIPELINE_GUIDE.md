# Hydrogen Permeation Pipeline — Complete Guide

**Material:** Hastelloy N (Ni₇₁Mo₁₆Cr₇Fe₆ alloy)  
**Question:** How fast does hydrogen permeate through a structural membrane?  
**Method:** MACE ML potential + LAMMPS → NEB → TST → Richardson-Sieverts permeability

> **Scope note (2026-08).** The kinetic Monte Carlo stage has been removed. `models/kmc.py` is deleted and the four KMC-fed functions (`sweep_pressure`, `check_sieverts_law`, `classify_sieverts_regime`, `fit_solubility_from_kmc`) are gone from `models/permeation.py`. Solubility and permeability are unaffected — they were always computed from energies and TST rates, never from the KMC. What is lost is the **Sieverts-regime classifier**, the only test of whether `c ∝ √P` actually holds; Sieverts' law is now *assumed*. Oxides are treated as membranes rather than as adsorbing surfaces, so the `surface_limited` regime it detected is out of scope. Sections below describing the KMC engine are retained for historical context and marked accordingly.

---

## Table of Contents

1. [Why We Are Doing This](#1-why-we-are-doing-this)
2. [Physics Background](#2-physics-background)
   - 2.1 [Nudged Elastic Band (NEB)](#21-nudged-elastic-band-neb)
   - 2.2 [Zero-Point Energy Correction](#22-zero-point-energy-zpe-correction)
   - 2.3 [Transition State Theory and the Vineyard Prefactor](#23-transition-state-theory-tst--vineyard-prefactor)
   - 2.4 [Kinetic Monte Carlo (BKL algorithm)](#24-kinetic-monte-carlo-bkl-algorithm)
   - 2.5 [Richardson-Sieverts Permeability](#25-richardson-sieverts-permeability)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Quick Start](#4-quick-start)
5. [Part 1 — Surface NEB](#5-part-1--surface-neb-neb_calculationipynb)
6. [Part 3 — Bulk Diffusivity](#6-part-3--bulk-diffusivity-diffusivityipynb)
7. [Part 2 — Subsurface Permeation](#7-part-2--subsurface-permeation-permeationipynb)
8. [Master Pipeline](#8-master-pipeline-pipelineipynb)
9. [Models Reference](#9-models-reference)
10. [Reading the Key Output Files](#10-reading-the-key-output-files)
11. [Parameters Cheat Sheet](#11-parameters-cheat-sheet)

---

## 1. Why We Are Doing This

### The scientific question

Hastelloy N is a structural alloy used in molten-salt nuclear reactors (MSRs). Tritium, a radioactive hydrogen isotope produced by the molten-salt fuel, permeates through the alloy walls — potentially contaminating the coolant loop, escaping to the environment, or embrittling the metal. To assess this risk and design mitigation strategies (coatings, operation temperatures, membrane thickness), we need the **permeability** Φ(T): the rate at which hydrogen atoms pass through a unit-area, unit-thickness membrane at a given temperature and pressure differential.

Experimentally, measuring permeability at reactor temperatures (600–900 K) is slow and expensive. Computationally, we can extract it from first-principles atomistic simulations.

### The computational approach

A direct molecular dynamics (MD) simulation of H permeation would require milliseconds of simulation time — far beyond what MD can achieve. Instead, the pipeline uses a **multiscale strategy**:

1. **MACE ML potential** — A machine-learned interatomic potential (MACE-MH-1, trained for this project) that gives DFT-level accuracy at near-classical-MD cost. All energy evaluations in this project use MACE via LAMMPS with GPU (A100) acceleration through Kokkos.

2. **NEB** — Finds the minimum-energy path between two atomic configurations to get the energy barrier Eₐ for each H hop.

3. **TST + ZPE** — Converts barriers into rate constants k(T), corrected for quantum vibrational zero-point energy.

4. **Richardson-Sieverts** — Combines the energy-based solubility S(T) with the MD-derived bulk diffusivity D(T) to give the macroscopic permeability Φ(T) = D·S, and the flux at a stated feed pressure.

All heavy simulations run on a SLURM cluster. The notebooks generate run scripts and submit them automatically.

---

## 2. Physics Background

### 2.1 Nudged Elastic Band (NEB)

**What it is.** NEB is a method for finding the minimum energy path (MEP) between two known atomic configurations — an initial state (IS) and a final state (FS). A chain of images (intermediate atomic configurations) is interpolated between IS and FS, then simultaneously relaxed under a modified force that keeps images equally spaced along the path (the "spring" force) while each image descends toward the MEP (the "true" force projected perpendicular to the path).

**Why we need it.** Without knowing the barrier height Eₐ between two H positions, we cannot compute the hop rate. NEB gives us Eₐ directly from the potential energy landscape — no experimental input required.

**Climbing Image NEB (CINEB).** One image (the one closest to the saddle point) is released from the spring constraint and pushed uphill along the path direction. This image converges precisely to the transition state (TS), giving an accurate barrier.

**What the output means:**

- `Ea` (= `E_abs` in code) — forward barrier [eV]: energy you need to go from IS → FS
- `E_des` — reverse barrier [eV]: energy to go from FS → IS
- `delta_E` — reaction energy [eV]: FS energy minus IS energy (negative = exothermic hop)

For H₂ dissociation on the surface (IS = H₂* adsorbed, FS = 2 H* atoms), `Ea` is the dissociation barrier and `E_des` is the recombination barrier.

**Simulation parameters:**

- `N_REPLICAS = 9` — number of intermediate images. More images resolve the path better but cost linearly more CPU time. 9 is adequate for MACE.
- `SPRING_CONST = 1.0` eV/Å² — keeps images equally spaced. Too small and images bunch at IS/FS; too large and the path distorts.
- `NEB_FTOL = 0.1` eV/Å — convergence criterion on the maximum force component (a looser tolerance for faster NEB convergence).

**How it runs.** NEB in this project uses ASE's NEB implementation driven by LAMMPS (via `models/ase_neb.py`). Each NEB job is one SLURM task using 9 CPU cores (NEB is CPU-bound in this implementation; GPU NEB is less efficient for short paths).

---

### 2.2 Zero-Point Energy (ZPE) Correction

**What it is.** In quantum mechanics, even at 0 K, a harmonic oscillator has a ground-state energy of ½hν (where h is Planck's constant and ν is the vibrational frequency). This is the zero-point energy. Hydrogen, being the lightest atom, has large ZPE — often 0.05–0.20 eV per mode.

**Why it matters.** The classical NEB barrier Eₐ is the difference in potential energy between IS and TS. But the atom does not sit still at the classical minimum — it vibrates. The effective barrier that actually governs the rate is:

```
Eₐ(ZPE) = Eₐ(classical) + ZPE_TS − ZPE_IS
```

If the TS vibrational modes are softer (lower frequency) than the IS modes, ZPE_TS < ZPE_IS and the effective barrier is **lower** than the classical value. For H in metals, ZPE corrections typically lower the barrier by 0.05–0.15 eV — a factor of 2–5 in rate at 700 K.

**How we compute it.** A partial Hessian (second-derivative matrix of the potential energy) is computed at both IS and TS by finite differences using LAMMPS. Only the H atom and its nearest-neighbour metal atoms are displaced. Normal mode diagonalisation gives the vibrational frequencies. At the TS, one mode is imaginary (negative frequency) — this is the reaction coordinate mode and is excluded. The ZPE is:

```
ZPE = (1/2) × Σᵢ hνᵢ    [sum over real modes above 50 cm⁻¹]
```

The 50 cm⁻¹ threshold (`min_freq_cm1`) excludes near-zero translational/rotational artefacts from the partial Hessian.

**Where the results are stored.** ZPE-corrected barriers appear in `diss_vib_rates.json` (surface dissociation, Part 1) and in `rate_dict_T{T}K.json` (entry/exit hops, Part 2). Both files also contain the raw (uncorrected) barriers so you can see the correction size.

---

### 2.3 Transition State Theory (TST) & Vineyard Prefactor

**The rate equation.** TST gives the rate of a thermally activated hop as:

```
k(T) = ν × exp(−Eₐ / k_B T)
```

where:
- `ν` is the **attempt frequency** [s⁻¹] — how often the atom "tries" to cross the barrier
- `Eₐ` is the (ZPE-corrected) barrier [eV]
- `k_B = 8.617 × 10⁻⁵ eV/K` is Boltzmann's constant
- `T` is temperature [K]

**The Vineyard prefactor.** The naive choice ν = k_BT/h (Eyring theory) is too approximate for solid-state hops. The Vineyard (1957) formula uses the actual curvature of the energy landscape:

```
ν* = c × (∏ᵢ νᵢ^IS) / (∏ⱼ νⱼ^TS)
```

where the products are over all real normal-mode frequencies at IS and TS respectively, and `c = 2.998 × 10¹⁰ cm/s` is the speed of light (converts the one unpaired cm⁻¹ to s⁻¹). The TS has one fewer real mode than the IS (the imaginary mode is missing), so the ratio has one net cm⁻¹ dimension. Physically, ν* captures how "rigid" the binding site is relative to the transition state.

**Typical values:** 10¹²–10¹³ s⁻¹ for H in Ni-based alloys.

**Numerical stability.** The products of many small numbers would underflow to zero. The code evaluates everything in log-space (`sum of log(νᵢ)`) and converts back at the end.

**What the code produces.** `build_rate_dict()` in `models/tst_rates.py` assembles one entry per NEB label:

```python
{
  'k_forward': float,  # s⁻¹
  'k_reverse': float,  # s⁻¹
  'Ea_raw':    float,  # eV, classical NEB barrier
  'Ea_zpe':   float,   # eV, ZPE-corrected barrier
  'Ed_raw':   float,   # eV, classical reverse barrier
  'Ed_zpe':   float,   # eV, ZPE-corrected reverse barrier
  'nu':       float,   # s⁻¹, Vineyard attempt frequency
  'delta_e':  float,   # eV, FS − IS reaction energy
  'T_K':      float,   # K
}
```

---

### 2.4 Kinetic Monte Carlo (BKL algorithm)

**Why not just use MD?** Molecular dynamics integrates Newton's equations of motion with a timestep of ~0.5 fs. A typical H surface-to-subsurface hop at 700 K happens on a timescale of ~10⁻⁸ s. That would require 2 × 10¹⁰ MD steps — completely infeasible. KMC sidesteps this by treating each hop as a Poisson process with rate k(T) and advancing time by the actual physical waiting time between events.

**The BKL algorithm (Bortz-Kalos-Lebowitz, 1975):**

```
1. Build the event list: enumerate all possible events from the current
   grid state, each with its rate kᵢ.
2. Sum all rates:  Q = Σᵢ kᵢ
3. Advance time:   dt = −ln(u₁) / Q    where u₁ ~ Uniform(0,1)
4. Select event:   draw u₂ ~ Uniform(0, Q); find event i such that
                   Σⱼ<ᵢ kⱼ ≤ u₂ < Σⱼ≤ᵢ kⱼ  (binary search on cumulative rates)
5. Execute event i (update grid state in-place)
6. Repeat from step 1
```

This is "rejection-free" — every random draw results in an event, unlike rejection Monte Carlo which wastes draws on null moves. It is exact for systems where all rates are known.

**The physical model.** The KMC grid represents the FCC(111) surface plus **two** explicit subsurface interstitial layers — subsurface-1 (sub1) and subsurface-2 (sub2). Each grid point is one surface site (one metal atom of type Ni/Mo/Cr/Fe) with a sub1 and a sub2 interstitial site stacked beneath it. H enters at the surface, hops surface → sub1 → sub2, and drains from sub2 (the deepest explicit layer) into the bulk. Each sub1/sub2 site also carries an **environment label** — its coordination composition, e.g. `Ni6_oct` or `Ni4_tet` — so entry/exit rates are resolved per local environment rather than per element.

Events tracked at each step:

| Event | Kind | Rate | Source |
|---|---|---|---|
| H₂(g) adsorbs on empty pair → 2H* | `adsorb` | Hertz-Knudsen × k_diss | Part 1 NEB + pressure |
| 2H* desorbs → H₂(g) | `desorb` | k_des | Part 1 NEB |
| H* diffuses to neighbouring surface site | `surf_diff` | k_surf_diff | (optional) |
| H* surface → sub1 (Hop A) | `enter` | k_entry(sub1 env) | Part 2 Hop A NEB |
| H sub1 → surface (Hop A reverse) | `exit` | k_exit(sub1 env) | Part 2 Hop A NEB |
| H sub1 → sub2 (Hop B) | `hopB_enter` | k_hopB_entry(sub2 env) | Part 2 Hop B NEB |
| H sub2 → sub1 (Hop B reverse) | `hopB_exit` | k_hopB_exit(sub2 env) | Part 2 Hop B NEB |
| H sub2 → bulk drain | `drain` | k_drain | Part 3 diffusivity |

`k_diss`/`k_des` are keyed by sorted element pair; the four inter-layer rates are keyed by **oct-site environment** (surface⇄sub1 by the sub1 env, sub1⇄sub2 by the sub2 env), each with a per-class mean fallback so a site whose exact environment has no entry never becomes silently inert. `k_drain` applies 1-D Fick's law to a single oct–oct hop out of sub2: `k_drain = D / (a₀/√2)²`, where `a₀/√2` is the FCC oct–oct nearest-neighbour distance and D is the bulk diffusivity from Part 3.

**The Hertz-Knudsen adsorption rate** for one surface site:

```
R_strike = P × A_site / √(2π m_H₂ k_B T)
```

where `A_site = (a₀/√2)²` is the FCC(111) nearest-neighbour area. The actual adsorption rate is `k_diss × R_strike`.

**The grid.** A 40×40 alloy grid (default) with periodic boundary conditions. Each surface site is assigned an element (Ni/Mo/Cr/Fe) drawn from Hastelloy N composition:

```
Ni: 71%,  Mo: 16%,  Cr: 7%,  Fe: 6%
```

The sub1 and sub2 layers are populated with **environment labels** drawn from the real relaxed slab's interstitial-site environment distribution (done for every metal type now, not just oxides). A fixed `seed` makes the composition and environment draws reproducible across pressure points.

**Steady state.** `run_kmc_to_steady_state()` runs until the surface coverage θ and the sub2 population both converge (rolling-window check comparing successive windows). The reported `C0` is the **time-averaged sub2** concentration over the final window (mol H/m³) — sub2 being the layer that feeds the bulk — not a single final snapshot; averaging out the single-occupancy shot noise is what keeps the Sieverts C₀(√P) curve monotonic. `C0` feeds Fick's law to give the permeation flux J.

---

### 2.5 Richardson-Sieverts Permeability

**The macroscopic result.** For a membrane of thickness L under a pressure differential P_high − P_low, the steady-state H flux is:

```
J = Φ(T) × (√P_high − √P_low) / L      [mol H m⁻² s⁻¹]
```

The permeability Φ = D × S combines:
- **D(T)** — bulk diffusivity [m²/s], from Part 3 MD
- **S(T)** — Sieverts solubility [mol H m⁻³ Pa^(−½)], the H concentration per unit √P

**Sieverts' law.** For bulk-diffusion-limited transport, J ∝ √P. This arises because H₂ dissociates (H₂ ↔ 2H*) and the equilibrium H concentration scales as √P by detailed balance. If J–√P is **not** linear, it means the surface kinetics (adsorption/dissociation or subsurface entry) are rate-limiting, not bulk diffusion.

**Solubility as a per-environment Boltzmann sum.** The solubility is now built up **per interstitial environment** and Boltzmann-weighted, rather than from a single scalar enthalpy:

```
S(T) = S₀ × Σ_env  w_env · exp(−ΔH_sol(env) / k_BT)
```

where `w_env` is each environment's population weight and the sum runs over the distinct sub-site environments. Four routes are reported — two supply the prefactor S₀ for the Boltzmann sum, two are rate-based (reusing the well-sampled surface coverage rather than counting rare subsurface atoms) — plus a noise-limited counting diagnostic:

| Route | How S is obtained | Physical meaning |
|---|---|---|
| geometric | `S₀ = 4/a₀³/N_A` × per-env Boltzmann | 4 oct sites per FCC cell, **per mol** — the geometric site-density ceiling (`lattice_site_S0`). The `/N_A` reports S in mol H rather than atoms; results generated before commit `ed5bb11` (2026-07-28) omit it and are a factor of Avogadro too large |
| vibrational | partition-function S₀ × per-env Boltzmann | S₀ from the gas-phase-H₂ and dissolved-H vibrational partition functions (`vibrational_S0`); available only when the dissolved-H FS vibrations were computed |
| detailed_balance | `ρ_oct·(k_entry/k_exit)·√(k_diss·A/(k_des·…))`, population-weighted (`solubility_from_rates`) | **rate-based cross-check only** — routes the equilibrium solubility through kinetic rates and picks up a dissociation-rate-averaging artifact; not the reported solubility |

**geometric and vibrational are the solubility headline** — the equilibrium solubility is a thermodynamic quantity and is computed from energies (per-environment Boltzmann sum via `solubility_by_environment`, differing only in S₀). detailed_balance routes the same quantity through *kinetic* machinery (TST rates) and picks up a dissociation-rate-averaging artifact, so it is a **cross-check, not the reported solubility**. The `kmc_theta` and `option3` routes were removed with the KMC engine (2026-08), as was the older "detailed balance from a single representative TST rate" route.

**The Boltzmann sum is the dilute limit, and it has no upper bound.** `exp(−ΔH_sol/k_BT)` assumes θ ≪ 1, so an exothermic environment can drive S past one H per site — physically impossible, and observed: Hastelloy N 7's geometric route reaches `1.3e11` against a site density of `1.5e5`, driven by three tetrahedral environments holding ~10 % of the sites. The occupancy-limited counterpart `solubility_by_environment_saturating` replaces the bare Boltzmann factor with a Langmuir occupancy,

```
θ_env = K√(P/P_ref) / (1 + K√(P/P_ref)),   K = (S₀/ρ_site)·exp(−ΔH_sol(env)/k_BT)
S     = ρ_site · Σ_env w_env·θ_env / √(P/P_ref)
```

which reduces exactly to the Boltzmann form as θ → 0, so the dilute regime is unchanged. It is reported alongside (`saturating` in each route's payload: `S`, `S_dilute`, `theta_max`, `regime`) rather than replacing the dilute value, which remains the correct Sieverts constant where it applies. `regime` reuses the `classify_sieverts_regime` thresholds (`θ_max ≥ 0.85` → `saturated_only`, `≥ 0.4` → `partially_saturated`). Any route whose S exceeds `4/a₀³/N_A` should be read as "the dilute assumption has failed here", not as a solubility.

**Removed (2026-08).** The KMC's distinct deliverable was the **Sieverts-regime classifier** (`classify_sieverts_regime`): from the coverage isotherm's low-pressure exponent `θ ∝ P^n` it reported `sieverts_compatible` (n ≈ 0.5), `surface_limited` (n ≈ 1.0, e.g. oxides) or `saturated_only`. No thermodynamic route can answer that question, so it is simply gone; `sieverts_regime` is now written as `null`. The partial replacement is `solubility_by_environment_saturating`, whose `regime` field detects saturation (θ → 1) from the enthalpies alone but **cannot** detect a surface-limited surface.

**Per-environment solution enthalpy (referenced to sub1):**

```
ΔH_sol(env) = ½ ΔH_diss + ΔH_HopA(env)
```

Solubility stops at the first subsurface site (sub1); Hop B and deeper transport are bulk diffusion, carried by D, so `ΔH_HopB` is **not** part of the solubility (it is still computed and saved in `hopb_vib_rates.json`/`rate_dict_T{T}K.json` for other use). `ΔH_diss` is the H₂ dissociation reaction energy (mean `delta_E` from surface NEB); `ΔH_HopA(env)` is the H* → sub1 reaction energy for that sub1 environment. These are assembled per environment into `dH_sol_by_env.json` and auto-extracted from the NEB/vibration JSON files after Parts 1 and 2 run.

**Fick's flux (intermediate).** During the KMC pressure sweep, the pipeline also computes J via Fick's law to validate:

```
J = D × C₀ / L
```

This Fick estimate and the Richardson-Sieverts estimate should agree when Sieverts' law holds.

---

## 3. Pipeline Architecture

### Data flow

```
  ┌─────────────────────────────────────────────────────────┐
  │  INPUT: Hastelloy N slab (.lammps) + bulk structure     │
  └────────────────┬────────────────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
  PART 1                   PART 3
  neb_run.py               diffusivity_run.py
  ──────────               ──────────────────
  ├─ ranked_barriers.json  ├─ diffusivity_arrhenius.json
  ├─ diss_vib_rates.json   └─ lattice_params_vs_T.json
  └─ adsorption/h_atom/
       │                       │
       └──────────┬────────────┘
                  │  (both consumed by Part 2)
                  ▼
             PART 2
             permeation_run.py
             ──────────────────
             ├─ surface_sub1_sub2_map.json
             ├─ rate_dict_T{T}K.json
             ├─ dH_sol_by_env.json
             ├─ permeation_sweep_T{T}K.json
             ├─ permeability_T{T}K.json
             ├─ solubility_arrhenius.json
             ├─ permeability_arrhenius.json
             └─ solubility_arrhenius_kmc.json
```

Parts 1 and 3 are **independent** and run in parallel. Part 2 requires both to complete.

### Script-generator pattern

The notebooks in `calculation/` do **not** run the heavy simulations directly. Instead, each notebook:

1. Takes user parameters from Cell 2
2. Generates a self-contained Python script (`*_run.py`) with all parameters embedded
3. Generates a SLURM shell script (`*_run.sh`) that loads the correct environment and calls the Python script
4. (Optionally) submits the job with `sbatch`

The generated `*_run.py` scripts are the ones that actually import from `models/` and do the work. This means you can re-run a script on the cluster without re-running the notebook. The notebook is only needed to change parameters or to run analysis cells after completion.

### Execution environment

- **Cluster:** Northeastern Explorer, SLURM with A100 GPUs and CPU nodes, split across five partitions by job category:

  | Partition | Wall time | Used for |
  |---|---|---|
  | `gpu` | 8 h | Real chained MD only — slab surface relaxation (min → heat → NVT → quench) and bulk NPT lattice equilibration. These are the only two job types that actually integrate long trajectories on the GPU. |
  | `sharing` | 1 h | Every quick one-shot CG minimisation — H₂ reference energy, H₂*/H* adsorption energy, FS-min before NEB, bare-bulk min, bulk+H min, pre-pipeline bulk min, Hop A/B FS-min. |
  | `short` | 12 h (NEB), 6 h (vibrations) | CPU-only SLURM arrays: CI-NEB runs and vibrational-frequency (Hessian) jobs — no Kokkos GPU support for multi-replica NEB in LAMMPS. |
  | `multigpu` | 24 h | Long NVT production MD for bulk diffusivity (Phase 2) — the only job category that can genuinely run for a full day. |
  | `west` | 30 days | `pipeline_run.py` itself — a lightweight CPU orchestrator that only submits and polls child jobs, never runs LAMMPS/MACE directly. |

  The rule of thumb: if a job runs actual dynamics for tens of picoseconds or more, it goes on `gpu` (or `multigpu` for the longest NVT runs); if it is a single CG minimisation to a local energy minimum, it goes on `sharing`.
- **Python env:** `mace_env` (Python 3.9.23) with MACE, ASE, networkx, numpy, scipy
- **LAMMPS build:** custom build with MLIAP + Kokkos for GPU acceleration
- **MACE model:** MACE-MH-1 (project-trained potential; `.model` file for ASE, `.pt` file for LAMMPS)

---

## 4. Quick Start

### Full pipeline (recommended for a new material)

```bash
# 1. Open the master notebook
#    calculation/pipeline.ipynb

# 2. Edit Cell 2 — set your parameters:
#    WORK_DIR         = BASE_DIR/calculation                (cluster path, from config.py)
#    INPUT_STRUCTURES = [list of per-metal .lammps files]    (one entry per material)
#    TEMPERATURES     = [400, 600, 800]                      # K
#    dry_run          = True                                 # writes scripts only, no sbatch

# 3. Run Cells 1-2, then run the E_H2_GAS cell at the bottom of Cell 2 (see note below)

# 4. Run Cells 3-6 → generates neb_run_{stem}.py / diffusivity_run.py /
#    permeation_run_{stem}.py / pipeline_run.py + pipeline_run.sh, one
#    NEB + permeation script pair per metal in INPUT_STRUCTURES

# 5. Submit to SLURM (set dry_run=False in Cell 6, or submit the printed sbatch commands)
sbatch calculation/pipeline_run.sh

# 6. Monitor
squeue -u $USER

# 7. After completion, open each notebook and run the analysis cells
#    for plots and tables
```

**Note on `E_H2_GAS`:** This is the MACE energy of an isolated H₂ molecule in vacuum [eV] — a shared thermodynamic reference computed **once** and reused by every metal (not recomputed per material). Cell 2 calls `calculate_ref_adsorbate_energy(..., dry_run=dry_run)`, which checks `adsorption/ref_energies/h2_ref_energy.json`: if it already exists, `E_H2_GAS` is read from the cache immediately. If not, it writes (but does not submit, even with `dry_run=False`) an `h2_ref.sh` script and returns `None` — submit that script yourself with `sbatch`, wait for it to finish, then re-run Cell 2; it will detect the completed log and populate the cache without resubmitting. Every metal's `neb_run_{stem}.py` reads this same cached value at runtime and raises `RuntimeError` if it is still missing — no metal computes its own copy.

### Partial runs

Each part can be run independently from its own notebook:

| Notebook | Generates | Run when |
|---|---|---|
| `calculation/neb_calculation.ipynb` | `neb_run.py` | Surface NEB only |
| `calculation/diffusivity.ipynb` | `diffusivity_run.py` | Bulk diffusivity only |
| `calculation/permeation.ipynb` | `permeation_run.py` | After Parts 1 + 3 |
| `calculation/tst_calculation.ipynb` | TST rates inline | Debug/check rates |
| `calculation/subsurface_neb_calculation.ipynb` | Subsurface NEB inline | Debug Hop A/B |
| `calculation/kmc_calculation.ipynb` | KMC inline | Debug KMC |

---

## 5. Part 1 — Surface NEB (`neb_calculation.ipynb`)

### What it computes

H₂ dissociation barriers on the Hastelloy N FCC(111) surface. The surface has many different site-pairs (e.g., Ni-Ni, Ni-Mo, Mo-Cr, ...) because of the alloy's chemical disorder. Part 1 computes the NEB barrier for each sampled pair, ranks them by Eₐ, and computes ZPE-corrected dissociation/desorption rate constants via TST.

### Five phases of `neb_run.py`

| Phase | Name | What happens | Hardware |
|---|---|---|---|
| A | Slab relaxation | Build slab (`build_slab`, routed by `metal_type`); freeze bottom layers; **four-phase** relaxation — CG min → thermal anneal → NVT → quench (second CG min); enumerate surface sites | `gpu`, 8 h (real chained MD) |
| B | Site enumeration + adsorption | Use ACAT (alloy/pure) or the ontop+bridge oxide enumerator (`metal_type='oxide'`) to identify all unique surface sites; compute H₂* and H* adsorption energies for each site | `sharing`, 1 h (one-shot CG min) |
| C | NEB initial path | Generate IS (H₂* at site) and FS (2×H* at adjacent sites) pairs; FS-minimise; create linear interpolation; submit CINEB | `sharing` (FS-min) + `short` (CI-NEB array, 9 cores) |
| D | Barrier parsing | Read converged NEB log files; extract Eₐ, E_des, ΔH per transition; rank and write `ranked_barriers.json` | Local Python (no SLURM) |
| E | Vibrational analysis | Compute partial Hessian at IS and TS for each converged NEB; extract frequencies; apply ZPE; save `diss_vib_rates.json` | `short`, 6 h (CPU array) |

### Key parameters (Cell 2 of `neb_calculation.ipynb` / per-metal block of `pipeline.ipynb`)

```python
WORK_DIR        = os.path.join(BASE_DIR, 'calculation')
INPUT_STRUCTURES = [...]        # one .lammps file per metal — see multi-metal routing below
E_H2_GAS        = calculate_ref_adsorbate_energy(...)  # auto-computed once, shared cache (see Quick Start note)
LAYERS          = 12             # slab thickness; ignored for oxides (auto-matched to ~22 Å instead)
N_IMAGES        = 9              # NEB interpolation images
SPRING_CONST    = 1.0            # eV/Å²
NEB_FTOL        = 0.1            # eV/Å — convergence tolerance
TEMPERATURES    = [400, 600, 800]  # K — for rate computation in Phase E
```

### How the slab is built

`build_slab()` routes on `metal_type` (`'alloy'`, `'pure'`, or `'oxide'`, assigned per input structure by `classify_metal()` in `pipeline.ipynb` Cell 2):

- **alloy** (e.g. Hastelloy N, `bestsqs*`): Ni-FCC geometry template; element symbols randomly shuffled from the bulk composition fractions, seeded per-structure (`slab_seed` derived from the numeric suffix in the stem, e.g. `Hastelloy_N_42` → seed 42).
- **pure** (e.g. `Al_supercell`, `Ni_supercell`, `Fe_supercell`): correct FCC or BCC geometry from the crystal-structure map; every site is the single element, no shuffle.
- **oxide** (e.g. `Cr_oxide_supercell`): primitive cell extracted via spglib; stoichiometry preserved exactly; `LAYERS` is ignored — the repeat count is chosen automatically to match a ~22 Å target thickness instead.

Bottom layers are frozen (`Z_FREEZE_CUTOFF`, auto = bottom 1/3 of slab thickness unless overridden) to simulate the bulk; the remainder relax freely through the four-phase protocol above. H atoms are placed at hollow/bridge/ontop sites enumerated in Phase B. Element types for alloy/pure structures: Al, B, C, Cr, Fe, Mo, Ni = types 1–7, H = type 8 (`ELEM_STR_7`); oxide structures add O as type 8 and H as type 9 (`ELEM_STR_10`).

Two structures are currently skipped for Parts 1/2 (surface NEB + permeation) via `SKIP_OXIDE_STEMS` / `is_pure_bcc_structure()` in `pipeline.ipynb` Cell 2, though they still run through Part 3 (bulk diffusivity) unaffected: `Ni_oxide_supercell` (polar Tasker-III NiO(111) termination — GitHub #5) and any pure BCC structure (surface/subsurface untested — GitHub #6).

### Key output files

**`neb/ranked_barriers.json`** — all converged NEB barriers, sorted by Eₐ:
```json
[
  {"Ea": 0.312, "E_des": 0.418, "delta_E": -0.106, "pair": ["Ni","Ni"],
   "sid": "Ni3Mo", "converged": true},
  ...
]
```

**`neb/diss_vib_rates.json`** — ZPE-corrected TST rates per element-pair label:
```json
{
  "hopa_Ni3Mo": {
    "pair": ["Ni", "Mo"],
    "Ea_raw": 0.312, "Ea_zpe": 0.274,
    "Ed_raw": 0.418, "Ed_zpe": 0.380,
    "nu": 2.31e12,
    "label": "hopa_Ni3Mo"
  }
}
```

**`adsorption/h_atom/h_atom_{sid}_relaxed.lammps`** — relaxed single H* structures. These become the initial states (IS) for Part 2 Hop A NEB.

---

## 6. Part 3 — Bulk Diffusivity (`diffusivity.ipynb`)

### What it computes

Temperature-dependent bulk H diffusivity D(T) and equilibrium lattice parameter a₀(T) from classical MD. This is done by tracking the mean-squared displacement (MSD) of H atoms in a bulk Hastelloy N supercell across multiple temperatures, then fitting the Arrhenius equation.

### Why this is Part 3 (not Part 2)

Part 3 runs in parallel with Part 1. Part 2 requires D(T) from Part 3 to compute the bulk drain rate in KMC and the final permeability. Part 3 is independent of the surface NEB work.

### Four phases of `diffusivity_run.py`

| Phase | Name | What happens | Hardware |
|---|---|---|---|
| 1a | CG minimisation | Conjugate-gradient energy minimisation of the bare bulk structure to find the T=0 reference geometry. Runs **once per structure**, shared across all `N_H_VALUES` (not recomputed per H concentration). | `sharing`, 1 h |
| 1b | NPT equilibration | Run NPT MD at each T to let the box relax to the thermal lattice constant a₀(T). Also runs **once per structure** (hoisted out of the per-`n_H` loop, since a₀(T) does not depend on H concentration); insert N_H hydrogen atoms and re-minimise the bulk+H structure separately per `(structure, n_H, T)` combination. | `gpu`, 8 h (NPT — real chained MD); `sharing`, 1 h (bulk+H re-min) |
| 2 | NVT production | Long NVT MD (fixed volume, fixed T); dump atomic positions every N steps for MSD calculation; auto-restarts if SLURM wall time is hit (job chaining) | `multigpu`, 24 h |
| 3 | MSD analysis | Compute MSD(t) from position dumps; fit to `MSD = 6Dt` for 3D diffusion; repeat across temperatures; Arrhenius fit: `ln(D) vs 1/T` → D₀ and E_D | Local Python |

Phases 1a and 1b-NPT are the two costliest steps per structure, so hoisting them out of the `N_H_VALUES` loop (they used to be recomputed redundantly for every H concentration) was a significant fix — see `audits/task_B_audit.md` follow-up note. Each phase writes its own failure to a shared `_FAILURES` list rather than crashing the whole run; `diffusivity_failures.json` records which `(structure, n_H)` combinations failed, and the script exits non-zero only if at least one failure was recorded.

### Arrhenius diffusivity

The temperature-dependent bulk diffusivity follows:

```
D(T) = D₀ × exp(−E_D / k_B T)      [m²/s]
```

`D₀` is the diffusivity pre-exponential [m²/s] and `E_D` is the activation energy for bulk H diffusion [eV]. Both come from the Arrhenius fit to the MSD data.

### Key parameters (Cell 2 of `diffusivity.ipynb`)

```python
INPUT_STRUCTURES = [...]                      # one entry per metal (11 structures currently)
TEMPERATURES     = [400, 600, 800]            # K — shared with Parts 1 and 2
N_H_VALUES       = [1, 3, 5, 10]              # H atoms to insert
N_PROD_STEPS     = 5_000_000                  # timestep 0.5 fs → 2.5 ns
NVT_WALL_TIME    = '24:00:00'                 # jobs auto-resubmit if needed
```

**Why multiple N_H values?** Dilute limit is what we want (one H atom in a sea of metal). Running several concentrations lets you extrapolate D to the dilute limit and check that D is concentration-independent (a sign that the force field is behaving correctly).

### Key output files

**`results/diffusivity_arrhenius.json`** — Arrhenius fit parameters:
```json
{
  "D0_m2s":      2.14e-7,
  "E_D_eV":      0.421,
  "D0_err":      1.5e-8,
  "E_D_err_eV":  0.015,
  "R2_fit":      0.9987
}
```
This file is **automatically read by Part 2** — no manual transcription needed.

**`results/lattice_params_vs_T.json`** — thermal lattice parameter:
```json
{
  "temperatures": [400, 600, 800],
  "a0_m": [3.514e-10, 3.522e-10, 3.530e-10]
}
```
This file is also **automatically read by Part 2** so that KMC and permeability calculations use the correct a₀ at each T (rather than a fixed room-temperature value).

**`{run}/analysis/diffusivity_table.txt`** — human-readable D ± σ at each temperature for checking.

---

## 7. Part 2 — Subsurface Permeation (`permeation.ipynb`)

### What it computes

Everything from H* on the surface to the final permeability number:
- Hop A barriers (H* → first subsurface octahedral site)
- Hop B barriers (first → second subsurface octahedral site)
- ZPE-corrected TST rates for all hops
- KMC pressure sweep → steady-state H concentration vs pressure
- Richardson-Sieverts permeability Φ(T) and flux J

### What "Hop A" and "Hop B" mean physically

**Hop A** — H* surface → subsurface-1 (sub1):  
H leaves the FCC(111) surface hollow site and enters the first subsurface interstitial site (octahedral **or** tetrahedral) directly beneath it. The H atoms that attempt this are exactly the **dissociation products** — the 2H* final states of each converged surface-dissociation NEB — not a wholesale enumeration of every adsorption site; the H that permeates is the H that dissociated. Each entry H* is mapped to its nearest sub1 interstitial and is never dropped (Finding A). This is the critical entry step — if k_entry is slow, surface kinetics limit permeation.

**Hop B** — sub1 → subsurface-2 (sub2):  
H moves from the first subsurface interstitial site to the second one directly below. This hop determines how fast H moves from the surface region into the bulk. After sub2, H is effectively in the bulk and diffuses at the bulk rate D(T). The sub1/sub2 layer indices are derived from the slab's actual layer count rather than hardcoded — see Section 9, `subsurface_graph.py`.

### Six phases of `permeation_run_{stem}.py`

| Phase | Name | What happens | Hardware |
|---|---|---|---|
| 1 | Hop A NEB | Build subsurface graph (NetworkX); seed entry H* from the dissociation products (`entry_h_sources.json`); map each to its nearest sub1 interstitial (`surface_sub1_sub2_map.json`); FS-min; run CINEB for H*→sub1 | `sharing` (FS-min) + `short` (CI-NEB array) |
| 2 | Hop B NEB | Use Hop A relaxed sub1 structures as IS; generate sub2 FS structures via the `sub1↔sub2` map; run CINEB for sub1→sub2 | `sharing` (FS-min) + `short` (CI-NEB array) |
| 3 | Vibrations | Hessian + normal modes at IS, TS, and FS (dissolved-H) for all Hop A and Hop B NEB jobs; ZPE-corrected barriers; FS modes also feed the vibrational-S₀ route | `short`, 6 h (CPU array) |
| 4 | TST rates | Vineyard prefactor; Arrhenius rates at each temperature; per-environment rate assembly (`env_rate_dict`); write `rate_dict_T{T}K.json` + the env-carrying `hopa_ranked.json`/`hopb_ranked.json` and `hopa_vib_rates.json`/`hopb_vib_rates.json` | Local Python |
| 5 | Permeability | Per-environment Boltzmann solubility, geometric + vibrational S₀ routes plus the detailed-balance cross-check; Φ(T) = D×S at each T; Arrhenius fits of S(T) and Φ(T) (Φ₀ = D₀·S₀, E_Φ = E_D + ΔH_sol); Richardson flux at `OPERATING_P_HIGH_PA` | Local Python |

Each phase (Hop A/B submission, permeability per T) is guarded by an existence check on its own output file, so a restarted `permeation_run_{stem}.py` skips whatever already completed rather than resubmitting — see `audits/task_F_audit.md`.

### Auto-extracted values

After Parts 1 and 3 finish, `permeation_run_{stem}.py` reads the following at startup, per metal, without any manual input:

| Value | Source | What it is |
|---|---|---|
| `DH_DISS_EV` | lowest-barrier entry in `neb/ranked_barriers.json` | H₂ dissociation reaction energy |
| `DH_ENTRY_EV` | converged Hop A NEB result | H* → H_sub reaction energy |
| `D0_M2S`, `E_D_EV` | `results/{stem}_{n_h}H/diffusivity_arrhenius.json` — loaded **per `(stem, n_H)` pair**, not once per metal | Bulk diffusivity Arrhenius parameters |
| `a₀(T)` | `lattice_params_vs_T.json` | Thermal lattice constant per temperature |

If Part 3 has not been run yet, Part 2 falls back to the fixed `A0_M` value from Cell 2. If a given `(stem, n_H)`'s `diffusivity_arrhenius.json` is missing or invalid, that H-concentration is **skipped entirely** (not substituted with a placeholder D value) — `permeation_status.json` records which `n_H` were skipped and why, and the script exits non-zero only if zero H-concentrations produced a result.

### Key parameters (Cell 2 of `permeation.ipynb` / shared config in `pipeline.ipynb`)

```python
WORK_DIR        = os.path.join(BASE_DIR, 'calculation')
TEMPERATURES    = [400, 600, 800]              # K — shared with Parts 1 and 3
A0_M            = 3.52e-10                    # m — fallback only if Part 3 not run
L_M             = 1e-3                        # m — membrane thickness (1 mm)
NX, NY          = 40, 40                      # KMC grid dimensions
KMC_MAX_STEPS   = 500_000                     # hard cap on KMC steps
P_VALS_PA       = list(np.logspace(-5, 6, 40))  # Pa — 40 log-spaced points, 1e-5 to 1e6
DH_DISS_EV      = None   # set to override auto-extraction
DH_ENTRY_EV     = None   # set to override auto-extraction
```

### Key output files

**`results/rate_dict_T700K.json`** — TST rates at 700 K:
```json
{
  "hopa_Ni3Mo": {
    "k_forward": 1.24e8, "k_reverse": 3.11e6,
    "Ea_raw": 0.421,     "Ea_zpe": 0.389,
    "Ed_raw": 0.318,     "Ed_zpe": 0.286,
    "nu": 2.31e12,       "delta_e": -0.103,
    "T_K": 700.0
  },
  ...
}
```

**`results/permeation_sweep_T700K.json`** — KMC pressure sweep results:
```json
{
  "P_vals":     [1e3, 1e4, 1e5, 5e5, 1e6],
  "J_vals":     [1.2e17, 3.8e17, 1.2e18, 2.7e18, 3.8e18],
  "C0_vals":    [3.1e24, 9.8e24, 3.1e25, 6.9e25, 9.8e25],
  "converged":  [true, true, true, true, true]
}
```

**`results/{stem}_{n_H}H/permeability_T700K.json`** — all three Φ options at 700 K (written per H-concentration):
```json
{
  "T_K": 700.0, "n_H": 1,
  "D0_m2s": 2.14e-7, "E_D_eV": 0.421,
  "dH_sol_mean_eV": 0.18, "n_env": 6,
  "option1": {"S0": 4.3e23, "S": 4.3e14, "Phi": 1.8e-15, "J": ...,
              "route": "geometric S0, per-env Boltzmann"},
  "option2": {"S0": ..., "S": ..., "Phi": ..., "J": ...,
              "route": "vibrational S0, per-env Boltzmann"},
  "option3": {"S": ..., "Phi": ..., "J": ..., "S_std": ..., "n_converged": ...,
              "route": "KMC empirical Sieverts fit"}
}
```

When the dissolved-H FS vibrations were not computed, `option2`'s values are `null` and its `route` reads `"vibrational S0 unavailable (no FS vibrations)"`.

**`results/{stem}_{n_H}H/solubility_arrhenius.json`** — multi-T `ln(S) vs 1/T` fit for **each** solubility route (`geometric`, `vibrational`, `kmc`), giving `S0`, `dH_sol_eV`, and the fit `r2` (which doubles as a curvature flag — a per-env S(T) is a sum of Arrhenius terms, so `r2 < 1` is physical, not an error).

**`results/{stem}_{n_H}H/permeability_arrhenius.json`** — the Arrhenius permeability per route: `Phi0 = D0·S0` and `E_phi_eV = E_D + dH_sol`.

**`results/{stem}_{n_H}H/solubility_arrhenius_kmc.json`** — the KMC-route-only Arrhenius fit, kept for backward compatibility (consumed by `plot_arrhenius_S0`).

### Plots produced

- `barriers_overview.png` — histogram of all Hop A and Hop B barriers
- `mep_hopA.png` / `mep_hopB.png` — minimum energy path for the lowest-barrier Hop A and Hop B
- `sieverts_check.png` — J vs √P scatter with linear fit; R² value tells you whether bulk or surface is limiting
- `permeability_vs_T.png` — ln(Φ) vs 1/T Arrhenius plot comparing all three options
- `solubility_arrhenius.png` — ln(S) vs 1/T for Option 3
- `bottleneck.png` — comparison of k_entry vs k_exit vs k_drain to identify rate-limiting step

---

## 8. Master Pipeline (`pipeline.ipynb`)

### Multi-metal routing (Cell 2)

`pipeline.ipynb` runs the whole campaign across every structure in `INPUT_STRUCTURES` in one pass, not just one material. Cell 2's `classify_metal(path)` inspects the filename stem and tags each structure `'oxide'` (name contains `oxide`), `'alloy'` (name contains `hastelloy`/`bestsqs`/`sqs`/`alloy`), or `'pure'` (everything else — `Al_supercell`, `Fe_supercell`, `Ni_supercell`, ...). This builds `METAL_CONFIGS`, a list of per-structure dicts (`stem`, `type`, `elem_str`/`e2t`/`masses` — 7-element table for alloy/pure, 10-element table with O for oxide). A `skip_surface` flag is also computed per structure (see the skip list at the end of "How the slab is built" in Section 5) — skipped structures still run through Part 3 but are excluded from Cells 3 and 5.

Cells 3 and 5 loop over `METAL_CONFIGS` and write one `neb_run_{stem}.py` and one `permeation_run_{stem}.py` per (non-skipped) structure; Cell 4 writes a single `diffusivity_run.py` that internally loops over all of `INPUT_STRUCTURES × N_H_VALUES` (diffusivity doesn't need the surface-NEB skip list). The per-metal `slab_seed` is derived from the numeric suffix in the stem (`Hastelloy_N_42_supercell` → 42; falls back to 7 if no digits are found).

### What `pipeline_run.py` does

Orchestrates every metal's Part 1, the shared Part 3, and every metal's Part 2, in the correct order, with no manual intervention between steps.

```
sbatch pipeline_run.sh
    └─ python pipeline_run.py
           │
           ├── subprocess.Popen(neb_run_{stem}.py)     # one per metal, Part 1 — all launched in parallel
           ├── subprocess.Popen(diffusivity_run.py)     # Part 3 — also launched in parallel with the NEBs
           │                                            # (shared across all metals internally)
           ├── proc.wait() for every launched process   # block until ALL NEB scripts + diffusivity finish
           ├── check each return code — any non-zero → recorded, does not block the rest
           └── for stem in metals:                      # Part 2 — run SEQUENTIALLY, one metal at a time
                   subprocess.run(permeation_run_{stem}.py)
                   non-zero return code → exit 1
```

The pipeline script itself is lightweight — it just launches and waits. All the physics happens inside the sub-scripts. Permeation runs sequentially (not in parallel) because each `permeation_run_{stem}.py` submits its own SLURM arrays for Hop A/B and can saturate the `short`/`sharing` partitions on its own; running all metals' Part 2 simultaneously would oversubscribe those partitions for no benefit (Part 2 is far cheaper than Parts 1/3 per metal).

### When to use individual notebooks vs the master pipeline

| Situation | Use |
|---|---|
| First full run on a new material | `pipeline.ipynb` → `sbatch pipeline_run.sh` |
| Re-run only Part 2 with new KMC parameters | `permeation.ipynb` directly |
| Debug a failed NEB job | `neb_calculation.ipynb` Phase C/D cells |
| Check rates without re-running NEB | `tst_calculation.ipynb` |
| Test KMC at a single temperature | `kmc_calculation.ipynb` |

---

## 9. Models Reference

The `models/` directory contains the reusable library that all notebooks and generated scripts import from. Here is what each module does.

### Foundational modules

**`config.py`** — Project-wide constants. LAMMPS binary path, MACE model paths, `BASE_DIR`, element-to-type mappings (`E2T_7`, `E2T_10`), mass tables, SLURM defaults, and simulation constants (`N_REPLICAS`, `SPRING_CONST`, `NEB_FTOL`, `TIMESTEP`). Change this file if you move the cluster installation.

**`lammps_script.py`** — Writes LAMMPS input files programmatically. Functions for CG minimisation, NPT equilibration, NVT production, and NVT restart scripts. Handles Kokkos flags, MACE pair style, and atom type mappings automatically.

**`create_slurm.py`** — SLURM job submission utilities. `write_slurm_job()` writes a `.sh` file with module loads, conda activation, and LD_LIBRARY_PATH setup. `submit_slurm_job()` calls `sbatch`. `wait_for_jobs()` polls `squeue` until job IDs complete.

**`parsers.py`** — Parse LAMMPS output. `parse_barrier_file()` reads `neb_barrier.txt` and returns `{Ea, E_des, delta_E, converged}`. `parse_minimisation_log()` extracts final energy and force from a minimisation log. `parse_thermo_output()` reads temperature, pressure, and lattice parameters from NVT/NPT thermo output.

### Structure and energetics

**`structure.py`** — Build and manipulate atomic structures. `build_slab()` creates the surface slab with frozen layers, routed by `metal_type` (`'alloy'`/`'pure'`/`'oxide'` — see "How the slab is built" in Section 5). `insert_hydrogen()` places H atoms at FCC octahedral interstitial sites. `get_lattice_parameter_from_dump()` reads the thermally-averaged a₀(T) from an NPT box-dimension dump (averages the last N frames rather than trusting a single snapshot — see `audits/task_G_audit.md`); `get_lattice_parameter()` remains for reading a₀ from a single minimised structure file (used inside `insert_hydrogen()`, not for NPT output). `is_pure_bcc_structure()` flags pure BCC inputs for the surface-step skip list.

**`energetics.py`** — Compute and summarise energetics. Parses adsorption energies from batched LAMMPS runs. `summarise_neb_barriers()` reads all `neb_barrier.txt` files in a directory and produces `ranked_barriers.json` sorted by Eₐ.

### NEB modules

**`ase_neb.py`** — Runs a single NEB calculation as an ASE NEB object backed by LAMMPS. Called by SLURM array job workers (one job per IS/FS pair). Handles CINEB image convergence and writes `neb_barrier.txt`.

**`neb_workflow.py`** — Orchestrates the full surface NEB pipeline (Phases A–E), threading `metal_type` and `slab_seed` through every phase. Generates all job scripts, submits SLURM arrays, waits for completion, and parses results. Contains the `_NEB_BODY` f-string that becomes `neb_run_{stem}.py` (one per metal) when `write_neb_run_script()` is called from `pipeline.ipynb` Cell 3. Also exposes `calculate_ref_adsorbate_energy()` — the shared H₂-gas reference-energy helper used once in Cell 2, not per metal.

**`neb_subsurface.py`** — Orchestrates Hop A and Hop B NEB and builds the entry maps. `collect_entry_h_sources()` seeds the entry H* from the converged dissociation NEB products (`entry_h_sources.json`); `build_sub1_sub2_map()` maps each sub1 interstitial to its nearest sub2 (`sub1_sub2_map.json`); `build_surface_sub1_sub2_map()` maps each entry H* to its nearest sub1, never dropping one (Finding A; `surface_sub1_sub2_map.json`). `orchestrate_hopa_neb()`/`orchestrate_hopb_neb()` are map-driven (one job per mapped path) and carry the sub1/sub2 environment + oct/tet type in each job dict. `classify_relaxed_h_env()` re-classifies each hop's environment from where H actually sits in the relaxed FS (Finding B). Writes the barrier files that feed Phases 3 and 4 of `permeation_run_{stem}.py`.

**`subsurface_graph.py`** — Builds a NetworkX graph of interstitial sites in the slab (octahedral/tetrahedral for alloy/pure metals via rank-based layer binning; any-coordination "interstitial" sites for oxides via gap-based layer binning, `metal_type='oxide'`). The total layer count `N` is derived from the slab's construction metadata (`_n_layers_from_metadata()`: `n_atoms_total // n_atoms_surface` from `surface_sites.json`) for metals, with gap-based z-clustering kept only as a fallback — gap clustering over-counted a relaxed 12-layer slab as 17. `subsurface_1`/`subsurface_2` are then derived as `N-1`/`N-2`. Each node is a sub-site; edges connect geometrically adjacent sites, including the `sub1↔sub2` edges needed for Hop B (previously missing — see `audits/oxide_support_plan.md`). `build_subsurface_graph()` parses the relaxed slab structure and identifies all sub1 and sub2 sites, keyed by the overlying surface site ID. This graph determines which FS structures to generate for Hop A/B.

### Vibrational analysis and rates

**`vibrations.py`** — Computes partial Hessian by finite displacements via LAMMPS. `orchestrate_vibrations()` submits a SLURM array of displacement jobs, collects forces, assembles the Hessian, diagonalises it, and writes `vib_frequencies.json` containing real and imaginary mode frequencies. The partial Hessian displaces only the H atom and its coordination shell (~10 metal atoms) rather than all atoms, making it computationally feasible.

**`tst_rates.py`** — Converts NEB barriers and vibrational frequencies into TST rate constants. Pipeline: `collect_neb_results` → `split_vib_results` → `apply_zpe_correction` → `vineyard_prefactor` → `arrhenius_rate` → `build_rate_dict` → `rates_to_json`. `env_rate_dict()` groups the Hop A/B rates by oct-site environment (arithmetic mean of the Arrhenius rates within each environment) to produce the env-keyed `k_entry`/`k_exit` and `k_hopB_entry`/`k_hopB_exit` the KMC consumes; `write_hop_ranked()`/`write_hop_vib_rates()` emit the env-carrying per-hop artifacts; `vib_partition_function()` and `h2_gas_partition_function()` supply the dissolved-H and gas-phase-H₂ partition functions for the vibrational-S₀ route. All steps are pure Python; no LAMMPS calls.

### KMC and macroscopic transport

**`kmc.py`** — Two-layer BKL KMC engine. `make_grid(nx, ny, composition, seed, sub1_env_composition, sub2_env_composition)` creates the surface + sub1 + sub2 grid, with occupancy arrays (`surface_occ`, `sub1_occ`, `sub2_occ`) and per-cell environment labels (`sub1_env`, `sub2_env`). `build_event_list()` enumerates the `enter`/`exit` (surface⇄sub1), `hopB_enter`/`hopB_exit` (sub1⇄sub2), `diss`/`des`, and `drain` (sub2→bulk) events, looking rates up per environment with a per-class mean fallback (`_rate_lookup`, never a silent 0.0). `run_kmc()` runs a fixed number of steps; `run_kmc_to_steady_state()` runs until θ and the sub2 population converge and returns `{C0, t_total, theta_ss, converged, n_steps}`, where `C0` is the time-averaged sub2 concentration.

**`permeation.py`** — Macroscopic permeability from KMC results. `sweep_pressure()` calls `run_kmc_to_steady_state` at each pressure point (threading the sub1/sub2 environment compositions through to `make_grid`) and returns J vs P data. `check_sieverts_law()` fits J vs √P and diagnoses the rate-limiting step. Solubility: `build_dh_sol_by_env()` assembles the per-environment ΔH_sol, `solubility_by_environment()` does the Boltzmann-weighted sum, and `lattice_site_S0()` (geometric) and `vibrational_S0()` (partition-function) supply the two S₀ prefactors, while `fit_solubility_from_kmc()` gives the KMC-empirical route. `fit_arrhenius()`, `permeability()`, `permeability_arrhenius()`, and `richardson_flux()` give the Arrhenius S/Φ parameters and the final Φ and J.

### Diffusivity

**`diffusivity_post_processing.py`** — Analyses LAMMPS dump files from NVT MD. Computes MSD(t) by unwrapping periodic boundary conditions. Fits `MSD = 6Dt` using linear regression over the diffusive regime. `fit_arrhenius_diffusivity()` takes `{T: D}` data and returns `D₀`, `E_D`, and fit quality R².

### Workflow orchestrators

**`neb_workflow.py`**, **`permeation_workflow.py`**, **`diffusivity_workflow.py`**, **`pipeline_workflow.py`** — Each contains a function that writes a complete self-contained `*_run.py` script (as an f-string header + raw r-string body) plus a matching SLURM `.sh` file. The notebooks call these functions, passing the user's Cell 2 parameters, and the workflows embed those parameters in the generated scripts.

---

## 10. Reading the Key Output Files

All output files live under `{WORK_DIR}/results/`. Here is the exact structure of each key file.

### `ranked_barriers.json`
Sorted list of all converged surface NEB barriers (smallest Eₐ first):
```json
[
  {
    "Ea":        0.312,
    "E_des":     0.418,
    "delta_E":  -0.106,
    "pair":     ["Ni", "Ni"],
    "sid":      "Ni3Mo",
    "converged": true
  },
  ...
]
```
Note: the code uses the key `"Ea"` (not `"E_abs"`). `delta_E` = FS energy − IS energy.

### `diss_vib_rates.json`
ZPE-corrected TST rates for surface dissociation, keyed by label:
```json
{
  "hopa_Ni3Mo": {
    "pair":   ["Ni", "Mo"],
    "Ea_raw": 0.312,
    "Ea_zpe": 0.274,
    "Ed_raw": 0.418,
    "Ed_zpe": 0.380,
    "nu":     2.31e12,
    "label":  "hopa_Ni3Mo"
  }
}
```

### `diffusivity_arrhenius.json`
Bulk diffusivity Arrhenius fit:
```json
{
  "D0_m2s":     2.14e-7,
  "E_D_eV":     0.421,
  "D0_err":     1.5e-8,
  "E_D_err_eV": 0.015,
  "R2_fit":     0.9987
}
```
Use `D(T) = D0_m2s × exp(−E_D_eV / (8.617e-5 × T))` to get D at any T.

### `lattice_params_vs_T.json`
Thermal equilibrium lattice parameter at each temperature:
```json
{
  "temperatures": [400, 600, 800],
  "a0_m":         [3.514e-10, 3.522e-10, 3.530e-10]
}
```
Indexing: `a0_m[i]` corresponds to `temperatures[i]`.

### `rate_dict_T700K.json`
TST rates at 700 K for all converged hops:
```json
{
  "hopa_Ni3Mo": {
    "k_forward": 1.24e8,
    "k_reverse": 3.11e6,
    "Ea_raw":    0.421,
    "Ea_zpe":    0.389,
    "Ed_raw":    0.318,
    "Ed_zpe":    0.286,
    "nu":        2.31e12,
    "delta_e":  -0.103,
    "T_K":       700.0
  },
  "hopb_Ni3Mo": { ... },
  ...
}
```
This file keeps the per-hop schema (one entry per `hopa_`/`hopb_` label). The env-keyed KMC `rate_dict` (`k_entry`/`k_exit`/`k_hopB_entry`/`k_hopB_exit`, keyed by oct-site environment) is assembled from these rates by `env_rate_dict()`, which arithmetic-means the Arrhenius rates within each environment — not a direct 1:1 copy of `k_forward`/`k_reverse`.

### `permeability_T700K.json`
All three permeability options at 700 K (written per H-concentration under `results/{stem}_{n_H}H/`):
```json
{
  "T_K":    700.0,
  "n_H":    1,
  "D0_m2s": 2.14e-7,
  "E_D_eV": 0.421,
  "dH_sol_mean_eV": 0.18,
  "a0_m":   3.52e-10,
  "n_env":  6,
  "option1": {
    "S0":    4.3e23,
    "S":     4.3e14,
    "Phi":   1.8e-15,
    "J":     1.8e12,
    "route": "geometric S0, per-env Boltzmann"
  },
  "option2": {
    "S0": ..., "S": ..., "Phi": ..., "J": ...,
    "route": "vibrational S0, per-env Boltzmann"
  },
  "option3": {
    "S": ..., "Phi": ..., "J": ...,
    "S_std": ..., "n_converged": ...,
    "route": "KMC empirical Sieverts fit"
  },
  "P_high_Pa": 1e6,
  "L_m": 1e-3
}
```
`Phi` has units **mol H·m⁻¹·s⁻¹·Pa^(−½)** — the H-atom counts are converted to moles internally (÷ Avogadro, `_N_A`), so S, Φ, and J are all reported per mol H. Two conventions to note when comparing to literature: (1) results are per **√Pa** — literature per √bar/√atm differs by √(10⁵) ≈ 316; (2) per **mol H** (atomic) — for a flux per mol H₂, divide J by 2. When the dissolved-H FS vibrations were not run, `option2`'s numeric fields are `null` and its `route` says so.

---

## 11. Parameters Cheat Sheet

### Parameters that must be set manually

| Parameter | Notebook Cell | Physical meaning | How to determine | Typical range |
|---|---|---|---|---|
| `E_H2_GAS` | pipeline Cell 2 | MACE energy of gas-phase H₂ [eV] | Auto-computed once via `calculate_ref_adsorbate_energy()` and cached (see Quick Start note) — not hand-entered | ~−6.79 eV (MACE-MH-1) |
| `L_M` | permeation Cell 2 | Membrane thickness [m] | Experimental membrane geometry | 0.5 mm – 5 mm |

### Parameters auto-extracted after pipeline runs

| Parameter | Source file | Physical meaning |
|---|---|---|
| `DH_DISS_EV` | `ranked_barriers.json` (mean delta_E) | H₂ dissociation reaction energy [eV] |
| `DH_ENTRY_EV` | Hop A NEB results (mean delta_E) | H* → H_sub reaction energy [eV] |
| `D0_M2S` | `results/{stem}_{n_h}H/diffusivity_arrhenius.json` (per stem, per `n_H`) | Diffusivity pre-exponential [m²/s] |
| `E_D_EV` | same file | Bulk diffusion activation energy [eV] |
| `a₀(T)` | `lattice_params_vs_T.json` | Thermal lattice parameter [m] per T |

### Tunable simulation parameters

| Parameter | Default | Physical meaning | When to change |
|---|---|---|---|
| `TEMPERATURES` | [400, 600, 800] K | Temperatures for rate/KMC computation, shared by all 3 parts | Match your reactor operating range |
| `NX, NY` | 40, 40 | KMC grid size | 1600 surface + 1600 sub1 + 1600 sub2 sites; large enough to avoid finite-size artefacts |
| `KMC_MAX_STEPS` | 500,000 | Hard cap on KMC steps | Increase if convergence warnings appear in `permeation_sweep*.json` |
| `P_VALS_PA` | 40 log-spaced points, 1e-5 to 1e6 Pa | Pressure sweep range | Match your reactor H₂ partial pressures |
| `A0_M` | 3.52e-10 m | FCC lattice constant fallback | Only used if Part 3 has not run yet |
| `N_IMAGES` | 9 | NEB interpolation images | 18 for finer paths; 24 for steep/narrow barriers |
| `SPRING_CONST` | 1.0 eV/Å² | NEB spring force constant | 0.5 for smooth paths; 2.0 for high-curvature paths |
| `NEB_FTOL` | 0.1 eV/Å | NEB convergence criterion | 0.05 for tighter convergence; 0.02 for high-accuracy barriers |
| `N_H_VALUES` | [1, 3, 5, 10] | H concentrations for diffusivity runs | At minimum include 1 for dilute-limit D |
| `N_PROD_STEPS` | 5,000,000 | NVT production steps (0.5 fs/step = 2.5 ns) | Increase for low-T where diffusion is slow |
| `LAYERS` | 12 | Slab thickness (must be even; ignored for oxides — auto-matched to ~22 Å) | 12 is standard; 16 for thicker membranes |

### Physical constants used internally

| Constant | Value | Used in |
|---|---|---|
| `k_B` | 8.617 × 10⁻⁵ eV/K | All Arrhenius expressions |
| Speed of light | 2.998 × 10¹⁰ cm/s | Vineyard prefactor (converts cm⁻¹ → s⁻¹) |
| m(H₂) | 2 × 1.674 × 10⁻²⁷ kg | Hertz-Knudsen adsorption rate |
| ZPE threshold | 50 cm⁻¹ | Filters near-zero artefact modes from Hessian |
| Sieverts R² threshold | 0.98 | Classifies transport as bulk-limited vs surface-limited |

---

*End of guide. For code questions, open the relevant notebook and read the comments in Cell 1 (module docstring) and Cell 2 (parameter descriptions). For physics questions, the key references are: Vineyard (1957) J. Phys. Chem. Solids 3:121; Richardson & Sieverts original permeability formulation; Bortz, Kalos & Lebowitz (1975) J. Comput. Phys. 17:10.*

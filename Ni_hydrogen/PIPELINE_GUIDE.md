# Hydrogen Permeation Pipeline — Complete Guide

**Material:** Hastelloy N (Ni₇₁Mo₁₆Cr₇Fe₆ alloy)  
**Question:** How fast does hydrogen permeate through a structural membrane?  
**Method:** MACE ML potential + LAMMPS → NEB → TST → KMC → Richardson-Sieverts permeability

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

1. **MACE ML potential** — A machine-learned interatomic potential (MACE-MP-0b2-medium) that gives DFT-level accuracy at near-classical-MD cost. All energy evaluations in this project use MACE via LAMMPS with GPU (A100) acceleration through Kokkos.

2. **NEB** — Finds the minimum-energy path between two atomic configurations to get the energy barrier Eₐ for each H hop.

3. **TST + ZPE** — Converts barriers into rate constants k(T), corrected for quantum vibrational zero-point energy.

4. **KMC** — Simulates the stochastic kinetics on the surface/subsurface layer using TST rates as input.

5. **Richardson-Sieverts** — Combines the KMC-derived surface concentration with the MD-derived bulk diffusivity to give the macroscopic permeability Φ(T).

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

- `N_REPLICAS = 18` — number of intermediate images. More images resolve the path better but cost linearly more CPU time. 18 is adequate for MACE.
- `SPRING_CONST = 1.0` eV/Å² — keeps images equally spaced. Too small and images bunch at IS/FS; too large and the path distorts.
- `NEB_FTOL = 0.05` eV/Å — convergence criterion on the maximum force component. Standard value for MACE.

**How it runs.** NEB in this project uses ASE's NEB implementation driven by LAMMPS (via `models/ase_neb.py`). Each NEB job is one SLURM task using 16 CPU cores (NEB is CPU-bound in this implementation; GPU NEB is less efficient for short paths).

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

**The physical model.** The KMC grid represents the FCC(111) surface + one subsurface-1 octahedral layer. Each grid point is one surface site (one metal atom of type Ni/Mo/Cr/Fe). A corresponding subsurface-1 oct site sits directly beneath each surface site.

Events tracked at each step:

| Event | Rate | Source |
|---|---|---|
| H₂(g) adsorbs on empty pair → 2H* | Hertz-Knudsen × k_diss | Part 1 NEB + pressure |
| 2H* desorbs → H₂(g) | k_des | Part 1 NEB |
| H* diffuses to neighbouring site | k_surf_diff | (optional) |
| H* → H subsurface (Hop A) | k_entry | Part 2 NEB |
| H subsurface → H* (Hop A reverse) | k_exit | Part 2 NEB |
| H subsurface → bulk drain | k_drain | Part 3 diffusivity |

`k_drain` uses Fick's law: `k_drain = D / (a₀² × N_sub)` — the rate at which each subsurface H atom is "absorbed" into the bulk given bulk diffusivity D.

**The Hertz-Knudsen adsorption rate** for one surface site:

```
R_strike = P × A_site / √(2π m_H₂ k_B T)
```

where `A_site = (a₀/√2)²` is the FCC(111) nearest-neighbour area. The actual adsorption rate is `k_diss × R_strike`.

**The grid.** A 20×20 alloy grid (default) with periodic boundary conditions. Each site is assigned an element (Ni/Mo/Cr/Fe) drawn from Hastelloy N composition:

```
Ni: 71%,  Mo: 16%,  Cr: 7%,  Fe: 6%
```

The same `seed=42` is used for composition so results are reproducible across pressure points.

**Steady state.** `run_kmc_to_steady_state()` runs until the mean subsurface H concentration converges (rolling window check, relative tolerance 1%). The steady-state `C0` (surface-side H concentration in atoms/m³) feeds Fick's law to give the permeation flux J.

---

### 2.5 Richardson-Sieverts Permeability

**The macroscopic result.** For a membrane of thickness L under a pressure differential P_high − P_low, the steady-state H flux is:

```
J = Φ(T) × (√P_high − √P_low) / L      [atoms m⁻² s⁻¹]
```

The permeability Φ = D × S combines:
- **D(T)** — bulk diffusivity [m²/s], from Part 3 MD
- **S(T)** — Sieverts solubility [atoms m⁻³ Pa^(−½)], the H concentration per unit √P

**Sieverts' law.** For bulk-diffusion-limited transport, J ∝ √P. This arises because H₂ dissociates (H₂ ↔ 2H*) and the equilibrium H concentration scales as √P by detailed balance. If J–√P is **not** linear, it means the surface kinetics (adsorption/dissociation or subsurface entry) are rate-limiting, not bulk diffusion.

**Three routes to solubility S₀** (the pre-exponential factor in `S(T) = S₀ exp(−ΔH_sol / k_BT)`):

| Option | Formula | Physical meaning |
|---|---|---|
| Option 1 | `S₀ = 4/a₀³` | Geometric maximum: all oct sites filled at P=1 Pa, ΔH_sol=0 |
| Option 2 | Detailed balance from TST rates | Equilibrium C_sub/√P from surface + entry rate balance |
| Option 3 | KMC pressure sweep fit: `S = C₀/√P` | Empirical, fully self-consistent with the KMC model |

**Solution enthalpy:**

```
ΔH_sol = ΔH_diss/2 + ΔH_entry
```

`ΔH_diss` is the H₂ dissociation reaction energy (mean `delta_E` from surface NEB).  
`ΔH_entry` is the H* → H_sub reaction energy (mean `delta_E` from Hop A NEB).  
Both are auto-extracted from the corresponding JSON files after Parts 1 and 2 run.

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
             ├─ rate_dict_T{T}K.json
             ├─ permeation_sweep_T{T}K.json
             ├─ permeability_T{T}K.json
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

- **Cluster:** SLURM with A100 GPUs (partition `multigpu`) and CPU nodes
- **Python env:** `mace_env` (Python 3.9.23) with MACE, ASE, networkx, numpy, scipy
- **LAMMPS build:** custom build with MLIAP + Kokkos for GPU acceleration
- **MACE model:** `mace-mp-0b2-medium` — universal ML potential

---

## 4. Quick Start

### Full pipeline (recommended for a new material)

```bash
# 1. Open the master notebook
#    calculation/pipeline.ipynb

# 2. Edit Cell 2 — set your parameters:
#    WORK_DIR         = '/projects/.../your_run'
#    INPUT_STRUCTURES = ['/path/to/hastelloy_slab.lammps']
#    TEMPERATURES     = [600, 700, 800, 900]   # K
#    E_H2_GAS         = -6.7640                # eV (run separately — see note below)

# 3. Run all cells → generates pipeline_run.py + pipeline_run.sh

# 4. Submit to SLURM
sbatch calculation/pipeline_run.sh

# 5. Monitor
squeue -u $USER

# 6. After completion, open each notebook and run the analysis cells
#    for plots and tables
```

**Note on `E_H2_GAS`:** This is the MACE energy of an isolated H₂ molecule in vacuum [eV]. It cannot be extracted automatically from the pipeline. Run a single LAMMPS minimisation of a H₂ molecule in a large box, read the final energy, and enter it once. For MACE-MP-0b2-medium on a gas-phase H₂, the value is approximately −6.76 eV (you should compute your own to be exact).

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
| A | Slab relaxation | Build FCC(111) 12-layer slab; assign Hastelloy N composition; relax surface with CG minimisation | GPU array |
| B | Site enumeration | Use ACAT to identify all unique surface sites; compute H₂* and H* adsorption energies for each site | GPU array |
| C | NEB initial path | Generate IS (H₂* at site) and FS (2×H* at adjacent sites) pairs; create linear interpolation; submit CINEB | CPU array (16 cores each) |
| D | FS refinement | Re-minimise FS structures found by Phase C; re-run NEB from Phase D FS | GPU+CPU sequential |
| E | Vibrational analysis | Compute partial Hessian at IS and TS for each converged NEB; extract frequencies; apply ZPE; save `diss_vib_rates.json` | CPU array |

### Key parameters (Cell 2 of `neb_calculation.ipynb`)

```python
WORK_DIR        = '/projects/.../run_dir'
INPUT_SLAB      = '/path/to/hastelloy_slab.lammps'
E_H2_GAS        = -6.7640        # eV — H₂ gas reference energy (MUST BE SET)
LAYERS          = 12             # slab thickness; 12 = standard, must be even
N_IMAGES        = 18             # NEB interpolation images
SPRING_CONST    = 1.0            # eV/Å²
NEB_FTOL        = 0.05           # eV/Å — convergence tolerance
TEMPERATURES    = [600, 700, 800, 900]  # K — for rate computation in Phase E
```

### How the slab is built

The slab is an FCC(111) supercell of Hastelloy N:
- 12 layers of metal atoms
- Bottom 4 layers are frozen (Cartesian z positions held fixed) to simulate the bulk
- Top 8 layers are free to relax
- H atoms are placed at FCC hollow sites or bridge sites (enumerated by ACAT)
- Element types: Al, B, C, Cr, Fe, Mo, Ni = types 1–7; H = type 8

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

| Phase | Name | What happens |
|---|---|---|
| 1a | CG minimisation | Conjugate-gradient energy minimisation of the bare bulk structure to find the T=0 reference geometry |
| 1b | NPT equilibration | Run NPT MD at each T to let the box relax to the thermal lattice constant a₀(T); insert N_H hydrogen atoms; re-minimise the bulk+H structure |
| 2 | NVT production | Long NVT MD (fixed volume, fixed T); dump atomic positions every N steps for MSD calculation; auto-restarts if SLURM wall time is hit (job chaining) |
| 3 | MSD analysis | Compute MSD(t) from position dumps; fit to `MSD = 6Dt` for 3D diffusion; repeat across temperatures; Arrhenius fit: `ln(D) vs 1/T` → D₀ and E_D |

### Arrhenius diffusivity

The temperature-dependent bulk diffusivity follows:

```
D(T) = D₀ × exp(−E_D / k_B T)      [m²/s]
```

`D₀` is the diffusivity pre-exponential [m²/s] and `E_D` is the activation energy for bulk H diffusion [eV]. Both come from the Arrhenius fit to the MSD data.

### Key parameters (Cell 2 of `diffusivity.ipynb`)

```python
INPUT_STRUCTURES = ['/path/to/bulk_hastelloy.lammps']
TEMPERATURES     = [500, 600, 700, 800, 900]  # K
N_H_VALUES       = [1, 3, 5, 7, 10]          # H atoms to insert
N_PROD_STEPS     = 1_500_000                  # timestep 0.5 fs → 750 ps
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
  "temperatures": [500, 600, 700, 800, 900],
  "a0_m": [3.518e-10, 3.522e-10, 3.526e-10, 3.530e-10, 3.534e-10]
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
H leaves the FCC(111) surface hollow site and enters the first octahedral interstitial site between slab layers 11 and 12 (numbering from the top). This is the critical entry step — if k_entry is slow, surface kinetics limit permeation.

**Hop B** — sub1 → subsurface-2 (sub2):  
H moves from the first subsurface oct site to the second oct site (between layers 10 and 11). This hop determines how fast H moves from the surface region into the bulk. After sub2, H is effectively in the bulk and diffuses at the bulk rate D(T).

### Six phases of `permeation_run.py`

| Phase | Name | What happens | Hardware |
|---|---|---|---|
| 1 | Hop A NEB | Build subsurface graph (NetworkX); generate H_sub FS structures; FS-min; run CINEB for H*→sub1 | GPU+CPU arrays |
| 2 | Hop B NEB | Use Hop A relaxed sub1 structures as IS; generate sub2 FS structures; run CINEB for sub1→sub2 | GPU+CPU arrays |
| 3 | Vibrations | Hessian + normal modes at IS and TS for all Hop A and Hop B NEB jobs; ZPE-corrected barriers | CPU array |
| 4 | TST rates | Vineyard prefactor; Arrhenius rates at each temperature; write `rate_dict_T{T}K.json` | Local Python |
| 5 | KMC sweep | Load rate_dict + D(T) + a₀(T); run BKL KMC at each pressure; record steady-state C₀ and J; Sieverts check | Local Python |
| 6 | Permeability | Three S₀ options; Φ(T) = D×S at each T; Arrhenius fit of Φ(T) | Local Python |

### Auto-extracted values

After a full pipeline run, Part 2 reads the following without any manual input:

| Value | Source | What it is |
|---|---|---|
| `DH_DISS_EV` | mean `delta_E` from `ranked_barriers.json` | H₂ dissociation reaction energy |
| `DH_ENTRY_EV` | mean `delta_E` from converged Hop A jobs | H* → H_sub reaction energy |
| `D0_M2S`, `E_D_EV` | `diffusivity_arrhenius.json` | Bulk diffusivity Arrhenius parameters |
| `a₀(T)` | `lattice_params_vs_T.json` | Thermal lattice constant per temperature |

If Part 3 has not been run yet, Part 2 falls back to the fixed `A0_M` value from Cell 2. If `diffusivity_arrhenius.json` is missing, it uses a placeholder D value and prints a warning.

### Key parameters (Cell 2 of `permeation.ipynb`)

```python
WORK_DIR        = '/projects/.../run_dir'    # same as Part 1
TEMPERATURES    = [600, 700, 800, 900]        # K
A0_M            = 3.52e-10                    # m — fallback only if Part 3 not run
L_M             = 1e-3                        # m — membrane thickness (1 mm)
NX, NY          = 20, 20                      # KMC grid dimensions
KMC_MAX_STEPS   = 500_000                     # hard cap on KMC steps
P_VALS_PA       = [1e3, 1e4, 1e5, 5e5, 1e6]  # Pa — pressure sweep
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

**`results/permeability_T700K.json`** — all three Φ options at 700 K:
```json
{
  "T_K": 700.0,
  "D_m2s": 4.2e-9,
  "option1": {"Phi": 1.8e-15, "S0": 4.3e23, "S": 4.3e14},
  "option2": {"Phi": 2.1e-15, "S0": ...,     "S": ...},
  "option3": {"Phi": 1.9e-15, "S0": ...,     "S": ..., "from_kmc": true}
}
```

**`results/solubility_arrhenius_kmc.json`** — multi-temperature S(T) Arrhenius fit from Option 3.

### Plots produced

- `barriers_overview.png` — histogram of all Hop A and Hop B barriers
- `mep_hopA.png` / `mep_hopB.png` — minimum energy path for the lowest-barrier Hop A and Hop B
- `sieverts_check.png` — J vs √P scatter with linear fit; R² value tells you whether bulk or surface is limiting
- `permeability_vs_T.png` — ln(Φ) vs 1/T Arrhenius plot comparing all three options
- `solubility_arrhenius.png` — ln(S) vs 1/T for Option 3
- `bottleneck.png` — comparison of k_entry vs k_exit vs k_drain to identify rate-limiting step

---

## 8. Master Pipeline (`pipeline.ipynb`)

### What it does

Orchestrates Parts 1, 3, and 2 in the correct order with no manual intervention between steps.

```
sbatch pipeline_run.sh
    └─ python pipeline_run.py
           │
           ├── subprocess.Popen(neb_run.py)          # Part 1 (GPU+CPU, hours–days)
           ├── subprocess.Popen(diffusivity_run.py)   # Part 3 (GPU+CPU, hours–days)
           │                                          # Parts 1 + 3 run in parallel
           ├── p1.wait() + p3.wait()                  # block until both finish
           └── subprocess.run(permeation_run.py)      # Part 2 (GPU+CPU, hours–days)
```

The pipeline script itself is lightweight — it just launches and waits. All the physics happens inside the sub-scripts.

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

**`structure.py`** — Build and manipulate atomic structures. `build_fcc111_slab()` creates the Hastelloy N slab with frozen layers. `insert_h_octahedral()` places H atoms at FCC oct interstitial sites. `extract_lattice_param()` reads a₀ from NPT thermo output.

**`energetics.py`** — Compute and summarise energetics. Parses adsorption energies from batched LAMMPS runs. `summarise_neb_barriers()` reads all `neb_barrier.txt` files in a directory and produces `ranked_barriers.json` sorted by Eₐ.

### NEB modules

**`ase_neb.py`** — Runs a single NEB calculation as an ASE NEB object backed by LAMMPS. Called by SLURM array job workers (one job per IS/FS pair). Handles CINEB image convergence and writes `neb_barrier.txt`.

**`neb_workflow.py`** — Orchestrates the full surface NEB pipeline (Phases A–E). Generates all job scripts, submits SLURM arrays, waits for completion, and parses results. Contains the `_NEB_BODY` f-string that becomes `neb_run.py` when the notebook is run.

**`neb_subsurface.py`** — Orchestrates Hop A and Hop B NEB. `orchestrate_hopa_neb()` generates H_sub FS positions from the subsurface graph and submits the NEB array. `orchestrate_hopb_neb()` uses Hop A output as input. Writes the barrier files that feed Phase 3 and 4 of `permeation_run.py`.

**`subsurface_graph.py`** — Builds a NetworkX graph of FCC octahedral interstitial sites in the slab. Each node is a sub oct site; edges connect geometrically adjacent sites. `build_subsurface_graph()` parses the relaxed slab structure and identifies all sub1 and sub2 sites, keyed by the overlying surface site ID. This graph determines which FS structures to generate for Hop A/B.

### Vibrational analysis and rates

**`vibrations.py`** — Computes partial Hessian by finite displacements via LAMMPS. `orchestrate_vibrations()` submits a SLURM array of displacement jobs, collects forces, assembles the Hessian, diagonalises it, and writes `vib_frequencies.json` containing real and imaginary mode frequencies. The partial Hessian displaces only the H atom and its coordination shell (~10 metal atoms) rather than all atoms, making it computationally feasible.

**`tst_rates.py`** — Converts NEB barriers and vibrational frequencies into TST rate constants. Pipeline: `collect_neb_results` → `split_vib_results` → `apply_zpe_correction` → `vineyard_prefactor` → `arrhenius_rate` → `build_rate_dict` → `rates_to_json`. All steps are pure Python; no LAMMPS calls.

### KMC and macroscopic transport

**`kmc.py`** — BKL KMC engine. `make_grid(nx, ny, composition, seed)` creates the 2D alloy surface grid. `run_kmc(grid, rate_dict, P, T, D, a0, n_steps)` runs a fixed number of KMC steps. `run_kmc_to_steady_state(grid, rate_dict, P, T, D, a0)` runs until the subsurface H concentration converges to steady state and returns `{C0, time, converged, n_steps}`.

**`permeation.py`** — Macroscopic permeability from KMC results. `sweep_pressure()` calls `run_kmc_to_steady_state` at each pressure point and returns J vs P data. `check_sieverts_law()` fits J vs √P and diagnoses the rate-limiting step. `lattice_site_S0()`, `solubility_from_rates()`, `fit_solubility_from_kmc()` implement the three S₀ routes. `permeability()` and `richardson_flux()` give the final Φ and J.

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
  "temperatures": [500, 600, 700, 800, 900],
  "a0_m":         [3.518e-10, 3.522e-10, 3.526e-10, 3.530e-10, 3.534e-10]
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
`k_forward` and `k_reverse` are the rates you put into the KMC `rate_dict` as `k_entry` and `k_exit`.

### `permeability_T700K.json`
All three permeability options at 700 K:
```json
{
  "T_K":    700.0,
  "D_m2s":  4.2e-9,
  "option1": {
    "label": "lattice_site",
    "S0":    4.3e23,
    "S":     4.3e14,
    "Phi":   1.8e-15,
    "J_at_1e5Pa": 1.8e12
  },
  "option2": {
    "label": "detailed_balance",
    "S0": ..., "S": ..., "Phi": ..., "J_at_1e5Pa": ...
  },
  "option3": {
    "label": "kmc_empirical",
    "S0": ..., "S": ..., "Phi": ..., "J_at_1e5Pa": ...,
    "from_kmc": true,
    "r_squared_sieverts": 0.9973
  }
}
```
`Phi` has units atoms·m⁻¹·s⁻¹·Pa^(−½). For comparison with literature, convert to mol·m⁻¹·s⁻¹·Pa^(−½) by dividing by Avogadro's number (6.022 × 10²³).

---

## 11. Parameters Cheat Sheet

### Parameters that must be set manually

| Parameter | Notebook Cell | Physical meaning | How to determine | Typical range |
|---|---|---|---|---|
| `E_H2_GAS` | neb Cell 2 | MACE energy of gas-phase H₂ [eV] | Single-point MACE of H₂ in vacuum | ~−6.76 eV (MACE-MP-0b2) |
| `L_M` | permeation Cell 2 | Membrane thickness [m] | Experimental membrane geometry | 0.5 mm – 5 mm |

### Parameters auto-extracted after pipeline runs

| Parameter | Source file | Physical meaning |
|---|---|---|
| `DH_DISS_EV` | `ranked_barriers.json` (mean delta_E) | H₂ dissociation reaction energy [eV] |
| `DH_ENTRY_EV` | Hop A NEB results (mean delta_E) | H* → H_sub reaction energy [eV] |
| `D0_M2S` | `diffusivity_arrhenius.json` | Diffusivity pre-exponential [m²/s] |
| `E_D_EV` | `diffusivity_arrhenius.json` | Bulk diffusion activation energy [eV] |
| `a₀(T)` | `lattice_params_vs_T.json` | Thermal lattice parameter [m] per T |

### Tunable simulation parameters

| Parameter | Default | Physical meaning | When to change |
|---|---|---|---|
| `TEMPERATURES` | [600, 700, 800, 900] K | Temperatures for rate/KMC computation | Match your reactor operating range |
| `NX, NY` | 20, 20 | KMC grid size | Increase to 40×40 for publication-quality statistics (4× slower) |
| `KMC_MAX_STEPS` | 500,000 | Hard cap on KMC steps | Increase if convergence warnings appear in `permeation_sweep*.json` |
| `P_VALS_PA` | [1e3, 1e4, 1e5, 5e5, 1e6] Pa | Pressure sweep range | Match your reactor H₂ partial pressures |
| `A0_M` | 3.52e-10 m | FCC lattice constant fallback | Only used if Part 3 has not run yet |
| `N_IMAGES` | 18 | NEB interpolation images | 12 for quick tests; 24 for steep/narrow barriers |
| `SPRING_CONST` | 1.0 eV/Å² | NEB spring force constant | 0.5 for smooth paths; 2.0 for high-curvature paths |
| `NEB_FTOL` | 0.05 eV/Å | NEB convergence criterion | 0.1 for quick scans; 0.02 for high-accuracy barriers |
| `N_H_VALUES` | [1,3,5,7,10] | H concentrations for diffusivity runs | At minimum include 1 for dilute-limit D |
| `N_PROD_STEPS` | 1,500,000 | NVT production steps (0.5 fs/step = 750 ps) | Increase for low-T where diffusion is slow |
| `LAYERS` | 12 | Slab thickness (must be even) | 12 is standard; 16 for thicker membranes |

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

# Multiscale H Permeation Pipeline (NEB → TST → KMC → Flux)

## Context

Extend the existing MACE/LAMMPS codebase into a complete five-phase permeation
pipeline that bridges atomic-scale barriers to macroscopic permeation flux J.
Phases 1 (H2 dissociation NEB) and 3 (bulk D via MSD) are largely complete.
This plan fills in the remaining pieces: subsurface/migration NEB, vibrational
frequency calculations, TST rate conversion, KMC simulation, and flux evaluation.

---

## Pipeline Overview

```
[ PHASE 1: Energy Landscaping ]  ──> NEB/MACE → ΔE barriers + vibrational frequencies
               │
               ▼
[ PHASE 2: TST Rate Conversion ] ──> ZPE + Vineyard → k_i(T) rate constants
               │
               ▼
[ PHASE 3: Bulk Transport ]      ──> LAMMPS MSD → D(T)   (already complete)
               │
               ▼
[ PHASE 4: Kinetic Monte Carlo ] ──> 2D alloy grid → steady-state C0
               │
               ▼
[ PHASE 5: Flux Evaluation ]     ──> Fick's Law → J(P)  →  Sievert's check
```

---

## What Already Exists (Do Not Reimplement)

| What | Where | Phase |
|---|---|---|
| H2 → 2H* dissociation NEB (k_diss, k_des) | `models/neb_workflow.py` + `models/ase_neb.py` | 1 |
| Barrier parsing (Ea, E_des) | `models/parsers.py::parse_barrier_file` | 1 |
| Octahedral bulk site insertion | `models/structure.py::insert_hydrogen` | 1 |
| Bulk diffusivity D(T) via MSD | `models/diffusivity_workflow.py` | 3 |
| SLURM job generation | `models/create_slurm.py::write_slurm_job` | all |
| Surface site enumeration | `models/neb_workflow.py` Section A Phase 3 | 1 |
| Adsorbed H* structures | `models/neb_workflow.py` Section B Phase 2 | 1 |
| Two-phase CINEB runner | `models/ase_neb.py::run_neb_pipeline` | 1 |
| Energetics helpers | `models/energetics.py` | 2 |

---

## New Files (5 modules + 3 notebooks)

```
models/neb_subsurface.py       Phase 1 — surface→subsurface and bulk migration NEB
models/vibrations.py           Phase 1 — frozen-phonon frequencies via ASE + MACE
models/tst_rates.py            Phase 2 — ZPE, Vineyard prefactor, rate dict
models/kmc.py                  Phase 4 — 2D alloy KMC engine (BKL algorithm)
models/permeation.py           Phase 5 — Fick's law, pressure sweep, Sieverts check

calculation/subsurface_neb_calculation.ipynb
calculation/tst_calculation.ipynb
calculation/kmc_calculation.ipynb
```

---

## Phase 1 Extension — `models/neb_subsurface.py`

### Subsurface Entry/Exit NEB

NEB type: IS = relaxed slab + H* at surface site (from Section B h_atom_{sid}_relaxed.lammps)
          FS = same slab + H at nearest subsurface octahedral cavity

```python
def find_subsurface_oct_site(slab_path, surface_xy, z_surface_top) -> tuple[float,float,float]:
    """Return (x, y, z) of the nearest FCC octahedral cavity below a surface site."""

def build_subsurface_is(h_surface_path, out_path) -> str:
    """IS: copy the existing h_atom_{sid}_relaxed.lammps (H at surface, no modification)."""

def build_subsurface_fs(slab_relaxed_path, oct_xyz, masses, e2t, out_path) -> str:
    """FS: place H at subsurface oct site (same metal atoms as IS)."""

def orchestrate_surface_subsurface_neb(
    phase2_h_dir, relaxed_slab_path, outdir,
    slurm_opts=None, neb_slurm_opts=None,
    n_images=18, spring_const=1.0, neb_ftol=0.05,
    dry_run=True,
) -> dict:
    """
    For each H* site in the Section B FS pool:
      1. Build IS (H at surface) and FS (H at subsurface oct).
      2. Generate NEB scripts via run_neb_pipeline.
      3. Write SLURM array.
    Returns: {entry_barriers, exit_barriers, array_script, n_jobs, status}
    """
```

### Bulk Migration NEB

NEB type: IS = bulk supercell + H at oct site 1
          FS = bulk supercell + H at adjacent oct site (a0/sqrt(2) away in FCC)

```python
def build_bulk_migration_pair(bulk_min_path, a0, n_oct_pairs, masses, e2t, outdir) -> list[dict]:
    """
    Enumerate symmetry-unique oct→oct hops in bulk FCC supercell.
    Returns list of {is_path, fs_path, hop_vector, label}.
    Uses insert_hydrogen from structure.py to seed oct sites.
    """

def orchestrate_bulk_migration_neb(
    bulk_min_path, a0, outdir,
    slurm_opts=None, neb_slurm_opts=None,
    n_images=18, spring_const=1.0, neb_ftol=0.05,
    n_unique_hops=5, dry_run=True,
) -> dict:
    """
    Generate and submit NEB jobs for bulk oct→oct hops.
    Returns: {migration_barriers, array_script, n_jobs, status}
    """
```

---

## Phase 1 Extension — `models/vibrations.py`

Use ASE `Vibrations` class with MACECalculator to compute frozen-phonon frequencies.
Run on the cluster (GPU job) after NEB convergence for each IS and TS geometry.

```python
def write_vibration_script(
    structure_path,       # LAMMPS data file (IS or TS geometry)
    mace_model_path,
    out_path,             # path to written vib_run.py
    outdir,               # directory for .json vib data files
    z_freeze_cutoff=22.115,
    delta=0.01,           # displacement in Å
    device='cpu',
) -> str:
    """
    Write a standalone Python script that:
      1. Loads structure via ASE.
      2. Attaches MACECalculator.
      3. Runs ASE Vibrations (finite differences, delta displacement).
      4. Saves frequencies to {outdir}/vib_frequencies.json.
    Returns path to written script.
    """

def load_vibration_results(vib_json_path) -> dict:
    """Load {frequencies: [cm-1 list], imaginary: [cm-1 list]} from vib_frequencies.json."""

def orchestrate_vibrations(
    structure_paths,      # list of (label, lammps_path) for IS and TS of each job
    outdir,
    slurm_opts=None,
    dry_run=True,
) -> dict:
    """
    Write and submit one vib_run.py + SLURM job per structure.
    Returns {label: vib_json_path, ...}
    """
```

---

## Phase 2 — `models/tst_rates.py`

Converts NEB barriers + vibrational frequencies into kinetic rate constants k(T).

```python
PLANCK_H   = 6.626e-34    # J·s
BOLTZMANN  = 8.617e-5     # eV/K

def apply_zpe_correction(E_barrier_eV, freqs_initial_cm1, freqs_ts_cm1) -> float:
    """
    ΔE_corrected = E_barrier + (ZPE_TS - ZPE_IS)
    ZPE = 0.5 * h * sum(nu_real)  [real frequencies only, converted to eV]
    """

def vineyard_prefactor(freqs_initial_cm1, freqs_ts_cm1) -> float:
    """
    nu = prod(nu_IS_real) / prod(nu_TS_real)   [Vineyard 1957]
    TS has one imaginary mode (excluded from denominator).
    Units: s^-1  (~10^12 s^-1 typical)
    """

def arrhenius_rate(nu_s1, delta_e_eV, T_K) -> float:
    """k = nu * exp(-deltaE / kB T)"""

def build_rate_dict(
    neb_results,        # {label: {Ea, E_des, ...}} from collect_neb_results
    vib_results_is,     # {label: vib_json_path} for IS geometries
    vib_results_ts,     # {label: vib_json_path} for TS geometries
    T_K,
    apply_zpe=True,
) -> dict:
    """
    For each NEB label, compute:
      k_forward  = arrhenius_rate(vineyard(...), zpe_correct(Ea, ...), T_K)
      k_reverse  = arrhenius_rate(vineyard(...), zpe_correct(E_des, ...), T_K)

    Rate types emitted:
      k_diss, k_des        — from H2 dissociation NEB
      k_entry, k_exit      — from surface→subsurface NEB
      k_mig                — from bulk migration NEB
      k_surf_diff          — from surface diffusion NEB

    Each keyed by local chemical environment e.g. ('Ni','Ni'), ('Ni','Mo').
    Returns: {label: {k_forward, k_reverse, Ea_raw, Ea_zpe, nu}}
    """

def rates_to_json(rate_dict, out_path) -> str:
    """Serialize rate dict to JSON for use by KMC simulation."""
```

---

## Phase 4 — `models/kmc.py`

Rejection-free BKL kinetic Monte Carlo on a 2D dual-layer alloy grid.

### Grid Architecture

```
Surface grid  [N×M]: element[i,j] ∈ {Ni,Cr,Mo,Fe}, occupancy[i,j] ∈ {0,1}
Subsurface    [N×M]: occupancy[i,j] ∈ {0,1}
Alloy comp:   71% Ni, 16% Mo, 7% Cr, 6% Fe  (Hastelloy N)
```

### Event Catalogue

| Event | Rate |
|---|---|
| H2 adsorption + dissociation | R_strike × k_diss(elem1, elem2) |
| H* + H* recombination + desorption | k_des(elem1, elem2) |
| H* surface diffusion | k_surf_diff(elem1, elem2) |
| H* surface → H subsurface entry | k_entry(elem) |
| H subsurface → H* surface exit | k_exit(elem) if surface site empty |
| H subsurface → bulk drainage | k_drain = D / dx² |

Gas strike flux (kinetic gas theory):  `R_strike = P / sqrt(2π m_H2 kB T)`

Bulk drainage calibrated from LAMMPS:  `k_drain = D / dx²,  dx = a0 / sqrt(2)`

### Core Classes

```python
@dataclass
class KMCEvent:
    kind: str            # 'adsorb','desorb','surf_diff','enter','exit','drain'
    sites: list          # affected (i,j) grid indices
    rate: float          # s^-1

class AlloyGrid:
    def __init__(self, nx, ny, composition=None, seed=42): ...
    def neighbors(self, i, j) -> list[tuple]: ...
    def element_pair(self, i1, j1, i2, j2) -> tuple: ...
    def surface_coverage(self) -> float: ...           # theta
    def subsurface_population(self) -> int: ...
    def subsurface_concentration(self, a0_m) -> float: ...  # C0 in atoms/m^3

class KMCSim:
    """BKL algorithm: sum rates → advance clock → binary search → execute."""

    def __init__(self, grid, rate_dict, P_high_Pa, T_K, D_bulk_m2s, a0_m): ...
    def build_event_list(self) -> list[KMCEvent]: ...
    def step(self) -> float: ...                       # returns dt (s)
    def run(self, n_steps) -> dict: ...                # trajectory of (t, theta, n_sub)
    def run_to_steady_state(
        self, window=5000, rtol=0.02, max_steps=5_000_000
    ) -> dict:
        """Runs until theta and n_sub stable. Returns {t_total, theta_ss, C0, n_steps}."""
```

---

## Phase 5 — `models/permeation.py`

```python
def fick_flux(D_m2s, C0_m3, L_m, C_low_m3=0.0) -> float:
    """J = D * (C0 - C_low) / L  [atoms/(m^2 s)]"""

def sweep_pressure(P_vals_Pa, kmc_factory_fn, D_m2s, L_m, T_K) -> dict:
    """
    For each P in P_vals_Pa:
      1. Build KMCSim via kmc_factory_fn(P).
      2. Run to steady state → C0.
      3. J = fick_flux(D, C0, L).
    Returns {P_vals, J_vals, C0_vals, sqrt_P_vals}.
    """

def check_sieverts_law(P_vals_Pa, J_vals, plot=True) -> dict:
    """
    Fit J vs sqrt(P) linearly.
    R^2 ~ 1  →  bulk diffusion limited (Sievert's law holds).
    R^2 < 1  →  surface kinetics are the bottleneck.
    Returns {slope, intercept, r_squared, is_sieverts}.
    """
```

---

## Notebooks

### `calculation/subsurface_neb_calculation.ipynb`

Config: `PHASE2_H_DIR`, `RELAXED_SLAB_PATH`, `BULK_MIN_PATH`, `A0`

Cells:
1. `orchestrate_surface_subsurface_neb()` — H* surface → H subsurface NEB
2. `orchestrate_bulk_migration_neb()` — H oct→oct bulk NEB
3. `orchestrate_vibrations()` — frequencies for IS + TS of each NEB type
4. Analysis: load barrier files, plot entry/exit/migration MEPs

### `calculation/tst_calculation.ipynb`

Config: `T_K`, paths to all NEB results, paths to all vibration results

Cells:
1. `build_rate_dict()` at target T — inspect k values per environment
2. Sensitivity: plot k(T) vs T for each rate type
3. `rates_to_json()` → `rate_dict.json`

### `calculation/kmc_calculation.ipynb`

Config: `RATE_DICT_JSON`, `D_BULK`, `A0`, `L_MEMBRANE`, `P_VALS`, `NX`, `NY`, `T_K`

Cells:
1. Load `rate_dict.json`, build `AlloyGrid`, run single-P KMC to steady state → plot θ(t), C(t)
2. `sweep_pressure(P_vals)` → collect J values
3. Plot J vs sqrt(P) — linear = Sievert's; curved = surface bottleneck
4. `check_sieverts_law()` → report R² and conclusion

---

## Implementation Order

1. `models/neb_subsurface.py` — subsurface entry/exit + bulk migration NEB
2. `models/vibrations.py` — frozen-phonon frequencies
3. `models/tst_rates.py` — ZPE, Vineyard, rate constants
4. `models/kmc.py` — AlloyGrid + KMCSim (BKL)
5. `models/permeation.py` — Fick's law, pressure sweep, Sieverts check
6. Three notebooks

---

## Verification

1. **Phase 1**: `orchestrate_surface_subsurface_neb(dry_run=True)` → scripts written, no SLURM submission.
2. **Phase 2**: `build_rate_dict` with mock barriers at T=600 K → k values ~10⁻²–10⁸ s⁻¹; `vineyard_prefactor` ~10¹² s⁻¹.
3. **Phase 4**: `KMCSim.run(10000)` on 10×10 grid with synthetic rates → θ stabilises, time monotonically increasing, `subsurface_concentration()` > 0.
4. **Phase 5**: `check_sieverts_law` on synthetic J ∝ sqrt(P) data → R² > 0.99.

---

---

## Phase 6 — Richardson-Sieverts Permeability (IMPLEMENTED)

**Status:** Complete. All functions live in `models/permeation.py` Section 4.
Notebook integration: Cells 7–8 of `calculation/kmc_calculation.ipynb`.

---

## 6.1 Physical Motivation — Arrhenius Factorisation

Fick's law (Phase 5) computes flux as:

```text
J = D · (C₀ - C_low) / L
```

where C₀ comes from KMC at a single (P, T). This is correct but requires a fresh
KMC run for every (P, T, L) combination. The Richardson-Sieverts formulation
replaces this with a single **permeability** Φ(T) that absorbs both diffusion
and solubility, valid in the bulk-diffusion-limited (Sieverts) regime:

```text
J = Φ(T) · (√P_high − √P_low) / L
```

Because both D(T) and S(T) are Arrhenius, their product Φ is also Arrhenius:

```text
D(T)  = D₀  · exp(−E_D   / k_B T)     [m²/s]
S(T)  = S₀  · exp(−ΔH_sol / k_B T)    [atoms m⁻³ Pa^(−½)]

Φ(T)  = D(T) · S(T)
       = (D₀ · S₀) · exp(−(E_D + ΔH_sol) / k_B T)
       =  Φ₀        · exp(−E_Φ / k_B T)
```

**Effective permeation activation energy:**

```text
E_Φ = E_D + ΔH_sol
```

The slope of an Arrhenius plot of Φ vs 1/T gives E_Φ directly.
This is the quantity compared to experimental permeation measurements.

---

## 6.2 Derivation of ΔH_sol — Thermodynamic Cycle

The dissolved-H solution enthalpy per H atom is constructed from a two-step
thermodynamic cycle:

```text
Step 1:  ½ H₂(g)  →  H*  (surface adsorption / dissociation)
         ΔH_step1 = ΔH_diss / 2
         Source: H₂ dissociation NEB reaction energy (delta_E from barrier file)
         Sign convention: delta_E < 0 for exothermic dissociation

Step 2:  H*  →  H_sub  (subsurface entry, Hop A)
         ΔH_step2 = ΔH_entry
         Source: Hop A NEB reaction energy (delta_E from neb_barrier.txt)
         Sign convention: delta_E > 0 for endothermic entry into subsurface
```

Total:

```text
ΔH_sol = ΔH_diss / 2 + ΔH_entry        [eV per H atom]
```

In code (from NEB barrier files):

```python
DH_SOL_EV = delta_E_diss / 2.0 + delta_E_hopa
```

where `delta_E` is the NEB reaction energy (FS energy − IS energy), read from
`parse_barrier_file()` output key `'delta_E'`.

---

## 6.3 Solubility S₀ — Three Approaches

### Option 1 — Lattice-Site Density (Geometric Upper Bound)

**Physical basis:** In an FCC crystal, there are 4 octahedral interstitial sites
per unit cell of volume a₀³. The maximum dissolved-H concentration (all oct
sites filled at P = 1 Pa with zero thermodynamic penalty) defines S₀.

**Formula:**

```text
S₀ = N_oct / V = 4 / a₀³      [atoms m⁻³ Pa^(−½)]
```

The Pa^(−½) unit is implicit: when ΔH_sol = 0 the Sieverts law reads
C = S₀ · √P, so S₀ carries the Pa^(−½) factor by convention.

**Implementation:**

```python
def lattice_site_S0(a0_m: float) -> float:
    return 4.0 / (a0_m ** 3)
```

**Use case:** Upper bound. Useful as a reference; actual S is smaller because
ΔH_sol > 0 (subsurface entry is endothermic in most metals).

---

### Option 2 — TST Rates via Detailed Balance

**Physical basis:** At thermodynamic equilibrium every microscopic process
balances its reverse. We exploit this to derive S(T) directly from the same
TST rate constants that feed the KMC engine — no additional inputs required.

#### Step-by-step derivation

##### Equilibrium 1: Adsorption ↔ Desorption on the surface

The H₂ molecule strikes a surface site pair at rate R_strike [s⁻¹], a fraction
k_diss (dimensionless sticking coefficient) of which stick. Two H* are produced.
Recombinative desorption occurs at rate k_des [s⁻¹].

At equilibrium for a single site pair (i, j):

```text
R_strike · k_diss · (1 − θ)²  =  k_des · θ²
```

where θ is surface H coverage (fraction of sites occupied).

The Hertz-Knudsen gas-strike rate per site of area A_site [m²]:

```text
R_strike = P · A_site / √(2π m_H2 k_B T)     [s⁻¹]
```

Solving the balance equation in the dilute limit (θ ≪ 1, (1−θ)² ≈ 1):

```text
θ²  ≈  R_strike · k_diss / k_des
     =  k_diss · P · A_site / (k_des · √(2π m_H2 k_B T))

θ_eq  =  √[ k_diss · A_site · P / (k_des · √(2π m_H2 k_B T)) ]
```

##### Equilibrium 2: Surface ↔ Subsurface (Hop A)

The subsurface site below each surface site has concentration C_sub [atoms/m³].
The entry rate k_entry [s⁻¹] and exit rate k_exit [s⁻¹] are the Hop A forward
and reverse TST rates. At steady state:

```text
k_entry · θ  =  k_exit · (C_sub / ρ_oct)
```

where ρ_oct = 4/a₀³ is the oct-site density [m⁻³].

Solving:

```text
C_sub  =  ρ_oct · (k_entry / k_exit) · θ_eq
```

**Combining both equilibria:**

```text
C_sub  =  ρ_oct · (k_entry / k_exit)
           · √[ k_diss · A_site · P / (k_des · √(2π m_H2 k_B T)) ]

       =  S(T) · √P
```

Therefore:

```text
S(T)  =  ρ_oct · (k_entry / k_exit)
          · √[ k_diss · A_site / (k_des · √(2π m_H2 k_B T)) ]
```

**Site area** used in the KMC engine (consistent with `gas_strike_rate`):

```text
A_site = (a₀ / √2)²  =  a₀² / 2        [m²]
```

This is the nearest-neighbour distance squared in FCC — the area associated
with one surface site in the (111) plane.

**Implementation:**

```python
def solubility_from_rates(k_diss, k_des_s1, k_entry_s1, k_exit_s1, a0_m, T_K):
    rho_oct = 4.0 / (a0_m ** 3)
    A_site  = (a0_m / np.sqrt(2.0)) ** 2
    denom   = k_des_s1 * np.sqrt(2.0 * np.pi * M_H2_KG * KB_J * T_K)
    return rho_oct * (k_entry_s1 / k_exit_s1) * np.sqrt(k_diss * A_site / denom)
```

**Use case:** Most physically consistent — encodes the full kinetic pathway
from gas-phase H₂ to dissolved H, using the same rates as the KMC engine.
Naturally captures compositional and temperature dependence.

---

### Option 3 — KMC Empirical Fit

**Physical basis:** The KMC engine produces steady-state subsurface
concentration C₀ [atoms/m³] at each pressure P. If Sieverts' law holds
(confirmed by `check_sieverts_law`, R² ≥ 0.98), then:

```text
C₀(P)  =  S(T) · √P     →     S(T) = C₀(P) / √P
```

Evaluating at each converged pressure point in the sweep gives one S estimate
per point. Their mean is the empirical S at the sweep temperature.

To extract the Arrhenius parameters S₀ and ΔH_sol, repeat the sweep at several
temperatures and fit:

```text
ln S(T)  =  ln S₀  −  ΔH_sol / (k_B T)
```

**Implementation:**

```python
def fit_solubility_from_kmc(sweep_result):
    S_vals = [C0 / np.sqrt(P) for C0, P, ok
              in zip(sweep_result['C0_vals'], sweep_result['P_vals'],
                     sweep_result['converged']) if ok and P > 0]
    return {'S_mean': np.mean(S_vals), 'S_std': np.std(S_vals, ddof=1), ...}
```

**Use case:** Fully self-consistent with KMC — no separate NEB inputs needed
beyond what already ran. Acts as a cross-check on Options 1 and 2.
Only valid when R² ≥ 0.98 (Sieverts regime confirmed).

---

## 6.4 Dimensional Analysis

### Option 1

```text
S₀  =  4 / a₀³

Units:  [atoms] / [m³]  =  [atoms m⁻³]

With implicit Pa^(−½):  [atoms m⁻³ Pa^(−½)]   ✓
(Sieverts law: C [atoms/m³] = S [atoms m⁻³ Pa^(−½)] × √P [Pa^(½)])
```

### Option 2 — Full unit tracking

```text
ρ_oct              [m⁻³]
k_entry / k_exit   [s⁻¹ / s⁻¹]  =  [dimensionless]
k_diss             [dimensionless]
A_site             [m²]
k_des              [s⁻¹]
m_H2               [kg]
k_B                [J K⁻¹]  =  [kg m² s⁻² K⁻¹]
T                  [K]
```

Building the argument of the outer √:

```text
Numerator:    k_diss · A_site             →  [1] · [m²]  =  [m²]

Denominator:  k_des · √(2π m_H2 k_B T)

  √(m_H2 · k_B · T)  →  √([kg] · [kg m² s⁻²])
                      =  √([kg² m² s⁻²])
                      =  [kg m s⁻¹]           (momentum)

  k_des · [kg m s⁻¹]  →  [s⁻¹] · [kg m s⁻¹]
                       =  [kg m s⁻²]
                       =  [N]                 (force)

Ratio:  [m²] / [N]  =  [m² / (kg m s⁻²)]
                    =  [m s² / kg]
                    =  [m s² kg⁻¹]

√(ratio)  →  [m^(1/2) s kg^(−1/2)]
```

Full S(T):

```text
S  =  [m⁻³] · [1] · [m^(1/2) s kg^(−1/2)]
   =  [m^(−5/2) s kg^(−1/2)]
```

Target units [atoms m⁻³ Pa^(−½)]:

```text
Pa^(−½)  =  (kg m⁻¹ s⁻²)^(−½)
          =  kg^(−1/2) m^(1/2) s

[atoms m⁻³ Pa^(−½)]  =  [m⁻³] · [kg^(−1/2) m^(1/2) s]
                      =  [m^(−5/2) s kg^(−1/2)]           ✓
```

The units match exactly. The formula is dimensionally consistent.

### Option 3

```text
S  =  C₀ / √P

C₀  [atoms m⁻³]
√P  [Pa^(1/2)]  =  [kg^(1/2) m^(−1/2) s⁻¹]

S  =  [atoms m⁻³] / [kg^(1/2) m^(−1/2) s⁻¹]
   =  [atoms m^(−5/2) s kg^(−1/2)]
   =  [atoms m⁻³ Pa^(−½)]                    ✓
```

### Permeability Φ = D × S

```text
D    [m² s⁻¹]
S    [atoms m⁻³ Pa^(−½)]  =  [atoms m^(−5/2) s kg^(−1/2)]

Φ  =  [m² s⁻¹] · [atoms m^(−5/2) s kg^(−1/2)]
   =  [atoms m^(−1/2) kg^(−1/2)]

Pa^(−½) · m^(−1)  =  [kg^(−1/2) m^(1/2) s] · [m⁻¹]
                   =  [kg^(−1/2) m^(−1/2) s]

So Φ  ≡  [atoms m⁻¹ s⁻¹ Pa^(−½)]             ✓
```

### Richardson flux J = Φ (√P_high − √P_low) / L

```text
Φ          [atoms m⁻¹ s⁻¹ Pa^(−½)]
√P_high    [Pa^(½)]
L          [m]

J  =  [atoms m⁻¹ s⁻¹ Pa^(−½)] · [Pa^(½)] / [m]
   =  [atoms m⁻¹ s⁻¹] / [m]
   =  [atoms m⁻² s⁻¹]                         ✓
```

---

## 6.5 Richardson Flux ↔ Fick Flux — When They Agree

Fick's law at the high-pressure surface with a H-free permeate (C_low = 0):

```text
J_Fick  =  D · C₀ / L
```

In the Sieverts regime, C₀ = S · √P, so:

```text
J_Fick  =  D · S · √P / L
         =  Φ · √P / L
```

The Richardson equation with P_low = 0:

```text
J_Richardson  =  Φ · (√P_high − 0) / L
              =  Φ · √P / L
```

They are **identical** when:
1. P_low = 0 (or P_low ≪ P_high)
2. Sieverts' law holds (C₀ ∝ √P, confirmed by R² ≥ 0.98)

The two diverge when:
- P_low > 0 (Richardson accounts for back-pressure; Fick needs explicit C_low)
- Surface kinetics limit transport (C₀ ∝ P or C₀ ∝ P^n with n ≠ ½)

This equivalence is verified numerically in Cell 7 of `kmc_calculation.ipynb`
(consistency check: `J_Fick(S_KMC) vs J_Richardson(S_KMC)`).

---

## 6.6 Implemented Functions — `models/permeation.py` Section 4

| Function | Inputs | Output | Units |
|---|---|---|---|
| `arrhenius_diffusivity(D0, E_D, T)` | D₀ [m²/s], E_D [eV], T [K] | D(T) | m²/s |
| `lattice_site_S0(a0)` | a₀ [m] | S₀ | atoms m⁻³ Pa^(−½) |
| `solubility_from_rates(k_diss, k_des, k_entry, k_exit, a0, T)` | rates, a₀, T | S(T) | atoms m⁻³ Pa^(−½) |
| `fit_solubility_from_kmc(sweep_result)` | sweep dict | S_mean, S_std | atoms m⁻³ Pa^(−½) |
| `sieverts_solubility(dH_sol, S0, T)` | ΔH_sol [eV], S₀, T [K] | S(T) | atoms m⁻³ Pa^(−½) |
| `permeability(D, S)` | D [m²/s], S [atoms m⁻³ Pa^(−½)] | Φ | atoms m⁻¹ s⁻¹ Pa^(−½) |
| `richardson_flux(Phi, P_high, P_low, L)` | Φ, pressures [Pa], L [m] | J | atoms m⁻² s⁻¹ |

**Physical constants defined at module level (Section 4):**

```python
_KB_EV   = 8.617333262e-5    # eV / K
_KB_J    = 1.380649e-23      # J / K
_M_H2_KG = 2.0 × 1.6735575e-27  # kg  (H₂ molecule, 2 × proton mass)
```

---

## 6.7 Notebook Integration — `kmc_calculation.ipynb`

Two cells appended after Cell 6 (save sweep results):

### Cell 7 — All three S₀ options at target T

Inputs (user-editable):
- `D0_M2S`, `E_D_EV` — from LAMMPS MSD Arrhenius fit
- `DH_DISS_EV`, `DH_ENTRY_EV` — from NEB `delta_E` keys
- `P_HIGH_PA`, `P_LOW_PA`, `L_M` — membrane operating conditions

Outputs:

```text
Option 1 (lattice site density):
  S₀      = 9.171e+28 atoms/m³/Pa^½
  S(700K) = 3.330e+27 atoms/m³/Pa^½
  Φ       = ...
  J       = ...

Option 2 (detailed balance from TST rates):
  S(700K) = ...  [uses first available k_entry/k_exit from kmc_rate_dict]

Option 3 (KMC empirical, N converged points):
  S̄(700K) = ...  ± ...

Consistency: J_Fick(S_KMC) vs J_Richardson(S_KMC)  [should agree to <1%]
```

Saves `permeability_T{T}K.json` with all three options.

### Cell 8 — Φ(T) Arrhenius plot (400–1100 K)

- Left panel: Φ(T) vs T (linear scale) — all three options overlaid
- Right panel: log₁₀(Φ) vs 1000/T (Arrhenius plot) — slopes give E_Φ
- Vertical dashed line at simulation temperature T_K
- Prints effective activation energy: `E_Φ = E_D + ΔH_sol`
- Saves `permeability_vs_T.png`

---

## 6.8 Three-Option Comparison — Physical Interpretation

| Option | S₀ | ΔH_sol source | Best used when |
|---|---|---|---|
| 1 (lattice) | 4/a₀³ (fixed) | From NEB delta_E | Quick estimate, no rate data needed |
| 2 (rates) | Implicit in S(T) | Encoded in k_entry/k_exit ratio | Full TST pipeline complete |
| 3 (KMC) | Fitted from C₀/√P | Empirical at one T | Cross-check; requires Sieverts regime |

Option 2 is the most physically self-consistent because it derives S from the
same rate constants that govern the KMC dynamics. The ratio k_entry/k_exit
encodes ΔH_entry exactly (via detailed balance: k_fwd/k_rev = exp(−ΔG/kBT)),
and k_diss/k_des encodes ΔH_diss/2. No additional thermodynamic inputs beyond
the NEB barriers are needed.

Option 3 will match Option 2 in the Sieverts regime (R² ≥ 0.98) and diverge
from it when surface kinetics limit transport — this divergence is itself
diagnostic: if S_KMC < S_rates, the surface is the bottleneck.



# Plan: Part 2 Pipeline — `permeation_workflow.py` + `permeation.ipynb`

## Context

The full H permeation system has three parts:

| Part | Notebook | Workflow module | Status |
|---|---|---|---|
| 1 — Surface NEB | `neb_calculation.ipynb` | `models/neb_workflow.py` | ✅ Done |
| 3 — Bulk diffusivity | `diffusivity.ipynb` | `models/diffusivity_workflow.py` | ✅ Done |
| 2 — Surface→Subsurface→Permeation | *(missing)* | *(missing)* | ❌ To build |

Parts 1 and 3 follow the same architecture:
- The **notebook** is a thin **script generator + local analyser**. It writes a standalone Python orchestrator (`*_run.py`) and a SLURM wrapper (`*_run.sh`), then has Phase 4 analysis cells that run after the cluster jobs complete.
- The **workflow module** holds all the heavy logic: `generate_*_scripts()`, `generate_*_sh()`, and analysis functions (`load_*`, `plot_*`, `qc_*`).

Currently Part 2 has three *execution* notebooks (`subsurface_neb_calculation.ipynb`, `tst_calculation.ipynb`, `kmc_calculation.ipynb`) and five physics modules, but **no script-generator notebook** and **no workflow orchestration module**. This plan builds those two missing artefacts so Part 2 matches the architecture of Parts 1 and 3.

---

## What to Build

### 1. `models/permeation_workflow.py`

Mirrors `models/diffusivity_workflow.py` and `models/neb_workflow.py` in structure.

#### Section A — Script generation

```python
def generate_permeation_scripts(
    work_dir, relaxed_slab_path, surface_sites_json, phase2_h_dir,
    sub_neb_dir, vib_dir, results_dir,
    temperatures, p_vals_pa, a0_m, l_m,
    d0_m2s, e_d_ev,                       # from diffusivity_arrhenius.json
    dh_diss_ev, dh_entry_ev,              # from NEB delta_E (user fills)
    nx, ny, seed, kmc_max_steps,
    gpu_slurm_cfg, neb_slurm_cfg,
    vib_slurm_cfg,
    n_images, spring_const, neb_ftol,
    out_py,
) -> str
```

Uses identical **f-string header + raw r-string body** pattern as `diffusivity_workflow.py`.  
Header injects all config literals; body is the fixed orchestrator logic.  
Returns path to written `permeation_run.py`.

```python
def generate_permeation_sh(
    orch_job_name, orch_partition, orch_cpus_per_task, orch_mem, orch_time,
    orch_openmpi_ver, orch_cuda_version, orch_conda_env, orch_ld_paths,
    work_dir, out_py, out_sh,
) -> str
```

Writes SLURM wrapper. Returns path to `permeation_run.sh`.

#### Section B — Analysis functions (Phase 4, run locally)

| Function | Reads | Returns |
|---|---|---|
| `load_barrier_summary(sub_neb_dir)` | `hopa_jobs.json`, `hopb_jobs.json`, `neb_barrier.txt` per job | `pd.DataFrame` (sid, hop, E_abs, E_des, delta_E, converged) |
| `load_rate_summary(results_dir, temperatures)` | `rate_dict_T{T}K.json` | `pd.DataFrame` (label, T_K, k_forward, k_reverse, Ea_zpe, nu) |
| `load_kmc_sweeps(results_dir, temperatures)` | `permeation_sweep_T{T}K.json` | `dict[T → sweep_dict]` |
| `load_permeability_results(results_dir, temperatures)` | `permeability_T{T}K.json` | `dict[T → perm_dict]` |
| `plot_barrier_overview(df, sub_neb_dir)` | barrier DataFrame | bar chart Ea distribution (Hop A/B), saves `barriers_overview.png` |
| `plot_mep_overlay(sub_neb_dir)` | NEB path files | MEP overlay per hop, saves `mep_hopA.png`, `mep_hopB.png` |
| `plot_kmc_sieverts(results_dir, temperatures)` | sweep JSONs | J vs √P for each T, saves `sieverts_check.png` |
| `plot_permeability_vs_T(results_dir, temperatures)` | permeability JSONs | Φ(T) linear + Arrhenius (all 3 options), saves `permeability_vs_T.png` |
| `plot_arrhenius_S0(results_dir)` | `solubility_arrhenius_kmc.json` | Multi-T ln(S) vs 1/T fit overlay, saves `solubility_arrhenius.png` |
| `plot_bottleneck(results_dir, temperatures)` | sweep + Sieverts JSONs | R² vs T bar chart (1 = bulk-limited, <0.98 = surface-limited), saves `bottleneck.png` |

---

### 2. `calculation/permeation.ipynb`

Thin notebook — no heavy computation inline. Mirrors `neb_calculation.ipynb` and `diffusivity.ipynb`.

#### Script-generation cells

| Cell | Purpose |
|---|---|
| Cell 1 | `import os, sys, json` + `sys.path` setup |
| Cell 2 | All config vars (paths, temperatures, pressures, lattice, KMC params, SLURM configs) |
| Cell 3 | Call `generate_permeation_scripts(...)` → writes `permeation_run.py` |
| Cell 4 | Call `generate_permeation_sh(...)` + `submit_slurm_job(dry_run=True)` |

#### Phase 4 — Local analysis cells (run after cluster jobs)

| Phase | Cell function called |
|---|---|
| 4a. Barrier overview | `load_barrier_summary()`, `plot_barrier_overview()` |
| 4b. MEP overlay | `plot_mep_overlay()` |
| 4c. TST rate summary | `load_rate_summary()`, rate vs T plot |
| 4d. KMC Sieverts check | `load_kmc_sweeps()`, `plot_kmc_sieverts()` |
| 4e. Φ(T) plot | `load_permeability_results()`, `plot_permeability_vs_T()` |
| 4f. Arrhenius S₀ | `plot_arrhenius_S0()` |
| 4g. Bottleneck diagnosis | `plot_bottleneck()` |

---

## Generated `permeation_run.py` — Phases and Execution Order

```
Phase 1a — Hop A FS-min  (GPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 1b — Hop A NEB     (CPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 2a — Hop B FS-min  (GPU SLURM array, parallel; IS = Hop A relaxed sub1)
    ↓ wait_for_jobs
Phase 2b — Hop B NEB     (CPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 3  — Vibrations    (CPU SLURM array; IS + TS for both hops, all parallel)
    ↓ wait_for_jobs
Phase 4  — TST rates     (local inside orchestrator; fast; one rate_dict_T{T}K.json per T)
Phase 5  — KMC sweeps    (local inside orchestrator; sweep_pressure() at each T)
Phase 6  — Permeability  (local; all 3 S₀ options; Arrhenius fit; save all JSONs)
```

Phases 1–3 use `submit_slurm_job()` + `wait_for_jobs()` (same pattern as `neb_run.py`).  
Phases 4–6 run inside the orchestrator process (no additional SLURM jobs needed for the default grid size NX=NY=20).

---

## Reused Functions (no reimplementation needed)

| Function | Module |
|---|---|
| `orchestrate_hopa_neb(...)` | `models/neb_subsurface.py` |
| `orchestrate_hopb_neb(...)` | `models/neb_subsurface.py` |
| `collect_is_ts_paths(...)` | `models/vibrations.py` |
| `orchestrate_vibrations(...)` | `models/vibrations.py` |
| `collect_neb_results(...)` | `models/tst_rates.py` |
| `split_vib_results(...)` | `models/tst_rates.py` |
| `build_rate_dict(...)` | `models/tst_rates.py` |
| `rates_to_json(...)` | `models/tst_rates.py` |
| `sweep_pressure(...)` | `models/permeation.py` |
| `fit_solubility_from_kmc(...)` | `models/permeation.py` |
| `arrhenius_diffusivity(...)` | `models/permeation.py` |
| `permeability(...)`, `richardson_flux(...)` | `models/permeation.py` |
| `submit_slurm_job(...)`, `wait_for_jobs(...)` | `models/create_slurm.py` |
| `parse_barrier_file(...)` | `models/parsers.py` |

---

## Inputs Required (from Parts 1 and 3)

| Variable | Source |
|---|---|
| `RELAXED_SLAB_PATH` | Part 1 — `calculation/slabs/slab_relaxed.lammps` |
| `SURFACE_SITES_JSON` | Part 1 — `calculation/slabs/surface_sites.json` |
| `PHASE2_H_DIR` | Part 1 — `calculation/adsorption/h_atom/` (dedup IS `.lammps` files) |
| `D0_M2S`, `E_D_EV` | Part 3 — `results/diffusivity_arrhenius.json` (loaded in Cell 2 if present) |
| `DH_DISS_EV`, `DH_ENTRY_EV` | User fills from NEB `delta_E` values after cluster run |

---

## Key Config Variables (Cell 2 of notebook)

```python
# Paths
WORK_DIR            = os.path.join(BASE_DIR, 'calculation')
RELAXED_SLAB_PATH   = os.path.join(WORK_DIR, 'slabs', 'slab_relaxed.lammps')
SURFACE_SITES_JSON  = os.path.join(WORK_DIR, 'slabs', 'surface_sites.json')
PHASE2_H_DIR        = os.path.join(WORK_DIR, 'adsorption', 'h_atom')
SUB_NEB_DIR         = os.path.join(WORK_DIR, 'neb_subsurface')
VIB_DIR             = os.path.join(WORK_DIR, 'vibrations')
RESULTS_DIR         = os.path.join(WORK_DIR, 'results')

# Physical parameters
TEMPERATURES   = [500, 600, 700, 800, 900]   # K
P_VALS_PA      = [1e3, 1e4, 1e5, 5e5, 1e6]  # Pa
A0_M           = 3.52e-10   # m
L_M            = 1e-3       # m

# Diffusivity — loaded from Part 3 output if available
D0_M2S = 1.5e-7    # placeholder
E_D_EV = 0.40      # placeholder

# NEB thermodynamics — user fills after NEB jobs complete
DH_DISS_EV  = None   # eV (dissociation NEB delta_E)
DH_ENTRY_EV = None   # eV (Hop A delta_E)

# KMC
NX, NY      = 20, 20
SEED        = 42
KMC_MAX_STEPS = 500_000

# SLURM
GPU_SLURM_CFG = dict(SLURM_DEFAULTS, partition='multigpu', time='04:00:00')
NEB_SLURM_CFG = dict(SLURM_DEFAULTS, partition='short', gpu=None, cpus_per_task=16, time='12:00:00')
VIB_SLURM_CFG = dict(SLURM_DEFAULTS, partition='short', gpu=None, cpus_per_task=8,  time='06:00:00')

# NEB
N_IMAGES    = N_REPLICAS
SPRING_CONST = SPRING_CONST
NEB_FTOL    = NEB_FTOL
```

---

## Verification

1. **Script generation**: Run Cells 1–4 with `dry_run=True` → `permeation_run.py` and `permeation_run.sh` are written, no SLURM jobs submitted. Check that the injected config matches the notebook variables.
2. **Orchestrator structure**: Open `permeation_run.py` and verify 6 phases are present and functions are correctly called.
3. **Phase 4 analysis cells**: Run each analysis cell with mock JSON files (or real output from a dry-run test at one temperature) → plots saved without error.
4. **End-to-end**: After a cluster run at one temperature (T=700K), run all Phase 4 cells — barrier DataFrame has at least one row, Sieverts R² prints, Φ(T) plot saves.

---

## Files to Create

| File | Action |
|---|---|
| `models/permeation_workflow.py` | New file — ~500 lines |
| `calculation/permeation.ipynb` | New notebook — Cells 1–4 (script gen) + Phase 4 analysis cells |

The three existing notebooks (`subsurface_neb_calculation.ipynb`, `tst_calculation.ipynb`, `kmc_calculation.ipynb`) remain valid for interactive step-by-step use. `permeation.ipynb` is the canonical automated entry point.





# Plan: Full Pipeline Automation — ZPE, a₀(T), and Master Orchestrator

## Context

The three-part pipeline is structurally complete (`permeation_workflow.py` + `permeation.ipynb` were built in the previous session). Two physics gaps and one architectural gap remain before the pipeline is fully automated:

1. **ZPE missing for dissociation NEB (Part 1)** — `neb_workflow.py` runs the H₂* → 2H* NEB but never computes vibrational frequencies at IS or TS. `k_diss` and `k_des` in the KMC engine therefore use raw barriers, inconsistent with the Vineyard+ZPE treatment for Hop A/B in Part 2.

2. **a₀(T) not saved (Part 3)** — `diffusivity_run.py` computes temperature-dependent lattice parameter from NPT MD (`get_lattice_parameter()`) but discards it. Part 2 KMC uses a fixed `A0_M = 3.52e-10 m` at all temperatures, ignoring thermal expansion.

3. **No master orchestrator** — Parts 1, 2, 3 are run independently with manual handoffs for `DH_DISS_EV`, `DH_ENTRY_EV`, `D0_M2S`, `E_D_EV`. The goal is full automation: one command starts everything, values flow automatically from outputs of earlier parts into later parts.

---

## Current status of three parts

| Part | Notebook | Workflow module | Status |
|---|---|---|---|
| 1 — Surface NEB | `neb_calculation.ipynb` | `models/neb_workflow.py` | ✅ Done (missing diss vib) |
| 3 — Bulk diffusivity | `diffusivity.ipynb` | `models/diffusivity_workflow.py` | ✅ Done (missing a₀ save) |
| 2 — Surface→Subsurface | `permeation.ipynb` | `models/permeation_workflow.py` | ✅ Done (missing T-dep a₀, ZPE diss) |

---

## What to Change

### Fix 1 — Save a₀(T) in Part 3 (`models/diffusivity_workflow.py`)

In the raw `_body` r-string, inside the NPT loop where `a0_T = get_lattice_parameter(...)` is computed (Phase 1b), collect all T→a₀ pairs and after the loop write:

```
{RESULTS_DIR}/lattice_params_vs_T.json
→ {"temperatures": [T1, T2, ...], "a0_m": [a0_1, a0_2, ...]}
```

This is a 5-line addition to the existing body — no structural changes.

---

### Fix 2 — Add ZPE to dissociation NEB in Part 1 (`models/neb_workflow.py`)

The dissociation NEB orchestrator (`generate_neb_scripts()` → `neb_run.py` body) currently has two phases: FS-min SLURM array → NEB SLURM array. Add **Phase 3: vibrations for dissociation IS + TS**.

**How:**
- The NEB run script already saves `neb_initial.lammps` (IS = H₂* structure).
- Need to also save `ts_image.lammps` (peak-energy NEB image). Add a post-step inside `run_neb.py` body: after CINEB converges, identify the image index with maximum ΔE, write it as `ts_image.lammps` using `ase.io.write`.
- After NEB SLURM array finishes: collect `('{label}_is', 'neb_initial.lammps')` and `('{label}_ts', 'ts_image.lammps')` pairs → call `orchestrate_vibrations()` → submit vib SLURM array → wait.
- After vib finishes: call `build_rate_dict()` (already in `tst_rates.py`) per-pair, save `diss_vib_rates.json`:

```json
{"Ni-Ni": {"Ea_zpe": 0.51, "Ed_zpe": 1.18, "nu_diss": 1.2e13, "nu_des": 9.8e12}, ...}
```

The pair key comes from the site labels (same logic already used in `kmc_calculation.ipynb` Cell 2 for building `k_diss`/`k_des` lookup).

**Files changed:**
- `models/neb_workflow.py` — (a) add TS extraction to `run_neb.py` body; (b) add Phase 3 to the `neb_run.py` body r-string after NEB wait
- `neb_calculation.ipynb` — no cell changes needed (Phase 3 runs automatically inside `neb_run.py`)

---

### Fix 3 — Use a₀(T) + ZPE diss rates in Part 2 (`models/permeation_workflow.py`)

In `_PERMEATION_BODY`, Phase 5 (KMC loop), replace fixed `A0_M` per temperature:

```python
# Load lattice_params_vs_T.json if available
_LAT_JSON = os.path.join(RESULTS_DIR, 'lattice_params_vs_T.json')
if os.path.exists(_LAT_JSON):
    with open(_LAT_JSON) as _f: _lat = json.load(_f)
    _a0_dict = dict(zip(_lat['temperatures'], _lat['a0_m']))
else:
    _a0_dict = {}

# In per-T loop:
_a0_T = _a0_dict.get(_T, A0_M)   # falls back to notebook constant if Part 3 not run
```

Replace raw-barrier `k_diss`/`k_des` with ZPE-corrected rates when `diss_vib_rates.json` exists:

```python
_DISS_VIB_JSON = os.path.join(WORK_DIR, 'neb', 'diss_vib_rates.json')
if os.path.exists(_DISS_VIB_JSON):
    # use nu * exp(-Ea_zpe / kBT) per pair
else:
    # fallback: existing diss_jobs.json / placeholder logic (unchanged)
```

Auto-extract `DH_ENTRY_EV` inside Phase 6 from Hop A barrier results instead of requiring `DH_ENTRY_EV is not None`:

```python
# After Phase 1 Hop A NEB completes, collect mean converged delta_E:
_dh_entry = np.mean([job['delta_E'] for job in hopa_jobs if job.get('converged')])
```

Auto-extract `DH_DISS_EV` from `ranked_barriers.json` (mean of converged delta_E values):

```python
_ranked = os.path.join(WORK_DIR, 'neb', 'ranked_barriers.json')
if os.path.exists(_ranked):
    with open(_ranked) as _f: _rb = json.load(_f)
    _dh_diss = np.mean([r['delta_E'] for r in _rb if r.get('converged')])
```

When both are auto-extracted, Phase 6 runs unconditionally (removes the `if DH_DISS_EV is None` guard).

**Also fix bug in `permeation.ipynb` Cell 2:** `SLURM_DEFAULTS.get('cuda_ver', '')` → `SLURM_DEFAULTS.get('cuda_version', '')` (key mismatch with actual config).

---

### Fix 4 — Master pipeline notebook (NEW)

**New files:**
- `models/pipeline_workflow.py` — `generate_pipeline_scripts()`, `generate_pipeline_sh()`
- `calculation/pipeline.ipynb` — master notebook

**Architecture:**

```
pipeline_run.py
    ├── subprocess.Popen(neb_run.py)        ← Part 1
    ├── subprocess.Popen(diffusivity_run.py) ← Part 3  (in parallel with Part 1)
    ├── p1.wait() + p3.wait()
    └── subprocess.run(permeation_run.py)   ← Part 2  (after both complete)
```

Both Part 1 and Part 3 are independent and can run in parallel. Part 2 depends on both (needs `ranked_barriers.json` + `adsorption/h_atom/` from Part 1; `diffusivity_arrhenius.json` + `lattice_params_vs_T.json` from Part 3).

The `pipeline_run.sh` SLURM wrapper submits this as a single long-running orchestrator job.

`pipeline.ipynb` cells:
1. Shared config (superset of all three parts' configs)
2. Generate all three run scripts (calls `generate_neb_scripts()`, `generate_diffusivity_scripts()`, `generate_permeation_scripts()`)
3. Generate `pipeline_run.py` + `pipeline_run.sh`, submit with `dry_run=True`

**Note:** The three individual notebooks remain valid for partial re-runs or debug.

---

## Data flow diagram

```
Part 1 (neb_run.py)                    Part 3 (diffusivity_run.py)
  └─ ranked_barriers.json ─────────┐     └─ diffusivity_arrhenius.json ─┐
  └─ diss_vib_rates.json ──────────┤     └─ lattice_params_vs_T.json ───┤
  └─ adsorption/h_atom/ ───────────┤                                     │
                                   ▼                                     ▼
                           Part 2 (permeation_run.py)
                             auto-reads all ↑ at runtime
                             no manual filling needed
```

---

## Files to change / create

| File | Change |
|---|---|
| `models/diffusivity_workflow.py` | Add a₀(T) JSON save to `_body` (~5 lines) |
| `models/neb_workflow.py` | Add TS extraction to NEB body; add Phase 3 (diss vib) to neb_run.py body |
| `models/permeation_workflow.py` | Update `_PERMEATION_BODY`: T-dep a₀, ZPE diss rates, auto DH values |
| `calculation/permeation.ipynb` | Fix `cuda_ver` → `cuda_version` key bug in Cell 2 |
| `models/pipeline_workflow.py` | NEW — `generate_pipeline_scripts()`, `generate_pipeline_sh()` |
| `calculation/pipeline.ipynb` | NEW — master generator notebook |

---

## Verification

1. `diffusivity_run.py` writes `lattice_params_vs_T.json` with one entry per temperature.
2. `permeation_run.py` prints `a0(T)` per temperature from the JSON (not the fixed constant).
3. After Part 1, `diss_vib_rates.json` exists with ZPE-corrected rates per element pair.
4. Phase 6 of `permeation_run.py` runs without requiring manual `DH_DISS_EV`/`DH_ENTRY_EV`.
5. `pipeline.ipynb` Cell 3 generates `pipeline_run.py`; opening it shows Part 1 + Part 3 started in parallel then Part 2 after both complete.

---

## Previous plan content (already implemented)

Parts 1 and 3 follow the same architecture:
- The **notebook** is a thin **script generator + local analyser**. It writes a standalone Python orchestrator (`*_run.py`) and a SLURM wrapper (`*_run.sh`), then has Phase 4 analysis cells that run after the cluster jobs complete.
- The **workflow module** holds all the heavy logic: `generate_*_scripts()`, `generate_*_sh()`, and analysis functions (`load_*`, `plot_*`, `qc_*`).

Currently Part 2 has three *execution* notebooks (`subsurface_neb_calculation.ipynb`, `tst_calculation.ipynb`, `kmc_calculation.ipynb`) and five physics modules, but **no script-generator notebook** and **no workflow orchestration module**. This plan builds those two missing artefacts so Part 2 matches the architecture of Parts 1 and 3.

---

## What to Build

### 1. `models/permeation_workflow.py`

Mirrors `models/diffusivity_workflow.py` and `models/neb_workflow.py` in structure.

#### Section A — Script generation

```python
def generate_permeation_scripts(
    work_dir, relaxed_slab_path, surface_sites_json, phase2_h_dir,
    sub_neb_dir, vib_dir, results_dir,
    temperatures, p_vals_pa, a0_m, l_m,
    d0_m2s, e_d_ev,                       # from diffusivity_arrhenius.json
    dh_diss_ev, dh_entry_ev,              # from NEB delta_E (user fills)
    nx, ny, seed, kmc_max_steps,
    gpu_slurm_cfg, neb_slurm_cfg,
    vib_slurm_cfg,
    n_images, spring_const, neb_ftol,
    out_py,
) -> str
```

Uses identical **f-string header + raw r-string body** pattern as `diffusivity_workflow.py`.  
Header injects all config literals; body is the fixed orchestrator logic.  
Returns path to written `permeation_run.py`.

```python
def generate_permeation_sh(
    orch_job_name, orch_partition, orch_cpus_per_task, orch_mem, orch_time,
    orch_openmpi_ver, orch_cuda_version, orch_conda_env, orch_ld_paths,
    work_dir, out_py, out_sh,
) -> str
```

Writes SLURM wrapper. Returns path to `permeation_run.sh`.

#### Section B — Analysis functions (Phase 4, run locally)

| Function | Reads | Returns |
|---|---|---|
| `load_barrier_summary(sub_neb_dir)` | `hopa_jobs.json`, `hopb_jobs.json`, `neb_barrier.txt` per job | `pd.DataFrame` (sid, hop, E_abs, E_des, delta_E, converged) |
| `load_rate_summary(results_dir, temperatures)` | `rate_dict_T{T}K.json` | `pd.DataFrame` (label, T_K, k_forward, k_reverse, Ea_zpe, nu) |
| `load_kmc_sweeps(results_dir, temperatures)` | `permeation_sweep_T{T}K.json` | `dict[T → sweep_dict]` |
| `load_permeability_results(results_dir, temperatures)` | `permeability_T{T}K.json` | `dict[T → perm_dict]` |
| `plot_barrier_overview(df, sub_neb_dir)` | barrier DataFrame | bar chart Ea distribution (Hop A/B), saves `barriers_overview.png` |
| `plot_mep_overlay(sub_neb_dir)` | NEB path files | MEP overlay per hop, saves `mep_hopA.png`, `mep_hopB.png` |
| `plot_kmc_sieverts(results_dir, temperatures)` | sweep JSONs | J vs √P for each T, saves `sieverts_check.png` |
| `plot_permeability_vs_T(results_dir, temperatures)` | permeability JSONs | Φ(T) linear + Arrhenius (all 3 options), saves `permeability_vs_T.png` |
| `plot_arrhenius_S0(results_dir)` | `solubility_arrhenius_kmc.json` | Multi-T ln(S) vs 1/T fit overlay, saves `solubility_arrhenius.png` |
| `plot_bottleneck(results_dir, temperatures)` | sweep + Sieverts JSONs | R² vs T bar chart (1 = bulk-limited, <0.98 = surface-limited), saves `bottleneck.png` |

---

### 2. `calculation/permeation.ipynb`

Thin notebook — no heavy computation inline. Mirrors `neb_calculation.ipynb` and `diffusivity.ipynb`.

#### Script-generation cells

| Cell | Purpose |
|---|---|
| Cell 1 | `import os, sys, json` + `sys.path` setup |
| Cell 2 | All config vars (paths, temperatures, pressures, lattice, KMC params, SLURM configs) |
| Cell 3 | Call `generate_permeation_scripts(...)` → writes `permeation_run.py` |
| Cell 4 | Call `generate_permeation_sh(...)` + `submit_slurm_job(dry_run=True)` |

#### Phase 4 — Local analysis cells (run after cluster jobs)

| Phase | Cell function called |
|---|---|
| 4a. Barrier overview | `load_barrier_summary()`, `plot_barrier_overview()` |
| 4b. MEP overlay | `plot_mep_overlay()` |
| 4c. TST rate summary | `load_rate_summary()`, rate vs T plot |
| 4d. KMC Sieverts check | `load_kmc_sweeps()`, `plot_kmc_sieverts()` |
| 4e. Φ(T) plot | `load_permeability_results()`, `plot_permeability_vs_T()` |
| 4f. Arrhenius S₀ | `plot_arrhenius_S0()` |
| 4g. Bottleneck diagnosis | `plot_bottleneck()` |

---

## Generated `permeation_run.py` — Phases and Execution Order

```
Phase 1a — Hop A FS-min  (GPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 1b — Hop A NEB     (CPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 2a — Hop B FS-min  (GPU SLURM array, parallel; IS = Hop A relaxed sub1)
    ↓ wait_for_jobs
Phase 2b — Hop B NEB     (CPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 3  — Vibrations    (CPU SLURM array; IS + TS for both hops, all parallel)
    ↓ wait_for_jobs
Phase 4  — TST rates     (local inside orchestrator; fast; one rate_dict_T{T}K.json per T)
Phase 5  — KMC sweeps    (local inside orchestrator; sweep_pressure() at each T)
Phase 6  — Permeability  (local; all 3 S₀ options; Arrhenius fit; save all JSONs)
```

Phases 1–3 use `submit_slurm_job()` + `wait_for_jobs()` (same pattern as `neb_run.py`).  
Phases 4–6 run inside the orchestrator process (no additional SLURM jobs needed for the default grid size NX=NY=20).

---

## Reused Functions (no reimplementation needed)

| Function | Module |
|---|---|
| `orchestrate_hopa_neb(...)` | `models/neb_subsurface.py` |
| `orchestrate_hopb_neb(...)` | `models/neb_subsurface.py` |
| `collect_is_ts_paths(...)` | `models/vibrations.py` |
| `orchestrate_vibrations(...)` | `models/vibrations.py` |
| `collect_neb_results(...)` | `models/tst_rates.py` |
| `split_vib_results(...)` | `models/tst_rates.py` |
| `build_rate_dict(...)` | `models/tst_rates.py` |
| `rates_to_json(...)` | `models/tst_rates.py` |
| `sweep_pressure(...)` | `models/permeation.py` |
| `fit_solubility_from_kmc(...)` | `models/permeation.py` |
| `arrhenius_diffusivity(...)` | `models/permeation.py` |
| `permeability(...)`, `richardson_flux(...)` | `models/permeation.py` |
| `submit_slurm_job(...)`, `wait_for_jobs(...)` | `models/create_slurm.py` |
| `parse_barrier_file(...)` | `models/parsers.py` |

---

## Inputs Required (from Parts 1 and 3)

| Variable | Source |
|---|---|
| `RELAXED_SLAB_PATH` | Part 1 — `calculation/slabs/slab_relaxed.lammps` |
| `SURFACE_SITES_JSON` | Part 1 — `calculation/slabs/surface_sites.json` |
| `PHASE2_H_DIR` | Part 1 — `calculation/adsorption/h_atom/` (dedup IS `.lammps` files) |
| `D0_M2S`, `E_D_EV` | Part 3 — `results/diffusivity_arrhenius.json` (loaded in Cell 2 if present) |
| `DH_DISS_EV`, `DH_ENTRY_EV` | User fills from NEB `delta_E` values after cluster run |

---

## Key Config Variables (Cell 2 of notebook)

```python
# Paths
WORK_DIR            = os.path.join(BASE_DIR, 'calculation')
RELAXED_SLAB_PATH   = os.path.join(WORK_DIR, 'slabs', 'slab_relaxed.lammps')
SURFACE_SITES_JSON  = os.path.join(WORK_DIR, 'slabs', 'surface_sites.json')
PHASE2_H_DIR        = os.path.join(WORK_DIR, 'adsorption', 'h_atom')
SUB_NEB_DIR         = os.path.join(WORK_DIR, 'neb_subsurface')
VIB_DIR             = os.path.join(WORK_DIR, 'vibrations')
RESULTS_DIR         = os.path.join(WORK_DIR, 'results')

# Physical parameters
TEMPERATURES   = [500, 600, 700, 800, 900]   # K
P_VALS_PA      = [1e3, 1e4, 1e5, 5e5, 1e6]  # Pa
A0_M           = 3.52e-10   # m
L_M            = 1e-3       # m

# Diffusivity — loaded from Part 3 output if available
D0_M2S = 1.5e-7    # placeholder
E_D_EV = 0.40      # placeholder

# NEB thermodynamics — user fills after NEB jobs complete
DH_DISS_EV  = None   # eV (dissociation NEB delta_E)
DH_ENTRY_EV = None   # eV (Hop A delta_E)

# KMC
NX, NY      = 20, 20
SEED        = 42
KMC_MAX_STEPS = 500_000

# SLURM
GPU_SLURM_CFG = dict(SLURM_DEFAULTS, partition='multigpu', time='04:00:00')
NEB_SLURM_CFG = dict(SLURM_DEFAULTS, partition='short', gpu=None, cpus_per_task=16, time='12:00:00')
VIB_SLURM_CFG = dict(SLURM_DEFAULTS, partition='short', gpu=None, cpus_per_task=8,  time='06:00:00')

# NEB
N_IMAGES    = N_REPLICAS
SPRING_CONST = SPRING_CONST
NEB_FTOL    = NEB_FTOL
```

---

## Verification

1. **Script generation**: Run Cells 1–4 with `dry_run=True` → `permeation_run.py` and `permeation_run.sh` are written, no SLURM jobs submitted. Check that the injected config matches the notebook variables.
2. **Orchestrator structure**: Open `permeation_run.py` and verify 6 phases are present and functions are correctly called.
3. **Phase 4 analysis cells**: Run each analysis cell with mock JSON files (or real output from a dry-run test at one temperature) → plots saved without error.
4. **End-to-end**: After a cluster run at one temperature (T=700K), run all Phase 4 cells — barrier DataFrame has at least one row, Sieverts R² prints, Φ(T) plot saves.

---

## Files to Create

| File | Action |
|---|---|
| `models/permeation_workflow.py` | New file — ~500 lines |
| `calculation/permeation.ipynb` | New notebook — Cells 1–4 (script gen) + Phase 4 analysis cells |

The three existing notebooks (`subsurface_neb_calculation.ipynb`, `tst_calculation.ipynb`, `kmc_calculation.ipynb`) remain valid for interactive step-by-step use. `permeation.ipynb` is the canonical automated entry point.



Ready for review
Select text to add comments on the plan
Plan: Full Pipeline Automation — ZPE, a₀(T), and Master Orchestrator
Context
The three-part pipeline is structurally complete (permeation_workflow.py + permeation.ipynb were built in the previous session). Two physics gaps and one architectural gap remain before the pipeline is fully automated:

ZPE missing for dissociation NEB (Part 1) — neb_workflow.py runs the H₂* → 2H* NEB but never computes vibrational frequencies at IS or TS. k_diss and k_des in the KMC engine therefore use raw barriers, inconsistent with the Vineyard+ZPE treatment for Hop A/B in Part 2.

a₀(T) not saved (Part 3) — diffusivity_run.py computes temperature-dependent lattice parameter from NPT MD (get_lattice_parameter()) but discards it. Part 2 KMC uses a fixed A0_M = 3.52e-10 m at all temperatures, ignoring thermal expansion.

No master orchestrator — Parts 1, 2, 3 are run independently with manual handoffs for DH_DISS_EV, DH_ENTRY_EV, D0_M2S, E_D_EV. The goal is full automation: one command starts everything, values flow automatically from outputs of earlier parts into later parts.

Current status of three parts
Part	Notebook	Workflow module	Status
1 — Surface NEB	neb_calculation.ipynb	models/neb_workflow.py	✅ Done (missing diss vib)
3 — Bulk diffusivity	diffusivity.ipynb	models/diffusivity_workflow.py	✅ Done (missing a₀ save)
2 — Surface→Subsurface	permeation.ipynb	models/permeation_workflow.py	✅ Done (missing T-dep a₀, ZPE diss)
What to Change
Fix 1 — Save a₀(T) in Part 3 (models/diffusivity_workflow.py)
In the raw _body r-string, inside the NPT loop where a0_T = get_lattice_parameter(...) is computed (Phase 1b), collect all T→a₀ pairs and after the loop write:

{RESULTS_DIR}/lattice_params_vs_T.json
→ {"temperatures": [T1, T2, ...], "a0_m": [a0_1, a0_2, ...]}
This is a 5-line addition to the existing body — no structural changes.

Fix 2 — Add ZPE to dissociation NEB in Part 1 (models/neb_workflow.py)
The dissociation NEB orchestrator (generate_neb_scripts() → neb_run.py body) currently has two phases: FS-min SLURM array → NEB SLURM array. Add Phase 3: vibrations for dissociation IS + TS.

How:

The NEB run script already saves neb_initial.lammps (IS = H₂* structure).
Need to also save ts_image.lammps (peak-energy NEB image). Add a post-step inside run_neb.py body: after CINEB converges, identify the image index with maximum ΔE, write it as ts_image.lammps using ase.io.write.
After NEB SLURM array finishes: collect ('{label}_is', 'neb_initial.lammps') and ('{label}_ts', 'ts_image.lammps') pairs → call orchestrate_vibrations() → submit vib SLURM array → wait.
After vib finishes: call build_rate_dict() (already in tst_rates.py) per-pair, save diss_vib_rates.json:
{"Ni-Ni": {"Ea_zpe": 0.51, "Ed_zpe": 1.18, "nu_diss": 1.2e13, "nu_des": 9.8e12}, ...}
The pair key comes from the site labels (same logic already used in kmc_calculation.ipynb Cell 2 for building k_diss/k_des lookup).

Files changed:

models/neb_workflow.py — (a) add TS extraction to run_neb.py body; (b) add Phase 3 to the neb_run.py body r-string after NEB wait
neb_calculation.ipynb — no cell changes needed (Phase 3 runs automatically inside neb_run.py)
Fix 3 — Use a₀(T) + ZPE diss rates in Part 2 (models/permeation_workflow.py)
In _PERMEATION_BODY, Phase 5 (KMC loop), replace fixed A0_M per temperature:

# Load lattice_params_vs_T.json if available
_LAT_JSON = os.path.join(RESULTS_DIR, 'lattice_params_vs_T.json')
if os.path.exists(_LAT_JSON):
    with open(_LAT_JSON) as _f: _lat = json.load(_f)
    _a0_dict = dict(zip(_lat['temperatures'], _lat['a0_m']))
else:
    _a0_dict = {}

# In per-T loop:
_a0_T = _a0_dict.get(_T, A0_M)   # falls back to notebook constant if Part 3 not run
Replace raw-barrier k_diss/k_des with ZPE-corrected rates when diss_vib_rates.json exists:

_DISS_VIB_JSON = os.path.join(WORK_DIR, 'neb', 'diss_vib_rates.json')
if os.path.exists(_DISS_VIB_JSON):
    # use nu * exp(-Ea_zpe / kBT) per pair
else:
    # fallback: existing diss_jobs.json / placeholder logic (unchanged)
Auto-extract DH_ENTRY_EV inside Phase 6 from Hop A barrier results instead of requiring DH_ENTRY_EV is not None:

# After Phase 1 Hop A NEB completes, collect mean converged delta_E:
_dh_entry = np.mean([job['delta_E'] for job in hopa_jobs if job.get('converged')])
Auto-extract DH_DISS_EV from ranked_barriers.json (mean of converged delta_E values):

_ranked = os.path.join(WORK_DIR, 'neb', 'ranked_barriers.json')
if os.path.exists(_ranked):
    with open(_ranked) as _f: _rb = json.load(_f)
    _dh_diss = np.mean([r['delta_E'] for r in _rb if r.get('converged')])
When both are auto-extracted, Phase 6 runs unconditionally (removes the if DH_DISS_EV is None guard).

Also fix bug in permeation.ipynb Cell 2: SLURM_DEFAULTS.get('cuda_ver', '') → SLURM_DEFAULTS.get('cuda_version', '') (key mismatch with actual config).

Fix 4 — Master pipeline notebook (NEW)
New files:

models/pipeline_workflow.py — generate_pipeline_scripts(), generate_pipeline_sh()
calculation/pipeline.ipynb — master notebook
Architecture:

pipeline_run.py
    ├── subprocess.Popen(neb_run.py)        ← Part 1
    ├── subprocess.Popen(diffusivity_run.py) ← Part 3  (in parallel with Part 1)
    ├── p1.wait() + p3.wait()
    └── subprocess.run(permeation_run.py)   ← Part 2  (after both complete)
Both Part 1 and Part 3 are independent and can run in parallel. Part 2 depends on both (needs ranked_barriers.json + adsorption/h_atom/ from Part 1; diffusivity_arrhenius.json + lattice_params_vs_T.json from Part 3).

The pipeline_run.sh SLURM wrapper submits this as a single long-running orchestrator job.

pipeline.ipynb cells:

Shared config (superset of all three parts' configs)
Generate all three run scripts (calls generate_neb_scripts(), generate_diffusivity_scripts(), generate_permeation_scripts())
Generate pipeline_run.py + pipeline_run.sh, submit with dry_run=True
Note: The three individual notebooks remain valid for partial re-runs or debug.

Data flow diagram
Part 1 (neb_run.py)                    Part 3 (diffusivity_run.py)
  └─ ranked_barriers.json ─────────┐     └─ diffusivity_arrhenius.json ─┐
  └─ diss_vib_rates.json ──────────┤     └─ lattice_params_vs_T.json ───┤
  └─ adsorption/h_atom/ ───────────┤                                     │
                                   ▼                                     ▼
                           Part 2 (permeation_run.py)
                             auto-reads all ↑ at runtime
                             no manual filling needed
Files to change / create
File	Change
models/diffusivity_workflow.py	Add a₀(T) JSON save to _body (~5 lines)
models/neb_workflow.py	Add TS extraction to NEB body; add Phase 3 (diss vib) to neb_run.py body
models/permeation_workflow.py	Update _PERMEATION_BODY: T-dep a₀, ZPE diss rates, auto DH values
calculation/permeation.ipynb	Fix cuda_ver → cuda_version key bug in Cell 2
models/pipeline_workflow.py	NEW — generate_pipeline_scripts(), generate_pipeline_sh()
calculation/pipeline.ipynb	NEW — master generator notebook
Verification
diffusivity_run.py writes lattice_params_vs_T.json with one entry per temperature.
permeation_run.py prints a0(T) per temperature from the JSON (not the fixed constant).
After Part 1, diss_vib_rates.json exists with ZPE-corrected rates per element pair.
Phase 6 of permeation_run.py runs without requiring manual DH_DISS_EV/DH_ENTRY_EV.
pipeline.ipynb Cell 3 generates pipeline_run.py; opening it shows Part 1 + Part 3 started in parallel then Part 2 after both complete.
Previous plan content (already implemented)
Parts 1 and 3 follow the same architecture:

The notebook is a thin script generator + local analyser. It writes a standalone Python orchestrator (*_run.py) and a SLURM wrapper (*_run.sh), then has Phase 4 analysis cells that run after the cluster jobs complete.
The workflow module holds all the heavy logic: generate_*_scripts(), generate_*_sh(), and analysis functions (load_*, plot_*, qc_*).
Currently Part 2 has three execution notebooks (subsurface_neb_calculation.ipynb, tst_calculation.ipynb, kmc_calculation.ipynb) and five physics modules, but no script-generator notebook and no workflow orchestration module. This plan builds those two missing artefacts so Part 2 matches the architecture of Parts 1 and 3.

What to Build
1. models/permeation_workflow.py
Mirrors models/diffusivity_workflow.py and models/neb_workflow.py in structure.

Section A — Script generation
def generate_permeation_scripts(
    work_dir, relaxed_slab_path, surface_sites_json, phase2_h_dir,
    sub_neb_dir, vib_dir, results_dir,
    temperatures, p_vals_pa, a0_m, l_m,
    d0_m2s, e_d_ev,                       # from diffusivity_arrhenius.json
    dh_diss_ev, dh_entry_ev,              # from NEB delta_E (user fills)
    nx, ny, seed, kmc_max_steps,
    gpu_slurm_cfg, neb_slurm_cfg,
    vib_slurm_cfg,
    n_images, spring_const, neb_ftol,
    out_py,
) -> str
Uses identical f-string header + raw r-string body pattern as diffusivity_workflow.py.
Header injects all config literals; body is the fixed orchestrator logic.
Returns path to written permeation_run.py.

def generate_permeation_sh(
    orch_job_name, orch_partition, orch_cpus_per_task, orch_mem, orch_time,
    orch_openmpi_ver, orch_cuda_version, orch_conda_env, orch_ld_paths,
    work_dir, out_py, out_sh,
) -> str
Writes SLURM wrapper. Returns path to permeation_run.sh.

Section B — Analysis functions (Phase 4, run locally)
Function	Reads	Returns
load_barrier_summary(sub_neb_dir)	hopa_jobs.json, hopb_jobs.json, neb_barrier.txt per job	pd.DataFrame (sid, hop, E_abs, E_des, delta_E, converged)
load_rate_summary(results_dir, temperatures)	rate_dict_T{T}K.json	pd.DataFrame (label, T_K, k_forward, k_reverse, Ea_zpe, nu)
load_kmc_sweeps(results_dir, temperatures)	permeation_sweep_T{T}K.json	dict[T → sweep_dict]
load_permeability_results(results_dir, temperatures)	permeability_T{T}K.json	dict[T → perm_dict]
plot_barrier_overview(df, sub_neb_dir)	barrier DataFrame	bar chart Ea distribution (Hop A/B), saves barriers_overview.png
plot_mep_overlay(sub_neb_dir)	NEB path files	MEP overlay per hop, saves mep_hopA.png, mep_hopB.png
plot_kmc_sieverts(results_dir, temperatures)	sweep JSONs	J vs √P for each T, saves sieverts_check.png
plot_permeability_vs_T(results_dir, temperatures)	permeability JSONs	Φ(T) linear + Arrhenius (all 3 options), saves permeability_vs_T.png
plot_arrhenius_S0(results_dir)	solubility_arrhenius_kmc.json	Multi-T ln(S) vs 1/T fit overlay, saves solubility_arrhenius.png
plot_bottleneck(results_dir, temperatures)	sweep + Sieverts JSONs	R² vs T bar chart (1 = bulk-limited, <0.98 = surface-limited), saves bottleneck.png
2. calculation/permeation.ipynb
Thin notebook — no heavy computation inline. Mirrors neb_calculation.ipynb and diffusivity.ipynb.

Script-generation cells
Cell	Purpose
Cell 1	import os, sys, json + sys.path setup
Cell 2	All config vars (paths, temperatures, pressures, lattice, KMC params, SLURM configs)
Cell 3	Call generate_permeation_scripts(...) → writes permeation_run.py
Cell 4	Call generate_permeation_sh(...) + submit_slurm_job(dry_run=True)
Phase 4 — Local analysis cells (run after cluster jobs)
Phase	Cell function called
4a. Barrier overview	load_barrier_summary(), plot_barrier_overview()
4b. MEP overlay	plot_mep_overlay()
4c. TST rate summary	load_rate_summary(), rate vs T plot
4d. KMC Sieverts check	load_kmc_sweeps(), plot_kmc_sieverts()
4e. Φ(T) plot	load_permeability_results(), plot_permeability_vs_T()
4f. Arrhenius S₀	plot_arrhenius_S0()
4g. Bottleneck diagnosis	plot_bottleneck()
Generated permeation_run.py — Phases and Execution Order
Phase 1a — Hop A FS-min  (GPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 1b — Hop A NEB     (CPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 2a — Hop B FS-min  (GPU SLURM array, parallel; IS = Hop A relaxed sub1)
    ↓ wait_for_jobs
Phase 2b — Hop B NEB     (CPU SLURM array, parallel per IS site)
    ↓ wait_for_jobs
Phase 3  — Vibrations    (CPU SLURM array; IS + TS for both hops, all parallel)
    ↓ wait_for_jobs
Phase 4  — TST rates     (local inside orchestrator; fast; one rate_dict_T{T}K.json per T)
Phase 5  — KMC sweeps    (local inside orchestrator; sweep_pressure() at each T)
Phase 6  — Permeability  (local; all 3 S₀ options; Arrhenius fit; save all JSONs)
Phases 1–3 use submit_slurm_job() + wait_for_jobs() (same pattern as neb_run.py).
Phases 4–6 run inside the orchestrator process (no additional SLURM jobs needed for the default grid size NX=NY=20).

Reused Functions (no reimplementation needed)
Function	Module
orchestrate_hopa_neb(...)	models/neb_subsurface.py
orchestrate_hopb_neb(...)	models/neb_subsurface.py
collect_is_ts_paths(...)	models/vibrations.py
orchestrate_vibrations(...)	models/vibrations.py
collect_neb_results(...)	models/tst_rates.py
split_vib_results(...)	models/tst_rates.py
build_rate_dict(...)	models/tst_rates.py
rates_to_json(...)	models/tst_rates.py
sweep_pressure(...)	models/permeation.py
fit_solubility_from_kmc(...)	models/permeation.py
arrhenius_diffusivity(...)	models/permeation.py
permeability(...), richardson_flux(...)	models/permeation.py
submit_slurm_job(...), wait_for_jobs(...)	models/create_slurm.py
parse_barrier_file(...)	models/parsers.py
Inputs Required (from Parts 1 and 3)
Variable	Source
RELAXED_SLAB_PATH	Part 1 — calculation/slabs/slab_relaxed.lammps
SURFACE_SITES_JSON	Part 1 — calculation/slabs/surface_sites.json
PHASE2_H_DIR	Part 1 — calculation/adsorption/h_atom/ (dedup IS .lammps files)
D0_M2S, E_D_EV	Part 3 — results/diffusivity_arrhenius.json (loaded in Cell 2 if present)
DH_DISS_EV, DH_ENTRY_EV	User fills from NEB delta_E values after cluster run
Key Config Variables (Cell 2 of notebook)
# Paths
WORK_DIR            = os.path.join(BASE_DIR, 'calculation')
RELAXED_SLAB_PATH   = os.path.join(WORK_DIR, 'slabs', 'slab_relaxed.lammps')
SURFACE_SITES_JSON  = os.path.join(WORK_DIR, 'slabs', 'surface_sites.json')
PHASE2_H_DIR        = os.path.join(WORK_DIR, 'adsorption', 'h_atom')
SUB_NEB_DIR         = os.path.join(WORK_DIR, 'neb_subsurface')
VIB_DIR             = os.path.join(WORK_DIR, 'vibrations')
RESULTS_DIR         = os.path.join(WORK_DIR, 'results')

# Physical parameters
TEMPERATURES   = [500, 600, 700, 800, 900]   # K
P_VALS_PA      = [1e3, 1e4, 1e5, 5e5, 1e6]  # Pa
A0_M           = 3.52e-10   # m
L_M            = 1e-3       # m

# Diffusivity — loaded from Part 3 output if available
D0_M2S = 1.5e-7    # placeholder
E_D_EV = 0.40      # placeholder

# NEB thermodynamics — user fills after NEB jobs complete
DH_DISS_EV  = None   # eV (dissociation NEB delta_E)
DH_ENTRY_EV = None   # eV (Hop A delta_E)

# KMC
NX, NY      = 20, 20
SEED        = 42
KMC_MAX_STEPS = 500_000

# SLURM
GPU_SLURM_CFG = dict(SLURM_DEFAULTS, partition='multigpu', time='04:00:00')
NEB_SLURM_CFG = dict(SLURM_DEFAULTS, partition='short', gpu=None, cpus_per_task=16, time='12:00:00')
VIB_SLURM_CFG = dict(SLURM_DEFAULTS, partition='short', gpu=None, cpus_per_task=8,  time='06:00:00')

# NEB
N_IMAGES    = N_REPLICAS
SPRING_CONST = SPRING_CONST
NEB_FTOL    = NEB_FTOL
Verification
Script generation: Run Cells 1–4 with dry_run=True → permeation_run.py and permeation_run.sh are written, no SLURM jobs submitted. Check that the injected config matches the notebook variables.
Orchestrator structure: Open permeation_run.py and verify 6 phases are present and functions are correctly called.
Phase 4 analysis cells: Run each analysis cell with mock JSON files (or real output from a dry-run test at one temperature) → plots saved without error.
End-to-end: After a cluster run at one temperature (T=700K), run all Phase 4 cells — barrier DataFrame has at least one row, Sieverts R² prints, Φ(T) plot saves.
Files to Create
File	Action
models/permeation_workflow.py	New file — ~500 lines
calculation/permeation.ipynb	New notebook — Cells 1–4 (script gen) + Phase 4 analysis cells
The three existing notebooks (subsurface_neb_calculation.ipynb, tst_calculation.ipynb, kmc_calculation.ipynb) remain valid for interactive step-by-step use. permeation.ipynb is the canonical automated entry point.
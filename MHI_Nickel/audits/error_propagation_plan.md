# Error / uncertainty conventions and propagation plan

Status: **convention agreed; propagation NOT yet implemented on the permeation
side.** This doc is the reference for how uncertainties are represented and the
plan for making error handling consistent across Parts 2 and 3.

---

## 1. Representation convention

Different quantities take different error representations, dictated by how they
are combined:

| quantity type | examples | represent as | why |
|---|---|---|---|
| **energies** (in an exponent, combined by **addition**) | `E_D`, `ΔH_diss`, `ΔH_HopA`, `ΔH_sol`, `E_Φ` | **absolute σ** (same units, eV) | additive → absolute errors add in quadrature; symmetric; never crosses a forbidden bound |
| **prefactors** (log-normal, positive-definite, combined by **multiplication**) | `D0`, `S0`, `Φ0` | **fractional σ** (`σ/x`, dimensionless) or a multiplicative ×/÷ factor | multiplicative → fractional errors add in quadrature; an absolute ± can go **negative** (unphysical) |

**Why prefactors must NOT be reported as absolute ±.** They are fit in log
space (`D0 = exp(intercept)`) and span orders of magnitude. Example (Ni Part 3):
`D0 = 7.73e-7`, `σ_D0 = 7.81e-7` → `D0 − σ = −0.07e-7 < 0`, a negative
diffusivity prefactor. The honest statement is fractional: `σ_D0/D0 ≈ 101 %`,
i.e. **D0 known to ×/÷ e¹ ≈ 2.7**, a *geometric* interval `[D0/2.7, D0×2.7]` —
not a symmetric `±`.

**R² is not an uncertainty.** The `r2` / `r2_S` fields are goodness-of-fit /
curvature flags (a per-environment `S(T)` is a *sum* of Arrhenius terms, so
`R² < 1` is physical curvature, not error). Never read R² as an error bar.

**JSON field naming (to adopt):**
- energies → `*_err_eV` (absolute).
- prefactors → `*_rel_err` (fractional) and optionally `*_factor` (the ×/÷
  multiplier `exp(σ_ln)`).
- keep `r2` separate and labelled as fit quality, not error.

---

## 2. Propagation rules

Standard first-order (Gaussian) propagation:

| operation | rule |
|---|---|
| scale `Z = a·X` | `σ_Z = |a|·σ_X` (absolute) |
| sum `Z = X + Y` | `σ_Z = √(σ_X² + σ_Y²)` (absolute) |
| product `Z = X·Y` | `σ_Z/Z = √((σ_X/X)² + (σ_Y/Y)²)` (fractional) |
| exponential `Z = exp(−E/kT)` | `σ_Z/Z = σ_E/(k_B T)` (fractional from an **absolute** energy error, **amplified by 1/k_BT**) |

The chain we need, end to end:

```
ΔH_sol(env) = ½·ΔH_diss + ΔH_HopA(env)
    σ_ΔHsol = √( (½·σ_ΔHdiss)² + σ_ΔHopA² )                    [absolute, eV]

S(T) = S0 · exp(−ΔH_sol/kT)
    σ_S/S = √( (σ_S0/S0)² + (σ_ΔHsol/kT)² )                    [fractional]

Φ0 = D0 · S0
    σ_Φ0/Φ0 = √( (σ_D0/D0)² + (σ_S0/S0)² )                     [fractional]

E_Φ = E_D + ΔH_sol
    σ_EΦ = √( σ_ED² + σ_ΔHsol² )                               [absolute, eV]

Φ(T) = Φ0 · exp(−E_Φ/kT)
    σ_Φ/Φ = √( (σ_Φ0/Φ0)² + (σ_EΦ/kT)² )                       [fractional]

J = Φ · (√P_high − √P_low) / L        (P, L treated as exact)
    σ_J/J = σ_Φ/Φ                                              [fractional]
```

**Dominant term.** The `σ_EΦ/(k_BT)` factor amplifies the energy error at low T:
Ni's `E_D_err = 0.10 eV` alone gives `σ_lnΦ ≈ 0.10/0.0345 ≈ 2.9` at 400 K → a
factor **≈ 18×** on Φ. So the permeability/flux uncertainty is **dominated by
E_D**, and no amount of solubility precision fixes that — a production-grade D
(not the current validation-grade old-MSD extrapolation) is the real lever.

---

## 3. Current state (audit)

**Done correctly — the diffusivity side (Part 3, `diffusivity_post_processing.py`):**
- per-T `σ_D = (MSD-fit stderr)/6` (the /6 from MSD = 6·D·t in 3D);
- inverse-variance-weighted `ln D` vs `1/T` regression (`_weighted_linregress`,
  proper OLS `σ² = ss_res/(n−2)`);
- propagated: `E_D_err = σ_slope·k_B` (absolute), `D0_err = D0·σ_intercept`
  (i.e. `σ_ln D0` — fractional in disguise);
- reported in `diffusivity_arrhenius.json` + error-bar plots.

**Missing — the solubility / permeability side (Part 2, Phase 6, `permeation.py`):**
- `fit_arrhenius` (permeation) is unweighted `polyfit` and returns only `r2` —
  **no standard errors** on S0 or ΔH_sol.
- `permeability_arrhenius` returns only `{Phi0, E_phi_eV}` — **no propagation**;
  the `D0_err`/`E_D_err` that Part 3 computed are **silently dropped**.
- `S(T)` routes give point values; the NEB pathway-to-pathway spread that feeds
  `ΔH_diss` / `ΔH_HopA` is never turned into `σ_ΔHsol`.
- KMC `S_std` is a dispersion over pressure points, not a propagated σ (and the
  KMC-counting route is a demoted diagnostic anyway).

**Name collision:** two functions named `fit_arrhenius` —
`diffusivity_post_processing.fit_arrhenius(T, D, D_err)` (weighted, returns
errors) vs `permeation.fit_arrhenius(T, y)` (unweighted, no errors). A trap.

---

## 4. Implementation plan

### Step A — source uncertainties (inputs)
1. **NEB reaction energies** (`build_dh_sol_by_env`, and the diss mean in the
   template): add the **standard error of the mean** over pathways,
   `σ = std(values)/√N`, for `ΔH_diss` (23 pathways) and `ΔH_HopA(env)` (N per
   env). Absolute (eV). Store `dH_hopA_err_eV`, and a `dH_diss_err_eV`.
2. **Per-env solution enthalpy**: `σ_ΔHsol(env) = √((½·σ_ΔHdiss)² + σ_ΔHopA(env)²)`.
   Add `dH_sol_err_eV` to each env entry of `dH_sol_by_env.json`.
3. **Diffusivity**: thread `D0_err`, `E_D_err` from `diffusivity_arrhenius.json`
   through `resolve_nh_diffusivity` into Phase 6 (currently dropped).
4. **S0 prefactors**:
   - geometric `S0 = 4/(a₀³·N_A)` → `σ_S0/S0 = 3·σ_a0/a0` (≈ 0 when a₀ is fixed;
     nonzero once `lattice_params_vs_T.json` supplies a₀(T) with a fit error).
   - vibrational → fractional spread of `S0` over the FS structures
     (`std/√N` of the per-structure `q_H`).

### Step B — solubility uncertainty
`σ_S(T)/S = √((σ_S0/S0)² + (σ_ΔHsol/kT)²)`, population-weighted consistently
with `solubility_by_environment`. Report `S_rel_err` per T.
- Also report the **analytic** `ΔH_sol_err` (from Step A) as the physics
  uncertainty, kept separate from the fit `r2` (curvature flag). Do **not**
  conflate the two.

### Step C — permeability propagation (`permeability_arrhenius`)
Extend the signature to accept `D0_err`, `E_D_err`, `S0_rel_err`, `dH_sol_err`
and return, using §2:
- `E_phi_err_eV = √(E_D_err² + dH_sol_err²)`  (absolute)
- `Phi0_rel_err = √((D0_err/D0)² + S0_rel_err²)`  (fractional) + `Phi0_factor = exp(Phi0_rel_err)`
- keep `r2_S` as the (separate) fit-quality flag.

### Step D — flux uncertainty
Per T: `σ_J/J = √(Phi0_rel_err² + (E_phi_err/kT)²)`. Report `J_rel_err` (and note
it is E_Φ/kT-dominated).

### Step E — unify the fits
Give `permeation.fit_arrhenius` an optional weighted mode + standard-error
output (`Ea_err`, `prefactor_rel_err`), matching the diffusivity version — or
rename one to remove the collision. Prefer a single shared weighted-linregress
helper used by both.

### Step F — outputs, tests, plots
- Add the `*_err_eV` / `*_rel_err` / `*_factor` fields to
  `permeability_T{T}K.json`, `solubility_arrhenius.json`,
  `permeability_arrhenius.json`.
- Unit tests: the propagation identities (sum→absolute quadrature,
  product→fractional quadrature, exp→/kT amplification), and that a prefactor
  error is never reported as a zero-crossing absolute ±.
- Error-bar plots for `S(T)` and `Φ(T)` (mirror the diffusivity plots), with
  Φ shown as a **geometric** band (×/÷ factor), not symmetric.

### Step G — units in every result payload
**Audit finding:** no output JSON carries a units block; units are conveyed
*only* by field-name suffixes, applied inconsistently. Fields that carry units:
`*_eV`, `*_m2s`, `*_Pa`, `_K`, `_m`. Fields with **no** unit annotation:
`S0`, `S`, `Phi`, `J` (`permeability_T{T}K.json`), `Phi0`
(`permeability_arrhenius.json`), `S0` (`solubility_arrhenius.json`), and the
sweep arrays `P_vals`, `sqrt_P_vals`, `J_vals`/`J_sub1_vals`,
`C0_vals`/`C0_sub1_vals`/`C0_sub2_vals`, plus `S_vals`/`S_mean`/`S_std`
(`fit_solubility_from_kmc`). (`theta_vals`, `*_rel_err`, `Phi0_factor`, `w_env`,
`r2` are genuinely dimensionless.)

**Fix (non-breaking):** add a `"units"` metadata block to each result payload
(`sweep_pressure` output, `permeability_T{T}K.json`, `solubility_arrhenius.json`,
`permeability_arrhenius.json`) mapping every value field → its unit string,
**without renaming** existing keys (so plots/consumers keep working)::

    "units": {"S0": "mol H m^-3 Pa^-0.5", "S": "mol H m^-3 Pa^-0.5",
              "Phi": "mol H m^-1 s^-1 Pa^-0.5", "Phi0": "mol H m^-1 s^-1 Pa^-0.5",
              "J": "mol H m^-2 s^-1", "P_vals": "Pa", "sqrt_P_vals": "Pa^0.5",
              "C0_vals": "mol H m^-3", "E_phi_eV": "eV", "dH_sol_eV": "eV",
              "theta_vals": "dimensionless", "S0_rel_err": "dimensionless (fractional)",
              "Phi0_factor": "dimensionless (x/÷ 1σ band)"}

Also add units to the Phase-6 summary prints (the S and Φ0 lines currently print
bare numbers, though the library functions — `lattice_site_S0`, `permeability`,
`richardson_flux` — already print units).

---

## 5. Scope notes
- The energy-error handling on the diffusivity side is already correct and is
  the template to follow.
- Because the permeability uncertainty is **E_D-dominated** (validation-grade
  D), the headline honest statement after this work will still be "Φ known to
  ~1–2 orders over 400–800 K, limited by the extrapolated D," until a
  production diffusivity replaces the old-MSD fit. The point of the propagation
  is to *state that honestly* rather than report bare point values.
- Nothing here changes the physics or the `Φ = D·S`, `E_Φ = E_D + ΔH_sol`
  identities — it only attaches uncertainties to them.

---

## 6. Appendix — complete equation set

All formulas used by the plan. `k` = `k_B` (Boltzmann), `T` = temperature.
Absolute errors are `σ_x`; fractional errors are `σ_x/x`.

### 6.1 Sample statistics (NEB pathway means)
For `N` pathway values `xᵢ` (the 23 `ΔH_diss`, or the `N` per-env `ΔH_HopA`):
```
mean         x̄ = (1/N) Σ xᵢ
sample std   s  = √( Σ(xᵢ − x̄)² / (N − 1) )
SEM          σ_x̄ = s / √N                     [absolute, eV]   <- use this
```
The standard error of the mean `σ_x̄` (not the raw std `s`) is the uncertainty on
`ΔH_diss` and on each `ΔH_HopA(env)`.

### 6.2 (Weighted) linear regression standard errors
Fit `y = a + b·x` over `n` points (used for MSD→D and every Arrhenius fit).
With weights `wᵢ` (unweighted ⇒ `wᵢ = 1`), `Sw = Σwᵢ`, `Sxx = Σwᵢxᵢ²`,
`denom = Sw·Sxx − (Σwᵢxᵢ)²`, residuals `eᵢ = yᵢ − (a + b·xᵢ)`:
```
residual variance   s² = Σ wᵢ eᵢ² / (n − 2)
slope error         σ_b = √( s² · Sw  / denom )
intercept error     σ_a = √( s² · Sxx / denom )
```
Inverse-variance weights for an Arrhenius fit done in **log space** (positive
`y` with absolute error `σ_y`, so `σ_(ln y) = σ_y/y`):
```
wᵢ = 1 / σ_(ln yᵢ)² = ( yᵢ / σ_yᵢ )²
```

### 6.3 MSD → diffusivity
3-D isotropic Einstein relation `MSD = 6·D·t + c`:
```
D   = slope / 6
σ_D = σ_slope / 6                     [absolute, m²/s]   (then Å²/ps → m²/s)
```

### 6.4 Arrhenius parameters
Fit `ln y = ln A − (Ea/k)·(1/T)`  →  slope `m = −Ea/k`, intercept `c = ln A`:
```
Ea = −k·m        →  σ_Ea = k·σ_m                    [absolute, eV]
A  = exp(c)      →  σ_A/A = σ_c  (fractional);  σ_A = A·σ_c
geometric factor g_A = exp(σ_c)  →  A ∈ [A/g_A, A·g_A]
```
Instances: D→(A=D0, Ea=E_D); S→(A=S0, Ea=ΔH_sol); Φ→(A=Φ0, Ea=E_Φ).

### 6.5 General first-order propagation (master formula)
For `f(x₁,…,xₙ)` with independent inputs:
```
σ_f² = Σᵢ (∂f/∂xᵢ)² · σ_xᵢ²
```
Standard cases:
```
Z = a·X            σ_Z = |a|·σ_X
Z = X ± Y          σ_Z = √(σ_X² + σ_Y²)
Z = X·Y  or  X/Y   σ_Z/Z = √((σ_X/X)² + (σ_Y/Y)²)
Z = Xⁿ             σ_Z/Z = |n|·(σ_X/X)
Z = exp(a·X)       σ_Z/Z = |a|·σ_X
Z = ln(X)          σ_Z = σ_X/X
```

### 6.6 The permeation chain — per-step partial derivatives
(the boxed end-to-end chain is in §2; these are the partials it comes from)
```
ΔH_sol = ½ΔH_diss + ΔH_HopA   ∂/∂ΔH_diss = ½ ,  ∂/∂ΔH_HopA = 1
S  = S0·exp(−ΔH_sol/kT)       ∂lnS/∂lnS0 = 1 ,  ∂lnS/∂ΔH_sol = −1/kT
Φ0 = D0·S0                    ∂lnΦ0/∂lnD0 = 1,  ∂lnΦ0/∂lnS0 = 1
E_Φ = E_D + ΔH_sol            ∂/∂E_D = 1     ,  ∂/∂ΔH_sol = 1
Φ  = Φ0·exp(−E_Φ/kT)          ∂lnΦ/∂lnΦ0 = 1 ,  ∂lnΦ/∂E_Φ = −1/kT
J  = Φ·(√P_hi − √P_lo)/L      ∂lnJ/∂lnΦ = 1    (P, L exact ⇒ σ_J/J = σ_Φ/Φ)
```
If the pressures carry error, add `(∂J/∂P)²σ_P²` per endpoint with
`∂J/∂P = Φ/(2√P·L)`.

### 6.7 Per-environment Boltzmann-sum solubility error
`S(T) = S0·B(T)`, `B(T) = Σ_env w_env·exp(−ΔH_env/kT)`. Define the fractional
Boltzmann contribution `f_env = w_env·exp(−ΔH_env/kT) / B` (so `Σ f_env = 1`):
```
∂B/∂ΔH_env = −(1/kT)·w_env·exp(−ΔH_env/kT)
σ_B/B  = (1/kT)·√( Σ_env f_env² · σ_(ΔH_env)² )
σ_S/S  = √( (σ_S0/S0)² + (σ_B/B)² )
```
Because `Σ f_env² ≤ 1`, the Boltzmann average *reduces* the effective ΔH
uncertainty relative to one environment — except when a single environment
dominates (`f → 1`), where it reduces to the single-env limit `σ_B/B → σ_ΔH/kT`
used in §2.

### 6.8 Representation conversions
```
fractional → absolute   σ_x = x·(σ_x/x)
absolute → fractional   σ_x/x
geometric (×/÷) factor  g = exp(σ_x/x)  →  x ∈ [x/g, x·g]   (D0, S0, Φ0)
exp amplification       Z = exp(−E/kT)  →  σ_Z/Z = σ_E/(kT)
```

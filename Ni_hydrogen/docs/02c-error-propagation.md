# Part II-C — Error propagation, as built

**Purpose.** [Part II §10](02-theory.md) states how an uncertainty *should* be
represented, and [Part II-B §4](02b-aggregation.md) states what each
aggregation rung *should* do to one. This document states what the code
**actually does**, stage by stage, with the file and line that does it.

Where the two differ, the divergence is marked and carried to the
[decision register](#12-decision-register) at the end. Nothing here is a
proposal; it is a record to review.

Equations are tagged `(E…)` to keep them distinct from the `(T…)` numbering in
Part II.

---

## Contents

| § | stage | what it produces |
|---|---|---|
| [1](#1-the-chain) | — | the chain at a glance |
| [2](#2-stage-a--σ_d-at-one-temperature) | A | $\sigma_D$ at a single $T$ |
| [3](#3-stage-b--dt--d_0-e_d) | B | $\sigma_{D_0}$, $\sigma_{E_D}$ |
| [4](#4-stage-c--per-environment-δh_sol) | C | $\sigma_{\Delta H}$ per environment |
| [5](#5-stage-d--averaging-over-environments) | D | $\sigma_S/S$ at a single $T$ |
| [6](#6-stage-e--st--s_0-δh_sol) | E | $\sigma_{S_0}$, $\sigma_{\Delta H_\text{sol}}$ |
| [7](#7-stage-f--φ--d·s) | F | $\sigma_{\Phi_0}$, $\sigma_{E_\Phi}$, $\sigma_J(T)$ |
| [8](#8-the-scale-invariance-property) | — | **the single most consequential fact** |
| [9](#9-zeros-nans-and-silent-substitutions) | — | where a $\sigma$ becomes 0 |
| [10](#10-realised-magnitudes-ni) | — | what actually dominates |
| [11](#11-two-fitters-two-behaviours) | — | an inconsistency between stages B and E |
| [12](#12-decision-register) | — | items to keep or change |

---

## 1. The chain

```
MSD(t)  ──A──►  D(T), σ_D  ──B──►  D₀, σ_D₀   E_D, σ_E_D ──┐
                                                            │
NEB energies ──► ΔE per pathway ──C──► ΔH_env, σ_env        │
                        │                                   ├──F──►  Φ₀, σ_Φ₀
                        └──D──► S(T), σ_S  ──E──►  S₀, σ_S₀ │        E_Φ, σ_E_Φ
                                                  ΔH, σ_ΔH ─┘        σ_J(T)
```

Two independent branches meet once, at F. Each branch ends in a regression
(B and E), and **both regressions share the property described in §8**.

---

## 2. Stage A — $\sigma_D$ at one temperature

`fit_diffusivity` — [`diffusivity_post_processing.py:151-205`](../models/diffusivity_post_processing.py#L151-L205)

A single ordinary least-squares line is fitted to MSD against time over the
middle of the trace (default window $0.2$–$0.8$ of the span, masking the
ballistic rise and the noisy tail), then

$$D = \frac{\text{slope}}{6}\,c, \qquad \sigma_D = \frac{\text{stderr}}{6}\,c \tag{E1}$$

with $c$ the Å²/ps → m²/s conversion. `stderr` is `scipy.stats.linregress`'s
standard error of the slope.

**What it measures.** How well a straight line fits those MSD points.

**What it does not measure.** Any ensemble spread. `compute_msd`
([`:100-144`](../models/diffusivity_post_processing.py#L100-L144)) averages over
time origins *and* over H atoms into one scalar per lag
(`msd_arr[lag-1] = float(np.mean(sq))`, [`:140`](../models/diffusivity_post_processing.py#L140)),
so the spread across origins and across walkers is discarded before any
uncertainty is formed. There is no block averaging, bootstrap, jackknife or
walker-ensemble code anywhere in `models/` or `calculation/`.

> [!IMPORTANT]
> Successive MSD lags share most of their data and are strongly
> autocorrelated. `linregress` assumes independent residuals, so (E1) is
> structurally an **underestimate**, and it is the only uncertainty the entire
> diffusivity branch is built on. The code applies no autocorrelation
> correction.

Key names for the same quantity: `sigma_D` (in-memory dict and the text
table), `D_err_arr` (JSON), `D_err_m2s` (CSV).

The half-vs-half MSD slope ratio (`CONVERGED_BAND = (0.75, 1.25)`,
[`msd_plot.py:65`](../models/plots/msd_plot.py#L65)) is a **pass/fail verdict
only**. It never enters any $\sigma$; its sole consumer is run exclusion.

---

## 3. Stage B — $D(T) \to D_0,\ E_D$

`fit_arrhenius` → `_weighted_linregress` —
[`diffusivity_post_processing.py:336-430`](../models/diffusivity_post_processing.py#L336-L430)

Weighted least squares of $\ln D$ against $1/T$, with inverse-variance weights
in log space:

$$w_i = \left(\frac{D_i}{\sigma_{D_i}}\right)^2 = \frac{1}{\sigma_{\ln D_i}^2} \tag{E2}$$

which is the correct weight, since $\sigma_{\ln D} = \sigma_D / D$. Parameter
errors come from a $\chi^2$-rescaled covariance:

$$s^2 = \frac{\text{ss}_\text{res}}{\max(n-2,\,1)}, \qquad
  \sigma_{E_D} = k_B \sqrt{\frac{s^2 \sum w}{\Delta}}, \qquad
  \sigma_{D_0} = D_0 \sqrt{\frac{s^2 \sum w x^2}{\Delta}} \tag{E3}$$

with $\Delta = \sum w \sum w x^2 - (\sum w x)^2$.

Three properties of (E3) worth recording:

- $\sigma_{D_0} = D_0\,\sigma_{\ln D_0}$ is a **first-order symmetric**
  approximation to a log-normal quantity. For Ni 1H this gives
  $\sigma_{D_0} \approx 3 D_0$, so the lower error bar is clipped
  (`LOG_ERR_FLOOR = 0.99`,
  [`arrhenius_params_plot.py:54`](../models/plots/arrhenius_params_plot.py#L54)).
  Stage F does report a proper multiplicative band; this stage does not.
- The **slope–intercept correlation is discarded.** $\sigma_{D_0}$ and
  $\sigma_{E_D}$ are propagated as if independent, though a least-squares line
  always correlates them.
- Degrees of freedom are $n-2$ **clamped to a floor of 1**, so a two-point fit
  returns a finite, meaningless $\sigma$ rather than infinity. The fit is
  admitted for $n \ge 2$.

A temperature whose $\sigma_D = 0$ is silently dropped from the fit
([`:610-611`](../models/diffusivity_post_processing.py#L610-L611)).

$R^2$ is weighted and computed about the weighted mean.

> [!NOTE]
> $\sigma_{D_0}$ is stored **absolute**, in m²/s, under the key `D0_err`.
> Two legacy converters write the same absolute value under `D0_err_m2s`
> instead. Only one reader in the codebase tries both spellings
> ([`permeation.py:986`](../models/permeation.py#L986)); it carries an in-code
> postmortem, because reading only `D0_err_m2s` once silently returned `None`
> and kept $\sigma_{D_0}$ out of $\Phi_0$ entirely.

---

## 4. Stage C — per-environment $\Delta H_\text{sol}$

`build_dh_sol_by_env` — [`permeation.py:442-510`](../models/permeation.py#L442-L510)

For each Hop A pathway, a reaction energy $\Delta E = E_a^\text{zpe} - E_d^\text{zpe}$
is formed. Pathways are grouped by destination environment, and within each
environment:

$$\overline{\Delta H}_\text{HopA} = \frac1n \sum_i \Delta E_i, \qquad
  \sigma_\text{HopA} = \frac{s}{\sqrt n}\ (n \ge 2), \qquad 0 \ (n < 2) \tag{E4}$$

$$\Delta H_\text{sol} = \tfrac12 \Delta H_\text{diss} + \overline{\Delta H}_\text{HopA},
  \qquad
  \sigma_{\Delta H} = \sqrt{\left(\tfrac12\sigma_\text{diss}\right)^2 + \sigma_\text{HopA}^2} \tag{E5}$$

Absolute, in eV. $\sigma_\text{diss}$ is itself the standard error of the mean
over converged dissociation pathways
([`permeation_workflow.py:504`](../models/permeation_workflow.py#L504)).

The weight $w_\text{env}$ is a **pathway-count fraction**
($n_\text{env}/n_\text{total}$) and carries no uncertainty of its own.

> [!IMPORTANT]
> **No per-pathway uncertainty exists anywhere upstream.** The rate artifacts
> written by `write_hop_vib_rates`
> ([`tst_rates.py:686-728`](../models/tst_rates.py#L686-L728)) carry
> `Ea_zpe, Ed_zpe, Ea_raw, Ed_raw, nu, nu_reverse, zpe_source` and **no error
> field**. So the only uncertainty available at this stage is the *scatter
> between pathways*. A single-pathway environment reports $\sigma = 0$, meaning
> unmeasurable — see Part II §10.2.

---

## 5. Stage D — averaging over environments

`solubility_by_environment` — [`permeation.py:513-551`](../models/permeation.py#L513-L551)

$$S(T) = S_0 \sum_\text{env} w_\text{env}\, e^{-\Delta H_\text{env}/k_BT}
  \;\equiv\; S_0 B(T) \tag{E6}$$

No single effective $\Delta H$ is formed here; the Boltzmann sum is kept intact
per temperature. An effective $\Delta H$ appears only later, as the *slope* of
the stage-E fit.

`solubility_env_rel_err` — [`permeation.py:648-687`](../models/permeation.py#L648-L687)

$$f_\text{env} = \frac{w_\text{env} e^{-\Delta H_\text{env}/k_BT}}{B}, \qquad
  \frac{\sigma_B}{B} = \frac{1}{k_BT}\sqrt{\sum_\text{env} f_\text{env}^2 \sigma_\text{env}^2} \tag{E7}$$

$$\frac{\sigma_S}{S} = \sqrt{\left(\frac{\sigma_{S_0}}{S_0}\right)^2 + \left(\frac{\sigma_B}{B}\right)^2} \tag{E8}$$

Because $\sum f^2 \le 1$, the Boltzmann average can only **reduce** the
effective enthalpy error relative to a single environment; the single-
environment limit is $\sigma_B/B \to \sigma_{\Delta H}/k_BT$. Both limits are
pinned by unit tests.

> [!WARNING]
> (E7) is computed correctly and then **functionally discarded**. Its only
> consumers are the per-temperature JSON field `S_rel_err` and the fit weights
> at stage E — and by §8 those weights cannot affect the reported magnitude.
> The environment spread therefore contributes **no absolute information** to
> $\Phi$.

---

## 6. Stage E — $S(T) \to S_0,\ \Delta H_\text{sol}$

`fit_arrhenius` — [`permeation.py:746-813`](../models/permeation.py#L746-L813)

Structurally the same weighted least squares as stage B, with the same
inverse-variance weights (E2) and the same $\chi^2$ rescaling:

$$s^2 = \frac{\text{ss}_\text{res}}{n-2}, \qquad
  \sigma_{\Delta H} = k_B\sqrt{\frac{s^2\sum w}{\Delta}}, \qquad
  \frac{\sigma_{S_0}}{S_0} = \sqrt{\frac{s^2 \sum w x^2}{\Delta}} \tag{E9}$$

Outputs: `dH_sol_err_eV` (absolute, eV) and `S0_rel_err` (fractional).

Two behaviours specific to this stage:

- **Weighting is all-or-nothing.** `_yerr` is used only if *every* temperature
  has a finite positive $\sigma_S$
  ([`permeation_workflow.py:752`](../models/permeation_workflow.py#L752)); one
  missing value silently demotes the whole route to unweighted.
- **`detailed_balance` is permanently unweighted** — its $\sigma$ list is
  structurally `None` ([`:741`](../models/permeation_workflow.py#L741)), and its
  per-temperature payload carries no `S_rel_err` key at all.

The three routes supply $\sigma_{S_0}$ at the per-temperature stage very
differently:

| route | per-$T$ $\sigma_{S_0}$ input | source |
|---|---|---|
| `geometric` | **hard-zeroed**, `S0_rel_err=0.0` | comment: "a0 fixed → S0 exact" |
| `vibrational` | SEM over FS structures / mean | fractional |
| `detailed_balance` | **none at any stage** | no $S_0$ is ever formed |

The geometric zero traces to source: `get_lattice_parameter_from_dump`
([`structure.py:156-203`](../models/structure.py#L156-L203)) averages the last
50 NPT frames and never returns their spread, so $\sigma_{a_0}$ is discarded
where it is measured. `lattice_params_vs_T.json` carries two keys,
`temperatures` and `a0_m`, and no $\sigma$.

---

## 7. Stage F — $\Phi = D\cdot S$

`permeability_arrhenius` — [`permeation.py:816-863`](../models/permeation.py#L816-L863)

$$\Phi_0 = D_0 S_0, \qquad E_\Phi = E_D + \Delta H_\text{sol} \tag{E10}$$

$$\frac{\sigma_{\Phi_0}}{\Phi_0} = \sqrt{\left(\frac{\sigma_{D_0}}{D_0}\right)^2 + \left(\frac{\sigma_{S_0}}{S_0}\right)^2},
  \qquad
  \sigma_{E_\Phi} = \sqrt{\sigma_{E_D}^2 + \sigma_{\Delta H}^2} \tag{E11}$$

This is exactly (T29) of Part II — a product combines fractional errors, a sum
combines absolute ones. The prefactor is reported as a multiplicative band,

$$f = \exp\!\left(\frac{\sigma_{\Phi_0}}{\Phi_0}\right), \qquad
  \Phi_0 \in \left[\Phi_0/f,\ \Phi_0 f\right] \tag{E12}$$

never as a symmetric $\pm$. Per-temperature flux error
([`permeation_workflow.py:787-789`](../models/permeation_workflow.py#L787-L789)):

$$\frac{\sigma_J}{J}(T) = \sqrt{\left(\frac{\sigma_{\Phi_0}}{\Phi_0}\right)^2 + \left(\frac{\sigma_{E_\Phi}}{k_BT}\right)^2} \tag{E13}$$

Since $J = \Phi(\sqrt{P_\text{hi}} - \sqrt{P_\text{lo}})/L$, pressures and
thickness are treated as **exact**: $\sigma_J/J = \sigma_\Phi/\Phi$ identically.
No operating-condition uncertainty exists anywhere in the pipeline.

Input conversion at the boundary: $\sigma_{D_0}$ arrives **absolute** (m²/s) and
is divided by $D_0$ to become fractional
([`permeation_workflow.py:556`](../models/permeation_workflow.py#L556));
$\sigma_{E_D}$ and $\sigma_{\Delta H}$ stay absolute in eV.

> [!NOTE]
> `J_rel_err_by_T` is keyed over the **full requested temperature list**, not
> over the temperatures that survived the fit. A route fitted on 3 of 5
> temperatures still emits 5 flux-error entries.

---

## 8. The scale-invariance property

This is the single fact that most changes how the reported numbers should be
read, and it applies to **both** regressions (B and E).

Let every input uncertainty be scaled, $\sigma_y \to c\,\sigma_y$. Then
$w \to w/c^2$, so each weighted sum scales by $1/c^2$, $\Delta \to \Delta/c^4$,
and $\text{ss}_\text{res} \to \text{ss}_\text{res}/c^2$, hence
$s^2 \to s^2/c^2$. The products that appear in (E3) and (E9) are therefore

$$\frac{s^2 \sum w}{\Delta} \ \longrightarrow\ \frac{(s^2/c^2)(\sum w/c^2)}{\Delta/c^4} = \frac{s^2\sum w}{\Delta}$$

**exactly unchanged.** The same holds for the intercept term.

> [!IMPORTANT]
> The per-temperature $\sigma$ set only the **relative weighting** between
> temperatures. Their magnitude cancels identically. What the reported
> $\sigma_{D_0}$, $\sigma_{E_D}$, $\sigma_{S_0}$ and $\sigma_{\Delta H}$ actually
> measure is **scatter of the points about the Arrhenius line** — not sampling
> quality, not MSD statistics, not environment spread.
>
> This is `curve_fit(..., absolute_sigma=False)` semantics. It is a defensible
> choice, but it means the careful work at stages A, C and D affects the
> answer only through point weighting, and a quantity with no uncertainty
> input at all can still be assigned a small $\sigma$.

The clearest demonstration is in §10: `detailed_balance`, which has no
uncertainty input at any stage, reports the **tightest** $\sigma_{\Delta H}$ of
the three routes, because its $S(T)$ happens to be the most log-linear.

### Reproducing this

Measured on Ni 10H data, scaling every input $\sigma$ by $10^{-2}$ and $10^{2}$:

```
Stage B (diffusivity)                         Stage E (permeation)
sigma x0.01  E_D=0.39729659  s_E_D=0.01418294   dH=0.14523906  s_dH=0.00448445
sigma x1     E_D=0.39729659  s_E_D=0.01418294   dH=0.14523906  s_dH=0.00448445
sigma x100   E_D=0.39729659  s_E_D=0.01418294   dH=0.14523906  s_dH=0.00448445
```

Bit-identical across four orders of magnitude. Changing the *relative* pattern
instead — inflating only the 400 K uncertainty twentyfold — does move the
answer, from $E_D = 0.397297$ to $0.356452$ eV. Both fitters behave the same
way.

```python
from models.diffusivity_post_processing import fit_arrhenius
import numpy as np
T = np.array([400., 600., 800., 1000., 1200.])
D = np.array([2.870e-12, 1.977e-10, 1.115e-09, 3.604e-09, 5.713e-09])
for c in (0.01, 1.0, 100.0):
    print(fit_arrhenius(T, D, D * 0.05 * c)[:2])   # (E_D, sigma_E_D)
```

---

## 9. Zeros, NaNs and silent substitutions

Every place an uncertainty becomes zero or is invented, as built.

| # | where | behaviour |
|---|---|---|
| 1 | [`permeation_workflow.py:781-782`](../models/permeation_workflow.py#L781-L782) | `fit_arrhenius` returns **NaN** for $\sigma$ when $n \le 2$, meaning "cannot estimate". These lines convert NaN → **0.0**, meaning "exact". Nothing records the substitution. |
| 2 | [`permeation_workflow.py:644`](../models/permeation_workflow.py#L644) | geometric $\sigma_{S_0}$ hard-zeroed; root cause is $\sigma_{a_0}$ discarded in `structure.py:200` |
| 3 | [`permeation_workflow.py:482`](../models/permeation_workflow.py#L482) | `_DH_DISS_ERR = 0.0`, overwritten **only** on the auto-extract path. A hand-set `DH_DISS_EV` — a path the code advertises — leaves it 0.0 and deletes the $\tfrac12\sigma_\text{diss}$ term from every environment. |
| 4 | [`permeation.py:684`](../models/permeation.py#L684) | `d.get('dH_sol_err_eV', 0.0)` — an environment missing its $\sigma$ contributes zero variance while keeping full weight in $B$ |
| 5 | [`permeation.py:443`](../models/permeation.py#L443), [`:818-819`](../models/permeation.py#L818-L819) | API defaults are `0.0` for every error argument; a caller that omits them gets a zero-error answer indistinguishable from an exact one |
| 6 | [`permeation_workflow.py:554-556`](../models/permeation_workflow.py#L554-L556) | `or 0.0` on $\sigma_{D_0}$, $\sigma_{E_D}$. No `isfinite` guard, so a NaN survives and poisons the quadrature instead |
| 7 | [`permeation_workflow.py:752`](../models/permeation_workflow.py#L752) | one missing $\sigma_S$ discards weighting for **all** temperatures of that route |
| 8 | [`permeation_workflow.py:1190-1191`](../models/permeation_workflow.py#L1190-L1191) | missing/NaN errors plot a **zero-width** band — visually identical to a perfectly determined permeability |
| 9 | [`permeation_workflow.py:488,499`](../models/permeation_workflow.py#L488), [`tst_rates.py:494`](../models/tst_rates.py#L494), [`neb_workflow.py:2943-2945`](../models/neb_workflow.py#L2943-L2945) | a missing reaction energy enters a mean as **0.0 eV** rather than being skipped, biasing both the mean and the SEM |
| 10 | [`environment_plot.py:158-184`](../models/plots/environment_plot.py#L158-L184) | `dH_sol_err_eV` is present in the JSON being read and is never drawn — the bars have no `xerr` |
| 11 | [`solubility_plot.py:277-278`](../models/plots/solubility_plot.py#L277-L278) | a falsy `S0_rel_err` prints no `±` at all, so a hard-zeroed route renders like an exactly known one. Note the asymmetry: the very next line tests `if fit.dH_err is not None`, so a zero *enthalpy* error prints `±0.0000` while a zero *prefactor* error prints nothing |

The one consumer that refuses to invent a zero is the CSV exporter
([`export_results.py:70-75`](../models/plots/export_results.py#L70-L75)):
missing errors export as **blank cells**.

---

## 10. Realised magnitudes (Ni)

Geometric route, the three analysed loadings (1H excluded):

| loading | $\sigma_{D_0}/D_0$ | $\sigma_{E_D}$ | $\sigma_{S_0}/S_0$ | $\sigma_{\Delta H}$ | $\sigma_{\Phi_0}/\Phi_0$ | $\sigma_{E_\Phi}$ | $\times/\div$ |
|---|---|---|---|---|---|---|---|
| 3H | 0.2671 | 0.0193 | 0.0833 | 0.0062 | 0.2798 | 0.0203 | 1.32 |
| 5H | 0.3457 | 0.0221 | 0.0833 | 0.0062 | 0.3556 | 0.0229 | 1.43 |
| 10H | 0.1747 | 0.0120 | 0.0833 | 0.0062 | 0.1935 | 0.0135 | 1.21 |

The solubility columns are identical across loadings because $S$ is
loading-independent by construction — it comes from `dH_sol_by_env.json`, not
from the MD box. Only $D$ varies with loading.

Share of variance, Ni 10H:

| quantity | diffusivity term | solubility term |
|---|---|---|
| $\sigma_{\Phi_0}/\Phi_0$ | $\sigma_{D_0}$ — **81.5 %** | $\sigma_{S_0}$ — 18.5 % |
| $\sigma_{E_\Phi}$ | $\sigma_{E_D}$ — **79.1 %** | $\sigma_{\Delta H}$ — 20.9 % |

So $\Phi$ is diffusivity-dominated, but **not** overwhelmingly: solubility
carries about a fifth of the variance. (On the excluded 1H run, where
$\sigma_{D_0}/D_0 \approx 1.03$, diffusivity reaches ~99 % — a figure that
should not be quoted for the analysed loadings.)

The three routes' enthalpy errors, Ni 10H:

| route | $\sigma_{\Delta H}$ (eV) | $r^2$ of the $S(T)$ fit | uncertainty input |
|---|---|---|---|
| geometric | 0.006159 | 0.99506 | $\sigma_{S_0}$ zeroed; env spread (weights only) |
| vibrational | 0.009398 | 0.97870 | SEM over FS structures (weights only) |
| `detailed_balance` | **0.000496** | 0.99999 | **none at any stage** |

> [!WARNING]
> The route with no uncertainty input reports the smallest uncertainty. This
> follows directly from §8: the reported $\sigma$ is a measure of Arrhenius
> linearity, and `detailed_balance` is the most linear. It is a diagnostic
> cross-check, not a headline result — but its $\sigma$ does not say so.

---

## 11. Two fitters, two behaviours

Stages B and E implement the same algebra in two places, and they differ:

| | stage B (`diffusivity_post_processing`) | stage E (`permeation`) |
|---|---|---|
| dof | $\max(n-2,\,1)$ — **clamped** | $n-2$, and **NaN** if $n \le 2$ |
| $n = 2$ | finite, meaningless $\sigma$ | $\sigma = $ NaN (honest) |
| prefactor $\sigma$ | absolute, $D_0\sigma_{\ln D_0}$ | fractional, $\sigma_{\ln S_0}$ |
| $r^2$ when $\text{ss}_\text{tot}=0$ | NaN | 1.0 |

Stage E's NaN is then converted to 0.0 downstream (§9 item 1), so the honest
behaviour is discarded at the boundary.

A related inconsistency in reporting: the CSV exporter emits an explicit
degrees-of-freedom warning for diffusivity fits — "a $n$-point Arrhenius fit
has $\max(0, n-2)$ degrees of freedom, so its $R^2$ is arithmetic, not
evidence" — using the *unclamped* convention, which contradicts the clamp
actually applied in (E3). **No equivalent warning exists for the solubility
fits.**

---

## 12. Decision register

Each item is independent. Mark each **keep** or **change**.

| # | item | as built | if left alone |
|---|---|---|---|
| D1 | $\sigma_D$ source | OLS stderr of one MSD slope; no ensemble statistics | $\sigma_D$ understates, because MSD lags are autocorrelated |
| D2 | Regression $\sigma$ semantics (§8) | `absolute_sigma=False`; input $\sigma$ magnitudes cancel | reported $\sigma$ means "scatter about the line", not sampling quality |
| D3 | dof clamp in stage B | $\max(n-2,1)$ | 2-point fits report a finite $\sigma$ that means nothing |
| D4 | Slope–intercept correlation | discarded | $\sigma_{\Phi_0}$ and $\sigma_{E_\Phi}$ are not independent, but are combined as if they were |
| D5 | $\sigma_{a_0}$ | discarded in `structure.py:200`; geometric $\sigma_{S_0}$ hard-zeroed | a measurable thermal uncertainty is reported as exact |
| D6 | NaN → 0.0 at the permeability boundary | converts "cannot estimate" to "exact" | a 2-point route publishes a confident $\Phi$ and a zero-width band |
| D7 | `detailed_balance` $\sigma$ | no input, tightest output | the diagnostic route looks like the best-determined result |
| D8 | Per-pathway NEB $\sigma$ | does not exist | environment $\sigma$ can only ever be between-pathway scatter |
| D9 | All-or-nothing fit weighting | one missing $\sigma_S$ unweights the route | silent, and unrecorded in the output |
| D10 | Zero-substitution for missing energies | missing $\Delta E$ enters means as 0.0 eV | biases both the mean and its SEM toward zero |
| D11 | Error bars not drawn | env plot has no `xerr`; zeroed routes print no `±` | the figures look more certain than the JSON is |
| D12 | Operating conditions | $P$ and $L$ exact | no pressure/thickness uncertainty in $\sigma_J$ |
| D13 | Two fitters (§11) | divergent dof and $r^2$ conventions | the same fit behaves differently on the two branches |

---

*Cross-references: [Part II §10](02-theory.md) for the intended representation,
[Part II-B §4](02b-aggregation.md) for what each aggregation rung is meant to do
to an uncertainty, [stage 9](03-stages/09-solubility.md),
[stage 10](03-stages/10-bulk-diffusivity.md) and
[stage 11](03-stages/11-permeability-assembly.md) for the surrounding physics.*

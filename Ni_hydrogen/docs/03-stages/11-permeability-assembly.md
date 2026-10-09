# Stage 11 — Permeability assembly

## Purpose

Combine the two transport properties into the quantity the whole workflow
exists to produce: $\Phi = DS$, reported as the Arrhenius pair $\Phi_0$ and
$E_\Phi$, and as a flux through a membrane of given thickness.

This is the only stage where the surface chain and the bulk chain meet.

## At a glance

| | |
|---|---|
| **Inputs** | $S(T)$ per route from [Stage 9](09-solubility.md); $D_0, E_D$ per loading from [Stage 10](10-bulk-diffusivity.md) |
| **Outputs** | `permeability_T{T}K.json` per loading per temperature; `permeability_arrhenius.json` per loading |
| **Code** | [`models/permeation.py`](../../models/permeation.py), [`models/permeation_workflow.py`](../../models/permeation_workflow.py) |
| **Theory** | [Part II §9](../02-theory.md#9-permeability) |
| **Aggregation** | rungs [A11, A12](../02b-aggregation.md#1-the-ladder) |

## Concepts

**Route.** Inherited from [Stage 9](09-solubility.md): each solubility
construction produces its own $\Phi$. They are carried separately to the end.

**Headline loading.** The one loading whose $\Phi$ is reported as the material's
result. Chosen, not averaged.

## Procedure

For each loading, then for each temperature within it:

1. **Resolve the diffusivity for this loading.** A missing or non-finite fit
   means the concentration is **skipped entirely** — no permeability is computed
   for it and the reason is recorded. Nothing is substituted.

2. **Evaluate $D$ at this temperature** from the fitted parameters (T22). It is
   *not* re-measured here — see [Design choices](#why-d-is-evaluated-not-measured).

3. **Take the lattice parameter at this temperature** from the thermal-expansion
   table where it exists, so the geometric prefactor tracks expansion.

4. **Evaluate $S$** by each route (T14), with its saturating counterpart.

5. **Form $\Phi = DS$** (T24) per route, and the flux (T26) at the configured
   feed and permeate pressures.

6. **Record the validity flags**: whether the loading is dilute, and whether the
   solubility sits below the site-density ceiling.

After the temperature loop, per loading:

7. **Compute the Arrhenius pair analytically** from (T25) — $\Phi_0 = D_0 S_0$
   and $E_\Phi = E_D + \Delta H_{\mathrm{sol}}$ — and propagate uncertainties by
   (T29): absolute in quadrature for the energies, fractional in quadrature for
   the prefactors.

8. **Fit $\ln\Phi$ against $1/T$ as a cross-check**, not as the primary result.

Finally, across loadings:

9. **Select the headline loading.** Not an average.

## Design choices

### Why the Arrhenius pair is computed, not fitted

Substituting two Arrhenius laws into a product gives a third exactly:

$$\Phi = D_0 S_0 \exp\!\left(-\frac{E_D + \Delta H_{\mathrm{sol}}}{k_B T}\right)$$

so $\Phi_0$ and $E_\Phi$ follow from the inputs with no fitting at all. Fitting
$\ln\Phi$ instead would introduce regression error into a quantity that is known
in closed form.

The fit is still performed, as an **independent check**: if the analytic pair
and the fitted pair disagree, something upstream is inconsistent. Its $R^2$
appears in the output for that purpose and is not an uncertainty.

### Why $D$ is evaluated, not measured

Step 2 computes $D$ from the fitted $D_0$ and $E_D$ rather than from a
trajectory at that temperature. The permeation temperatures and the molecular
dynamics temperatures are **independent lists**, and the former may interpolate
or even extrapolate the latter.

> [!IMPORTANT]
> This couples the two stages more tightly than it appears. Changing which
> points enter the diffusivity fit — excluding a temperature, dropping a
> non-positive $D$ — changes $D_0$ and $E_D$, and therefore moves $\Phi$ at
> **every** permeation temperature, including ones the removed point did not
> touch. A permeation result is only as stable as the fit beneath it.

### Why routes are never merged

Each solubility route produces its own $\Phi_0$ and $E_\Phi$, and they are
carried to the end and reported side by side. The geometric and vibrational
routes differ by orders of magnitude for the reason given in
[Stage 9](09-solubility.md#why-two-prefactor-routes-are-reported-rather-than-one):
they count different things.

The rate-based construction is also reported, and is labelled a **diagnostic**
in the output payload rather than a solubility. Its averaging is not the same
operation as the Boltzmann-weighted sum, and by Jensen's inequality
[(A-2)](../02b-aggregation.md#22-arithmetic-mean-a3-a4-a6) the two cannot agree
except when every barrier is identical. Agreement is reassuring; disagreement
measures the artefact, not the physics.

### Why loadings are selected rather than averaged

Averaging $\Phi$ over loadings would combine different physical systems. The
loadings differ in hydrogen–hydrogen interaction and in available site fraction,
so their spread is not measurement noise about a common value.

Selection has its own cost: the choice is a judgement, and the reported
uncertainty is that of the chosen loading alone, carrying no information about
how much the answer would change under a different choice. The spread across
loadings is worth reporting alongside the headline for exactly this reason.

### Why pressure appears in two unrelated places

Two pressures exist in this stage and they are not interchangeable.

| pressure | where | what it is |
|---|---|---|
| reference, fixed at $1\ \mathrm{Pa}$ | inside both $S_0$ routes | an SI normalisation; the half-power units are *per* $\sqrt{1\ \mathrm{Pa}}$ |
| operating feed and permeate | only in the flux (T26) | a condition of the membrane, chosen by the caller |

> [!IMPORTANT]
> Changing the reference would move the vibrational route by
> $\sqrt{P_{\mathrm{ref}}}$ while leaving the geometric route untouched, silently
> putting the two on different footings — invariant
> [(I8)](../02-theory.md#11-invariants). The operating pressure is a separate
> quantity and affects only the flux, never $\Phi$ itself.

## Checks and failure modes

| symptom | meaning | what to do |
|---|---|---|
| a loading absent from the output | its diffusivity fit was missing or non-finite | complete [Stage 10](10-bulk-diffusivity.md) for it; nothing was fabricated |
| no loading produced anything | the solution enthalpies were unavailable | see [Stage 9](09-solubility.md); the gate is upstream |
| analytic and fitted Arrhenius pairs disagree | an inconsistency upstream, not a fitting problem | compare $E_D + \Delta H_{\mathrm{sol}}$ against the fitted $E_\Phi$ directly |
| $E_\Phi$ varies across loadings | expected — all of the variation is $E_D$'s, since $S$ is loading-independent | report the spread; it bounds the selection in step 9 |
| the dilute flag is false | the loading is not in the regime (T24)–(T26) assume | state it; it cannot be fixed by choosing another loading ([Stage 10](10-bulk-diffusivity.md#the-tension-that-cannot-be-designed-away)) |
| occupancy near or above the ceiling | the dilute expression has been pushed past saturation | use the saturating counterpart |
| the rate-based $\Phi$ differs greatly from the others | the averaging artefact, as expected | quote the energy-based routes |
| $\Phi_0$ reported as $x \pm \sigma$ with a negative lower bound | a log-normal prefactor written as a symmetric interval | report a fraction or $\times/\div$ band |

> [!IMPORTANT]
> A permeability can be arithmetically perfect and still meaningless. Every
> caveat from Stages 7 to 10 — an unresolved saddle, a singleton environment
> with zero recorded uncertainty, a two-point diffusivity fit — propagates here
> silently, because none of them makes (T24) fail. The validity flags and the
> per-stage censuses exist so those conditions travel with the number instead of
> being lost at the last multiplication.

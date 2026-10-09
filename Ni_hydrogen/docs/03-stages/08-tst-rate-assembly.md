# Stage 8 — Rate assembly

## Purpose

Combine the barriers from [Stage 6](06-neb.md) with the frequencies from
[Stage 7](07-vibrations-phva.md) into rate constants: a zero-point-corrected
barrier and an attempt frequency in each direction, for every pathway, at a
given temperature.

This is also where **state quality is judged**. A pathway arrives here as a set
of files; it leaves here either as a rate with a recorded quality verdict, or
not at all.

## At a glance

| | |
|---|---|
| **Inputs** | barrier results per pathway; `vib_frequencies.json` for IS, TS and — where available — FS |
| **Outputs** | one rate dictionary per pathway, written as JSON |
| **Code** | [`models/tst_rates.py`](../../models/tst_rates.py) |
| **Runs on** | anywhere — this is arithmetic over stored numbers, seconds of work |
| **Theory** | [Part II §4–§5](../02-theory.md#4-vibrational-analysis) |

## Concepts

**Significance threshold.** A frequency below which an imaginary mode is treated
as finite-difference noise rather than a real instability. Default
$50\ \mathrm{cm^{-1}}$. Needed because
[Stage 7 step 5](07-vibrations-phva.md#procedure) records near-zero eigenvalues
as imaginary.

**State-quality census.** Per pathway, whether the initial state looks like a
minimum and the transition state like a first-order saddle, judged against that
threshold. Recorded as two booleans alongside the raw and significant mode
counts.

**Zero-point source.** Which state supplied the reverse correction — the final
state, as it should, or the initial state as a fallback. Recorded, never
silent.

## Procedure

For each pathway, in this order:

1. **Require IS and TS frequencies.** Missing either, the pathway is skipped and
   named in the skip list. No placeholder is substituted.

2. **Census the imaginary modes** against the significance threshold. Warn if
   the initial state has any significant imaginary mode (expected: none), or if
   the transition state does not have exactly one (expected: one). A warning
   does **not** discard the pathway — the verdict is recorded and passed on.

3. **Compare degrees of freedom.** Total modes, real plus imaginary, for each
   state. Equal totals mean the same atoms were displaced. Unequal totals mean
   they were not, and no prefactor can be formed — see
   [Design choices](#why-the-degrees-of-freedom-are-compared-before-anything-else).

4. **Load the final state if present.** Warn if it carries any imaginary mode.
   If it is absent or unreadable, fall back to the initial state for the reverse
   direction and set the source marker to the fallback value.

5. **Forward attempt frequency** from (T8), initial over transition state. On
   failure the behaviour depends on whether a forward prefactor is required:
   - *required* (hops): skip the pathway.
   - *not required* (dissociation): keep the pathway with a null forward
     prefactor, and record it. The forward dissociation rate is a dimensionless
     sticking probability, not a transition-state rate, so a missing prefactor
     costs nothing there.

6. **Reverse attempt frequency** from (T8) with the final state in place of the
   initial. On failure, fall back to the forward value and downgrade the source
   marker.

7. **Zero-point corrections**, taking **every** real mode:
   - forward against the initial state, (T6a);
   - reverse against the **final** state, (T6b) — or the initial state if step 4
     fell back.

8. **Rates** from (T10) and (T10a), one per direction, null where its prefactor
   is null.

9. **Assemble the payload**: both barriers raw and corrected, both prefactors,
   the source marker, the census verdicts, the raw and significant imaginary
   counts, and the threshold used.

## Design choices

### Why the degrees of freedom are compared before anything else

Equation (T8) is only a frequency if the initial state carries exactly one more
real mode than the transition state, invariant
[(I1)](../02-theory.md#11-invariants). The underlying requirement is stronger:
the two states must describe **the same subsystem**, because each state's mode
count is three times its own displaced-atom count.

Comparing *total* modes — real plus imaginary — tests exactly that, and is
independent of how many modes happen to be imaginary at either state. Comparing
only the real lists would conflate two different problems: different atom sets,
and a saddle that failed to produce its imaginary mode.

> [!IMPORTANT]
> A prefactor built from mismatched subsystems is not obviously wrong. It is a
> finite, plausible-looking number that can be off by many orders of magnitude.
> This is why the mismatch is detected rather than tolerated, and why
> [Stage 7](07-vibrations-phva.md#why-the-dissociation-variant-takes-its-mobile-set-from-the-caller)
> takes its mobile set from the caller for multi-atom pathways.

### Why a failed census does not discard the pathway

An unsound state and a missing file are different problems and get different
treatment. A missing file means nothing can be computed, so the pathway is
skipped. A questionable census means the numbers exist but may not mean what
they claim — which is a judgement for whoever reads them, in the light of what
else is available.

So the verdict travels **with** the rate, as `is_minimum` and `ts_saddle`,
alongside both the significant and the raw imaginary counts. Where those two
counts disagree, the state had artefacts that the threshold absorbed.

> [!IMPORTANT]
> Filtering on the census is therefore a **downstream** decision, and where it
> happens matters: an average taken before the filter silently includes
> pathways already known to be unsound. See
> [the aggregation ladder §3.2](../02b-aggregation.md#32-averaging-survives-quality-filtering).

### Why the threshold exists at all

Without it, an initial state carrying a single imaginary mode of a few
$\mathrm{cm^{-1}}$ — far below even its own lowest real mode — would be
condemned as "not a minimum". Such modes are the frozen-slab artefact of
[Part II §4.1](../02-theory.md#41-what-is-computed), not instabilities.
Counting imaginary modes with no magnitude threshold discards sound structures
over finite-difference noise.

> [!IMPORTANT]
> The threshold is a noise floor, not a physical boundary. Where a material's
> genuine low modes overlap the artefact range, **no threshold separates them**,
> and the census cannot resolve that material's saddles. That is a real limit of
> the method, not a tuning problem: the correct response is to report the
> pathways as unresolved, not to move the threshold until the answer looks
> better.

### Why the reverse direction is built from the final state

$E_{\mathrm{des}} = E_{\mathrm{TS}} - E_{\mathrm{FS}}$, so its zero-point
correction is $\mathrm{ZPE_{TS}} - \mathrm{ZPE_{FS}}$ and its attempt frequency
is built from the final state's modes.

Using the initial state for both directions is the error this stage is shaped to
avoid, and it is nearly invisible: the correction cancels exactly in
$E_a^{\mathrm{ZPE}} - E_{\mathrm{des}}^{\mathrm{ZPE}}$, so the reaction energy
comes out *uncorrected* while both barriers appear corrected. Nothing looks
wrong. The `zpe_source` field exists so that a fallback can never be mistaken
for the real thing.

### Why zero-point sums take every mode, but the prefactor does not

Two different thresholds, because the two quantities fail differently.

| quantity | form | low-mode behaviour | treatment |
|---|---|---|---|
| zero-point energy (T6) | **sum** | a small mode contributes a small amount | include everything |
| attempt frequency (T8) | **product** | a near-zero mode in the denominator is unbounded | raise to a floor (T7) |

A sum is robust to a bad small term; a product is not. Applying one threshold to
both would either discard real zero-point contributions or leave the prefactor
unbounded.

> [!IMPORTANT]
> Three thresholds act in this stage and they are **not** interchangeable:
> the significance threshold ($50\ \mathrm{cm^{-1}}$, judges imaginary modes),
> the zero-point inclusion threshold ($0\ \mathrm{cm^{-1}}$, includes every real
> mode), and the quasi-harmonic floor ($100\ \mathrm{cm^{-1}}$, raises low real
> modes in the prefactor). A fourth, with the same default as the first, drops
> low modes from the partition function (T9).

### Rejected: rebalancing mode counts after a cut

An earlier approach discarded real modes below a cutoff and then trimmed the
longer list so the counts matched again. This was abandoned. Discarding changes
the counts, so the dimensional invariant (I1) has to be *restored* artificially
rather than *checked* — which means it can no longer detect the mismatched-atoms
failure it exists to catch.

Raising the modes instead (T7) leaves the counts untouched, so the invariant
stays meaningful and the artefact is damped symmetrically in both states. See
[Part II §4.3](../02-theory.md#43-quasi-harmonic-treatment-of-low-modes).

## Checks and failure modes

| symptom | meaning | what to do |
|---|---|---|
| pathway in the skip list, "vib missing" | Stage 7 did not complete for that state | re-run Stage 7 for it; the marker gates the retry |
| warning: IS has significant imaginary modes | the initial geometry is not a minimum | re-relax it; the census records the verdict either way |
| warning: TS has none, or several | the band did not resolve one saddle | the barrier does not describe a single process |
| warning: different degrees of freedom | the states displaced different atoms | recompute with a shared mobile set; do not trim the lists |
| `zpe_source` is the fallback value | no usable final-state frequencies | the reverse barrier and prefactor are initial-state-derived; treat reverse quantities as provisional |
| forward prefactor is null | (T8) failed and was not required — a dissociation pathway | expected; the forward rate there is a sticking probability |
| prefactor far outside $10^{12}$–$10^{14}\ \mathrm{s^{-1}}$ | a low mode is dominating the product, or the subsystems differ | check the degrees-of-freedom warning first, then the floor |
| significant and raw imaginary counts disagree | the state had artefacts the threshold absorbed | normal for a partial Hessian; worth noting if the gap is large |

> [!NOTE]
> **Planned:** the four thresholds above are spread across functions and two of
> them still share a parameter name, a survival from when one threshold served
> every purpose. Renaming them to say what each does would remove the most
> likely misreading of this stage.

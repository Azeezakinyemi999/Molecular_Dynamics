# Appendices

- [A. Notation](#a-notation)
- [B. Settings reference](#b-settings-reference)
- [C. Output files](#c-output-files)
- [D. Unit conversions](#d-unit-conversions)
- [E. Checking the equations](#e-checking-the-equations)
- [F. Glossary](#f-glossary)
- [G. References](#g-references)

---

## A. Notation

| symbol | meaning | unit |
|---|---|---|
| $E$ | electronic energy | eV |
| $E_a$ | forward barrier, $E_{\mathrm{TS}} - E_{\mathrm{IS}}$ | eV |
| $E_{\mathrm{des}}$ | reverse barrier, $E_{\mathrm{TS}} - E_{\mathrm{FS}}$ | eV |
| $\Delta E$ | reaction energy, $E_{\mathrm{FS}} - E_{\mathrm{IS}}$ | eV |
| $\mathrm{ZPE}$ | zero-point energy | eV |
| $\tilde\nu$ | vibrational wavenumber | cm⁻¹ |
| $\tilde\nu_{\mathrm{floor}}$ | quasi-harmonic floor | cm⁻¹ |
| $\nu^*$ | attempt frequency (Vineyard) | s⁻¹ |
| $\nu^*_{\mathrm{rev}}$ | reverse attempt frequency | s⁻¹ |
| $k$ | rate constant | s⁻¹ |
| $q_{\mathrm{vib}}$ | vibrational partition function | — |
| $q^{\mathrm{gas}}$ | gas-phase partition function | — |
| $\Gamma$ | gas strike rate per area | m⁻² s⁻¹ |
| $\theta$ | surface or site occupancy | — |
| $e$ | an interstitial environment | — |
| $w_e$ | population weight of environment $e$ | — |
| $\Delta H_{\mathrm{diss}}$ | dissociation reaction energy, per molecule | eV |
| $\Delta H_{\mathrm{entry}}$ | entry-hop reaction energy | eV |
| $\Delta H_{\mathrm{sol}}$ | solution enthalpy, per atom | eV |
| $S_0$, $S$ | solubility prefactor and solubility | mol m⁻³ Pa^(−½) |
| $\rho_{\mathrm{site}}$ | interstitial site density | mol m⁻³ |
| $D_0$, $D$ | diffusivity prefactor and diffusivity | m² s⁻¹ |
| $E_D$ | diffusion activation energy | eV |
| $\mathrm{MSD}$ | mean-squared displacement | Å² |
| $\tau$ | lag time | ps |
| $r$ | half-vs-half slope ratio | — |
| $\Phi_0$, $\Phi$ | permeability prefactor and permeability | mol m⁻¹ s⁻¹ Pa^(−½) |
| $E_\Phi$ | permeation activation energy | eV |
| $J$ | flux | mol m⁻² s⁻¹ |
| $L$ | membrane thickness | m |
| $a_0$ | lattice parameter | m |
| $P$, $P_{\mathrm{ref}}$ | pressure, reference pressure | Pa |
| $n$ | hydrogen count in the box | — |
| $x_{\mathrm{at\%}}$ | hydrogen content | at.% |
| $\sigma$ | standard uncertainty | as its quantity, or dimensionless if fractional |
| $n_{\mathrm{IS}}$, $n_{\mathrm{TS}}$ | real mode counts | — |

**States.** IS initial, TS transition, FS final.

## B. Settings reference

Values are the shipped defaults. Those marked **cluster** are
environment-specific and must be checked before running.

### Potential and environment

| setting | stage | meaning |
|---|---|---|
| potential file, ASE and engine forms | all | the model supplying every energy and force |
| model head | all | which trained head of the model is selected |
| base directory | all | **cluster** — root under which everything is written |
| scheduler defaults, partitions, submit limits | all | **cluster** |

### Structure and slab

| setting | value | stage | meaning |
|---|---|---|---|
| minimisation force tolerance | 1e-6 eV/Å | 1, 2 | convergence for structure relaxation |
| minimisation iteration caps | 50000 / 500000 | 1, 2 | iteration and force-evaluation limits |
| freeze fraction | 1/3 | 2 | lower fraction of slab thickness held fixed |
| freeze cutoff | derived | 2 | computed per slab; a configured value exists but is superseded |
| surface relaxation tolerances | 1e-6 eV/Å, 10000/100000 | 2 | as above, for the slab |
| surface heating and equilibration steps | 10000 / 100000 | 2 | thermal relaxation of the surface |
| thermostat coupling | 0.05 ps | 2 | surface equilibration |

### Sites

| setting | value | stage | meaning |
|---|---|---|---|
| bond cutoff | 3.2 Å | 3 | surface-graph edge length |
| top-layer tolerance | 1.8 Å | 3 | height window defining the top layer |
| site-merge tolerance | 0.5 Å | 3 | merges near-degenerate adsorption sites |
| oxide bond cutoff | 2.6 Å | 3 | shorter bonds in an oxide |
| clustering tolerance | 0.75 Å | 4 | merges Voronoi vertices |
| minimum atom distance | 0.5 Å | 4 | discards vertices too close to an atom |
| coordination cutoff | 2.2 Å | 4 | neighbour count for classification |
| layer mode | rank / gaps | 4 | equal-count binning, or gap detection for oxides |

### Band

| setting | value | stage | meaning |
|---|---|---|---|
| intermediate images | 9 | 6 | chain length between endpoints |
| spring constant | 1.0 eV/Å² | 6 | coupling along the band |
| force tolerance | 0.1 eV/Å | 6 | convergence for the climbing phase |
| phase-1 tolerance | 3× the above | 6 | looser, deliberately |
| interpolation | pair-potential | 6 | avoids atom overlap |
| step caps | 5000 / 10000 | 6 | per phase |

### Vibrations and rates

| setting | value | stage | meaning |
|---|---|---|---|
| displacement | 0.01 Å | 7 | finite-difference step |
| metal neighbours per hydrogen | 6 | 7 | mobile cage size; 0 gives hydrogen-only modes |
| quasi-harmonic floor | 100 cm⁻¹ | 8 | modes below are **raised** to it |
| zero-point inclusion | 0 cm⁻¹ | 8 | every real mode contributes |
| imaginary significance | 50 cm⁻¹ | 8 | below this an imaginary mode is noise |
| partition-function cutoff | 50 cm⁻¹ | 9 | modes below are **dropped** |

> [!IMPORTANT]
> The last four are four different thresholds with three different actions.
> Two still share a parameter name. See
> [Part II §4.3](02-theory.md#43-quasi-harmonic-treatment-of-low-modes).

### Dynamics

| setting | value | stage | meaning |
|---|---|---|---|
| timestep | 0.0005 ps | 10 | integration step |
| constant-pressure heating / production | 20000 / 200000 | 10 | steps |
| barostat coupling | 1.0 ps | 10 | |
| box output interval | 100 steps | 10 | lattice parameter sampling |
| fit window | 0.2–0.8 | 10 | fraction of the lag range regressed |
| maximum lag | half the trajectory | 10 | |
| convergence band | 0.75–1.25 | 10 | accepted half-vs-half slope ratio |

### Permeation

| setting | stage | meaning |
|---|---|---|
| temperatures | 8, 9, 11 | where rates, solubility and permeability are evaluated |
| hydrogen loadings | 10, 11 | one diffusivity and one permeability per loading |
| reference pressure | 9 | **fixed at 1 Pa** — an SI normalisation, not a choice |
| feed and permeate pressure | 11 | operating conditions; affect the flux only |
| membrane thickness | 11 | flux only |

## C. Output files

### Per material

| file | contents |
|---|---|
| surface site list | position, type and environment label per site |
| subsurface site list | position, classification, label, layer |
| ranked barriers | per pathway: endpoints, barrier, reverse barrier, reaction energy, convergence flag, final force |
| `vib_frequencies.json` | per state: real and imaginary frequency lists, displaced-atom count, indices, displacement |
| hop rate dictionaries | per pathway: both barriers raw and corrected, both prefactors, environment labels, zero-point source |
| `rate_dict_T{T}K.json` | the above plus forward and reverse rates, reaction energy, quality census |
| `dH_sol_by_env.json` | per environment: solution enthalpy and uncertainty, entry enthalpy, weight, site count |
| `lattice_params_vs_T.json` | lattice parameter against temperature |

### Per material and loading

| file | contents |
|---|---|
| `diffusivity_arrhenius.json` | temperatures, diffusivity and uncertainty per temperature, fitted prefactor and activation energy with errors, goodness of fit |
| `permeability_T{T}K.json` | diffusivity and solubility at that temperature, permeability and flux per route, saturating counterparts, validity flags, units |
| `permeability_arrhenius.json` | per route: prefactor, activation energy, uncertainties, cross-check statistic |
| `solubility_arrhenius.json` | per route: solubility against temperature, prefactor, mean solution enthalpy |

Every payload carries a `units` block listing the unit of each field present.

> [!IMPORTANT]
> A field absent from the units block is absent from the payload. The block is
> built by walking the payload, so it never describes a field that is not there.

## D. Unit conversions

| from | to | factor |
|---|---|---|
| cm⁻¹ | eV | $hc = 1.23984198\times10^{-4}$ |
| cm⁻¹ | s⁻¹ | $c = 2.99792458\times10^{10}$ |
| Å² ps⁻¹ | m² s⁻¹ | $10^{-8}$ |
| Å | m | $10^{-10}$ |
| ps | s | $10^{-12}$ |
| eV | J | $1.602176634\times10^{-19}$ |
| eV/atom | kJ/mol | $96.485$ |

## E. Checking the equations

```bash
~/anaconda3/envs/mace_env/bin/python docs/figures/check_equations.py
```

Each check recomputes a quantity twice — once through the project's own
function, once from the equation as this documentation states it — on the
fictitious system in [`figures/example/toy.py`](figures/example/toy.py). One
`PASS` or `FAIL` line per check, then a count. Exit status is 0 only if all
pass.

| group | covers |
|---|---|
| units | wavenumber to energy; mean-squared displacement to SI; site density |
| zero-point | forward correction; **reverse from the final state**; inclusion of every real mode |
| prefactor | the Vineyard expression; the floor; the floor disabled; the mode-count rule; refusal when it fails |
| partition function | the harmonic product; its lower bound |
| rates | the Arrhenius form |
| diffusion | the Einstein relation; slope recovery; **both argument orders** of the Arrhenius helper; fit round-trip |
| constants | one Boltzmann constant across four modules |
| solubility | both prefactor routes; the per-environment enthalpy; the weighted sum; reduction to Sieverts; weight normalisation; the bound |
| permeability | the product; both Arrhenius identities; the flux |
| uncertainty | absolute and fractional quadrature; the multiplicative band |

Expected output ends with `40 / 40 checks passed`.

> [!IMPORTANT]
> Two of these exist because of bugs found while writing: the reversed
> argument orders of the two Arrhenius helpers, and the Boltzmann constant that
> differed between modules. Both were silent.

## F. Glossary

**Aggregation rung.** One of the twelve points where many quantities become
fewer. See [the ladder](02b-aggregation.md).

**Band.** A chain of images between two endpoints, relaxed onto the
minimum-energy path.

**Climbing image.** The highest image on a band, driven uphill to converge onto
the saddle.

**Degrees of freedom.** For a Hessian, three times the displaced-atom count. For
a fit, the number of points minus the number of parameters.

**Dilute limit.** Hydrogen sparse enough that atoms do not interact. Assumed by
Sieverts' law.

**Environment.** A class of site sharing a neighbour shell; the unit of
averaging and weighting.

**Headline loading.** The one hydrogen loading whose permeability is reported.

**Idempotency marker.** A file written after a stage completes, so a re-run
skips it.

**Imaginary mode.** A vibrational mode with a negative eigenvalue, reported as
a positive magnitude.

**Loading.** The number of hydrogen atoms in the simulation box.

**Partial Hessian.** A Hessian over a chosen subset of atoms, the rest frozen.

**Quasi-harmonic floor.** A frequency below which modes are raised to it rather
than discarded.

**Route.** An independent construction of the solubility prefactor.

**Saddle.** A stationary point with exactly one imaginary mode.

**Sieverts' law.** Dissolved concentration proportional to the square root of
pressure, for a diatomic gas dissolving as atoms.

**State-quality census.** Per-pathway verdicts on whether a minimum is a
minimum and a saddle a saddle.

**Stem.** A material's identity: its input filename without extension.

**Time origin.** A frame from which a displacement is measured.

**Unwrapping.** Removing periodic-boundary jumps from a trajectory before
computing displacement.

## G. References

Cited where the method rests on a published result.

| topic | where it is used |
|---|---|
| Vineyard's many-body rate expression | the attempt frequency (T8), [Stage 8](03-stages/08-tst-rate-assembly.md) |
| Truhlar's quasi-harmonic treatment of low frequencies | the floor (T7); the same convention appears in standard thermochemistry tooling |
| Richardson–Sieverts permeation | the permeability and flux, [Part II §9](02-theory.md#9-permeability) |
| Einstein relation for diffusion | (T20), [Stage 10](03-stages/10-bulk-diffusivity.md) |
| climbing-image nudged elastic band | [Stage 6](03-stages/06-neb.md) |
| image-dependent pair-potential interpolation | the initial band chain, [Stage 6](03-stages/06-neb.md) |
| Hertz–Knudsen gas kinetics | the strike rate (T11) |
| Langmuir adsorption | site saturation (T17) |

> [!NOTE]
> **Planned:** full bibliographic entries. The code cites these by name in
> docstrings; the formal references are not yet collected.

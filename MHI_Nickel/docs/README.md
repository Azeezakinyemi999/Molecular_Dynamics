# Hydrogen permeation — method reference

How this codebase computes hydrogen permeability through a metal: what each
stage does, why it is done that way, and the equations behind it.

> [!NOTE]
> **Scope.** This documents the **method, not results**. It works for any
> material the workflow accepts, and reports no numbers from any real run —
> every worked value and figure comes from a small fictitious example, or is a
> schematic. Numbers that are part of the method, such as defaults, tolerances
> and unit factors, do appear.
>
> Checked against ASE 3.26.0, MACE 0.3.8, ACAT 2.0.1, NumPy 1.26.4,
> SciPy 1.13.1, Python 3.9.23, on 2026-10-09. Re-check the equations with:
>
> ```bash
> ~/anaconda3/envs/mace_env/bin/python docs/figures/check_equations.py
> ```

## Contents

| part | what it holds |
|---|---|
| [Part I — Overview](01-overview.md) | what the workflow answers, how to run it, the stage map, assumptions and limits |
| [Part II — Theory](02-theory.md) | every equation, definition and invariant, with units |
| [Part II-B — The aggregation ladder](02b-aggregation.md) | how many quantities become one, at each of twelve points |
| [Part III — Stages](03-stages/README.md) | one section per stage, eleven in all |
| [Part IV — Practical](04-practical.md) | troubleshooting, adding a material, re-running, reproducibility, code map |
| [Appendices](05-appendices.md) | notation, settings, output files, unit conversions, equation checks, glossary, references |

### Part III at a glance

| # | stage | | # | stage |
|---|---|---|---|---|
| [1](03-stages/01-materials-and-structures.md) | materials and structures | | [7](03-stages/07-vibrations-phva.md) | vibrations |
| [2](03-stages/02-slabs-and-relaxation.md) | slabs and relaxation | | [8](03-stages/08-tst-rate-assembly.md) | rate assembly |
| [3](03-stages/03-surface-site-mapping.md) | surface site mapping | | [9](03-stages/09-solubility.md) | solubility |
| [4](03-stages/04-subsurface-site-mapping.md) | subsurface site mapping | | [10](03-stages/10-bulk-diffusivity.md) | bulk diffusivity |
| [5](03-stages/05-adsorption-and-enumeration.md) | adsorption and enumeration | | [11](03-stages/11-permeability-assembly.md) | permeability assembly |
| [6](03-stages/06-neb.md) | nudged elastic band | | | |

## How to use this document

### Reading order

| if you want to | read |
|---|---|
| understand what the workflow is for | [Part I](01-overview.md), then stop |
| follow the physics end to end | Part I, [Part II](02-theory.md), then stages 9, 10, 11 |
| understand one stage | its section alone — each is self-contained and links out for shared material |
| work out why a number looks wrong | [Part IV §1](04-practical.md#1-troubleshooting), then the stage it came from |
| change a setting | [Appendix B](05-appendices.md#b-settings-reference), then the stage's design-choices section |
| add a material | [Part IV §2](04-practical.md#2-adding-a-new-material) |
| understand why an average was taken that way | [the aggregation ladder](02b-aggregation.md) |

### The shape of a stage section

Every one of the eleven follows the same layout, so a section can be read whole
or skimmed for one part:

| section | contains |
|---|---|
| **Purpose** | what the stage is for |
| **At a glance** | inputs, outputs, code, where it runs |
| **Concepts** | terms introduced here, linked to Part II rather than re-derived |
| **Procedure** | numbered, in the order the code performs them |
| **Design choices** | why it is done this way; what was tried and rejected |
| **Checks and failure modes** | symptom, meaning, what to do |

### Symbols and units

Generic symbols are used in the text; project-specific names appear only in
examples, the code map and the settings reference. Energies are in
electronvolts, frequencies in wavenumbers, and all derived quantities in SI.
Full notation is [Appendix A](05-appendices.md#a-notation).

### The two boxes

> [!IMPORTANT]
> Behaviour that is easy to get wrong, usually because the wrong version does
> not raise an error. These mark the traps.

> [!NOTE]
> **Planned:** something described but not implemented, or a decision taken
> without a recorded justification. These mark the gaps.

## Supporting files

| file | purpose |
|---|---|
| [`figures/example/toy.py`](https://github.com/Azeezakinyemi999/Molecular_Dynamics/blob/main/MHI_Nickel/docs/figures/example/toy.py) | a fictitious material that runs through every stage in seconds, so nothing here needs a real run |
| [`figures/check_equations.py`](https://github.com/Azeezakinyemi999/Molecular_Dynamics/blob/main/MHI_Nickel/docs/figures/check_equations.py) | verifies every equation in Part II against the code; prints `PASS`/`FAIL` per check |
| [`figures/make_figures.py`](https://github.com/Azeezakinyemi999/Molecular_Dynamics/blob/main/MHI_Nickel/docs/figures/make_figures.py) | regenerates all ten drawn figures from that example; the mermaid diagrams live inline in the text |
| [`CHECKPOINT.md`](https://github.com/Azeezakinyemi999/Molecular_Dynamics/blob/main/MHI_Nickel/docs/CHECKPOINT.md) | working state: decisions taken, archaeology, and what remains |

## What this supersedes

Three earlier documents described the pipeline before the physics corrections
of October 2026 and are superseded by this one:

| superseded | now in |
|---|---|
| the pipeline guide | [Part I](01-overview.md) and [Part III](03-stages/README.md) |
| the workflow reference | [Part III](03-stages/README.md) |
| the error-propagation plan | [Part II §10](02-theory.md#10-uncertainty) and [Appendix A](05-appendices.md#a-notation) |

They remain in place as history. The two graph explainers in
`Project2_surface_labeling/` are **not** superseded — their algorithms are
unchanged, and stages 3 and 4 adapt them.

> [!IMPORTANT]
> The superseded documents predate the reverse-barrier, prefactor, mode-census
> and quasi-harmonic corrections. Where they disagree with this reference, this
> one is current.

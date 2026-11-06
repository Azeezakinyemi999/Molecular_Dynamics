# Stage 1 — Materials and structures

## Purpose

Decide **which** materials the workflow processes, classify each one, and build
the atomic cells every later stage consumes.

Small, cheap, and the place where a mistake is least visible: an error here
produces no exception, only a material that quietly never appears in the
results.

## At a glance

| | |
|---|---|
| **Inputs** | structure files in the input directory |
| **Outputs** | absolute paths, a classification per material, and built bulk cells with hydrogen inserted |
| **Code** | [`models/materials.py`](../../models/materials.py), [`models/structure.py`](../../models/structure.py) |
| **Theory** | [Part II §1](../02-theory.md#1-conventions) for naming |

## Concepts

**Stem.** A material's identity throughout the workflow: the input filename
without its extension. Every output directory, marker and artifact is keyed on
it.

**Classification.** One of *pure*, *alloy* or *oxide*, decided from the stem.
It routes slab construction in [Stage 2](02-slabs-and-relaxation.md) and the
surface skip rules below — it no longer has any say in the element table.

**Auto-detected tables.** There is no fixed built-in element/mass table.
Every material's table comes from `build_element_tables_from_structure` (in
`structure.py`): it reads the structure file with ASE and builds the table
from whatever elements are actually present, instead of a material having to
fall inside some pre-typed set. Type IDs are assigned by a fixed rule —
elements alphabetically, hydrogen last — not by the order elements happen to
appear in the file, so re-detecting on a re-saved copy of the same structure
always reproduces the same IDs. For a structure that already carries its own
LAMMPS type numbering (an existing `Masses` section), those exact type IDs
are preserved rather than renumbered, so a file already run once never gets
silently reassigned.

**Skip rule.** A reason a material is excluded from the **surface** half of the
workflow. It does not exclude it from bulk diffusivity.

## Procedure

1. **Read the material list** — one list, in one module.

2. **Classify** each by substring of its stem: *oxide* if the stem says so,
   *alloy* if it names an alloy or a special quasirandom structure, *pure*
   otherwise.

3. **Apply the surface skip rules**, in order:
   1. a named polar-oxide termination;
   2. any pure body-centred-cubic structure, by reading the file.

   The second reads from disk, so it runs last and **tolerates failure**: an
   unreadable structure is *not* skipped, it is left in to fail loudly
   downstream.

4. **Build the bulk cell** and minimise it.

5. **Insert hydrogen** at octahedral interstitial sites, subject to minimum
   separations from the host atoms and from other hydrogens, from a seeded
   random choice.

```mermaid
flowchart TD
    A["structure file"] --> B{"stem contains<br/>'oxide'?"}
    B -->|yes| C1["class: oxide"]
    B -->|no| D{"stem names an alloy<br/>or a quasirandom structure?"}
    D -->|yes| C2["class: alloy"]
    D -->|no| C3["class: pure"]
    C1 --> E{"a named polar<br/>termination?"}
    C2 --> G
    C3 --> F{"reads as<br/>body-centred cubic?"}
    E -->|yes| SKIP["surface work SKIPPED<br/>reason recorded"]
    E -->|no| G
    F -->|yes| SKIP
    F -->|"no, or unreadable"| G["full workflow"]
    SKIP --> H["bulk diffusivity still runs"]

    style SKIP fill:#fdf6e3,stroke:#b8860b
    style H fill:#e8f2e8,stroke:#2e7d32
```

**Figure 1.** Classification and the skip decision. Two things to note: an
**unreadable** structure takes the "no" branch rather than being skipped, so a
corrupt file fails loudly downstream instead of vanishing silently; and a skip
removes only the *surface* work — bulk diffusivity runs for every material.

## Design choices

### Why the material list is a module and not a literal

The list was once written out four times — once in each of three script
generators and once in the master notebook — along with copies of the
classification function and the skip rules.

> [!IMPORTANT]
> Nothing kept the four copies in agreement, and the failure mode was
> **silent**. Adding a material meant four correct edits, and missing one did
> not raise: the part whose list you forgot simply never generated a script for
> that material, and you noticed only when its results were absent, possibly
> weeks later.

Adding a material is now one edit in one place, and a regression test asserts
that no generator keeps a local copy.

### Why the skip rules apply only to the surface half

The skipped cases fail for reasons that are specifically about **building a
surface**: a polar termination needs a per-structure Miller-index and
termination analysis that does not exist yet, and the body-centred-cubic
surface and subsurface mapping is untested.

Bulk diffusivity builds no surface, so none of that applies. Those materials run
through [Stage 10](10-bulk-diffusivity.md) normally, and a reader looking only
at the diffusivity results would never know a skip rule existed.

### Why an unreadable structure is not skipped

The body-centred-cubic test opens the file, and that can fail for reasons having
nothing to do with the crystal structure. Treating a read failure as "skip"
would turn a corrupt or missing file into a silent omission — exactly the class
of failure this stage is built to avoid.

Returning "do not skip" instead lets the material proceed and fail at the point
where the real problem is visible.

### Why hydrogen insertion is seeded

Insertion picks interstitial sites at random subject to separation constraints.
A fixed seed makes the choice reproducible, so re-running a material produces
the same cell and results can be compared across runs rather than differing by
placement.

### Why element tables are detected from the structure, not hand-typed

This project used to hand-type two fixed tables (an 8-type and a 9-type
variant, covering nine elements total) and pick between them by
classification. That worked for the materials on hand at the time but did not
extend to a new alloy the way adding a material does: a hand-typed element
list for a new composition can quietly drift out of sync with the structure
file as it is edited, with no error — just a mislabeled atom type once the
generated scripts run. It also meant the H₂ gas-phase reference energy
(computed once, shared by every material) was carrying several elements that
no material in the project actually has — harmless to MACE-MH-1 (verified in
`calculation/tools/verify_padded_species_mace.py`: an unused declared species
changes nothing about the energy of the atoms that are present), but the kind
of hand-maintained leftover this removes along with the rest.

`build_element_tables_from_structure` removes that hand-typed list entirely.
It derives the element set from the atoms in the file itself, so the table is
reproducible by construction instead of something to keep correct by hand. It
does not replace classification, which still decides *pure*/*alloy*/*oxide*
slab routing and the surface skip rules — it only ever decided the element
table, and now nothing does that by hand.

## Checks and failure modes

| symptom | meaning | what to do |
|---|---|---|
| a material has no results at all | it was never in the list, or it is skipped for the surface half | check the list and the skip reason, which is printed |
| a material has bulk results but no surface ones | a skip rule applies to it | expected; the reason names the issue |
| a material routes to the wrong slab template or skip rule | the classification is wrong for that stem | the classification is substring-based; rename the stem or extend the rule |
| a new alloy's generated scripts have the wrong masses or species labels | its element table wasn't built from its own structure | use `build_element_tables_from_structure` (or `build_element_tables` with an explicit list) and pass the result into the relevant generator — there is no built-in default to fall back on |
| `build_element_tables_from_structure` can't identify an element | the file has no real `Masses` section, or placeholder mass values | use `build_element_tables` with an explicit element list instead |
| hydrogen placement differs between runs | the seed changed | insertion is seeded for exactly this reason |
| insertion fails to place the requested count | the separation constraints cannot be satisfied at that loading | lower the loading or the minimum separations |

> [!NOTE]
> **Planned:** classification is decided by substrings of the filename. It works
> because the naming is disciplined, but a material named outside the convention
> is classified wrongly with no warning. A declared type per entry would remove
> the guesswork.

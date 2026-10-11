# Part III — Stages

One section per stage, in the order the pipeline runs them. Each follows the
same layout, so a stage can be read in isolation or skimmed for one of its parts:

| section | contains |
|---|---|
| **Purpose** | what the stage is for, in one paragraph |
| **At a glance** | inputs, outputs, and the code that implements it |
| **Concepts** | terms introduced here, linked to [Part II](../02-theory.md) rather than re-derived |
| **Procedure** | numbered, in the order the code performs them |
| **Design choices** | why it is done this way; what was tried and rejected |
| **Checks and failure modes** | symptom and meaning, as a table |

| # | stage | file |
|---|---|---|
| 1 | Materials and structures | `01-materials-and-structures.md` |
| 2 | Slabs and relaxation | `02-slabs-and-relaxation.md` |
| 3 | Surface site mapping | `03-surface-site-mapping.md` |
| 4 | Subsurface site mapping | `04-subsurface-site-mapping.md` |
| 5 | Adsorption and pathway enumeration | `05-adsorption-and-enumeration.md` |
| 6 | Nudged elastic band | `06-neb.md` |
| 7 | Vibrations (PHVA) | `07-vibrations-phva.md` |
| 8 | Rate assembly | `08-tst-rate-assembly.md` |
| 9 | Solubility | `09-solubility.md` |
| 10 | Bulk diffusivity | `10-bulk-diffusivity.md` |
| 11 | Permeability assembly | `11-permeability-assembly.md` |

> [!NOTE]
> All eleven sections are written. Outstanding method-level gaps are collected
> in [Part IV §6](../04-practical.md#6-planned-improvements); working state is in
> [`../CHECKPOINT.md`](https://github.com/Azeezakinyemi999/Molecular_Dynamics/blob/main/MHI_Nickel/docs/CHECKPOINT.md).

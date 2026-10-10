# Hydrogen permeation through metal membranes

**Question.** How fast does hydrogen permeate a structural metal membrane?

**Method.** A MACE machine-learned potential drives LAMMPS and ASE to get
barriers (NEB), turns them into rate constants (harmonic TST), and combines
those with a bulk diffusivity into a Richardson–Sieverts permeability
`Φ = D·S`.

**Three Parts**, run as SLURM jobs on the cluster:

| | what it computes | depends on |
|---|---|---|
| **Part 1** — surface NEB | H₂ dissociation on the slab surface | — |
| **Part 2** — permeation | subsurface entry (Hop A/B), TST rates, solubility, Φ(T) | Parts 1 and 3 |
| **Part 3** — bulk diffusivity | D(T) from NVT MD mean-squared displacement | — |

Parts 1 and 3 are independent of each other and can run at the same time.
Part 2 needs both.

---

## 1. Before you run anything

Everything below assumes you are **on the cluster**, in a checkout of this
repo. Nothing here runs usefully on a laptop: the paths in
`models/config.py` are cluster paths, and the work is SLURM jobs.

Open `models/config.py` and **check these match your environment** — they are
currently set for the account this was developed on:

```python
BASE_DIR          = '/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel'
LAMMPS_CMD        = '/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp'
MACE_MODEL_ASE    = '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mh-1.model'
MACE_MODEL_LAMMPS = '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mh-1.model-mliap_lammps.pt'
MACE_HEAD         = 'omat_pbe'

SLURM_DEFAULTS['conda_env'] = '/home/akinyemi.az/miniforge3/envs/mace-lammps'
```

`BASE_DIR` is the important one: it must point at **your** checkout, because
every generated run script bakes it in. The conda env must have MACE, ASE and
PyTorch; LAMMPS must be built with the `mliap` package.

**The one check that works with no data yet:**

```bash
python -m pytest tests/ -q        # expect 1626 passed
```

That exercises the physics and the script generators without touching SLURM,
LAMMPS or MACE. If it passes, the code is sound and anything that fails later
is environment or data.

---

## 2. How to approach this codebase

The single most useful thing to know:

> **`models/` is a library you almost never run. It *generates* standalone
> scripts into `calculation/`, and those are what SLURM runs.**

So `models/permeation_workflow.py` does not compute a permeability. It writes
`calculation/permeation_run_{stem}.py`, a self-contained script with all its
configuration baked in as constants at the top, and *that* computes the
permeability on a compute node.

Two consequences worth internalising:

- **Editing a file in `models/` changes nothing until you regenerate.** The
  generated script still holds the old code.
- A generated script is readable top to bottom. When a run misbehaves, read
  `calculation/permeation_run_{stem}.py` directly — it is the thing that ran.

Every notebook has the same four-step shape:

```
configure  →  generate *_run.py  →  submit it  →  analyse results
  cell 2        cell 3              cell 4        later cells
```

### What lives where

| path | what it is |
|---|---|
| `models/` | the library: physics, graph building, script generators |
| `models/materials.py` | **the list of materials** — add one here, see §4a |
| `models/plots/` | plotting scripts, run after results exist |
| `calculation/*.ipynb` | **the entry points** — one per Part, plus a master |
| `calculation/input_structure/` | input `.lammps` structures |
| `calculation/tools/` | maintenance tools — **not** the normal way in, see below |
| `calculation/tools/legacy/` | superseded; kept for reference |
| `tests/` | 1626 tests |
| `audits/`, `Project2_surface_labeling/` | design docs and audits, see §6 |

**`calculation/tools/` is not how you run the workflow.** Those are patch
tools, used when you have edited something in `models/` and need the generated
scripts rebuilt without stepping through a notebook. The notebooks are the way
in. The tools are there for maintenance:

| tool | when you want it |
|---|---|
| `regenerate_{neb,permeation,diffusivity}_scripts.py` | you edited a workflow in `models/` and want every metal's run script rebuilt |
| `wrap_*_runs_west.py` | write the SLURM wrappers for the generated scripts |
| `check_permeation_maps.py` | check a metal has everything Part 2 needs |
| `backfill_done_markers.py` | reconstruct checkpoint markers for a run that predates them |

### Where results appear

Nothing below exists until you run something.

| you want | look in |
|---|---|
| final Φ(T), S(T), flux | `calculation/results/{stem}_{n}H/permeability_T*K.json` |
| the Arrhenius fits | `calculation/results/{stem}_{n}H/{solubility,permeability}_arrhenius.json` |
| a summary figure | `calculation/results/{stem}_{n}H/permeation_summary.png` |
| diffusivity D₀, E_D | `calculation/results/{stem}_{n}H/diffusivity_arrhenius.json` |
| lattice parameter vs T | `calculation/results/{stem}/lattice_params_vs_T.json` |
| per-environment ΔH_sol | `calculation/results/{stem}/dH_sol_by_env.json` |
| TST rate constants | `calculation/results/{stem}/rate_dict_T*K.json` |
| did the run succeed | `calculation/results/{stem}/permeation_status.json` |
| Hop A/B barriers | `calculation/neb_subsurface/{stem}/hop{a,b}/hop*_ranked.json` |
| dissociation barriers | `calculation/neb/{stem}/ranked_barriers.json` |
| vibrational frequencies | `calculation/vibrations/{stem}/{label}_{IS,TS,FS}/vib_frequencies.json` |
| the slab and its sites | `calculation/slabs/{stem}/` |

`{stem}` is the structure file name without its extension — `Al_supercell`.
`{n}H` is the hydrogen loading.

---

## 3. The pipeline

### Part 1 — surface NEB (`neb_calculation.ipynb`)
Builds the slab, relaxes it, finds surface sites, places H₂, and runs NEB for
dissociation. Produces `ranked_barriers.json` and, in its Phase E, the
dissociation vibrational frequencies.

### Part 3 — bulk diffusivity (`diffusivity.ipynb`)
NPT to get the lattice parameter against temperature, then NVT MD at each
temperature and H loading. Fits the mean-squared displacement to get D(T), and
from that D₀ and E_D. Independent of Part 1 — start both together.

### Part 2 — permeation (`permeation.ipynb`)
Needs Parts 1 and 3 finished. Runs in numbered phases:

| phase | does | where |
|---|---|---|
| 1 | Hop A NEB — surface H* → first subsurface site | SLURM array |
| 2 | Hop B NEB — subsurface 1 → 2 | SLURM array |
| 3 | vibrational frequencies at IS, TS and FS | SLURM array |
| 4 | TST rate constants | local to the job |
| 5 | *retired* | — |
| 6 | solubility and Richardson–Sieverts permeability | local to the job |

**Phase 5 is deliberately absent.** It was a pressure-sweep stage that has been
removed; the number is left empty so Phase 6 keeps the name it carries in
`permeability_T*K.done` markers and throughout the docs. A gap in the numbering
is intentional, not a missing step.

---

## 4. Running it

### 4a. Add a material

1. Put the structure in `calculation/input_structure/`, e.g. `MyAlloy.lammps`.
2. Add one line to `INPUT_STRUCTURE_FILES` in `models/materials.py`:

```python
INPUT_STRUCTURE_FILES = [
    ...
    'MyAlloy.lammps',
]
```

That is the whole registration. `models/materials.py` also decides the
material's class from its name — anything containing `oxide` is an oxide,
anything containing `hastelloy`/`sqs`/`alloy` is an alloy, everything else is
pure — which selects the element and mass tables.

Two materials are deliberately excluded from the **surface** work (Parts 1 and
2) by `skip_surface_reason()`: a pure BCC structure, and the named polar-oxide
termination. Both still run through Part 3, which builds no surface. If your
material is skipped unexpectedly, that function says why.

### 4b. Run everything — `pipeline.ipynb`

The master orchestrator. Run cells 1–2 to configure, then the generator cells,
which write a `*_run.py` per metal for all three Parts plus a `pipeline_run.py`
that sequences them. Keep `dry_run = True` until the generated scripts look
right; it writes files without submitting.

### 4c. Part 1 alone — `neb_calculation.ipynb`
### 4d. Part 3 alone — `diffusivity.ipynb`

Same shape: configure, generate `*_run.py`, generate the `.sh`, submit.

### 4e. Part 2 alone — `permeation.ipynb`

Set `STEM` at the top of the configuration cell:

```python
STEM = 'Al_supercell'
N_H_VALUES = [1, 3, 5, 10]
```

Everything else derives from it. Before running, that metal needs:

- **from Part 1** — `calculation/neb/{stem}/ranked_barriers.json`, a relaxed
  slab under `calculation/slabs/{stem}/`, and the H adsorption results;
- **from Part 3** — `results/{stem}_{n}H/diffusivity_arrhenius.json` for every
  loading in `N_H_VALUES`.

The configuration cell prints which loadings have a diffusivity fit. A loading
without one is **skipped loudly** — no placeholder D₀/E_D is ever substituted,
so a missing fit costs you that loading, never a wrong number.

To check readiness before submitting:

```bash
python calculation/tools/check_permeation_maps.py {stem}
```

### 4f. Is it done?

```bash
squeue -u $USER
cat calculation/results/{stem}/permeation_status.json
```

`permeation_status.json` records which loadings produced a permeability and
which were skipped and why. The run exits non-zero only if *every* loading
failed, so a partial success is a success — check the file, not just the exit
code.

### 4g. Forcing a rerun

The pipeline is checkpointed: finished work is skipped on re-submission. There
are exactly **two** guards, so to redo a stage you delete its marker.

| to redo | delete |
|---|---|
| Phase 4 (TST rates) | `results/{stem}/rate_T*K.done` and `rate_dict_T*K.json` |
| Phase 6 (permeability) | `results/{stem}_*H/permeability_T*K.done` |
| dissociation vibrations | `calculation/neb/{stem}/vibrations_diss/` and `diss_vib_rates.json` |

Everything else — `dH_sol_by_env.json`, `hop*_vib_rates.json`,
`hop*_ranked.json` — is rewritten on every run and needs no deletion.

**Keep the hop vibrations.** `calculation/vibrations/{stem}/` holds the most
expensive output in the pipeline and is rarely invalidated. Delete it only if
the vibration calculation itself changed.

---

## 5. Getting the results out

Everything above stops when the jobs finish. This is how a finished run becomes
figures and tables. None of it touches the cluster — it reads artifacts that are
already on disk and runs in seconds, so it is safe to re-run freely.

### 5a. One command for all of it

```bash
$PY models/plots/postprocess.py --stem Ni_supercell
$PY models/plots/postprocess.py --all          # every material, most complete first
```

This runs the eight plot scripts and then the export, in the right order, and
prints what ran, what was skipped and why. **The pipeline calls it for you** —
`pipeline_run.py` after each metal, and `permeation_run.py` when Part 2 alone
finishes — so you only need the command above to re-do analysis after a change.

Each stage is gated on **its own** inputs. A material with only Part 3 gets its
diffusivity figures and a CSV export; the permeation figures are skipped with
the reason stated, because "Part 2 isn't finished" is a normal state and not an
error. A plot script that fails is reported and stepped over — analysis never
takes down a run that already produced the physics.

The rest of this section is what that command does, in case you want a step on
its own.

### 5b. Plot

Eight scripts under `models/plots/`. They share run discovery, so they all take
the same `--pattern` and `--outdir`.

**Diffusivity side** — needs Part 3 only:

```bash
PY=/home/akinyemi.az/miniforge3/envs/mace-lammps/bin/python   # check this matches your environment
OUT=calculation/results/plots

$PY models/plots/diffusivity_plot.py       --pattern 'Ni_supercell_*' --outdir $OUT
$PY models/plots/msd_plot.py               --pattern 'Ni_supercell_*' --outdir $OUT
$PY models/plots/arrhenius_params_plot.py  --pattern 'Ni_supercell_*' --outdir $OUT
$PY models/plots/thermal_expansion.py      --outdir $OUT
```

**Permeation side** — needs Parts 1, 2 and 3:

```bash
$PY models/plots/permeability_plot.py  --pattern 'Ni_supercell_*H' --outdir $OUT
$PY models/plots/solubility_plot.py    --pattern 'Ni_supercell_*H' --outdir $OUT
$PY models/plots/environment_plot.py   --pattern 'Ni_supercell'    --outdir $OUT
$PY models/plots/neb_mep_plot.py       --pattern 'Ni_supercell'    --outdir $OUT
```

Two patterns, not one: `environment_plot` and `neb_mep_plot` work on the
**material** (`Ni_supercell`), because per-environment enthalpies and barriers
do not depend on H loading. The rest work on **runs** (`Ni_supercell_*H`).

Per-run figures land in `results/{run}/analysis/`, cross-run ones in
`results/plots/`. Pass two or more materials to a pattern and the cross-material
comparison panels appear as well.

Appendix C2 of [`docs/`](docs/05-appendices.md#c2-figure-catalogue) lists every
figure type, what it shows, and what to look at in it.

### 5c. Export

One command collects a finished material into a folder you can hand to someone:

```bash
$PY models/plots/export_results.py --stem Ni_supercell
```

Writes to `calculation/results/export/{stem}/`:

| | |
|---|---|
| `part1_dissociation.csv` | per pathway: sites, barriers both directions, ZPE-corrected values, prefactors, state-quality census |
| `part2_hop_barriers.csv` | Hop A/B per site, with environments |
| `part2_tst_rates.csv` | forward and reverse rates per temperature |
| `part2_solubility_by_env.csv` | ΔH_sol per environment with weights |
| `part3_diffusivity_vs_T.csv` | D(T) per loading, in at.% H |
| `part3_diffusivity_fits.csv` | D₀, E_D, errors, **and how many positive points each fit used** |
| `overall_permeability.csv` | Φ₀, E_Φ per loading per route, each marked `headline` or `diagnostic` |
| `figures/` | every figure belonging to that material |
| `SUMMARY.md` | the above, plus caveats generated from the data |

Read `SUMMARY.md` first. Its caveats are derived, not written by hand, so they
catch things that are easy to miss by eye — how many dissociation pathways have
both a relaxed initial state and a resolved saddle, which hop lost sites to an
unconverged band, and whether any loading sits in the dilute limit.

### 5d. Before quoting any number

Five checks, in the order they bite:

1. **Did the MSD converge?** `msd_plot.py` prints a half-vs-half slope ratio per
   run and temperature. Outside 0.75–1.25 means the trace is not diffusive
   there. **A per-temperature R² above 0.99 does not catch this.**
2. **How many points are behind the fit?** `part3_diffusivity_fits.csv` has a
   `n_positive_D_points` column. Two points is zero degrees of freedom, and the
   R² of 1.0 it reports is arithmetic, not evidence.
3. **Which solubility route?** `geometric` and `vibrational` are the reported
   pair and differ by orders of magnitude by construction — quote both as a
   bracket. `detailed_balance` is a **diagnostic**, not a solubility.
4. **Is the loading dilute?** Richardson–Sieverts assumes it. The permeability
   payload carries a flag; no loading here is both dilute and well sampled, so
   state which compromise the number represents.
5. **Is a quoted σ real?** A standard error of exactly zero means the group had
   one member, i.e. unmeasurable — not precise.

### 5e. Flags that change the answer

| flag | default | effect |
|---|---|---|
| `--include-1h` | off | single-H runs are **excluded by default** — one H atom is one random walker with no ensemble average, and every 1H fit inspected so far is unsound |
| `--mep-reference formation` | `chained` | draws the NEB stages at their true formation energy instead of shifting them to meet, so the H–H interaction at the seam is shown rather than assumed |
| `--pattern` | varies | **check it.** `diffusivity_plot`, `msd_plot` and `arrhenius_params_plot` default to `Al*`; the permeation-side scripts default to `*`. Running one unqualified silently plots the wrong material, or every material at once. |

---

## 6. Conventions that will bite you

**Use the absolute path to Python.** `conda activate` gets killed on the login
node:

```bash
/home/akinyemi.az/miniforge3/envs/mace-lammps/bin/python   # check this matches your environment
```

**Anything importing torch needs `srun`.** The login node kills it (exit 137).
This includes the regenerate tools, because the workflow imports `vibrations`,
which imports MACE:

```bash
srun --partition=short --time=00:12:00 --cpus-per-task=2 --mem=8G \
  /home/akinyemi.az/miniforge3/envs/mace-lammps/bin/python calculation/tools/regenerate_permeation_scripts.py
```

**Results are not in git.** `calculation/results/`, `neb/`, `neb_subsurface/`,
`vibrations/`, `adsorption/` and `slabs/` are all `.gitignore`d — they run to
hundreds of gigabytes. They move by `rsync`, and they are **not backed up by
committing**. Only source is tracked.

**Some vibration jobs fail transiently.** Roughly 7% die with
`Illegal instruction (core dumped)`, intermittently, on nodes that run other
jobs fine. They pass on plain resubmission. For a run with thousands of
vibration jobs, script the retry rather than resubmitting by hand.

**A `.done` marker means "finished", not "correct".** The markers are
checkpoints to avoid recomputation. If the physics changed, delete them (§4g)
or you will keep the old answer.

---

## 7. Where to read more

**Start with [`docs/`](docs/README.md)** — the method reference. It documents
how the codebase works: what each stage does, why it is done that way, and the
equations behind it. Eleven stage sections, every equation with its units, and
a troubleshooting guide organised by symptom.

| document | what it is for |
|---|---|
| [`docs/README.md`](docs/README.md) | index and reading order — go here first |
| [`docs/01-overview.md`](docs/01-overview.md) | what the workflow answers, the stage map, assumptions and limits |
| [`docs/02-theory.md`](docs/02-theory.md) | every equation, definition and invariant, with units |
| [`docs/02b-aggregation.md`](docs/02b-aggregation.md) | how many quantities become one, at each of twelve points |
| [`docs/03-stages/`](docs/03-stages/) | one section per stage, eleven in all |
| [`docs/04-practical.md`](docs/04-practical.md) | troubleshooting, the marker inventory, the code map |
| [`docs/05-appendices.md`](docs/05-appendices.md) | notation, every setting, output-file schemas, the figure catalogue |

Still current, not superseded:

| document | what it is for |
|---|---|
| `Project2_surface_labeling/Project2_Surface_Graph_Explainer (1).md` | surface site identification and labelling, stage by stage. **Despite the name this is the newer, fuller version** — twice the length of the file without the `(1)`. |
| `Project2_surface_labeling/Project2_Subsurface_Graph_Explainer.md` | the subsurface interstitial graph that Hop A/B traverse |
| `Project2_surface_labeling/pipeline_changes_plan.md` | the idempotency/checkpoint design |
| `audits/test_plan.md`, `audits/functional_test_plan.md` | what the tests cover and why |
| `audits/task_*_audit.md` | per-task audits from specific pieces of work |
| `audits/oxide_support_plan.md` | what would be needed to support oxides properly |

> **Superseded — kept for history only.** `PIPELINE_GUIDE.md`,
> `multiscale_permeation_plan.md` and `audits/error_propagation_plan.md`
> describe the pipeline as it stood before the October 2026 physics
> corrections — the reverse barrier and prefactor built from the final state,
> the refusal of a prefactor when the mode counts cannot be a frequency, the
> state-quality census, and the quasi-harmonic floor. Each carries a header
> saying so. Where they disagree with `docs/`, `docs/` is current.

Section numbers in the audits refer to each other; a number left empty is
deliberate, the same convention as Phase 5.

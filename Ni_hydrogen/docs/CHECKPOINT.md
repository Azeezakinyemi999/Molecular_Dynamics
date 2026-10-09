# Documentation checkpoint

Working state for writing `docs/`. Read this first when resuming; it carries
every decision already made so none of it has to be re-litigated.

- **Opened:** 2026-10-08
- **Baseline commit:** `ed8f96d` (main)
- **Status:** outline approved, two code fixes landed, **no documentation written yet**
- **Next action:** build `docs/figures/example/` + `check_equations.py`

---

## 1. Mission

Write a reference that explains **how the hydrogen-permeation codebase works** —
what each stage does, why it is done that way, and the equations behind it.

- **Audience:** the author in six months, and a new student with a physical-chemistry
  background who has not seen this code.
- **It documents the method, not results.**
- **It supersedes** three stale documents (§7).

### Hard constraints

| rule | detail |
|---|---|
| No results | No numbers from real runs **in the body**. Method numbers (defaults, tolerances, unit factors) are fine. |
| Design-choices exception | **Approved:** "Design choices" subsections *may* cite real-run numbers as evidence for a decision. This is how the Truhlar floor is justified in code. |
| Read before writing | Procedures follow the code's real order, conditions and defaults. No describing from memory. |
| Verify every equation | Via `check_equations.py` where a library can confirm it; otherwise derive it and say so. |
| Generic symbols | Project names belong in examples, the code map and the settings reference only. |
| Flag, don't guess | Where code, notes and library docs disagree, list it rather than resolving silently. |
| Mark unimplemented | `> [!NOTE] Planned:` boxes. |

### Style

Plain sentences, one idea per paragraph. LaTeX (`$...$`, `$$...$$`) always with units.
Mermaid for pipeline, decision logic and data flow. Every figure numbered with a caption
saying what to look at. GFM tables, `[!NOTE]` / `[!IMPORTANT]` boxes, relative links to code.

---

## 2. Approved file layout

```
docs/
  CHECKPOINT.md          <- this file
  README.md              index: scope NOTE, contents, how to use, reading order
  01-overview.md         Part I
  02-theory.md           Part II   — every equation in one place
  02b-aggregation.md     the aggregation ladder (cross-cutting; see §5)
  03-stages/
    01-materials-and-structures.md
    02-slabs-and-relaxation.md
    03-surface-site-mapping.md
    04-subsurface-site-mapping.md
    05-adsorption-and-enumeration.md
    06-neb.md
    07-vibrations-phva.md
    08-tst-rate-assembly.md
    09-solubility.md
    10-bulk-diffusivity.md
    11-permeability-assembly.md
  04-practical.md        Part IV: troubleshooting, adapting, reproducibility, code map
  05-appendices.md       A notation · B settings · C output files · D unit conversions
                         · E equation checks · F glossary · G references
  figures/
    example/             fictitious toy cell, runs end-to-end in seconds
    make_figures.py
    check_equations.py
```

Every stage section uses the same layout:
**Purpose · At a glance (inputs, outputs, code) · Concepts · Procedure (numbered, in code
order) · Design choices (why; what was tried and rejected) · Checks and failure modes (table).**

---

## 3. The 12 stages

| # | Stage | Primary code |
|---|---|---|
| 1 | Material registry, input structures, bulk/alloy build | `materials.py`, `structure.py` |
| 2 | Slab construction, surface relaxation, z-freeze cutoff | `structure.py`, `lammps_script.py` |
| 3 | Surface site mapping — layers, ACAT sites, graph | `surface_graph.py` |
| 4 | Subsurface mapping — Voronoi, classification, surface↔sub1↔sub2 | `subsurface_graph.py`, `neb_subsurface.py` |
| 5 | Adsorption and dissociation-pathway enumeration | `neb_workflow.py` phases 1–2b |
| 6 | NEB — dissociation, Hop A, Hop B | `ase_neb.py`, `neb_subsurface.py` |
| 7 | Vibrations (PHVA) — single-H and 2-H variants | `vibrations.py` |
| 8 | TST rate assembly — ZPE, quasi-harmonic, Vineyard, census | `tst_rates.py` |
| 9 | Solubility — ΔH_sol by environment → S(T) | `permeation.py` §3b |
| 10 | Bulk diffusivity — NPT → NVT → MSD → Arrhenius | `diffusivity_*.py` |
| 11 | Permeability assembly — Φ, Arrhenius, flux | `permeation.py`, `permeation_workflow.py` |
| — | Job generation, SLURM, idempotency (cross-cutting) | `create_slurm.py` → Part IV |

---

## 4. Part I questions

1. What physical question does the workflow answer, and what is Φ?
2. Why three Parts, and why does Part 2 need both 1 and 3?
3. What are the ways to run it, and when is each right?
4. What does each stage consume and produce?
5. What does the method assume, and where do those assumptions break?
6. What software, versions and hardware does it need?

---

## 5. The aggregation ladder (`02b-aggregation.md`)

Answers a question the author asked explicitly: *how do many things become one?*
Twelve collapse points, six distinct operators. Each stage section links to its rung
rather than re-explaining.

| # | from → to | operator | weight | error rule |
|---|---|---|---|---|
| A1 | atoms → surface sites | ACAT enumeration + graph | — | — |
| A2 | sites → environment labels | geometric/coordination classification | — | — |
| A3 | dissociation pathways → ΔH_diss | arithmetic mean | uniform | SEM |
| A4 | Hop A pathways → ΔH_HopA(env) | arithmetic mean **within env** | uniform within env | SEM, **0 when n=1** |
| A5 | environments → S(T) | **Boltzmann-weighted sum** | `w_env` | weighted-sum rule |
| A6 | per-site rates → per-env rates | arithmetic mean of Arrhenius rates | uniform within env | — |
| A7 | per-env rates → ⟨k_f/k_r⟩ | population-weighted mean of ratios | grid site counts | — |
| A8 | H atoms + time origins → MSD(τ) | **ensemble + time average** | uniform | — |
| A9 | MSD(t) → D(T) | regression on fit window, ÷6 | — | σ from fit |
| A10 | D(T) across T → D₀, E_D | log-space regression | optionally 1/σ² | σ from fit, DOF = n−2 |
| A11 | D × S → Φ | **product** | — | fractional σ in quadrature |
| A12 | Φ across loadings → headline | **selection**, not averaging | — | — |

Three points this chapter must make, which no single stage section would:

1. **Two weightings stack, and differ in kind.** A4 averages over whatever pathways were
   computed, so NEB sampling acts as an implicit uniform weight. A5 then applies `w_env`
   *and* a Boltzmann factor on top. A site's influence on S is a product of three things,
   only one of which is physics.
2. **Averaging survives quality filtering.** The census marks pathways unsound, but A3's
   mean runs over the set it is given. Where the filter is applied versus where the mean
   is taken is the difference between a defensible ΔH_diss and a contaminated one.
3. **SEM = 0 means unmeasurable, not precise.** A4 records this whenever an environment
   holds one pathway, and it propagates into `dH_sol_err_eV` as a zero contribution.

---

## 6. Part II equation inventory

21 are already LaTeX in docstrings — harvest, do not reinvent.

**Thermodynamics / transport** — Φ = D·S; Sieverts; Φ₀ = D₀·S₀ and E_Φ = E_D + ΔH_sol;
Fick vs Richardson flux; μ_H ↔ P.

**Statistical mechanics** — harmonic TST; Vineyard with quasi-harmonic floor; ZPE
(forward TS−IS, reverse **TS−FS**); vibrational and H₂ gas partition functions;
Hertz–Knudsen strike rate; detailed-balance θ_eq; Langmuir occupancy.

**Solubility** — ΔH_sol(env) = ½ΔH_diss + ΔH_HopA(env); environment-weighted Boltzmann
sum; geometric S₀ = 4/(a₀³N_A); vibrational S₀; site-density ceiling.

**Diffusion** — MSD definition; MSD = 6Dt; fit window; half-vs-half convergence ratio;
Arrhenius D(T); at.% conversion.

**Energetics** — binding energy; E_a = E_TS − E_IS; ΔE = E_FS − E_IS; E_des = E_TS − E_FS.

**Error propagation** — absolute σ for energies, fractional σ for prefactors; quadrature
per operation; regression σ and DOF; why R² is not an uncertainty.
Source: `audits/error_propagation_plan.md` §2 (rules) and §6 (full equation set) — the one
genuinely reusable existing document.

### `check_equations.py` must verify

ZPE from frequencies · Vineyard product and the floor · the `n_IS = n_TS + 1` rule ·
partition functions · MSD/Einstein slope · Arrhenius round-trip · Φ₀ = D₀·S₀ and
E_Φ = E_D + ΔH_sol · Boltzmann-sum normalisation · oct-site density · unit factors
(cm⁻¹→eV, Å²/ps→m²/s). Prints PASS/FAIL per check and `N / N checks passed`.
Appendix E lists what it checks and the expected output.

---

## 7. Decisions already made

Do not reopen these.

| # | Question | Decision |
|---|---|---|
| 1 | Is Hop B in ΔH_sol? | **Out.** sub1-referenced; Hop B is bulk transport carried by D. Stale comment **fixed** in `ebfe680`. |
| 2 | `S_sub2` / `C0_sub2_*` unit keys | **Abandoned.** 20 dead keys **removed** in `ebfe680`. |
| 3 | Three meanings of `min_freq_cm1` | **Drift, not design** — see §8. Document as IMPORTANT box + Planned rename. |
| 4 | `CONVERGED_BAND = (0.75, 1.25)` | **No recorded rationale.** Document as chosen, not derived; flag it. |
| 5 | `w_env` = Hop A pathway fraction, not site density | **Deliberate.** Document as an approximation with its consequence. |
| 6 | Document `detailed_balance`? | **Yes**, as a supported output, with its averaging artifact stated plainly. |
| 7 | May design rationales cite real-run numbers? | **Yes**, in "Design choices" only. |

### Supersession

`PIPELINE_GUIDE.md`, `multiscale_permeation_plan.md`, `error_propagation_plan.md` get a
header pointing at `docs/` and stay in place for history. **Not deleted.**

The two Project2 explainers (surface and subsurface graphs) are **older but not wrong** —
their algorithms have not changed. Stages 3 and 4 adapt them and note provenance.

---

## 8. Archaeology findings that must reach the docs

### 8a. All six October physics fixes are undocumented

The three big docs were last touched `10-06 14:21` (the KMC-removal commit). Every physics
fix landed after:

```
a3c53fe 15:44  reverse barrier/prefactor from FS
2ba456a 16:06  refuse Vineyard when mode counts cannot be a frequency
8e49ba2 16:48  real desorption prefactor for dissociation
01f5d42 17:05  state-quality census (is_minimum / ts_saddle)
e2e4712 23:28  rebalance mode counts after the frequency cut
580038d 00:16  Truhlar quasi-harmonic floor
```

Grepping docs for `nu_reverse`, `quasi-harmonic`, `is_minimum`, `ts_saddle`, `imag_cut`,
`zpe_source`, `low_freq`, `Truhlar` → **0 hits in docs, 1–4 files each in code.**
Stages 7 and 8 are therefore written from scratch and are the highest-value chapters.

### 8b. `min_freq_cm1` drift (answers decision #3)

Originally (`7bc380b`, 2026-06-17) the name was **coherent**: one threshold meaning
*discard real modes below 50 cm⁻¹*, used identically in three functions. The October
fixes pulled it apart and nobody renamed the survivors:

| function | then | now | changed by |
|---|---|---|---|
| `vineyard_prefactor` | discard below 50 | **raise to** `low_freq_cm1` = 100 | `580038d` |
| `build_rate_dict` | discard below 50 | **imaginary**-mode significance cut → `imag_cut_cm1` | `580038d` |
| `apply_zpe_correction` | discard below 50 | include all (default 0.0) | `a3c53fe`–`580038d` |
| `vib_partition_function` | discard below 50 | **unchanged** — original rule survives | never touched |

Document all four in an IMPORTANT box. Planned rename: `qho_floor_cm1`, `imag_sig_cm1`,
`zpe_min_cm1`, `qvib_min_cm1`.

### 8c. Truhlar floor is well documented in code

`vineyard_prefactor`'s docstring already gives the rationale and precedent
(GoodVibes, Shermo). No gap — harvest it.

---

## 9. Sources to read

| what | where |
|---|---|
| Science modules | `models/{permeation,tst_rates,vibrations,energetics,diffusivity_post_processing}.py` |
| Site mapping | `models/{surface_graph,subsurface_graph,neb_subsurface,site_identifier}.py` |
| Orchestration | `models/{neb,diffusivity,permeation,pipeline}_workflow.py` |
| Structures / inputs | `models/{structure,materials,config}.py` |
| Infrastructure | `models/{create_slurm,lammps_script,parsers,utils}.py` |
| Analysis | `models/plots/*.py` |
| Notebooks | `calculation/{pipeline,neb_calculation,diffusivity,permeation}.ipynb` |
| Reusable prose | `audits/error_propagation_plan.md`, the two Project2 explainers |
| Superseded (read for history) | `PIPELINE_GUIDE.md`, `multiscale_permeation_plan.md` |
| Formats, not results | `calculation/results/*/`, `calculation/results/export/*/` |

---

## 10. Environment

Do not install anything into it.

```
env: mace_env   (~/anaconda3/envs/mace_env/bin/python)
python 3.9.23   numpy 1.26.4   scipy 1.13.1   ase 3.26.0
mace 0.3.8      acat 2.0.1     matplotlib 3.9.4   networkx 3.2.1
```

Re-check command for the scope NOTE:

```bash
~/anaconda3/envs/mace_env/bin/python docs/figures/check_equations.py
```

> [!IMPORTANT]
> The base anaconda environment has a broken matplotlib (NumPy 2 vs 1.x).
> Always invoke `mace_env` by absolute path.

---

## 11. Order of work

1. `docs/figures/example/` — fictitious toy cell running every stage in seconds
2. `check_equations.py` — **before** writing Part II, so every equation is verified first
3. `02-theory.md`
4. `02b-aggregation.md`
5. `03-stages/` in pipeline order — but write **07 (PHVA)** and **08 (TST)** early while
   the archaeology in §8 is fresh
6. `01-overview.md` (easier once stages exist)
7. `04-practical.md`, `05-appendices.md`, `README.md`
8. `make_figures.py`; run both scripts; fix failures before handover
9. Add supersession headers to the three old docs (§7)

### Progress

- [x] Sources surveyed, outline approved
- [x] Decisions 1–7 settled
- [x] Code fixes landed (`ebfe680`); k_B unified across all six definitions
- [x] Toy example — `figures/example/toy.py`
- [x] `check_equations.py` — **40 / 40 passing**
- [x] `02-theory.md` — 37 tagged equations, 9 invariants, equation→code→check index
- [x] `02b-aggregation.md` — the 12-rung ladder
- [x] `03-stages/07-vibrations-phva.md`
- [x] `03-stages/08-tst-rate-assembly.md`
- [x] `03-stages/09-solubility.md`
- [x] `03-stages/10-bulk-diffusivity.md`
- [x] `03-stages/11-permeability-assembly.md`
- [x] `03-stages/01` … `06` — the structural and setup half
- [x] **Part III complete — all 11 stages written**
- [ ] `01-overview.md` · `04-practical.md` · `05-appendices.md` · `README.md`
- [ ] `make_figures.py`
- [ ] Supersession headers

### Found while writing (not in the original survey)

1. **Two Arrhenius helpers with reversed argument order.**
   `permeation.arrhenius_diffusivity(D0, E_D, T)` versus
   `diffusivity_post_processing.arrhenius_D(T, Ea, D0)`. Calling either the
   wrong way round does not raise — it underflows to exactly `0.0`. Both orders
   are now pinned by checks. Documented in Part II §12.
2. **k_B was not shared.** Two modules carried a 7-digit truncation; unified to
   CODATA 2018. Worst effect on any stored D(T) was 4e-7, so no reruns. Four
   checks now pin every definition. Invariant (I7).
3. **A fourth frequency threshold** exists beyond the three in §8b: the
   partition-function cutoff in `vib_partition_function`. Part II §4.3 lists all
   four.
4. **A missing reaction energy defaults to zero** when ΔH_diss is auto-extracted,
   so an incomplete ranked file biases the mean toward zero rather than raising.
   Documented as a failure mode in stage 09.
5. **ΔH_diss is filtered on NEB convergence, not on the vibrational census.**
   Defensible — a reaction energy needs sound endpoint energies, not sound
   frequencies — but it means the census does not gate that average. Stage 09
   states this explicitly.
6. **Dynamics and fit are gated by separate markers** in Part 3, so a loading can
   hold complete trajectories and no Arrhenius fit. Re-deriving costs seconds;
   repeating the dynamics costs GPU-days. Stage 10 failure-mode table.

---

## 12. Still open

Nothing blocking. Carry these into the text as flagged items:

1. **`CONVERGED_BAND` has no justification** and is now load-bearing — it decides which
   diffusivity points are trustworthy. Either justify or make configurable.
2. **`min_freq_cm1` rename** (§8b) — Planned improvement, not yet done.
3. **No test coverage for `models/plots/`** — note in Part IV; the single-H exclusion
   default shipped unguarded.
4. **Ni 10H and Hastelloy N 7 10H** have no Part 3 fit; Hastelloy N 42 has no Part 1 or 2.
   Affects which materials can illustrate a complete chain — the toy example exists so the
   docs never depend on this.

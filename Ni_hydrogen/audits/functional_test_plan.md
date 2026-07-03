# Functional Test Suite — H-in-Alloy Pipeline

## Context

Before running the full pipeline on `main`, we need functional tests that verify
the Python logic — script generation, data flow, parsers, and pure-Python
computation — without needing a GPU or cluster. The pipeline has 5 major sections,
each with sub-phases. Tests cover each section's checkpoints.

**Golden rule:** No LAMMPS runs, no SLURM submission, no GPU required.
Everything runs locally via `dry_run=True` or pure-Python computation.

---

## The 4 Categories of Functional Tests

| Category | What it tests | Speed |
|---|---|---|
| **A — Script generation** | Call orchestrator with `dry_run=True`; assert correct files written with right SLURM config | Fast (ms) |
| **B — Parser correctness** | Feed real fixture log files into each parser; assert extracted values match known answers | Fast (ms) |
| **C — Data flow / interface contracts** | Verify output format of stage N is valid input for stage N+1 (schema/key/type checks) | Fast (ms) |
| **D — Pure-Python computation** | Run KMC, TST rates, Arrhenius fits, permeability math with synthetic inputs; assert numerically correct | Fast (ms–s) |

---

## File Structure

```
tests/
  functional/
    conftest.py                        # shared fixtures: tiny slab, temp work_dir, fixture paths
    fixtures/
      fs_min.log                       # real LAMMPS CG-min log (copy from cluster: neb/*/fs_min.log)
      neb_barrier.txt                  # real barrier file (copy from cluster)
      neb_path.dat                     # real NEB path file (copy from cluster)
      msd_prod.dat                     # real MSD output from NVT production run (copy from cluster)
    test_ft_structure.py               # Section 1: slab + surface sites
    test_ft_surface_neb.py             # Section 2: adsorption + NEB scripts + parsers
    test_ft_subsurface_neb.py          # Section 3: Hop A + Hop B scripts
    test_ft_diffusivity.py             # Section 4: script gen + MSD parse + Arrhenius
    test_ft_permeability.py            # Section 5: TST + KMC + Sieverts + permeability
```

---

## Section 1 — Structure Generation

**File:** `test_ft_structure.py`
**Model functions:** `build_slab()`, `build_surface_graph()`, `save_surface_sites()`
**Sub-phases covered:** A1 (slab construction), A2 (frozen atom count), A3 (site enumeration)

| Test | Category | What it checks |
|---|---|---|
| `test_slab_builds_correct_atom_count` | B | Given a tiny 2×2 Ni FCC slab (3 layers), output `.lammps` has correct atom count |
| `test_slab_frozen_atom_count` | B | Bottom layer atoms ≤ `z_freeze_cutoff` are counted correctly |
| `test_surface_sites_json_schema` | C | `surface_sites.json` has `sites`, `metadata`, `surface_atoms` keys; each site has `site_id`, `site_type`, `position`, `composition` |
| `test_surface_graph_site_count_nonzero` | C | Graph has > 0 site nodes and > 0 site-site edges |

**Fixture:** Small 3-layer pure Ni FCC(111) 2×2 slab built in the test (no cluster needed).

---

## Section 2 — Surface NEB (Adsorption + NEB)

**File:** `test_ft_surface_neb.py`
**Model functions:** `run_h2_minimization_jobs()`, `run_h_minimization_jobs()`,
`prepare_neb_jobs()`, `write_neb_run_script()`, `parse_energy_log()`,
`_parse_pe_final()`, `parse_barrier_file()`, `parse_neb_path()`
**Sub-phases covered:** B1 (H2* adsorption), B2 (H* adsorption), C1–C3 (NEB pair setup), D (FS-min + NEB scripts)

| Test | Category | What it checks |
|---|---|---|
| `test_h2_min_scripts_written` | A | `dry_run=True` writes SLURM array `.sh` + LAMMPS `.in` files; `partition=sharing`, `time=00:20:00` |
| `test_h_min_scripts_written` | A | Same for H* minimization |
| `test_neb_script_written` | A | `run_neb.py` and `slurm_neb_*.sh` exist; TRAJ files use `.traj` (not `.lammpstrj`) |
| `test_neb_traj_write_format_is_extxyz` | A | `run_neb.py` source contains `format="extxyz"`, not `format="lammps-dump-text"` |
| `test_parse_pe_final_skips_echo_line` | B | `_parse_pe_final(fixtures/fs_min.log)` returns float, does not raise ValueError (the LAMMPS command-echo bug) |
| `test_parse_energy_log` | B | `parse_energy_log(fixtures/fs_min.log)` returns dict with `pe_final_eV`, `fmax_eV_per_Ang`, `natoms` as expected values |
| `test_parse_barrier_file` | B | `parse_barrier_file(fixtures/neb_barrier.txt)` returns correct `E_abs`, `E_des`, `delta_E`, `converged=True` |
| `test_parse_neb_path_length` | B | `parse_neb_path(fixtures/neb_path.dat)` returns arrays of length == N_images |
| `test_neb_pair_enumeration_filters_proximity` | C | `enumerate_fs_pairs()` excludes pairs outside 2.5–6.0 Å XY separation |
| `test_neb_dedup_reduces_count` | C | `deduplicate_fs_pairs()` returns ≤ input count |
| `test_neb_pair_output_schema` | C | Each entry in `neb_pairs.json` has `label_is`, `label_fs`, `is_path`, `fs_path`, `delta_E` |

---

## Section 3 — Subsurface NEB (Hop A + Hop B)

**File:** `test_ft_subsurface_neb.py`
**Model functions:** `orchestrate_hopa_neb()`, `orchestrate_hopb_neb()`,
`build_subsurface_graph()`, `connect_to_surface()`
**Sub-phases covered:** Hop A (surface → sub1), Hop B (sub1 → sub2)

| Test | Category | What it checks |
|---|---|---|
| `test_hopa_scripts_written` | A | `dry_run=True` writes `hopa_fsmin_array.sh`, `hopa_neb_array.sh`, `hopa/job_index.txt`; `concurrent=2` in array header |
| `test_hopb_scripts_written` | A | Same for Hop B |
| `test_hopa_slurm_partition` | A | `hopa_fsmin_array.sh` contains `#SBATCH --partition=sharing` and `#SBATCH --time=00:20:00` |
| `test_hopa_returns_jobs_dict` | C | Return dict has `n_jobs`, `fsmin_array`, `neb_array`, `jobs` (list of dicts with `sid`, `is_path`, `fs_path`) |
| `test_hopb_requires_hopa_output` | C | Calling `orchestrate_hopb_neb()` before Hop A FS results exist raises a clear error or returns `n_jobs=0` |
| `test_subsurface_graph_layers` | B | `build_subsurface_graph()` on fixture slab returns sites with `layer_classification` in `{subsurface_1, subsurface_2}` |

---

## Section 4 — Diffusivity (NPT + NVT + MSD + Arrhenius)

**File:** `test_ft_diffusivity.py`
**Model functions:** `generate_diffusivity_scripts()`, MSD parsers,
`run_arrhenius_pipeline()` (`diffusivity_post_processing.py`)
**Sub-phases covered:** 1a (bulk min), 1b (NPT + insert H), 2 (NVT script), 3 (MSD → D → Arrhenius)

| Test | Category | What it checks |
|---|---|---|
| `test_diffusivity_script_written` | A | `generate_diffusivity_scripts(..., dry_run=True)` writes `diffusivity_run.py`; contains `SHORT_GPU_PARTITION=sharing` |
| `test_diffusivity_script_has_all_phases` | A | Generated script contains Phase 1a, 1b, 2, 3 markers |
| `test_diffusivity_nvt_partition` | A | NVT SLURM config in script uses `multigpu` (long run), not `sharing` |
| `test_msd_parse_returns_positive_diffusivity` | B | Parse `fixtures/msd_prod.dat` → D value is positive float |
| `test_arrhenius_fit_known_answer` | D | Feed synthetic D(T) from `D = D0·exp(-Ea/kT)` with `D0=1e-7`, `Ea=0.4 eV` → fit recovers both within 2% |
| `test_lattice_params_json_schema` | C | `lattice_params_vs_T.json` has temperature keys with `a0_angstrom` float values |

---

## Section 5 — Permeability (TST + KMC + Sieverts)

**File:** `test_ft_permeability.py`
**Model functions:** `build_rate_dict()` (`tst_rates.py`), `sweep_pressure()` (`permeation.py`),
`permeability()`, `richardson_flux()`, `check_sieverts_law()`, `fit_solubility_from_kmc()`
**Sub-phases covered:** Phase 4 (TST rates), Phase 5 (KMC), Phase 6 (Sieverts + permeability)

| Test | Category | What it checks |
|---|---|---|
| `test_tst_rate_detailed_balance` | D | Synthetic barrier: `Ea=0.5 eV`, `dE=-0.1 eV`, `T=700K` → `k_fwd/k_rev = exp(dE/kT)` within 0.1% |
| `test_kmc_flux_positive` | D | `sweep_pressure()` with synthetic rates → `J > 0` for all P values |
| `test_kmc_sieverts_linear` | D | `J` vs `√P` is linear (`R² > 0.98`) for bulk-diffusion-limited synthetic rates |
| `test_kmc_convergence_flag` | D | Very high rate → `sweep_pressure()` returns `converged=True` for all pressures |
| `test_permeability_is_D_times_S` | D | `permeability(D, S)` returns `D × S` exactly |
| `test_richardson_flux_scales_with_L` | D | Doubling membrane thickness L halves flux J |
| `test_permeation_script_written` | A | `generate_permeation_scripts(..., dry_run=True)` writes `permeation_run.py`; `partition=sharing` in GPU_SLURM_CFG |
| `test_permeation_script_phases_present` | A | Generated script contains Phase 1–6 markers |
| `test_rate_dict_schema` | C | `rate_dict_T700K.json` has `k_forward`, `k_reverse`, `Ea_eV`, `T_K` keys per entry |

---

## Fixtures to copy from the cluster

Copy these real output files into `tests/functional/fixtures/` and commit them.
They are small text files (10–50 KB each) and never change.

```bash
# On the cluster:
scp cluster:/path/to/neb/s_40__s_89+s_90/fs_min.log       tests/functional/fixtures/
scp cluster:/path/to/neb/s_40__s_89+s_90/neb_barrier.txt  tests/functional/fixtures/
scp cluster:/path/to/neb/s_40__s_89+s_90/neb_path.dat     tests/functional/fixtures/
scp cluster:/path/to/diffusivity/msd_prod.dat              tests/functional/fixtures/
```

---

## Implementation order (most value first)

1. **Category B parsers** (`test_ft_surface_neb.py` parser tests) — directly catches the `_parse_pe_final` and `lammps-dump-text` class of bugs we just fixed; runs in milliseconds
2. **Category D pure-Python** (`test_ft_permeability.py`) — no files needed; catches TST/KMC math bugs; runs in milliseconds
3. **Category A script generation** (all 5 sections, `dry_run=True`) — catches SLURM partition/time regressions like the `multigpu` → `sharing` changes
4. **Category C data flow** — catches interface mismatches between stages (schema checks)
5. **Category B subsurface/diffusivity parsers** — needs fixture files from cluster

---

## How to run

```bash
# All functional tests
pytest tests/functional/ -v

# One section at a time
pytest tests/functional/test_ft_permeability.py -v
pytest tests/functional/test_ft_surface_neb.py -v

# Only fast tests (skip any marked slow)
pytest tests/functional/ -m "not slow" -v
```

**No GPU, no SLURM, no MACE model required.**

---

---

# Detailed Implementation Plans (per section)

---

## Implementation Plan: `test_ft_parsers.py` (Category B — Parser correctness)

### Context from reading the code

**What already exists (`tests/test_parsers.py`):**
All parsers in `models/parsers.py` are already unit-tested with clean, synthetic inline
string content. Every test writes a perfectly formed log with no noise. The functional
tests must cover what the unit tests cannot: **realistic, messy LAMMPS output** — the
exact content that trips up buggy parsers on the cluster.

**The `_parse_pe_final` is not in `parsers.py`:**
It is embedded as a string template inside `write_ase_neb_script()` in `models/ase_neb.py`
(lines 533–545). It is injected into every generated `run_neb.py` at write time. It
cannot be imported directly — it must be tested by either:
1. Extracting and `exec()`-ing the function string from `ase_neb.py` source, or
2. Duplicating the logic in the test with a separate regression guard that checks the
   source of `ase_neb.py` still contains the accumulator pattern.

**Why `parse_energy_log` is already safe:**
`parse_energy_log` in `parsers.py` uses `results[key] = float(...)` inside a
`try/except ValueError: pass`, so the echo line silently fails and the real line
overwrites. The functional test should **confirm** this, not assume it.

**The echo-line trap — what the realistic fixture must contain:**
LAMMPS echoes `print` commands verbatim before executing them. A real `fs_min.log`
contains these two consecutive lines for every `print` statement:
```
print "  pe_final_eV     : ${pe_final}"    ← echo: pe_final_eV + colon, but value = ${pe_final}
  pe_final_eV     : -2271.05882318558      ← real output: same key, numeric value
```
The old broken code returned on the first match → `float('${pe_final}"')` → ValueError.
The fix accumulates and returns the last valid float.

---

### File to create: `tests/functional/test_ft_parsers.py`

#### Inline fixture strings (no external files needed for this test)

```
_FS_MIN_LOG_REALISTIC
```
Must replicate the real LAMMPS log structure including:
- LAMMPS banner + KOKKOS preamble lines (non-numeric noise)
- `minimize` thermo block (Step / PotEng / Fmax / Fnorm / Press rows)
- `Loop time` line (ends thermo block)
- `Minimization stats` block
- `print "### Minimization complete ###"` echo
- The three `print "  pe_final_eV : ${pe_final}"` echo lines + their numeric outputs
- `write_data ...` line
- `Total wall time` line

```
_NEB_BARRIER_REALISTIC
```
Must match real `neb_barrier.txt` format written by `write_neb_results()` in
`ase_neb.py` (lines 737–758). Inspect that function to get exact key names and format:
- Header comment block (`IS :`, `FS :`, `N images :`, `fmax_final :`, `Converged :`)
- Blank line
- `E_IS = ...`, `E_FS = ...`, `E_abs = ...`, `E_des = ...`, `delta_E = ...`
- `Image energies:` section with per-image lines
- `# image_frac  E_eV  dE_from_IS_eV` header
- Numeric data rows

```
_NEB_PATH_REALISTIC
```
Must match real `neb_path.dat` format (3-column, comment header, 20 rows for 18+2 images).

---

#### Test classes

**`class TestParseEnergyLogRealisticFormat`**

Fixture: write `_FS_MIN_LOG_REALISTIC` to `tmp_path / 'fs_min.log'`; call `parse_energy_log`.

| Test method | Asserts |
|---|---|
| `test_returns_dict_not_none` | result is not None |
| `test_pe_final_is_float` | `result['pe_final_eV']` is a float (not a string, not NaN) |
| `test_pe_final_correct_value` | `math.isclose(result['pe_final_eV'], -2271.05882318558, rel_tol=1e-9)` |
| `test_fmax_correct_value` | `math.isclose(result['fmax_eV_per_Ang'], 1.85109507932005e-07, rel_tol=1e-6)` |
| `test_natoms_correct` | `result['natoms'] == 362.0` |
| `test_echo_line_does_not_corrupt_value` | `result['pe_final_eV'] < 0` — confirms numeric value, not a parse of `${pe_final}` |

---

**`class TestParsePeFinalAccumulator`**

This class tests the `_parse_pe_final` logic (the bug we fixed). Since the function
lives in `ase_neb.py` as a string template, we use two approaches in parallel:

*Approach A — inline reimplementation test:*
Define `_parse_pe_final_impl(log_path)` locally in the test file (exact copy of the
accumulator logic from `ase_neb.py`). Test it with the realistic fixture. This confirms
the logic is correct regardless of how it's embedded.

*Approach B — source regression guard:*
Read `models/ase_neb.py` as a string and assert the template block still contains
`_val = None`, `_val = float`, `return _val`. This catches any future regression where
someone reverts the fix or changes the template.

| Test method | Asserts |
|---|---|
| `test_accumulator_skips_echo_returns_float` | `_parse_pe_final_impl(log)` returns `-2271.05882318558` |
| `test_does_not_raise_on_echo_line` | no `ValueError` raised |
| `test_raises_if_key_absent` | `ValueError` raised when `pe_final_eV` not in log at all |
| `test_ase_neb_template_has_accumulator_pattern` | `ase_neb.py` source contains `'_val = None'` and `'return _val'` |
| `test_ase_neb_template_no_bare_return` | `ase_neb.py` `_parse_pe_final` block does **not** contain `'return float('` (old pattern) |

---

**`class TestParseBarrierFileRealisticFormat`**

Fixture: write `_NEB_BARRIER_REALISTIC` to `tmp_path / 'neb_barrier.txt'`; call
`parse_barrier_file`.

| Test method | Asserts |
|---|---|
| `test_E_abs_extracted` | `math.isclose(result['E_abs'], expected_Ea)` |
| `test_E_des_extracted` | `math.isclose(result['E_des'], expected_Ed)` |
| `test_delta_E_extracted` | `math.isclose(result['delta_E'], expected_dE)` |
| `test_converged_is_bool_true` | `result['converged'] is True` |
| `test_image_energies_block_does_not_corrupt` | none of the per-image lines accidentally overwrite `E_abs` |
| `test_comment_header_skipped` | `'IS'` key not in result (it's a comment-block line, not a float key) |

---

**`class TestParseNebPathRealisticFormat`**

Fixture: write `_NEB_PATH_REALISTIC` to `tmp_path / 'neb_path.dat'`; call `parse_neb_path`.

| Test method | Asserts |
|---|---|
| `test_array_length_matches_n_images` | `len(frac) == 20` (18 intermediate + IS + FS) |
| `test_first_frac_is_zero` | `math.isclose(frac[0], 0.0)` |
| `test_last_frac_is_one` | `math.isclose(frac[-1], 1.0)` |
| `test_dE_IS_is_zero` | `math.isclose(dE[0], 0.0)` — IS is the energy reference |
| `test_peak_is_interior_image` | `0 < dE.argmax() < len(dE) - 1` — TS is not IS or FS |
| `test_returns_numpy_arrays` | all three outputs are `np.ndarray` |

---

### What is NOT tested here (already covered in `test_parsers.py`)

- Missing file raises `FileNotFoundError` — already in unit tests
- `=` vs `:` separator variants — already in unit tests
- Empty files return `None` — already in unit tests
- Comment-only files — already in unit tests

The functional tests add only what the unit tests cannot: **realistic format with noise**.

---

---

## Implementation Plan: `test_ft_dataflow.py` (Category C — Data flow / interface contracts)

### Purpose

Category C tests verify that the **output format of stage N is a valid input for stage N+1**.
No LAMMPS, no SLURM, no cluster required. Every test either:
- Reads a **synthetic inline fixture** (a minimal dict/JSON that represents one valid stage output), OR
- Reads a real **production file** from `calculation/` if it exists (skipped gracefully otherwise).

These tests catch schema regressions — e.g. a writer adding a new key without updating readers,
renaming `E_abs` → `Ea`, changing `site_id` from `str` to `int`, or changing `a0_m` units.

---

### Interface map (from code inspection)

| Artifact | Writer | Reader | Critical type constraints |
|---|---|---|---|
| `surface_sites.json` | `surface_graph.save_surface_sites()` | `neb_workflow.load_neb_pools()`, `run_phase1/2_*` | `site_id` is `str` ("s_0"); `level2` keys are `str(int)`; `position` is list of 3 floats |
| `neb_pairs.json` | `neb_workflow.orchestrate_neb()` | `neb_workflow.collect_neb_results()` | 15 required keys; `label` format `<sid>__<sid1>+<sid2>`; all energies `float` |
| `hopa_jobs.json` | `neb_subsurface.orchestrate_hopa_neb()` | `orchestrate_hopb_neb()`, `tst_rates.collect_neb_results()` | `sub1_xyz` is list of 3 floats; `e_is` is `float`; `barrier_file` is `str` |
| `hopb_jobs.json` | `neb_subsurface.orchestrate_hopb_neb()` | `tst_rates.collect_neb_results()` | has `ss2_id` + `hopb_is` (not `is_path`); same `barrier_file` key as Hop A |
| `rate_dict_T{T}K.json` | `tst_rates.rates_to_json()` | `permeation_workflow` Phase 6 body | 9 float keys per label; label starts with `hopa_` or `hopb_` |
| `lattice_params_vs_T.json` | `diffusivity_run.py` (generated) | `permeation_workflow` Phase 5 body | 2 keys: `temperatures` (list), `a0_m` (list, metres, ~3.5e-10); same length |
| `neb_barrier.txt` → barrier dict | `parse_barrier_file()` | `tst_rates.build_rate_dict()` | must have `E_abs` key (not `E_a` which is aliased at parse time); `converged` is `bool` |

---

### Strategy

All tests use **inline synthetic fixtures** — minimal dicts/strings that represent one valid entry.
This keeps tests fully offline with no dependency on `calculation/` files.

Where production files exist locally, a separate optional `@pytest.mark.skipif` class reads
them and validates schema (catches real regressions on developer machines without failing CI).

---

### File to create: `tests/functional/test_ft_dataflow.py`

#### Test classes

---

**`class TestSurfaceSitesJsonSchema`**

Fixture: build a minimal synthetic `surface_sites.json` dict as a Python variable.
Includes `metadata`, `surface_atoms`, and one site entry with `level1`, `level2`, `level3`.

| Test method | Asserts |
|---|---|
| `test_top_level_keys_present` | dict has `metadata`, `surface_atoms`, `sites` |
| `test_metadata_required_keys` | metadata has `n_atoms_total`, `n_sites_total`, `cell`, `slab_composition`, `site_type_counts` |
| `test_site_id_is_string` | `sites[0]['site_id']` is `str` (not `int`) |
| `test_level1_has_required_keys` | `level1` has `site_type`, `composition`, `full_label`, `position`, `atom_indices` |
| `test_level1_position_is_list_of_3_floats` | `level1['position']` is a list of length 3; each element is `float` |
| `test_level2_keys_are_strings` | all keys of `level2` are `str` (atom indices as strings, not ints) |
| `test_level3_is_list` | `level3` is a list (empty OK for isolated site) |
| `test_n_sites_metadata_equals_sites_length` | `metadata['n_sites_total'] == len(sites)` |
| `test_load_neb_pools_consumes_schema` | Call `load_neb_pools()` internal logic with synthetic data → builds `{s['site_id']: s}` dict without KeyError |

---

**`class TestNebPairsJsonSchema`**

Fixture: build a synthetic `neb_pairs.json` list with 2 entries (IS→FS1+FS2).

| Test method | Asserts |
|---|---|
| `test_is_a_list` | parsed JSON is a `list` |
| `test_required_15_keys_present` | each entry has all 15 required keys (`label`, `is_site`, `fs_site1`, `fs_site2`, `E_IS`, `E_FS`, `delta_E`, `is_fs_dist`, `neb_script`, `min_script`, `fsmin_sh`, `neb_sh`, `barrier_file`, `path_file`, `job_dir`) |
| `test_label_format` | `label` matches `r'^s_\d+__s_\d+\+s_\d+$'` (IS__FS1+FS2 pattern) |
| `test_energies_are_float` | `E_IS`, `E_FS`, `delta_E`, `is_fs_dist` are `float` |
| `test_path_keys_are_strings` | `barrier_file`, `path_file`, `job_dir` are all `str` |
| `test_collect_neb_results_reads_barrier_file_key` | `collect_neb_results()` accesses `job['barrier_file']` — verify key accessible without KeyError |

---

**`class TestHopaJobsSchema`**

Fixture: synthetic `hopa_jobs` list with 1 job entry (all required keys present with correct types).

| Test method | Asserts |
|---|---|
| `test_required_keys_present` | job dict has `sid`, `is_path`, `e_is`, `ss1_id`, `sub1_xyz`, `fs_raw`, `fs_relaxed`, `fsmin_script`, `neb_script`, `fsmin_sh`, `neb_sh`, `barrier_file`, `path_file`, `job_dir` |
| `test_sid_is_string` | `job['sid']` is `str` |
| `test_e_is_is_float` | `job['e_is']` is `float` |
| `test_sub1_xyz_is_list_of_3` | `job['sub1_xyz']` has length 3; all elements `float` |
| `test_tst_collect_accesses_sid_and_barrier` | `job['sid']` and `job['barrier_file']` accessible (keys used by `tst_rates.collect_neb_results()`) |
| `test_hopb_can_read_hopa_keys` | `ha_job['sid']` and `ha_job['ss1_id']` accessible (keys read by `orchestrate_hopb_neb()`) |

---

**`class TestHopbJobsSchema`**

Fixture: synthetic `hopb_jobs` list derived from a Hop A job (simulating the Hop B output).

| Test method | Asserts |
|---|---|
| `test_required_keys_present` | job has `sid`, `ss1_id`, `ss2_id`, `hopb_is`, `e_is`, `sub2_xyz`, `fs_raw`, `fs_relaxed`, `fsmin_script`, `neb_script`, `barrier_file`, `path_file`, `job_dir` |
| `test_ss2_id_present_not_ss1_id_only` | `'ss2_id'` in job (Hop B adds this; Hop A only has `ss1_id`) |
| `test_hopb_is_key_not_is_path` | `'hopb_is'` in job; `'is_path'` NOT expected in Hop B (different key than Hop A) |
| `test_e_is_is_float` | `job['e_is']` is `float` |
| `test_sub2_xyz_is_list_of_3` | `job['sub2_xyz']` has length 3 |
| `test_tst_collect_accesses_barrier_file` | `job['barrier_file']` accessible |

---

**`class TestRateDictJsonSchema`**

Fixture: synthetic `rate_dict` with 2 entries (one `hopa_Ni`, one `hopb_Ni`).

| Test method | Asserts |
|---|---|
| `test_is_dict_not_list` | top-level JSON is `dict` |
| `test_required_9_keys_per_entry` | each value has `k_forward`, `k_reverse`, `Ea_raw`, `Ea_zpe`, `Ed_raw`, `Ed_zpe`, `nu`, `delta_e`, `T_K` |
| `test_all_values_are_float` | every value in each entry is `float` or `nan` (no strings, no ints) |
| `test_label_prefix_hopa_or_hopb` | all top-level keys start with `hopa_` or `hopb_` |
| `test_T_K_is_correct_temperature` | `T_K` value matches the temperature in the filename |
| `test_permeation_reader_accesses_k_forward_k_reverse` | `r['k_forward']` and `r['k_reverse']` accessible (used by Phase 6 body to build KMC rate dict) |

---

**`class TestLatticeParamsJsonSchema`**

Fixture: synthetic `lattice_params_vs_T.json` dict with 3 temperature entries.

| Test method | Asserts |
|---|---|
| `test_top_level_keys` | dict has exactly `temperatures` and `a0_m` (no extra keys) |
| `test_lists_same_length` | `len(temperatures) == len(a0_m)` |
| `test_a0_m_values_are_floats` | every element of `a0_m` is `float` |
| `test_a0_m_in_si_metres` | all `a0_m` values are in range (3.4e-10, 3.7e-10) — confirms units are metres not Å |
| `test_temperatures_are_numeric` | all temperature entries are `int` or `float` |
| `test_permeation_reader_can_zip` | `dict(zip(temperatures, a0_m))` works and produces float values (exercises reader pattern from `permeation_workflow.py` line 250) |

---

**`class TestBarrierDictToRateInterface`**

Tests the interface between `parse_barrier_file()` output and `build_rate_dict()` input.
Fixture: write a synthetic `neb_barrier.txt` with `E_a` (alias) and `Converged` (capital C)
as written by the real NEB script — exactly the production format.

| Test method | Asserts |
|---|---|
| `test_parse_barrier_returns_E_abs_not_E_a` | `parse_barrier_file()` normalises `E_a` alias → result key is `'E_abs'`, not `'E_a'` |
| `test_converged_bool_type` | `result['converged']` is exactly `True` (bool, not string) |
| `test_build_rate_dict_reads_E_abs` | `build_rate_dict()` (or equivalent call) reads `neb['E_abs']` — key accessible without KeyError |
| `test_missing_E_des_defaults_to_zero` | if `E_des` absent in barrier file → `build_rate_dict()` uses `neb.get('E_des', 0.0)` without error |

---

### What is NOT tested here

- Round-trip serialisation correctness (JSON → Python → JSON) — not needed; trust `json.dumps/loads`
- Absolute correctness of energies or rates — that is Category D
- Whether the files actually exist on the cluster — these are schema-only tests
- Parser correctness — already in `test_ft_parsers.py`

---

---

## Implementation Plan: Extended test coverage (Phase 2)

**STATUS: COMPLETE (2026-07-02) — 235/235 tests passing**

Covers the remaining gaps identified after Phase 1:

---

### 1. LAMMPS log variants — add to `test_ft_parsers.py`

**File:** `test_ft_parsers.py` (add `TestParseEnergyLogVariants` class)

LAMMPS output differs across versions and KOKKOS configurations. The
parsers must be robust to these variations:

| Fixture variant | What it tests |
|---|---|
| `_FS_MIN_LOG_NO_KOKKOS` | Bare LAMMPS log with no KOKKOS preamble (older cluster modules) |
| `_FS_MIN_LOG_WITH_WARNINGS` | `WARNING:` lines interspersed — must not corrupt pe_final |
| `_FS_MIN_LOG_MULTI_BLOCK` | Two `minimize` sections (restart/continuation) — last value must win |
| `_FS_MIN_LOG_OPENMP` | KOKKOS with OpenMP threads instead of GPU device string |

Tests: `test_no_kokkos_returns_correct_pe`, `test_warning_lines_skipped`,
`test_multi_block_returns_last_value`, `test_openmp_kokkos_returns_correct_pe`

---

### 2. `.wrap()` regression guards — add to `test_ft_script_generation.py`

**Class:** `TestWrapRegressionGuards` (new class in existing file)

Source-code grep guards confirming `.wrap()` immediately follows every
`read(format='lammps-data')` call across 7 model files.

| Guard | File | What regressing would reintroduce |
|---|---|---|
| `surface_graph.py` | `models/surface_graph.py` | Bug 2 — wrong site coordinates |
| `structure.py add_adsorbate` | `models/structure.py` | Wrong adsorption structures |
| `structure.py build_fs_raw` | `models/structure.py` | Wrong NEB FS structures |
| `ase_neb.py IS read` | `models/ase_neb.py` | Wrong NEB IS images |
| `ase_neb.py FS read` | `models/ase_neb.py` | Wrong NEB FS images |
| `neb_subsurface.py` | `models/neb_subsurface.py` | Wrong Hop A/B structures |
| `neb_workflow.py` | `models/neb_workflow.py` | Wrong adsorption pool energies |
| `subsurface_graph.py` | `models/subsurface_graph.py` | Wrong subsurface site positions |

---

### 3. Generated-script AST validity — add to `test_ft_script_generation.py`

**Class:** `TestGeneratedScriptAstValidity`

Generated `permeation_run.py` and `diffusivity_run.py` are Python scripts
run on the cluster. A syntax error in the template would cause a silent
failure hours into a job. Use `ast.parse()` to catch template bugs:

| Test | What it catches |
|---|---|
| `test_permeation_run_py_is_valid_python` | Un-closed brackets, bad f-strings in template |
| `test_diffusivity_run_py_is_valid_python` | Same for diffusivity script |

---

### 4. Phase 1 + 2 adsorption dry-run — `test_ft_surface_neb.py`

**File:** `test_ft_surface_neb.py` (new)
**Functions:** `run_phase1_h2_adsorption()`, `run_phase2_h_adsorption()`

**Fixture strategy:**
- Synthetic 2×2 Ni FCC(111) slab written as LAMMPS data file using ASE
- Synthetic `surface_sites.json` with 2 sites (positions within slab cell)
- `e_clean=0.0`, `e_h2_gas=0.0` safe for `dry_run=True` (not used in script gen)
- `e2t={'Ni': 1, 'H': 2}`, `masses={'Ni': 58.693, 'H': 1.008}`, `elem_str='Al B C Cr Fe Mo Ni H'`

| Test | Category | What it checks |
|---|---|---|
| `test_h2_status_is_generated` | A | `result['status'] == 'generated'` |
| `test_h2_structure_files_created` | A | `structures/slab_h2_{sid}.lammps` exists per site |
| `test_h2_lammps_script_created` | A | `scripts/h2_min_{sid}.in` exists, contains `minimize` |
| `test_h2_slurm_partition_is_sharing` | A | `slurm/h2_slurm_{sid}.sh` contains `sharing` |
| `test_h2_array_script_created` | A | `run_h2_array.sh` exists |
| `test_h2_job_index_contains_sid` | A | `h2_job_index.txt` contains the site id |
| `test_h_status_is_generated` | A | Phase 2 same as above |
| `test_h_structure_files_created` | A | `structures/slab_h_{sid}.lammps` exists per site |
| `test_h_lammps_script_created` | A | `scripts/h_min_{sid}.in` exists, contains `minimize` |
| `test_h_slurm_partition_is_sharing` | A | Sharing partition in Phase 2 SLURM script |
| `test_h_array_script_created` | A | `run_h_array.sh` exists |

---

### 5. Subsurface dry-run — `test_ft_subsurface_neb.py`

**File:** `test_ft_subsurface_neb.py` (new)
**Functions:** `connect_to_surface()`, `orchestrate_hopa_neb()`

**`connect_to_surface()` tests** — fully synthetic (no slab file needed):

| Test | What it checks |
|---|---|
| `test_returns_list` | Return value is a list |
| `test_only_subsurface_1_connected` | `subsurface_2` sites not in result |
| `test_correct_surface_site_matched` | XY-closest surface site is chosen |
| `test_xy_dist_is_float` | `xy_dist` element is numeric |
| `test_no_connections_for_distant_site` | Site >xy_tol from all surface sites → not in result |

**`orchestrate_hopa_neb(dry_run=True)` tests** — synthetic IS file + synthetic graph:
- Synthetic IS `.lammps` file: 2×2 Ni(111) slab + 1 H atom (written with ASE)
- Synthetic `(G, subsurface_sites)` with 1 `subsurface_1` site
- Synthetic `surface_connections = [('ss_0', 's_0', 1.5)]`

| Test | What it checks |
|---|---|
| `test_hopa_status_is_generated` | `result['status'] == 'generated'` |
| `test_hopa_n_jobs_correct` | `result['n_jobs'] == 1` |
| `test_hopa_neb_script_uses_extxyz` | `run_hopa.py` contains `.extxyz` not `.lammpstrj` |
| `test_hopa_fsmin_slurm_has_sharing` | `slurm_fsmin_{sid}.sh` has `partition=sharing` |
| `test_hopa_array_scripts_created` | `hopa_fsmin_array.sh` and `hopa_neb_array.sh` exist |
| `test_hopa_jobs_json_schema` | `hopa_jobs.json` has required keys per entry |
| `test_hopa_sub1_xyz_within_cell` | `sub1_xyz` is a 3-element list of floats |

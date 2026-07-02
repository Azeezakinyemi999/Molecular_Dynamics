# Comprehensive pytest Plan — MHI Nickel Pipeline

**Date:** 2026-07-01  
**Scope:** All testable modules in `models/` + generated-script content  
**Reference style:** `tests/test_nvt_restart.py` — class-per-functional-area, docstring per test, no mocking of core logic, `tmp_path` fixtures for disk I/O  

---

## Ground Rules

1. **One section at a time.** Each section is confirmed passing before the next begins.  
2. **No production execution.** All tests work offline: no SLURM, no LAMMPS, no cluster.  
3. **Generated-script tests.** For workflow generators, we inspect the *content* of the written Python/shell script as a string — same pattern used in `test_nvt_restart.py::TestDiffusivityWorkflowBody`.  
4. **Pure-math tests.** Physics/math functions tested with known-answer inputs (analytical or tabulated values).  
5. **File-I/O tests.** Parser tests write synthetic input files to `tmp_path` then call the parser.  
6. **Failure mode tests.** Every documented failure mode (wrong args, missing file, edge cases) gets at least one test.  

---

## Section Order

| # | Test file | Covers | Rationale |
|---|-----------|--------|-----------|
| 1 | `test_parsers.py` | `models/parsers.py` | Foundational — all other sections consume parser output |
| 2 | `test_structure.py` | `models/structure.py` | Core slab builder — pure-metal/alloy/oxide branches |
| 3 | `test_create_slurm.py` | `models/create_slurm.py` | SLURM utilities used by every workflow |
| 4 | `test_lammps_scripts.py` | `models/lammps_script.py` | LAMMPS script generators (non-NVT parts not in existing test) |
| 5 | `test_diffusivity_post_processing.py` | `models/diffusivity_post_processing.py` | MSD + Arrhenius math |
| 6 | `test_energetics.py` | `models/energetics.py` | NEB energy analysis helpers |
| 7 | `test_tst_rates.py` | `models/tst_rates.py` | ZPE, Vineyard, rate dict assembly |
| 8 | `test_kmc.py` | `models/kmc.py` | KMC grid, events, BKL stepper |
| 9 | `test_permeation.py` | `models/permeation.py` | Permeation flux, Sieverts law, solubility |
| 10 | `test_diffusivity_workflow.py` | `models/diffusivity_workflow.py` | Generated `diffusivity_run.py` content |
| 11 | `test_permeation_workflow.py` | `models/permeation_workflow.py` | Generated `permeation_run.py` content |
| 12 | `test_pipeline_workflow.py` | `models/pipeline_workflow.py` | Generated `pipeline_run.py` content |
| 13 | `test_neb_workflow.py` | `models/neb_workflow.py` | Slab-phase orchestration + generated `neb_run.py` |

---

---

## Section 1 — `test_parsers.py`

### What it covers
All 10 functions in `models/parsers.py`. Every parser reads files (LAMMPS logs, dump trajectories, barrier files, NEB paths, diffusivity tables). Tests write synthetic files to `tmp_path` then call the parser and assert the returned data.

### Test Classes

#### `TestParseLammpsLogMinimization`
Tests `parse_minimization_log()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_thermo_and_meta` | Returns 2-tuple `(thermo, meta)` |
| `test_thermo_lists_same_length` | All thermo keys (`step`, `pe`, `fmax`, `lx`, `press`) have identical list lengths |
| `test_total_energy_in_meta` | `meta['Total_energy_eV']` present and float |
| `test_natoms_in_meta` | `meta['Natoms']` > 0 |
| `test_ecoh_in_meta` | `meta['Ecoh_eV_per_atom']` ≈ total_energy / natoms |
| `test_a0_in_meta` | `meta['a0_Angstrom']` > 0 |
| `test_fmax_in_meta` | `meta['Fmax_eV_per_Ang']` > 0 |
| `test_stop_criterion_in_meta` | `meta['stop_criterion']` is str |
| `test_missing_file_raises` | `FileNotFoundError` on non-existent log |
| `test_missing_results_block_returns_empty_meta` | Synthetic log with no `MINIMIZATION_RESULTS_START` → meta = {} |
| `test_malformed_lines_skipped` | Lines without `:` separator in results block → no crash |

#### `TestParseSurfaceRelaxationLog`
Tests `parse_surface_relaxation_log()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_tuple` | Returns `(thermo, meta)` |
| `test_thermo_has_time_temp_pe` | Keys `time`, `temp`, `pe` present and non-empty |
| `test_z_top_before_in_meta` | `meta['z_top_before_Ang']` present |
| `test_z_top_after_nvt_in_meta` | `meta['z_top_after_nvt_Ang']` present |
| `test_frozen_free_counts_in_meta` | `meta['frozen_count']` + `meta['free_count']` is integer total |
| `test_missing_groups_line_graceful` | No `GROUPS:` line → keys absent but no crash |
| `test_surface_contraction_sign` | `meta['surface_contraction']` is float (can be negative) |

#### `TestParseEquilLog`
Tests `parse_equil_log()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_dict_on_valid_file` | Returns dict with `T_K`, `pe_final_eV`, `temp_final_K`, `press_final` |
| `test_returns_none_on_missing_file` | Missing path → `None` (no exception) |
| `test_returns_none_on_empty_file` | Empty file → `None` |
| `test_pe_is_float` | `result['pe_final_eV']` is float |

#### `TestParseEnergyLog`
Tests `parse_energy_log()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_dict_on_valid_file` | Returns dict with `pe_final_eV`, `fmax_eV_per_Ang`, `natoms` |
| `test_returns_none_on_missing_file` | Missing → `None` |
| `test_returns_none_on_empty_results` | File present but no key-value pairs → `None` |

#### `TestParseThermoSeries`
Tests `parse_thermo_series()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_three_arrays` | Returns `(steps, temps, pes)` all `ndarray` |
| `test_arrays_same_length` | All three arrays identical length |
| `test_positional_parsing_correct` | Steps match column 0, temps column 1, pes column 2 |
| `test_missing_file_returns_none` | Missing file → `None` |
| `test_no_header_returns_none` | Synthetic file without `Step Temp` header → `None` |
| `test_malformed_row_skipped` | Row with < 3 columns → skipped, remaining rows still parsed |

#### `TestParseLammpsDump`
Tests `parse_lammps_dump()`. Most complex parser — many edge cases.

| Test | What it asserts |
|------|----------------|
| `test_returns_three_arrays` | Returns `(t_arr, pos_arr, box_arr)` all `ndarray` |
| `test_t_arr_shape` | `t_arr.shape == (n_frames,)` |
| `test_pos_arr_shape` | `pos_arr.shape == (n_frames, n_h_max, 3)` |
| `test_box_arr_shape` | `box_arr.shape == (n_frames, 3)` |
| `test_time_calculation` | `t_arr[i] == steps[i] * timestep` |
| `test_h_positions_extracted` | Positions for atoms of `h_type` match synthetic input |
| `test_non_h_atoms_excluded` | Atoms with type != `h_type` not in `pos_arr` |
| `test_nan_padding_single_frame_zero_h` | Frame with no H atoms → row of `[NaN, NaN, NaN]` |
| `test_nan_padding_variable_h_count` | Frames with different H counts → padded to max with NaN |
| `test_missing_file_returns_triple_none` | Missing file → `(None, None, None)` |
| `test_empty_file_returns_triple_none` | Empty file → `(None, None, None)` |
| `test_box_lengths_correct` | Lx, Ly, Lz from `BOX BOUNDS` match known values |
| `test_missing_atoms_block_fills_nan` | Frame without `ITEM: ATOMS` → NaN positions, no crash |
| `test_multiple_frames_parsed` | N-frame synthetic dump → `t_arr.shape == (N,)` |
| `test_monotonic_time` | `t_arr` strictly increasing |
| `test_header_column_fallback` | Dump without explicit column names → falls back to `[1,2,3,4]`, no crash |

#### `TestParseDiffusivityFile`
Tests `parse_diffusivity_file()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_four_arrays` | Returns `(T_arr, D_arr, Derr_arr, R2_arr)` all `ndarray` |
| `test_all_same_length` | All four arrays have identical length |
| `test_values_correct` | Known synthetic table → values match row-by-row |
| `test_header_lines_skipped` | Lines starting with `#`, `T_K`, `=`, `-`, `:` are skipped |
| `test_missing_file_raises` | `FileNotFoundError` on missing path |
| `test_r2_range` | All R² values between 0 and 1 (for well-formed data) |

#### `TestParseBarrierFile`
Tests `parse_barrier_file()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_e_abs_present` | `result['E_abs']` is float |
| `test_converged_true_variants` | Values `'true'`, `'1'`, `'yes'` (case-insensitive) → `True` |
| `test_converged_false_variants` | Values `'false'`, `'0'`, `'no'` → `False` |
| `test_key_alias_e_a` | File has `E_a:` key → normalized to `E_abs` |
| `test_colon_and_equals_delimiters` | Both `key: value` and `key = value` parsed correctly |
| `test_missing_file_raises` | `FileNotFoundError` on missing path |
| `test_empty_file_returns_empty_dict` | Empty file → `{}` (no crash) |
| `test_e_des_optional` | File without `E_des` → `E_des` key absent, no crash |

#### `TestParseNebPath`
Tests `parse_neb_path()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_three_arrays` | Returns `(frac_arr, E_arr, dE_arr)` all `ndarray` |
| `test_all_same_length` | All three arrays identical length |
| `test_comment_lines_skipped` | `#` lines → skipped |
| `test_values_match_input` | Known synthetic path file → values match |
| `test_missing_file_raises` | `FileNotFoundError` on missing path |
| `test_peak_energy_is_ts` | `E_arr.max()` corresponds to transition state |

---

## Section 2 — `test_structure.py`

### What it covers
`models/structure.py`: `build_slab()` with all three branches, `get_lattice_parameter()`, `get_lattice_parameter_from_dump()`, `_CRYSTAL_STRUCT_MAP`, and the `write_lammps_data()` function.

### Approach
`build_slab` requires a minimized LAMMPS data file as input. Tests will write synthetic LAMMPS-format files to `tmp_path` with known compositions and lattice parameters, then inspect the written slab output.

### Test Classes

#### `TestGetLatticeParameter`
Tests `get_lattice_parameter()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_float` | Output is `float` |
| `test_correct_value_from_known_supercell` | Synthetic 5×5×5 Ni supercell at a=3.52 Å → `a0 ≈ 3.52` |
| `test_scales_with_supercell_reps` | Same box, `supercell_reps=(3,3,3)` vs `(5,5,5)` → different a0 by ratio |
| `test_missing_file_raises` | `FileNotFoundError` on non-existent path |

#### `TestGetLatticeParameterFromDump`
Tests `get_lattice_parameter_from_dump()`.

| Test | What it asserts |
|------|----------------|
| `test_returns_float` | Output is `float` |
| `test_correct_value_from_known_lx` | Synthetic dump with `Lx=17.6` and `(5,5,5)` reps → `a0 ≈ 3.52` |
| `test_comment_lines_skipped` | Lines starting with `#` → skipped |
| `test_uses_last_n_rows` | File with 100 rows, `n_last=10` → averages only last 10 |
| `test_fewer_rows_than_n_last` | 3 rows, `n_last=50` → averages all 3 (graceful) |
| `test_missing_file_raises` | `FileNotFoundError` |
| `test_zero_supercell_rep_raises` | `supercell_reps=(0,5,5)` → `ZeroDivisionError` |

#### `TestCrystalStructMap`
Tests `_CRYSTAL_STRUCT_MAP` lookup table.

| Test | What it asserts |
|------|----------------|
| `test_fcc_elements` | `Ni`, `Al`, `Cu` → `'fcc'` crystal structure |
| `test_bcc_elements` | `Fe`, `Cr`, `Mo`, `W`, `V` → `'bcc'` crystal structure |
| `test_template_element_present` | Each entry has a valid template element string |
| `test_unknown_element_raises` | `_CRYSTAL_STRUCT_MAP['Zr']` → `KeyError` |

#### `TestBuildSlabAlloy`
Tests `build_slab(metal_type='alloy')`.

| Test | What it asserts |
|------|----------------|
| `test_returns_path_and_a0` | Returns `(str, float)` |
| `test_output_file_created` | `out_path` exists on disk |
| `test_a0_positive` | `a0 > 0` |
| `test_slab_composition_matches_bulk_fractions` | Element counts in slab match bulk composition fractions (within rounding) |
| `test_different_seeds_give_different_slabs` | Same input, `seed=7` vs `seed=42` → different element assignments |
| `test_same_seed_gives_same_slab` | Two calls with same seed → identical output |
| `test_lateral_repeat_scales_atom_count` | `lat_repeat=(2,2)` → 4× atom count vs `(1,1)` |
| `test_missing_bulk_file_raises` | `FileNotFoundError` on bad path |
| `test_vacuum_in_cell` | Output slab cell z-length > `vacuum` |

#### `TestBuildSlabPure`
Tests `build_slab(metal_type='pure')`.

| Test | What it asserts |
|------|----------------|
| `test_returns_path_and_a0` | Returns `(str, float)` |
| `test_all_atoms_are_pure_element` | All atoms in output slab are single element (no others) |
| `test_fe_uses_bcc_geometry` | Fe slab with `miller=(1,1,0)` (BCC cleavage) — atom count consistent with BCC layers |
| `test_ni_uses_fcc_geometry` | Ni slab — atom count consistent with FCC layers |
| `test_two_non_h_elements_raises` | Bulk with `Fe` and `Ni` → `ValueError` |
| `test_unknown_element_raises` | Bulk with `Zr` → `KeyError` referencing `_CRYSTAL_STRUCT_MAP` |
| `test_deterministic_no_seed_effect` | `seed=7` vs `seed=99` → identical output (pure is deterministic) |

#### `TestBuildSlabOxide`
Tests `build_slab(metal_type='oxide')`.

| Test | What it asserts |
|------|----------------|
| `test_returns_path_and_a0` | Returns `(str, float)` |
| `test_stoichiometry_preserved` | O:metal ratio in slab matches input bulk (±1 atom rounding) |
| `test_a0_from_primitive_cell` | `a0` equals `primitive_atoms.cell.lengths()[0]` |
| `test_spglib_none_raises_runtime_error` | Monkeypatched `spglib.find_primitive` returning `None` → `RuntimeError` |
| `test_output_file_created` | File exists on disk |
| `test_deterministic` | Same input → identical output (no random shuffle) |
| `test_symprec_fallback` | Slightly distorted structure still finds primitive cell within `symprec=1e-2` |

#### `TestBuildSlabInvalidMetalType`
| Test | What it asserts |
|------|----------------|
| `test_unknown_metal_type_raises` | `metal_type='crystal'` → `ValueError` or `NotImplementedError` |

---

## Section 3 — `test_create_slurm.py`

### What it covers
`models/create_slurm.py`: `write_slurm_job()`, `write_chained_slurm_job()`, `_hms_to_seconds()`. Does NOT test cluster-facing functions (`submit_slurm_job`, `wait_for_jobs`, `auto_submit`) — those require a live SLURM environment.

Note: `test_nvt_restart.py` already tests `write_chained_slurm_job` for the NVT phase-aware path. This file extends that coverage to: `write_slurm_job` (all modes), `_hms_to_seconds`, and `write_chained_slurm_job` edge cases not yet covered.

### Test Classes

#### `TestHmsToSeconds`
Tests `_hms_to_seconds()`.

| Test | What it asserts |
|------|----------------|
| `test_simple_hms` | `'01:30:00'` → 5400 |
| `test_with_days` | `'1-02:00:00'` → 93600 |
| `test_zero_time` | `'00:00:00'` → 0 |
| `test_large_hours` | `'48:00:00'` → 172800 |
| `test_malformed_raises` | `'invalid'` → `ValueError` |
| `test_missing_seconds_raises` | `'01:30'` → `ValueError` |

#### `TestWriteSlurmJobCommands`
Tests `write_slurm_job()` with `commands=` argument.

| Test | What it asserts |
|------|----------------|
| `test_file_created` | Output `.sh` exists |
| `test_file_executable` | `st_mode & stat.S_IXUSR` is set |
| `test_sbatch_header_present` | `#SBATCH --job-name` present |
| `test_partition_in_header` | `#SBATCH --partition={partition}` present |
| `test_ntasks_in_header` | `#SBATCH --ntasks=` present |
| `test_cpus_per_task_in_header` | `#SBATCH --cpus-per-task=` present |
| `test_time_in_header` | `#SBATCH --time=` present |
| `test_gpu_requested_when_set` | `slurm_cfg['gpu']='a100:1'` → `#SBATCH --gres=gpu:a100:1` present |
| `test_no_gpu_line_when_none` | `slurm_cfg['gpu']=None` → no `--gres` line |
| `test_conda_activation_present` | `conda activate {env}` in script |
| `test_module_load_openmpi` | `module load OpenMPI/` present |
| `test_module_load_cuda` | `module load cuda/` present |
| `test_commands_present` | Every string in `commands` appears in script |
| `test_ld_path_export` | Each path in `ld_paths` appears in `LD_LIBRARY_PATH` export |
| `test_no_ld_path_when_empty` | `ld_paths=[]` → no `LD_LIBRARY_PATH=` line |
| `test_extra_env_vars_present` | `extra_env_vars={'MY_VAR': '42'}` → `export MY_VAR=42` in script |
| `test_array_range_in_header` | `array_range=(0, 9)` → `#SBATCH --array=0-9` |
| `test_concurrent_throttle` | `concurrent=4` → `%4` in array header |
| `test_neither_commands_nor_script_raises` | Calling with neither → `ValueError` |

#### `TestWriteSlurmJobLammps`
Tests `write_slurm_job()` with `runner='lmp'`.

| Test | What it asserts |
|------|----------------|
| `test_lammps_command_in_script` | `lammps_cmd -in {script_path}` present |
| `test_kokkos_flags_in_script` | `-k on g 1 -sf kk` (or configured flags) present |
| `test_log_argument_present` | `-log` flag present |
| `test_missing_lammps_cmd_raises` | `lammps_cmd=None` with `runner='lmp'` → `ValueError` |

#### `TestWriteSlurmJobPython`
Tests `write_slurm_job()` with `runner='python'`.

| Test | What it asserts |
|------|----------------|
| `test_python_command_in_script` | `python {script_path}` present |
| `test_no_lammps_command` | `lmp` not in commands section |

#### `TestWriteChainedSlurmJobEdgeCases`
Extends `TestWriteChainedSlurmJobPhaseAware` / `TestWriteChainedSlurmJobLegacy` in `test_nvt_restart.py`.

| Test | What it asserts |
|------|----------------|
| `test_cutoff_appears_in_script` | `cutoff` HH:MM:SS string appears in script |
| `test_flush_wait_configurable` | `flush_wait=60` → 60 appears in sleep command |
| `test_work_dir_cd_present` | `cd {work_dir}` present when `work_dir` provided |
| `test_exit_nonzero_not_resubmitted` | Exit code other than 0 or 124 → no `sbatch` call for that branch |
| `test_restart_glob_in_script` | `restart_glob` pattern appears in checkpoint detection |
| `test_first_run_command_list_all_present` | All items in `first_commands` list appear in script |
| `test_missing_equil_restart_with_n_equil_raises` | `n_equil` set but `equil_restart_commands` absent → `ValueError` |

---

## Section 4 — `test_lammps_scripts.py`

### What it covers
`models/lammps_script.py` functions NOT already covered by `test_nvt_restart.py`:
- `write_minimization_script()`
- `write_npt_script()` (fresh-run)
- `write_npt_restart_script()` (Task A3)
- `write_surface_relaxation_script()` (including Task H quench phase)
- `write_surface_relaxation_restart_script()`
- `write_nvt_bulk_restart_script()` (backward-compat stub)

### Test Classes

#### `TestWriteMinimizationScript`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | Output `.lammps` exists |
| `test_read_data_present` | `read_data` command references `input_file` |
| `test_pair_style_present` | `pair_style {pair_style}` present |
| `test_pair_coeff_present` | `pair_coeff` references `mace_model` |
| `test_elem_str_in_pair_coeff` | `elem_str` appears in `pair_coeff` line |
| `test_minimize_command_present` | `minimize` command with `etol ftol maxiter maxeval` present |
| `test_write_data_output` | `write_data` references `out_file` |
| `test_thermo_output` | `thermo` and `thermo_style` present |
| `test_results_block_written` | `MINIMIZATION_RESULTS_START` sentinel present |
| `test_total_energy_reported` | `Total_energy_eV` key in results block |
| `test_fmax_reported` | `Fmax_eV_per_Ang` key in results block |
| `test_a0_reported` | `a0_Angstrom` key in results block |

#### `TestWriteNptScript`
Tests `write_npt_script()` — fresh-run NPT.

| Test | What it asserts |
|------|----------------|
| `test_file_created` | Output `.lammps` exists |
| `test_read_data_present` | `read_data` references bulk file |
| `test_stage1_heat_steps` | `run {heat_steps}` present for Stage 1 |
| `test_stage2_npt_steps` | `run {npt_steps}` present for Stage 2 |
| `test_fix_npt_present` | `fix {name} all npt` present |
| `test_box_dims_dump_present` | `fix boxdump` with box dimensions output |
| `test_no_append_on_first_run` | Stage 2 box dump does NOT use `append yes` |
| `test_write_restart_present` | `write_restart` command saves checkpoint |
| `test_target_temperature_correct` | Both ramp endpoints and production temp use `target_t` |
| `test_thermo_damp_correct` | `{thermo_damp}` appears in `fix npt` line |

#### `TestWriteNptRestartScript`
Tests `write_npt_restart_script()` (Task A3).

| Test | What it asserts |
|------|----------------|
| `test_file_created` | Output `.lammps` exists |
| `test_read_restart_not_read_data` | `read_restart` present, `read_data` absent |
| `test_no_velocity_create` | `velocity all create` absent (restart restores velocities) |
| `test_append_yes_on_box_dump` | Box dims dump uses `append yes` |
| `test_no_heat_stage` | `heat_steps` absent from script (no Stage 1) |
| `test_npt_steps_present` | `run {npt_steps}` for Stage 2 |
| `test_target_temperature` | `target_t` appears in `fix npt` |

#### `TestWriteSurfaceRelaxationScript`
Tests `write_surface_relaxation_script()` including Task H Phase 4 quench.

| Test | What it asserts |
|------|----------------|
| `test_file_created` | Output `.in` exists |
| `test_four_phases_present` | Markers for Phase 1 (minimize), 2 (anneal), 3 (NVT), 4 (quench) all present |
| `test_phase3_output_filename` | `write_data` in Phase 3 references `phase3_nvt.lammps` (not final output) |
| `test_phase4_input_is_phase3_output` | Phase 4 `read_data` references Phase 3 output filename |
| `test_final_write_data_is_relaxed_slab` | Last `write_data` references `relaxed_slab` filename |
| `test_freeze_bottom_layers` | `group frozen` and `fix setforce` present |
| `test_z_freeze_cutoff_in_group_def` | `z_freeze_cutoff` value appears in group selection |
| `test_nvt_temperature_correct` | `{target_t}.0 {target_t}.0` in `fix nvt_relax` |
| `test_quench_minimize_present` | `minimize` command in Phase 4 |
| `test_quench_etol_ftol` | `etol=0.0 ftol=1e-6` in Phase 4 minimize |
| `test_trajectory_dumps_present` | `dump` commands for NVT and quench trajectories |
| `test_results_block_includes_quench_z` | `z_top_after_quench_Ang` key in results block |
| `test_restart_files_written` | `write_restart` for Phase 3 and 4 |

#### `TestWriteSurfaceRelaxationRestartScript`
Tests `write_surface_relaxation_restart_script()` (Task C restart + H quench).

| Test | What it asserts |
|------|----------------|
| `test_read_restart_not_read_data` | `read_restart` present |
| `test_phase3_continues_nvt` | NVT fix with same `target_t` present |
| `test_traj_append_yes` | Dump for NVT trajectory uses `append yes` |
| `test_thermo_append_yes` | `ave/time` fix for thermo uses `append yes` |
| `test_phase4_quench_present` | Phase 4 minimize present |
| `test_quench_params_match_fresh` | Same `etol`, `ftol`, `maxiter`, `maxeval` as fresh script |
| `test_final_write_data_present` | `write_data` for `relaxed_slab` present |

#### `TestNvtBulkRestartScriptBackwardCompat`
Tests that `write_nvt_bulk_restart_script()` still exists (backward-compat stub).

| Test | What it asserts |
|------|----------------|
| `test_importable` | `from models.lammps_script import write_nvt_bulk_restart_script` → callable |
| `test_returns_path` | Calling it returns a string path |

---

## Section 5 — `test_diffusivity_post_processing.py`

### What it covers
`models/diffusivity_post_processing.py`: all 11 functions. Pure math + I/O. All tests work with synthetic numpy arrays or synthetic files — no LAMMPS required.

### Test Classes

#### `TestPhysicsConstants`
| Test | What it asserts |
|------|----------------|
| `test_kb_ev_value` | `KB_EV ≈ 8.617333e-5` within 1e-10 |
| `test_unit_conversion` | `ANG2_PS_TO_M2S == 1e-8` |

#### `TestUnwrapTrajectory`
| Test | What it asserts |
|------|----------------|
| `test_no_jump_unchanged` | Small displacements (< L/2) → wrapped == unwrapped |
| `test_single_positive_jump_corrected` | Atom moves +L → unwrapped coordinate increases by L |
| `test_single_negative_jump_corrected` | Atom moves −L → coordinate decreases by L |
| `test_multiple_jumps_accumulate` | 3 crossings → continuous trajectory without discontinuities |
| `test_output_shape_matches_input` | `(n, m, 3)` → `(n, m, 3)` |
| `test_nan_propagates` | NaN in positions → NaN in unwrapped (doesn't crash) |
| `test_box_length_zero_raises` | `box_lengths=[0, 5, 5]` → `ZeroDivisionError` or `ValueError` |

#### `TestComputeMsd`
| Test | What it asserts |
|------|----------------|
| `test_zero_displacement_gives_zero_msd` | Stationary atoms → `msd_arr` all zeros |
| `test_linear_motion_correct` | Atom at const velocity `v` → `MSD(τ) = v² τ²` (ballistic) |
| `test_msd_is_nonnegative` | `msd_arr >= 0` always |
| `test_lag_frames_length` | `len(lag_frames) == len(msd_arr)` |
| `test_default_max_lag` | Default `max_lag = n_frames // 2` |
| `test_custom_max_lag` | `max_lag=10` → `lag_frames.max() == 10` |
| `test_single_atom_single_frame_raises` | Single-frame trajectory → graceful (lag 0 = MSD 0) |
| `test_multi_atom_average` | Two atoms with different velocities → MSD = mean of individual MSDs |
| `test_nan_in_positions_gives_nan_msd` | NaN position → NaN in some MSD lags (no crash) |

#### `TestFitDiffusivity`
| Test | What it asserts |
|------|----------------|
| `test_perfect_linear_msd` | `MSD = 6D*t` → `D` recovered within 1% |
| `test_returns_d_sigma_r2` | Returns `(D, sigma_D, R2)` |
| `test_r2_near_one_for_linear` | Perfect linear data → `R2 ≥ 0.999` |
| `test_d_positive` | `D > 0` for positive slope MSD |
| `test_fit_window_respected` | Only central `(0.2, 0.8)` fraction of t_arr used |
| `test_custom_fit_window` | `fit_window=(0.1, 0.9)` → more points, lower uncertainty |
| `test_empty_window_raises` | `fit_window=(0.5, 0.5)` → `ValueError` or `RuntimeError` |
| `test_unit_conversion_applied` | Returns D in m²/s (not Å²/ps) |

#### `TestSaveDiffusivityTable`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | Output file exists |
| `test_parseable_by_parse_diffusivity_file` | Written file read back → arrays match |
| `test_column_order` | Columns are T_K, D, sigma_D, R2 |
| `test_length_mismatch_raises` | Arrays of different lengths → `ValueError` |

#### `TestRunDiffusivityPipeline`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict_with_required_keys` | Keys: `D`, `sigma_D`, `R2`, `msd_file`, `t_lag`, `msd_arr` |
| `test_msd_file_written` | `msd_{T}K.txt` exists in outdir |
| `test_missing_dump_raises` | `FileNotFoundError` on bad dump path |
| `test_d_positive` | `D > 0` for synthetic diffusing H trajectory |

#### `TestWeightedLinregress`
Tests `_weighted_linregress()` (internal helper).

| Test | What it asserts |
|------|----------------|
| `test_uniform_weights_matches_ols` | Equal weights → same as unweighted OLS |
| `test_high_weight_dominates` | One point with 100× weight → slope biased toward that point |
| `test_slope_intercept_correct` | Known 2-point line → exact slope, intercept |
| `test_all_zero_weights_raises` | `weights=[0,0,0]` → `ZeroDivisionError` |
| `test_identical_x_raises` | All `x` equal → singular system |

#### `TestFitArrhenius`
| Test | What it asserts |
|------|----------------|
| `test_recovers_ea` | Synthetic D(T) from known Eₐ → recovered within 1% |
| `test_recovers_d0` | Synthetic D(T) → recovered D₀ within 2% |
| `test_r2_near_one` | Perfect Arrhenius data → `R2 ≥ 0.999` |
| `test_returns_five_values` | Returns `(Ea, Ea_err, D0, D0_err, R2)` |
| `test_single_temperature_raises` | Single T point → undefined regression |
| `test_d_zero_raises` | `D_arr` contains 0 → `log(0)` → handled (`ValueError` or `-inf` propagation) |
| `test_ea_units_ev` | Eₐ is in eV (< 5.0 for H diffusion in metals) |
| `test_d0_positive` | `D0 > 0` |

#### `TestArrheniusD`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `D(T, Ea, D0)` matches `D0 * exp(-Ea / (KB_EV * T))` |
| `test_array_input` | T as ndarray → D as ndarray same shape |
| `test_t_zero_raises` | `T=0` → `ZeroDivisionError` |
| `test_ea_zero_gives_d0` | `Ea=0` → `D = D0` at any T |
| `test_decreases_with_ea` | Higher Eₐ → lower D at same T |

---

## Section 6 — `test_energetics.py`

### What it covers
`models/energetics.py`: all energy calculation helpers. All pure math — no disk I/O except `summarise_neb()` and `plot_mep()`.

### Test Classes

#### `TestCalcBindingEnergy`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `E_bind = E_slab_h - E_clean - 0.5 * E_h2` matches hand calculation |
| `test_n_h2_ref_one` | `n_h2_ref=1.0` → subtracts full H₂ energy |
| `test_negative_binding_stable` | Negative binding = exothermic = stable (not an error) |
| `test_zero_energy_difference` | `E_with_h == E_clean + 0.5*E_h2` → `E_bind = 0` |

#### `TestCalcDissociationBarrier`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `E_ts - E_is` matches |
| `test_zero_barrier` | `E_ts == E_is` → `Ea = 0` |
| `test_negative_barrier_allowed` | `E_ts < E_is` → negative value returned (no crash) |

#### `TestCalcReactionEnergy`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `E_fs - E_is` matches |
| `test_exothermic` | `E_fs < E_is` → negative ΔE |
| `test_endothermic` | `E_fs > E_is` → positive ΔE |

#### `TestMuHPressureConversions`
| Test | What it asserts |
|------|----------------|
| `test_roundtrip_consistency` | `pressure_to_mu_h(mu_h_to_pressure(mu, e_h2, T), e_h2, T) ≈ mu` |
| `test_higher_pressure_higher_mu` | Higher P → higher μ_H |
| `test_zero_pressure_raises` | `pressure_to_mu_h(0.0, ...)` → math domain error |
| `test_t_zero_raises` | `T=0` → `ValueError` |

#### `TestRankSites`
| Test | What it asserts |
|------|----------------|
| `test_sorted_ascending` | Output is sorted ascending by energy |
| `test_all_sites_returned` | All sites in input dict appear in output |
| `test_most_stable_first` | `result[0][1]` is minimum energy |
| `test_empty_dict_returns_empty` | `{}` → `[]` |

#### `TestBestNSites`
| Test | What it asserts |
|------|----------------|
| `test_returns_n_sites` | Returns exactly `n` sites if available |
| `test_returns_fewer_if_not_enough` | < n total sites → returns all available |
| `test_most_stable_sites` | First site is globally most stable |
| `test_separation_constraint_enforced` | With `min_separation` set, all returned pairs have distance ≥ min_separation |
| `test_separation_without_coords_raises` | `min_separation` set but `site_coords=None` → `ValueError` |
| `test_no_sites_satisfy_constraint` | All sites within separation radius → returns 1 (just the best) |
| `test_empty_input_returns_empty` | `{}` → `[]` |

#### `TestSummariseNeb`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_ea_from_barrier_file` | `result['Ea']` matches `parse_barrier_file` output |
| `test_converged_flag` | `result['converged']` is bool |
| `test_missing_barrier_file_graceful` | Non-existent path → `Ea=None` or 0.0, no crash |
| `test_missing_path_file_graceful` | No path file → `rxn_coord` and `energies` are empty/None |
| `test_ts_index_from_path` | `result['ts_index']` corresponds to peak energy in path |

---

## Section 7 — `test_tst_rates.py`

### What it covers
`models/tst_rates.py`: ZPE correction, Vineyard prefactor, Arrhenius rate, `build_rate_dict` assembly, and result serialization.

### Test Classes

#### `TestApplyZpeCorrection`
| Test | What it asserts |
|------|----------------|
| `test_no_frequencies_returns_raw` | Empty freq lists → ZPE correction = 0, returns `E_barrier` |
| `test_known_correction` | Synthetic IS/TS frequencies → correction matches `0.5 * Σ(ν_TS - ν_IS) * CM1_TO_EV` |
| `test_below_threshold_excluded` | Freq below `min_freq_cm1=50` → excluded from sum |
| `test_threshold_default_50` | Frequency at 49 cm⁻¹ excluded by default; at 50 cm⁻¹ included |
| `test_lowers_barrier_when_is_stiffer` | Stiffer IS (higher ν_IS) → negative ΔE_zpe → corrected barrier lower |
| `test_raises_barrier_when_ts_stiffer` | Stiffer TS (higher ν_TS) → positive ΔE_zpe → corrected barrier higher |

#### `TestVineyardPrefactor`
| Test | What it asserts |
|------|----------------|
| `test_returns_float` | Returns `float` |
| `test_positive_output` | `ν* > 0` always |
| `test_order_of_magnitude` | `1e11 < ν* < 1e14` (typical solid-state range) |
| `test_identical_freqs_gives_one_ratio` | `freqs_is == freqs_ts` → `ν* = c` (ratio = 1) |
| `test_more_is_freqs_higher_nu` | Adding IS frequencies increases numerator → higher `ν*` |
| `test_empty_is_raises` | `freqs_is_cm1=[]` → `ValueError` |
| `test_empty_ts_raises` | `freqs_ts_cm1=[]` → `ValueError` |
| `test_all_below_threshold_raises` | All freqs < `min_freq_cm1` → `ValueError` |
| `test_threshold_filter_warning` | Some freqs below threshold → no crash, warning issued |

#### `TestArrheniusRate`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `k = ν * exp(-Ea / kB*T)` — hand-computed value matches |
| `test_zero_barrier` | `Ea=0` → `k = ν` |
| `test_high_barrier_near_zero` | Very high Ea → k ≈ 0 (underflow) |
| `test_temperature_dependence` | Higher T → higher k (for fixed Ea > 0) |
| `test_t_zero_raises` | `T_K=0` → `ValueError` |
| `test_negative_temperature_raises` | `T_K=-300` → `ValueError` |
| `test_positive_output` | `k >= 0` always |

#### `TestCollectNebResults`
| Test | What it asserts |
|------|----------------|
| `test_loads_valid_barrier_files` | N jobs with readable barrier files → dict with N entries |
| `test_skips_missing_barrier_file` | Job with non-existent barrier file → skipped, no crash |
| `test_label_format` | Labels are `'{hop}_{sid}'` (e.g., `'hopa_Ni3Mo'`) |
| `test_missing_sid_key_raises` | Job dict without `'sid'` → `KeyError` |
| `test_empty_job_list` | `neb_jobs=[]` → empty dict |

#### `TestSplitVibResults`
| Test | What it asserts |
|------|----------------|
| `test_is_and_ts_separated` | IS keys in `vib_is`, TS keys in `vib_ts` |
| `test_base_label_preserved` | `'hopa_Ni3Mo_IS'` → key `'hopa_Ni3Mo'` in output |
| `test_no_suffix_warns_and_skips` | Key without `_IS` or `_TS` → warning, skipped |
| `test_empty_input` | `{}` → `({}, {})` |

#### `TestBuildRateDict`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_all_expected_keys_per_label` | Each entry has `k_forward`, `k_reverse`, `Ea_raw`, `Ea_zpe`, `nu`, `T_K` |
| `test_k_forward_positive` | `k_forward > 0` for all labels |
| `test_k_forward_gt_k_reverse_when_ea_lt_edes` | When `Ea < E_des` → `k_fwd > k_rev` |
| `test_apply_zpe_false_skips_correction` | `apply_zpe=False` → `Ea_raw == Ea_zpe` |
| `test_missing_vib_label_skipped` | Label in NEB results but not vib_is → skipped with warning |
| `test_skip_count_correct` | All missing labels counted in `skipped` total |
| `test_missing_e_des_defaults_to_zero` | NEB result without `E_des` → `E_des=0.0` (not crash) |

#### `TestRatesToJson`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | JSON file exists |
| `test_loadable` | `json.load(f)` on output → dict |
| `test_all_labels_present` | All keys from input dict appear in JSON |
| `test_creates_parent_dirs` | Parent directory auto-created |
| `test_returns_path` | Returns path string of written file |

---

## Section 8 — `test_kmc.py`

### What it covers
`models/kmc.py`: grid creation, event enumeration, BKL stepper, full KMC run, steady-state convergence.

### Test Classes

#### `TestGasStrikeRate`
| Test | What it asserts |
|------|----------------|
| `test_positive_output` | `R > 0` for all positive inputs |
| `test_scales_with_pressure` | `R ∝ P` (linear) |
| `test_scales_with_area` | `R ∝ A_site` (linear) |
| `test_scales_with_temperature` | `R ∝ 1/√T` |
| `test_known_value` | Hand-computed Hertz-Knudsen at 300K, 1e5 Pa matches within 1% |
| `test_zero_pressure_gives_zero` | `P=0` → `R=0` |
| `test_negative_temperature_gives_nan_or_raises` | `T_K<=0` → unphysical (documented) |

#### `TestDrainRate`
| Test | What it asserts |
|------|----------------|
| `test_positive_output` | `k_drain > 0` for positive inputs |
| `test_formula_correct` | `k_drain == D / (a0/√2)²` hand computation |
| `test_scales_with_diffusivity` | `k_drain ∝ D` |
| `test_zero_a0_raises` | `a0_m=0` → `ZeroDivisionError` |
| `test_zero_diffusivity_gives_zero` | `D_m2s=0` → `k_drain=0` |

#### `TestMakeGrid`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_required_keys_present` | Keys: `surface_elem`, `surface_occ`, `subsurface_occ`, `nx`, `ny` |
| `test_surface_elem_shape` | `surface_elem.shape == (nx, ny)` |
| `test_surface_occ_all_zero_initially` | `surface_occ` all False/0 initially |
| `test_subsurface_occ_all_zero_initially` | `subsurface_occ` all False/0 initially |
| `test_composition_respected` | Pure-Ni composition → all sites == 'Ni' |
| `test_mixed_composition` | `{'Ni': 0.5, 'Mo': 0.5}` → roughly 50% each |
| `test_same_seed_reproducible` | Two calls with same seed → identical `surface_elem` |
| `test_different_seeds_differ` | Different seeds → different element placement |
| `test_composition_sums_to_one` | Fractional grid composition sums to 1.0 (within rounding) |
| `test_invalid_composition_raises` | Fractions not summing to 1 → `ValueError` |

#### `TestGridNeighbors`
| Test | What it asserts |
|------|----------------|
| `test_returns_four_neighbors` | Always returns exactly 4 tuples |
| `test_interior_site` | Site (3,3) on 10×10 grid → 4 adjacent interior sites |
| `test_corner_wraps` | Site (0,0) → wraps to (nx-1, 0), (0, ny-1), etc. |
| `test_edge_wraps` | Site (0, j) → top row wraps to (nx-1, j) |
| `test_all_in_bounds` | All returned coordinates 0 ≤ i < nx, 0 ≤ j < ny |
| `test_no_self_neighbor` | No returned tuple equals input `(i, j)` |

#### `TestGridMetrics`
Tests `surface_coverage()`, `subsurface_population()`, `subsurface_concentration()`.

| Test | What it asserts |
|------|----------------|
| `test_empty_surface_coverage_zero` | Empty grid → coverage = 0.0 |
| `test_full_surface_coverage_one` | All occupied → coverage = 1.0 |
| `test_half_coverage` | Exactly half sites occupied → coverage = 0.5 |
| `test_subsurface_population_zero_initially` | Fresh grid → `n_sub = 0` |
| `test_subsurface_population_count` | Set some subsurface_occ → count matches |
| `test_concentration_formula` | `C = n_sub / (nx * ny * a0³/√2)` |
| `test_concentration_zero_a0_raises` | `a0_m=0` → `ZeroDivisionError` |

#### `TestBuildEventList`
| Test | What it asserts |
|------|----------------|
| `test_returns_list` | Returns `list` |
| `test_event_dict_has_kind_sites_rate` | Each event has `'kind'`, `'sites'`, `'rate'` |
| `test_rates_nonnegative` | All `event['rate'] >= 0` |
| `test_adsorption_requires_empty_pair` | Adsorb events only for unoccupied pairs |
| `test_desorption_requires_occupied_pair` | Desorb events only for occupied pairs |
| `test_empty_grid_no_desorb_events` | Fresh empty grid → no desorb events |
| `test_full_grid_no_adsorb_events` | All sites occupied → no adsorb events |
| `test_missing_rate_dict_key_defaults_zero` | Missing key in rate_dict → rate=0, no crash |
| `test_drain_event_in_list` | Occupied subsurface → drain event present |
| `test_entry_event_in_list` | Occupied surface → entry event possible |

#### `TestKmcStep`
| Test | What it asserts |
|------|----------------|
| `test_returns_float_dt` | Returns `float >= 0` |
| `test_empty_events_returns_zero` | `events=[]` → `dt = 0.0` |
| `test_all_zero_rates_returns_zero` | All events with `rate=0` → `dt = 0.0` |
| `test_dt_positive_for_valid_events` | Positive rates → `dt > 0` |
| `test_grid_mutated` | Grid state changes after step |
| `test_adsorb_event_sets_occupancy` | An adsorb event at (i,j) sets `surface_occ[i,j] = True` |
| `test_desorb_event_clears_occupancy` | A desorb event clears `surface_occ[i,j]` |

#### `TestRunKmc`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict_with_arrays` | Keys: `t_arr`, `theta_arr`, `n_sub_arr` |
| `test_array_length` | Each array length = `n_steps + 1` |
| `test_t_arr_monotonic` | `t_arr` strictly non-decreasing |
| `test_initial_state_at_index_zero` | `theta_arr[0]` matches initial coverage before any steps |
| `test_coverage_bounded` | `0.0 <= theta_arr <= 1.0` always |
| `test_n_sub_nonnegative` | `n_sub_arr >= 0` always |
| `test_zero_steps` | `n_steps=0` → arrays length 1 (initial state only) |

#### `TestRunKmcToSteadyState`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_required_keys` | Keys: `t_total`, `theta_ss`, `C0`, `n_steps`, `converged` |
| `test_converges_simple_case` | Balanced adsorb/desorb rates → `converged=True` within max_steps |
| `test_converged_false_if_not_converged` | Tiny `max_steps=10` → `converged=False` (unless trivially fast) |
| `test_theta_ss_in_range` | `0.0 <= theta_ss <= 1.0` |
| `test_c0_nonnegative` | `C0 >= 0.0` |
| `test_n_steps_le_max_steps` | `result['n_steps'] <= max_steps` |

---

## Section 9 — `test_permeation.py`

### What it covers
`models/permeation.py`: flux computations, Sieverts law fit, solubility routes, permeability.

### Test Classes

#### `TestFickFlux`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `J = D * (C0 - C_low) / L` hand computation |
| `test_zero_length_raises` | `L_m=0` → `ValueError` |
| `test_zero_concentration_gradient` | `C0 == C_low` → `J = 0` |
| `test_negative_gradient_negative_flux` | `C0 < C_low` → negative flux (not an error) |
| `test_scales_with_diffusivity` | `J ∝ D` |

#### `TestSweepPressure`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_output_lists_same_length` | All output lists same length as `P_vals_Pa` |
| `test_empty_pressure_list_returns_empty` | `P_vals_Pa=[]` → all lists empty |
| `test_convergence_flag_present` | `result['converged']` list present |
| `test_j_nonnegative` | `J_vals[i] >= 0` for positive pressures |
| `test_higher_pressure_higher_flux` | Sorted P → monotonic J (in Sieverts regime) |
| `test_theta_bounded` | `0 <= theta_vals[i] <= 1` |

#### `TestCheckSievertsLaw`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_required_keys` | Keys: `slope`, `intercept`, `r_squared`, `is_sieverts` |
| `test_perfect_sieverts_r2_near_one` | `J = k * √P` data → `R2 ≥ 0.98`, `is_sieverts=True` |
| `test_non_sieverts_r2_low` | `J = k * P` (surface limited) → `R2 < 0.98`, `is_sieverts=False` |
| `test_single_point_raises_or_returns` | < 2 points → handled gracefully |
| `test_slope_positive` | Positive pressures → positive slope |

#### `TestArrheniusDiffusivity`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `D0 * exp(-ED / kB*T)` matches |
| `test_t_zero_raises` | `T_K=0` → `ValueError` |
| `test_negative_t_raises` | `T_K < 0` → `ValueError` |
| `test_ea_zero_gives_d0` | `E_D=0` → `D = D0` |
| `test_decreases_with_ea` | Higher E_D → lower D |

#### `TestLatticeS0`
| Test | What it asserts |
|------|----------------|
| `test_formula_correct` | `S0 = 4 / a0³` (4 oct sites per FCC cell) |
| `test_a0_zero_raises` | `a0_m=0` → `ZeroDivisionError` |
| `test_positive_result` | `S0 > 0` for positive `a0_m` |

#### `TestSolubilityFromRates`
| Test | What it asserts |
|------|----------------|
| `test_returns_positive_float` | `S(T) > 0` for physical inputs |
| `test_t_zero_raises` | `T_K=0` → `ValueError` |
| `test_k_des_zero_raises` | `k_des_s1=0` → `ValueError` |
| `test_k_exit_zero_raises` | `k_exit_s1=0` → `ValueError` |
| `test_temperature_dependence` | Higher T → different S (depends on `dH_sol`) |

#### `TestFitSolubilityFromKmc`
| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` with `S_vals`, `S_mean`, `S_std`, `n_converged` |
| `test_no_converged_gives_zero_mean` | `converged=False` for all → `S_mean=0.0`, `n_converged=0` |
| `test_zero_pressure_excluded` | `P=0` entries → excluded from S calculation |
| `test_s_mean_positive` | For physical KMC sweep → `S_mean > 0` |
| `test_single_converged_point_std_zero` | 1 converged point → `S_std=0.0` |

#### `TestSievertsSolubility`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `S = S0 * exp(-dH / kB*T)` matches |
| `test_t_zero_raises` | `T_K=0` → `ValueError` |
| `test_positive_s0_positive_output` | `S > 0` for positive `S0` and any T |

#### `TestPermeability`
| Test | What it asserts |
|------|----------------|
| `test_formula` | `Phi = D * S` |
| `test_zero_d_gives_zero_phi` | `D=0` → `Phi=0` |
| `test_zero_s_gives_zero_phi` | `S=0` → `Phi=0` |

#### `TestRichardsonFlux`
| Test | What it asserts |
|------|----------------|
| `test_known_value` | `J = (Phi/L) * (√P_high - √P_low)` hand computation |
| `test_zero_length_raises` | `L_m=0` → `ValueError` |
| `test_equal_pressures_zero_flux` | `P_high == P_low` → `J = 0` |
| `test_negative_p_low_clamped` | `P_low < 0` → treated as 0 (based on docs) |

---

## Section 10 — `test_diffusivity_workflow.py`

### What it covers
`models/diffusivity_workflow.py`: `generate_diffusivity_scripts()` and `generate_orchestrator_sh()`. Tests inspect content of the generated Python/shell scripts.

### Approach
Call `generate_diffusivity_scripts(...)` with a full minimal config, read the output `diffusivity_run.py` as a string, inspect it. Extends the `TestDiffusivityWorkflowBody` class already in `test_nvt_restart.py` (that class tests the `_body` raw string; this file tests the whole generated output and `generate_orchestrator_sh`).

### Test Classes

#### `TestGenerateDiffusivityScripts`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | `diffusivity_run.py` exists on disk |
| `test_returns_path_string` | Return value is path to written file |
| `test_input_structures_embedded` | `INPUT_STRUCTURES` appears with correct value |
| `test_temperatures_embedded` | `TEMPERATURES` list matches input |
| `test_n_h_values_embedded` | `N_H_VALUES` list matches input |
| `test_work_dir_embedded` | `WORK_DIR` matches input |
| `test_nvt_wall_time_embedded` | `NVT_WALL_TIME` matches input |
| `test_cutoff_embedded` | `CUTOFF` matches input |
| `test_gpu_partition_embedded` | `GPU_PARTITION` matches input |
| `test_short_gpu_partition_embedded` | `SHORT_GPU_PARTITION` matches input |
| `test_short_gpu_cutoff_auto_derived` | If `short_gpu_cutoff=None`, auto-derived = `short_gpu_time - 5 min` |
| `test_timestep_embedded` | `TIMESTEP_PS` matches input |
| `test_n_equil_steps_embedded` | `N_EQUIL_STEPS` matches input |
| `test_n_prod_steps_embedded` | `N_PROD_STEPS` matches input |
| `test_phase1a_guard_present` | `if not os.path.exists(min_bare_out)` in body |
| `test_phase1b_npt_guard_present` | NPT guard uses per-T output path |
| `test_phase2_sentinel_guard_present` | Phase 2 checks `.done` sentinel (not msd file) |
| `test_nvt_uses_multigpu_partition` | NVT job config uses `GPU_PARTITION` (multigpu) |
| `test_npt_uses_short_gpu_partition` | NPT job config uses `SHORT_GPU_SLURM_CFG` |
| `test_get_lattice_parameter_from_dump_imported` | `get_lattice_parameter_from_dump` imported (Task G) |
| `test_dead_single_frame_import_absent` | `get_lattice_parameter` (old) NOT used in Phase 1b-B |
| `test_write_nvt_equil_restart_imported` | `write_nvt_equil_restart_script` imported |
| `test_write_nvt_prod_restart_imported` | `write_nvt_prod_restart_script` imported |
| `test_write_npt_restart_imported` | `write_npt_restart_script` imported (Task A3) |
| `test_chained_job_for_npt` | NPT uses `write_chained_slurm_job` (not `write_slurm_job`) |
| `test_chained_job_for_nvt` | NVT uses `write_chained_slurm_job` |
| `test_n_equil_passed_to_nvt_chain` | `n_equil=N_EQUIL_STEPS` in NVT chain call |
| `test_phase1a_uses_wait_for_jobs` | Bare minimization waits via `wait_for_jobs` |
| `test_phase1b_npt_waits` | NPT uses `wait_for_jobs(npt_job_ids)` |
| `test_phase2_polls_done_sentinel` | Phase 2 polls `.done` files (not job IDs) |
| `test_phase3_arrhenius_present` | Phase 3 fits Arrhenius in body |
| `test_metal_table_embedded_when_provided` | `metal_table` dict serialized into script |
| `test_short_gpu_cutoff_positive` | `short_gpu_time - 300s > 0` → cutoff is valid |

#### `TestGenerateOrchestratorSh`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | `diffusivity_run.sh` exists |
| `test_executable` | `st_mode & stat.S_IXUSR` |
| `test_sbatch_header` | `#SBATCH` lines present |
| `test_job_name_in_header` | `--job-name` matches input |
| `test_partition_in_header` | `--partition` matches input |
| `test_cpus_per_task_in_header` | `--cpus-per-task` matches input |
| `test_time_in_header_when_provided` | `--time` present when `orch_time` non-empty |
| `test_no_time_in_header_when_empty` | `orch_time=''` → no `--time` line |
| `test_module_load_openmpi` | `module load OpenMPI/` line present |
| `test_module_load_cuda` | `module load cuda/` line present |
| `test_conda_activate` | `conda activate {env}` present |
| `test_cd_work_dir` | `cd {work_dir}` present |
| `test_python_call` | `python {out_py}` present |
| `test_ld_paths_exported` | Each `ld_paths` entry in `LD_LIBRARY_PATH` export |

---

## Section 11 — `test_permeation_workflow.py`

### What it covers
`models/permeation_workflow.py`: `generate_permeation_scripts()`, `generate_permeation_sh()`, and analysis helpers `load_barrier_summary()`, `load_rate_summary()`.

### Test Classes

#### `TestGeneratePermeationScripts`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | `permeation_run.py` exists |
| `test_temperatures_embedded` | `TEMPERATURES` list in generated script |
| `test_p_vals_embedded` | `P_VALS_PA` list in generated script |
| `test_a0_m_embedded` | `A0_M` value in generated script |
| `test_d0_m2s_embedded` | `D0_M2S` value |
| `test_e_d_ev_embedded` | `E_D_EV` value |
| `test_dh_diss_none_handled` | `dh_diss_ev=None` → `DH_DISS_EV = None` in script, Phase 6 skipped with warning |
| `test_dh_entry_none_handled` | Same for `dh_entry_ev=None` |
| `test_work_dir_embedded` | `WORK_DIR` in script |
| `test_surface_sites_json_embedded` | `SURFACE_SITES_JSON` in script |
| `test_relaxed_slab_path_embedded` | `RELAXED_SLAB_PATH` in script |
| `test_kmc_max_steps_embedded` | `KMC_MAX_STEPS` in script |
| `test_seed_embedded` | `SEED` in script |
| `test_phase6_guard_on_dh` | Phase 6 wrapped in `if DH_DISS_EV is not None and DH_ENTRY_EV is not None` |
| `test_hop_a_guard_present` | Guard checks `hopa_jobs.json` existence |
| `test_hop_b_guard_present` | Guard checks `hopb_jobs.json` existence |
| `test_kmc_sweep_guard_per_temperature` | Per-T guard on `permeation_sweep_T{T}K.json` |

#### `TestGeneratePermeationSh`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | `.sh` exists |
| `test_executable` | `st_mode & stat.S_IXUSR` |
| `test_sbatch_headers` | Standard `#SBATCH` lines present |
| `test_python_call` | `python {out_py}` in script |
| `test_conda_activate` | `conda activate` line present |

#### `TestLoadBarrierSummary`
Uses synthetic `hopa_jobs.json` and `hopb_jobs.json` written to `tmp_path`.

| Test | What it asserts |
|------|----------------|
| `test_returns_dataframe` | Returns `pd.DataFrame` |
| `test_required_columns` | Columns: `E_abs`, `E_des`, `delta_E`, `converged`, `sid`, `hop` |
| `test_hop_column_values` | `hop` column contains `'hopa'` and `'hopb'` |
| `test_missing_barrier_file_skipped` | Job with non-existent barrier file → skipped, no crash |
| `test_empty_json_files_returns_empty_df` | Empty job lists → empty DataFrame |
| `test_missing_hopa_json_returns_hopb_only` | Missing hopa JSON → only hopb data |

#### `TestLoadRateSummary`
| Test | What it asserts |
|------|----------------|
| `test_returns_dataframe` | Returns `pd.DataFrame` |
| `test_t_k_column` | `T_K` column matches input temperatures |
| `test_missing_json_for_temp_skipped` | Missing rate JSON for one T → that T absent, no crash |
| `test_k_forward_present` | `k_forward` column in output |

---

## Section 12 — `test_pipeline_workflow.py`

### What it covers
`models/pipeline_workflow.py`: `generate_pipeline_scripts()` and `generate_pipeline_sh()`.

### Test Classes

#### `TestGeneratePipelineScripts`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | `pipeline_run.py` exists |
| `test_neb_scripts_launched_in_parallel` | `subprocess.Popen` used for NEB scripts (parallel) |
| `test_diffusivity_launched_in_parallel` | `diffusivity_run.py` also via `Popen` (parallel with NEB) |
| `test_all_metal_neb_scripts_present` | Each `stem`'s `neb_run_py` path appears in script |
| `test_wait_before_permeation` | `proc.wait()` or equivalent called before Part 2 starts |
| `test_failure_check_before_permeation` | Non-zero return code check present before permeation |
| `test_permeation_sequential` | `permeation_run.py` scripts run one at a time (not Popen) |
| `test_permeation_failure_exits` | Per-permeation failure → exit 1 |
| `test_all_permeation_scripts_present` | Each `stem`'s `permeation_run_py` appears in script |
| `test_empty_metals_list` | `metals=[]` → script still valid, loop runs zero iterations |
| `test_file_is_valid_python` | `compile(source, ...)` → no `SyntaxError` |

#### `TestGeneratePipelineSh`
| Test | What it asserts |
|------|----------------|
| `test_file_created` | `.sh` exists |
| `test_executable` | `st_mode & stat.S_IXUSR` |
| `test_sbatch_headers` | `#SBATCH` lines present |
| `test_python_call` | `python {pipeline_run.py}` present |

---

## Section 13 — `test_neb_workflow.py`

### What it covers
`models/neb_workflow.py`: `build_phase1_slab()`, `orchestrate_slab_prep()` (phase chaining and checkpoint detection), and `write_neb_run_script()` (generated script content). Tests run offline with `dry_run=True` and synthetic input files.

### Test Classes

#### `TestBuildPhase1Slab`
| Test | What it asserts |
|------|----------------|
| `test_returns_tuple` | Returns `(str, float)` |
| `test_slab_file_created` | Output path exists on disk |
| `test_a0_positive` | `a0 > 0` |
| `test_metal_type_alloy_default` | Default `metal_type='alloy'` → Ni FCC template used |
| `test_metal_type_pure_routes_correctly` | `metal_type='pure'` → calls `build_slab` with `metal_type='pure'` |
| `test_metal_type_oxide_routes_correctly` | `metal_type='oxide'` → routes to oxide branch |
| `test_slab_seed_passed_through` | `slab_seed=42` → `seed=42` passed to `build_slab` |

#### `TestOrchestrateSlabrep`
Tests `orchestrate_slab_prep()` with `dry_run=True`.

| Test | What it asserts |
|------|----------------|
| `test_returns_dict` | Returns `dict` |
| `test_required_keys` | Keys: `phase1_slab`, `phase2_relaxed`, `phase3_sites`, `n_sites`, `outdir`, `status` |
| `test_phase1_slab_created` | `result['phase1_slab']` file exists on disk |
| `test_phase2_scripts_created` | LAMMPS `.in` file and SLURM `.sh` exist (dry_run=True) |
| `test_dry_run_does_not_submit` | No SLURM submission when `dry_run=True` |
| `test_checkpoint_detection_skips_phase2` | Pre-create `relaxed_slab.lammps` → Phase 2 skipped |
| `test_output_dirs_created` | `phase1_slab/`, `phase2_relax/`, `phase3_sites/` subdirs exist |
| `test_metal_type_threaded` | `metal_type='pure'` passed through to `build_phase1_slab` |
| `test_slab_seed_threaded` | `slab_seed=99` appears in phase 3 ACAT enumeration call |

#### `TestWriteNebRunScript`
Inspects generated `neb_run.py` content.

| Test | What it asserts |
|------|----------------|
| `test_file_created` | `out_py` exists |
| `test_bulk_min_path_embedded` | `BULK_MIN_PATH = {path!r}` in script |
| `test_work_dir_embedded` | `WORK_DIR = {path!r}` |
| `test_e_h2_gas_embedded` | `E_H2_GAS = {value}` |
| `test_miller_embedded` | `MILLER = (h, k, l)` |
| `test_layers_embedded` | `LAYERS = {n}` |
| `test_slab_seed_embedded` | `SLAB_SEED = {seed}` |
| `test_metal_type_embedded` | `METAL_TYPE = {type!r}` |
| `test_n_images_embedded` | `N_IMAGES = {n}` |
| `test_spring_const_embedded` | `SPRING_CONST = {k}` |
| `test_gpu_slurm_cfg_embedded` | `GPU_SLURM_CFG` dict in script |
| `test_neb_slurm_cfg_embedded` | `NEB_SLURM_CFG` dict in script |
| `test_vib_slurm_defaults_to_neb_slurm` | `vib_slurm_cfg=None` → `VIB_SLURM_CFG` = copy of neb_slurm_cfg |
| `test_phases_a_to_e_present` | Comments or markers for all 5 phases |
| `test_phase_a_to_d_guard_present` | `if not os.path.exists(_ranked_f)` wraps Phases A-D |
| `test_phase_e_separate_guard` | Phase E has its own existence check |
| `test_orchestrate_full_neb_called` | `orchestrate_full_neb_workflow(` in script |
| `test_slab_seed_passed_to_orchestrate` | `slab_seed=SLAB_SEED` in `orchestrate_full_neb_workflow` call |
| `test_metal_type_passed_to_orchestrate` | `metal_type=METAL_TYPE` in call |
| `test_auto_submit_in_phase_d` | `auto_submit(` present for FS-min array |
| `test_neb_array_submission_present` | `submit_slurm_job(result['neb_array_script'])` present |
| `test_phase_e_loads_ranked_barriers` | `ranked_barriers.json` loaded in Phase E |
| `test_phase_e_vibrations_called` | `orchestrate_vibrations` called in Phase E |
| `test_phase_e_build_rate_dict_called` | `build_rate_dict(` called in Phase E |
| `test_diss_rates_json_saved` | `diss_vib_rates.json` written |
| `test_valid_python_syntax` | `compile(source, ...)` → no `SyntaxError` |
| `test_elem_str_embedded` | `ELEM_STR` in script |
| `test_e2t_embedded` | `E2T` dict in script |
| `test_masses_embedded` | `MASSES` dict in script |

---

## Execution Instructions

For each section:
```bash
cd /Users/akinyemi.az/Desktop/PhD_Folder/research/MHI/MD/Molecular_Dynamics/MHI_Nickel
python -m pytest tests/test_{section}.py -v
```

All tests in a section must pass before moving to the next.

**Run the existing NVT test to confirm baseline:**
```bash
python -m pytest tests/test_nvt_restart.py -v
```

---

## Summary

| Section | Test File | Est. Test Count |
|---------|-----------|----------------|
| 1 | `test_parsers.py` | ~55 |
| 2 | `test_structure.py` | ~35 |
| 3 | `test_create_slurm.py` | ~35 |
| 4 | `test_lammps_scripts.py` | ~45 |
| 5 | `test_diffusivity_post_processing.py` | ~50 |
| 6 | `test_energetics.py` | ~35 |
| 7 | `test_tst_rates.py` | ~40 |
| 8 | `test_kmc.py` | ~55 |
| 9 | `test_permeation.py` | ~40 |
| 10 | `test_diffusivity_workflow.py` | ~40 |
| 11 | `test_permeation_workflow.py` | ~30 |
| 12 | `test_pipeline_workflow.py` | ~15 |
| 13 | `test_neb_workflow.py` | ~35 |
| **Total** | **13 files** | **~510 tests** |

"""
tests/functional/test_ft_script_generation.py
===============================================
Category A functional tests — script generation (dry_run / pure writers).

Verifies that all SLURM configuration changes (partition='sharing',
time='00:20:00') and the NEB trajectory format fix (.extxyz not .lammpstrj)
are correctly embedded in generated scripts.

No LAMMPS, no SLURM, no GPU, no cluster files required.
Functions that are pure script writers are called directly with dummy
cluster paths; source-code regression guards catch accidental reverts.
"""

import math
import os
import sys
import pathlib

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import MagicMock

# acat, surface_graph, and matplotlib are not available in the offline test env
# (matplotlib compiled against NumPy 1.x; incompatible with NumPy 2.x at import time)
for _m in ('acat', 'acat.adsorption_sites'):
    sys.modules.setdefault(_m, MagicMock())
sys.modules.setdefault('models.surface_graph', MagicMock())
for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.patches',
           'matplotlib.cm', 'matplotlib.ticker', 'matplotlib.transforms',
           'matplotlib.colors', 'matplotlib.scale', 'matplotlib._path',
           'matplotlib._api', 'matplotlib.cbook', 'matplotlib.rcsetup'):
    sys.modules.setdefault(_m, MagicMock())

from models.ase_neb import write_ase_neb_script, run_neb_pipeline
from models.diffusivity_workflow import generate_diffusivity_scripts
from models.permeation_workflow import generate_permeation_scripts


# ─── shared dummy config ──────────────────────────────────────────────────────

_DUMMY = dict(
    is_file='/cluster/neb/is.lammps',
    fs_file='/cluster/neb/fs.lammps',
    e_is=-2271.0,
    e_fs=-2271.06,          # required: either e_fs or fs_log_file must be provided
    mace_model_path='/cluster/models/mace.model',
    barrier_file='/cluster/neb/neb_barrier.txt',
    path_file='/cluster/neb/neb_path.dat',
)

_DIFF_CFG = dict(
    input_structures=['/work/Ni_bulk.lammps'],
    n_h_values=[1],
    temperatures=[600, 700, 800],
    work_dir='/work/diffusivity',
    nvt_wall_time='48:00:00',
    cutoff=6.0,
    gpu_partition='gpu_p100',
    gpu_time='48:00:00',
    timestep_ps=0.0005,
    tau_t_ps=0.1,
    n_equil_steps=50000,
    n_prod_steps=500000,
    thermo_every=100,
    dump_every=200,
    velocity_seed=42,
    restart_every=10000,
)

_PERM_CFG = dict(
    work_dir='/work/permeation',
    stem='test_metal',
    relaxed_slab_path='/work/relaxed_slab.lammps',
    surface_sites_json='/work/surface_sites.json',
    phase2_h_dir='/work/phase2_h',
    sub_neb_dir='/work/sub_neb',
    vib_dir='/work/vib',
    results_dir='/work/results',
    temperatures=[600, 700, 800],
    n_h_values=[1, 3, 5, 10],
    p_vals_pa=[1e3, 1e4, 1e5],
    a0_m=3.52e-10,
    l_m=1e-3,
    dh_diss_ev=-0.5,
    dh_entry_ev=0.2,
    nx=5,
    ny=5,
    seed=42,
    kmc_max_steps=1_000_000,
    gpu_slurm_cfg={'partition': 'sharing', 'time': '00:20:00'},
    neb_slurm_cfg={'partition': 'short',   'time': '12:00:00'},
    vib_slurm_cfg={'partition': 'short',   'time': '06:00:00'},
    n_images=18,
    spring_const=1.0,
    neb_ftol=0.05,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. ASE NEB script — trajectory format fix
# ═════════════════════════════════════════════════════════════════════════════

class TestNebScriptTrajFormat:
    """
    Verify that write_ase_neb_script produces a run_neb.py that uses .extxyz
    (not .lammpstrj) for trajectory visualisation exports, and that the read/write
    format is "extxyz" (not "lammps-dump-text").

    The bug was that ASE's lammps-dump-text writer does not exist; calling it
    raised ValueError on the cluster after Phase 1 completed.  The fix replaces
    both the extension and the format string.
    """

    @pytest.fixture()
    def script_content(self, tmp_path):
        out = str(tmp_path / 'run_neb.py')
        write_ase_neb_script(
            **_DUMMY,
            logfile_phase1=str(tmp_path / 'neb_phase1.log'),
            logfile_phase2=str(tmp_path / 'neb_phase2.log'),
            traj_phase1=str(tmp_path / 'neb_phase1.traj'),
            traj_phase2=str(tmp_path / 'neb_phase2.traj'),
            out_path=out,
        )
        return pathlib.Path(out).read_text()

    def test_script_file_created(self, tmp_path):
        out = str(tmp_path / 'run_neb.py')
        write_ase_neb_script(
            **_DUMMY,
            logfile_phase1=str(tmp_path / 'p1.log'),
            logfile_phase2=str(tmp_path / 'p2.log'),
            out_path=out,
        )
        assert pathlib.Path(out).exists()

    def test_traj_phase1_uses_dot_traj(self, script_content):
        # TRAJ_PHASE1 variable embedded in script must end with .traj
        assert 'neb_phase1.traj' in script_content

    def test_traj_phase2_uses_dot_traj(self, script_content):
        assert 'neb_phase2.traj' in script_content

    def test_extxyz_extension_used_not_lammpstrj(self, script_content):
        # The visualisation export replaces .traj with .extxyz
        assert '.extxyz' in script_content
        assert '.lammpstrj' not in script_content

    def test_extxyz_format_string_present(self, script_content):
        # write() must be called with format="extxyz"
        assert 'format="extxyz"' in script_content

    def test_lammps_dump_text_format_absent(self, script_content):
        # The old broken format must not appear in any generated script
        assert 'lammps-dump-text' not in script_content

    def test_run_neb_pipeline_also_uses_extxyz(self, tmp_path):
        # run_neb_pipeline is the high-level entry point; verify same contract
        script_path = run_neb_pipeline(
            **_DUMMY,
            outdir=str(tmp_path),
            traj_phase1=str(tmp_path / 'neb_phase1.traj'),
            traj_phase2=str(tmp_path / 'neb_phase2.traj'),
        )
        content = pathlib.Path(script_path).read_text()
        assert '.extxyz' in content
        assert '.lammpstrj' not in content
        assert 'lammps-dump-text' not in content

    def test_source_ase_neb_no_lammpstrj(self):
        src = pathlib.Path(PROJECT_ROOT, 'models', 'ase_neb.py').read_text()
        assert '.lammpstrj' not in src, (
            "models/ase_neb.py still contains '.lammpstrj' — "
            "the traj format fix may have been reverted"
        )

    def test_source_ase_neb_no_lammps_dump_text(self):
        src = pathlib.Path(PROJECT_ROOT, 'models', 'ase_neb.py').read_text()
        assert 'lammps-dump-text' not in src, (
            "models/ase_neb.py still contains 'lammps-dump-text' — "
            "the format= fix may have been reverted"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Diffusivity script — sharing partition defaults
# ═════════════════════════════════════════════════════════════════════════════

class TestDiffusivityScriptSharingDefaults:
    """
    generate_diffusivity_scripts default parameters were changed to
    short_gpu_partition='sharing' and short_gpu_time='01:00:00' (raised
    from '00:20:00' to match what's actually deployed on the cluster and
    give bare-bulk/bulk+H minimization -- now self-resubmitting via
    write_chained_slurm_job -- more headroom before its first timeout leg).

    Verify these values appear correctly in the generated diffusivity_run.py.
    """

    @pytest.fixture()
    def content_defaults(self, tmp_path):
        out = str(tmp_path / 'diffusivity_run.py')
        generate_diffusivity_scripts(**_DIFF_CFG, out_py=out)
        return pathlib.Path(out).read_text()

    def test_file_created(self, tmp_path):
        out = str(tmp_path / 'diffusivity_run.py')
        generate_diffusivity_scripts(**_DIFF_CFG, out_py=out)
        assert pathlib.Path(out).exists()

    def test_default_short_gpu_partition_is_sharing(self, content_defaults):
        # Default: short_gpu_partition='sharing' → embedded as SHORT_GPU_PARTITION = 'sharing'
        assert "SHORT_GPU_PARTITION  = 'sharing'" in content_defaults

    def test_default_short_gpu_time_is_01_00_00(self, content_defaults):
        assert "SHORT_GPU_TIME       = '01:00:00'" in content_defaults

    def test_custom_short_gpu_partition_overrides_default(self, tmp_path):
        out = str(tmp_path / 'diff_custom.py')
        generate_diffusivity_scripts(
            **_DIFF_CFG, out_py=out,
            short_gpu_partition='multigpu',
            short_gpu_time='04:00:00',
        )
        content = pathlib.Path(out).read_text()
        assert "SHORT_GPU_PARTITION  = 'multigpu'" in content
        assert "SHORT_GPU_TIME       = '04:00:00'" in content

    def test_phase_labels_present(self, content_defaults):
        for label in ('Phase 1a', 'Phase 1b', 'Phase 2', 'Phase 3'):
            assert label in content_defaults, f"'{label}' missing from diffusivity script"

    def test_source_diffusivity_default_is_sharing(self):
        src = pathlib.Path(PROJECT_ROOT, 'models', 'diffusivity_workflow.py').read_text()
        assert "short_gpu_partition='sharing'" in src, (
            "generate_diffusivity_scripts default short_gpu_partition is not 'sharing' — "
            "the SLURM config change may have been reverted"
        )
        assert "short_gpu_time='01:00:00'" in src, (
            "generate_diffusivity_scripts default short_gpu_time is not '01:00:00'"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Permeation script — partition embedded from caller config
# ═════════════════════════════════════════════════════════════════════════════

class TestPermeationScriptSharingConfig:
    """
    generate_permeation_scripts takes gpu_slurm_cfg as a required parameter.
    The notebooks (and our changes) use partition='sharing'.
    Verify the config is correctly embedded in the generated permeation_run.py.
    """

    @pytest.fixture()
    def content(self, tmp_path):
        out = str(tmp_path / 'permeation_run.py')
        generate_permeation_scripts(**_PERM_CFG, out_py=out)
        return pathlib.Path(out).read_text()

    def test_file_created(self, tmp_path):
        out = str(tmp_path / 'permeation_run.py')
        generate_permeation_scripts(**_PERM_CFG, out_py=out)
        assert pathlib.Path(out).exists()

    def test_gpu_slurm_cfg_sharing_embedded(self, content):
        # GPU_SLURM_CFG = {'partition': 'sharing', 'time': '00:20:00'}
        assert "'partition': 'sharing'" in content

    def test_gpu_slurm_cfg_time_embedded(self, content):
        assert "'time': '00:20:00'" in content

    def test_all_six_phase_labels_present(self, content):
        for label in ('Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6'):
            assert label in content, f"'{label}' missing from permeation script"

    def test_temperatures_embedded(self, content):
        assert '600' in content and '700' in content and '800' in content

    def test_richardson_flux_referenced(self, content):
        assert 'richardson_flux' in content


# ═════════════════════════════════════════════════════════════════════════════
# 4. Source regression guards — SLURM defaults across models
# ═════════════════════════════════════════════════════════════════════════════

class TestSlurmDefaultRegressionGuards:
    """
    Verify the SLURM defaults are still correct in the model source files.

    Quick one-shot minimizations (H2*/H* adsorption, FS-min before NEB) stay
    on partition='sharing', time='01:00:00'. Slab surface relaxation
    (run_phase2_surface_relaxation) is real chained MD -- same category as
    diffusivity's NPT -- and defaults to partition='gpu', time='08:00:00'.

    These guards catch accidental reverts or merge conflicts that restore the
    old partition='multigpu' default, or the pre-fix partition='sharing'
    default for surface relaxation specifically.
    """

    def _src(self, filename):
        return pathlib.Path(PROJECT_ROOT, 'models', filename).read_text()

    # ── neb_workflow.py ──────────────────────────────────────────────────────

    def test_neb_workflow_h2_adsorption_default_is_sharing(self):
        src = self._src('neb_workflow.py')
        # run_phase1_h2_adsorption default slurm_opts
        assert "'partition': 'sharing'" in src, (
            "neb_workflow.py still contains non-sharing partition for H2 adsorption"
        )

    def test_neb_workflow_phase2_default_is_gpu_08_00(self):
        """run_phase2_surface_relaxation (slab relaxation -- real chained
        heat/NVT/quench MD) defaults to partition='gpu', time='08:00:00',
        the same category as diffusivity's NPT -- not the quick-minimization
        'sharing' default used by H2*/H* adsorption and FS-min."""
        src = self._src('neb_workflow.py')
        assert "{**SLURM_DEFAULTS, 'partition': 'gpu', 'time': '08:00:00'}" in src

    def test_neb_workflow_min_functions_default_time_is_01_00(self):
        src = self._src('neb_workflow.py')
        assert "{**SLURM_DEFAULTS, 'partition': 'sharing', 'time': '01:00:00'}" in src

    def test_neb_workflow_no_multigpu_as_default(self):
        src = self._src('neb_workflow.py')
        # 'multigpu' should not appear as a hard-coded default slurm_opts value.
        # It may appear in comments or string literals inside if-branches, but
        # not in the pattern "'partition': 'multigpu'" (which was the old default).
        assert "'partition': 'multigpu'" not in src, (
            "neb_workflow.py has 'multigpu' as a hard-coded partition default — "
            "check the slurm_opts None-branch defaults"
        )

    # ── neb_subsurface.py ────────────────────────────────────────────────────

    def test_neb_subsurface_hopa_default_is_sharing(self):
        src = self._src('neb_subsurface.py')
        assert "'partition': 'sharing'" in src, (
            "neb_subsurface.py orchestrate_hopa_neb default partition is not 'sharing'"
        )

    def test_neb_subsurface_no_multigpu_as_default(self):
        src = self._src('neb_subsurface.py')
        assert "'partition': 'multigpu'" not in src

    # ── parsers.py — barrier file unit/case fixes (cross-check) ─────────────

    def test_parsers_barrier_file_uses_split_for_units(self):
        src = self._src('parsers.py')
        assert 'float(val.split()[0])' in src, (
            "parsers.py parse_barrier_file no longer uses val.split()[0] — "
            "the unit-suffix fix may have been reverted"
        )

    def test_parsers_barrier_file_case_insensitive_converged(self):
        src = self._src('parsers.py')
        assert "key.lower() == 'converged'" in src, (
            "parsers.py parse_barrier_file no longer uses key.lower() — "
            "the 'Converged' case-insensitive fix may have been reverted"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. .wrap() regression guards — every lammps-data read must be followed by .wrap()
# ═════════════════════════════════════════════════════════════════════════════

def _lines_after_lammps_read(src_text):
    """
    Return a list of (read_lineno, next_10_lines) tuples for every line in
    src_text that is a lammps-data READ call (not a write call).  Used to
    check that .wrap() (or an equivalent periodic-consistent realignment,
    e.g. find_mic(...)-based) appears within the lines immediately after.

    Write calls use `write(`, `_ase_write(`, or `ase_write(` and must be
    excluded — they do not need .wrap() and do not unwrap image flags.

    Window is 13 lines (not 2) to accommodate ase_neb.py's FS read (followed by
    a multi-line explanatory comment before its find_mic(...) realignment call)
    and neb_subsurface.py's classification read, whose classify_site(...)
    consumer applies its own minimum-image PBC a few lines down (see
    TestWrapRegressionGuards docstring).
    """
    lines = src_text.splitlines()
    result = []
    for i, line in enumerate(lines):
        if "format='lammps-data'" not in line and 'format="lammps-data"' not in line:
            continue
        # Exclude write calls (ase_write, _ase_write, write — anything ending in write)
        stripped = line.lstrip()
        if stripped.startswith('_ase_write') or stripped.startswith('ase_write'):
            continue
        # Generic write: if the function call token right before '(' is 'write'
        if 'write(' in line and 'read(' not in line and 'ase_read(' not in line:
            continue
        following = lines[i + 1 : i + 14]
        result.append((i + 1, following))
    return result


class TestWrapRegressionGuards:
    """
    Bug 2 fix: every `read(format='lammps-data')` call must be immediately
    followed by `.wrap()` — or an equivalent periodic-consistent realignment
    — within the next source lines.

    If any of these guards fails it means the wrap fix was reverted (e.g. via
    a merge conflict, regenerated .py, or accidental revert), which would
    silently produce atoms outside the box and corrupt site coordinates.

    Exception: permeation_workflow.py:85 — the read is inside a generated
    script body (template string) and only uses get_cell(); wrapping has no
    effect on cell dimensions so it is intentionally omitted.

    Second accepted pattern: ase_neb.py's FS read is followed by a
    find_mic(...)-based realignment to IS's frame instead of a bare
    .wrap() — independently wrapping IS and FS let an atom whose real
    relaxation crosses a cell boundary in one endpoint but not the other
    get mapped to opposite sides of the cell, which NEB.interpolate() (raw
    Cartesian, no periodic awareness) would treat as a literal ~cell-length
    path to traverse (confirmed on a real pair: metal atoms with ~12.6 Å
    naive displacement vs ~0.01 Å true displacement). find_mic(...)
    guarantees the same "atoms end up in a well-defined, bounded frame"
    property .wrap() does, just relative to IS instead of the origin.

    Third accepted pattern: neb_subsurface.py's classify_relaxed_h_env reads a
    relaxed FS purely to CLASSIFY where H sits (no NEB interpolation, no IS to
    anchor to). Its consumer classify_site(...) -> _find_coordinating_atoms
    applies its own xy minimum-image PBC (subsurface_graph.py), so the atoms
    object is deliberately left unwrapped -- consistent with the FS-not-bare-
    wrapped design. classify_site( is therefore treated like find_mic( (a
    consumer that handles periodicity) when scanning the post-read window.
    """

    def _src(self, filename):
        return pathlib.Path(PROJECT_ROOT, 'models', filename).read_text()

    def _assert_all_reads_wrapped(self, filename):
        src = self._src(filename)
        reads = _lines_after_lammps_read(src)
        assert reads, f"No lammps-data reads found in {filename} — guards may be stale"
        unwrapped = [
            lineno for lineno, following in reads
            if not any('.wrap()' in ln or 'find_mic(' in ln or 'classify_site(' in ln
                       for ln in following)
        ]
        assert not unwrapped, (
            f"{filename}: lammps-data read(s) at line(s) {unwrapped} are NOT "
            f"followed by .wrap() (or a find_mic(...) realignment) within "
            f"the next 10 lines — Bug 2 may be re-introduced"
        )

    def test_surface_graph_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('surface_graph.py')

    def test_structure_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('structure.py')

    def test_ase_neb_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('ase_neb.py')

    def test_neb_subsurface_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('neb_subsurface.py')

    def test_neb_workflow_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('neb_workflow.py')

    def test_subsurface_graph_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('subsurface_graph.py')

    def test_site_identifier_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('site_identifier.py')

    def test_vibrations_reads_are_wrapped(self):
        self._assert_all_reads_wrapped('vibrations.py')

    def test_permeation_workflow_wrap_exemption_documented(self):
        # permeation_workflow.py:85 reads the slab only to call get_cell().
        # Cell dimensions are invariant to wrapping, so the omission is safe.
        # This guard ensures the read still exists (not removed entirely) and
        # that it targets the slab path variable, not a structure-consuming path.
        src = self._src('permeation_workflow.py')
        assert "format='lammps-data'" in src or 'format="lammps-data"' in src, (
            "permeation_workflow.py no longer has a lammps-data read — "
            "verify the exemption comment is still valid"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. Generated-script AST validity
# ═════════════════════════════════════════════════════════════════════════════

import ast as _ast


class TestGeneratedScriptAstValidity:
    """
    Generated scripts (permeation_run.py, diffusivity_run.py, run_neb.py) are
    executed on the cluster hours after submission.  A syntax error in the
    template would silently fail at import time.  Use ast.parse() to catch
    template bugs (unclosed brackets, bad f-strings, wrong indentation).
    """

    def test_permeation_run_py_is_valid_python(self, tmp_path):
        out = str(tmp_path / 'permeation_run.py')
        generate_permeation_scripts(**_PERM_CFG, out_py=out)
        src = pathlib.Path(out).read_text()
        try:
            _ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(
                f"permeation_run.py has a SyntaxError: {exc}\n"
                f"  line {exc.lineno}: {exc.text}"
            )

    def test_diffusivity_run_py_is_valid_python(self, tmp_path):
        out = str(tmp_path / 'diffusivity_run.py')
        generate_diffusivity_scripts(**_DIFF_CFG, out_py=out)
        src = pathlib.Path(out).read_text()
        try:
            _ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(
                f"diffusivity_run.py has a SyntaxError: {exc}\n"
                f"  line {exc.lineno}: {exc.text}"
            )

    def test_run_neb_py_is_valid_python(self, tmp_path):
        out = str(tmp_path / 'run_neb.py')
        write_ase_neb_script(
            **_DUMMY,
            logfile_phase1=str(tmp_path / 'neb_phase1.log'),
            logfile_phase2=str(tmp_path / 'neb_phase2.log'),
            out_path=out,
        )
        src = pathlib.Path(out).read_text()
        try:
            _ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(
                f"run_neb.py has a SyntaxError: {exc}\n"
                f"  line {exc.lineno}: {exc.text}"
            )

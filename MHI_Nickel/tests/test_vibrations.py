"""
tests/test_vibrations.py
=========================
Unit tests for models/vibrations.py — fully offline, no MACE or LAMMPS.

Covers:
  extract_ts_structure   — parses neb_barrier.txt, returns TS image path
  write_vibration_script — generates standalone vib_run.py
  collect_is_ts_paths    — builds (label, path) list from job dicts
  load_vibration_results — reads vib_frequencies.json
  orchestrate_vibrations — writes scripts + SLURM jobs in dry_run mode
"""

import json
import os
import pathlib
import sys
import warnings
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.vibrations import (
    extract_ts_structure,
    write_vibration_script,
    collect_is_ts_paths,
    load_vibration_results,
    orchestrate_vibrations,
)

# ── shared constants ──────────────────────────────────────────────────────────

_STRUCT = '/data/structure.lammps'
_MACE   = '/models/mace.model'
_OUTDIR = '/work/vib_out'

_VIB_DATA = {
    'structure'            : _STRUCT,
    'n_atoms_displaced'    : 7,
    'h_index'              : 42,
    'metal_indices'        : [0, 1, 2, 3, 4, 5],
    'delta_ang'            : 0.01,
    'frequencies_real_cm1' : [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    'frequencies_imag_cm1' : [750.0],
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _write_barrier(job_dir: pathlib.Path, lines: list[str]) -> pathlib.Path:
    p = job_dir / 'neb_barrier.txt'
    p.write_text('\n'.join(lines))
    return p


def _make_ts_image(job_dir: pathlib.Path, ts_idx: int) -> pathlib.Path:
    img_dir = job_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    fname = f'image_{ts_idx:02d}_img_{ts_idx:02d}.lammps'
    p = img_dir / fname
    p.write_text('dummy lammps image')
    return p


@pytest.fixture
def barrier_job(tmp_path):
    """NEB job dir with barrier file + TS image at index 2 (highest energy)."""
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    _write_barrier(job_dir, [
        'Image energies:',
        'IS: -100.0',
        'img_1: -99.5 eV',
        'img_2: -99.0 eV',
        'img_3: -99.3 eV',
        'FS: -100.2',
    ])
    _make_ts_image(job_dir, 2)
    return job_dir


@pytest.fixture
def is_file(tmp_path):
    p = tmp_path / 'is_structure.lammps'
    p.write_text('dummy IS')
    return p


@pytest.fixture
def neb_job(tmp_path, is_file, barrier_job):
    return {
        'sid'    : 'Ni3Mo_s0_s1',
        'is_path': str(is_file),
        'job_dir': str(barrier_job),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. extract_ts_structure
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractTsStructure:

    def test_returns_string_path(self, barrier_job):
        ts = extract_ts_structure(str(barrier_job))
        assert isinstance(ts, str)

    def test_selects_highest_energy_image(self, barrier_job):
        ts = extract_ts_structure(str(barrier_job))
        # img_2 has energy -99.0, the maximum → ts_index=2
        assert 'image_02_img_02' in ts

    def test_returned_path_is_absolute(self, barrier_job):
        ts = extract_ts_structure(str(barrier_job))
        assert os.path.isabs(ts)

    def test_raises_if_barrier_file_missing(self, tmp_path):
        empty = tmp_path / 'empty_job'
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match='neb_barrier.txt'):
            extract_ts_structure(str(empty))

    def test_raises_if_no_intermediate_images(self, tmp_path):
        job = tmp_path / 'j'
        job.mkdir()
        _write_barrier(job, [
            'Image energies:',
            'IS: -100.0',
            'FS: -100.2',
        ])
        with pytest.raises(ValueError, match='No intermediate image'):
            extract_ts_structure(str(job))

    def test_raises_if_ts_image_file_missing(self, tmp_path):
        job = tmp_path / 'j'
        job.mkdir()
        _write_barrier(job, [
            'Image energies:',
            'IS: -100.0',
            'img_1: -99.5 eV',
            'FS: -100.2',
        ])
        # images/ dir not created — ts file is missing
        with pytest.raises(FileNotFoundError, match='TS image file not found'):
            extract_ts_structure(str(job))

    def test_single_image_found(self, tmp_path):
        job = tmp_path / 'j'
        job.mkdir()
        _write_barrier(job, [
            'Image energies:',
            'IS: -100.0',
            'img_1: -99.5 eV',
            'FS: -100.2',
        ])
        _make_ts_image(job, 1)
        ts = extract_ts_structure(str(job))
        assert 'image_01_img_01' in ts


# ═══════════════════════════════════════════════════════════════════════════
# 2. write_vibration_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteVibrationScript:

    @pytest.fixture
    def script_result(self, tmp_path):
        out_path = str(tmp_path / 'scripts' / 'vib_run.py')
        ret = write_vibration_script(
            structure_path  = _STRUCT,
            mace_model_path = _MACE,
            out_path        = out_path,
            outdir          = _OUTDIR,
            delta           = 0.01,
            device          = 'cpu',
        )
        content = pathlib.Path(out_path).read_text()
        return ret, out_path, content

    def test_file_created(self, script_result):
        _, out_path, _ = script_result
        assert pathlib.Path(out_path).exists()

    def test_returns_out_path(self, script_result):
        ret, out_path, _ = script_result
        assert ret == out_path

    def test_structure_path_embedded(self, script_result):
        _, _, content = script_result
        assert _STRUCT in content

    def test_mace_model_embedded(self, script_result):
        _, _, content = script_result
        assert _MACE in content

    def test_outdir_embedded(self, script_result):
        _, _, content = script_result
        assert _OUTDIR in content

    def test_delta_embedded(self, script_result):
        _, _, content = script_result
        assert '0.01' in content

    def test_device_embedded(self, script_result):
        _, _, content = script_result
        assert "'cpu'" in content

    def test_mace_calculator_imported(self, script_result):
        _, _, content = script_result
        assert 'MACECalculator' in content

    def test_vibrations_class_imported(self, script_result):
        _, _, content = script_result
        assert 'Vibrations' in content

    def test_vib_frequencies_json_in_script(self, script_result):
        _, _, content = script_result
        assert 'vib_frequencies.json' in content

    def test_file_is_executable(self, script_result):
        _, out_path, _ = script_result
        assert os.access(out_path, os.X_OK)

    def test_parent_directory_created(self, tmp_path):
        nested = str(tmp_path / 'a' / 'b' / 'c' / 'vib_run.py')
        write_vibration_script(_STRUCT, _MACE, nested, _OUTDIR)
        assert pathlib.Path(nested).exists()

    def test_custom_delta_and_device(self, tmp_path):
        out = str(tmp_path / 'vib.py')
        write_vibration_script(_STRUCT, _MACE, out, _OUTDIR,
                               delta=0.05, device='cuda')
        content = pathlib.Path(out).read_text()
        assert '0.05' in content
        assert "'cuda'" in content

    def test_default_dtype_is_float32(self, script_result):
        _, _, content = script_result
        assert "DTYPE      = 'float32'" in content
        assert 'default_dtype=DTYPE' in content
        assert 'default_dtype="float64"' not in content

    def test_custom_dtype_override(self, tmp_path):
        out = str(tmp_path / 'vib.py')
        write_vibration_script(_STRUCT, _MACE, out, _OUTDIR, dtype='float64')
        content = pathlib.Path(out).read_text()
        assert "DTYPE      = 'float64'" in content


# ═══════════════════════════════════════════════════════════════════════════
# 3. collect_is_ts_paths
# ═══════════════════════════════════════════════════════════════════════════

class TestCollectIsTsPaths:

    def test_returns_is_entry(self, neb_job):
        pairs = collect_is_ts_paths([neb_job], hop='hopa')
        labels = [p[0] for p in pairs]
        assert 'hopa_Ni3Mo_s0_s1_IS' in labels

    def test_returns_ts_entry(self, neb_job):
        pairs = collect_is_ts_paths([neb_job], hop='hopa')
        labels = [p[0] for p in pairs]
        assert 'hopa_Ni3Mo_s0_s1_TS' in labels

    def test_labels_use_hop_prefix(self, neb_job):
        pairs = collect_is_ts_paths([neb_job], hop='hopb')
        assert all(lbl.startswith('hopb_') for lbl, _ in pairs)

    def test_missing_is_file_warns_and_skips_is(self, neb_job, tmp_path):
        bad_job = {**neb_job, 'is_path': str(tmp_path / 'nonexistent.lammps')}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            pairs = collect_is_ts_paths([bad_job], hop='hopa')
        labels = [p[0] for p in pairs]
        assert 'hopa_Ni3Mo_s0_s1_IS' not in labels
        assert any('IS file not found' in str(x.message) for x in w)

    def test_ts_failure_warns_and_skips_ts(self, is_file):
        job = {'sid': 'test', 'is_path': str(is_file), 'job_dir': '/nonexistent/dir'}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            pairs = collect_is_ts_paths([job], hop='hopa')
        labels = [p[0] for p in pairs]
        assert 'hopa_test_TS' not in labels
        assert any('TS extraction failed' in str(x.message) for x in w)

    def test_empty_jobs_returns_empty_list(self):
        assert collect_is_ts_paths([], hop='hopa') == []

    def test_is_path_is_in_pairs(self, neb_job, is_file):
        pairs = collect_is_ts_paths([neb_job], hop='hopa')
        is_paths = [path for lbl, path in pairs if '_IS' in lbl]
        assert str(is_file) in is_paths


# ═══════════════════════════════════════════════════════════════════════════
# 4. load_vibration_results
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadVibrationResults:

    @pytest.fixture
    def vib_json(self, tmp_path):
        p = tmp_path / 'vib_frequencies.json'
        p.write_text(json.dumps(_VIB_DATA))
        return str(p)

    def test_returns_dict(self, vib_json):
        result = load_vibration_results(vib_json)
        assert isinstance(result, dict)

    def test_expected_keys_present(self, vib_json):
        result = load_vibration_results(vib_json)
        for key in ('frequencies_real_cm1', 'frequencies_imag_cm1',
                    'h_index', 'metal_indices', 'delta_ang', 'structure'):
            assert key in result, f'Missing key: {key}'

    def test_frequencies_real_loaded(self, vib_json):
        result = load_vibration_results(vib_json)
        assert result['frequencies_real_cm1'] == _VIB_DATA['frequencies_real_cm1']

    def test_frequencies_imag_loaded(self, vib_json):
        result = load_vibration_results(vib_json)
        assert result['frequencies_imag_cm1'] == _VIB_DATA['frequencies_imag_cm1']

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_vibration_results(str(tmp_path / 'missing.json'))

    def test_h_index_correct(self, vib_json):
        result = load_vibration_results(vib_json)
        assert result['h_index'] == 42


# ═══════════════════════════════════════════════════════════════════════════
# 5. orchestrate_vibrations
# ═══════════════════════════════════════════════════════════════════════════

class TestOrchestrateVibrations:

    _PAIRS = [
        ('hopa_Ni3Mo_IS', '/data/is.lammps'),
        ('hopa_Ni3Mo_TS', '/data/ts.lammps'),
    ]
    _SLURM = {
        'partition'    : 'cpu',
        'ntasks'       : 1,
        'cpus_per_task': 4,
        'time'         : '02:00:00',
        'mem'          : '8G',
        'conda_env'    : 'ase-env',
        'openmpi_ver'  : '4.1.4',
    }

    def test_creates_vib_run_py_per_label(self, tmp_path):
        orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE, dry_run=True)
        for label, _ in self._PAIRS:
            p = tmp_path / 'out' / label / 'vib_run.py'
            assert p.exists(), f'{label}/vib_run.py not found'

    def test_returns_dict_keyed_by_label(self, tmp_path):
        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE, dry_run=True)
        for label, _ in self._PAIRS:
            assert label in result

    def test_result_has_vib_script_key(self, tmp_path):
        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE, dry_run=True)
        for label, _ in self._PAIRS:
            assert 'vib_script' in result[label]

    def test_result_has_vib_json_key(self, tmp_path):
        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE, dry_run=True)
        for label, _ in self._PAIRS:
            assert 'vib_json' in result[label]

    def test_no_slurm_file_when_opts_none(self, tmp_path):
        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE,
            slurm_opts=None, dry_run=True)
        for label, _ in self._PAIRS:
            assert result[label]['slurm'] is None

    def test_slurm_file_created_when_opts_provided(self, tmp_path):
        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE,
            slurm_opts=self._SLURM, dry_run=True)
        for label, _ in self._PAIRS:
            slurm_path = result[label]['slurm']
            assert slurm_path is not None
            assert pathlib.Path(slurm_path).exists()

    def test_empty_structure_list_returns_empty_dict(self, tmp_path):
        result = orchestrate_vibrations(
            [], str(tmp_path / 'out'), _MACE, dry_run=True)
        assert result == {}


class TestOrchestrateVibrationsCheckpoint:
    """orchestrate_vibrations previously had zero skip-on-rerun logic --
    every call regenerated vib_run.py/vib_job.sh and (if not dry_run)
    resubmitted every label unconditionally, even for already-converged
    structures. Covers the new vib.done marker check."""

    _PAIRS = [('hopa_Ni3Mo_IS', '/data/is.lammps')]
    _SLURM = {
        'partition'    : 'cpu',
        'ntasks'       : 1,
        'cpus_per_task': 4,
        'time'         : '02:00:00',
        'mem'          : '8G',
        'conda_env'    : 'ase-env',
        'openmpi_ver'  : '4.1.4',
    }

    def test_skips_regeneration_when_done_marker_present(self, tmp_path, monkeypatch):
        job_outdir = tmp_path / 'out' / 'hopa_Ni3Mo_IS'
        job_outdir.mkdir(parents=True)
        (job_outdir / 'vib.done').touch()

        calls = []
        monkeypatch.setattr(
            'models.vibrations.write_vibration_script',
            lambda **kw: calls.append(kw),
        )

        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE,
            slurm_opts=self._SLURM, dry_run=True)

        assert calls == []
        assert result['hopa_Ni3Mo_IS']['slurm'] is None

    def test_regenerates_when_done_marker_absent(self, tmp_path):
        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE,
            slurm_opts=self._SLURM, dry_run=True)

        assert result['hopa_Ni3Mo_IS']['slurm'] is not None
        assert pathlib.Path(result['hopa_Ni3Mo_IS']['vib_script']).exists()

    def test_generated_vib_run_py_touches_done_marker_last(self, tmp_path):
        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE, dry_run=True)
        script = pathlib.Path(result['hopa_Ni3Mo_IS']['vib_script']).read_text()
        done_idx = script.index('vib.done')
        json_idx = script.index('vib_frequencies.json')
        assert done_idx > json_idx, (
            'vib.done must be touched after vib_frequencies.json is written, '
            'so a killed job never leaves a false "done" marker'
        )

    def test_skip_never_returns_stale_slurm_path(self, tmp_path):
        """A skipped label's 'slurm' must be None, not a path to a stale
        script from a prior run -- otherwise callers that do
        `if v.get('slurm'): submit_slurm_job(v['slurm'])` (neb_workflow.py
        Phase E, permeation_workflow.py Phase 3) would resubmit the
        already-done label's old (real, non-stub) command."""
        job_outdir = tmp_path / 'out' / 'hopa_Ni3Mo_IS'
        job_outdir.mkdir(parents=True)
        (job_outdir / 'vib.done').touch()
        # Stale slurm script left over from whatever run produced vib.done.
        (job_outdir / 'vib_job.sh').write_text('#!/bin/bash\necho real work\n')

        result = orchestrate_vibrations(
            self._PAIRS, str(tmp_path / 'out'), _MACE,
            slurm_opts=self._SLURM, dry_run=True)

        assert result['hopa_Ni3Mo_IS']['slurm'] is None

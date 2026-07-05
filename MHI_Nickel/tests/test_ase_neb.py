"""
tests/test_ase_neb.py
======================
Tests for models/ase_neb.py — offline, no LAMMPS/SLURM/cluster required.

Covers:
  write_ase_neb_script / run_neb_pipeline — embedded config, and the
  per-image LAMMPS-data write regression (Bug 20).
"""

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.ase_neb import run_neb_pipeline
from models.config import MASSES_7, E2T_7


def _generate(tmp_path, **overrides):
    kwargs = dict(
        is_file='/x/is.lammps',
        fs_file='/x/fs.lammps',
        e_is=-100.0,
        mace_model_path='/x/model.model',
        barrier_file='/x/barrier.txt',
        path_file='/x/path.dat',
        outdir=str(tmp_path),
        fs_log_file='/x/fs_min.log',
        job_name='test',
        n_images=6,
        spring_const=1.0,
        neb_ftol=0.1,
        z_freeze_cutoff=10.0,
        device='cpu',
        label_is='IS',
        label_fs='FS',
    )
    kwargs.update(overrides)
    out_py = run_neb_pipeline(**kwargs)
    return out_py, pathlib.Path(out_py).read_text()


class TestRunNebPipeline:

    def test_file_created_and_compiles(self, tmp_path):
        out_py, content = _generate(tmp_path)
        assert pathlib.Path(out_py).exists()
        compile(content, out_py, 'exec')

    def test_per_image_write_uses_write_lammps_data(self, tmp_path):
        """Regression test for Bug 20: a bare ase.io.write of NEB image
        structures omits the Masses section, so a later read() (no
        explicit type-to-element override) falls back to treating the
        raw LAMMPS atom type id as the atomic number -- with this
        project's E2T_7 mapping (Ni=7, H=8) that silently collides with
        real elements (N=7, O=8) instead of erroring. vib_run.py found
        "0 H atoms" on a structure that actually has exactly one.
        write_lammps_data (this project's canonical writer) always
        writes a real Masses section, so the round trip is correct
        regardless of what later reads the file."""
        _, content = _generate(tmp_path, masses=MASSES_7, e2t=E2T_7)
        assert 'write_lammps_data(' in content
        assert 'from models.structure import write_lammps_data' in content
        assert '_ase_write' not in content

    def test_masses_and_e2t_embedded(self, tmp_path):
        _, content = _generate(tmp_path, masses=MASSES_7, e2t=E2T_7)
        assert 'MASSES          = ' in content
        assert 'E2T             = ' in content
        assert "'Ni': 7" in content
        assert "'H': 8" in content

    def test_masses_and_e2t_default_when_not_provided(self, tmp_path):
        # Callers that don't pass masses/e2t must still get a valid,
        # non-empty mapping (MASSES_7/E2T_7), not an empty/None embed.
        _, content = _generate(tmp_path)
        assert 'MASSES          = {' in content
        assert 'E2T             = {' in content

    def test_generated_script_can_import_models_package(self, tmp_path):
        """The generated script runs standalone on a cluster node --
        it must add the project root to sys.path before importing
        models.structure, or the import will fail there."""
        _, content = _generate(tmp_path)
        assert 'sys.path.insert(0, _parent)' in content
        import_idx = content.index('from models.structure import write_lammps_data')
        syspath_idx = content.index('sys.path.insert(0, _parent)')
        assert syspath_idx < import_idx

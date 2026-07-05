"""
tests/test_utils.py
====================
Tests for models/utils.py — fully offline, only touches tmp_path.

Covers:
  make_run_dirs — the directory-layout function underlying Part 3's
  per-(stem, n_H) results directories. Its correctness is what the
  Bug 12 fix (per-n_H Arrhenius overwrite) actually depends on: two
  different n_H values must resolve to two different `root` paths.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.utils import make_run_dirs


class TestMakeRunDirs:

    def test_root_path_is_base_dir_plus_name(self, tmp_path):
        dirs = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        assert dirs['root'] == str(tmp_path / 'ni_bulk_test_1H')

    def test_root_dir_created_on_disk(self, tmp_path):
        dirs = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        assert pathlib.Path(dirs['root']).is_dir()

    def test_structures_subdir_created(self, tmp_path):
        dirs = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        assert dirs['structures'] == str(tmp_path / 'ni_bulk_test_1H' / 'structures')
        assert pathlib.Path(dirs['structures']).is_dir()

    def test_per_temperature_subdirs_created(self, tmp_path):
        dirs = make_run_dirs('ni_bulk_test_1H', [600, 700], base_dir=str(tmp_path))
        for T in (600, 700):
            for d in ('lammps_scripts', 'slurm_scripts', 'results'):
                p = pathlib.Path(dirs[T][d])
                assert p.is_dir()
                assert p.name == f'{T}K'

    def test_temperature_key_cast_to_int(self, tmp_path):
        dirs = make_run_dirs('ni_bulk_test_1H', [600.0], base_dir=str(tmp_path))
        t_keys = [k for k in dirs if k not in ('root', 'structures')]
        assert t_keys == [600]
        assert isinstance(t_keys[0], int)

    def test_idempotent_on_rerun(self, tmp_path):
        # exist_ok=True — calling twice must not raise
        make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        dirs2 = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        assert pathlib.Path(dirs2['root']).is_dir()

    # ── direct regression proof for Bug 12 (per-n_H Arrhenius overwrite) ──────

    def test_different_names_give_different_root_paths(self, tmp_path):
        """The bug: every n_H shared one directory. The fix relies on
        make_run_dirs(name=f'{stem}_{n_h}H', ...) actually producing a
        distinct root per n_H — verify that directly."""
        dirs_1h = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        dirs_3h = make_run_dirs('ni_bulk_test_3H', [600], base_dir=str(tmp_path))
        assert dirs_1h['root'] != dirs_3h['root']
        assert pathlib.Path(dirs_1h['root']).is_dir()
        assert pathlib.Path(dirs_3h['root']).is_dir()

    def test_different_names_give_independent_result_dirs(self, tmp_path):
        dirs_1h = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        dirs_3h = make_run_dirs('ni_bulk_test_3H', [600], base_dir=str(tmp_path))
        assert dirs_1h[600]['results'] != dirs_3h[600]['results']

    def test_same_name_reused_across_calls_gives_same_root(self, tmp_path):
        # Same n_H called twice (e.g. resumed run) must land in the same
        # place, not fork into a second directory.
        dirs_a = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        dirs_b = make_run_dirs('ni_bulk_test_1H', [600], base_dir=str(tmp_path))
        assert dirs_a['root'] == dirs_b['root']

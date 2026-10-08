"""
tests/test_materials.py
=======================
models/materials.py is the single source of truth for which materials the
pipeline processes. Its value is entirely in being single -- the list, the
classifier and the surface-skip rules used to be duplicated across the three
regenerate_*_scripts.py and pipeline.ipynb, where a missed edit silently
dropped a material from one Part.

The last two classes below are the ones that matter: they fail if anyone
re-introduces a local copy.
"""

import ast
import io
import json
import os
import pathlib
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.materials import (
    INPUT_STRUCTURE_FILES,
    SKIP_SURFACE_STEMS,
    classify_metal,
    input_structures,
    skip_surface_reason,
    stem_of,
)

_TOOLS = pathlib.Path(PROJECT_ROOT) / 'calculation' / 'tools'
_REGENERATORS = [
    'regenerate_neb_scripts.py',
    'regenerate_permeation_scripts.py',
    'regenerate_diffusivity_scripts.py',
]


class TestStemOf:
    def test_strips_directory_and_extension(self):
        assert stem_of('/a/b/Al_supercell.lammps') == 'Al_supercell'

    def test_handles_lmp_extension(self):
        assert stem_of('/a/bestsqs3.lmp') == 'bestsqs3'


class TestClassifyMetal:
    @pytest.mark.parametrize('name,expected', [
        ('Ni_oxide_supercell.lammps', 'oxide'),
        ('Cr_oxide_supercell.lammps', 'oxide'),
        ('Hastelloy_N_7_supercell.lammps', 'alloy'),
        ('bestsqs3.lmp', 'alloy'),
        ('Al_supercell.lammps', 'pure'),
        ('Ni_supercell.lammps', 'pure'),
        ('Fe_supercell.lammps', 'pure'),
    ])
    def test_classification(self, name, expected):
        assert classify_metal('/w/input_structure/' + name) == expected

    def test_oxide_beats_alloy(self):
        """An oxide of an alloy is still an oxide -- order of the checks."""
        assert classify_metal('/w/hastelloy_oxide_supercell.lammps') == 'oxide'

    def test_is_case_insensitive(self):
        assert classify_metal('/w/AL_SUPERCELL.LAMMPS') == 'pure'


class TestInputStructures:
    def test_paths_are_under_input_structure(self):
        for p in input_structures('/w'):
            assert p.startswith(os.path.join('/w', 'input_structure'))

    def test_one_path_per_declared_file(self):
        assert len(input_structures('/w')) == len(INPUT_STRUCTURE_FILES)

    def test_every_declared_file_exists_on_disk(self):
        """A name in the list with no file behind it would fail far downstream,
        in a generated script, instead of here."""
        wd = os.path.join(PROJECT_ROOT, 'calculation')
        missing = [p for p in input_structures(wd) if not os.path.exists(p)]
        assert not missing, f'declared but absent: {missing}'

    def test_no_duplicate_entries(self):
        assert len(INPUT_STRUCTURE_FILES) == len(set(INPUT_STRUCTURE_FILES))


class TestSkipSurfaceReason:
    def test_named_polar_oxide_is_skipped(self):
        r = skip_surface_reason('/w/input_structure/Ni_oxide_supercell.lammps')
        assert r and 'polar oxide' in r

    def test_fcc_alloy_is_not_skipped(self):
        assert skip_surface_reason('/w/input_structure/Hastelloy_N_7_supercell.lammps') is None

    def test_unreadable_structure_is_not_silently_skipped(self):
        """A missing file must not be mistaken for 'skip me' -- it should flow
        on and fail loudly downstream instead."""
        assert skip_surface_reason('/w/input_structure/Does_Not_Exist.lammps') is None

    def test_skip_set_entries_carry_a_reason(self):
        for stem, reason in SKIP_SURFACE_STEMS.items():
            assert isinstance(reason, str) and reason.strip(), stem


class TestNoLocalCopiesInRegenerators:
    """The regression guard. Re-introducing any of these in a regenerator
    recreates the four-way duplication this module exists to remove."""

    @pytest.mark.parametrize('fname', _REGENERATORS)
    def test_no_local_input_structures_literal(self, fname):
        src = io.open(_TOOLS / fname, encoding='utf-8').read()
        assert 'INPUT_STRUCTURES = [' not in src, \
            f'{fname} declares its own list; import input_structures() instead'

    @pytest.mark.parametrize('fname', _REGENERATORS)
    def test_no_local_classify_metal(self, fname):
        src = io.open(_TOOLS / fname, encoding='utf-8').read()
        assert 'def classify_metal' not in src, \
            f'{fname} defines its own classify_metal; import it instead'

    @pytest.mark.parametrize('fname', _REGENERATORS)
    def test_imports_from_the_shared_module(self, fname):
        src = io.open(_TOOLS / fname, encoding='utf-8').read()
        assert 'from models.materials import' in src

    @pytest.mark.parametrize('fname', _REGENERATORS)
    def test_regenerator_still_parses(self, fname):
        ast.parse(io.open(_TOOLS / fname, encoding='utf-8').read())


class TestPipelineNotebookUsesSharedModule:
    """pipeline.ipynb was the fourth copy."""

    def _config_cell(self):
        nb = json.load(io.open(pathlib.Path(PROJECT_ROOT) / 'calculation' /
                               'pipeline.ipynb', encoding='utf-8'))
        for c in nb['cells']:
            s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
            if 'INPUT_STRUCTURES' in s:
                return s
        pytest.fail('no cell defines INPUT_STRUCTURES')

    def test_imports_shared_module(self):
        assert 'from models.materials import' in self._config_cell()

    def test_has_no_literal_list(self):
        assert 'INPUT_STRUCTURES = [' not in self._config_cell()

    def test_has_no_local_classify_metal(self):
        assert 'def classify_metal' not in self._config_cell()

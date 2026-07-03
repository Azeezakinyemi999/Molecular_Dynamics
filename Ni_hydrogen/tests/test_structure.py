"""
tests/test_structure.py
=======================
Unit tests for models/structure.py.

Covers:
  write_lammps_data              — low-level LAMMPS file writer
  get_lattice_parameter          — a0 from minimized bulk LAMMPS file
  get_lattice_parameter_from_dump — a0 from NPT ave/time box-dim file
  build_alloy_bulk               — random FCC alloy builder
  build_slab (alloy)             — Ni-FCC template + composition shuffle
  build_slab (pure)              — correct FCC/BCC geometry from crystal map
  build_slab (oxide)             — spglib primitive-cell path (skipped if unavailable)
  add_adsorbate                  — H and H2 placement above a surface
  build_fs_raw_structure         — NEB final-state raw structure

All tests are fully offline (no LAMMPS, no SLURM).
"""

import os
import sys
import math

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.structure import (
    write_lammps_data,
    get_lattice_parameter,
    get_lattice_parameter_from_dump,
    build_alloy_bulk,
    build_slab,
    add_adsorbate,
    build_fs_raw_structure,
    _CRYSTAL_STRUCT_MAP,
)
from models.config import H2_HEIGHT, H2_BOND


# ─── file-inspection helpers ──────────────────────────────────────────────────

def _atom_count_header(path):
    """Return the N from the 'N atoms' header line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.endswith('atoms') and 'types' not in line:
                try:
                    return int(line.split()[0])
                except (ValueError, IndexError):
                    pass
    return 0


def _atom_types_in_atoms_block(path):
    """Return the set of integer type IDs in the Atoms section."""
    types = set()
    in_atoms = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Atoms'):
                in_atoms = True
                continue
            if in_atoms and line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        types.add(int(parts[1]))
                    except (ValueError, IndexError):
                        pass
    return types


def _atom_rows(path):
    """Return list of (type, x, y, z) for every atom in the Atoms block."""
    rows = []
    in_atoms = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Atoms'):
                in_atoms = True
                continue
            if in_atoms and line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        rows.append((int(parts[1]),
                                     float(parts[2]),
                                     float(parts[3]),
                                     float(parts[4])))
                    except (ValueError, IndexError):
                        pass
    return rows


# ─── shared constants ─────────────────────────────────────────────────────────

_A0_NI   = 3.52   # Å  FCC Ni
_A0_CR   = 2.91   # Å  BCC Cr

_MASSES_NI = {1: (58.6934, 'Ni')}
_E2T_NI    = {'Ni': 1}

_MASSES_NI_H = {1: (58.6934, 'Ni'), 2: (1.0080, 'H')}
_E2T_NI_H   = {'Ni': 1, 'H': 2}

_MASSES_MO_NI = {1: (58.6934, 'Ni'), 2: (95.96, 'Mo')}
_E2T_MO_NI   = {'Ni': 1, 'Mo': 2}


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def ni_2x2x2_file(tmp_path):
    """2×2×2 FCC Ni supercell written with write_lammps_data.

    Uses cubic=True so the conventional 4-atom cell produces an orthogonal
    box (Lx = 2*a0 = 7.04 Å), and the # Ni comment lets ASE identify Ni
    when reading back.  32 atoms total.
    """
    from ase.build import bulk, make_supercell
    unit = bulk('Ni', 'fcc', a=_A0_NI, cubic=True)   # 4-atom conventional cell
    sc   = make_supercell(unit, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])  # 32 atoms
    fpath = str(tmp_path / 'ni_2x2x2.lammps')
    write_lammps_data(
        list(sc.get_chemical_symbols()), sc.get_positions(),
        sc.cell.lengths(), _MASSES_NI, _E2T_NI, fpath,
        comment='Ni FCC cubic 2x2x2',
    )
    return fpath  # 32 atoms, Lx = 7.04 Å


@pytest.fixture()
def cr_2x2x2_file(tmp_path):
    """2×2×2 BCC Cr supercell written with write_lammps_data.

    Uses cubic=True for an orthogonal box (Lx = 2*a0 = 5.82 Å).
    16 atoms total.
    """
    from ase.build import bulk, make_supercell
    masses_cr = {1: (51.9961, 'Cr')}
    e2t_cr    = {'Cr': 1}
    unit = bulk('Cr', 'bcc', a=_A0_CR, cubic=True)   # 2-atom conventional cell
    sc   = make_supercell(unit, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])  # 16 atoms
    fpath = str(tmp_path / 'cr_2x2x2.lammps')
    write_lammps_data(
        list(sc.get_chemical_symbols()), sc.get_positions(),
        sc.cell.lengths(), masses_cr, e2t_cr, fpath,
        comment='Cr BCC cubic 2x2x2',
    )
    return fpath  # 16 atoms, Lx = 5.82 Å


@pytest.fixture()
def simple_slab_file(tmp_path):
    """Minimal 2-atom Ni slab written with write_lammps_data (z_top = 2.0 Å)."""
    pos = np.array([[0.0, 0.0, 0.0], [1.76, 1.76, 2.0]])
    fpath = str(tmp_path / 'slab.lammps')
    write_lammps_data(['Ni', 'Ni'], pos, [3.52, 3.52, 20.0],
                      _MASSES_NI_H, _E2T_NI_H, fpath, comment='test slab')
    return fpath  # z_top = 2.0


@pytest.fixture()
def is_slab_file(tmp_path):
    """Ni slab + H2 molecule (IS structure) for build_fs_raw_structure tests."""
    # 2 Ni atoms (z_top = 2.0) + 2 H atoms above
    pos = np.array([
        [0.00,  0.00, 0.0],   # Ni  (z=0)
        [1.76,  1.76, 2.0],   # Ni  (z=2 = z_top)
        [1.00,  1.00, 4.5],   # H1
        [1.741, 1.00, 4.5],   # H2
    ])
    fpath = str(tmp_path / 'is_slab.lammps')
    write_lammps_data(
        ['Ni', 'Ni', 'H', 'H'], pos, [3.52, 3.52, 20.0],
        _MASSES_NI_H, _E2T_NI_H, fpath, comment='IS slab'
    )
    return fpath


# ═════════════════════════════════════════════════════════════════════════════
# 1. write_lammps_data
# ═════════════════════════════════════════════════════════════════════════════

class TestWriteLammpsData:

    @pytest.fixture()
    def written(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0], [1.76, 1.76, 1.76]])
        fpath = str(tmp_path / 'out.lammps')
        ret = write_lammps_data(
            symbols=['Ni', 'Ni'],
            positions=pos,
            cell_lengths=[3.52, 3.52, 3.52],
            masses=_MASSES_NI,
            e2t=_E2T_NI,
            out_path=fpath,
            comment='test comment',
        )
        return fpath, ret, open(fpath).read()

    def test_file_created(self, written):
        fpath, _, _ = written
        assert os.path.exists(fpath)

    def test_returns_absolute_path(self, written):
        fpath, ret, _ = written
        assert os.path.isabs(ret)
        assert ret == os.path.abspath(fpath)

    def test_atom_count_in_header(self, written):
        _, _, content = written
        assert '2 atoms' in content

    def test_atom_types_in_header(self, written):
        _, _, content = written
        assert '1 atom types' in content

    def test_cell_x_in_header(self, written):
        _, _, content = written
        assert 'xlo xhi' in content
        assert '3.52' in content

    def test_cell_y_in_header(self, written):
        _, _, content = written
        assert 'ylo yhi' in content

    def test_cell_z_in_header(self, written):
        _, _, content = written
        assert 'zlo zhi' in content

    def test_masses_section_present(self, written):
        _, _, content = written
        assert 'Masses' in content
        assert '58.6934' in content

    def test_atoms_section_present(self, written):
        _, _, content = written
        assert 'Atoms # atomic' in content

    def test_comment_written(self, written):
        _, _, content = written
        assert 'test comment' in content

    def test_two_atom_rows_written(self, written):
        fpath, _, _ = written
        rows = _atom_rows(fpath)
        assert len(rows) == 2

    def test_position_first_atom(self, written):
        fpath, _, _ = written
        rows = _atom_rows(fpath)
        x, y, z = rows[0][1], rows[0][2], rows[0][3]
        assert math.isclose(x, 0.0) and math.isclose(y, 0.0) and math.isclose(z, 0.0)

    def test_parent_dirs_created(self, tmp_path):
        deep = str(tmp_path / 'a' / 'b' / 'c' / 'test.lammps')
        pos = np.array([[0.0, 0.0, 0.0]])
        write_lammps_data(['Ni'], pos, [3.52, 3.52, 3.52],
                          _MASSES_NI, _E2T_NI, deep)
        assert os.path.exists(deep)

    def test_two_types_written(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0], [1.76, 1.76, 1.76]])
        fpath = str(tmp_path / 'mixed.lammps')
        write_lammps_data(
            ['Ni', 'H'], pos, [3.52, 3.52, 3.52],
            _MASSES_NI_H, _E2T_NI_H, fpath,
        )
        content = open(fpath).read()
        assert '2 atom types' in content
        types = _atom_types_in_atoms_block(fpath)
        assert types == {1, 2}


# ═════════════════════════════════════════════════════════════════════════════
# 2. get_lattice_parameter
# ═════════════════════════════════════════════════════════════════════════════

class TestGetLatticeParameter:

    def test_correct_a0_from_2x2x2(self, ni_2x2x2_file):
        a0 = get_lattice_parameter(ni_2x2x2_file, supercell_reps=(2, 2, 2))
        assert math.isclose(a0, _A0_NI, rel_tol=1e-4)

    def test_different_reps_give_different_result(self, ni_2x2x2_file):
        # Lx = 2*a0; with reps=(1,1,1) → a0_wrong = 2*a0
        a0_wrong = get_lattice_parameter(ni_2x2x2_file, supercell_reps=(1, 1, 1))
        assert math.isclose(a0_wrong, 2 * _A0_NI, rel_tol=1e-4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_lattice_parameter(str(tmp_path / 'ghost.lammps'))

    def test_returns_float(self, ni_2x2x2_file):
        a0 = get_lattice_parameter(ni_2x2x2_file, supercell_reps=(2, 2, 2))
        assert isinstance(a0, float)


# ═════════════════════════════════════════════════════════════════════════════
# 3. get_lattice_parameter_from_dump
# ═════════════════════════════════════════════════════════════════════════════

_DUMP_CONTENT = """\
# Step Lx Ly Lz
0     17.6  17.6  17.6
100   17.61 17.61 17.61
200   17.59 17.59 17.59
300   17.60 17.60 17.60
"""


class TestGetLatticeParameterFromDump:

    @pytest.fixture()
    def dump_file(self, tmp_path):
        f = tmp_path / 'boxdims.dat'
        f.write_text(_DUMP_CONTENT)
        return str(f)

    def test_correct_a0(self, dump_file):
        # mean of all 4 Lx values = (17.6+17.61+17.59+17.60)/4 = 17.6; /5 = 3.52
        a0 = get_lattice_parameter_from_dump(dump_file, n_last=4, supercell_reps=(5, 5, 5))
        assert math.isclose(a0, 3.52, rel_tol=1e-3)

    def test_uses_last_n_rows(self, dump_file):
        # With n_last=1 → last row Lx=17.60; a0=17.60/5 = 3.52
        a0 = get_lattice_parameter_from_dump(dump_file, n_last=1, supercell_reps=(5, 5, 5))
        assert math.isclose(a0, 17.60 / 5, rel_tol=1e-6)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_lattice_parameter_from_dump(str(tmp_path / 'ghost.dat'))

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / 'empty.dat'
        f.write_text('')
        with pytest.raises(ValueError):
            get_lattice_parameter_from_dump(str(f))

    def test_header_comment_lines_skipped(self, tmp_path):
        txt = '# comment\n# another\n17.6 17.6 17.6 17.6\n'
        f = tmp_path / 'hdr.dat'
        f.write_text(txt)
        a0 = get_lattice_parameter_from_dump(str(f), n_last=1, supercell_reps=(5, 5, 5))
        assert math.isclose(a0, 17.6 / 5, rel_tol=1e-6)

    def test_returns_float(self, dump_file):
        a0 = get_lattice_parameter_from_dump(dump_file, supercell_reps=(5, 5, 5))
        assert isinstance(a0, float)


# ═════════════════════════════════════════════════════════════════════════════
# 4. build_alloy_bulk
# ═════════════════════════════════════════════════════════════════════════════

_COMP_NI_MO = {'Ni': 0.70, 'Mo': 0.30}
_SC_2x2x2   = (2, 2, 2)  # 4 atoms/cell × 8 = 32 atoms


class TestBuildAlloyBulk:

    @pytest.fixture()
    def bulk_path(self, tmp_path):
        return str(tmp_path / 'alloy.lammps')

    def test_file_created(self, bulk_path):
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, bulk_path)
        assert os.path.exists(bulk_path)

    def test_returns_absolute_path(self, bulk_path):
        ret = build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                               _MASSES_MO_NI, _E2T_MO_NI, bulk_path)
        assert os.path.isabs(ret)

    def test_correct_fcc_atom_count(self, bulk_path):
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, bulk_path)
        # FCC: 4 atoms/conventional cell × 2³ = 32 atoms
        assert _atom_count_header(bulk_path) == 32

    def test_both_element_types_present(self, bulk_path):
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, bulk_path)
        types = _atom_types_in_atoms_block(bulk_path)
        assert 1 in types and 2 in types

    def test_composition_fractions_approximately_respected(self, bulk_path):
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, bulk_path)
        rows = _atom_rows(bulk_path)
        n_ni = sum(1 for r in rows if r[0] == 1)   # type 1 = Ni
        n_mo = sum(1 for r in rows if r[0] == 2)   # type 2 = Mo
        assert n_ni + n_mo == 32   # 4 atoms/cell × 2³ = 32 atoms
        # 70% Ni → ~22 Ni atoms; allow ±2 rounding
        assert abs(n_ni - 22) <= 2

    def test_raises_on_bad_composition(self, bulk_path):
        bad_comp = {'Ni': 0.5, 'Mo': 0.3}   # sums to 0.8
        with pytest.raises(ValueError):
            build_alloy_bulk(bad_comp, _A0_NI, _SC_2x2x2,
                             _MASSES_MO_NI, _E2T_MO_NI, bulk_path)

    def test_reproducible_with_same_seed(self, tmp_path):
        p1 = str(tmp_path / 'a1.lammps')
        p2 = str(tmp_path / 'a2.lammps')
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, p1, seed=99)
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, p2, seed=99)
        rows1 = _atom_rows(p1)
        rows2 = _atom_rows(p2)
        assert rows1 == rows2

    def test_different_seeds_differ(self, tmp_path):
        p1 = str(tmp_path / 'b1.lammps')
        p2 = str(tmp_path / 'b2.lammps')
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, p1, seed=1)
        build_alloy_bulk(_COMP_NI_MO, _A0_NI, _SC_2x2x2,
                         _MASSES_MO_NI, _E2T_MO_NI, p2, seed=2)
        types1 = [r[0] for r in _atom_rows(p1)]
        types2 = [r[0] for r in _atom_rows(p2)]
        assert types1 != types2


# ═════════════════════════════════════════════════════════════════════════════
# 5. _CRYSTAL_STRUCT_MAP
# ═════════════════════════════════════════════════════════════════════════════

class TestCrystalStructMap:

    def test_ni_is_fcc(self):
        assert _CRYSTAL_STRUCT_MAP['Ni'] == ('fcc', 'Ni')

    def test_fe_is_bcc(self):
        assert _CRYSTAL_STRUCT_MAP['Fe'][0] == 'bcc'

    def test_mo_is_bcc(self):
        assert _CRYSTAL_STRUCT_MAP['Mo'][0] == 'bcc'

    def test_cr_is_bcc(self):
        assert _CRYSTAL_STRUCT_MAP['Cr'][0] == 'bcc'

    def test_al_is_fcc(self):
        assert _CRYSTAL_STRUCT_MAP['Al'][0] == 'fcc'

    def test_w_is_bcc(self):
        assert _CRYSTAL_STRUCT_MAP['W'][0] == 'bcc'


# ═════════════════════════════════════════════════════════════════════════════
# 6. build_slab — alloy branch
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildSlabAlloy:

    @pytest.fixture()
    def slab_result(self, ni_2x2x2_file, tmp_path):
        out = str(tmp_path / 'alloy_slab.lammps')
        return build_slab(
            ni_2x2x2_file, miller=(1, 1, 1), layers=4, vacuum=10.0,
            masses=_MASSES_NI, e2t=_E2T_NI, out_path=out,
            supercell_reps=(2, 2, 2), metal_type='alloy',
        )

    def test_file_created(self, slab_result):
        out_path, _ = slab_result
        assert os.path.exists(out_path)

    def test_returns_tuple(self, slab_result):
        assert isinstance(slab_result, tuple) and len(slab_result) == 2

    def test_a0_approximately_correct(self, slab_result):
        _, a0 = slab_result
        assert math.isclose(a0, _A0_NI, rel_tol=1e-3)

    def test_has_atoms(self, slab_result):
        out_path, _ = slab_result
        assert _atom_count_header(out_path) > 0

    def test_lateral_repeat_doubles_atoms(self, ni_2x2x2_file, tmp_path):
        out1 = str(tmp_path / 'slab_11.lammps')
        out2 = str(tmp_path / 'slab_21.lammps')
        build_slab(ni_2x2x2_file, (1, 1, 1), 4, 10.0,
                   _MASSES_NI, _E2T_NI, out1,
                   supercell_reps=(2, 2, 2), lateral_repeat=(1, 1),
                   metal_type='alloy')
        build_slab(ni_2x2x2_file, (1, 1, 1), 4, 10.0,
                   _MASSES_NI, _E2T_NI, out2,
                   supercell_reps=(2, 2, 2), lateral_repeat=(2, 1),
                   metal_type='alloy')
        n1 = _atom_count_header(out1)
        n2 = _atom_count_header(out2)
        assert n2 == 2 * n1


# ═════════════════════════════════════════════════════════════════════════════
# 7. build_slab — pure branch
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildSlabPure:

    @pytest.fixture()
    def ni_pure_slab(self, ni_2x2x2_file, tmp_path):
        out = str(tmp_path / 'ni_pure_slab.lammps')
        return build_slab(
            ni_2x2x2_file, miller=(1, 1, 1), layers=4, vacuum=10.0,
            masses=_MASSES_NI, e2t=_E2T_NI, out_path=out,
            supercell_reps=(2, 2, 2), metal_type='pure',
        )

    def test_file_created(self, ni_pure_slab):
        out_path, _ = ni_pure_slab
        assert os.path.exists(out_path)

    def test_a0_correct(self, ni_pure_slab):
        _, a0 = ni_pure_slab
        assert math.isclose(a0, _A0_NI, rel_tol=1e-3)

    def test_all_atoms_are_ni(self, ni_pure_slab):
        out_path, _ = ni_pure_slab
        types = _atom_types_in_atoms_block(out_path)
        assert types == {_E2T_NI['Ni']}  # only Ni type

    def test_cr_bcc_pure_slab(self, cr_2x2x2_file, tmp_path):
        # cr_2x2x2_file was written with masses_cr={1:(51.9961,'Cr')} / e2t_cr={'Cr':1}
        masses_cr = {1: (51.9961, 'Cr'), 2: (1.0080, 'H')}
        e2t_cr    = {'Cr': 1, 'H': 2}
        out = str(tmp_path / 'cr_slab.lammps')
        out_path, a0 = build_slab(
            cr_2x2x2_file, miller=(1, 1, 0), layers=4, vacuum=10.0,
            masses=masses_cr, e2t=e2t_cr, out_path=out,
            supercell_reps=(2, 2, 2), metal_type='pure',
        )
        assert os.path.exists(out_path)
        assert math.isclose(a0, _A0_CR, rel_tol=1e-3)

    def test_raises_keyerror_for_unknown_element(self, tmp_path):
        # Au is not in _CRYSTAL_STRUCT_MAP — pure branch must raise KeyError.
        # Write with write_lammps_data so ASE reads back '# Au' → 'Au' symbol.
        from ase.build import bulk
        masses_au = {1: (196.97, 'Au')}
        e2t_au    = {'Au': 1}
        unit = bulk('Au', 'fcc', a=4.08, cubic=True)
        fpath = str(tmp_path / 'au.lammps')
        write_lammps_data(
            ['Au'] * len(unit), unit.get_positions(),
            unit.cell.lengths(), masses_au, e2t_au, fpath,
        )
        out = str(tmp_path / 'au_slab.lammps')
        with pytest.raises(KeyError):
            build_slab(fpath, (1, 1, 1), 4, 10.0,
                       masses_au, e2t_au, out,
                       supercell_reps=(1, 1, 1), metal_type='pure')

    def test_raises_valueerror_for_multi_element_bulk(self, tmp_path):
        # A bulk file with two non-H elements is invalid for metal_type='pure'
        pos = np.array([[0.0, 0.0, 0.0], [1.76, 1.76, 1.76]])
        fpath = str(tmp_path / 'mixed.lammps')
        write_lammps_data(['Ni', 'Mo'], pos, [3.52, 3.52, 3.52],
                          _MASSES_MO_NI, _E2T_MO_NI, fpath)
        out = str(tmp_path / 'bad_slab.lammps')
        with pytest.raises(ValueError):
            build_slab(fpath, (1, 1, 1), 4, 10.0,
                       _MASSES_NI, _E2T_NI, out,
                       supercell_reps=(1, 1, 1), metal_type='pure')


# ═════════════════════════════════════════════════════════════════════════════
# 8. build_slab — oxide branch (requires spglib)
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildSlabOxide:

    def test_oxide_slab_created(self, tmp_path):
        spglib = pytest.importorskip('spglib')
        from ase.build import bulk
        from ase.io import write as ase_write
        # Use NiO rock-salt as a simple oxide
        nio = bulk('NiO', crystalstructure='rocksalt', a=4.177, cubic=True)
        fpath = str(tmp_path / 'nio.lammps')
        ase_write(fpath, nio, format='lammps-data')
        masses_nio = {1: (58.6934, 'Ni'), 2: (15.999, 'O')}
        e2t_nio    = {'Ni': 1, 'O': 2}
        out = str(tmp_path / 'nio_slab.lammps')
        out_path, a0 = build_slab(
            fpath, miller=(1, 0, 0), layers=2, vacuum=10.0,
            masses=masses_nio, e2t=e2t_nio, out_path=out,
            metal_type='oxide',
        )
        assert os.path.exists(out_path)
        assert a0 > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 9. add_adsorbate
# ═════════════════════════════════════════════════════════════════════════════

class TestAddAdsorbate:
    """simple_slab_file: 2 Ni atoms, z_top = 2.0 Å."""

    def test_h_file_created(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h_ads.lammps')
        add_adsorbate(simple_slab_file, [1.76, 1.76], 'H',
                      _MASSES_NI_H, _E2T_NI_H, out)
        assert os.path.exists(out)

    def test_h_atom_count(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h_ads.lammps')
        add_adsorbate(simple_slab_file, [1.76, 1.76], 'H',
                      _MASSES_NI_H, _E2T_NI_H, out)
        assert _atom_count_header(out) == 3   # 2 Ni + 1 H

    def test_h_placed_at_correct_z(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h_ads.lammps')
        add_adsorbate(simple_slab_file, [1.0, 1.0], 'H',
                      _MASSES_NI_H, _E2T_NI_H, out, height=2.0)
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        assert len(h_rows) == 1
        # z_top=2.0 + height=2.0 = 4.0
        assert math.isclose(h_rows[0][3], 4.0, rel_tol=1e-6)

    def test_h2_parallel_atom_count(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h2_ads.lammps')
        add_adsorbate(simple_slab_file, [1.76, 1.76], 'H2',
                      _MASSES_NI_H, _E2T_NI_H, out)
        assert _atom_count_header(out) == 4   # 2 Ni + 2 H

    def test_h2_parallel_both_h_at_same_z(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h2_parallel.lammps')
        add_adsorbate(simple_slab_file, [1.76, 1.76], 'H2',
                      _MASSES_NI_H, _E2T_NI_H, out,
                      h2_orientation='parallel')
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        assert len(h_rows) == 2
        assert math.isclose(h_rows[0][3], h_rows[1][3], rel_tol=1e-6)

    def test_h2_parallel_x_offset(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h2_xoff.lammps')
        x_site = 1.76
        add_adsorbate(simple_slab_file, [x_site, 1.76], 'H2',
                      _MASSES_NI_H, _E2T_NI_H, out,
                      h2_bond=1.0, h2_orientation='parallel')
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        xs = sorted(r[1] for r in h_rows)
        # H1 at x_site - 0.5, H2 at x_site + 0.5
        assert math.isclose(xs[1] - xs[0], 1.0, rel_tol=1e-6)

    def test_h2_vertical_different_z(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h2_vert.lammps')
        add_adsorbate(simple_slab_file, [1.76, 1.76], 'H2',
                      _MASSES_NI_H, _E2T_NI_H, out,
                      h2_bond=1.0, h2_orientation='vertical')
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        assert len(h_rows) == 2
        assert not math.isclose(h_rows[0][3], h_rows[1][3])

    def test_h2_vertical_z_symmetric(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h2_vert_sym.lammps')
        add_adsorbate(simple_slab_file, [1.76, 1.76], 'H2',
                      _MASSES_NI_H, _E2T_NI_H, out,
                      height=2.0, h2_bond=1.0, h2_orientation='vertical')
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        zs = sorted(r[3] for r in h_rows)
        z_mid = (zs[0] + zs[1]) / 2.0
        # z_top=2.0 + height=2.0 = 4.0
        assert math.isclose(z_mid, 4.0, rel_tol=1e-6)
        assert math.isclose(zs[1] - zs[0], 1.0, rel_tol=1e-6)

    def test_default_height_from_config(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'h_default.lammps')
        add_adsorbate(simple_slab_file, [1.76, 1.76], 'H',
                      _MASSES_NI_H, _E2T_NI_H, out)
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        # z_top=2.0 + H2_HEIGHT=2.5 = 4.5
        assert math.isclose(h_rows[0][3], 2.0 + H2_HEIGHT, rel_tol=1e-6)

    def test_raises_bad_species(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'bad.lammps')
        with pytest.raises(ValueError):
            add_adsorbate(simple_slab_file, [0, 0], 'N2',
                          _MASSES_NI_H, _E2T_NI_H, out)

    def test_raises_bad_orientation(self, simple_slab_file, tmp_path):
        out = str(tmp_path / 'bad2.lammps')
        with pytest.raises(ValueError):
            add_adsorbate(simple_slab_file, [0, 0], 'H2',
                          _MASSES_NI_H, _E2T_NI_H, out,
                          h2_orientation='diagonal')

    def test_raises_missing_slab(self, tmp_path):
        out = str(tmp_path / 'h_ads.lammps')
        with pytest.raises(FileNotFoundError):
            add_adsorbate(str(tmp_path / 'ghost.lammps'), [0, 0], 'H',
                          _MASSES_NI_H, _E2T_NI_H, out)


# ═════════════════════════════════════════════════════════════════════════════
# 10. build_fs_raw_structure
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildFsRawStructure:
    """is_slab_file: 2 Ni (z=0, z=2) + 2 H (z=4.5). Cell = 3.52×3.52×20 Å."""

    def test_file_created(self, is_slab_file, tmp_path):
        out = str(tmp_path / 'fs.lammps')
        build_fs_raw_structure(is_slab_file, (1.0, 1.0), (1.5, 1.5),
                               _MASSES_NI_H, _E2T_NI_H, out)
        assert os.path.exists(out)

    def test_output_atom_count(self, is_slab_file, tmp_path):
        out = str(tmp_path / 'fs.lammps')
        build_fs_raw_structure(is_slab_file, (1.0, 1.0), (1.5, 1.5),
                               _MASSES_NI_H, _E2T_NI_H, out)
        # IS had 2 Ni + 2 H; FS has 2 Ni + 2 new H = 4
        assert _atom_count_header(out) == 4

    def test_strips_original_h_adds_two_new(self, is_slab_file, tmp_path):
        out = str(tmp_path / 'fs.lammps')
        build_fs_raw_structure(is_slab_file, (1.0, 1.0), (1.5, 1.5),
                               _MASSES_NI_H, _E2T_NI_H, out)
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        assert len(h_rows) == 2

    def test_h_at_correct_z(self, is_slab_file, tmp_path):
        out = str(tmp_path / 'fs_z.lammps')
        h_height = 1.5
        build_fs_raw_structure(is_slab_file, (1.0, 1.0), (1.5, 1.5),
                               _MASSES_NI_H, _E2T_NI_H, out, h_height=h_height)
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        # z_top of metal = 2.0; z_fs = 2.0 + 1.5 = 3.5
        for r in h_rows:
            assert math.isclose(r[3], 3.5, rel_tol=1e-6)

    def test_h_xy_positions(self, is_slab_file, tmp_path):
        out = str(tmp_path / 'fs_xy.lammps')
        build_fs_raw_structure(is_slab_file, (1.0, 1.0), (2.0, 2.0),
                               _MASSES_NI_H, _E2T_NI_H, out)
        rows = _atom_rows(out)
        h_rows = sorted([r for r in rows if r[0] == _E2T_NI_H['H']],
                        key=lambda r: r[1])
        xy1 = (h_rows[0][1], h_rows[0][2])
        xy2 = (h_rows[1][1], h_rows[1][2])
        # Cell is 3.52×3.52; coordinates already in-bounds, no PBC wrapping needed
        assert math.isclose(xy1[0], 1.0, rel_tol=1e-6)
        assert math.isclose(xy2[0], 2.0, rel_tol=1e-6)

    def test_pbc_wrapping(self, is_slab_file, tmp_path):
        out = str(tmp_path / 'fs_pbc.lammps')
        # x=5.0 is outside the 3.52 Å cell → should wrap to 5.0 % 3.52 ≈ 1.48
        build_fs_raw_structure(is_slab_file, (5.0, 1.0), (1.0, 1.0),
                               _MASSES_NI_H, _E2T_NI_H, out)
        rows = _atom_rows(out)
        h_rows = [r for r in rows if r[0] == _E2T_NI_H['H']]
        xs = [r[1] for r in h_rows]
        assert all(0.0 <= x < 3.52 for x in xs)

    def test_returns_absolute_path(self, is_slab_file, tmp_path):
        out = str(tmp_path / 'fs_ret.lammps')
        ret = build_fs_raw_structure(is_slab_file, (1.0, 1.0), (1.5, 1.5),
                                     _MASSES_NI_H, _E2T_NI_H, out)
        assert os.path.isabs(ret)

    def test_raises_missing_is_file(self, tmp_path):
        out = str(tmp_path / 'fs.lammps')
        with pytest.raises(FileNotFoundError):
            build_fs_raw_structure(str(tmp_path / 'ghost.lammps'),
                                   (1.0, 1.0), (1.5, 1.5),
                                   _MASSES_NI_H, _E2T_NI_H, out)

"""
tests/test_parsers.py
=====================
Unit tests for models/parsers.py.

Covers all 9 public entry points (plus the private _parse_lammps_log base):
  parse_minimization_log
  parse_surface_relaxation_log
  parse_equil_log
  parse_energy_log
  parse_thermo_series
  parse_lammps_dump
  parse_diffusivity_file
  parse_barrier_file
  parse_neb_path

All tests are fully offline — no LAMMPS, no SLURM, no cluster.
"""

import os
import sys
import math

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.parsers import (
    parse_minimization_log,
    parse_surface_relaxation_log,
    parse_equil_log,
    parse_energy_log,
    parse_dissociated_h2_log,
    parse_thermo_series,
    parse_lammps_dump,
    parse_diffusivity_file,
    parse_barrier_file,
    parse_neb_path,
)


# ─── small helpers ────────────────────────────────────────────────────────────

def _write(path, text):
    path.write_text(text)
    return str(path)


# ═════════════════════════════════════════════════════════════════════════════
# 1. parse_minimization_log
# ═════════════════════════════════════════════════════════════════════════════

_MIN_LOG = """\
LAMMPS (29 Sep 2021)

Step PotEng Fmax Lx Press
0 -1234.5 3.21 35.2 100.0
100 -1250.0 0.05 35.2 95.0
Loop time of 10.0

Stopping criterion : energy tolerance

MINIMIZATION_RESULTS_START
Total_energy_eV : -1250.0
Natoms : 108
Ecoh_eV_per_atom : -11.574074
a0_Angstrom : 3.52
Fmax_eV_per_Ang : 0.05
MINIMIZATION_RESULTS_END
"""


class TestParseMinimizationLog:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'min.log', _MIN_LOG)
        return parse_minimization_log(f)

    def test_returns_tuple(self, result):
        assert isinstance(result, tuple) and len(result) == 2

    def test_thermo_step_parsed(self, result):
        thermo, _ = result
        assert thermo['step'] == [0.0, 100.0]

    def test_thermo_pe_parsed(self, result):
        thermo, _ = result
        assert thermo['pe'] == [-1234.5, -1250.0]

    def test_meta_total_energy(self, result):
        _, meta = result
        assert math.isclose(meta['Total_energy_eV'], -1250.0)

    def test_meta_natoms(self, result):
        _, meta = result
        assert math.isclose(meta['Natoms'], 108.0)

    def test_meta_ecoh(self, result):
        _, meta = result
        assert math.isclose(meta['Ecoh_eV_per_atom'], -11.574074, rel_tol=1e-5)

    def test_meta_a0(self, result):
        _, meta = result
        assert math.isclose(meta['a0_Angstrom'], 3.52)

    def test_meta_fmax(self, result):
        _, meta = result
        assert math.isclose(meta['Fmax_eV_per_Ang'], 0.05)

    def test_stop_criterion_captured(self, result):
        _, meta = result
        assert 'stop_criterion' in meta
        assert 'energy tolerance' in meta['stop_criterion']

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            parse_minimization_log(str(tmp_path / 'nope.log'))

    def test_empty_thermo_when_no_step_line(self, tmp_path):
        f = _write(tmp_path / 'bare.log', 'nothing here\n')
        thermo, meta = parse_minimization_log(f)
        assert thermo['step'] == []
        assert thermo['pe'] == []


# ═════════════════════════════════════════════════════════════════════════════
# 2. parse_surface_relaxation_log
# ═════════════════════════════════════════════════════════════════════════════

_SURF_LOG = """\
LAMMPS (29 Sep 2021)

GROUPS: frozen=60  free=300

Step Time Temp PotEng Press
0 0.0 300.0 -5000.0 1.0
500 0.25 302.1 -5010.5 0.8
Loop time of 5.0

RELAXATION_RESULTS_START
z_top_before_Ang : 18.50
z_top_after_min_Ang : 18.42
z_top_after_nvt_Ang : 18.40
surface_contraction : 0.10
pe_final_eV : -5010.5
RELAXATION_RESULTS_END
"""


class TestParseSurfaceRelaxationLog:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'surf.log', _SURF_LOG)
        return parse_surface_relaxation_log(f)

    def test_returns_tuple(self, result):
        assert isinstance(result, tuple) and len(result) == 2

    def test_thermo_step_parsed(self, result):
        thermo, _ = result
        assert thermo['step'] == [0.0, 500.0]

    def test_meta_z_top_before(self, result):
        _, meta = result
        assert math.isclose(meta['z_top_before_Ang'], 18.50)

    def test_meta_surface_contraction(self, result):
        _, meta = result
        assert math.isclose(meta['surface_contraction'], 0.10)

    def test_meta_pe_final(self, result):
        _, meta = result
        assert math.isclose(meta['pe_final_eV'], -5010.5)

    def test_frozen_count_parsed(self, result):
        _, meta = result
        assert meta['frozen_count'] == 60

    def test_free_count_parsed(self, result):
        _, meta = result
        assert meta['free_count'] == 300

    def test_missing_groups_line_ok(self, tmp_path):
        log = _SURF_LOG.replace('GROUPS: frozen=60  free=300\n', '')
        f = _write(tmp_path / 'nogroups.log', log)
        _, meta = parse_surface_relaxation_log(f)
        assert 'frozen_count' not in meta
        assert 'free_count' not in meta


# ═════════════════════════════════════════════════════════════════════════════
# 3. parse_equil_log
# ═════════════════════════════════════════════════════════════════════════════

_EQUIL_LOG = """\
EQUIL_RESULTS_START
T_K : 1000.0
pe_final_eV : -8000.1
temp_final_K : 999.8
press_final : 0.52
EQUIL_RESULTS_END
"""


class TestParseEquilLog:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'equil.log', _EQUIL_LOG)
        return parse_equil_log(f)

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_T_K(self, result):
        assert math.isclose(result['T_K'], 1000.0)

    def test_pe_final(self, result):
        assert math.isclose(result['pe_final_eV'], -8000.1)

    def test_temp_final(self, result):
        assert math.isclose(result['temp_final_K'], 999.8)

    def test_press_final(self, result):
        assert math.isclose(result['press_final'], 0.52)

    def test_missing_file_returns_none(self, tmp_path):
        result = parse_equil_log(str(tmp_path / 'ghost.log'))
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        f = _write(tmp_path / 'empty.log', '')
        assert parse_equil_log(f) is None


# ═════════════════════════════════════════════════════════════════════════════
# 4. parse_energy_log
# ═════════════════════════════════════════════════════════════════════════════

_ENERGY_LOG = """\
Some preamble text
pe_final_eV: -1234.5678
fmax_eV_per_Ang: 0.0123
natoms: 432
More text
"""


class TestParseEnergyLog:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'energy.log', _ENERGY_LOG)
        return parse_energy_log(f)

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_pe_final(self, result):
        assert math.isclose(result['pe_final_eV'], -1234.5678)

    def test_fmax(self, result):
        assert math.isclose(result['fmax_eV_per_Ang'], 0.0123)

    def test_natoms(self, result):
        assert math.isclose(result['natoms'], 432.0)

    def test_missing_file_returns_none(self, tmp_path):
        assert parse_energy_log(str(tmp_path / 'nope.log')) is None

    def test_file_with_no_keys_returns_none(self, tmp_path):
        f = _write(tmp_path / 'no_keys.log', 'irrelevant: 99\n')
        assert parse_energy_log(f) is None


# ═════════════════════════════════════════════════════════════════════════════
# 5. parse_thermo_series
# ═════════════════════════════════════════════════════════════════════════════

_THERMO_LOG = """\
LAMMPS output

Step Temp PotEng KinEng
0 300.0 -1000.0 120.0
1000 350.5 -990.2 140.3
2000 400.1 -980.5 160.7
Loop time of 50.0
"""


class TestParseThermoSeries:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'thermo.log', _THERMO_LOG)
        return parse_thermo_series(f)

    def test_returns_three_arrays(self, result):
        assert result is not None
        assert len(result) == 3

    def test_steps(self, result):
        steps, _, _ = result
        np.testing.assert_array_equal(steps, [0, 1000, 2000])

    def test_temps(self, result):
        _, temps, _ = result
        np.testing.assert_allclose(temps, [300.0, 350.5, 400.1])

    def test_pes(self, result):
        _, _, pes = result
        np.testing.assert_allclose(pes, [-1000.0, -990.2, -980.5])

    def test_missing_file_returns_none(self, tmp_path):
        assert parse_thermo_series(str(tmp_path / 'nope.log')) is None

    def test_no_thermo_block_returns_none(self, tmp_path):
        f = _write(tmp_path / 'plain.log', 'hello world\n')
        assert parse_thermo_series(f) is None

    def test_multiple_blocks_concatenated(self, tmp_path):
        log = """\
Step Temp PotEng
0 300.0 -1000.0
Loop time of 1.0
Step Temp PotEng
1000 400.0 -900.0
Loop time of 1.0
"""
        f = _write(tmp_path / 'multi.log', log)
        result = parse_thermo_series(f)
        assert result is not None
        steps, _, _ = result
        assert len(steps) == 2


# ═════════════════════════════════════════════════════════════════════════════
# 6. parse_lammps_dump
# ═════════════════════════════════════════════════════════════════════════════

def _make_dump(path, frames):
    """Write a minimal LAMMPS custom dump with columns id type x y z."""
    lines = []
    for ts, atoms in frames:
        lines.append('ITEM: TIMESTEP')
        lines.append(str(ts))
        lines.append('ITEM: NUMBER OF ATOMS')
        lines.append(str(len(atoms)))
        lines.append('ITEM: BOX BOUNDS pp pp pp')
        lines.append('0.0 10.0')
        lines.append('0.0 10.0')
        lines.append('0.0 10.0')
        lines.append('ITEM: ATOMS id type x y z')
        for aid, atype, x, y, z in atoms:
            lines.append(f'{aid} {atype} {x} {y} {z}')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


_H_TYPE = 9
_FRAMES = [
    (0,    [(1, 1, 1.0, 2.0, 3.0), (2, _H_TYPE, 4.0, 5.0, 6.0)]),
    (1000, [(1, 1, 1.1, 2.1, 3.1), (2, _H_TYPE, 4.1, 5.1, 6.1)]),
]


class TestParseLammpsDump:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _make_dump(tmp_path / 'traj.lammpstrj', _FRAMES)
        return parse_lammps_dump(f, h_type=_H_TYPE, timestep=0.0005)

    def test_returns_three_arrays(self, result):
        t, pos, box = result
        assert t is not None and pos is not None and box is not None

    def test_frame_count(self, result):
        t, pos, box = result
        assert len(t) == 2

    def test_time_conversion(self, result):
        t, _, _ = result
        np.testing.assert_allclose(t, [0.0, 0.5])

    def test_h_positions_frame0(self, result):
        _, pos, _ = result
        np.testing.assert_allclose(pos[0, 0], [4.0, 5.0, 6.0])

    def test_h_positions_frame1(self, result):
        _, pos, _ = result
        np.testing.assert_allclose(pos[1, 0], [4.1, 5.1, 6.1])

    def test_box_shape(self, result):
        _, _, box = result
        assert box.shape == (2, 3)

    def test_box_edge_lengths(self, result):
        _, _, box = result
        np.testing.assert_allclose(box[0], [10.0, 10.0, 10.0])

    def test_missing_file_returns_nones(self, tmp_path):
        t, pos, box = parse_lammps_dump(str(tmp_path / 'ghost.lammpstrj'))
        assert t is None and pos is None and box is None

    def test_no_h_atoms_fills_nan(self, tmp_path):
        # Frame with no H atoms at all
        frames = [(0, [(1, 1, 1.0, 2.0, 3.0)])]
        f = _make_dump(tmp_path / 'noh.lammpstrj', frames)
        _, pos, _ = parse_lammps_dump(f, h_type=_H_TYPE)
        assert np.all(np.isnan(pos[0]))

    def test_multi_h_shape(self, tmp_path):
        frames = [(0, [(1, _H_TYPE, 1.0, 2.0, 3.0), (2, _H_TYPE, 4.0, 5.0, 6.0)])]
        f = _make_dump(tmp_path / 'twoh.lammpstrj', frames)
        _, pos, _ = parse_lammps_dump(f, h_type=_H_TYPE)
        assert pos.shape == (1, 2, 3)

    def test_empty_file_returns_nones(self, tmp_path):
        f = _write(tmp_path / 'empty.lammpstrj', '')
        t, pos, box = parse_lammps_dump(f)
        assert t is None and pos is None and box is None


# ═════════════════════════════════════════════════════════════════════════════
# 7. parse_diffusivity_file
# ═════════════════════════════════════════════════════════════════════════════

_DIFF_FILE = """\
# Diffusivity results
# ===================
T_K    D          sigma_D    R2
300    1.23e-09   0.05e-09   0.998
600    4.56e-09   0.10e-09   0.995
900    9.87e-09   0.15e-09   0.991
"""


class TestParseDiffusivityFile:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'diffusivity.txt', _DIFF_FILE)
        return parse_diffusivity_file(f)

    def test_returns_four_arrays(self, result):
        assert len(result) == 4

    def test_T_array(self, result):
        T, _, _, _ = result
        np.testing.assert_array_equal(T, [300.0, 600.0, 900.0])

    def test_D_array(self, result):
        _, D, _, _ = result
        np.testing.assert_allclose(D, [1.23e-9, 4.56e-9, 9.87e-9])

    def test_Derr_array(self, result):
        _, _, Derr, _ = result
        np.testing.assert_allclose(Derr, [0.05e-9, 0.10e-9, 0.15e-9])

    def test_R2_array(self, result):
        _, _, _, R2 = result
        np.testing.assert_allclose(R2, [0.998, 0.995, 0.991])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_diffusivity_file(str(tmp_path / 'nope.txt'))

    def test_comment_and_header_lines_skipped(self, tmp_path):
        txt = '# comment\n= separator\n- separator\nT_K D s R2\n300 1e-9 0.1e-9 0.99\n'
        f = _write(tmp_path / 'd.txt', txt)
        T, D, Derr, R2 = parse_diffusivity_file(f)
        assert len(T) == 1

    def test_returns_numpy_arrays(self, result):
        for arr in result:
            assert isinstance(arr, np.ndarray)


# ═════════════════════════════════════════════════════════════════════════════
# 8. parse_barrier_file
# ═════════════════════════════════════════════════════════════════════════════

_BARRIER_COLON = """\
# NEB barrier summary
E_IS: -100.0
E_FS: -99.5
E_abs: 0.80
E_des: 0.30
delta_E: 0.50
fmax_final: 0.012
converged: True
"""

_BARRIER_EQ = """\
E_IS = -100.0
E_FS = -99.5
E_abs = 0.80
E_des = 0.30
delta_E = 0.50
fmax_final = 0.012
converged = yes
"""

_BARRIER_ALIAS = """\
E_a: 0.75
delta_E: 0.40
converged: false
"""


class TestParseBarrierFile:

    @pytest.fixture()
    def result_colon(self, tmp_path):
        f = _write(tmp_path / 'barrier_colon.txt', _BARRIER_COLON)
        return parse_barrier_file(f)

    @pytest.fixture()
    def result_eq(self, tmp_path):
        f = _write(tmp_path / 'barrier_eq.txt', _BARRIER_EQ)
        return parse_barrier_file(f)

    @pytest.fixture()
    def result_alias(self, tmp_path):
        f = _write(tmp_path / 'barrier_alias.txt', _BARRIER_ALIAS)
        return parse_barrier_file(f)

    def test_returns_dict(self, result_colon):
        assert isinstance(result_colon, dict)

    def test_E_IS(self, result_colon):
        assert math.isclose(result_colon['E_IS'], -100.0)

    def test_E_FS(self, result_colon):
        assert math.isclose(result_colon['E_FS'], -99.5)

    def test_E_abs(self, result_colon):
        assert math.isclose(result_colon['E_abs'], 0.80)

    def test_E_des(self, result_colon):
        assert math.isclose(result_colon['E_des'], 0.30)

    def test_delta_E(self, result_colon):
        assert math.isclose(result_colon['delta_E'], 0.50)

    def test_fmax_final(self, result_colon):
        assert math.isclose(result_colon['fmax_final'], 0.012)

    def test_converged_true(self, result_colon):
        assert result_colon['converged'] is True

    def test_equals_separator_works(self, result_eq):
        assert math.isclose(result_eq['E_abs'], 0.80)
        assert result_eq['converged'] is True   # 'yes' → True

    def test_converged_false(self, result_alias):
        assert result_alias['converged'] is False

    def test_E_a_alias_normalised(self, result_alias):
        assert 'E_abs' in result_alias
        assert math.isclose(result_alias['E_abs'], 0.75)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_barrier_file(str(tmp_path / 'ghost.txt'))

    def test_comment_lines_skipped(self, tmp_path):
        txt = '# this is a comment\nE_abs: 0.50\nconverged: true\n'
        f = _write(tmp_path / 'c.txt', txt)
        result = parse_barrier_file(f)
        assert math.isclose(result['E_abs'], 0.50)


# ═════════════════════════════════════════════════════════════════════════════
# 9. parse_neb_path
# ═════════════════════════════════════════════════════════════════════════════

_NEB_PATH = """\
# image_fraction  E_eV  dE_from_IS_eV
0.0   -100.000   0.000
0.25  -99.600    0.400
0.50  -99.200    0.800
0.75  -99.500    0.500
1.0   -99.700    0.300
"""


class TestParseNebPath:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'neb_path.dat', _NEB_PATH)
        return parse_neb_path(f)

    def test_returns_three_arrays(self, result):
        assert len(result) == 3

    def test_frac_array(self, result):
        frac, _, _ = result
        np.testing.assert_allclose(frac, [0.0, 0.25, 0.50, 0.75, 1.0])

    def test_E_array(self, result):
        _, E, _ = result
        np.testing.assert_allclose(E, [-100.0, -99.6, -99.2, -99.5, -99.7])

    def test_dE_array(self, result):
        _, _, dE = result
        np.testing.assert_allclose(dE, [0.0, 0.4, 0.8, 0.5, 0.3])

    def test_peak_is_third_image(self, result):
        _, _, dE = result
        assert dE.argmax() == 2

    def test_returns_numpy_arrays(self, result):
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_neb_path(str(tmp_path / 'ghost.dat'))

    def test_comment_and_blank_lines_skipped(self, tmp_path):
        txt = '# header\n\n0.0 -100.0 0.0\n1.0 -99.5 0.5\n'
        f = _write(tmp_path / 'short.dat', txt)
        frac, E, dE = parse_neb_path(f)
        assert len(frac) == 2

    def test_partial_lines_skipped(self, tmp_path):
        txt = '0.0 -100.0 0.0\nbad line\n1.0 -99.5 0.5\n'
        f = _write(tmp_path / 'partial.dat', txt)
        frac, _, _ = parse_neb_path(f)
        assert len(frac) == 2


# ═════════════════════════════════════════════════════════════════════════════
# 10. parse_dissociated_h2_log
# ═════════════════════════════════════════════════════════════════════════════

# Synthetic adsorbate minimization log — H2 dissociates (energy drops monotonically)
_DISS_LOG_MONOTONIC = """\
LAMMPS (29 Sep 2021)

Step PotEng Fmax Fnorm Press
0 -1200.0 5.20 8.10 10.0
100 -1210.0 2.10 3.50 9.5
200 -1218.0 0.80 1.20 9.1
300 -1220.0 0.12 0.18 9.0
Loop time of 8.0

### Minimization complete ###

  pe_final_eV     : -1220.0
  fmax_eV_per_Ang : 0.12
  natoms          : 362
"""

# Synthetic log — apparent local maximum (energy rises briefly then falls)
_DISS_LOG_WITH_BARRIER = """\
LAMMPS (29 Sep 2021)

Step PotEng Fmax Fnorm Press
0 -1200.0 5.20 8.10 10.0
50  -1198.0 4.80 7.50 10.1
100 -1210.0 2.10 3.50 9.5
200 -1220.0 0.12 0.18 9.0
Loop time of 8.0

### Minimization complete ###

  pe_final_eV     : -1220.0
  fmax_eV_per_Ang : 0.12
  natoms          : 362
"""

# Log with only a single thermo row (should return None)
_DISS_LOG_SINGLE_ROW = """\
Step PotEng Fmax Fnorm Press
0 -1200.0 5.20 8.10 10.0
Loop time of 0.1
"""


class TestParseDissociatedH2Log:

    @pytest.fixture()
    def result_monotonic(self, tmp_path):
        f = _write(tmp_path / 'h2_min_s_1.log', _DISS_LOG_MONOTONIC)
        return parse_dissociated_h2_log(f)

    @pytest.fixture()
    def result_with_barrier(self, tmp_path):
        f = _write(tmp_path / 'h2_min_s_2.log', _DISS_LOG_WITH_BARRIER)
        return parse_dissociated_h2_log(f)

    # ── return type ──────────────────────────────────────────────────────────

    def test_returns_dict(self, result_monotonic):
        assert isinstance(result_monotonic, dict)

    def test_required_keys_present(self, result_monotonic):
        for key in ('e_initial_eV', 'e_final_eV', 'delta_e_eV',
                    'e_max_eV', 'apparent_barrier_eV', 'has_local_max'):
            assert key in result_monotonic, f'missing key: {key}'

    # ── monotonic (downhill) log ──────────────────────────────────────────────

    def test_e_initial_is_first_pe(self, result_monotonic):
        assert math.isclose(result_monotonic['e_initial_eV'], -1200.0)

    def test_e_final_is_last_pe(self, result_monotonic):
        assert math.isclose(result_monotonic['e_final_eV'], -1220.0)

    def test_delta_e_is_e_final_minus_e_initial(self, result_monotonic):
        expected = -1220.0 - (-1200.0)   # -20.0
        assert math.isclose(result_monotonic['delta_e_eV'], expected)

    def test_delta_e_negative_for_exothermic(self, result_monotonic):
        assert result_monotonic['delta_e_eV'] < 0.0

    def test_e_max_equals_e_initial_for_monotonic(self, result_monotonic):
        assert math.isclose(result_monotonic['e_max_eV'], -1200.0)

    def test_apparent_barrier_zero_for_monotonic(self, result_monotonic):
        assert math.isclose(result_monotonic['apparent_barrier_eV'], 0.0)

    def test_has_local_max_false_for_monotonic(self, result_monotonic):
        assert result_monotonic['has_local_max'] is False

    # ── log with apparent local maximum ──────────────────────────────────────

    def test_e_initial_with_barrier(self, result_with_barrier):
        assert math.isclose(result_with_barrier['e_initial_eV'], -1200.0)

    def test_e_final_with_barrier(self, result_with_barrier):
        assert math.isclose(result_with_barrier['e_final_eV'], -1220.0)

    def test_e_max_is_peak(self, result_with_barrier):
        # step=50 has PE=-1198.0, which is the highest value
        assert math.isclose(result_with_barrier['e_max_eV'], -1198.0)

    def test_apparent_barrier_positive(self, result_with_barrier):
        # -1198.0 - (-1200.0) = 2.0 eV
        assert math.isclose(result_with_barrier['apparent_barrier_eV'], 2.0)

    def test_has_local_max_true(self, result_with_barrier):
        assert result_with_barrier['has_local_max'] is True

    # ── edge cases ───────────────────────────────────────────────────────────

    def test_missing_file_returns_none(self, tmp_path):
        result = parse_dissociated_h2_log(str(tmp_path / 'nonexistent.log'))
        assert result is None

    def test_single_thermo_row_returns_none(self, tmp_path):
        f = _write(tmp_path / 'single.log', _DISS_LOG_SINGLE_ROW)
        assert parse_dissociated_h2_log(f) is None

    def test_empty_file_returns_none(self, tmp_path):
        f = _write(tmp_path / 'empty.log', '')
        assert parse_dissociated_h2_log(f) is None


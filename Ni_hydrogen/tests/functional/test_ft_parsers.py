"""
tests/functional/test_ft_parsers.py
=====================================
Category B functional tests — parser correctness with realistic LAMMPS output.

Unlike tests/test_parsers.py (minimal synthetic fixtures), these tests replicate
the exact log format produced by the project's LAMMPS and ASE-NEB scripts,
including the LAMMPS echo-line behaviour where every ``print`` command appears
verbatim in the log immediately before its numeric output.

No GPU, no SLURM, no MACE required.
"""

import math
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.parsers import parse_energy_log, parse_barrier_file, parse_neb_path


# ─── helper ──────────────────────────────────────────────────────────────────

def _write(path, text):
    path.write_text(text)
    return str(path)


# ─── inline copy of _parse_pe_final logic from ase_neb.py ────────────────────
# The real function is injected as a string template inside
# write_ase_neb_script() (models/ase_neb.py lines 533-545); it cannot be
# imported directly.  This copy is used to test the accumulator logic itself.

def _parse_pe_final_impl(log_path):
    _val = None
    with open(log_path) as _f:
        for _line in _f:
            if 'pe_final_eV' in _line and ':' in _line:
                try:
                    _val = float(_line.split(':')[1].strip())
                except ValueError:
                    pass
    if _val is None:
        raise ValueError(f"pe_final_eV not found in {log_path}")
    return _val


# ─── realistic fixtures ───────────────────────────────────────────────────────
#
# _FS_MIN_LOG: Mirrors a real LAMMPS CG-minimization log.
#   The critical echo-line pattern appears after the minimisation stats block:
#
#   print "  pe_final_eV     : ${pe_final}"   ← echoed verbatim (not numeric)
#     pe_final_eV     : -2271.05882318558     ← actual output (numeric)
#
#   A parser that returns on the first match hits the echo line and raises
#   ValueError on float("${pe_final}").  The accumulator pattern (try/except +
#   overwrite) silently discards the echo line and keeps the numeric value.

_FS_MIN_LOG = """\
LAMMPS (29 Aug 2024 - Update 1)
KOKKOS mode with 1 GPU and 1 thread(s) per MPI task
  using KOKKOS_DEVICES=Cuda KOKKOS_ARCH=Volta70
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread.
  using 1 OpenMP thread(s) per MPI task
Per MPI rank memory allocation (min/avg/max) = 1024 | 1024 | 1024 Mbytes
Step PotEng Fmax Fnorm Press
0 -2265.12345678 8.34215 45.32192 -1234.567
100 -2270.00123456 0.00423 0.01234 -987.654
200 -2271.05882318558 1.85109507932005e-07 5.12e-07 -956.234
Loop time of 45.2341 on 1 procs for 200 steps with 362 atoms
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = -2265.123 -2271.058 -2271.05882318558
  Force two-norm initial, final = 45.32192 5.12e-07
  Force max-component initial, final = 8.34215 1.85109507932005e-07
  Final line search alpha, max atom move = 1 1.85109507932005e-07
  Iterations, force evaluations = 200 400
print "### Minimization complete ###"
### Minimization complete ###
print "  pe_final_eV     : ${pe_final}"
  pe_final_eV     : -2271.05882318558
print "  fmax_eV_per_Ang : ${fmax}"
  fmax_eV_per_Ang : 1.85109507932005e-07
print "  natoms          : ${n_atoms}"
  natoms          : 362
Total wall time: 0:00:45
"""

# _NEB_BARRIER: Matches the exact format written by write_neb_results()
# in models/ase_neb.py (lines 402-418), including "Converged" (capital C)
# and unit suffixes "eV" / "eV/A" after numeric values.

_NEB_BARRIER = """\
IS          : s_40
FS          : s_89+s_90
N images    : 20 (18 intermediate + IS + FS)
fmax_final  : 0.0023 eV/A
Converged   : True

E_IS    = -2271.058823 eV
E_FS    = -2271.118823 eV
E_abs   = 0.5134 eV
E_des   = 0.4534 eV
delta_E = -0.0600 eV

Image energies:
  IS    : -2271.058823 eV  (+0.0000 eV)
  img_1 : -2270.963456 eV  (+0.0954 eV)
  img_2 : -2270.823456 eV  (+0.2354 eV)
  img_3 : -2270.623456 eV  (+0.4354 eV)
  img_4 : -2270.523456 eV  (+0.5354 eV)
  img_5 : -2270.545490 eV  (+0.5134 eV)
  img_6 : -2270.623456 eV  (+0.4354 eV)
  img_7 : -2270.703456 eV  (+0.3554 eV)
  img_8 : -2270.783456 eV  (+0.2754 eV)
  img_9 : -2270.853456 eV  (+0.2054 eV)
  img_10: -2270.913456 eV  (+0.1454 eV)
  img_11: -2270.963456 eV  (+0.0954 eV)
  img_12: -2271.003456 eV  (+0.0554 eV)
  img_13: -2271.033456 eV  (+0.0254 eV)
  img_14: -2271.053456 eV  (+0.0054 eV)
  img_15: -2271.063456 eV  (-0.0046 eV)
  img_16: -2271.083456 eV  (-0.0246 eV)
  img_17: -2271.103456 eV  (-0.0446 eV)
  FS    : -2271.118823 eV  (-0.0600 eV)
"""

# _NEB_PATH: 20-image MEP (18 intermediate + IS + FS).
# frac = i / (n-1), i in 0..19, so step ≈ 0.0526.
# TS is at image 4 (frac ≈ 0.2105), where dE is highest.

_NEB_PATH = """\
# image_frac  E_eV  dE_from_IS_eV
0.0000  -2271.058823  +0.0000
0.0526  -2270.963456  +0.0954
0.1053  -2270.823456  +0.2354
0.1579  -2270.623456  +0.4354
0.2105  -2270.523456  +0.5354
0.2632  -2270.545490  +0.5134
0.3158  -2270.623456  +0.4354
0.3684  -2270.703456  +0.3554
0.4211  -2270.783456  +0.2754
0.4737  -2270.853456  +0.2054
0.5263  -2270.913456  +0.1454
0.5789  -2270.963456  +0.0954
0.6316  -2271.003456  +0.0554
0.6842  -2271.033456  +0.0254
0.7368  -2271.053456  +0.0054
0.7895  -2271.063456  -0.0046
0.8421  -2271.083456  -0.0246
0.8947  -2271.103456  -0.0446
0.9474  -2271.113456  -0.0546
1.0000  -2271.118823  -0.0600
"""


# ═════════════════════════════════════════════════════════════════════════════
# 1. parse_energy_log — realistic LAMMPS format with echo lines
# ═════════════════════════════════════════════════════════════════════════════

class TestParseEnergyLogRealisticFormat:

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'fs_min.log', _FS_MIN_LOG)
        return parse_energy_log(f)

    def test_returns_dict_not_none(self, result):
        assert result is not None

    def test_pe_final_is_float(self, result):
        assert isinstance(result['pe_final_eV'], float)
        assert not math.isnan(result['pe_final_eV'])

    def test_pe_final_correct_value(self, result):
        assert math.isclose(result['pe_final_eV'], -2271.05882318558, rel_tol=1e-9)

    def test_fmax_correct_value(self, result):
        assert math.isclose(result['fmax_eV_per_Ang'], 1.85109507932005e-07, rel_tol=1e-6)

    def test_natoms_correct(self, result):
        assert result['natoms'] == 362.0

    def test_echo_line_does_not_corrupt_value(self, result):
        # The echo line contains "${pe_final}" — if mistakenly parsed it would
        # raise ValueError (caught) and leave pe_final_eV unset or wrong.
        # A negative float confirms the real line was parsed, not the echo.
        assert result['pe_final_eV'] < 0


# ═════════════════════════════════════════════════════════════════════════════
# 2. _parse_pe_final accumulator — echo-line regression
# ═════════════════════════════════════════════════════════════════════════════

class TestParsePeFinalAccumulator:
    """
    Tests the accumulator logic used in the _parse_pe_final function that is
    injected into every generated run_neb.py script.

    Approach A: test the replicated logic (_parse_pe_final_impl above).
    Approach B: regression guard — verify the source of ase_neb.py still
                contains the accumulator pattern so future edits cannot
                silently revert the fix.
    """

    @pytest.fixture()
    def log_path(self, tmp_path):
        return _write(tmp_path / 'fs_min.log', _FS_MIN_LOG)

    @pytest.fixture()
    def log_no_pe(self, tmp_path):
        return _write(tmp_path / 'empty.log', "LAMMPS output\nLoop time of 1.0\n")

    def test_accumulator_returns_correct_float(self, log_path):
        val = _parse_pe_final_impl(log_path)
        assert math.isclose(val, -2271.05882318558, rel_tol=1e-9)

    def test_does_not_raise_on_echo_line(self, log_path):
        # Would raise ValueError if echo line ("${pe_final}") triggered a bare
        # return instead of being caught by the try/except accumulator.
        _parse_pe_final_impl(log_path)  # must not raise

    def test_raises_if_key_absent(self, log_no_pe):
        with pytest.raises(ValueError, match='pe_final_eV not found'):
            _parse_pe_final_impl(log_no_pe)

    def test_ase_neb_template_has_accumulator_pattern(self):
        src_path = os.path.join(PROJECT_ROOT, 'models', 'ase_neb.py')
        src = open(src_path).read()
        assert "'    _val = None\\n'" in src, (
            "ase_neb.py _parse_pe_final template is missing '_val = None' — "
            "the accumulator fix may have been reverted"
        )
        assert "'    return _val\\n'" in src, (
            "ase_neb.py _parse_pe_final template is missing 'return _val' — "
            "the accumulator fix may have been reverted"
        )

    def test_ase_neb_template_no_bare_return_float(self):
        src_path = os.path.join(PROJECT_ROOT, 'models', 'ase_neb.py')
        src = open(src_path).read()
        # The old broken pattern was "return float(...)" directly in the loop.
        # It should not appear in the _parse_pe_final block.
        assert "'            return float(" not in src, (
            "ase_neb.py _parse_pe_final template contains a bare 'return float(' "
            "inside the loop — this is the old broken pattern that returns on the "
            "first match (echo line) and raises ValueError"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. parse_barrier_file — realistic write_neb_results format
# ═════════════════════════════════════════════════════════════════════════════

class TestParseBarrierFileRealisticFormat:
    """
    Tests parse_barrier_file against the exact format written by
    write_neb_results() in models/ase_neb.py, which includes:
      - "Converged" (capital C) — previously failed due to case-sensitive check
      - Unit suffixes "eV" and "eV/A" after numeric values — previously failed
        because float(val) cannot parse "0.5134 eV"
    Both bugs were fixed: float(val.split()[0]) and key.lower() == 'converged'.
    """

    @pytest.fixture()
    def result(self, tmp_path):
        f = _write(tmp_path / 'neb_barrier.txt', _NEB_BARRIER)
        return parse_barrier_file(f)

    def test_E_abs_extracted(self, result):
        assert math.isclose(result['E_abs'], 0.5134, rel_tol=1e-6)

    def test_E_des_extracted(self, result):
        assert math.isclose(result['E_des'], 0.4534, rel_tol=1e-6)

    def test_delta_E_extracted(self, result):
        assert math.isclose(result['delta_E'], -0.0600, rel_tol=1e-4)

    def test_E_IS_extracted(self, result):
        assert math.isclose(result['E_IS'], -2271.058823, rel_tol=1e-9)

    def test_E_FS_extracted(self, result):
        assert math.isclose(result['E_FS'], -2271.118823, rel_tol=1e-9)

    def test_fmax_final_extracted(self, result):
        assert math.isclose(result['fmax_final'], 0.0023, rel_tol=1e-4)

    def test_converged_is_bool_true(self, result):
        # "Converged   : True" — capital C, parsed via key.lower() == 'converged'
        assert result.get('converged') is True

    def test_image_lines_do_not_corrupt_E_abs(self, result):
        # Image energy lines like "img_4 : -2270.523456 eV  (+0.5354 eV)"
        # have "img_4" as key — not in _FLOAT_KEYS — so they must not overwrite E_abs.
        assert math.isclose(result['E_abs'], 0.5134, rel_tol=1e-6)

    def test_header_comment_keys_not_in_result(self, result):
        # "IS : s_40", "FS : s_89+s_90", "N images : ..." are header lines
        # and must not appear as numeric keys.
        for spurious_key in ('IS', 'FS', 'N images'):
            assert spurious_key not in result


# ═════════════════════════════════════════════════════════════════════════════
# 4. parse_neb_path — realistic 20-image MEP format
# ═════════════════════════════════════════════════════════════════════════════

class TestParseNebPathRealisticFormat:

    @pytest.fixture()
    def arrays(self, tmp_path):
        f = _write(tmp_path / 'neb_path.dat', _NEB_PATH)
        return parse_neb_path(f)

    def test_returns_three_numpy_arrays(self, arrays):
        frac, E, dE = arrays
        for arr in (frac, E, dE):
            assert isinstance(arr, np.ndarray)

    def test_array_length_matches_n_images(self, arrays):
        frac, _, _ = arrays
        assert len(frac) == 20

    def test_first_frac_is_zero(self, arrays):
        frac, _, _ = arrays
        assert math.isclose(frac[0], 0.0, abs_tol=1e-6)

    def test_last_frac_is_one(self, arrays):
        frac, _, _ = arrays
        assert math.isclose(frac[-1], 1.0, rel_tol=1e-6)

    def test_dE_IS_is_zero(self, arrays):
        _, _, dE = arrays
        assert math.isclose(dE[0], 0.0, abs_tol=1e-6)

    def test_peak_dE_is_interior_image(self, arrays):
        _, _, dE = arrays
        peak = int(np.argmax(dE))
        assert 0 < peak < len(dE) - 1, (
            f"TS peak at image {peak} — should be interior, not IS (0) or FS ({len(dE)-1})"
        )

    def test_peak_dE_matches_E_abs(self, arrays):
        _, _, dE = arrays
        assert math.isclose(dE.max(), 0.5354, rel_tol=1e-4)

    def test_FS_dE_matches_delta_E(self, arrays):
        _, _, dE = arrays
        assert math.isclose(dE[-1], -0.0600, rel_tol=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# 5. parse_energy_log — LAMMPS log format variants
# ═════════════════════════════════════════════════════════════════════════════

# Bare LAMMPS log with no KOKKOS preamble — plain MPI run
_FS_MIN_LOG_NO_KOKKOS = """\
LAMMPS (29 Aug 2024 - Update 1)
Per MPI rank memory allocation (min/avg/max) = 1024 | 1024 | 1024 Mbytes
Step PotEng Fmax Fnorm Press
0 -2265.12345678 8.34215 45.32192 -1234.567
100 -2270.00123456 0.00423 0.01234 -987.654
200 -2271.09999999 1.11e-07 3.14e-07 -956.234
Loop time of 38.1234 on 1 procs for 200 steps with 362 atoms
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = -2265.123 -2271.099 -2271.09999999
  Force two-norm initial, final = 45.32192 1.11e-07
  Force max-component initial, final = 8.34215 1.11e-07
  Final line search alpha, max atom move = 1 1.11e-07
  Iterations, force evaluations = 200 400
print "### Minimization complete ###"
### Minimization complete ###
print "  pe_final_eV     : ${pe_final}"
  pe_final_eV     : -2271.09999999
print "  fmax_eV_per_Ang : ${fmax}"
  fmax_eV_per_Ang : 1.11e-07
print "  natoms          : ${n_atoms}"
  natoms          : 362
Total wall time: 0:00:38
"""

# LAMMPS log with WARNING lines interspersed between thermo output
_FS_MIN_LOG_WITH_WARNINGS = """\
LAMMPS (29 Aug 2024 - Update 1)
KOKKOS mode with 1 GPU and 1 thread(s) per MPI task
  using KOKKOS_DEVICES=Cuda KOKKOS_ARCH=Volta70
WARNING: Ignoring unknown variable ploop (src/variable.cpp:756)
WARNING: Neighbor list request was already made (src/pair.cpp:221)
Per MPI rank memory allocation (min/avg/max) = 1024 | 1024 | 1024 Mbytes
Step PotEng Fmax Fnorm Press
0 -2265.12345678 8.34215 45.32192 -1234.567
WARNING: Energy not converging -- taking smaller steps (src/min.cpp:300)
100 -2270.00123456 0.00423 0.01234 -987.654
200 -2271.07777777 9.87e-08 2.71e-07 -956.234
Loop time of 41.5678 on 1 procs for 200 steps with 362 atoms
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = -2265.123 -2271.077 -2271.07777777
  Force two-norm initial, final = 45.32192 9.87e-08
  Force max-component initial, final = 8.34215 9.87e-08
  Final line search alpha, max atom move = 1 9.87e-08
  Iterations, force evaluations = 200 400
print "### Minimization complete ###"
### Minimization complete ###
print "  pe_final_eV     : ${pe_final}"
  pe_final_eV     : -2271.07777777
print "  fmax_eV_per_Ang : ${fmax}"
  fmax_eV_per_Ang : 9.87e-08
print "  natoms          : ${n_atoms}"
  natoms          : 362
Total wall time: 0:00:41
"""

# LAMMPS log with TWO complete minimize sections — parser must return the LAST value
_FS_MIN_LOG_MULTI_BLOCK = """\
LAMMPS (29 Aug 2024 - Update 1)
KOKKOS mode with 1 GPU and 1 thread(s) per MPI task
Per MPI rank memory allocation (min/avg/max) = 512 | 512 | 512 Mbytes
Step PotEng Fmax Fnorm Press
0 -2260.00000000 12.5 60.0 -2000.0
50 -2264.11111111 5.2 25.0 -1500.0
100 -2266.22222222 0.5 2.0 -1100.0
Loop time of 20.0 on 1 procs for 100 steps with 362 atoms
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = -2260.0 -2266.0 -2266.22222222
  Force two-norm initial, final = 60.0 2.0
  Force max-component initial, final = 12.5 0.5
  Final line search alpha, max atom move = 1 0.5
  Iterations, force evaluations = 100 200
print "  pe_final_eV     : ${pe_final}"
  pe_final_eV     : -2266.22222222
print "  fmax_eV_per_Ang : ${fmax}"
  fmax_eV_per_Ang : 0.5
print "  natoms          : ${n_atoms}"
  natoms          : 362
Step PotEng Fmax Fnorm Press
0 -2266.22222222 0.5 2.0 -1100.0
50 -2270.33333333 0.1 0.5 -990.0
150 -2271.44444444 6.66e-08 1.99e-07 -955.0
Loop time of 35.0 on 1 procs for 150 steps with 362 atoms
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = -2266.222 -2271.444 -2271.44444444
  Force two-norm initial, final = 2.0 6.66e-08
  Force max-component initial, final = 0.5 6.66e-08
  Final line search alpha, max atom move = 1 6.66e-08
  Iterations, force evaluations = 150 300
print "  pe_final_eV     : ${pe_final}"
  pe_final_eV     : -2271.44444444
print "  fmax_eV_per_Ang : ${fmax}"
  fmax_eV_per_Ang : 6.66e-08
print "  natoms          : ${n_atoms}"
  natoms          : 362
Total wall time: 0:00:55
"""

# LAMMPS log with KOKKOS/OpenMP (CPU threads, no GPU)
_FS_MIN_LOG_OPENMP = """\
LAMMPS (29 Aug 2024 - Update 1)
KOKKOS mode with 8 OpenMP thread(s) per MPI task
  using KOKKOS_DEVICES=OpenMP KOKKOS_ARCH=SKX
OMP_NUM_THREADS environment set to 8.
  using 8 OpenMP thread(s) per MPI task
Per MPI rank memory allocation (min/avg/max) = 2048 | 2048 | 2048 Mbytes
Step PotEng Fmax Fnorm Press
0 -2265.12345678 8.34215 45.32192 -1234.567
100 -2270.00123456 0.00423 0.01234 -987.654
200 -2271.05555555 2.22e-07 6.66e-07 -956.234
Loop time of 52.3456 on 8 procs for 200 steps with 362 atoms
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = -2265.123 -2271.055 -2271.05555555
  Force two-norm initial, final = 45.32192 2.22e-07
  Force max-component initial, final = 8.34215 2.22e-07
  Final line search alpha, max atom move = 1 2.22e-07
  Iterations, force evaluations = 200 400
print "### Minimization complete ###"
### Minimization complete ###
print "  pe_final_eV     : ${pe_final}"
  pe_final_eV     : -2271.05555555
print "  fmax_eV_per_Ang : ${fmax}"
  fmax_eV_per_Ang : 2.22e-07
print "  natoms          : ${n_atoms}"
  natoms          : 362
Total wall time: 0:00:52
"""


class TestParseEnergyLogVariants:
    """
    parse_energy_log must handle the full range of LAMMPS header variations
    seen in practice: plain MPI (no KOKKOS), WARNING lines, multi-block logs
    from two-stage minimization, and KOKKOS/OpenMP (CPU-thread) runs.
    """

    def test_no_kokkos_returns_correct_pe(self, tmp_path):
        f = _write(tmp_path / 'no_kokkos.log', _FS_MIN_LOG_NO_KOKKOS)
        result = parse_energy_log(f)
        assert result is not None
        assert math.isclose(result['pe_final_eV'], -2271.09999999, rel_tol=1e-9)

    def test_no_kokkos_correct_natoms(self, tmp_path):
        f = _write(tmp_path / 'no_kokkos.log', _FS_MIN_LOG_NO_KOKKOS)
        result = parse_energy_log(f)
        assert result['natoms'] == 362.0

    def test_warning_lines_skipped(self, tmp_path):
        f = _write(tmp_path / 'warnings.log', _FS_MIN_LOG_WITH_WARNINGS)
        result = parse_energy_log(f)
        assert result is not None
        assert math.isclose(result['pe_final_eV'], -2271.07777777, rel_tol=1e-9)

    def test_warning_lines_do_not_corrupt_fmax(self, tmp_path):
        f = _write(tmp_path / 'warnings.log', _FS_MIN_LOG_WITH_WARNINGS)
        result = parse_energy_log(f)
        assert math.isclose(result['fmax_eV_per_Ang'], 9.87e-08, rel_tol=1e-6)

    def test_multi_block_returns_last_pe(self, tmp_path):
        # Two complete minimize blocks: first ends at -2266.222, second at -2271.444.
        # The parser must return the LAST value (-2271.444).
        f = _write(tmp_path / 'multi.log', _FS_MIN_LOG_MULTI_BLOCK)
        result = parse_energy_log(f)
        assert result is not None
        assert math.isclose(result['pe_final_eV'], -2271.44444444, rel_tol=1e-9), (
            f"pe_final={result['pe_final_eV']:.8f} — multi-block log must return last block's value"
        )

    def test_multi_block_last_fmax(self, tmp_path):
        f = _write(tmp_path / 'multi.log', _FS_MIN_LOG_MULTI_BLOCK)
        result = parse_energy_log(f)
        assert math.isclose(result['fmax_eV_per_Ang'], 6.66e-08, rel_tol=1e-6)

    def test_openmp_kokkos_returns_correct_pe(self, tmp_path):
        f = _write(tmp_path / 'openmp.log', _FS_MIN_LOG_OPENMP)
        result = parse_energy_log(f)
        assert result is not None
        assert math.isclose(result['pe_final_eV'], -2271.05555555, rel_tol=1e-9)

    def test_openmp_kokkos_correct_natoms(self, tmp_path):
        f = _write(tmp_path / 'openmp.log', _FS_MIN_LOG_OPENMP)
        result = parse_energy_log(f)
        assert result['natoms'] == 362.0

    def test_all_variants_return_negative_pe(self, tmp_path):
        logs = [
            (tmp_path / 'v1.log', _FS_MIN_LOG_NO_KOKKOS),
            (tmp_path / 'v2.log', _FS_MIN_LOG_WITH_WARNINGS),
            (tmp_path / 'v3.log', _FS_MIN_LOG_MULTI_BLOCK),
            (tmp_path / 'v4.log', _FS_MIN_LOG_OPENMP),
        ]
        for p, content in logs:
            f = _write(p, content)
            result = parse_energy_log(f)
            assert result['pe_final_eV'] < 0, f"Expected negative pe_final for {p.name}"

    def test_all_variants_return_dict_with_required_keys(self, tmp_path):
        required = {'pe_final_eV', 'fmax_eV_per_Ang', 'natoms'}
        logs = [
            (tmp_path / 'w1.log', _FS_MIN_LOG_NO_KOKKOS),
            (tmp_path / 'w2.log', _FS_MIN_LOG_WITH_WARNINGS),
            (tmp_path / 'w3.log', _FS_MIN_LOG_MULTI_BLOCK),
            (tmp_path / 'w4.log', _FS_MIN_LOG_OPENMP),
        ]
        for p, content in logs:
            f = _write(p, content)
            result = parse_energy_log(f)
            assert result is not None
            missing = required - result.keys()
            assert not missing, f"{p.name} result missing keys: {missing}"

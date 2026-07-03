"""
tests/functional/test_ft_diffusivity.py
=========================================
Category D functional tests — Arrhenius fit math (Section 4: Diffusivity).
Category B functional tests — diffusivity file parser with realistic fixture.

No LAMMPS, no GPU, no cluster files required.

Category D tests feed synthetic D(T) data with known D0 and Ea into
fit_arrhenius() and assert the recovered parameters are within 2%.

Category B tests write a realistic diffusivity.txt fixture (matching the
format produced by save_diffusivity_table()) and assert parse_diffusivity_file()
extracts correct values.
"""

import math
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.diffusivity_post_processing import fit_arrhenius
from models.parsers import parse_diffusivity_file

_KB = 8.617333262e-5   # eV/K


# ─── synthetic diffusivity fixture ───────────────────────────────────────────

_DIFF_FILE_CONTENT = """\
# Hydrogen self-diffusivity from MSD linear fit
# D = slope/6  (Einstein relation, 3D)
# Fit window: 20-80% of trajectory
#
#        T_K    D (m2/s)    sigma_D (m2/s)       R2
# ========  ==============  ================  ========
  400.0    3.221400e-10    1.610700e-11    0.9991
  600.0    2.187300e-09    1.093650e-10    0.9988
  800.0    8.234100e-09    4.117050e-10    0.9985
 1000.0    2.156700e-08    1.078350e-09    0.9979
"""


def _make_synthetic_D(T_K, D0, Ea_eV):
    """True Arrhenius diffusivity at the given temperatures."""
    return D0 * np.exp(-Ea_eV / (_KB * T_K))


# ═════════════════════════════════════════════════════════════════════════════
# Category D — Arrhenius fit math
# ═════════════════════════════════════════════════════════════════════════════

class TestArrheniusFitKnownAnswer:
    """
    fit_arrhenius(T, D, D_err) must recover D0 and Ea within 2% on perfect
    synthetic data.  Tests also verify error/R2 contract and edge behaviour.
    """

    @pytest.fixture()
    def perfect_fit(self):
        D0_true = 1e-7    # m²/s — typical bulk Ni pre-exponential
        Ea_true = 0.4     # eV   — typical H-in-Ni activation energy
        T = np.array([400., 500., 600., 700., 800., 900., 1000.])
        D = _make_synthetic_D(T, D0_true, Ea_true)
        D_err = 0.01 * D  # 1% relative errors → well-conditioned weights
        Ea, Ea_err, D0, D0_err, R2 = fit_arrhenius(T, D, D_err)
        return dict(Ea=Ea, Ea_err=Ea_err, D0=D0, D0_err=D0_err, R2=R2,
                    Ea_true=Ea_true, D0_true=D0_true)

    def test_Ea_recovered_within_2pct(self, perfect_fit):
        assert math.isclose(perfect_fit['Ea'], perfect_fit['Ea_true'], rel_tol=0.02), (
            f"Ea={perfect_fit['Ea']:.4f} eV, expected {perfect_fit['Ea_true']:.4f} eV"
        )

    def test_D0_recovered_within_2pct(self, perfect_fit):
        assert math.isclose(perfect_fit['D0'], perfect_fit['D0_true'], rel_tol=0.02), (
            f"D0={perfect_fit['D0']:.3e} m²/s, expected {perfect_fit['D0_true']:.3e} m²/s"
        )

    def test_R2_near_unity_for_perfect_data(self, perfect_fit):
        assert perfect_fit['R2'] > 0.9999, (
            f"R2={perfect_fit['R2']:.6f} should be >0.9999 for exact Arrhenius data"
        )

    def test_Ea_err_is_positive_float(self, perfect_fit):
        assert perfect_fit['Ea_err'] > 0

    def test_D0_err_is_positive_float(self, perfect_fit):
        assert perfect_fit['D0_err'] > 0

    def test_Ea_positive(self, perfect_fit):
        assert perfect_fit['Ea'] > 0

    def test_D0_positive(self, perfect_fit):
        assert perfect_fit['D0'] > 0

    def test_higher_Ea_lower_D_at_low_T(self):
        T = np.array([500., 700., 900.])
        D_low_Ea  = _make_synthetic_D(T, 1e-7, 0.2)
        D_high_Ea = _make_synthetic_D(T, 1e-7, 0.8)
        err = 0.01 * D_low_Ea
        Ea_low,  *_ = fit_arrhenius(T, D_low_Ea,  err)
        Ea_high, *_ = fit_arrhenius(T, D_high_Ea, 0.01 * D_high_Ea)
        assert Ea_high > Ea_low, "Higher true Ea must produce higher fitted Ea"

    def test_D0_scales_correctly(self):
        T   = np.array([600., 700., 800., 900.])
        D1  = _make_synthetic_D(T, 1e-7, 0.4)
        D2  = _make_synthetic_D(T, 1e-6, 0.4)   # 10× larger D0, same Ea
        err1 = 0.01 * D1
        err2 = 0.01 * D2
        _, _, D0_fit1, _, _ = fit_arrhenius(T, D1, err1)
        _, _, D0_fit2, _, _ = fit_arrhenius(T, D2, err2)
        assert math.isclose(D0_fit2 / D0_fit1, 10.0, rel_tol=0.02), (
            "10× D0 scaling must be recovered within 2%"
        )

    def test_Ea_invariant_to_D0_scaling(self):
        T    = np.array([500., 700., 900.])
        D1   = _make_synthetic_D(T, 1e-7, 0.4)
        D100 = _make_synthetic_D(T, 1e-5, 0.4)   # 100× D0
        Ea1,  *_ = fit_arrhenius(T, D1,   0.01 * D1)
        Ea100,*_ = fit_arrhenius(T, D100, 0.01 * D100)
        assert math.isclose(Ea1, Ea100, rel_tol=1e-6), (
            "Ea must not depend on the D0 prefactor"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Category B — diffusivity file parser with realistic fixture
# ═════════════════════════════════════════════════════════════════════════════

class TestParseDiffusivityFileRealisticFormat:
    """
    parse_diffusivity_file() must correctly extract T, D, sigma_D, R2 from
    the format produced by save_diffusivity_table().

    The fixture includes a full header block (comment lines, separator lines,
    column-label line) plus 4 data rows — exactly what the writer emits.
    """

    @pytest.fixture()
    def parsed(self, tmp_path):
        p = tmp_path / 'diffusivity.txt'
        p.write_text(_DIFF_FILE_CONTENT)
        return parse_diffusivity_file(str(p))

    def test_returns_four_arrays(self, parsed):
        assert len(parsed) == 4, "parse_diffusivity_file must return (T, D, D_err, R2)"

    def test_four_rows_extracted(self, parsed):
        T, D, Derr, R2 = parsed
        assert len(T) == 4

    def test_all_same_length(self, parsed):
        T, D, Derr, R2 = parsed
        assert len(T) == len(D) == len(Derr) == len(R2)

    def test_temperatures_correct(self, parsed):
        T, *_ = parsed
        expected = [400., 600., 800., 1000.]
        for t_fit, t_exp in zip(T, expected):
            assert math.isclose(t_fit, t_exp), f"T={t_fit} != {t_exp}"

    def test_D_values_correct(self, parsed):
        _, D, *_ = parsed
        expected = [3.2214e-10, 2.1873e-9, 8.2341e-9, 2.1567e-8]
        for d_fit, d_exp in zip(D, expected):
            assert math.isclose(d_fit, d_exp, rel_tol=1e-5), f"D={d_fit:.4e} != {d_exp:.4e}"

    def test_D_err_is_1pct_of_D(self, parsed):
        _, D, Derr, _ = parsed
        for d, de in zip(D, Derr):
            assert math.isclose(de / d, 0.05, rel_tol=1e-5), (
                "sigma_D should be ~5% of D in this fixture"
            )

    def test_R2_values_all_above_0_99(self, parsed):
        *_, R2 = parsed
        for r in R2:
            assert r > 0.99

    def test_all_D_positive(self, parsed):
        _, D, *_ = parsed
        for d in D:
            assert d > 0

    def test_returns_ndarrays(self, parsed):
        for arr in parsed:
            assert isinstance(arr, np.ndarray)

    def test_header_lines_not_in_output(self, parsed):
        T, *_ = parsed
        # Header rows with non-numeric content must not appear as data
        assert not any(math.isnan(t) for t in T)
        assert not any(t < 0 for t in T)

"""
tests/test_diffusivity_post_processing.py
==========================================
Tests for models/diffusivity_post_processing.py — offline, no LAMMPS required.

Numeric fixtures are constructed analytically so expected values are exact.
matplotlib is avoided by skipping plot_arrhenius; pipeline tests use
synthetic dump files written with tmp_path.
"""

import sys
import pathlib
import pytest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.diffusivity_post_processing import (
    KB_EV,
    ANG2_PS_TO_M2S,
    unwrap_trajectory,
    compute_msd,
    fit_diffusivity,
    save_diffusivity_table,
    fit_arrhenius,
    arrhenius_D,
    run_diffusivity_pipeline,
    run_arrhenius_pipeline,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_dump(path, frames):
    """Write a minimal LAMMPS custom dump with one H atom (type 9) per frame."""
    lines = []
    for step, (x, y, z) in enumerate(frames):
        lines += [
            'ITEM: TIMESTEP', str(step * 1000),
            'ITEM: NUMBER OF ATOMS', '2',
            'ITEM: BOX BOUNDS pp pp pp',
            '0.0 20.0', '0.0 20.0', '0.0 20.0',
            'ITEM: ATOMS id type x y z',
            f'1 9 {x} {y} {z}',   # H atom (type 9)
            '2 1 2.0 2.0 2.0',    # metal atom (type 1)
        ]
    pathlib.Path(path).write_text('\n'.join(lines) + '\n')


def _linear_msd(n_frames=200, velocity=1.0, dt=0.5):
    """Build a constant-velocity 1D trajectory and its exact MSD time array."""
    pos = np.zeros((n_frames, 1, 3))
    pos[:, 0, 0] = np.arange(n_frames) * velocity   # x = v * frame
    t_lag = np.arange(1, n_frames // 2 + 1) * dt    # ps
    return pos, t_lag


# ═══════════════════════════════════════════════════════════════════════════
# 1. Physical constants
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:

    def test_kb_ev_value(self):
        assert KB_EV == pytest.approx(8.617333262e-5, rel=1e-12)

    def test_ang2_ps_to_m2s_value(self):
        assert ANG2_PS_TO_M2S == pytest.approx(1e-8, rel=1e-10)

    def test_unit_conversion_product(self):
        # 1 Å²/ps × ANG2_PS_TO_M2S = 1e-8 m²/s
        assert 1.0 * ANG2_PS_TO_M2S == pytest.approx(1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 2. unwrap_trajectory
# ═══════════════════════════════════════════════════════════════════════════

class TestUnwrapTrajectory:

    def _box(self):
        return np.array([10.0, 10.0, 10.0])

    def test_output_shape_matches_input(self):
        pos = np.zeros((5, 3, 3))
        out = unwrap_trajectory(pos, self._box())
        assert out.shape == pos.shape

    def test_static_trajectory_unchanged(self):
        pos = np.ones((8, 2, 3)) * 5.0
        out = unwrap_trajectory(pos, self._box())
        np.testing.assert_allclose(out, pos)

    def test_no_crossing_unchanged(self):
        # particle moves 0.5 Å per frame — never crosses boundary
        n = 6
        pos = np.zeros((n, 1, 3))
        pos[:, 0, 0] = np.arange(n) * 0.5
        out = unwrap_trajectory(pos, self._box())
        np.testing.assert_allclose(out, pos)

    def test_single_positive_x_crossing_unwrapped(self):
        # particle at 9.5, jumps to 0.5 in the next frame (+x crossing)
        pos = np.zeros((3, 1, 3))
        pos[0, 0, 0] = 8.0
        pos[1, 0, 0] = 9.5
        pos[2, 0, 0] = 0.5   # wrapped back after crossing
        out = unwrap_trajectory(pos, self._box())
        # unwrapped frame 2 should be 10.5, not 0.5
        assert out[2, 0, 0] == pytest.approx(10.5)

    def test_single_negative_x_crossing_unwrapped(self):
        # particle at 0.5, jumps to 9.5 in the next frame (-x crossing)
        pos = np.zeros((3, 1, 3))
        pos[0, 0, 0] = 1.5
        pos[1, 0, 0] = 0.5
        pos[2, 0, 0] = 9.5   # wrapped to far end after crossing
        out = unwrap_trajectory(pos, self._box())
        # unwrapped frame 2 should be -0.5, not 9.5
        assert out[2, 0, 0] == pytest.approx(-0.5)

    def test_multiple_crossings_cumulative(self):
        # 0→1: 8→0.5, min-image +2.5, unwrapped=10.5 (1st crossing)
        # 1→2: 0.5→9.0, min-image −1.5, unwrapped=9.0  (goes back)
        # 2→3: 9.0→0.2, min-image +1.2, unwrapped=10.2 (2nd crossing)
        pos = np.zeros((4, 1, 3))
        pos[0, 0, 0] = 8.0
        pos[1, 0, 0] = 0.5
        pos[2, 0, 0] = 9.0
        pos[3, 0, 0] = 0.2
        out = unwrap_trajectory(pos, self._box())
        assert out[1, 0, 0] == pytest.approx(10.5)
        assert out[3, 0, 0] == pytest.approx(10.2)

    def test_multi_atom_each_unwrapped_independently(self):
        pos = np.zeros((3, 2, 3))
        # atom 0: crosses +x
        pos[0, 0, 0] = 9.0; pos[1, 0, 0] = 0.5; pos[2, 0, 0] = 1.0
        # atom 1: stays still
        pos[:, 1, 0] = 5.0
        out = unwrap_trajectory(pos, self._box())
        assert out[1, 0, 0] == pytest.approx(10.5)
        assert out[1, 1, 0] == pytest.approx(5.0)

    def test_y_and_z_crossings_unwrapped(self):
        pos = np.zeros((2, 1, 3))
        pos[0, 0] = [5.0, 9.5, 9.5]
        pos[1, 0] = [5.0, 0.5, 0.5]   # y and z both cross
        out = unwrap_trajectory(pos, self._box())
        assert out[1, 0, 1] == pytest.approx(10.5)
        assert out[1, 0, 2] == pytest.approx(10.5)


# ═══════════════════════════════════════════════════════════════════════════
# 3. compute_msd
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMsd:

    def test_static_trajectory_msd_zero(self):
        pos = np.ones((20, 1, 3)) * 5.0
        lag_frames, msd = compute_msd(pos)
        np.testing.assert_allclose(msd, 0.0, atol=1e-12)

    def test_default_max_lag_is_half_frames(self):
        pos = np.zeros((40, 1, 3))
        lag_frames, msd = compute_msd(pos)
        assert len(lag_frames) == 20

    def test_custom_max_lag_respected(self):
        pos = np.zeros((40, 1, 3))
        lag_frames, msd = compute_msd(pos, max_lag=10)
        assert len(lag_frames) == 10
        assert lag_frames[-1] == 10

    def test_lag_frames_starts_at_one(self):
        pos = np.zeros((10, 1, 3))
        lag_frames, _ = compute_msd(pos)
        assert lag_frames[0] == 1

    def test_constant_velocity_msd_equals_v2_tau2(self):
        # particle at x = v*t; MSD(τ) = v²τ² + 0 + 0
        n = 50
        v = 2.0
        pos = np.zeros((n, 1, 3))
        pos[:, 0, 0] = np.arange(n) * v
        lag_frames, msd = compute_msd(pos)
        expected = (v * lag_frames) ** 2
        np.testing.assert_allclose(msd, expected, rtol=1e-10)

    def test_output_shapes_consistent(self):
        pos = np.zeros((30, 1, 3))
        lag_frames, msd = compute_msd(pos)
        assert lag_frames.shape == msd.shape

    def test_multi_atom_msd_averaged(self):
        # two atoms both move at v=1 in x — MSD same as single atom
        n = 20
        pos = np.zeros((n, 2, 3))
        pos[:, 0, 0] = np.arange(n) * 1.0
        pos[:, 1, 0] = np.arange(n) * 1.0
        lag_frames, msd = compute_msd(pos)
        expected = lag_frames ** 2
        np.testing.assert_allclose(msd, expected, rtol=1e-10)

    def test_msd_nonnegative(self):
        rng = np.random.default_rng(42)
        pos = rng.random((30, 3, 3)) * 10.0
        _, msd = compute_msd(pos)
        assert np.all(msd >= 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. fit_diffusivity
# ═══════════════════════════════════════════════════════════════════════════

class TestFitDiffusivity:

    def _linear_data(self, slope=0.6, n=500):
        """Return (t, MSD) for MSD = slope*t + 5 with n points."""
        t = np.linspace(0.5, 100.0, n)
        msd = slope * t + 5.0
        return t, msd

    def test_returns_three_values(self):
        t, msd = self._linear_data()
        result = fit_diffusivity(t, msd)
        assert len(result) == 3

    def test_perfect_linear_recovers_d(self):
        slope = 0.6  # Å²/ps
        t, msd = self._linear_data(slope=slope)
        D, _, _ = fit_diffusivity(t, msd)
        expected_D = (slope / 6.0) * ANG2_PS_TO_M2S
        assert D == pytest.approx(expected_D, rel=1e-4)

    def test_perfect_linear_r2_near_one(self):
        t, msd = self._linear_data()
        _, _, R2 = fit_diffusivity(t, msd)
        assert R2 == pytest.approx(1.0, abs=1e-8)

    def test_d_positive_for_positive_slope(self):
        t, msd = self._linear_data(slope=0.3)
        D, _, _ = fit_diffusivity(t, msd)
        assert D > 0.0

    def test_sigma_d_nonnegative(self):
        t, msd = self._linear_data()
        _, sigma_D, _ = fit_diffusivity(t, msd)
        assert sigma_D >= 0.0

    def test_custom_fit_window_applied(self):
        # slope=1.0 in [0.2, 0.8], different slope elsewhere
        t = np.linspace(0.0, 100.0, 1000)
        msd = np.where((t >= 20) & (t <= 80), t * 1.0, t * 99.0)
        D_narrow, _, _ = fit_diffusivity(t, msd, fit_window=(0.2, 0.8))
        expected = (1.0 / 6.0) * ANG2_PS_TO_M2S
        assert D_narrow == pytest.approx(expected, rel=0.01)

    def test_d_scales_linearly_with_slope(self):
        t = np.linspace(1.0, 100.0, 500)
        D1, _, _ = fit_diffusivity(t, 0.3 * t, fit_window=(0.0, 1.0))
        D2, _, _ = fit_diffusivity(t, 0.6 * t, fit_window=(0.0, 1.0))
        assert D2 == pytest.approx(2.0 * D1, rel=1e-4)

    def test_units_correct_via_ang2_ps_to_m2s(self):
        # slope of 6.0 Å²/ps → D = 1.0 Å²/ps → D = ANG2_PS_TO_M2S m²/s
        t = np.linspace(1.0, 100.0, 500)
        D, _, _ = fit_diffusivity(t, 6.0 * t, fit_window=(0.0, 1.0))
        assert D == pytest.approx(ANG2_PS_TO_M2S, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# 5. save_diffusivity_table
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveDiffusivityTable:

    @pytest.fixture()
    def diff_file(self, tmp_path):
        p = str(tmp_path / 'diffusivity.txt')
        save_diffusivity_table(
            temperatures=[300.0, 600.0, 900.0],
            D_vals=[1e-9, 3e-9, 7e-9],
            D_errs=[1e-10, 2e-10, 5e-10],
            R2_vals=[0.998, 0.995, 0.990],
            outfile=p,
        )
        return p

    def test_file_created(self, diff_file):
        assert pathlib.Path(diff_file).exists()

    def test_header_lines_present(self, diff_file):
        content = pathlib.Path(diff_file).read_text()
        assert 'T_K' in content
        assert 'D (m2/s)' in content

    def test_temperatures_in_output(self, diff_file):
        content = pathlib.Path(diff_file).read_text()
        assert '300.0' in content
        assert '600.0' in content
        assert '900.0' in content

    def test_d_values_in_output(self, diff_file):
        content = pathlib.Path(diff_file).read_text()
        assert '1.000000e-09' in content
        assert '3.000000e-09' in content

    def test_r2_values_in_output(self, diff_file):
        content = pathlib.Path(diff_file).read_text()
        assert '0.9980' in content

    def test_row_count_matches_input(self, diff_file):
        data_rows = [
            ln for ln in pathlib.Path(diff_file).read_text().splitlines()
            if ln.strip() and not ln.startswith('#')
        ]
        assert len(data_rows) == 3

    def test_parseable_by_parse_diffusivity_file(self, diff_file):
        from models.parsers import parse_diffusivity_file
        T, D, Derr, R2 = parse_diffusivity_file(diff_file)
        assert len(T) == 3
        assert T[0] == pytest.approx(300.0)
        assert D[0] == pytest.approx(1e-9, rel=1e-6)
        assert Derr[0] == pytest.approx(1e-10, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 6. fit_arrhenius
# ═══════════════════════════════════════════════════════════════════════════

class TestFitArrhenius:

    @pytest.fixture()
    def arrhenius_data(self):
        Ea_true = 0.45   # eV
        D0_true = 5e-7   # m²/s
        T = np.array([300., 400., 500., 600., 800., 1000.])
        D = D0_true * np.exp(-Ea_true / (KB_EV * T))
        return T, D, Ea_true, D0_true

    @pytest.fixture()
    def noisy_data(self):
        rng = np.random.default_rng(3)
        T = np.array([400., 600., 800., 1000., 1200.])
        D = 1e-7 * np.exp(-0.45 / (KB_EV * T)) * np.exp(rng.normal(0, 0.2, T.size))
        return T, D

    def test_returns_five_values(self, arrhenius_data):
        T, D, _, _ = arrhenius_data
        result = fit_arrhenius(T, D)
        assert len(result) == 5

    def test_recovers_activation_energy(self, arrhenius_data):
        T, D, Ea_true, _ = arrhenius_data
        Ea, *_ = fit_arrhenius(T, D)
        assert Ea == pytest.approx(Ea_true, rel=0.01)

    def test_recovers_prefactor(self, arrhenius_data):
        T, D, _, D0_true = arrhenius_data
        _, _, D0, _, _ = fit_arrhenius(T, D)
        assert D0 == pytest.approx(D0_true, rel=0.01)

    def test_ea_positive_for_thermally_activated(self, arrhenius_data):
        T, D, _, _ = arrhenius_data
        Ea, *_ = fit_arrhenius(T, D)
        assert Ea > 0.0

    def test_d0_positive(self, arrhenius_data):
        T, D, _, _ = arrhenius_data
        _, _, D0, _, _ = fit_arrhenius(T, D)
        assert D0 > 0.0

    def test_r2_near_one_for_ideal_data(self, arrhenius_data):
        T, D, _, _ = arrhenius_data
        _, _, _, _, R2 = fit_arrhenius(T, D)
        assert R2 == pytest.approx(1.0, abs=1e-4)

    def test_ea_err_nonnegative(self, arrhenius_data):
        T, D, _, _ = arrhenius_data
        _, Ea_err, _, _, _ = fit_arrhenius(T, D)
        assert Ea_err >= 0.0

    def test_d0_err_nonnegative(self, arrhenius_data):
        T, D, _, _ = arrhenius_data
        _, _, _, D0_err, _ = fit_arrhenius(T, D)
        assert D0_err >= 0.0

    def test_matches_numpy_polyfit_on_noisy_data(self, noisy_data):
        T, D = noisy_data
        Ea, _, D0, _, _ = fit_arrhenius(T, D)
        slope, intercept = np.polyfit(1.0 / T, np.log(D), 1)
        assert Ea == pytest.approx(-slope * KB_EV, rel=1e-10)
        assert D0 == pytest.approx(np.exp(intercept), rel=1e-10)

    def test_errors_are_least_squares_standard_errors(self, noisy_data):
        # σ_slope² = [SSR/(n−2)] / Σ(x−x̄)²,   σ_intercept = σ_slope·√(Σx²/n)
        T, D = noisy_data
        _, Ea_err, D0, D0_err, _ = fit_arrhenius(T, D)
        x, y = 1.0 / T, np.log(D)
        slope, intercept = np.polyfit(x, y, 1)
        s2 = ((y - (slope * x + intercept)) ** 2).sum() / (len(x) - 2)
        sig_slope = np.sqrt(s2 / ((x - x.mean()) ** 2).sum())
        sig_int = sig_slope * np.sqrt((x ** 2).mean())
        assert Ea_err == pytest.approx(sig_slope * KB_EV, rel=1e-8)
        assert D0_err == pytest.approx(D0 * sig_int, rel=1e-8)

    def test_two_points_give_zero_uncertainty(self):
        # two points fit exactly: no degrees of freedom left for an error estimate
        T = np.array([500., 900.])
        D = np.array([3e-11, 4e-9])
        _, Ea_err, _, D0_err, R2 = fit_arrhenius(T, D)
        assert Ea_err == 0.0 and D0_err == 0.0
        assert R2 == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. arrhenius_D
# ═══════════════════════════════════════════════════════════════════════════

class TestArrheniusD:

    def test_scalar_input_returns_float_like(self):
        D = arrhenius_D(T=500., Ea=0.4, D0=1e-7)
        assert float(D) == D

    def test_ea_zero_returns_d0(self):
        D = arrhenius_D(T=300., Ea=0.0, D0=1e-7)
        assert D == pytest.approx(1e-7)

    def test_higher_temperature_higher_d(self):
        D_lo = arrhenius_D(T=300., Ea=0.4, D0=1e-7)
        D_hi = arrhenius_D(T=600., Ea=0.4, D0=1e-7)
        assert D_hi > D_lo

    def test_array_input_shape_preserved(self):
        T = np.array([300., 400., 500., 600.])
        D = arrhenius_D(T, Ea=0.4, D0=1e-7)
        assert D.shape == (4,)

    def test_exact_value(self):
        Ea, D0, T = 0.5, 1e-7, 400.0
        expected = D0 * np.exp(-Ea / (KB_EV * T))
        assert arrhenius_D(T, Ea, D0) == pytest.approx(expected, rel=1e-10)

    def test_larger_ea_smaller_d_at_same_t(self):
        D_low_ea  = arrhenius_D(T=300., Ea=0.2, D0=1e-7)
        D_high_ea = arrhenius_D(T=300., Ea=0.8, D0=1e-7)
        assert D_high_ea < D_low_ea


# ═══════════════════════════════════════════════════════════════════════════
# 8. run_diffusivity_pipeline (uses synthetic LAMMPS dump)
# ═══════════════════════════════════════════════════════════════════════════

class TestRunDiffusivityPipeline:

    @pytest.fixture()
    def pipeline_result(self, tmp_path):
        # Build a synthetic trajectory: H atom moves at v=0.1 Å/step
        n_frames = 60
        v = 0.1  # Å / dump-step
        frames = [(2.0 + i * v, 5.0, 5.0) for i in range(n_frames)]
        dump_path = str(tmp_path / 'traj.dump')
        _make_dump(dump_path, frames)

        result = run_diffusivity_pipeline(
            dump_file=dump_path,
            temperature=300,
            h_type=9,
            timestep=0.001,
            outdir=str(tmp_path / 'out'),
        )
        return result, tmp_path

    def test_returns_dict_with_required_keys(self, pipeline_result):
        result, _ = pipeline_result
        for key in ('D', 'sigma_D', 'R2', 'msd_file'):
            assert key in result

    def test_d_positive(self, pipeline_result):
        result, _ = pipeline_result
        assert result['D'] > 0.0

    def test_msd_file_created(self, pipeline_result):
        result, _ = pipeline_result
        assert pathlib.Path(result['msd_file']).exists()

    def test_msd_file_has_header(self, pipeline_result):
        result, _ = pipeline_result
        content = pathlib.Path(result['msd_file']).read_text()
        assert '# t_ps  MSD_ang2' in content

    def test_r2_bounded(self, pipeline_result):
        result, _ = pipeline_result
        # R² must be ≤ 1 (may not be near 1 for very short trajectories)
        assert result['R2'] <= 1.0

    def test_msd_file_in_outdir(self, pipeline_result):
        result, tmp_path = pipeline_result
        assert pathlib.Path(result['msd_file']).parent == tmp_path / 'out'

    def test_missing_dump_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, Exception)):
            run_diffusivity_pipeline(
                dump_file=str(tmp_path / 'nonexistent.dump'),
                temperature=300,
                h_type=9,
                outdir=str(tmp_path),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 9. run_arrhenius_pipeline (uses synthetic diffusivity.txt)
# ═══════════════════════════════════════════════════════════════════════════

class TestRunArrheniusPipeline:

    @pytest.fixture()
    def arrhenius_setup(self, tmp_path):
        Ea_true = 0.40
        D0_true = 2e-7
        T = np.array([300., 400., 600., 800.])
        D = D0_true * np.exp(-Ea_true / (KB_EV * T))
        Derr = D * 0.01

        diff_file = str(tmp_path / 'diffusivity.txt')
        save_diffusivity_table(
            temperatures=T.tolist(),
            D_vals=D.tolist(),
            D_errs=Derr.tolist(),
            R2_vals=[0.999] * 4,
            outfile=diff_file,
        )
        return diff_file, tmp_path, Ea_true, D0_true

    def test_returns_dict_with_ea(self, arrhenius_setup):
        diff_file, tmp_path, _, _ = arrhenius_setup
        result = run_arrhenius_pipeline(diff_file, outdir=str(tmp_path))
        assert 'Ea' in result

    def test_ea_close_to_true(self, arrhenius_setup):
        diff_file, tmp_path, Ea_true, _ = arrhenius_setup
        result = run_arrhenius_pipeline(diff_file, outdir=str(tmp_path))
        assert result['Ea'] == pytest.approx(Ea_true, rel=0.02)

    def test_d0_close_to_true(self, arrhenius_setup):
        diff_file, tmp_path, _, D0_true = arrhenius_setup
        result = run_arrhenius_pipeline(diff_file, outdir=str(tmp_path))
        assert result['D0'] == pytest.approx(D0_true, rel=0.05)

    def test_plot_file_created(self, arrhenius_setup):
        diff_file, tmp_path, _, _ = arrhenius_setup
        result = run_arrhenius_pipeline(diff_file, outdir=str(tmp_path))
        assert pathlib.Path(result['plot_file']).exists()

    def test_fit_ignores_sigma_D(self, tmp_path, monkeypatch):
        """σ_D only draws error bars: changing it, even to 0 or NaN, leaves
        the fit unchanged and drops no temperature."""
        import models.diffusivity_post_processing as dpp
        monkeypatch.setattr(dpp, 'plot_arrhenius', lambda *a, **k: None)
        T = np.array([400., 600., 800., 1000.])
        rng = np.random.default_rng(5)
        D = 2e-7 * np.exp(-0.40 / (KB_EV * T)) * np.exp(rng.normal(0, 0.2, T.size))
        fits = []
        for name, Derr in (('a', 0.01 * D), ('b', np.array([1e-3, 10., 0., np.nan]) * D)):
            f = str(tmp_path / f'diff_{name}.txt')
            save_diffusivity_table(T.tolist(), D.tolist(), Derr.tolist(), [0.99] * 4, f)
            fits.append(run_arrhenius_pipeline(f, outdir=str(tmp_path)))
        for k in ('Ea', 'Ea_err', 'D0', 'D0_err', 'R2'):
            assert fits[1][k] == pytest.approx(fits[0][k], rel=1e-12)

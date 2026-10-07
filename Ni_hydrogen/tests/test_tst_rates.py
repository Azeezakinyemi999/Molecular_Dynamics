"""
tests/test_tst_rates.py
========================
Tests for models/tst_rates.py — offline, no LAMMPS or cluster required.

Synthetic vib_frequencies.json and neb_barrier.txt files are written to
tmp_path.  All numeric assertions use analytic expected values derived from
the Vineyard / Arrhenius formulas.
"""

import json
import math
import os
import sys
import pathlib
import warnings
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.tst_rates import (
    BOLTZMANN_eV,
    CM1_TO_EV,
    SPEED_LIGHT_CM_S,
    split_vib_results,
    apply_zpe_correction,
    vineyard_prefactor,
    arrhenius_rate,
    collect_neb_results,
    build_rate_dict,
    rates_to_json,
)


# ── file helpers ─────────────────────────────────────────────────────────────

def _write_vib_json(path, real_freqs, imag_freqs=None):
    data = {
        'frequencies_real_cm1': list(real_freqs),
        'frequencies_imag_cm1': list(imag_freqs or []),
    }
    pathlib.Path(path).write_text(json.dumps(data, indent=2))


def _write_barrier(path, E_abs=0.45, E_des=0.15, delta_E=0.30,
                   fmax=0.05, converged=True):
    pathlib.Path(path).write_text(
        f'E_IS: -100.5\n'
        f'E_FS: -100.2\n'
        f'E_abs: {E_abs}\n'
        f'E_des: {E_des}\n'
        f'delta_E: {delta_E}\n'
        f'fmax_final: {fmax}\n'
        f'converged: {converged}\n'
    )


# ── shared vib frequencies ────────────────────────────────────────────────────

_IS_FREQS = [300.0, 500.0, 800.0, 1000.0]   # 4 real IS modes
_TS_FREQS = [250.0, 450.0, 700.0]            # 3 real TS modes (imaginary excluded)
# 4 real FS modes, deliberately stiffer than IS so ZPE_FS > ZPE_IS and the
# forward/reverse asymmetry is detectable in both the ZPE and the prefactor.
_FS_FREQS = [400.0, 650.0, 900.0, 1200.0]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Physical constants
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:

    def test_boltzmann_ev(self):
        assert BOLTZMANN_eV == pytest.approx(8.617333262e-5, rel=1e-6)

    def test_cm1_to_ev(self):
        assert CM1_TO_EV == pytest.approx(1.2398419843e-4, rel=1e-6)

    def test_speed_light_cm_s(self):
        assert SPEED_LIGHT_CM_S == pytest.approx(2.99792458e10, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 2. split_vib_results
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitVibResults:

    def _make_vd(self, keys):
        return {k: {'vib_json': f'/path/{k}.json'} for k in keys}

    def test_returns_two_dicts(self):
        vd = self._make_vd(['hopa_Ni3Mo_IS', 'hopa_Ni3Mo_TS'])
        result = split_vib_results(vd)
        assert len(result) == 2

    def test_is_dict_keyed_by_base_label(self):
        vd = self._make_vd(['hopa_Ni3Mo_IS'])
        vis, _ = split_vib_results(vd)
        assert 'hopa_Ni3Mo' in vis

    def test_ts_dict_keyed_by_base_label(self):
        vd = self._make_vd(['hopa_Ni3Mo_TS'])
        _, vts = split_vib_results(vd)
        assert 'hopa_Ni3Mo' in vts

    def test_vib_json_path_preserved(self):
        vd = {'hopa_Ni3Mo_IS': {'vib_json': '/data/IS.json'}}
        vis, _ = split_vib_results(vd)
        assert vis['hopa_Ni3Mo'] == '/data/IS.json'

    def test_multiple_labels_split_correctly(self):
        vd = self._make_vd(['hopa_A_IS', 'hopa_A_TS', 'hopa_B_IS', 'hopa_B_TS'])
        vis, vts = split_vib_results(vd)
        assert set(vis.keys()) == {'hopa_A', 'hopa_B'}
        assert set(vts.keys()) == {'hopa_A', 'hopa_B'}

    def test_unexpected_key_emits_warning(self):
        vd = {'hopa_Ni3Mo': {'vib_json': '/p.json'}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            vis, vts = split_vib_results(vd)
        assert any('unexpected key' in str(wn.message) for wn in w)

    def test_unexpected_key_not_in_either_dict(self):
        vd = {'hopa_Ni3Mo': {'vib_json': '/p.json'}}
        vis, vts = split_vib_results(vd)
        assert 'hopa_Ni3Mo' not in vis
        assert 'hopa_Ni3Mo' not in vts

    def test_empty_input_returns_empty_dicts(self):
        vis, vts = split_vib_results({})
        assert vis == {} and vts == {}

    def test_fs_key_ignored_without_warning(self):
        # FS (dissolved-H) modes are consumed by split_vib_fs, not the
        # IS->TS barrier ZPE correction: split_vib_results must skip them
        # silently rather than warn (which would be spurious noise now that
        # the vibration set includes FS endpoints).
        vd = {'hopa_Ni3Mo_IS': {'vib_json': '/is.json'},
              'hopa_Ni3Mo_TS': {'vib_json': '/ts.json'},
              'hopa_Ni3Mo_FS': {'vib_json': '/fs.json'}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            vis, vts = split_vib_results(vd)
        assert not any('unexpected key' in str(wn.message) for wn in w)
        assert 'hopa_Ni3Mo' in vis and 'hopa_Ni3Mo' in vts


class TestSplitVibFs:

    def test_extracts_only_fs_entries(self):
        from models.tst_rates import split_vib_fs
        vd = {'hopa_A_IS': {'vib_json': '/a_is.json'},
              'hopa_A_TS': {'vib_json': '/a_ts.json'},
              'hopa_A_FS': {'vib_json': '/a_fs.json'},
              'hopb_B_FS': {'vib_json': '/b_fs.json'}}
        fs = split_vib_fs(vd)
        assert set(fs.keys()) == {'hopa_A', 'hopb_B'}
        assert fs['hopa_A'] == '/a_fs.json'

    def test_empty_when_no_fs(self):
        from models.tst_rates import split_vib_fs
        vd = {'hopa_A_IS': {'vib_json': '/a_is.json'}}
        assert split_vib_fs(vd) == {}


# ═══════════════════════════════════════════════════════════════════════════
# 3. apply_zpe_correction
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyZpeCorrection:

    def test_equal_freqs_no_correction(self):
        freqs = [100.0, 200.0, 300.0]
        result = apply_zpe_correction(0.4, freqs, freqs)
        assert result == pytest.approx(0.4)

    def test_higher_ts_freqs_increase_barrier(self):
        is_f = [100.0, 200.0]
        ts_f = [300.0, 400.0]    # ZPE_TS > ZPE_IS → positive correction
        result = apply_zpe_correction(0.4, is_f, ts_f)
        assert result > 0.4

    def test_lower_ts_freqs_decrease_barrier(self):
        is_f = [300.0, 400.0]
        ts_f = [100.0, 200.0]    # ZPE_TS < ZPE_IS → negative correction
        result = apply_zpe_correction(0.4, is_f, ts_f)
        assert result < 0.4

    def test_exact_correction_value(self):
        is_f  = [200.0]
        ts_f  = [400.0]
        E_bar = 0.3
        delta_zpe = 0.5 * (400.0 - 200.0) * CM1_TO_EV
        assert apply_zpe_correction(E_bar, is_f, ts_f) == pytest.approx(E_bar + delta_zpe)

    def test_min_freq_filter_excludes_low_modes(self):
        is_f  = [200.0, 10.0]   # 10 below default min=50
        ts_f  = [400.0, 5.0]    # 5 below default min=50
        result = apply_zpe_correction(0.3, is_f, ts_f)
        expected = 0.3 + 0.5 * (400.0 - 200.0) * CM1_TO_EV
        assert result == pytest.approx(expected)

    def test_custom_min_freq(self):
        is_f  = [60.0, 200.0]
        ts_f  = [60.0, 300.0]
        # with min_freq=100, the 60 cm^-1 modes are excluded
        result_default = apply_zpe_correction(0.3, is_f, ts_f, min_freq_cm1=100.0)
        result_low     = apply_zpe_correction(0.3, is_f, ts_f, min_freq_cm1=50.0)
        assert result_default != result_low

    def test_empty_freqs_return_barrier_unchanged(self):
        assert apply_zpe_correction(0.5, [], []) == pytest.approx(0.5)

    def test_zero_barrier_returns_only_delta_zpe(self):
        is_f = [100.0]
        ts_f = [200.0]
        delta_zpe = 0.5 * (200.0 - 100.0) * CM1_TO_EV
        assert apply_zpe_correction(0.0, is_f, ts_f) == pytest.approx(delta_zpe)


# ═══════════════════════════════════════════════════════════════════════════
# 4. vineyard_prefactor
# ═══════════════════════════════════════════════════════════════════════════

class TestVineyardPrefactor:

    def test_returns_positive_float(self):
        nu = vineyard_prefactor(_IS_FREQS, _TS_FREQS)
        assert isinstance(nu, float) and nu > 0.0

    def test_exact_two_mode_case(self):
        # IS: [f1, f2], TS: [f1]  →  nu = c * f2
        f1, f2 = 500.0, 1000.0
        nu = vineyard_prefactor([f1, f2], [f1])
        assert nu == pytest.approx(SPEED_LIGHT_CM_S * f2, rel=1e-8)

    def test_single_is_single_ts_ratio(self):
        # IS: [2000], TS: [1000]  (both above threshold of 50)
        # but we need at least 1 IS above and 1 TS above...
        # actually with 1 IS and 1 TS: nu = c * 2000 / 1000 = 2c
        # but IS has 2 modes (one imaginary excluded in TS context)
        # IS: [f_a, f_b], TS: [f_b]  →  nu = c * f_a
        nu = vineyard_prefactor([2000.0, 1000.0], [1000.0])
        assert nu == pytest.approx(SPEED_LIGHT_CM_S * 2000.0, rel=1e-8)

    def test_raises_if_no_valid_is_freqs(self):
        with pytest.raises(ValueError, match='IS'):
            vineyard_prefactor([20.0], [500.0], min_freq_cm1=50.0)

    def test_raises_if_no_valid_ts_freqs(self):
        with pytest.raises(ValueError, match='TS'):
            vineyard_prefactor([500.0, 300.0], [20.0], min_freq_cm1=50.0)

    def test_low_freqs_excluded_with_warning(self):
        # Counts must still satisfy len(IS) == len(TS) + 1 after the cut, so
        # this exercises the exclusion warning rather than the dimensional guard.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            vineyard_prefactor([500.0, 600.0, 30.0], [400.0, 25.0], min_freq_cm1=50.0)
        assert any('excluded' in str(wn.message) for wn in w)

    def test_raises_when_mode_counts_mismatch(self):
        """IS and TS displacing different atom sets leaves c x (cm^-1)^n, not a
        frequency. Real case: an H2 dissociation IS with 2H+6 metals (24 modes)
        against a TS with 2H+8 metals (29 real) produced 5e-10 s^-1."""
        with pytest.raises(ValueError, match='len\\(IS\\) == len\\(TS\\) \\+ 1'):
            vineyard_prefactor([500.0] * 24, [400.0] * 29)

    def test_raises_when_is_equals_ts_count(self):
        with pytest.raises(ValueError, match='dimensionally'):
            vineyard_prefactor([500.0, 600.0], [400.0, 300.0])

    def test_accepts_exact_one_extra_is_mode(self):
        nu = vineyard_prefactor([500.0, 600.0, 700.0], [400.0, 300.0])
        assert nu > 0.0

    def test_higher_is_product_higher_nu(self):
        # doubling IS freqs doubles nu
        nu1 = vineyard_prefactor([500.0, 1000.0], [500.0])
        nu2 = vineyard_prefactor([1000.0, 1000.0], [500.0])
        assert nu2 == pytest.approx(2.0 * nu1, rel=1e-8)

    def test_higher_ts_product_lower_nu(self):
        # doubling TS freqs halves nu
        nu1 = vineyard_prefactor([1000.0, 1000.0], [500.0])
        nu2 = vineyard_prefactor([1000.0, 1000.0], [1000.0])
        assert nu2 == pytest.approx(0.5 * nu1, rel=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 5. arrhenius_rate
# ═══════════════════════════════════════════════════════════════════════════

class TestArrheniusRate:

    _nu = 1e13   # s^-1 (typical attempt frequency)

    def test_zero_barrier_returns_nu(self):
        k = arrhenius_rate(self._nu, 0.0, T_K=300.0)
        assert k == pytest.approx(self._nu)

    def test_positive_barrier_reduces_rate(self):
        k = arrhenius_rate(self._nu, 0.5, T_K=300.0)
        assert k < self._nu

    def test_higher_temperature_higher_rate(self):
        k_lo = arrhenius_rate(self._nu, 0.3, T_K=300.0)
        k_hi = arrhenius_rate(self._nu, 0.3, T_K=600.0)
        assert k_hi > k_lo

    def test_higher_barrier_lower_rate(self):
        k_lo = arrhenius_rate(self._nu, 0.2, T_K=300.0)
        k_hi = arrhenius_rate(self._nu, 0.5, T_K=300.0)
        assert k_hi < k_lo

    def test_exact_value(self):
        nu, Ea, T = 1e13, 0.4, 500.0
        expected = nu * math.exp(-Ea / (BOLTZMANN_eV * T))
        assert arrhenius_rate(nu, Ea, T) == pytest.approx(expected, rel=1e-10)

    def test_raises_for_zero_temperature(self):
        with pytest.raises(ValueError):
            arrhenius_rate(self._nu, 0.3, T_K=0.0)

    def test_raises_for_negative_temperature(self):
        with pytest.raises(ValueError):
            arrhenius_rate(self._nu, 0.3, T_K=-100.0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. collect_neb_results
# ═══════════════════════════════════════════════════════════════════════════

class TestCollectNebResults:

    def test_empty_job_list_returns_empty_dict(self):
        result = collect_neb_results([])
        assert result == {}

    def test_single_valid_job_returns_one_entry(self, tmp_path):
        bf = str(tmp_path / 'barrier.txt')
        _write_barrier(bf)
        jobs = [{'sid': 'Ni3Mo', 'barrier_file': bf}]
        result = collect_neb_results(jobs, hop='hopa')
        assert 'hopa_Ni3Mo' in result

    def test_label_format_is_hop_sid(self, tmp_path):
        bf = str(tmp_path / 'barrier.txt')
        _write_barrier(bf)
        jobs = [{'sid': 'testsite', 'barrier_file': bf}]
        result = collect_neb_results(jobs, hop='hopb')
        assert 'hopb_testsite' in result

    def test_missing_barrier_file_skipped_with_warning(self, tmp_path):
        jobs = [{'sid': 'A', 'barrier_file': str(tmp_path / 'nonexistent.txt')}]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = collect_neb_results(jobs)
        assert result == {}
        assert len(w) >= 1

    def test_non_converged_job_included_with_warning(self, tmp_path):
        bf = str(tmp_path / 'barrier_notconv.txt')
        _write_barrier(bf, converged=False)
        jobs = [{'sid': 'A', 'barrier_file': bf}]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = collect_neb_results(jobs)
        assert 'hopa_A' in result
        assert any('not converged' in str(wn.message) for wn in w)

    def test_barrier_values_loaded_correctly(self, tmp_path):
        bf = str(tmp_path / 'barrier.txt')
        _write_barrier(bf, E_abs=0.38, E_des=0.12)
        jobs = [{'sid': 'X', 'barrier_file': bf}]
        result = collect_neb_results(jobs)
        assert result['hopa_X']['E_abs'] == pytest.approx(0.38)
        assert result['hopa_X']['E_des'] == pytest.approx(0.12)


# ═══════════════════════════════════════════════════════════════════════════
# 7. build_rate_dict
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildRateDict:

    @pytest.fixture()
    def rate_inputs(self, tmp_path):
        label = 'hopa_Ni3Mo'
        is_json = str(tmp_path / 'IS.json')
        ts_json = str(tmp_path / 'TS.json')
        _write_vib_json(is_json, _IS_FREQS, imag_freqs=[])
        _write_vib_json(ts_json, _TS_FREQS, imag_freqs=[200.0])

        neb = {label: {'E_abs': 0.45, 'E_des': 0.15, 'delta_E': 0.30, 'converged': True}}
        vis = {label: is_json}
        vts = {label: ts_json}
        return neb, vis, vts, label

    def test_returns_dict_with_label(self, rate_inputs):
        neb, vis, vts, label = rate_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0)
        assert label in rd

    def test_result_has_required_keys(self, rate_inputs):
        neb, vis, vts, label = rate_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0)
        for k in ('k_forward', 'k_reverse', 'Ea_raw', 'Ea_zpe', 'Ed_raw', 'Ed_zpe', 'nu', 'T_K'):
            assert k in rd[label]

    def test_k_forward_positive(self, rate_inputs):
        neb, vis, vts, label = rate_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0)
        assert rd[label]['k_forward'] > 0.0

    def test_k_reverse_positive(self, rate_inputs):
        neb, vis, vts, label = rate_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0)
        assert rd[label]['k_reverse'] > 0.0

    def test_k_forward_gt_k_reverse_for_positive_delta_e(self, rate_inputs):
        # E_abs > E_des implies k_forward < k_reverse (lower barrier → faster)
        neb, vis, vts, label = rate_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, apply_zpe=False)
        assert rd[label]['k_forward'] < rd[label]['k_reverse']

    def test_apply_zpe_false_ea_zpe_equals_ea_raw(self, rate_inputs):
        neb, vis, vts, label = rate_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, apply_zpe=False)
        assert rd[label]['Ea_zpe'] == pytest.approx(rd[label]['Ea_raw'])

    def test_missing_is_vib_label_skipped(self, rate_inputs):
        neb, _, vts, label = rate_inputs
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, {}, vts, T_K=300.0)
        assert label not in rd

    def test_t_k_stored_in_result(self, rate_inputs):
        neb, vis, vts, label = rate_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=700.0)
        assert rd[label]['T_K'] == pytest.approx(700.0)

    def test_missing_ts_vib_label_skipped(self, rate_inputs):
        # IS present but TS absent → label must be silently skipped
        neb, vis, _, label = rate_inputs
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, {}, T_K=300.0)
        assert label not in rd


class TestReverseFromFS:
    """The reverse direction must be built from FS, not IS.

    E_des is E_TS - E_FS, so its ZPE partner is ZPE_TS - ZPE_FS and its attempt
    frequency is c*prod(nu_FS)/prod(nu_TS). Using the IS quantities for both
    directions (the old behaviour) is wrong twice over, and the two ZPE terms
    then cancel in Ea_zpe - Ed_zpe, stripping the ZPE out of the reaction energy
    that feeds dH_sol.
    """

    @pytest.fixture()
    def fs_inputs(self, tmp_path):
        label = 'hopa_Ni3Mo'
        is_json, ts_json, fs_json = (str(tmp_path / f'{n}.json') for n in ('IS', 'TS', 'FS'))
        _write_vib_json(is_json, _IS_FREQS, imag_freqs=[])
        _write_vib_json(ts_json, _TS_FREQS, imag_freqs=[200.0])
        _write_vib_json(fs_json, _FS_FREQS, imag_freqs=[])
        neb = {label: {'E_abs': 0.45, 'E_des': 0.15, 'delta_E': 0.30, 'converged': True}}
        return neb, {label: is_json}, {label: ts_json}, {label: fs_json}, label

    def test_zpe_source_is_fs_when_supplied(self, fs_inputs):
        neb, vis, vts, vfs, label = fs_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, vib_results_fs=vfs)
        assert rd[label]['zpe_source'] == 'FS'

    def test_zpe_source_marks_fallback_without_fs(self, fs_inputs):
        neb, vis, vts, _, label = fs_inputs
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=300.0)
        assert rd[label]['zpe_source'] == 'IS_fallback'

    def test_ed_zpe_uses_fs_not_is(self, fs_inputs):
        neb, vis, vts, vfs, label = fs_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, vib_results_fs=vfs)
        expected = apply_zpe_correction(0.15, _FS_FREQS, _TS_FREQS)
        assert rd[label]['Ed_zpe'] == pytest.approx(expected)

    def test_nu_reverse_is_fs_over_ts(self, fs_inputs):
        neb, vis, vts, vfs, label = fs_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, vib_results_fs=vfs)
        assert rd[label]['nu_reverse'] == pytest.approx(
            vineyard_prefactor(_FS_FREQS, _TS_FREQS))

    def test_forward_prefactor_unchanged_by_fs(self, fs_inputs):
        """Supplying FS must not perturb the forward direction."""
        neb, vis, vts, vfs, label = fs_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, vib_results_fs=vfs)
        assert rd[label]['nu'] == pytest.approx(
            vineyard_prefactor(_IS_FREQS, _TS_FREQS))
        assert rd[label]['Ea_zpe'] == pytest.approx(
            apply_zpe_correction(0.45, _IS_FREQS, _TS_FREQS))

    def test_nu_reverse_differs_from_forward(self, fs_inputs):
        neb, vis, vts, vfs, label = fs_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, vib_results_fs=vfs)
        assert rd[label]['nu_reverse'] != pytest.approx(rd[label]['nu'])

    def test_reaction_energy_recovers_zpe_term(self, fs_inputs):
        """Ea_zpe - Ed_zpe must equal the raw reaction energy plus
        (ZPE_FS - ZPE_IS). With the IS-for-both behaviour this term cancelled
        to zero, which is the bug that silently de-ZPE'd dH_sol."""
        neb, vis, vts, vfs, label = fs_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, vib_results_fs=vfs)
        zpe = lambda fs: 0.5 * sum(f * CM1_TO_EV for f in fs if f >= 50.0)
        expected = (0.45 - 0.15) + (zpe(_FS_FREQS) - zpe(_IS_FREQS))
        got = rd[label]['Ea_zpe'] - rd[label]['Ed_zpe']
        assert got == pytest.approx(expected)
        # and it is genuinely non-zero, i.e. the test would catch a regression
        assert abs(got - (0.45 - 0.15)) > 1e-4

    def test_fallback_reaction_energy_loses_zpe(self, fs_inputs):
        """Pin the old behaviour as the documented fallback: with no FS the
        ZPE cancels and the reaction energy is raw."""
        neb, vis, vts, _, label = fs_inputs
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=300.0)
        assert (rd[label]['Ea_zpe'] - rd[label]['Ed_zpe']) == pytest.approx(0.45 - 0.15)

    def test_k_reverse_uses_reverse_prefactor(self, fs_inputs):
        neb, vis, vts, vfs, label = fs_inputs
        rd = build_rate_dict(neb, vis, vts, T_K=300.0, vib_results_fs=vfs)
        r = rd[label]
        assert r['k_reverse'] == pytest.approx(
            arrhenius_rate(r['nu_reverse'], r['Ed_zpe'], 300.0))

    def test_unreadable_fs_falls_back(self, fs_inputs, tmp_path):
        neb, vis, vts, _, label = fs_inputs
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=300.0,
                                 vib_results_fs={label: str(tmp_path / 'nope.json')})
        assert rd[label]['zpe_source'] == 'IS_fallback'
        assert rd[label]['nu_reverse'] == pytest.approx(rd[label]['nu'])


# ═══════════════════════════════════════════════════════════════════════════
# 8. rates_to_json
# ═══════════════════════════════════════════════════════════════════════════

class TestRatesToJson:

    @pytest.fixture()
    def sample_rate_dict(self):
        return {
            'hopa_Ni3Mo': {
                'k_forward': 1.23e10, 'k_reverse': 3.45e11,
                'Ea_raw': 0.45, 'Ea_zpe': 0.43,
                'Ed_raw': 0.15, 'Ed_zpe': 0.13,
                'nu': 1.2e13, 'delta_e': 0.30, 'T_K': 300.0,
            }
        }

    def test_returns_out_path(self, sample_rate_dict, tmp_path):
        p = str(tmp_path / 'rates.json')
        ret = rates_to_json(sample_rate_dict, p)
        assert ret == p

    def test_file_created(self, sample_rate_dict, tmp_path):
        p = str(tmp_path / 'rates.json')
        rates_to_json(sample_rate_dict, p)
        assert pathlib.Path(p).exists()

    def test_file_is_valid_json(self, sample_rate_dict, tmp_path):
        p = str(tmp_path / 'rates.json')
        rates_to_json(sample_rate_dict, p)
        loaded = json.loads(pathlib.Path(p).read_text())
        assert isinstance(loaded, dict)

    def test_label_present_in_json(self, sample_rate_dict, tmp_path):
        p = str(tmp_path / 'rates.json')
        rates_to_json(sample_rate_dict, p)
        loaded = json.loads(pathlib.Path(p).read_text())
        assert 'hopa_Ni3Mo' in loaded

    def test_k_forward_preserved_in_json(self, sample_rate_dict, tmp_path):
        p = str(tmp_path / 'rates.json')
        rates_to_json(sample_rate_dict, p)
        loaded = json.loads(pathlib.Path(p).read_text())
        assert loaded['hopa_Ni3Mo']['k_forward'] == pytest.approx(1.23e10, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 8. write_hop_ranked / write_hop_vib_rates  (per-hop artifacts w/ oct env)
# ═══════════════════════════════════════════════════════════════════════════

from models.tst_rates import write_hop_ranked, write_hop_vib_rates


class TestWriteHopRanked:

    def test_ranks_by_barrier_and_carries_env(self, tmp_path):
        bf_hi = str(tmp_path / 'hi.txt'); _write_barrier(bf_hi, E_abs=0.60)
        bf_lo = str(tmp_path / 'lo.txt'); _write_barrier(bf_lo, E_abs=0.30)
        jobs = [
            {'sid': 's_hi', 'barrier_file': bf_hi, 'sub1_env': 'Ni6_oct'},
            {'sid': 's_lo', 'barrier_file': bf_lo, 'sub1_env': 'Ni5Mo_oct'},
        ]
        out = str(tmp_path / 'hopa_ranked.json')
        rows = write_hop_ranked(jobs, 'hopa', out, env_key='sub1_env')
        # lower barrier first
        assert [r['sid'] for r in rows] == ['s_lo', 's_hi']
        assert rows[0]['env'] == 'Ni5Mo_oct'
        assert rows[0]['label'] == 'hopa_s_lo'
        assert os.path.exists(out)

    def test_missing_barrier_listed_not_dropped(self, tmp_path):
        jobs = [{'sid': 's_x', 'barrier_file': str(tmp_path / 'nope.txt'),
                 'sub1_env': 'Ni6_oct'}]
        rows = write_hop_ranked(jobs, 'hopa', str(tmp_path / 'r.json'))
        assert len(rows) == 1
        assert rows[0]['converged'] is None
        assert rows[0].get('Ea') is None


class TestWriteHopVibRates:

    def test_filters_by_hop_and_tags_env(self, tmp_path):
        rate_dict = {
            'hopa_s_0': {'Ea_zpe': 0.31, 'Ed_zpe': 0.12, 'Ea_raw': 0.33,
                         'Ed_raw': 0.14, 'nu': 1.2e13},
            'hopb_s_0': {'Ea_zpe': 0.55, 'Ed_zpe': 0.40, 'Ea_raw': 0.57,
                         'Ed_raw': 0.42, 'nu': 9.0e12},
        }
        jobs_a = [{'sid': 's_0', 'sub1_env': 'Ni6_oct'}]
        out = str(tmp_path / 'hopa_vib_rates.json')
        res = write_hop_vib_rates(rate_dict, jobs_a, 'hopa', out, env_key='sub1_env')
        assert set(res.keys()) == {'hopa_s_0'}          # hopb filtered out
        assert res['hopa_s_0']['env'] == 'Ni6_oct'
        assert res['hopa_s_0']['Ea_zpe'] == 0.31
        assert res['hopa_s_0']['nu'] == 1.2e13
        assert os.path.exists(out)

    def test_hopb_uses_sub2_env(self, tmp_path):
        rate_dict = {'hopb_s_0': {'Ea_zpe': 0.55, 'Ed_zpe': 0.40, 'Ea_raw': 0.57,
                                  'Ed_raw': 0.42, 'nu': 9.0e12}}
        jobs_b = [{'sid': 's_0', 'sub1_env': 'Ni6_oct', 'sub2_env': 'Ni4Mo2_oct'}]
        res = write_hop_vib_rates(rate_dict, jobs_b, 'hopb', str(tmp_path / 'v.json'),
                                  env_key='sub2_env')
        assert res['hopb_s_0']['env'] == 'Ni4Mo2_oct'


# ═══════════════════════════════════════════════════════════════════════════
# 9. Partition functions (vibrational solubility prefactor building blocks)
# ═══════════════════════════════════════════════════════════════════════════

from models.tst_rates import (
    vib_partition_function, h2_gas_partition_function,
    _THETA_ROT_H2_K, _SIGMA_H2,
)


class TestVibPartitionFunction:

    def test_at_least_one(self):
        assert vib_partition_function([500.0, 800.0, 1200.0], 600.0) >= 1.0

    def test_approaches_one_at_low_T(self):
        # all modes frozen out → q → 1
        assert vib_partition_function([1000.0, 1000.0, 1000.0], 1.0) == pytest.approx(1.0, abs=1e-9)

    def test_increases_with_temperature(self):
        q_lo = vib_partition_function([300.0, 300.0], 300.0)
        q_hi = vib_partition_function([300.0, 300.0], 900.0)
        assert q_hi > q_lo

    def test_soft_modes_below_threshold_dropped(self):
        # a 10 cm-1 mode is below the 50 cm-1 floor -> ignored
        q_with_soft = vib_partition_function([10.0, 500.0], 600.0)
        q_stiff     = vib_partition_function([500.0], 600.0)
        assert q_with_soft == pytest.approx(q_stiff)

    def test_single_mode_exact(self):
        nu, T = 500.0, 600.0
        from models.tst_rates import CM1_TO_EV, BOLTZMANN_eV
        x = (CM1_TO_EV * nu) / (BOLTZMANN_eV * T)
        assert vib_partition_function([nu], T) == pytest.approx(1.0 / (1.0 - math.exp(-x)))


class TestH2GasPartitionFunction:

    def test_returns_components(self):
        q = h2_gas_partition_function(600.0, 1.0)
        for k in ('trans', 'rot', 'vib', 'total'):
            assert k in q and q[k] > 0.0

    def test_translational_inverse_in_pressure(self):
        q_lo = h2_gas_partition_function(600.0, 1.0)['trans']
        q_hi = h2_gas_partition_function(600.0, 2.0)['trans']
        assert q_lo == pytest.approx(2.0 * q_hi, rel=1e-8)   # q_trans ∝ V ∝ 1/P

    def test_rotational_high_T_formula(self):
        q = h2_gas_partition_function(600.0, 1.0)
        assert q['rot'] == pytest.approx(600.0 / (_SIGMA_H2 * _THETA_ROT_H2_K), rel=1e-8)

    def test_vibrational_near_unity(self):
        # θ_vib(H2) ≈ 6332 K >> 600 K, so q_vib ≈ 1
        assert h2_gas_partition_function(600.0, 1.0)['vib'] == pytest.approx(1.0, abs=1e-3)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            h2_gas_partition_function(0.0, 1.0)
        with pytest.raises(ValueError):
            h2_gas_partition_function(600.0, 0.0)


class TestForwardPrefactorOptional:
    """H2 dissociation needs only the reverse prefactor.

    Its forward rate is a dimensionless sticking probability exp(-Ea_zpe/kT)
    multiplied by a Hertz-Knudsen strike rate, so no forward Vineyard prefactor
    is wanted. Dropping the whole label because that unused quantity could not
    be formed would throw away the barriers and the reverse prefactor that are
    actually consumed.
    """

    @pytest.fixture()
    def mismatched(self, tmp_path):
        """IS and TS mode counts that cannot form a forward prefactor, with a
        valid FS/TS pair for the reverse one."""
        label = 'diss_s_1'
        is_json, ts_json, fs_json = (str(tmp_path / f'{n}.json') for n in ('IS', 'TS', 'FS'))
        # IS has the same count as TS -> forward guard trips.
        _write_vib_json(is_json, [300.0] * 5, imag_freqs=[])
        _write_vib_json(ts_json, [250.0] * 5, imag_freqs=[200.0])
        # FS has one more than TS -> reverse prefactor is well formed.
        _write_vib_json(fs_json, [400.0] * 6, imag_freqs=[])
        neb = {label: {'E_abs': 0.60, 'E_des': 0.74, 'delta_E': -0.14, 'converged': True}}
        return neb, {label: is_json}, {label: ts_json}, {label: fs_json}, label

    def test_label_skipped_when_forward_required(self, mismatched):
        neb, vis, vts, vfs, label = mismatched
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=700.0, vib_results_fs=vfs)
        assert label not in rd

    def test_label_kept_when_forward_not_required(self, mismatched):
        neb, vis, vts, vfs, label = mismatched
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=700.0, vib_results_fs=vfs,
                                 require_forward_nu=False)
        assert label in rd

    def test_forward_fields_are_none(self, mismatched):
        neb, vis, vts, vfs, label = mismatched
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=700.0, vib_results_fs=vfs,
                                 require_forward_nu=False)
        assert rd[label]['nu'] is None
        assert rd[label]['k_forward'] is None

    def test_reverse_still_computed(self, mismatched):
        neb, vis, vts, vfs, label = mismatched
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=700.0, vib_results_fs=vfs,
                                 require_forward_nu=False)
        r = rd[label]
        assert r['nu_reverse'] == pytest.approx(
            vineyard_prefactor([400.0] * 6, [250.0] * 5))
        assert r['k_reverse'] == pytest.approx(
            arrhenius_rate(r['nu_reverse'], r['Ed_zpe'], 700.0))
        assert r['zpe_source'] == 'FS'

    def test_barriers_survive(self, mismatched):
        """The barriers are what the dissociation actually consumes."""
        neb, vis, vts, vfs, label = mismatched
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rd = build_rate_dict(neb, vis, vts, T_K=700.0, vib_results_fs=vfs,
                                 require_forward_nu=False)
        assert rd[label]['Ea_raw'] == pytest.approx(0.60)
        assert rd[label]['Ed_raw'] == pytest.approx(0.74)
        assert rd[label]['Ea_zpe'] is not None

    def test_warns_about_missing_forward_prefactor(self, mismatched):
        neb, vis, vts, vfs, label = mismatched
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            build_rate_dict(neb, vis, vts, T_K=700.0, vib_results_fs=vfs,
                            require_forward_nu=False)
        assert any('without a forward prefactor' in str(x.message) for x in w)

    def test_hop_path_default_still_requires_forward(self, mismatched):
        """Regression guard: the Hop A/B default must stay strict."""
        import inspect
        sig = inspect.signature(build_rate_dict)
        assert sig.parameters['require_forward_nu'].default is True

    def test_zero_cut_keeps_all_modes(self):
        """min_freq_cm1=0 is what lets the shared-atom-set dissociation states
        keep the len(IS) == len(TS) + 1 relation."""
        soft_is = [6.9, 11.0, 18.7, 29.9, 35.1] + [300.0] * 31
        ts      = [300.0] * 35
        with pytest.raises(ValueError):
            vineyard_prefactor(soft_is, ts, min_freq_cm1=50.0)
        assert vineyard_prefactor(soft_is, ts, min_freq_cm1=0.0) > 0.0


class TestCountMatchedTruncation:
    """The frequency cut can separate the mode counts even when both states
    displaced the same atoms.

    Ni's dissociation TS carries modes at 20.7-32 cm^-1 while its FS bottoms
    out at 89, so a 50 cm^-1 cut removes several from one side and none from
    the other. Those survivors sit in the denominator and, compounded over ~41
    modes, drove nu_rev to 6.2e14 s^-1 against a physical ~1e12-1e13.
    """

    def test_raises_without_opt_in(self):
        soft_ts = [20.0, 25.0] + [300.0] * 39
        fs      = [300.0] * 42
        with pytest.raises(ValueError, match='allow_count_match'):
            vineyard_prefactor(fs, soft_ts, min_freq_cm1=50.0)

    def test_opt_in_restores_the_relation(self):
        soft_ts = [20.0, 25.0] + [300.0] * 39
        fs      = [300.0] * 42
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            nu = vineyard_prefactor(fs, soft_ts, min_freq_cm1=50.0,
                                    allow_count_match=True)
        assert nu > 0.0

    def test_trimming_drops_the_lowest_not_the_highest(self):
        """Artifacts are the near-zero modes, so those are what must go."""
        fs = [60.0, 1000.0, 1000.0]
        ts = [1000.0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            nu = vineyard_prefactor(fs, ts, min_freq_cm1=50.0,
                                    allow_count_match=True)
        # dropping the 60.0 leaves 1000*1000/1000 = 1000 cm^-1 * c
        assert nu == pytest.approx(SPEED_LIGHT_CM_S * 1000.0, rel=1e-9)

    def test_no_op_when_counts_already_balanced(self):
        """Al's case: nothing below the cut, so the result must not move."""
        fs = [100.0, 200.0, 300.0]
        ts = [150.0, 250.0]
        assert (vineyard_prefactor(fs, ts, min_freq_cm1=50.0, allow_count_match=True)
                == pytest.approx(vineyard_prefactor(fs, ts, min_freq_cm1=50.0)))

    def test_matching_lowers_an_inflated_prefactor(self):
        """The regression this fixes: soft denominator modes inflate nu."""
        soft_ts = [20.0, 22.0, 23.0, 32.0] + [300.0] * 37
        fs      = [300.0] * 42
        inflated = vineyard_prefactor(fs, soft_ts, min_freq_cm1=0.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            fixed = vineyard_prefactor(fs, soft_ts, min_freq_cm1=50.0,
                                       allow_count_match=True)
        assert inflated > 100.0 * fixed

    def test_genuine_atom_set_mismatch_still_raises(self):
        """The 24-vs-29 dissociation bug must NOT be papered over by trimming.
        build_rate_dict only opts in after confirming equal degrees of freedom,
        so a caller that has not checked still gets the error."""
        with pytest.raises(ValueError, match='len\\(IS\\) == len\\(TS\\) \\+ 1'):
            vineyard_prefactor([500.0] * 24, [400.0] * 29)


class TestAtomSetGatesTrimming:
    """build_rate_dict may only enable trimming once the states are known to
    span the same degrees of freedom (real + imaginary)."""

    def _inputs(self, tmp_path, is_real, is_imag, ts_real, ts_imag):
        label = 'hopa_X'
        ij, tj = str(tmp_path / 'IS.json'), str(tmp_path / 'TS.json')
        _write_vib_json(ij, is_real, imag_freqs=is_imag)
        _write_vib_json(tj, ts_real, imag_freqs=ts_imag)
        neb = {label: {'E_abs': 0.4, 'E_des': 0.1, 'delta_E': 0.3, 'converged': True}}
        return neb, {label: ij}, {label: tj}, label

    def test_equal_dof_with_two_imaginary_ts_still_works(self):
        """A TS with 2 imaginary modes spans the same DOF as its IS, so the
        counts may be rebalanced — this is Al's hopb_s_1 (21 real/0 imag vs
        19 real/2 imag, both 21 DOF)."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            neb, vis, vts, label = self._inputs(
                tmp, [300.0] * 21, [], [300.0] * 19, [200.0, 210.0])
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                rd = build_rate_dict(neb, vis, vts, T_K=600.0)
            assert label in rd

    def test_unequal_dof_is_skipped(self):
        """24+0 vs 29+1 DOF — different atom sets, must not produce a rate."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            neb, vis, vts, label = self._inputs(
                tmp, [300.0] * 24, [], [300.0] * 29, [200.0])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                rd = build_rate_dict(neb, vis, vts, T_K=600.0)
            assert label not in rd
            assert any('different degrees of freedom' in str(x.message) for x in w)

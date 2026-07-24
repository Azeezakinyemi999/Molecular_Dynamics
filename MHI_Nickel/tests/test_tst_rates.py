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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            vineyard_prefactor([500.0, 30.0], [400.0, 25.0], min_freq_cm1=50.0)
        assert any('excluded' in str(wn.message) for wn in w)

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

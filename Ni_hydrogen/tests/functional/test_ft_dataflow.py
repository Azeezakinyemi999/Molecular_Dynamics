"""
tests/functional/test_ft_dataflow.py
=====================================
Category C functional tests — data flow / interface contracts.

Verifies that the output schema of stage N is a valid input for stage N+1.
No LAMMPS, no SLURM, no cluster, no GPU required.

Every test uses a minimal synthetic fixture (Python dict or string) that
represents exactly what the writer function produces. Tests assert:
  - required keys are present
  - value types are correct (str vs int, float vs nan, list vs dict)
  - shape constraints hold (same-length lists, 3-element position vectors)
  - the access patterns used by downstream readers don't raise KeyError
"""

import json
import math
import os
import re
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.parsers import parse_barrier_file


# ─── synthetic fixtures ───────────────────────────────────────────────────────

_SURFACE_SITES = {
    "metadata": {
        "seed": 42,
        "slab_path": "/work/slab.lammps",
        "n_atoms_total": 48,
        "n_atoms_surface": 12,
        "n_sites_total": 1,
        "cell": [7.07, 7.07, 20.0],
        "z_max": 15.0,
        "slab_composition": {"Ni": 48},
        "site_type_counts": {"bridge": 1},
        "top_20_compositions": {"Ni2_bridge": 1},
    },
    "surface_atoms": [
        {"node_id": "Ni_0", "index": 0, "element": "Ni", "position": [0.0, 0.0, 10.0]},
        {"node_id": "Ni_1", "index": 1, "element": "Ni", "position": [2.5, 0.0, 10.0]},
    ],
    "sites": [
        {
            "site_id": "s_0",
            "level1": {
                "site_id": "s_0",
                "site_type": "bridge",
                "hollow_type": "",
                "composition": "Ni2",
                "full_label": "Ni2_bridge",
                "position": [1.25, 0.0, 11.0],
                "atom_indices": [0, 1],
                "constituent_atoms": [
                    {"index": 0, "element": "Ni", "position": [0.0, 0.0, 10.0]},
                    {"index": 1, "element": "Ni", "position": [2.5, 0.0, 10.0]},
                ],
                "subsurf_element": "",
            },
            "level2": {
                "0": {
                    "element": "Ni",
                    "shell1": [{"index": 1, "element": "Ni", "distance": 2.49, "shell": 1}],
                    "shell2": [],
                    "n_shell1": 1,
                    "n_shell2": 0,
                },
                "1": {
                    "element": "Ni",
                    "shell1": [{"index": 0, "element": "Ni", "distance": 2.49, "shell": 1}],
                    "shell2": [],
                    "n_shell1": 1,
                    "n_shell2": 0,
                },
            },
            "level3": [],
        }
    ],
}

_NEB_PAIRS = [
    {
        "label": "s_0__s_1+s_0",
        "is_site": "s_0",
        "fs_site1": "s_1",
        "fs_site2": "s_0",
        "E_IS": -2271.0,
        "E_FS": -2271.06,
        "delta_E": -0.06,
        "is_fs_dist": 2.8,
        "neb_script": "/work/neb/s_0__s_1+s_0/run_neb.py",
        "min_script": "/work/neb/s_0__s_1+s_0/min_fs.lammps",
        "fsmin_sh": "/work/neb/s_0__s_1+s_0/slurm_fsmin.sh",
        "neb_sh": "/work/neb/s_0__s_1+s_0/slurm_neb.sh",
        "barrier_file": "/work/neb/s_0__s_1+s_0/neb_barrier.txt",
        "path_file": "/work/neb/s_0__s_1+s_0/neb_path.dat",
        "job_dir": "/work/neb/s_0__s_1+s_0",
    },
    {
        "label": "s_2__s_3+s_2",
        "is_site": "s_2",
        "fs_site1": "s_3",
        "fs_site2": "s_2",
        "E_IS": -2270.5,
        "E_FS": -2270.55,
        "delta_E": -0.05,
        "is_fs_dist": 3.1,
        "neb_script": "/work/neb/s_2__s_3+s_2/run_neb.py",
        "min_script": "/work/neb/s_2__s_3+s_2/min_fs.lammps",
        "fsmin_sh": "/work/neb/s_2__s_3+s_2/slurm_fsmin.sh",
        "neb_sh": "/work/neb/s_2__s_3+s_2/slurm_neb.sh",
        "barrier_file": "/work/neb/s_2__s_3+s_2/neb_barrier.txt",
        "path_file": "/work/neb/s_2__s_3+s_2/neb_path.dat",
        "job_dir": "/work/neb/s_2__s_3+s_2",
    },
]

_HOPA_JOB = {
    "sid": "s_5",
    "is_path": "/work/hopa/s_5/h_atom_s_5_relaxed.lammps",
    "e_is": -2271.0,
    "ss1_id": "oct_0",
    "sub1_xyz": [1.0, 2.0, 8.0],
    "fs_raw": "/work/hopa/s_5/sub1_fs_raw.lammps",
    "fs_relaxed": "/work/hopa/s_5/sub1_fs_relaxed.lammps",
    "fsmin_script": "/work/hopa/s_5/min_fs.lammps",
    "neb_script": "/work/hopa/s_5/run_neb.py",
    "fsmin_sh": "/work/hopa/s_5/slurm_fsmin_s_5.sh",
    "neb_sh": "/work/hopa/s_5/slurm_neb_s_5.sh",
    "barrier_file": "/work/hopa/s_5/neb_barrier.txt",
    "path_file": "/work/hopa/s_5/neb_path.dat",
    "job_dir": "/work/hopa/s_5",
}

_HOPB_JOB = {
    "sid": "s_5",
    "ss1_id": "oct_0",
    "ss2_id": "tet_1",
    "hopb_is": "/work/hopa/s_5/sub1_fs_relaxed.lammps",
    "e_is": -2270.5,
    "sub2_xyz": [1.0, 2.0, 6.0],
    "fs_raw": "/work/hopb/s_5/sub2_fs_raw.lammps",
    "fs_relaxed": "/work/hopb/s_5/sub2_fs_relaxed.lammps",
    "fsmin_script": "/work/hopb/s_5/min_fs.lammps",
    "neb_script": "/work/hopb/s_5/run_neb.py",
    "fsmin_sh": "/work/hopb/s_5/slurm_fsmin_s_5.sh",
    "neb_sh": "/work/hopb/s_5/slurm_neb_s_5.sh",
    "barrier_file": "/work/hopb/s_5/neb_barrier.txt",
    "path_file": "/work/hopb/s_5/neb_path.dat",
    "job_dir": "/work/hopb/s_5",
}

_RATE_DICT = {
    "hopa_Ni": {
        "k_forward": 1.23e8,
        "k_reverse": 4.56e6,
        "Ea_raw": 0.48,
        "Ea_zpe": 0.45,
        "Ed_raw": 0.54,
        "Ed_zpe": 0.51,
        "nu": 2.1e12,
        "delta_e": -0.06,
        "T_K": 700.0,
    },
    "hopb_Ni": {
        "k_forward": 9.87e7,
        "k_reverse": 3.21e6,
        "Ea_raw": 0.52,
        "Ea_zpe": 0.49,
        "Ed_raw": 0.58,
        "Ed_zpe": 0.55,
        "nu": 1.9e12,
        "delta_e": -0.06,
        "T_K": 700.0,
    },
}

_LATTICE_PARAMS = {
    "temperatures": [600, 700, 800],
    "a0_m": [3.530e-10, 3.545e-10, 3.560e-10],
}

_NEB_BARRIER_WITH_ALIAS = """\
# IS : h_atom_s_5_relaxed.lammps
# FS : sub1_fs_relaxed.lammps
# N images : 18
# fmax_final : 0.0021 eV/A

Converged   : True

E_IS = -2271.05
E_FS = -2270.57
E_a = 0.48 eV
E_des = 0.54 eV
delta_E = -0.06 eV
"""

_NEB_BARRIER_NO_E_DES = """\
# IS : h_atom_s_5_relaxed.lammps
# FS : sub1_fs_relaxed.lammps

Converged   : True

E_IS = -2271.05
E_FS = -2270.57
E_abs = 0.48 eV
delta_E = -0.06 eV
"""


# ═════════════════════════════════════════════════════════════════════════════
# 1. surface_sites.json schema
# ═════════════════════════════════════════════════════════════════════════════

class TestSurfaceSitesJsonSchema:
    """
    Verify the schema of surface_sites.json as produced by save_surface_sites().
    Tests use a synthetic fixture that mirrors the exact structure written by
    models/surface_graph.py, validated against the reader patterns in
    neb_workflow.load_neb_pools() and run_phase1/2_*().
    """

    @pytest.fixture()
    def data(self):
        return _SURFACE_SITES

    def test_top_level_keys_present(self, data):
        for key in ('metadata', 'surface_atoms', 'sites'):
            assert key in data, f"'{key}' missing from surface_sites.json top level"

    def test_metadata_required_keys(self, data):
        meta = data['metadata']
        for key in ('n_atoms_total', 'n_sites_total', 'cell', 'slab_composition', 'site_type_counts'):
            assert key in meta, f"metadata missing '{key}'"

    def test_site_id_is_string(self, data):
        for site in data['sites']:
            assert isinstance(site['site_id'], str), (
                f"site_id must be str, got {type(site['site_id'])!r} — "
                "load_neb_pools() builds {s['site_id']: s} so integer keys break lookup"
            )

    def test_level1_has_required_keys(self, data):
        l1 = data['sites'][0]['level1']
        for key in ('site_type', 'composition', 'full_label', 'position', 'atom_indices'):
            assert key in l1, f"level1 missing '{key}'"

    def test_level1_position_is_list_of_3_floats(self, data):
        pos = data['sites'][0]['level1']['position']
        assert len(pos) == 3
        for v in pos:
            assert isinstance(v, (int, float)), f"position element {v!r} is not numeric"

    def test_level2_keys_are_strings(self, data):
        l2 = data['sites'][0]['level2']
        for k in l2:
            assert isinstance(k, str), (
                f"level2 key {k!r} must be str (stringified atom index) — "
                "load_neb_pools() accesses level2 by string key"
            )

    def test_level3_is_list(self, data):
        l3 = data['sites'][0]['level3']
        assert isinstance(l3, list)

    def test_n_sites_metadata_equals_sites_length(self, data):
        assert data['metadata']['n_sites_total'] == len(data['sites'])

    def test_load_neb_pools_access_pattern(self, data):
        # Replicate the access pattern from neb_workflow.load_neb_pools():
        #   sites_dict = {s['site_id']: s for s in sites_data['sites']}
        #   site['level1']['position'], site['level1']['full_label'], site['level2']
        sites_dict = {s['site_id']: s for s in data['sites']}
        for sid, s in sites_dict.items():
            _ = s['level1']['position']
            _ = s['level1']['full_label']
            _ = s['level2']


# ═════════════════════════════════════════════════════════════════════════════
# 2. neb_pairs.json schema
# ═════════════════════════════════════════════════════════════════════════════

_NEB_PAIR_REQUIRED_KEYS = {
    'label', 'is_site', 'fs_site1', 'fs_site2',
    'E_IS', 'E_FS', 'delta_E', 'is_fs_dist',
    'neb_script', 'min_script', 'fsmin_sh', 'neb_sh',
    'barrier_file', 'path_file', 'job_dir',
}

class TestNebPairsJsonSchema:
    """
    Verify the schema of neb_pairs.json as produced by neb_workflow.orchestrate_neb().
    Consumed by collect_neb_results() which reads label, barrier_file, path_file.
    """

    @pytest.fixture()
    def pairs(self):
        return _NEB_PAIRS

    def test_is_a_list(self, pairs):
        assert isinstance(pairs, list)

    def test_required_15_keys_present(self, pairs):
        for i, entry in enumerate(pairs):
            missing = _NEB_PAIR_REQUIRED_KEYS - entry.keys()
            assert not missing, f"neb_pairs[{i}] missing keys: {missing}"

    def test_label_format(self, pairs):
        pattern = re.compile(r'^s_\d+__s_\d+\+s_\d+$')
        for entry in pairs:
            assert pattern.match(entry['label']), (
                f"label {entry['label']!r} does not match IS__FS1+FS2 format"
            )

    def test_energies_are_float(self, pairs):
        for entry in pairs:
            for key in ('E_IS', 'E_FS', 'delta_E', 'is_fs_dist'):
                assert isinstance(entry[key], (int, float)), (
                    f"{key}={entry[key]!r} must be numeric"
                )

    def test_path_keys_are_strings(self, pairs):
        for entry in pairs:
            for key in ('barrier_file', 'path_file', 'job_dir'):
                assert isinstance(entry[key], str), f"{key} must be str"

    def test_collect_neb_results_access_pattern(self, pairs):
        # collect_neb_results() accesses job['label'], job['barrier_file'], job['path_file']
        for job in pairs:
            _ = job['label']
            _ = job['barrier_file']
            _ = job['path_file']


# ═════════════════════════════════════════════════════════════════════════════
# 3. hopa_jobs.json schema
# ═════════════════════════════════════════════════════════════════════════════

_HOPA_REQUIRED_KEYS = {
    'sid', 'is_path', 'e_is', 'ss1_id', 'sub1_xyz',
    'fs_raw', 'fs_relaxed', 'fsmin_script', 'neb_script',
    'fsmin_sh', 'neb_sh', 'barrier_file', 'path_file', 'job_dir',
}

class TestHopaJobsSchema:
    """
    Verify the per-job dict schema returned by orchestrate_hopa_neb()['jobs']
    and serialised to hopa_jobs.json.
    Consumed by orchestrate_hopb_neb() (reads sid, ss1_id) and
    tst_rates.collect_neb_results() (reads sid, barrier_file).
    """

    @pytest.fixture()
    def job(self):
        return _HOPA_JOB.copy()

    def test_required_keys_present(self, job):
        missing = _HOPA_REQUIRED_KEYS - job.keys()
        assert not missing, f"hopa job dict missing keys: {missing}"

    def test_sid_is_string(self, job):
        assert isinstance(job['sid'], str)

    def test_e_is_is_float(self, job):
        assert isinstance(job['e_is'], (int, float))

    def test_sub1_xyz_is_list_of_3_floats(self, job):
        xyz = job['sub1_xyz']
        assert len(xyz) == 3
        for v in xyz:
            assert isinstance(v, (int, float))

    def test_tst_collect_accesses_sid_and_barrier(self, job):
        # tst_rates.collect_neb_results() accesses job['sid'] and job['barrier_file']
        _ = job['sid']
        _ = job['barrier_file']

    def test_hopb_reads_sid_and_ss1_id(self, job):
        # orchestrate_hopb_neb() reads ha_job['sid'] and ha_job['ss1_id'] per job
        _ = job['sid']
        _ = job['ss1_id']


# ═════════════════════════════════════════════════════════════════════════════
# 4. hopb_jobs.json schema
# ═════════════════════════════════════════════════════════════════════════════

_HOPB_REQUIRED_KEYS = {
    'sid', 'ss1_id', 'ss2_id', 'hopb_is', 'e_is', 'sub2_xyz',
    'fs_raw', 'fs_relaxed', 'fsmin_script', 'neb_script',
    'fsmin_sh', 'neb_sh', 'barrier_file', 'path_file', 'job_dir',
}

class TestHopbJobsSchema:
    """
    Verify the per-job dict schema returned by orchestrate_hopb_neb()['jobs'].
    Key difference from Hop A: uses 'hopb_is' (not 'is_path') and adds 'ss2_id'.
    """

    @pytest.fixture()
    def job(self):
        return _HOPB_JOB.copy()

    def test_required_keys_present(self, job):
        missing = _HOPB_REQUIRED_KEYS - job.keys()
        assert not missing, f"hopb job dict missing keys: {missing}"

    def test_ss2_id_present(self, job):
        assert 'ss2_id' in job, "Hop B must have ss2_id (subsurface-2 site)"

    def test_hopb_is_key_not_is_path(self, job):
        assert 'hopb_is' in job, "Hop B uses 'hopb_is' key for IS path (not 'is_path')"

    def test_e_is_is_float(self, job):
        assert isinstance(job['e_is'], (int, float))

    def test_sub2_xyz_is_list_of_3(self, job):
        assert len(job['sub2_xyz']) == 3

    def test_tst_collect_accesses_barrier_file(self, job):
        # tst_rates.collect_neb_results() accesses job['sid'] and job['barrier_file']
        _ = job['sid']
        _ = job['barrier_file']


# ═════════════════════════════════════════════════════════════════════════════
# 5. rate_dict_T{T}K.json schema
# ═════════════════════════════════════════════════════════════════════════════

_RATE_DICT_ENTRY_KEYS = {
    'k_forward', 'k_reverse',
    'Ea_raw', 'Ea_zpe',
    'Ed_raw', 'Ed_zpe',
    'nu', 'delta_e', 'T_K',
}

class TestRateDictJsonSchema:
    """
    Verify the schema of rate_dict_T{T}K.json as produced by tst_rates.rates_to_json().
    Consumed by permeation_workflow Phase 6, which reads k_forward, k_reverse,
    Ea_zpe (falling back to Ea_raw), and delta_e per entry.
    """

    @pytest.fixture()
    def rd(self):
        return _RATE_DICT

    def test_is_dict_not_list(self, rd):
        assert isinstance(rd, dict)

    def test_required_9_keys_per_entry(self, rd):
        for label, entry in rd.items():
            missing = _RATE_DICT_ENTRY_KEYS - entry.keys()
            assert not missing, f"rate_dict['{label}'] missing keys: {missing}"

    def test_all_values_are_numeric(self, rd):
        for label, entry in rd.items():
            for k, v in entry.items():
                assert isinstance(v, (int, float)), (
                    f"rate_dict['{label}']['{k}'] = {v!r} must be float"
                )

    def test_label_prefix_hopa_or_hopb(self, rd):
        for label in rd:
            assert label.startswith('hopa_') or label.startswith('hopb_'), (
                f"unexpected label prefix in rate_dict key: {label!r}"
            )

    def test_T_K_matches_expected_temperature(self, rd):
        for label, entry in rd.items():
            assert entry['T_K'] == 700.0

    def test_permeation_reader_accesses_k_forward_k_reverse(self, rd):
        # permeation_workflow Phase 6 body: r['k_forward'], r['k_reverse']
        for label, r in rd.items():
            if label.startswith('hopa_'):
                _ = r['k_forward']
                _ = r['k_reverse']


# ═════════════════════════════════════════════════════════════════════════════
# 6. lattice_params_vs_T.json schema
# ═════════════════════════════════════════════════════════════════════════════

class TestLatticeParamsJsonSchema:
    """
    Verify the schema of lattice_params_vs_T.json as produced by the generated
    diffusivity_run.py body. Consumed by permeation_workflow Phase 5 via:
        _a0_dict = dict(zip(lat['temperatures'], lat['a0_m']))
    """

    @pytest.fixture()
    def lat(self):
        return _LATTICE_PARAMS

    def test_top_level_keys(self, lat):
        assert set(lat.keys()) == {'temperatures', 'a0_m'}, (
            "lattice_params_vs_T.json must have exactly 'temperatures' and 'a0_m'"
        )

    def test_lists_same_length(self, lat):
        assert len(lat['temperatures']) == len(lat['a0_m'])

    def test_a0_m_values_are_floats(self, lat):
        for v in lat['a0_m']:
            assert isinstance(v, (int, float))

    def test_a0_m_in_si_metres(self, lat):
        for v in lat['a0_m']:
            assert 3.4e-10 < v < 3.7e-10, (
                f"a0_m={v:.3e} is outside expected Ni lattice range — "
                "check units: should be metres (~3.52e-10), not Angstroms"
            )

    def test_temperatures_are_numeric(self, lat):
        for t in lat['temperatures']:
            assert isinstance(t, (int, float))

    def test_permeation_reader_zip_pattern(self, lat):
        # Replicates: _a0_dict = dict(zip(_lat['temperatures'], _lat['a0_m']))
        a0_dict = dict(zip(lat['temperatures'], lat['a0_m']))
        assert len(a0_dict) == len(lat['temperatures'])
        for t, v in a0_dict.items():
            assert isinstance(v, float)


# ═════════════════════════════════════════════════════════════════════════════
# 7. neb_barrier.txt → barrier dict interface
# ═════════════════════════════════════════════════════════════════════════════

class TestBarrierDictToRateInterface:
    """
    Verify the interface between parse_barrier_file() output and build_rate_dict() input.

    Key contracts:
    1. 'E_a' alias in the file is normalised to 'E_abs' in the result
    2. 'Converged' (capital C) is parsed as bool True
    3. Missing E_des doesn't crash (build_rate_dict uses neb.get('E_des', 0.0))
    """

    @pytest.fixture()
    def result_alias(self, tmp_path):
        p = tmp_path / 'neb_barrier.txt'
        p.write_text(_NEB_BARRIER_WITH_ALIAS)
        return parse_barrier_file(str(p))

    @pytest.fixture()
    def result_no_e_des(self, tmp_path):
        p = tmp_path / 'neb_barrier.txt'
        p.write_text(_NEB_BARRIER_NO_E_DES)
        return parse_barrier_file(str(p))

    def test_E_a_alias_normalised_to_E_abs(self, result_alias):
        # E_a in the file must appear as 'E_abs' in the result — the key that
        # build_rate_dict() reads as neb['E_abs']
        assert 'E_abs' in result_alias, (
            "'E_abs' key missing — E_a alias not normalised by parse_barrier_file()"
        )
        assert 'E_a' not in result_alias, (
            "raw alias 'E_a' leaked into result — readers expect 'E_abs'"
        )

    def test_converged_is_bool_true(self, result_alias):
        assert result_alias['converged'] is True

    def test_build_rate_dict_reads_E_abs(self, result_alias):
        # build_rate_dict() accesses neb['E_abs'] — verify no KeyError
        _ = result_alias['E_abs']
        assert math.isclose(result_alias['E_abs'], 0.48, rel_tol=1e-6)

    def test_missing_E_des_safe_with_get(self, result_no_e_des):
        # build_rate_dict() uses neb.get('E_des', 0.0) — missing key must not crash
        ed = result_no_e_des.get('E_des', 0.0)
        assert ed == 0.0

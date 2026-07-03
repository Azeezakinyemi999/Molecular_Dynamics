"""
tests/functional/test_ft_surface_neb.py
=========================================
Category A functional tests — Phase 1 (H2*) and Phase 2 (H*) adsorption
dry-run script generation.

Verifies that run_phase1_h2_adsorption(dry_run=True) and
run_phase2_h_adsorption(dry_run=True) generate all expected files with
correct SLURM partition, LAMMPS minimize keyword, and per-site naming
conventions.  No LAMMPS, no SLURM, no GPU required.

Fixture strategy:
- 2×2 Ni FCC(111) slab (3 layers) built with ASE in a tmp_path
- Synthetic surface_sites.json with 2 sites (positions within slab cell)
- e_clean=0.0, e_h2_gas=0.0 safe for dry_run=True (not used in script gen)
"""

import json
import os
import sys
import pathlib

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import MagicMock

# acat and matplotlib are unavailable / NumPy-2.x-incompatible in the test env
for _m in ('acat', 'acat.adsorption_sites'):
    sys.modules.setdefault(_m, MagicMock())
sys.modules.setdefault('models.surface_graph', MagicMock())
for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.patches',
           'matplotlib.cm', 'matplotlib.ticker', 'matplotlib.transforms',
           'matplotlib.colors', 'matplotlib.scale', 'matplotlib._path',
           'matplotlib._api', 'matplotlib.cbook', 'matplotlib.rcsetup'):
    sys.modules.setdefault(_m, MagicMock())

from ase.build import fcc111
from ase.io import write as ase_write

from models.neb_workflow import run_phase1_h2_adsorption, run_phase2_h_adsorption
from models.config import E2T_7, MASSES_7, ELEM_STR_7


# ─── shared slab + sites fixture ─────────────────────────────────────────────

def _make_slab_and_sites(tmp_path):
    """
    Build a 2×2 Ni FCC(111) slab (3 layers, 12 atoms) and a matching
    surface_sites.json with 2 synthetic sites.  Returns (slab_path, sites_json).
    """
    slab = fcc111('Ni', size=(2, 2, 3), vacuum=10.0)
    slab.wrap()

    slab_path = str(tmp_path / 'slab.lammps')
    ase_write(slab_path, slab, format='lammps-data', atom_style='atomic')

    pos = slab.get_positions()
    surface_z = float(pos[:, 2].max())
    surface_atoms = [int(i) for i in range(len(slab))
                     if abs(pos[i, 2] - surface_z) < 0.5]

    sites_data = {
        'metadata': {
            'n_atoms_total': len(slab),
            'n_sites_total': 2,
            'cell': slab.get_cell().tolist(),
            'slab_composition': {'Ni': len(slab)},
            'site_type_counts': {'fcc': 2},
        },
        'surface_atoms': surface_atoms,
        'sites': [
            {
                'site_id': 's_0',
                'level1': {
                    'site_type': 'fcc',
                    'composition': {'Ni': 3},
                    'full_label': 'fcc_Ni3',
                    'position': [float(pos[surface_atoms[0], 0]),
                                 float(pos[surface_atoms[0], 1]),
                                 surface_z + 2.5],
                    'atom_indices': surface_atoms[:3],
                },
                'level2': {str(i): {'elem': 'Ni'} for i in surface_atoms[:3]},
                'level3': [],
            },
            {
                'site_id': 's_1',
                'level1': {
                    'site_type': 'fcc',
                    'composition': {'Ni': 3},
                    'full_label': 'fcc_Ni3',
                    'position': [float(pos[surface_atoms[1], 0]),
                                 float(pos[surface_atoms[1], 1]),
                                 surface_z + 2.5],
                    'atom_indices': surface_atoms[1:4],
                },
                'level2': {str(i): {'elem': 'Ni'} for i in surface_atoms[1:4]},
                'level3': [],
            },
        ],
    }

    sites_json = str(tmp_path / 'surface_sites.json')
    with open(sites_json, 'w') as f:
        json.dump(sites_data, f)

    return slab_path, sites_json


# ═════════════════════════════════════════════════════════════════════════════
# 1. Phase 1 — H2* adsorption dry-run
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase1H2AdsorptionDryRun:
    """
    run_phase1_h2_adsorption(dry_run=True) must generate all LAMMPS and
    SLURM files for every surface site without attempting any submission.
    """

    @pytest.fixture()
    def phase1_result(self, tmp_path):
        slab_path, sites_json = _make_slab_and_sites(tmp_path)
        result = run_phase1_h2_adsorption(
            surface_sites_json=sites_json,
            relaxed_slab_path=slab_path,
            e_clean=0.0,
            e_h2_gas=0.0,
            outdir=str(tmp_path),
            dry_run=True,
            elem_str=ELEM_STR_7,
            e2t=E2T_7,
            masses=MASSES_7,
        )
        return result, tmp_path / 'phase1_h2'

    def test_status_is_generated(self, phase1_result):
        result, _ = phase1_result
        assert result['status'] == 'generated'

    def test_n_sites_total_correct(self, phase1_result):
        result, _ = phase1_result
        assert result['n_sites_total'] == 2

    def test_structure_files_created_for_each_site(self, phase1_result):
        _, phase_dir = phase1_result
        struct_dir = phase_dir / 'structures'
        for sid in ('s_0', 's_1'):
            assert (struct_dir / f'slab_h2_{sid}.lammps').exists(), (
                f"slab_h2_{sid}.lammps not found in structures/"
            )

    def test_lammps_scripts_created_for_each_site(self, phase1_result):
        _, phase_dir = phase1_result
        script_dir = phase_dir / 'scripts'
        for sid in ('s_0', 's_1'):
            assert (script_dir / f'h2_min_{sid}.in').exists(), (
                f"h2_min_{sid}.in not found in scripts/"
            )

    def test_lammps_script_contains_minimize(self, phase1_result):
        _, phase_dir = phase1_result
        content = (phase_dir / 'scripts' / 'h2_min_s_0.in').read_text()
        assert 'minimize' in content

    def test_slurm_scripts_created_for_each_site(self, phase1_result):
        _, phase_dir = phase1_result
        slurm_dir = phase_dir / 'slurm'
        for sid in ('s_0', 's_1'):
            assert (slurm_dir / f'h2_slurm_{sid}.sh').exists(), (
                f"h2_slurm_{sid}.sh not found in slurm/"
            )

    def test_slurm_script_uses_sharing_partition(self, phase1_result):
        _, phase_dir = phase1_result
        content = (phase_dir / 'slurm' / 'h2_slurm_s_0.sh').read_text()
        assert 'sharing' in content, (
            "Phase 1 SLURM script does not use 'sharing' partition — "
            "the default was changed but may have been reverted in neb_workflow.py"
        )

    def test_array_script_created(self, phase1_result):
        _, phase_dir = phase1_result
        assert (phase_dir / 'run_h2_array.sh').exists()

    def test_job_index_contains_all_site_ids(self, phase1_result):
        _, phase_dir = phase1_result
        content = (phase_dir / 'h2_job_index.txt').read_text()
        for sid in ('s_0', 's_1'):
            assert sid in content, f"'{sid}' not found in h2_job_index.txt"

    def test_slab_structure_has_extra_atoms_for_h2(self, phase1_result):
        # The original slab has 12 atoms; add_adsorbate appends 2 H atoms → 14 total.
        # We read the LAMMPS header (atoms count line) to avoid a type-mapping issue.
        _, phase_dir = phase1_result
        content = (phase_dir / 'structures' / 'slab_h2_s_0.lammps').read_text()
        # LAMMPS data file has a line like "14 atoms"
        atom_count_lines = [l.strip() for l in content.splitlines()
                            if l.strip().endswith('atoms') and l.split()[0].isdigit()]
        assert atom_count_lines, "No 'N atoms' header line found in slab_h2 file"
        n_atoms = int(atom_count_lines[0].split()[0])
        assert n_atoms == 14, (
            f"Expected 14 atoms (12 Ni + 2 H) in H2-adsorbed slab, got {n_atoms}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Phase 2 — H* adsorption dry-run
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase2HAdsorptionDryRun:
    """
    run_phase2_h_adsorption(dry_run=True) must generate all LAMMPS and
    SLURM files for every surface site without attempting any submission.
    """

    @pytest.fixture()
    def phase2_result(self, tmp_path):
        slab_path, sites_json = _make_slab_and_sites(tmp_path)
        result = run_phase2_h_adsorption(
            surface_sites_json=sites_json,
            relaxed_slab_path=slab_path,
            e_clean=0.0,
            e_h2_gas=0.0,
            outdir=str(tmp_path),
            dry_run=True,
            elem_str=ELEM_STR_7,
            e2t=E2T_7,
            masses=MASSES_7,
        )
        return result, tmp_path / 'phase2_h'

    def test_status_is_generated(self, phase2_result):
        result, _ = phase2_result
        assert result['status'] == 'generated'

    def test_n_sites_total_correct(self, phase2_result):
        result, _ = phase2_result
        assert result['n_sites_total'] == 2

    def test_structure_files_created_for_each_site(self, phase2_result):
        _, phase_dir = phase2_result
        struct_dir = phase_dir / 'structures'
        for sid in ('s_0', 's_1'):
            assert (struct_dir / f'slab_h_{sid}.lammps').exists(), (
                f"slab_h_{sid}.lammps not found in structures/"
            )

    def test_lammps_scripts_created_for_each_site(self, phase2_result):
        _, phase_dir = phase2_result
        script_dir = phase_dir / 'scripts'
        for sid in ('s_0', 's_1'):
            assert (script_dir / f'h_min_{sid}.in').exists(), (
                f"h_min_{sid}.in not found in scripts/"
            )

    def test_lammps_script_contains_minimize(self, phase2_result):
        _, phase_dir = phase2_result
        content = (phase_dir / 'scripts' / 'h_min_s_0.in').read_text()
        assert 'minimize' in content

    def test_slurm_scripts_created_for_each_site(self, phase2_result):
        _, phase_dir = phase2_result
        slurm_dir = phase_dir / 'slurm'
        for sid in ('s_0', 's_1'):
            assert (slurm_dir / f'h_slurm_{sid}.sh').exists(), (
                f"h_slurm_{sid}.sh not found in slurm/"
            )

    def test_slurm_script_uses_sharing_partition(self, phase2_result):
        _, phase_dir = phase2_result
        content = (phase_dir / 'slurm' / 'h_slurm_s_0.sh').read_text()
        assert 'sharing' in content, (
            "Phase 2 SLURM script does not use 'sharing' partition"
        )

    def test_array_script_created(self, phase2_result):
        _, phase_dir = phase2_result
        assert (phase_dir / 'run_h_array.sh').exists()

    def test_job_index_contains_all_site_ids(self, phase2_result):
        _, phase_dir = phase2_result
        content = (phase_dir / 'h_job_index.txt').read_text()
        for sid in ('s_0', 's_1'):
            assert sid in content, f"'{sid}' not found in h_job_index.txt"

    def test_slab_structure_has_extra_atom_for_h(self, phase2_result):
        # The original slab has 12 atoms; add_adsorbate appends 1 H atom → 13 total.
        _, phase_dir = phase2_result
        content = (phase_dir / 'structures' / 'slab_h_s_0.lammps').read_text()
        atom_count_lines = [l.strip() for l in content.splitlines()
                            if l.strip().endswith('atoms') and l.split()[0].isdigit()]
        assert atom_count_lines, "No 'N atoms' header line found in slab_h file"
        n_atoms = int(atom_count_lines[0].split()[0])
        assert n_atoms == 13, (
            f"Expected 13 atoms (12 Ni + 1 H) in H-adsorbed slab, got {n_atoms}"
        )

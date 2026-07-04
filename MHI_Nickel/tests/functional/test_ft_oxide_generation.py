"""
tests/functional/test_ft_oxide_generation.py
=============================================
Category A/C functional tests — oxide support in the generated Part-2
permeation scripts, metal_type threading, and oxide slab auto-thickness.

No LAMMPS, no SLURM, no GPU required. build_slab's oxide branch needs
spglib and is skipped where unavailable (runs on the cluster).
"""

import os
import sys
import inspect

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import MagicMock

for _m in ('acat', 'acat.adsorption_sites'):
    sys.modules.setdefault(_m, MagicMock())
for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.patches',
           'matplotlib.cm', 'matplotlib.ticker', 'matplotlib.transforms',
           'matplotlib.colors', 'matplotlib.scale', 'matplotlib._path',
           'matplotlib._api', 'matplotlib.cbook', 'matplotlib.rcsetup'):
    sys.modules.setdefault(_m, MagicMock())

from models.permeation_workflow import generate_permeation_scripts

_SLURM = dict(partition='sharing', time='00:20:00')


def _generate(tmp_path, name, **kwargs):
    out = str(tmp_path / name)
    generate_permeation_scripts(
        work_dir=str(tmp_path), relaxed_slab_path='/x/slab.lammps',
        surface_sites_json='/x/surface_sites.json',
        phase2_h_dir=str(tmp_path), sub_neb_dir=str(tmp_path),
        vib_dir=str(tmp_path), results_dir=str(tmp_path),
        temperatures=[700], p_vals_pa=[1e5], a0_m=3.52e-10, l_m=1e-3,
        d0_m2s=1.5e-7, e_d_ev=0.4, dh_diss_ev=None, dh_entry_ev=None,
        nx=10, ny=10, seed=42, kmc_max_steps=1000,
        gpu_slurm_cfg=_SLURM, neb_slurm_cfg=_SLURM, vib_slurm_cfg=_SLURM,
        n_images=6, spring_const=1.0, neb_ftol=0.1, out_py=out,
        **kwargs,
    )
    return open(out).read()


# ─── permeation script generation ─────────────────────────────────────────────

def test_permeation_metal_type_injected_oxide(tmp_path):
    src = _generate(tmp_path, 'perm_oxide.py', metal_type='oxide')
    compile(src, 'perm_oxide.py', 'exec')
    assert "METAL_TYPE = 'oxide'" in src
    assert 'metal_type=METAL_TYPE' in src   # forwarded to build_subsurface_graph


def test_permeation_metal_type_defaults_to_alloy(tmp_path):
    src = _generate(tmp_path, 'perm_alloy.py')
    assert "METAL_TYPE = 'alloy'" in src


def test_permeation_species_are_slab_derived(tmp_path):
    """The k_entry scan and diss placeholder pairs must come from the slab,
    not a hardcoded metal tuple — oxides have O; other alloys differ too."""
    src = _generate(tmp_path, 'perm_species.py', metal_type='oxide')
    assert "('Ni', 'Mo', 'Cr', 'Fe')" not in src
    assert '_slab_species' in src
    assert 'combinations_with_replacement' in src
    # bare-sid fallback resolves site composition from surface_sites.json
    assert '_sid2comp' in src


def test_permeation_oxide_kmc_composition(tmp_path):
    src = _generate(tmp_path, 'perm_kmc.py', metal_type='oxide')
    assert '_kmc_composition' in src
    assert 'composition = _kmc_composition' in src
    # metals keep make_grid's default: composition only set when oxide
    assert "if METAL_TYPE == 'oxide':" in src


# ─── metal_type threading through Part 1 site enumeration ────────────────────

def test_site_enumeration_accepts_metal_type():
    from models import neb_workflow
    p3 = inspect.signature(neb_workflow.run_phase3_site_enumeration).parameters
    assert 'metal_type' in p3 and p3['metal_type'].default == 'alloy'
    assert 'n_layers_total' in p3 and p3['n_layers_total'].default == 12

    prep = inspect.signature(neb_workflow.orchestrate_slab_prep).parameters
    assert prep['z_freeze_cutoff'].default is None   # auto bottom-1/3


def test_subsurface_graph_accepts_metal_type():
    from models import subsurface_graph
    params = inspect.signature(subsurface_graph.build_subsurface_graph).parameters
    assert 'metal_type' in params and params['metal_type'].default == 'alloy'


# ─── oxide slab auto-thickness (needs spglib; runs on cluster) ────────────────

def test_build_slab_oxide_auto_thickness(tmp_path):
    pytest.importorskip('spglib')
    from ase.build import bulk
    from models.structure import build_slab, write_lammps_data
    from ase.io import read as ase_read

    masses = {1: (58.6934, 'Ni'), 2: (15.999, 'O')}
    e2t    = {'Ni': 1, 'O': 2}

    nio = bulk('NiO', 'rocksalt', a=4.17, cubic=True).repeat((2, 2, 2))
    bulk_path = str(tmp_path / 'nio_bulk.lammps')
    write_lammps_data(symbols=nio.get_chemical_symbols(),
                      positions=nio.get_positions(),
                      cell_lengths=nio.cell.diagonal(),
                      masses=masses, e2t=e2t, out_path=bulk_path,
                      comment='NiO bulk')

    thicknesses = []
    for layers in (3, 12):   # must be IGNORED on the oxide path
        out = str(tmp_path / f'slab_L{layers}.lammps')
        build_slab(bulk_min_path=bulk_path, miller=(1, 0, 0), layers=layers,
                   vacuum=10.0, masses=masses, e2t=e2t, out_path=out,
                   metal_type='oxide')
        z = ase_read(out, format='lammps-data',
                     atom_style='atomic').get_positions()[:, 2]
        thicknesses.append(float(z.max() - z.min()))

    # both requests land near the 22 Å target, independent of `layers`
    for t in thicknesses:
        assert abs(t - 22.0) < 4.0, f'thickness {t:.1f} Å far from target'
    assert thicknesses[0] == pytest.approx(thicknesses[1], abs=1e-6), \
        'layers argument must not change oxide slab thickness'

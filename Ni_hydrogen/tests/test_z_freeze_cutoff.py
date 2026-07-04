"""
tests/test_z_freeze_cutoff.py
==============================
Unit tests for models.structure.compute_z_freeze_cutoff — fully offline.

The auto freeze cutoff must:
  * freeze exactly round(N/3) layers for any equally spaced N-layer slab
    (8 → 3, 12 → 4, 13 → 4, ...);
  * never coincide with an atomic plane — the raw thickness/3 cutoff lands
    exactly ON a plane whenever N ≡ 1 (mod 3), so the returned value is
    snapped to the midpoint of the surrounding interlayer gap;
  * work for planes with unequal atom populations (oxide-like slabs).
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from ase.build import fcc111
from ase.io import write as ase_write

from models.structure import compute_z_freeze_cutoff, write_lammps_data

_MASSES_NI = {1: (58.6934, 'Ni')}
_E2T_NI    = {'Ni': 1}


def _write_slab(atoms, path):
    write_lammps_data(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell_lengths=atoms.cell.diagonal(),
        masses=_MASSES_NI,
        e2t=_E2T_NI,
        out_path=str(path),
        comment='z-freeze test slab',
    )
    return str(path)


@pytest.mark.parametrize('n_layers', list(range(3, 17)))
def test_frozen_count_is_round_n_over_3(tmp_path, n_layers):
    slab = fcc111('Ni', size=(2, 2, n_layers), vacuum=10.0)
    p = _write_slab(slab, tmp_path / f'slab_{n_layers}.lammps')

    cutoff = compute_z_freeze_cutoff(p)

    z = slab.get_positions()[:, 2]
    planes = np.unique(np.round(z, 3))
    frozen = int((planes < cutoff).sum())
    assert frozen == round(n_layers / 3), (
        f'N={n_layers}: froze {frozen} layers, expected {round(n_layers / 3)}'
    )


@pytest.mark.parametrize('n_layers', [12, 13, 14])
def test_cutoff_never_touches_a_plane(tmp_path, n_layers):
    """N=13 is the dangerous case: raw thickness/3 lands exactly ON plane 5."""
    slab = fcc111('Ni', size=(2, 2, n_layers), vacuum=10.0)
    p = _write_slab(slab, tmp_path / f'slab_{n_layers}.lammps')

    cutoff = compute_z_freeze_cutoff(p)
    z = slab.get_positions()[:, 2]

    # Ni(111) interlayer spacing d = a0/sqrt(3) ≈ 2.03 Å → midpoint snap
    # guarantees ≈ d/2 ≈ 1.0 Å clearance to the nearest atom.
    clearance = float(np.abs(z - cutoff).min())
    assert clearance > 0.9, (
        f'N={n_layers}: cutoff {cutoff:.3f} only {clearance:.3f} Å from an '
        f'atomic plane — boundary layer could split under thermal noise'
    )


def test_unequal_plane_populations(tmp_path):
    """Oxide-like slab: planes with different atom counts still freeze 1/3."""
    # 6 planes at z = 0, 2, 4, 6, 8, 10 with populations 3, 1, 3, 1, 3, 1
    positions = []
    for k, n_atoms in enumerate([3, 1, 3, 1, 3, 1]):
        for i in range(n_atoms):
            positions.append([1.0 + 2.0 * i, 1.0, 2.0 * k])
    positions = np.array(positions, dtype=float)

    p = str(tmp_path / 'uneven.lammps')
    write_lammps_data(
        symbols=['Ni'] * len(positions),
        positions=positions,
        cell_lengths=[8.0, 4.0, 20.0],
        masses=_MASSES_NI,
        e2t=_E2T_NI,
        out_path=p,
        comment='uneven-population test slab',
    )

    cutoff = compute_z_freeze_cutoff(p)
    planes = np.unique(positions[:, 2])
    frozen = int((planes < cutoff).sum())
    assert frozen == round(len(planes) / 3) == 2
    # snapped into the gap: at least ~0.9 Å from any atom
    assert float(np.abs(positions[:, 2] - cutoff).min()) > 0.9


def test_custom_fraction(tmp_path):
    slab = fcc111('Ni', size=(2, 2, 12), vacuum=10.0)
    p = _write_slab(slab, tmp_path / 'slab_frac.lammps')

    cutoff_half = compute_z_freeze_cutoff(p, fraction=0.5)
    z = slab.get_positions()[:, 2]
    planes = np.unique(np.round(z, 3))
    frozen = int((planes < cutoff_half).sum())
    assert frozen == 6   # half of 12

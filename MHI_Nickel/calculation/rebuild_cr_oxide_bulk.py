#!/usr/bin/env python3
"""
calculation/rebuild_cr_oxide_bulk.py
=====================================
One-off script: rebuilds Cr2O3's bulk supercell with its correct hexagonal
(R-3c, spacegroup 167, gamma=120deg) cell and overwrites the corrupted
calculation/input_structure/Cr_oxide_supercell.lammps.

Background
----------
The file currently on disk declares an ORTHOGONAL box (90/90/90) even
though Cr2O3 (corundum) is hexagonal (a=b=5.00, c=13.51, gamma=120deg).
old_notebooks/01_bulk_Cr2O3_generation.ipynb built the crystal correctly
via ase.spacegroup.crystal, but its hand-rolled LAMMPS writer only wrote
`L = supercell.cell.lengths()` -- silently dropping gamma. Every metal's
build_slab() 'oxide' branch runs spglib.find_primitive() on this cell to
find the true primitive cell before building a slab; fed a corrupted
90-degree box, spglib finds a bogus low-symmetry ~150-atom "primitive"
cell instead of the true 10-atom R-3c primitive, and the resulting slab
comes out ~15x too large (22,500 atoms instead of ~1500), which is what
caused Cr_oxide's Phase 2 CUDA OOM.

This script reconstructs the same crystal (same spacegroup, cell
parameters, and basis as the old notebook -- verified by direct read of
its cells) and writes it through the now-triclinic-capable
write_lammps_data() (models/structure.py), using ASE's Prism class to
convert the cell into LAMMPS's canonical lower-triangular triclinic form
and to transform atom positions into that frame.

Known, deliberate omission: the old notebook's AFM magnetic-moment
initialization (alternating +/-3.0 on Cr sites) is not reproduced here --
plain LAMMPS atomic-style data files have no magmom field, and nothing in
the current LAMMPS/MACE pipeline reads per-atom magnetic moments from this
file.

Safe to re-run: deterministically rebuilds the same structure every time.
Does NOT touch calculation/structures/bulk_min_Cr_oxide_supercell.lammps
(the already-minimized-from-the-wrong-box output) -- Phase 1a needs to be
re-run against the corrected input to regenerate that file.

Usage
-----
    python calculation/rebuild_cr_oxide_bulk.py [--dry-run]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ase.spacegroup import crystal
from ase.calculators.lammps.coordinatetransform import Prism

from models.config import E2T_10, MASSES_10
from models.structure import write_lammps_data

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(WORK_DIR, 'input_structure', 'Cr_oxide_supercell.lammps')

# Exact cell parameters, spacegroup, and basis as
# old_notebooks/01_bulk_Cr2O3_generation.ipynb (cells 4-5).
_ATOM_NAME = 'Cr'
_A, _B, _C = 5.00, 5.00, 13.51
_ALPHA, _BETA, _GAMMA = 90, 90, 120
_SPACEGROUP = 167
_BASIS = [(1 / 3, 2 / 3, 0.016523), (0.635025, 0.968359, 0.916667)]
_SUPERCELL_REPS = (5, 5, 5)


def build_supercell():
    unit_cell = crystal(
        symbols=[_ATOM_NAME, 'O'],
        basis=_BASIS,
        spacegroup=_SPACEGROUP,
        cellpar=[_A, _B, _C, _ALPHA, _BETA, _GAMMA],
    )
    return unit_cell * _SUPERCELL_REPS


def main(dry_run):
    supercell = build_supercell()
    n_atoms = len(supercell)
    print(f'Built Cr2O3 supercell: {n_atoms} atoms, cellpar={supercell.cell.cellpar()}')

    prism = Prism(supercell.cell)
    lx, ly, lz, xy, xz, yz = prism.get_lammps_prism()
    lammps_positions = prism.vector_to_lammps(supercell.get_positions())
    print(f'LAMMPS triclinic box: lx={lx:.6f} ly={ly:.6f} lz={lz:.6f} '
          f'xy={xy:.6f} xz={xz:.6f} yz={yz:.6f}')

    if dry_run:
        print(f'[dry-run] would overwrite: {OUT_PATH}')
        return

    write_lammps_data(
        symbols=supercell.get_chemical_symbols(),
        positions=lammps_positions,
        cell_lengths=[lx, ly, lz],
        masses=MASSES_10,
        e2t=E2T_10,
        out_path=OUT_PATH,
        comment='Cr2O3 corundum bulk (R-3c, spacegroup 167) -- rebuilt with correct '
                'hexagonal cell, replacing a corrupted orthogonal-box version',
        tilt_factors=(xy, xz, yz),
    )
    print(f'Written: {OUT_PATH} ({n_atoms} atoms)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                         help='Preview without overwriting the output file.')
    args = parser.parse_args()
    main(dry_run=args.dry_run)

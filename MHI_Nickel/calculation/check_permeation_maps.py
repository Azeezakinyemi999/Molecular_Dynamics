#!/usr/bin/env python3
"""
check_permeation_maps.py
========================
Fast, read-only Level-1 validation of the reframed subsurface-entry enumeration
on REAL data -- no SLURM, no LAMMPS/MACE, runs in seconds.

For every metal (or one named on the command line) it:
  1. reports upstream readiness (which Part 1 / Part 3 files exist), and
  2. for metals whose Part 1 outputs are present, builds the subsurface graph
     and the three entry maps and prints the key counts:
       - sub1 / sub2 octahedral site counts and how many sub1 map to a sub2
       - number of dissociation-product H* seeding Hop A
       - number of Hop A pathways after the one-per-sub1 collapse
         (this should be the dissociation-seeded set, NOT the ~171 adsorption
          sites the old wholesale glob produced)
       - the distinct sub1 / sub2 oct-site environment classes

Nothing is written or submitted. Use this to confirm Part 1 behaves on a
metal's real data before committing to a multi-hour permeation run.

Usage
-----
    python check_permeation_maps.py                 # all metals
    python check_permeation_maps.py Hastelloy_N_1234_supercell   # one metal
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ase.io import read as ase_read

from models.config import BASE_DIR
from models.subsurface_graph import build_subsurface_graph, connect_to_surface
from models.neb_subsurface import (
    build_sub1_sub2_map, collect_entry_h_sources, build_surface_sub1_sub2_map,
)

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

_STEMS = [
    'Hastelloy_N_7_supercell', 'Cr_oxide_supercell', 'Hastelloy_N_42_supercell',
    'Hastelloy_N_111_supercell', 'Hastelloy_N_1234_supercell',
    'Hastelloy_N_12345_supercell', 'Al_supercell', 'Fe_supercell',
    'Ni_supercell', 'bestsqs3', 'Ni_oxide_supercell',
]


def _classify(stem):
    s = stem.lower()
    if 'oxide' in s:
        return 'oxide'
    if any(k in s for k in ('hastelloy', 'bestsqs', 'sqs', 'alloy')):
        return 'alloy'
    return 'pure'


def _paths(stem):
    return {
        'slab':       os.path.join(WORK_DIR, f'slabs/{stem}/phase2_relax/relaxed_slab.lammps'),
        'sites':      os.path.join(WORK_DIR, f'slabs/{stem}/phase3_sites/surface_sites.json'),
        'neb_dir':    os.path.join(WORK_DIR, f'neb/{stem}'),
        'ranked':     os.path.join(WORK_DIR, f'neb/{stem}/ranked_barriers.json'),
        'phase2_h':   os.path.join(WORK_DIR, f'adsorption/{stem}/phase2_h/results'),
        'diff':       os.path.join(WORK_DIR, f'results/{stem}_1H/diffusivity_arrhenius.json'),
        'lattice':    os.path.join(WORK_DIR, 'results/lattice_params_vs_T.json'),
    }


def _readiness(p):
    h_atom = glob.glob(os.path.join(p['phase2_h'], 'h_atom_*_relaxed.lammps'))
    return {
        'relaxed_slab':        os.path.exists(p['slab']),
        'surface_sites':       os.path.exists(p['sites']),
        'ranked_barriers':     os.path.exists(p['ranked']),
        'h_atom_structures':   len(h_atom),
        'diffusivity_fit':     os.path.exists(p['diff']),
        'lattice_params_vs_T': os.path.exists(p['lattice']),
    }


def check_one(stem):
    print(f'\n{"="*72}\n{stem}  (metal_type={_classify(stem)})\n{"="*72}')
    p = _paths(stem)
    r = _readiness(p)
    for k, v in r.items():
        mark = 'OK ' if v else '-- '
        print(f'  [{mark}] {k}: {v}')

    part1_ready = r['relaxed_slab'] and r['surface_sites'] and \
        r['ranked_barriers'] and r['h_atom_structures'] > 0
    if not part1_ready:
        print('  Part 1 outputs incomplete — cannot build maps yet.')
        return
    part3_ready = r['diffusivity_fit'] and r['lattice_params_vs_T']
    print(f'  Part 3 (diffusivity) ready for a full run: {part3_ready}')

    # ── build the maps (read-only; out_json=None) ─────────────────────────────
    with open(p['sites']) as f:
        surf_data = json.load(f)
    G, subsurface_sites = build_subsurface_graph(
        p['slab'], p['sites'], seed=42, metal_type=_classify(stem))
    cell = ase_read(p['slab'], format='lammps-data', atom_style='atomic').get_cell().diagonal()
    surface_connections = connect_to_surface(subsurface_sites, surf_data, cell)

    n_sub1 = sum(1 for s in subsurface_sites if s.get('layer_classification') == 'subsurface_1')
    n_sub2 = sum(1 for s in subsurface_sites if s.get('layer_classification') == 'subsurface_2')

    sub1_sub2 = build_sub1_sub2_map((G, subsurface_sites))
    entry = collect_entry_h_sources(p['neb_dir'], p['phase2_h'])
    path_map = build_surface_sub1_sub2_map(entry, surface_connections, sub1_sub2,
                                           (G, subsurface_sites))

    n_matched = sum(1 for v in sub1_sub2.values() if v.get('sub2_id'))
    n_adsorption_sites = len(glob.glob(os.path.join(p['phase2_h'], 'h_atom_*_relaxed.lammps')))
    sub1_envs = sorted({e['sub1_env'] for e in path_map})
    sub2_envs = sorted({e.get('sub2_env') for e in path_map if e.get('sub2_env')})

    print(f'\n  RESULTS')
    print(f'    sub1 oct sites            : {n_sub1}')
    print(f'    sub2 oct sites            : {n_sub2}')
    print(f'    sub1→sub2 mapped          : {n_matched}/{n_sub1}')
    print(f'    dissociation-product H*   : {len(entry)}')
    print(f'    Hop A pathways (collapsed): {len(path_map)}')
    print(f'    (old wholesale glob would have been ~{n_adsorption_sites} adsorption sites)')
    print(f'    distinct sub1 env classes : {len(sub1_envs)}  {sub1_envs}')
    print(f'    distinct sub2 env classes : {len(sub2_envs)}  {sub2_envs}')


def main():
    stems = sys.argv[1:] or _STEMS
    for stem in stems:
        try:
            check_one(stem)
        except Exception as exc:                       # noqa: BLE001
            print(f'  ERROR for {stem}: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()

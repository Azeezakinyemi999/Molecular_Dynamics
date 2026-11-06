#!/usr/bin/env python3
"""
verify_padded_species_mace.py
===============================
One-off verification: does declaring extra, UNUSED species in `pair_coeff`
change MACE-MH-1's computed energy/forces for the atoms that are actually
present?

Why this script exists
-----------------------
`calculate_ref_adsorbate_energy()` (models/neb_workflow.py) computes the
isolated-H2 gas-phase reference energy once, shared by every metal
(pipeline.ipynb cell 5). Today it's called with `elem_str=ELEM_STR_7,
e2t=E2T_7, masses=MASSES_7` -- the historical fixed 8-type table (Al, B, C,
Cr, Fe, Mo, Ni, H) -- even though the H2 calculation only ever has 2 H
atoms in it. For a material whose own slab/adsorption tables don't even
contain those elements (e.g. a 5-element alloy), this violates the
function's own documented contract that `elem_str` "must be identical to
the slab and adsorption calculations so all three E_ads terms use the same
model."

Removing the hardcoded ELEM_STR_7/E2T_7/MASSES_7 constants means this one
call site needs a real replacement. There's no structure file to
auto-detect from here (H2 is a synthetic vacuum-box molecule, not read from
input_structure/), so the natural replacement is a MINIMAL table built
directly from what's physically present: `build_element_tables(['H'])` --
one type, pair_coeff "H". This script tests whether that minimal table
gives the identical physics to today's padded 8-type table, rather than
assuming it (same project ethos as verify_element_order_mace.py).

Two cases, identical physical system (2 H atoms, H2 bond length, vacuum box)
-- only the declared species list differs:

  D. Minimal table (the proposed replacement):
     build_element_tables(['H']) -> H=1; pair_coeff "H".

  E. Padded table (today's production behavior):
     build_element_tables(['Al','B','C','Cr','Fe','Mo','Ni','H']) -> H=8;
     pair_coeff "Al B C Cr Fe Mo Ni H". The other 7 types are declared but
     have zero atoms of that type anywhere in the box.

Usage
-----
    python calculation/tools/verify_padded_species_mace.py

Run on a login node (just generates files -- instant). Then:

    sbatch <path printed at the end>

What success looks like
------------------------
- E(D) and E(E) agree to a tight tolerance (not necessarily bit-identical --
  summation order inside mliap can differ -- but far closer than any
  physically meaningful difference), and per-atom forces/PE agree too.
  Confirms padding with unused species is harmless, and the minimal
  build_element_tables(['H']) table is a safe drop-in replacement for
  ELEM_STR_7/E2T_7/MASSES_7 at this call site.

- If E(D) and E(E) differ meaningfully, that's a real, important finding:
  it would mean today's shared, pre-computed E_H2_GAS value is NOT
  guaranteed consistent with a material whose own table differs from
  ELEM_STR_7, and the fix would need to go further than a drop-in
  substitution (e.g. computing E_H2_GAS per-material, using that
  material's own elem_str/e2t/masses, rather than one shared global value).
"""
import os
import sys

# Project root = the directory containing models/. Found by walking up
# rather than counting parents, so this keeps working wherever the script
# is moved to (same pattern as verify_element_order_mace.py).
_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, 'models')):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

import numpy as np

from models.config import (
    build_element_tables, LAMMPS_CMD, MACE_MODEL_LAMMPS, KOKKOS_FLAGS,
    PAIR_STYLE, PAIR_SUFFIX, SLURM_DEFAULTS, BASE_DIR,
)
from models.structure import write_lammps_data
from models.create_slurm import write_slurm_job

OUTDIR = os.path.join(BASE_DIR, 'calculation', 'padded_species_verification')
os.makedirs(OUTDIR, exist_ok=True)

# Same geometry calculate_ref_adsorbate_energy() uses for H2: a single bond
# length, centered in a cubic vacuum box.
BOX_SIZE = 15.0  # Angstrom
H2_BOND = 0.741  # Angstrom
_center = BOX_SIZE / 2.0
POSITIONS = np.array([
    [_center - H2_BOND / 2.0, _center, _center],
    [_center + H2_BOND / 2.0, _center, _center],
])
CELL = [BOX_SIZE, BOX_SIZE, BOX_SIZE]
SYMBOLS = ['H', 'H']

# Padded list mirrors models/config.py's ELEM_STR_7 exactly (Al, B, C, Cr,
# Fe, Mo, Ni, H) -- today's actual production call.
PADDED_ELEMENTS = ['Al', 'B', 'C', 'Cr', 'Fe', 'Mo', 'Ni', 'H']

E2T_D, T2E_D, MASSES_D, ELEM_STR_D = build_element_tables(['H'])
E2T_E, T2E_E, MASSES_E, ELEM_STR_E = build_element_tables(PADDED_ELEMENTS)

print(f'Case D (minimal)  : E2T={E2T_D}  pair_coeff "{ELEM_STR_D}"')
print(f'Case E (padded, today\'s ELEM_STR_7): E2T={E2T_E}  pair_coeff "{ELEM_STR_E}"')

data_D = write_lammps_data(
    symbols=SYMBOLS, positions=POSITIONS, cell_lengths=CELL,
    masses=MASSES_D, e2t=E2T_D, out_path=os.path.join(OUTDIR, 'struct_D.lammps'),
    comment='Case D: minimal table, H only',
)
data_E = write_lammps_data(
    symbols=SYMBOLS, positions=POSITIONS, cell_lengths=CELL,
    masses=MASSES_E, e2t=E2T_E, out_path=os.path.join(OUTDIR, 'struct_E.lammps'),
    comment='Case E: padded table, today\'s ELEM_STR_7 (H=type 8, 7 unused types)',
)

KK = ' '.join(KOKKOS_FLAGS)


def _write_input_script(label, data_path, masses, elem_str, out_path, forces_path):
    mass_lines = '\n'.join(
        f'mass           {t}  {m}  # {el}' for t, (m, el) in sorted(masses.items())
    )
    script = f'''# {label}
units           metal
atom_style      atomic
boundary        p p p

read_data       {data_path}

{mass_lines}

pair_style      {PAIR_STYLE} {MACE_MODEL_LAMMPS} {PAIR_SUFFIX}
pair_coeff      * * {elem_str}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

compute         peratom all pe/atom

thermo_style    custom step pe
thermo          1

dump            d1 all custom 1 {forces_path} id type x y z fx fy fz c_peratom
dump_modify     d1 sort id

run             0

variable        etot equal pe
print           "TOTAL_ENERGY ${{etot}}"
'''
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


in_D = _write_input_script(
    'Case D: minimal table, H only', data_D, MASSES_D, ELEM_STR_D,
    os.path.join(OUTDIR, 'in_D.lammps'), os.path.join(OUTDIR, 'forces_D.dump'),
)
in_E = _write_input_script(
    'Case E: padded table (today\'s ELEM_STR_7)', data_E, MASSES_E, ELEM_STR_E,
    os.path.join(OUTDIR, 'in_E.lammps'), os.path.join(OUTDIR, 'forces_E.dump'),
)

compare_py = os.path.join(OUTDIR, 'compare_results.py')
with open(compare_py, 'w') as f:
    f.write(f'''#!/usr/bin/env python3
"""Parse the two LAMMPS logs + force dumps and report the comparison."""
import re

OUTDIR = {OUTDIR!r}


def _read_energy(log_path):
    with open(log_path) as fh:
        text = fh.read()
    m = re.search(r'TOTAL_ENERGY\\s+([-\\d.eE+]+)', text)
    if not m:
        raise RuntimeError(f'No TOTAL_ENERGY found in {{log_path}} -- job likely crashed, check the .log/.out file')
    return float(m.group(1))


def _read_dump(dump_path):
    """Return {{atom_id: (type, fx, fy, fz, pe)}}, sorted/ided by dump_modify sort id."""
    with open(dump_path) as fh:
        lines = fh.readlines()
    i = lines.index('ITEM: ATOMS id type x y z fx fy fz c_peratom\\n') + 1
    out = {{}}
    for line in lines[i:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        aid, atype = int(parts[0]), int(parts[1])
        fx, fy, fz, pe = map(float, parts[5:9])
        out[aid] = (atype, fx, fy, fz, pe)
    return out


e_d = _read_energy(f'{{OUTDIR}}/log_D.out')
e_e = _read_energy(f'{{OUTDIR}}/log_E.out')

d_d = _read_dump(f'{{OUTDIR}}/forces_D.dump')
d_e = _read_dump(f'{{OUTDIR}}/forces_E.dump')

lines = []
lines.append(f'E(D) minimal table (H only)          = {{e_d:.8f}} eV')
lines.append(f'E(E) padded table (today\\'s ELEM_STR_7) = {{e_e:.8f}} eV')
lines.append('')
lines.append(f'|E(D) - E(E)| = {{abs(e_d - e_e):.3e}} eV  (expect ~0 -- unused declared species should not matter)')
lines.append('')
lines.append('Per-atom comparison, D vs E (same physical atoms, same positions):')
for aid in sorted(d_d):
    _td, fxd, fyd, fzd, ped = d_d[aid]
    _te, fxe, fye, fze, pee = d_e[aid]
    dpe = abs(ped - pee)
    dforce = ((fxd - fxe) ** 2 + (fyd - fye) ** 2 + (fzd - fze) ** 2) ** 0.5
    lines.append(f'  atom {{aid}}: |dPE|={{dpe:.3e}} eV, |dF|={{dforce:.3e}} eV/A')

report = '\\n'.join(lines)
print(report)
with open(f'{{OUTDIR}}/energy_comparison.txt', 'w') as fh:
    fh.write(report + '\\n')
''')

sh_path = os.path.join(OUTDIR, 'run_verification.sh')
write_slurm_job(
    job_name='verify_padded_species',
    slurm_config=dict(SLURM_DEFAULTS, partition='sharing', time='00:10:00'),
    out_path=sh_path,
    commands=[
        f'{LAMMPS_CMD} {KK} -in {in_D} -log {os.path.join(OUTDIR, "log_D.out")}',
        f'{LAMMPS_CMD} {KK} -in {in_E} -log {os.path.join(OUTDIR, "log_E.out")}',
        f'python {compare_py}',
    ],
)

print(f'\nWrote verification job: {sh_path}')
print('Nothing submitted. Next:')
print(f'  sbatch {sh_path}')
print(f'Then: cat {os.path.join(OUTDIR, "energy_comparison.txt")}')

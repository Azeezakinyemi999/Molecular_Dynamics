#!/usr/bin/env python3
"""
verify_element_order_mace.py
=============================
One-off verification: does MACE-MH-1 give the right per-element potential
when atom types are ordered alphabetically with H last (this project's
`build_element_tables` convention), and what does it do if the type-ID
arrangement is wrong instead?

Why this script exists
-----------------------
`models/config.py:build_element_tables` assigns LAMMPS atom type IDs by a
fixed rule -- elements alphabetically, H last -- and builds `MASSES` and
`ELEM_STR` (the `pair_coeff * * {elem_str}` line) from that same ordered
list, so a type ID's mass and its MACE species label always agree *by
construction*. The claim behind that design (argued, not yet measured) is
that MACE/LAMMPS's `mliap unified` interface resolves each atom type to a
species by the *name* in `elem_str`, not by its position matching some
fixed internal training order -- so alphabetical order itself carries no
physical meaning to MACE; only internal consistency between the data
file's type assignment and `elem_str` does. This script tests that claim
directly, on real cluster hardware with the real MACE-MH-1 potential,
rather than trusting the argument.

Three cases, identical 5-atom structure (2 Fe, 2 Ni, 1 H at an octahedral
site, one small periodic FCC-like cell) -- only the type-ID/species
bookkeeping differs between them:

  A. Alphabetical, self-consistent (this project's actual convention):
     build_element_tables(['Fe', 'Ni', 'H']) -> Fe=1, Ni=2, H=3;
     pair_coeff "Fe Ni H". Data file's Atoms section uses these type IDs.

  B. Non-alphabetical, self-consistent:
     build_element_tables(['Ni', 'Fe', 'H']) -> Ni=1, Fe=2, H=3;
     pair_coeff "Ni Fe H". Same physical atoms as A (same symbol at each
     position), just swapped type-ID/species bookkeeping -- but still
     internally matched: type 1's mass, type 1's pair_coeff label, and the
     atoms actually written as type 1 all agree (Ni, Ni, Ni).
     If A and B give the same energy and forces, order alone does not
     matter to MACE -- only internal consistency does.

  C. Mismatched ("wrong arrangement"): case A's data file (Fe=1, Ni=2,
     H=3 physically) run with case B's pair_coeff/mass bundle
     ("Ni Fe H"). Type 1 is physically Fe, but MACE is told it is Ni, and
     vice versa. This is a controlled, deliberate reproduction of the
     exact silent species-mislabeling failure mode already found once in
     this codebase -- see [[project_pipeline_test_bugs]] Bug #20
     (ase_neb.py's bare `ase.io.write` with no masses, which on read-back
     silently relabeled Ni/H as N/O). That bug was a missing-masses read
     issue, not a type-ID/pair_coeff mismatch, but the consequence --
     MACE silently evaluating the wrong species -- is the same class of
     error this script checks for directly. Note: `mass` command values
     do not affect `pair_style mliap`'s energy/force evaluation at all
     (they only matter for MD integration, and this is a single `run 0`,
     not MD) -- the only thing that matters for this test is `elem_str`.

Usage
-----
    python calculation/tools/verify_element_order_mace.py

Run on a login node (just generates files -- instant, no LAMMPS needed for
this step). Then:

    sbatch <path printed at the end>

That job runs all three cases (five LAMMPS atoms, `run 0`, each is a few
seconds at most) and writes `energy_comparison.txt` in the output
directory with the parsed result.

What success looks like
------------------------
- E(A) and E(B) agree to a tight tolerance (not necessarily bit-identical
  -- summation order inside mliap can differ -- but far closer than any
  physically meaningful difference). Confirms order itself is irrelevant
  to MACE; only data-file/pair_coeff consistency matters.
- E(A) and E(C) differ clearly, and the per-atom forces/energies on the
  Fe/Ni atoms (not H, whose identity wasn't swapped) differ between A and
  C. Confirms a type-ID/pair_coeff mismatch silently gives wrong physics
  -- LAMMPS raises no error for this at all.

If E(A) and E(C) come out equal instead, that is a real, useful, and
non-obvious finding in its own right (e.g. MACE-MH-1's learned Fe and Ni
descriptors happen to be very close at this particular local environment)
-- rerun with a more different element pair (edit SWAP_ELEMENTS below)
before trusting that conclusion in general.
"""
import os
import sys

# Project root = the directory containing models/. Found by walking up
# rather than counting parents, so this keeps working wherever the script
# is moved to (same pattern as calculation/tools/legacy/verify_neb_fixes.py).
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

OUTDIR = os.path.join(BASE_DIR, 'calculation', 'element_order_verification')
os.makedirs(OUTDIR, exist_ok=True)

# The two metals being swapped between cases. Change these (e.g. to a pair
# with very different cohesive energies) if A vs C ever comes out
# surprisingly close and you want a sharper contrast.
SWAP_ELEMENTS = ['Fe', 'Ni']

# ── Build one small, shared physical structure: a 1x1x1 FCC conventional
# cell (4 metal sites) + 1 H at the body-center octahedral site. Only the
# symbol assigned to each site and the type-ID bookkeeping change between
# cases -- the positions are identical in all three. ──────────────────────
A0 = 3.55  # Angstrom, generic FCC transition-metal lattice parameter
FRACTIONAL = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [0.5, 0.5, 0.5],   # octahedral interstitial -> H
])
POSITIONS = FRACTIONAL * A0
CELL = [A0, A0, A0]
EL1, EL2 = SWAP_ELEMENTS
SYMBOLS = [EL1, EL1, EL2, EL2, 'H']   # fixed physical assignment, every case

E2T_A, T2E_A, MASSES_A, ELEM_STR_A = build_element_tables([EL1, EL2, 'H'])
E2T_B, T2E_B, MASSES_B, ELEM_STR_B = build_element_tables([EL2, EL1, 'H'])

print(f'Case A (alphabetical)     : E2T={E2T_A}  pair_coeff "{ELEM_STR_A}"')
print(f'Case B (swapped, matched) : E2T={E2T_B}  pair_coeff "{ELEM_STR_B}"')
print(f'Case C (swapped, MISMATCHED): data file = A, pair_coeff = "{ELEM_STR_B}"')

data_A = write_lammps_data(
    symbols=SYMBOLS, positions=POSITIONS, cell_lengths=CELL,
    masses=MASSES_A, e2t=E2T_A, out_path=os.path.join(OUTDIR, 'struct_A.lammps'),
    comment='Case A: alphabetical, self-consistent',
)
data_B = write_lammps_data(
    symbols=SYMBOLS, positions=POSITIONS, cell_lengths=CELL,
    masses=MASSES_B, e2t=E2T_B, out_path=os.path.join(OUTDIR, 'struct_B.lammps'),
    comment='Case B: non-alphabetical, self-consistent',
)
# Case C deliberately reuses case A's data file (same physical type
# assignment, Fe=1/Ni=2/H=3) -- only its LAMMPS *input script* (below)
# claims the type-ID/species bundle actually belonging to case B.

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


in_A = _write_input_script(
    'Case A: alphabetical, self-consistent', data_A, MASSES_A, ELEM_STR_A,
    os.path.join(OUTDIR, 'in_A.lammps'), os.path.join(OUTDIR, 'forces_A.dump'),
)
in_B = _write_input_script(
    'Case B: non-alphabetical, self-consistent', data_B, MASSES_B, ELEM_STR_B,
    os.path.join(OUTDIR, 'in_B.lammps'), os.path.join(OUTDIR, 'forces_B.dump'),
)
in_C = _write_input_script(
    'Case C: MISMATCHED -- data file is A, pair_coeff/masses are B\'s',
    data_A, MASSES_B, ELEM_STR_B,
    os.path.join(OUTDIR, 'in_C.lammps'), os.path.join(OUTDIR, 'forces_C.dump'),
)

compare_py = os.path.join(OUTDIR, 'compare_results.py')
with open(compare_py, 'w') as f:
    f.write(f'''#!/usr/bin/env python3
"""Parse the three LAMMPS logs + force dumps and report the comparison."""
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
    \"\"\"Return {{atom_id: (type, fx, fy, fz, pe)}}, sorted/ided by dump_modify sort id.\"\"\"
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


e_a = _read_energy(f'{{OUTDIR}}/log_A.out')
e_b = _read_energy(f'{{OUTDIR}}/log_B.out')
e_c = _read_energy(f'{{OUTDIR}}/log_C.out')

d_a = _read_dump(f'{{OUTDIR}}/forces_A.dump')
d_c = _read_dump(f'{{OUTDIR}}/forces_C.dump')

lines = []
lines.append(f'E(A) alphabetical, self-consistent   = {{e_a:.8f}} eV')
lines.append(f'E(B) swapped, self-consistent         = {{e_b:.8f}} eV')
lines.append(f'E(C) swapped, MISMATCHED              = {{e_c:.8f}} eV')
lines.append('')
lines.append(f'|E(A) - E(B)| = {{abs(e_a - e_b):.3e}} eV  (expect ~0 -- order alone should not matter)')
lines.append(f'|E(A) - E(C)| = {{abs(e_a - e_c):.3e}} eV  (expect clearly nonzero -- mismatch should matter)')
lines.append('')
lines.append('Per-atom comparison, A vs C (same physical atoms, same positions):')
for aid in sorted(d_a):
    ta, fxa, fya, fza, pea = d_a[aid]
    tc, fxc, fyc, fzc, pec = d_c[aid]
    dpe = abs(pea - pec)
    dforce = ((fxa - fxc) ** 2 + (fya - fyc) ** 2 + (fza - fzc) ** 2) ** 0.5
    lines.append(f'  atom {{aid}}: |dPE|={{dpe:.3e}} eV, |dF|={{dforce:.3e}} eV/A')

report = '\\n'.join(lines)
print(report)
with open(f'{{OUTDIR}}/energy_comparison.txt', 'w') as fh:
    fh.write(report + '\\n')
''')

sh_path = os.path.join(OUTDIR, 'run_verification.sh')
write_slurm_job(
    job_name='verify_elem_order',
    slurm_config=dict(SLURM_DEFAULTS, partition='sharing', time='00:10:00'),
    out_path=sh_path,
    commands=[
        f'{LAMMPS_CMD} {KK} -in {in_A} -log {os.path.join(OUTDIR, "log_A.out")}',
        f'{LAMMPS_CMD} {KK} -in {in_B} -log {os.path.join(OUTDIR, "log_B.out")}',
        f'{LAMMPS_CMD} {KK} -in {in_C} -log {os.path.join(OUTDIR, "log_C.out")}',
        f'python {compare_py}',
    ],
)

print(f'\nWrote verification job: {sh_path}')
print('Nothing submitted. Next:')
print(f'  sbatch {sh_path}')
print(f'Then: cat {os.path.join(OUTDIR, "energy_comparison.txt")}')

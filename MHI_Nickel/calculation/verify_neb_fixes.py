#!/usr/bin/env python3
"""
verify_neb_fixes.py
====================
One-off verification script for the 2026-07-09/10 NEB fixes (Bug 21 restart
correctness, FIRE optimizer, parallel image evaluation, float32 dtype, and
the fmax-formula fix). Not part of the production pipeline -- run this once
after deploying the fix to confirm it actually works on real cluster
hardware with the real MACE model, before trusting it for production runs.

What this does
--------------
1. Builds a small, real Ni(100) + adsorbed H slab (2x2x3, ASE-built, no
   LAMMPS/slab-prep pipeline needed) and relaxes IS/FS at two adjacent
   hollow sites using the REAL MACE model directly through ASE -- this is
   deliberately small/quick, not a production-quality surface calculation.
2. Writes real LAMMPS-data IS/FS files via models.structure.write_lammps_data
   (the same canonical writer the real pipeline uses).
3. Generates run_neb.py via the REAL, fixed run_neb_pipeline() -- exercises
   the actual production code path (FIRE, parallel=True, dtype=float32,
   the N_IMAGES-mismatch guard, append_trajectory, the fixed fmax formula),
   not a reimplementation.
4. Generates a SLURM chain script via the REAL write_chained_slurm_job(),
   targeting the `west` partition with a DELIBERATELY SHORT cutoff (2 min)
   so the chain-resubmit-on-timeout path gets exercised for real within a
   short test, not just a single uninterrupted leg.
5. Prints the sbatch command to run next.

This was locally verified (with ASE's EMT calculator substituted for MACE,
since the real model isn't available off-cluster) to correctly interrupt
mid-Phase-1 via a genuine SIGTERM, detect the checkpoint on the next leg,
resume Phase 1 rather than skipping to Phase 2, append rather than truncate
the trajectory, and converge with a physically sensible fmax_final. This
script exercises the identical code path with the real MACE model.

Usage
-----
    python verify_neb_fixes.py

Run directly on a login node (the IS/FS relaxation is small -- a 26-atom
slab, a few dozen ASE optimizer steps -- typically seconds; if your
cluster's login-node policy restricts this, wrap it in a quick one-shot job
on the `sharing` partition instead). Then:

    sbatch <path printed at the end>

Then watch:
    squeue -u $USER -n verify_neb_fix
    tail -f <outdir>/verify_phase1.log <outdir>/verify_phase2.log
    cat <outdir>/slurm_verify_neb_fix_*.out   # look for "Restart: Phase 1
                                               #  checkpoint found" on the
                                               # second+ leg if a timeout
                                               # happens

What success looks like
------------------------
- At least one `slurm_verify_neb_fix_*.out` shows "Command block exited
  with code 124" (a real SLURM timeout) followed by "Resubmitting".
- The following leg's .out shows "Restart: Phase 1 checkpoint found --
  fmax=... (N/2000 steps done) -- resuming Phase 1" (NOT "starting Phase
  2" unless N/2000 genuinely already reached fmax<=0.15).
- Eventually: <outdir>/neb_barrier.txt shows "Converged   : True" and
  "fmax_final" at or below 0.05 eV/A (the fixed formula -- this is the
  real NEB-projected force, not the old inflated raw-force value).
- No Python tracebacks in any .out file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ase.build import fcc100, add_adsorbate
from ase.optimize import QuasiNewton
from mace.calculators import MACECalculator

from models.ase_neb import run_neb_pipeline
from models.create_slurm import write_chained_slurm_job
from models.structure import write_lammps_data
from models.config import (
    MASSES_7, E2T_7, MACE_MODEL_ASE, MACE_HEAD, SLURM_DEFAULTS, BASE_DIR,
)

OUTDIR = os.path.join(BASE_DIR, 'calculation', 'neb_fix_verification')
os.makedirs(OUTDIR, exist_ok=True)


def _make_calc():
    return MACECalculator(model_paths=MACE_MODEL_ASE, device='cpu',
                           default_dtype='float32', head=MACE_HEAD)


def _make_slab():
    slab = fcc100('Ni', size=(2, 2, 3), vacuum=10.0)
    slab.calc = _make_calc()
    return slab


print('Building IS (H at hollow site 1) ...')
slab_is = _make_slab()
add_adsorbate(slab_is, 'H', 1.5, 'hollow')
slab_is.calc = _make_calc()
QuasiNewton(slab_is, logfile=None).run(fmax=0.05, steps=200)
e_is = slab_is.get_potential_energy()
z_lo = slab_is.get_positions()[:, 2].min()
print(f'  E_IS = {e_is:.4f} eV')

print('Building FS (H at adjacent hollow site) ...')
slab_fs = _make_slab()
add_adsorbate(slab_fs, 'H', 1.5, 'hollow', offset=(1, 0))
slab_fs.calc = _make_calc()
QuasiNewton(slab_fs, logfile=None).run(fmax=0.05, steps=200)
e_fs = slab_fs.get_potential_energy()
print(f'  E_FS = {e_fs:.4f} eV')

is_path = os.path.join(OUTDIR, 'neb_initial.lammps')
fs_path = os.path.join(OUTDIR, 'neb_final_relaxed.lammps')
write_lammps_data(symbols=slab_is.get_chemical_symbols(), positions=slab_is.get_positions(),
                   cell_lengths=slab_is.cell.diagonal(), masses=MASSES_7, e2t=E2T_7,
                   out_path=is_path, comment='IS (verify_neb_fixes.py)')
write_lammps_data(symbols=slab_fs.get_chemical_symbols(), positions=slab_fs.get_positions(),
                   cell_lengths=slab_fs.cell.diagonal(), masses=MASSES_7, e2t=E2T_7,
                   out_path=fs_path, comment='FS (verify_neb_fixes.py)')

# n_images/cpus_per_task deliberately small -- this is a fast fix-verification
# run, not production (which uses n_images=18, cpus_per_task=18).
N_IMAGES = 6

neb_script = run_neb_pipeline(
    is_file=is_path, fs_file=fs_path, e_is=e_is, e_fs=e_fs,
    mace_model_path=MACE_MODEL_ASE,
    barrier_file=os.path.join(OUTDIR, 'neb_barrier.txt'),
    path_file=os.path.join(OUTDIR, 'neb_path.dat'),
    outdir=OUTDIR,
    job_name='verify',
    n_images=N_IMAGES,
    spring_const=1.0,
    neb_ftol=0.05,
    phase1_steps=2000,
    phase2_steps=2000,
    z_freeze_cutoff=z_lo - 10.0,   # freeze nothing -- this is a mechanics
                                   # test, not a surface-freezing physics test
    device='cpu',
    traj_phase1=os.path.join(OUTDIR, 'neb_phase1.traj'),
    traj_phase2=os.path.join(OUTDIR, 'neb_phase2.traj'),
)
print(f'\nGenerated: {neb_script}')

west_slurm = dict(
    SLURM_DEFAULTS,
    partition='west',
    gpu=None,
    cpus_per_task=N_IMAGES,
    time='00:10:00',
)
chain_sh = os.path.join(OUTDIR, 'slurm_verify_neb_fix.sh')
write_chained_slurm_job(
    job_name='verify_neb_fix',
    slurm_config=west_slurm,
    out_path=chain_sh,
    first_commands=[f'python {neb_script}'],
    restart_commands=[f'python {neb_script}'],
    restart_glob=os.path.join(OUTDIR, 'neb_phase2.traj'),
    cutoff='00:02:00',   # deliberately short -- force at least one real
                         # SLURM timeout+resubmit cycle within a quick test
    work_dir=OUTDIR,
)
print(f'Generated: {chain_sh}')

print(f'''
Next step -- submit the chain job:

    sbatch {chain_sh}

Then watch:
    squeue -u $USER -n verify_neb_fix
    tail -f {OUTDIR}/verify_phase1.log {OUTDIR}/verify_phase2.log

Success = eventually {OUTDIR}/neb_barrier.txt shows "Converged   : True"
and "fmax_final" <= 0.05 eV/A, with at least one .out file in {OUTDIR}
showing a timeout ("Command block exited with code 124") followed by the
NEXT leg printing "Restart: Phase 1 checkpoint found ... resuming Phase 1"
(not silently skipping to Phase 2) -- see this file's module docstring for
the full checklist.
''')

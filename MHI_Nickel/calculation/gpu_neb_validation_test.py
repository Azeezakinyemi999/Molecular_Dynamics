#!/usr/bin/env python3
"""
calculation/gpu_neb_validation_test.py
=======================================
Real-structure validation: is NEB's slowness caused by device (CPU vs GPU),
model (mace-mh-1/omat_pbe vs the old mace-mp-0b2-medium), or both?

Uses one real, already-completed pair as the fixed input for every variant --
no synthetic data, nothing new to build:

    calculation/neb/Hastelloy_N_1234_supercell/neb/s_145__s_73+s_144/

  - neb_initial.lammps / neb_final_relaxed.lammps: real, already-relaxed
    IS/FS structures (362 atoms) -- read-only, never modified.
  - Its existing neb_phase1.log is a real CPU+mh-1 baseline already on the
    books: starting fmax 1118.7 eV/A, ~45-48s/step, 935+ steps without
    convergence. That's corner 1 of the 2x2 grid below, for free.

               model=mh-1 (current)          model=mp-0b2-medium (old)
    CPU        existing real log               cpu_mp0b2  (this script)
    GPU        gpu_mh1    (this script)        gpu_mp0b2  (this script)

Each new variant writes to its own subdirectory under
calculation/gpu_neb_validation_test/ (sibling to calculation/neb/, never
inside it) -- guarantees zero interference with the real pair's own
production files regardless of whether its chain job is still active.

gpu_mh1 goes through the real, unmodified production code path
(models.ase_neb.run_neb_pipeline) with device='cuda', parallel=False --
this is the actual change being validated for rollout.

cpu_mp0b2 / gpu_mp0b2 use the old mace-mp-0b2-medium model, which has no
'head' argument (mace-mh-1/omat_pbe is a multi-head model; the old one
isn't). Rather than risk passing head=None into models.ase_neb's shared,
production make_mace_calc()/make_frozen_calc() helpers for the first time,
these two variants get their own small, self-contained driver script that
calls models.ase_neb's already-tested building blocks directly
(build_neb_images, run_cineb, extract_mep, write_neb_results -- the exact
pattern documented in that module's own "Typical usage" docstring) with a
local frozen-calculator wrapper that omits 'head' entirely. Zero changes
to any shared/production file.

Simplification vs production for cpu_mp0b2/gpu_mp0b2 only: no
trajectory-checkpoint restart. If a leg times out and self-resubmits, it
restarts NEB from scratch (fresh IDPP interpolation) rather than resuming
mid-optimization. Fine for a validation tool whose goal is comparative
early-behaviour data (seconds/step, starting fmax) -- not a converged
production barrier. gpu_mh1 (the real code path) keeps full checkpoint
restart, same as production.

Only WRITES scripts -- does not submit anything. Prints the sbatch
command for each variant; submit them yourself.

Usage
-----
    python calculation/gpu_neb_validation_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import (
    BASE_DIR, MACE_MODEL_ASE, E2T_7, MASSES_7,
    N_REPLICAS, SPRING_CONST, NEB_FTOL, SLURM_DEFAULTS,
)
from models.ase_neb import run_neb_pipeline
from models.create_slurm import write_chained_slurm_job

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

# ── Real pair, read-only ──────────────────────────────────────────────────
PAIR_DIR = os.path.join(
    WORK_DIR, 'neb', 'Hastelloy_N_1234_supercell', 'neb', 's_145__s_73+s_144')
IS_FILE  = os.path.join(PAIR_DIR, 'neb_initial.lammps')
FS_FILE  = os.path.join(PAIR_DIR, 'neb_final_relaxed.lammps')
FS_LOG   = os.path.join(PAIR_DIR, 'fs_min.log')
# Copied verbatim from this pair's already-generated run_neb.py.
E_IS     = -2265.59303235186
Z_CUTOFF = 22.21212258445

for _p in (IS_FILE, FS_FILE, FS_LOG):
    if not os.path.exists(_p):
        raise FileNotFoundError(
            f'{_p} not found -- this validation script depends on the real, '
            'already-completed Hastelloy_N_1234_supercell s_145__s_73+s_144 '
            'pair. Point PAIR_DIR at a different real, already-FS-minimized '
            'pair if this one has moved.'
        )

MP0B2_MODEL_ASE = '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mp-0b2-medium.model'
if not os.path.exists(MP0B2_MODEL_ASE):
    print(f'WARNING: {MP0B2_MODEL_ASE} not found on this machine -- if this '
          'is the cluster login node, the cpu_mp0b2/gpu_mp0b2 variants will '
          'fail at runtime. Confirm the file still exists before submitting '
          'those two (gpu_mh1 does not depend on it).')

TEST_DIR = os.path.join(WORK_DIR, 'gpu_neb_validation_test')

GPU_SHORT = dict(SLURM_DEFAULTS, partition='gpu-short', time='02:00:00', cpus_per_task=8)
CPU_SHORT = dict(SLURM_DEFAULTS, partition='short', time='12:00:00',
                  gpu=None, cpus_per_task=18)


def _cutoff_from(slurm_cfg):
    h, m, s = (int(x) for x in slurm_cfg['time'].split(':'))
    total = h * 3600 + m * 60 + s - 300
    return f'{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}'


print(f'Real pair: {PAIR_DIR}')
print('Existing real CPU+mh-1 baseline (corner 1, no new run needed):')
print(f'  {os.path.join(PAIR_DIR, "neb_phase1.log")}')
print()

sbatch_commands = []

# ═══════════════════════════════════════════════════════════════════════════
# Variant: gpu_mh1 -- real production code path, device flipped to GPU
# ═══════════════════════════════════════════════════════════════════════════
_outdir = os.path.join(TEST_DIR, 'gpu_mh1')
_traj_p2 = os.path.join(_outdir, 'neb_phase2.traj')
neb_script = run_neb_pipeline(
    is_file=IS_FILE,
    fs_file=FS_FILE,
    e_is=E_IS,
    mace_model_path=MACE_MODEL_ASE,
    barrier_file=os.path.join(_outdir, 'neb_barrier.txt'),
    path_file=os.path.join(_outdir, 'neb_path.dat'),
    outdir=_outdir,
    fs_log_file=FS_LOG,
    job_name='gpu_mh1',
    n_images=N_REPLICAS,
    spring_const=SPRING_CONST,
    neb_ftol=NEB_FTOL,
    z_freeze_cutoff=Z_CUTOFF,
    device='cuda',
    parallel=False,
    label_is='IS:s_145 [gpu_mh1 validation]',
    label_fs='FS:s_73+s_144 [gpu_mh1 validation]',
    traj_phase1=os.path.join(_outdir, 'neb_phase1.traj'),
    traj_phase2=_traj_p2,
    masses=MASSES_7,
    e2t=E2T_7,
)
_sh = os.path.join(_outdir, 'slurm_gpu_mh1.sh')
write_chained_slurm_job(
    job_name='gpu_mh1_validation',
    slurm_config=GPU_SHORT,
    out_path=_sh,
    first_commands=[f'python {neb_script}'],
    restart_commands=[f'python {neb_script}'],
    restart_glob=_traj_p2,
    cutoff=_cutoff_from(GPU_SHORT),
    work_dir=WORK_DIR,
)
sbatch_commands.append(('gpu_mh1', 'device=cuda  model=mh-1 (current, real code path)', _sh))
print(f'[gpu_mh1] real production run_neb_pipeline(), device=cuda, model=mh-1')
print(f'  run script  : {neb_script}')
print(f'  slurm script: {_sh}')
print()

# ═══════════════════════════════════════════════════════════════════════════
# Old-model driver template -- self-contained, no shared-file changes
# ═══════════════════════════════════════════════════════════════════════════
_OLD_MODEL_TEMPLATE = '''#!/usr/bin/env python3
# Validation-only NEB driver: {variant} (model=mace-mp-0b2-medium, device={device})
# Calls models.ase_neb's tested building blocks directly (build_neb_images,
# run_cineb, extract_mep, write_neb_results) -- the exact pattern documented
# in that module's own "Typical usage" docstring. Not a copy of production
# logic, and does not modify models/ase_neb.py.
#
# Simplification vs production: no trajectory-checkpoint restart -- if this
# leg times out and self-resubmits, it restarts NEB from scratch (fresh IDPP
# interpolation) rather than resuming mid-optimisation. Fine for comparative
# early-behaviour data (seconds/step, starting fmax); not a converged
# production barrier.
import os
import sys

sys.path.insert(0, {parent!r})

_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in _ld.split(":") if "stubs" not in p)

MACE_MODEL      = {model_path!r}
IS_FILE         = {is_file!r}
FS_FILE         = {fs_file!r}
E_IS            = {e_is!r}
Z_FREEZE_CUTOFF = {z_cutoff!r}
N_IMAGES        = {n_images!r}
SPRING_CONST    = {spring_const!r}
NEB_FTOL        = {neb_ftol!r}
DEVICE          = {device!r}
BARRIER_FILE    = {barrier_file!r}
PATH_FILE       = {path_file!r}
LOG_PHASE1      = {log_phase1!r}
LOG_PHASE2      = {log_phase2!r}
FS_LOG_FILE     = {fs_log_file!r}


def _parse_pe_final(log_path):
    val = None
    with open(log_path) as f:
        for line in f:
            if 'pe_final_eV' in line and ':' in line:
                try:
                    val = float(line.split(':')[1].strip())
                except ValueError:
                    pass
    if val is None:
        raise ValueError(f'pe_final_eV not found in {{log_path}}')
    return val


E_FS = _parse_pe_final(FS_LOG_FILE)

from models.ase_neb import build_neb_images, run_cineb, extract_mep, write_neb_results
from mace.calculators import MACECalculator


def make_frozen_calc_no_head(model_path, z_freeze_cutoff, device='cpu', dtype='float32'):
    """Same frozen-force pattern as models.ase_neb.make_frozen_calc, but
    deliberately omits the 'head' kwarg entirely -- mace-mp-0b2-medium is a
    single-head foundation model, unlike mace-mh-1/omat_pbe."""
    class MACEFrozenCalc(MACECalculator):
        def get_forces(self, atoms=None):
            forces = super().get_forces(atoms).copy()
            a = self.atoms if atoms is None else atoms
            forces[a.get_positions()[:, 2] < z_freeze_cutoff] = 0.0
            return forces
    return MACEFrozenCalc(model_paths=model_path, device=device, default_dtype=dtype)


print(f'Device: {{DEVICE}}')
print(f'Model : {{MACE_MODEL}}  (no head kwarg)')
print(f'E_IS = {{E_IS:.6f}} eV')
print(f'E_FS = {{E_FS:.6f}} eV')
sys.stdout.flush()

images = build_neb_images(
    IS_FILE, FS_FILE, n_images=N_IMAGES, e_is=E_IS, e_fs=E_FS, interpolation='idpp')

neb, converged, fmax_final = run_cineb(
    images,
    calc_fn=lambda: make_frozen_calc_no_head(MACE_MODEL, Z_FREEZE_CUTOFF, device=DEVICE),
    spring_const=SPRING_CONST,
    neb_ftol=NEB_FTOL,
    logfile_phase1=LOG_PHASE1,
    logfile_phase2=LOG_PHASE2,
)

mep = extract_mep(images, E_IS=E_IS, E_FS=E_FS)
write_neb_results(
    mep, fmax_final, converged, BARRIER_FILE, PATH_FILE,
    label_is='IS:s_145 [{variant} validation]',
    label_fs='FS:s_73+s_144 [{variant} validation]',
)
print(f'fmax_final={{fmax_final:.4f}}  converged={{converged}}')
'''


def _write_old_model_variant(variant, device, slurm_cfg):
    outdir = os.path.join(TEST_DIR, variant)
    os.makedirs(outdir, exist_ok=True)
    run_py = os.path.join(outdir, f'run_{variant}.py')
    content = _OLD_MODEL_TEMPLATE.format(
        variant=variant,
        device=device,
        parent=os.path.dirname(WORK_DIR),
        model_path=MP0B2_MODEL_ASE,
        is_file=IS_FILE,
        fs_file=FS_FILE,
        fs_log_file=FS_LOG,
        e_is=E_IS,
        z_cutoff=Z_CUTOFF,
        n_images=N_REPLICAS,
        spring_const=SPRING_CONST,
        neb_ftol=NEB_FTOL,
        barrier_file=os.path.join(outdir, 'neb_barrier.txt'),
        path_file=os.path.join(outdir, 'neb_path.dat'),
        log_phase1=os.path.join(outdir, 'neb_phase1.log'),
        log_phase2=os.path.join(outdir, 'neb_phase2.log'),
    )
    with open(run_py, 'w') as f:
        f.write(content)

    sh = os.path.join(outdir, f'slurm_{variant}.sh')
    write_chained_slurm_job(
        job_name=f'{variant}_validation',
        slurm_config=slurm_cfg,
        out_path=sh,
        first_commands=[f'python {run_py}'],
        restart_commands=[f'python {run_py}'],
        # No real checkpoint to detect for these two (see module docstring) --
        # point restart_glob at a file that will never exist, so every leg
        # (including any resubmit) always takes the "no restart file found"
        # branch and reruns first_commands from scratch.
        restart_glob=os.path.join(outdir, 'NEVER_EXISTS.restart'),
        cutoff=_cutoff_from(slurm_cfg),
        work_dir=WORK_DIR,
    )
    return run_py, sh


# ═══════════════════════════════════════════════════════════════════════════
# Variant: cpu_mp0b2 -- same device as the existing baseline, old model
# ═══════════════════════════════════════════════════════════════════════════
run_py, sh = _write_old_model_variant('cpu_mp0b2', 'cpu', CPU_SHORT)
sbatch_commands.append(('cpu_mp0b2', 'device=cpu   model=mp-0b2-medium (old)', sh))
print(f'[cpu_mp0b2] device=cpu, model=mp-0b2-medium (old)')
print(f'  run script  : {run_py}')
print(f'  slurm script: {sh}')
print()

# ═══════════════════════════════════════════════════════════════════════════
# Variant: gpu_mp0b2 -- both swapped, closest to the historical setup
# ═══════════════════════════════════════════════════════════════════════════
run_py, sh = _write_old_model_variant('gpu_mp0b2', 'cuda', GPU_SHORT)
sbatch_commands.append(('gpu_mp0b2', 'device=cuda  model=mp-0b2-medium (old)', sh))
print(f'[gpu_mp0b2] device=cuda, model=mp-0b2-medium (old)')
print(f'  run script  : {run_py}')
print(f'  slurm script: {sh}')
print()

print('=' * 70)
print('Nothing submitted. Submit whichever variants you want to compare:')
print('=' * 70)
for name, desc, sh in sbatch_commands:
    print(f'  {name:12s} {desc}')
    print(f'    sbatch {sh}')
print()
print('Once at least one is running, check progress anytime with:')
print('  python calculation/gpu_neb_validation_report.py')

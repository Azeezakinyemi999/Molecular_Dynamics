"""
models/config.py
================
Project-wide constants for Hastelloy-N hydrogen diffusion MD simulations.

All LAMMPS runs use the MACE-MH-1 ML potential on a single A100 GPU via
SLURM (partition: multigpu).  Import what you need and override only the
job-specific values (partition, time) in each notebook.

Usage
-----
    from models.config import (
        LAMMPS_CMD, MACE_MODEL_LAMMPS, KOKKOS_FLAGS,
        E2T_7, MASSES_7, ELEM_STR_7,
        SLURM_DEFAULTS,
    )
    slurm_cfg = {**SLURM_DEFAULTS, 'partition': 'multigpu', 'time': '06:00:00'}
"""

# =============================================================================
# Section 1 — Hardware paths
# =============================================================================

LAMMPS_CMD = (
    '/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp'
)

MACE_MODEL_LAMMPS = (
    '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mh-1.model-mliap_lammps.pt'
)
MACE_MODEL_ASE = (
    '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mh-1.model'
)
MACE_HEAD = 'omat_pbe'

BASE_DIR = '/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel'

# =============================================================================
# Section 2 — LAMMPS runtime settings
# =============================================================================

KOKKOS_FLAGS = [
    '-k', 'on', 'g', '1',
    '-sf', 'kk',
    '-pk', 'kokkos', 'newton', 'on', 'neigh', 'half',
]

PAIR_STYLE  = 'mliap unified'
PAIR_SUFFIX = '0'

# =============================================================================
# Section 3 — Element / mass tables
# =============================================================================
# _MASSES_ALL is the single source of truth for all 9 element types.
# Slices E2T_7 / E2T_10 are derived from it — do not edit them independently.
#
# Format:  type_id -> (atomic_mass_amu, element_symbol)

_MASSES_ALL = {
    1: (26.9815, 'Al'),
    2: (10.8110, 'B'),
    3: (12.0110, 'C'),
    4: (51.9961, 'Cr'),
    5: (55.8450, 'Fe'),
    6: (95.9600, 'Mo'),
    7: (58.6934, 'Ni'),
    8: (15.9990, 'O'),
    9: ( 1.0080, 'H'),
}

# --- Variant 7: 8-type system (no O) — surface, adsorption, NEB notebooks ---
# Al B C Cr Fe Mo Ni H
E2T_7 = {'Al': 1, 'B': 2, 'C': 3, 'Cr': 4,
          'Fe': 5, 'Mo': 6, 'Ni': 7, 'H': 8}
T2E_7 = {v: k for k, v in E2T_7.items()}
# Remap H from type 9 → type 8 in the mass table
MASSES_7 = {
    1: _MASSES_ALL[1],   # Al
    2: _MASSES_ALL[2],   # B
    3: _MASSES_ALL[3],   # C
    4: _MASSES_ALL[4],   # Cr
    5: _MASSES_ALL[5],   # Fe
    6: _MASSES_ALL[6],   # Mo
    7: _MASSES_ALL[7],   # Ni
    8: _MASSES_ALL[9],   # H  (type 8 in this variant)
}
ELEM_STR_7 = 'Al B C Cr Fe Mo Ni H'

# --- Variant 10: 9-type system (with O) — equilibration + oxide notebooks ---
# Al B C Cr Fe Mo Ni O H
E2T_10 = {'Al': 1, 'B': 2, 'C': 3, 'Cr': 4,
           'Fe': 5, 'Mo': 6, 'Ni': 7, 'O': 8, 'H': 9}
T2E_10 = {v: k for k, v in E2T_10.items()}
MASSES_10 = {i: _MASSES_ALL[i] for i in range(1, 10)}  # types 1-9, O at 8
ELEM_STR_10 = 'Al B C Cr Fe Mo Ni O H'

# =============================================================================
# Section 4 — SLURM defaults
# =============================================================================
# Keys NOT included here (set per-notebook): 'partition', 'time'

_LD_PATHS = [
    '/shared/EL9/explorer/cuda/12.3.0/lib64',
    '/shared/EL9/explorer/cuda/12.3.0/lib64/stubs',
    '/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap',
    '/home/akinyemi.az/miniforge3/envs/mace-lammps/lib',
]

SLURM_DEFAULTS = {
    'ntasks':        1,
    'cpus_per_task': 8,
    'gpu':           'a100:1',
    'conda_env':     'mace-lammps',
    'cuda_version':  '12.3.0',
    'openmpi_ver':   '4.1.6',
    'ld_paths':      _LD_PATHS,
}

# =============================================================================
# Section 5 — Simulation defaults
# =============================================================================

# --- Surface / slab ---
TIMESTEP        = 0.0005   # ps
Z_FREEZE_CUTOFF = 22.115   # Å  — bottom-layer freeze threshold

# --- Adsorption ---
H2_HEIGHT  = 2.5    # Å   — initial H₂ height above surface
H2_BOND    = 0.741  # Å   — experimental H₂ bond length
FTOL       = 1e-6   # eV/Å — force convergence tolerance (minimisation)

# --- NEB ---
N_REPLICAS   = 18    # intermediate images
SPRING_CONST = 1.0   # eV/Å²
NEB_FTOL     = 0.05  # eV/Å — NEB force convergence

# --- Bulk CG minimisation (Phase 1a / Phase 1b bulk+H) ---
MIN_ETOL    = 0.0       # energy tolerance (eV)
MIN_FTOL    = 1e-8      # force tolerance (eV/Å)
MIN_MAXITER = 50000     # CG iterations cap
MIN_MAXEVAL = 500000    # force evaluation cap

# --- NPT lattice equilibration (Phase 1b) ---
NPT_HEAT_STEPS = 20000    # velocity-ramp heating phase
NPT_PROD_STEPS = 200000   # constant-T/P production
NPT_BARO_DAMP  = 1.0      # ps — barostat coupling
NPT_DUMP_EVERY = 100      # box-dimension output frequency (steps)

# --- Surface relaxation (NEB workflow Phase A) ---
SURF_ETOL        = 0.0
SURF_FTOL        = 1e-6     # force tolerance (eV/Å)
SURF_MAXITER     = 10000    # CG iterations cap
SURF_MAXEVAL     = 100000   # force evaluation cap
SURF_HEAT_STEPS  = 10000    # velocity-rescale heating phase
SURF_NVT_STEPS   = 100000   # NVT equilibration
SURF_THERMO_DAMP = 0.05     # ps — Nosé-Hoover thermostat coupling

# --- Adsorbate CG minimisation (H2* and H*, NEB workflow) ---
ADS_MIN_ETOL    = 0.0
ADS_MIN_FTOL    = 1e-6      # force tolerance (eV/Å)
ADS_MIN_MAXITER = 10000
ADS_MIN_MAXEVAL = 100000

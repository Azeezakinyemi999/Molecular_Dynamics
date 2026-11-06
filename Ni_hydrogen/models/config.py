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
        SLURM_DEFAULTS,
    )
    slurm_cfg = {**SLURM_DEFAULTS, 'partition': 'multigpu', 'time': '06:00:00'}

There is no fixed built-in element table. Build one with `build_element_tables`
for a synthetic/known list, or -- for anything read from a structure file --
with `build_element_tables_from_structure` (models/structure.py), which
preserves whatever type IDs the file's own Masses section already uses:

    from models.config import build_element_tables
    e2t, t2e, masses, elem_str = build_element_tables(
        ['Fe', 'Mn', 'Ni', 'Ti', 'Zr', 'H']
    )
"""

# =============================================================================
# ►►► CONFIGURATION — EDIT THIS BLOCK FIRST ◄◄◄
# =============================================================================
#
# Everything in this block is specific to one machine and one account. Nothing
# below this block is: Sections 2-5 are LAMMPS settings and physical constants
# that are the same wherever you run.
#
# None of these values fails loudly when wrong. A BASE_DIR pointing at a
# directory that does not exist produces a FileNotFoundError deep inside a
# generator, long after you thought setup was done; a wrong MACE_MODEL path
# surfaces only once a job has already been queued and started. Check them
# before the first run:
#
#     python -c "from models.config import check_configuration as c; c()"
#
# Section 1 — Hardware paths
# -----------------------------------------------------------------------------

# The LAMMPS executable, built with the MLIAP package and Kokkos GPU support.
LAMMPS_CMD = (
    '/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp'
)

# The MACE potential, in two forms. These are data files you supply; they do
# not come from pip. LAMMPS needs the compiled .pt, ASE needs the .model.
MACE_MODEL_LAMMPS = (
    '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mh-1.model-mliap_lammps.pt'
)
MACE_MODEL_ASE = (
    '/projects/westgroup/akinyemi.az/mace_lammps/models/mace-mh-1.model'
)
# Which head of a multi-head MACE model to evaluate. Wrong head = wrong
# energies, silently -- the run completes and the numbers look plausible.
MACE_HEAD = 'omat_pbe'

# The repository root: the directory containing calculation/ and models/.
# Every generated script embeds absolute paths derived from this, so it must
# be correct on the machine where the JOBS run, not where you edit.
BASE_DIR = '/projects/westgroup/akinyemi.az/mace_lammps/Ni_hydrogen'

# The SLURM account/environment settings are further down, in Section 4
# (SLURM_DEFAULTS and _LD_PATHS) -- they are equally machine-specific, and are
# left beside the other SLURM values rather than duplicated here.
#
# ►►► END OF CONFIGURATION ◄◄◄
# =============================================================================

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
# _MASS_OVERRIDES pins the exact mass values this project has always used
# for its original 9 elements, so existing LAMMPS data files and validated
# runs never see so much as a trailing-digit change. Any element NOT in
# this dict is looked up from ASE's atomic data instead -- MACE-MH-1
# supports the full periodic table up to Pu, and ASE is already a hard
# dependency elsewhere in this project (MACE_MODEL_ASE).
#
# Format:  element_symbol -> atomic_mass_amu

_MASS_OVERRIDES = {
    'Al': 26.9815,
    'B':  10.8110,
    'C':  12.0110,
    'Cr': 51.9961,
    'Fe': 55.8450,
    'Mo': 95.9600,
    'Ni': 58.6934,
    'O':  15.9990,
    'H':   1.0080,
}


def build_element_tables(elements):
    """Build (E2T, T2E, MASSES, ELEM_STR) for an ordered list of elements.

    Type IDs are assigned by position, starting at 1 -- order matters,
    since it fixes the LAMMPS atom type IDs embedded in every generated
    data file, pair_coeff line, and already-written run script for a given
    system. Reordering `elements` for a system that already has output
    changes its type IDs and breaks compatibility with that output.

    Mass values come from `_MASS_OVERRIDES` where available (the original
    9 elements, byte-identical to this project's historical values) and
    from ASE's atomic data otherwise -- ASE is only imported if a requested
    element actually needs that fallback, so building the two variants
    below does not add an ASE dependency to this module.

    Parameters
    ----------
    elements : list of str
        Element symbols in the desired type-ID order, e.g.
        ``['Al', 'B', 'C', 'Cr', 'Fe', 'Mo', 'Ni', 'H']``.

    Returns
    -------
    E2T : dict
        ``{element_symbol: atom_type}``.
    T2E : dict
        ``{atom_type: element_symbol}``.
    MASSES : dict
        ``{atom_type: (mass_amu, element_symbol)}``.
    ELEM_STR : str
        Space-separated element symbols in type-ID order.
    """
    E2T = {sym: i for i, sym in enumerate(elements, start=1)}
    T2E = {t: sym for sym, t in E2T.items()}
    MASSES = {}
    for sym, t in E2T.items():
        if sym in _MASS_OVERRIDES:
            mass = _MASS_OVERRIDES[sym]
        else:
            from ase.data import atomic_masses, atomic_numbers
            if sym not in atomic_numbers:
                raise KeyError(
                    f"Unknown element symbol '{sym}' -- not in "
                    "_MASS_OVERRIDES and not recognized by ASE's atomic data."
                )
            mass = atomic_masses[atomic_numbers[sym]]
        MASSES[t] = (mass, sym)
    ELEM_STR = ' '.join(elements)
    return E2T, T2E, MASSES, ELEM_STR


# No fixed built-in table (the old 8-type ELEM_STR_7 / 9-type ELEM_STR_10
# variants, Al-B-C-Cr-Fe-Mo-Ni(-O)-H, were removed -- every call site now
# gets its element table from the structure it's actually working with, via
# build_element_tables_from_structure (models/structure.py) for anything
# read from a file, or a direct build_element_tables([...]) call for a
# synthetic system that has no structure file, e.g. the isolated-H2 gas
# reference. Verified empirically (calculation/tools/verify_padded_species_mace.py)
# that padding a pair_coeff with species unused by any atom in the box has
# no effect on MACE-MH-1's computed energy/forces, so a minimal table is
# always safe to use in place of a historical, often-irrelevant larger one.

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
    'conda_env':     '/home/akinyemi.az/miniforge3/envs/mace-lammps',
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
N_REPLICAS   = 9     # intermediate images
SPRING_CONST = 1.0   # eV/Å²
NEB_FTOL     = 0.1   # eV/Å — NEB force convergence

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

# =============================================================================
# Section 6 — SLURM partition submission limits
# =============================================================================
# (queue_max, concurrent) per partition, consumed by auto_submit() via
# partition_submit_limits() in models/create_slurm.py. queue_max = max
# pending+running jobs the partition/account allows at once; concurrent =
# max simultaneously RUNNING (the %N throttle on --array).
PARTITION_SUBMIT_LIMITS = {
    'short':     (1000, 50),
    'multigpu':  (8,    4),
    'gpu':       (8,    4),
    'sharing':   (4,    2),
    'gpu-short': (4,    2),
}


# =============================================================================
# Section 7 — Configuration check
# =============================================================================

def check_configuration(verbose: bool = True) -> list:
    """Verify the machine-specific paths at the top of this file.

    Returns the list of problems found, empty when everything resolves. Import
    of this module never checks anything -- a missing LAMMPS binary is not an
    error for the many code paths that only read results -- so this is a
    deliberate, explicit call:

        python -c "from models.config import check_configuration as c; c()"

    Run it before the first job. Every value it checks is one that otherwise
    surfaces late: BASE_DIR as a FileNotFoundError inside a generator, the
    MACE paths only once a queued job has started and burned its startup time.
    """
    import os

    problems = []
    checks = [
        ('BASE_DIR',          BASE_DIR,          'dir',  'repository root (contains calculation/ and models/)'),
        ('LAMMPS_CMD',        LAMMPS_CMD,        'exec', 'LAMMPS binary with MLIAP + Kokkos'),
        ('MACE_MODEL_ASE',    MACE_MODEL_ASE,    'file', 'MACE model for ASE (NEB, vibrations)'),
        ('MACE_MODEL_LAMMPS', MACE_MODEL_LAMMPS, 'file', 'compiled MACE model for LAMMPS'),
    ]
    for name, value, kind, what in checks:
        if kind == 'dir':
            ok = os.path.isdir(value)
        elif kind == 'exec':
            ok = os.path.isfile(value) and os.access(value, os.X_OK)
        else:
            ok = os.path.isfile(value)
        if not ok:
            problems.append(f'{name}: {value}  — not found ({what})')
        if verbose:
            print(f'  [{"ok" if ok else "MISSING":>7}] {name:<18} {value}')

    # BASE_DIR must be the root that holds both trees, not one of them.
    if os.path.isdir(BASE_DIR):
        for sub in ('models', 'calculation'):
            if not os.path.isdir(os.path.join(BASE_DIR, sub)):
                problems.append(
                    f'BASE_DIR: {BASE_DIR} has no {sub}/ — it should be the '
                    f'directory CONTAINING models/ and calculation/')

    env = SLURM_DEFAULTS.get('conda_env')
    if env and not os.path.isdir(env):
        problems.append(f"SLURM_DEFAULTS['conda_env']: {env} — not found "
                        f"(the environment SLURM jobs activate)")
    if verbose:
        print(f'  [{"ok" if (env and os.path.isdir(env)) else "MISSING":>7}] '
              f'{"conda_env":<18} {env}')
        print()
        if problems:
            print(f'  {len(problems)} problem(s) — edit the CONFIGURATION '
                  f'block at the top of models/config.py:')
            for p in problems:
                print(f'    - {p}')
        else:
            print('  Configuration OK.')
    return problems

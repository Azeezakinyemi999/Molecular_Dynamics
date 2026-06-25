# -----------------------SECTION A: Slab Construction & Surface Relaxation -----------------------
"""
Section A: Slab Construction & Surface Relaxation
==================================================
Prepares the FCC(111) Hastelloy N surface for adsorption and NEB calculations.

Phases:
    1. Slab Construction (NB03)
       - Load minimized bulk from NB02.
       - Build FCC(111) slab: 5×6 lateral tiling, 12 layers, 15 Å vacuum.
       - Randomly substitute Ni → Hastelloy N composition (seed=7).
       - Write LAMMPS data file for the initial slab.

    2. Surface Relaxation (NB04)
       - CG minimization with bottom 4 layers frozen (ftol=1e-6).
       - Thermal anneal from 10 K → 300 K over 10 ps (velocity rescaling).
       - NVT equilibration at 300 K for 100 ps (Nosé-Hoover thermostat).
       - Submit via SLURM; output is the relaxed slab LAMMPS file + log.

    3. ACAT Site Enumeration (NB04b)
       - Build surface connectivity graph from relaxed slab.
       - Identify all high-symmetry hollow/bridge/atop adsorption sites (171 total).
       - Assign site_id (0–170) and true_label (L1 or L2 layer).
       - Compute Level-1 and Level-2 neighbor chemical environments.
       - Save full inventory to surface_sites.json.

    4. Orchestrator
       - Single entry point that chains Phases 1 → 2 → 3.
       - Supports dry_run mode: generates all scripts without SLURM submission.
"""


# -----------------------SECTION A: Phase 1: Slab construction -----------------------

from __future__ import annotations

import json
from pathlib import Path

from models.config import E2T_7, MASSES_7
from models.structure import build_slab


def build_phase1_slab(
    bulk_min_path: str,
    out_path: str,
    miller: tuple[int, int, int] = (1, 1, 1),
    layers: int = 12,
    vacuum: float = 15.0,
    lateral_repeat: tuple[int, int] = (5, 6),
    supercell_reps: tuple[int, int, int] = (5, 5, 5),
    seed: int = 7,
    masses: dict = MASSES_7,
    e2t: dict = E2T_7,
) -> tuple[str, float]:
    """
    Section A Phase 1: Build the FCC(111) Hastelloy N slab from minimized bulk.

    This follows NB03:
        1. Load minimized bulk structure NB02.
        2. Build FCC(111) slab with 5×6 lateral tiling, 12 layers, and 15 Å vacuum.
        3. Randomly substitute Ni → Hastelloy N composition using a reproducible seed (seed=7).
        4. Initial magnetic moments are handled in Phase 2 via LAMMPS input script.
        5. Return the generated LAMMPS slab file path and lattice parameter a0.

    Parameters
    ----------
    bulk_min_path : str
        Path to minimized bulk LAMMPS file (from NB02).
    out_path : str
        Destination path for slab LAMMPS file.
    miller : tuple of int
        Miller indices. Default (1,1,1) for FCC(111).
    layers : int
        Number of atomic layers. Default 12.
    vacuum : float
        Vacuum thickness in Å. Default 15.0.
    lateral_repeat : tuple of int
        (p, q) lateral tiling. Default (5, 6).
    supercell_reps : tuple of int
        Bulk supercell repetitions. Default (5, 5, 5).
    seed : int
        Random seed for composition assignment. Default 7 (matches NB03).
    masses : dict
        Atom type → (mass, element) mapping. Default MASSES_7.
    e2t : dict
        Element → type mapping. Default E2T_7.

    Returns
    -------
    tuple of (str, float)
        (slab_path, a0) — absolute path to slab file and lattice parameter in Å.
    """
    slab_path, a0 = build_slab(
        bulk_min_path=bulk_min_path,
        miller=miller,
        layers=layers,
        vacuum=vacuum,
        masses=masses,
        e2t=e2t,
        out_path=str(out_path),
        supercell_reps=supercell_reps,
        lateral_repeat=lateral_repeat,
        seed=seed,
    )
    print(f"[Section A Phase 1] Built slab: {slab_path}")
    print(f"  a0 = {a0:.6f} Å")
    print(f"  Note: Initial magnetic moments (Ni/Co ferromagnetic) are set in Phase 2 LAMMPS script.")
    return slab_path, a0


# -----------------------SECTION A: Phase 2: Surface Relaxation  -----------------------

from models.lammps_script import write_surface_relaxation_script, write_surface_relaxation_restart_script
from models.create_slurm import write_slurm_job, write_chained_slurm_job, submit_slurm_job, wait_for_jobs


def run_phase2_surface_relaxation(
    slab_path: str,
    outdir: str,
    a0: float,
    timestep: float = 0.0005,
    z_freeze_cutoff: float = 22.115,
    slurm_opts: dict | None = None,
    restart_every: int = 10000,
    dry_run: bool = True,
) -> dict:
    """
    Section A Phase 2: Relax the slab surface via three-phase LAMMPS MD.

    This follows NB04:
        Phase 1: CG minimization (ftol=1e-6, freeze bottom 4 layers)
        Phase 2: Thermal anneal 10K → 300K over 10 ps (velocity rescaling)
        Phase 3: NVT equilibration at 300K for 100 ps (Nosé-Hoover, frozen bottom)

    Outputs per run:
        - {outdir}/relaxed_slab.lammps           — final relaxed structure
        - {outdir}/relaxed_slab_phase1_min.lammps — post-CG checkpoint
        - {outdir}/relaxed_slab_phase2_heat.lammps — post-heat checkpoint
        - {outdir}/relaxed_slab_phase*.restart    — binary restart files
        - {outdir}/relaxed_slab_nvt_traj.lammpstrj — NVT trajectory dump
        - {outdir}/relax_thermo.txt               — NVT temperature time-series
        - {outdir}/surface_relax.log              — LAMMPS log

    Parameters
    ----------
    slab_path : str
        Path to initial slab structure (LAMMPS data file from Phase 1).
    outdir : str
        Output directory for relaxation results.
    a0 : float
        Lattice parameter (Å) — carried forward to summary, not used internally.
    timestep : float
        MD timestep in ps. Default 0.0005.
    z_freeze_cutoff : float
        Z-coordinate threshold for freezing bottom layers (Å). Default 22.115.
    slurm_opts : dict, optional
        SLURM configuration. If None, uses SLURM_DEFAULTS with
        partition='multigpu', time='24:00:00'.
    restart_every : int
        Steps between periodic mid-run restart writes. Default 10000.
    dry_run : bool
        If True, generate scripts but do not submit. Default True.

    Returns
    -------
    dict
        {
            'slab_path'   : input slab file,
            'relaxed_slab': path to relaxed structure,
            'traj_file'   : NVT trajectory path,
            'log_file'    : LAMMPS log path,
            'job_id'      : SLURM job ID (if submitted, else None),
            'outdir'      : output directory,
            'a0'          : lattice parameter (Å),
            'status'      : 'submitted' or 'generated',
        }
    """
    from models.config import (
        SLURM_DEFAULTS, LAMMPS_CMD, MACE_MODEL_LAMMPS,
        PAIR_STYLE, PAIR_SUFFIX, ELEM_STR_7, KOKKOS_FLAGS,
        SURF_ETOL, SURF_FTOL, SURF_MAXITER, SURF_MAXEVAL,
        SURF_HEAT_STEPS, SURF_NVT_STEPS, SURF_THERMO_DAMP,
    )

    Path(outdir).mkdir(parents=True, exist_ok=True)

    if slurm_opts is None:
        slurm_opts = {**SLURM_DEFAULTS, 'partition': 'multigpu', 'time': '24:00:00'}

    # Derive all output paths up front
    relaxed_slab  = str(Path(outdir) / 'relaxed_slab.lammps')
    relax_thermo  = str(Path(outdir) / 'relax_thermo.txt')
    restart_dir   = str(Path(outdir) / 'restarts')
    lammps_in     = str(Path(outdir) / 'surface_relax.in')
    log_path      = str(Path(outdir) / 'surface_relax.log')
    slurm_script  = str(Path(outdir) / 'relax_slurm.sh')

    Path(restart_dir).mkdir(parents=True, exist_ok=True)

    write_surface_relaxation_script(
        slab_input=slab_path,
        slab_relaxed=relaxed_slab,
        relax_thermo=relax_thermo,
        out_path=lammps_in,
        pair_style=PAIR_STYLE,
        mace_model=MACE_MODEL_LAMMPS,
        pair_suffix=PAIR_SUFFIX,
        elem_str=ELEM_STR_7,
        z_freeze_cutoff=z_freeze_cutoff,
        timestep=timestep,
        thermo_damp=SURF_THERMO_DAMP,
        etol=SURF_ETOL,
        ftol=SURF_FTOL,
        maxiter=SURF_MAXITER,
        maxeval=SURF_MAXEVAL,
        heat_steps=SURF_HEAT_STEPS,
        nvt_steps=SURF_NVT_STEPS,
        restart_dir=restart_dir,
        restart_every=restart_every,
    )

    lammps_rst_in = str(Path(outdir) / 'surface_relax_restart.in')
    rst_glob      = str(Path(restart_dir) / 'surf_300K.*.restart')

    write_surface_relaxation_restart_script(
        restart_file=rst_glob,
        slab_relaxed=relaxed_slab,
        relax_thermo=relax_thermo,
        out_path=lammps_rst_in,
        pair_style=PAIR_STYLE,
        mace_model=MACE_MODEL_LAMMPS,
        pair_suffix=PAIR_SUFFIX,
        elem_str=ELEM_STR_7,
        z_freeze_cutoff=z_freeze_cutoff,
        timestep=timestep,
        thermo_damp=SURF_THERMO_DAMP,
        nvt_steps=SURF_NVT_STEPS,
        restart_dir=restart_dir,
        restart_every=restart_every,
    )

    kk          = ' '.join(KOKKOS_FLAGS)
    _wall       = slurm_opts.get('time', '24:00:00')
    _h, _m, _s  = map(int, _wall.split(':'))
    _cutoff_val = f'{(_h*3600+_m*60+_s-300)//3600:02d}:{((_h*3600+_m*60+_s-300)%3600)//60:02d}:{(_h*3600+_m*60+_s-300)%60:02d}'

    write_chained_slurm_job(
        job_name='SurfaceRelax',
        slurm_config=slurm_opts,
        out_path=slurm_script,
        first_commands=[f'{LAMMPS_CMD} {kk} -in {lammps_in} -log {log_path}'],
        restart_commands=[f'{LAMMPS_CMD} {kk} -in {lammps_rst_in} -log {log_path}'],
        restart_glob=rst_glob,
        cutoff=_cutoff_val,
        work_dir=outdir,
    )

    print(f"[Section A Phase 2] Surface relaxation setup:")
    print(f"  LAMMPS input : {lammps_in}")
    print(f"  SLURM script : {slurm_script}")
    print(f"  Relaxed slab : {relaxed_slab}")
    print(f"  NVT traj     : {Path(outdir)/'relaxed_slab_nvt_traj.lammpstrj'}")
    print(f"  Log          : {log_path}")

    if not dry_run:
        job_id = submit_slurm_job(slurm_script)
        print(f"  Submitted job {job_id}")
        status = 'submitted'
    else:
        print(f"  (dry_run=True, not submitting)")
        job_id = None
        status = 'generated'

    # Traj path matches what write_surface_relaxation_script derives from slab_relaxed stem
    traj_file = str(Path(outdir) / 'relaxed_slab_nvt_traj.lammpstrj')

    return {
        'slab_path'   : slab_path,
        'relaxed_slab': relaxed_slab,
        'traj_file'   : traj_file,
        'log_file'    : log_path,
        'job_id'      : job_id,
        'outdir'      : outdir,
        'a0'          : a0,
        'status'      : status,
    }

# -----------------------SECTION A: Phase 3: ACAT site Enumeration -----------------------

import networkx as nx

from models.surface_graph import build_surface_graph, build_site_environment, save_surface_sites


def run_phase3_site_enumeration(
    relaxed_slab_path: str,
    outdir: str,
    seed: int = 7,
    bond_cutoff: float = 3.2,
) -> tuple[str, int]:
    """
    Section A Phase 3: Enumerate all high-symmetry adsorption sites via ACAT.

    This follows NB04b:
        1. Load relaxed slab from Phase 2.
        2. Build surface graph via connectivity analysis.
        3. Identify high-symmetry hollow/atop sites.
        4. Assign site_id (0–170) and true_label (L1 or L2).
        5. Compute Level-1 and Level-2 neighbor environments.
        6. Save surface_sites.json with 171 entries.

    Parameters
    ----------
    relaxed_slab_path : str
        Path to relaxed slab structure (from Phase 2).
    outdir : str
        Output directory for surface sites JSON.
    seed : int
        Random seed for ACAT enumeration. Default 7.
    bond_cutoff : float
        Bond distance cutoff for connectivity (Å). Default 3.2.

    Returns
    -------
    tuple of (str, int)
        (surface_sites_json_path, n_sites_found) — path to output JSON and site count.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    # Build surface connectivity graph
    print(f"[Section A Phase 3] Building surface graph...")
    G, slab, _top3_slab, _sites = build_surface_graph(
        relaxed_slab_path,
        seed=seed,
        bond_cutoff=bond_cutoff,
    )
    
    # Compute site environments (Level-1 and Level-2 neighbors)
    print(f"[Section A Phase 3] Computing site environments...")
    environments = build_site_environment(G, slab, bond_cutoff=bond_cutoff)
    
    # Save surface sites inventory
    output_json = Path(outdir) / 'surface_sites.json'
    save_surface_sites(G, environments, slab, str(output_json), seed=seed)

    # Derive site count from saved JSON (avoids counting atom nodes in G)
    with open(output_json) as f:
        _saved = json.load(f)
    n_sites = len(_saved['sites'])

    # Save graph for reuse / debugging
    graph_path = Path(outdir) / 'surface_graph.gml'
    nx.write_gml(G, str(graph_path))

    print(f"[Section A Phase 3] Enumerated {n_sites} adsorption sites")
    print(f"  surface_sites.json : {output_json}")
    print(f"  surface_graph.gml  : {graph_path}")

    return str(output_json), n_sites

# -----------------------SECTION A: Phase 4: SECTION A Orchestrator -----------------------


def orchestrate_slab_prep(
    bulk_min_path: str,
    outdir: str = 'calculation/slabs',
    miller: tuple[int, int, int] = (1, 1, 1),
    layers: int = 12,
    vacuum: float = 15.0,
    lateral_repeat: tuple[int, int] = (5, 6),
    timestep: float = 0.0005,
    z_freeze_cutoff: float = 22.115,
    slurm_opts: dict | None = None,
    dry_run: bool = True,
) -> dict:
    """
    Section A Full Pipeline: Build slab → Relax surface → Enumerate sites.

    Chains Phase 1 (slab construction), Phase 2 (surface relaxation), and
    Phase 3 (ACAT site enumeration) into a single orchestrator.

    Parameters
    ----------
    bulk_min_path : str
        Path to minimized bulk structure (NB02 output).
    outdir : str
        Base output directory. Default 'calculation/slabs'.
    miller : tuple of int
        Miller indices. Default (1,1,1).
    layers : int
        Number of slab layers. Default 12.
    vacuum : float
        Vacuum thickness (Å). Default 15.0.
    lateral_repeat : tuple of int
        Lateral tiling (p, q). Default (5, 6).
    timestep : float
        MD timestep (ps). Default 0.0005.
    z_freeze_cutoff : float
        Z-freeze cutoff (Å). Default 22.115.
    slurm_opts : dict, optional
        SLURM configuration. If None, uses defaults.
    dry_run : bool
        If True, generate scripts without submission. Default True.

    Returns
    -------
    dict
        Summary with keys: {
            'phase1_slab': path to initial slab,
            'phase2_relaxed': path to relaxed slab,
            'phase3_sites': path to surface_sites.json,
            'n_sites': number of enumerated sites,
            'outdir': output directory,
            'status': overall completion status,
        }
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    hkl_str = ''.join(map(str, miller))
    phase1_dir = Path(outdir) / 'phase1_slab'
    phase2_dir = Path(outdir) / 'phase2_relax'
    phase3_dir = Path(outdir) / 'phase3_sites'
    
    print(f"\n{'='*80}")
    print(f"SECTION A: Full Slab Prep Pipeline")
    print(f"{'='*80}\n")
    
    # Phase 1: Build slab
    print(f"\n>>> PHASE 1: Slab Construction")
    slab_out = phase1_dir / f'slab_{hkl_str}.lammps'
    slab_path, a0 = build_phase1_slab(
        bulk_min_path=bulk_min_path,
        out_path=str(slab_out),
        miller=miller,
        layers=layers,
        vacuum=vacuum,
        lateral_repeat=lateral_repeat,
        seed=7,
    )
    
    # Phase 2: Relax surface (skip if dry_run, but generate scripts)
    print(f"\n>>> PHASE 2: Surface Relaxation")
    relax_result = run_phase2_surface_relaxation(
        slab_path=slab_path,
        outdir=str(phase2_dir),
        a0=a0,
        timestep=timestep,
        z_freeze_cutoff=z_freeze_cutoff,
        slurm_opts=slurm_opts,
        dry_run=dry_run,
    )
    
    relaxed_slab = relax_result['relaxed_slab']

    # Wait for the Phase 2 SLURM job to finish before reading its output
    if relax_result.get('job_id') is not None:
        wait_for_jobs({'surface_relax': relax_result['job_id']})

    # Phase 3: Enumerate sites (use input slab if relaxation was not run)
    print(f"\n>>> PHASE 3: ACAT Site Enumeration")
    if dry_run and relax_result['status'] == 'generated':
        # For dry-run, we use the input slab (Phase 2 would produce relaxed version)
        print(f"  Note: Using input slab for site enumeration (Phase 2 in dry_run)")
        enum_slab = slab_path
    else:
        enum_slab = relaxed_slab
    
    sites_json, n_sites = run_phase3_site_enumeration(
        relaxed_slab_path=enum_slab,
        outdir=str(phase3_dir),
        seed=7,
        bond_cutoff=3.2,
    )
    
    result = {
        'phase1_slab'   : slab_path,
        'phase2_relaxed': relaxed_slab,
        'phase2_traj'   : relax_result.get('traj_file'),
        'phase2_log'    : relax_result.get('log_file'),
        'phase3_sites'  : sites_json,
        'phase3_graph'  : str(phase3_dir / 'surface_graph.gml'),
        'n_sites'       : n_sites,
        'a0'            : a0,
        'outdir'        : outdir,
        'status'        : relax_result['status'],
    }

    summary_path = Path(outdir) / 'section_a_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SECTION A COMPLETE")
    print(f"{'='*80}")
    print(f"  Phase 1 slab    : {slab_path}")
    print(f"  Phase 2 relaxed : {relaxed_slab}")
    print(f"  Phase 2 traj    : {relax_result.get('traj_file')}")
    print(f"  Phase 3 sites   : {sites_json} ({n_sites} sites)")
    print(f"  Summary         : {summary_path}")
    print(f"  Status          : {relax_result['status']}")

    return result

# -----------------------SECTION B: Adsorption Energy Calculation -----------------------
"""
Section B: Adsorption Energy Calculation
=========================================
Computes H2* (molecular) and H* (atomic) adsorption energies for all 171
surface sites enumerated in Section A. Results feed directly into Section C
(NEB) as the IS and FS pools for H2 dissociation.

Reference energies (e_clean, e_h2_gas) are passed as inputs by the caller,
computed externally from the clean-slab single-point log and a separate
H2 gas minimization.

Phases:
    1. H2* Adsorption Energy (NB05b)
       - Place H2 molecule parallel to surface, 2.5 Å above each of the
         171 site centroids (height 2.5 Å, bond 0.741 Å, parallel to surface).
       - CG-minimize slab+H2 with bottom layers frozen (fixed box, p p f).
       - Check if H2 stays intact (H–H < 1.5 Å) or dissociates spontaneously.
       - E_H2* = E(slab+H2) - E(clean slab) - E(H2 gas); n_h2_ref=1.0.
       - Output IS pool: H2_site_coords.json — per-site status, centroid,
         E_ads, true_label.

    2. H* Adsorption Energy (NB06b)
       - Place single H atom 2.5 Å above each site centroid.
       - CG-minimize; E_H* = E(slab+H) - E(clean slab) - 0.5*E(H2 gas).
       - Filter FS pool to sites with E_ads < 0 (thermodynamically accessible).
       - Output FS pool: H_site_coords.json + per-site h_atom_{sid}_relaxed.lammps.

    3. Orchestrator
       - Chains Phase 1 → Phase 2.
       - Writes consolidated adsorption_summary.json.
"""

import json
import numpy as np
from ase.io import read as ase_read

from models.lammps_script import write_adsorbate_min_script
from models.structure import add_adsorbate
from models.energetics import calc_binding_energy, rank_sites
from models.parsers import parse_energy_log
from models.config import (
    LAMMPS_CMD, MACE_MODEL_LAMMPS, PAIR_STYLE, PAIR_SUFFIX,
    KOKKOS_FLAGS, ELEM_STR_7, MASSES_7, E2T_7,
    H2_HEIGHT, H2_BOND, FTOL, Z_FREEZE_CUTOFF,
    ADS_MIN_ETOL, ADS_MIN_FTOL, ADS_MIN_MAXITER, ADS_MIN_MAXEVAL,
)


def _check_h2_intact(relaxed_path: str, bond_cutoff: float = 1.5) -> bool | None:
    """Return True if H2 is intact (H–H distance < bond_cutoff Å) after relaxation.

    The two H atoms added by add_adsorbate are always the last H atoms in the
    structure.  Returns None if the file does not exist or cannot be parsed.
    """
    if not Path(relaxed_path).exists():
        return None
    try:
        struct = ase_read(relaxed_path, format='lammps-data', style='atomic')
        syms   = np.array(struct.get_chemical_symbols())
        pos    = struct.get_positions()
        h_idx  = np.where(syms == 'H')[0]
        if len(h_idx) < 2:
            return False
        d = float(np.linalg.norm(pos[h_idx[-1]] - pos[h_idx[-2]]))
        return d < bond_cutoff
    except Exception:
        return None


def _get_h_final_position(relaxed_path: str) -> list | None:
    """Return [x, y, z] of the H atom from a relaxed slab+H file.

    The H atom added by add_adsorbate is the last H in the structure.
    Returns None if the file does not exist.
    """
    if not Path(relaxed_path).exists():
        return None
    try:
        struct = ase_read(relaxed_path, format='lammps-data', style='atomic')
        syms   = np.array(struct.get_chemical_symbols())
        pos    = struct.get_positions()
        h_idx  = np.where(syms == 'H')[0]
        if len(h_idx) == 0:
            return None
        return pos[h_idx[-1]].tolist()
    except Exception:
        return None


# -----------------------SECTION B: Phase 1: H2* Adsorption Energy -----------------------

def run_phase1_h2_adsorption(
    surface_sites_json: str,
    relaxed_slab_path: str,
    e_clean: float,
    e_h2_gas: float,
    outdir: str,
    slurm_opts: dict | None = None,
    dry_run: bool = True,
) -> dict:
    """
    Section B Phase 1: H2* adsorption energy for all surface sites (NB05b).

    This follows NB05b:
        1. Load 171 site centroids from surface_sites.json (Section A output).
        2. Place H2 parallel to surface (height 2.5 Å, bond 0.741 Å).
        3. CG-minimize slab+H2 with bottom layers frozen (fixed box).
        4. Check H–H distance: intact if < 1.5 Å, dissociated otherwise.
        5. Compute E_H2* = E(slab+H2) - E(clean slab) - E(H2 gas).
        6. Save IS pool to H2_site_coords.json (status, centroid, E_ads, true_label).

    Parameters
    ----------
    surface_sites_json : str
        Path to surface_sites.json from Section A Phase 3.
    relaxed_slab_path : str
        Path to relaxed slab LAMMPS data file (Section A Phase 2 output).
    e_clean : float
        Potential energy of the clean relaxed slab (eV). Computed externally.
    e_h2_gas : float
        Potential energy of an isolated H2 molecule (eV). Computed externally.
    outdir : str
        Base output directory. Phase 1 outputs go to {outdir}/phase1_h2/.
    slurm_opts : dict, optional
        SLURM configuration. If None, uses SLURM_DEFAULTS with
        partition='multigpu', time='02:00:00'.
    dry_run : bool
        If True, generate all scripts without submitting. Default True.

    Returns
    -------
    dict
        {
          'h2_energies': {site_id: E_H2_eV},
          'ranked_sites': [[site_id, E_H2_eV], ...],
          'is_pool': {site_id: {status, centroid, E_ads, true_label}},
          'n_sites_computed': int,
          'n_sites_total': int,
          'outdir': str,
          'status': 'generated' or 'submitted',
        }
    """
    from models.config import SLURM_DEFAULTS

    if slurm_opts is None:
        slurm_opts = {**SLURM_DEFAULTS, 'partition': 'multigpu', 'time': '02:00:00'}

    phase_dir  = Path(outdir) / 'phase1_h2'
    struct_dir = phase_dir / 'structures'
    script_dir = phase_dir / 'scripts'
    slurm_dir  = phase_dir / 'slurm'
    result_dir = phase_dir / 'results'
    for d in [struct_dir, script_dir, slurm_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(surface_sites_json) as f:
        sites_data = json.load(f)
    sites = sites_data['sites']

    kk = ' '.join(KOKKOS_FLAGS)

    print(f"[Section B Phase 1] H2* adsorption (parallel): {len(sites)} sites")

    for site in sites:
        sid        = site['site_id']
        centroid   = site['level1']['position']    # [x, y, z]
        site_xy    = centroid[:2]

        struct_path  = str(struct_dir / f'slab_h2_{sid}.lammps')
        relaxed_path = str(result_dir  / f'h2_{sid}_relaxed.lammps')
        log_path     = str(result_dir  / f'h2_min_{sid}.log')
        lammps_in    = str(script_dir  / f'h2_min_{sid}.in')
        slurm_sh     = str(slurm_dir   / f'h2_slurm_{sid}.sh')

        add_adsorbate(
            slab_path=relaxed_slab_path,
            site_position=site_xy,
            species='H2',
            masses=MASSES_7,
            e2t=E2T_7,
            out_path=struct_path,
            height=H2_HEIGHT,
            h2_bond=H2_BOND,
            h2_orientation='parallel',
        )

        write_adsorbate_min_script(
            slab_ads_input=struct_path,
            ads_output=relaxed_path,
            out_path=lammps_in,
            pair_style=PAIR_STYLE,
            mace_model=MACE_MODEL_LAMMPS,
            pair_suffix=PAIR_SUFFIX,
            elem_str=ELEM_STR_7,
            z_freeze_cutoff=Z_FREEZE_CUTOFF,
            etol=ADS_MIN_ETOL,
            ftol=ADS_MIN_FTOL,
            maxiter=ADS_MIN_MAXITER,
            maxeval=ADS_MIN_MAXEVAL,
        )

        write_slurm_job(
            job_name=f'H2ads_{sid}',
            slurm_config=slurm_opts,
            out_path=slurm_sh,
            commands=[f'{LAMMPS_CMD} {kk} -in {lammps_in} -log {log_path}'],
        )

        if not dry_run:
            submit_slurm_job(slurm_sh)

    status = 'submitted' if not dry_run else 'generated'
    print(f"  Scripts in {script_dir}/  ({status})")

    # Collect results if logs already exist (post-run)
    h2_energies = {}
    is_pool     = {}

    for site in sites:
        sid        = site['site_id']
        centroid   = site['level1']['position']
        true_label = site['level1']['full_label']

        log_path     = str(result_dir / f'h2_min_{sid}.log')
        relaxed_path = str(result_dir / f'h2_{sid}_relaxed.lammps')

        parsed = parse_energy_log(log_path)
        if parsed and 'pe_final_eV' in parsed:
            e_bind = calc_binding_energy(
                e_with_h=parsed['pe_final_eV'],
                e_clean=e_clean,
                e_h2_gas=e_h2_gas,
                n_h2_ref=1.0,
            )
            h2_energies[sid] = e_bind

            intact = _check_h2_intact(relaxed_path)
            is_pool[sid] = {
                'status'    : 'intact' if intact else 'dissociated' if intact is not None else 'unknown',
                'centroid'  : centroid,
                'E_ads'     : e_bind,
                'true_label': true_label,
            }

    ranked = rank_sites(h2_energies) if h2_energies else []

    # H2_site_coords.json — all computed sites with intact/dissociated/unknown label (audit trail)
    h2_coords_path = phase_dir / 'H2_site_coords.json'
    with open(h2_coords_path, 'w') as f:
        json.dump(is_pool, f, indent=2)

    # H2_intact_pool.json — intact-only subset: actual NEB IS candidates
    intact_pool = {sid: v for sid, v in is_pool.items() if v['status'] == 'intact'}
    intact_pool_path = phase_dir / 'H2_intact_pool.json'
    with open(intact_pool_path, 'w') as f:
        json.dump(intact_pool, f, indent=2)

    result = {
        'h2_energies'     : h2_energies,
        'ranked_sites'    : ranked,
        'is_pool'         : is_pool,
        'intact_pool'     : intact_pool,
        'n_sites_computed': len(h2_energies),
        'n_sites_total'   : len(sites),
        'outdir'          : str(phase_dir),
        'status'          : status,
    }

    out_json = phase_dir / 'h2_adsorption_energies.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    n_intact = len(intact_pool)
    n_diss   = sum(1 for v in is_pool.values() if v['status'] == 'dissociated')
    print(f"  H2* energies     → {out_json}")
    print(f"  H2_site_coords   → {h2_coords_path}  (all {len(is_pool)} computed)")
    print(f"  H2_intact_pool   → {intact_pool_path}  ({n_intact} intact / {n_diss} dissociated)")
    return result


# -----------------------SECTION B: Phase 2: H* Adsorption Energy -----------------------

def run_phase2_h_adsorption(
    surface_sites_json: str,
    relaxed_slab_path: str,
    e_clean: float,
    e_h2_gas: float,
    outdir: str,
    slurm_opts: dict | None = None,
    dry_run: bool = True,
) -> dict:
    """
    Section B Phase 2: H* adsorption energy for all surface sites (NB06b).

    This follows NB06b:
        1. Place a single H atom 2.5 Å above each site centroid.
        2. CG-minimize slab+H with bottom layers frozen (fixed box).
        3. Compute E_H* = E(slab+H) - E(clean slab) - 0.5*E(H2 gas).
        4. FS pool = all sites where E_ads(H*) < 0 (negative binding energy).
        5. Save FS pool to H_site_coords.json (centroid, E_ads, true_label, relaxed_h_pos).

    Parameters
    ----------
    surface_sites_json : str
        Path to surface_sites.json from Section A Phase 3.
    relaxed_slab_path : str
        Path to relaxed slab LAMMPS data file.
    e_clean : float
        Potential energy of the clean relaxed slab (eV).
    e_h2_gas : float
        Potential energy of an isolated H2 molecule (eV).
    outdir : str
        Base output directory. Phase 2 outputs go to {outdir}/phase2_h/.
    slurm_opts : dict, optional
        SLURM configuration. If None, uses SLURM_DEFAULTS with
        partition='multigpu', time='01:00:00'.
    dry_run : bool
        If True, generate scripts without submitting. Default True.

    Returns
    -------
    dict
        {
          'h_energies': {site_id: E_H_eV},
          'ranked_sites': [[site_id, E_H_eV], ...],
          'fs_pool': {site_id: {centroid, E_ads, true_label, relaxed_h_pos}},
          'n_sites_computed': int,
          'n_sites_total': int,
          'outdir': str,
          'status': 'generated' or 'submitted',
        }
    """
    from models.config import SLURM_DEFAULTS

    if slurm_opts is None:
        slurm_opts = {**SLURM_DEFAULTS, 'partition': 'multigpu', 'time': '01:00:00'}

    phase_dir  = Path(outdir) / 'phase2_h'
    struct_dir = phase_dir / 'structures'
    script_dir = phase_dir / 'scripts'
    slurm_dir  = phase_dir / 'slurm'
    result_dir = phase_dir / 'results'
    for d in [struct_dir, script_dir, slurm_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(surface_sites_json) as f:
        sites_data = json.load(f)
    sites    = sites_data['sites']
    site_map = {site['site_id']: site for site in sites}

    kk = ' '.join(KOKKOS_FLAGS)

    print(f"[Section B Phase 2] H* adsorption: {len(sites)} sites")

    for site in sites:
        sid        = site['site_id']
        centroid   = site['level1']['position']    # [x, y, z]
        site_xy    = centroid[:2]

        struct_path  = str(struct_dir / f'slab_h_{sid}.lammps')
        relaxed_path = str(result_dir  / f'h_atom_{sid}_relaxed.lammps')
        log_path     = str(result_dir  / f'h_min_{sid}.log')
        lammps_in    = str(script_dir  / f'h_min_{sid}.in')
        slurm_sh     = str(slurm_dir   / f'h_slurm_{sid}.sh')

        add_adsorbate(
            slab_path=relaxed_slab_path,
            site_position=site_xy,
            species='H',
            masses=MASSES_7,
            e2t=E2T_7,
            out_path=struct_path,
            height=H2_HEIGHT,
        )

        write_adsorbate_min_script(
            slab_ads_input=struct_path,
            ads_output=relaxed_path,
            out_path=lammps_in,
            pair_style=PAIR_STYLE,
            mace_model=MACE_MODEL_LAMMPS,
            pair_suffix=PAIR_SUFFIX,
            elem_str=ELEM_STR_7,
            z_freeze_cutoff=Z_FREEZE_CUTOFF,
            etol=ADS_MIN_ETOL,
            ftol=ADS_MIN_FTOL,
            maxiter=ADS_MIN_MAXITER,
            maxeval=ADS_MIN_MAXEVAL,
        )

        write_slurm_job(
            job_name=f'Hads_{sid}',
            slurm_config=slurm_opts,
            out_path=slurm_sh,
            commands=[f'{LAMMPS_CMD} {kk} -in {lammps_in} -log {log_path}'],
        )

        if not dry_run:
            submit_slurm_job(slurm_sh)

    status = 'submitted' if not dry_run else 'generated'
    print(f"  Scripts in {script_dir}/  ({status})")

    # Collect results; FS pool = all sites with E_ads < 0
    h_energies = {}
    fs_pool    = {}

    for site in sites:
        sid        = site['site_id']
        centroid   = site['level1']['position']
        true_label = site['level1']['full_label']

        log_path     = str(result_dir / f'h_min_{sid}.log')
        relaxed_path = str(result_dir / f'h_atom_{sid}_relaxed.lammps')

        parsed = parse_energy_log(log_path)
        if parsed and 'pe_final_eV' in parsed:
            e_bind = calc_binding_energy(
                e_with_h=parsed['pe_final_eV'],
                e_clean=e_clean,
                e_h2_gas=e_h2_gas,
                n_h2_ref=0.5,
            )
            h_energies[sid] = e_bind

            if e_bind < 0:
                relaxed_h_pos = _get_h_final_position(relaxed_path)
                fs_pool[sid] = {
                    'centroid'     : centroid,
                    'E_ads'        : e_bind,
                    'true_label'   : true_label,
                    'relaxed_h_pos': relaxed_h_pos,
                }

    ranked = rank_sites(h_energies) if h_energies else []

    # H_all_energies.json — all computed sites before the E_ads < 0 filter (audit trail)
    all_h = {
        sid: {
            'centroid'  : site_map[sid]['level1']['position'],
            'E_ads'     : h_energies[sid],
            'true_label': site_map[sid]['level1']['full_label'],
        }
        for sid in h_energies
    }
    all_h_path = phase_dir / 'H_all_energies.json'
    with open(all_h_path, 'w') as f:
        json.dump(all_h, f, indent=2)

    # H_site_coords.json — FS pool: only sites with E_ads < 0
    h_coords_path = phase_dir / 'H_site_coords.json'
    with open(h_coords_path, 'w') as f:
        json.dump(fs_pool, f, indent=2)

    result = {
        'h_energies'      : h_energies,
        'ranked_sites'    : ranked,
        'all_h_energies'  : all_h,
        'fs_pool'         : fs_pool,
        'n_sites_computed': len(h_energies),
        'n_sites_total'   : len(sites),
        'outdir'          : str(phase_dir),
        'status'          : status,
    }

    out_json = phase_dir / 'h_adsorption_energies.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  H* energies      → {out_json}")
    print(f"  H_all_energies   → {all_h_path}  (all {len(all_h)} computed)")
    print(f"  H_site_coords    → {h_coords_path}  ({len(fs_pool)} with E_ads < 0)")
    return result


# -----------------------SECTION B: Phase 3: Adsorption Energy Orchestrator -----------------------

def orchestrate_adsorption_energies(
    surface_sites_json: str,
    relaxed_slab_path: str,
    e_clean: float,
    e_h2_gas: float,
    outdir: str = 'calculation/adsorption',
    slurm_opts: dict | None = None,
    dry_run: bool = True,
) -> dict:
    """
    Section B Full Pipeline: H2* adsorption energies → H* adsorption energies.

    Chains Phase 1 (H2*) and Phase 2 (H*) into a single call and writes a
    consolidated adsorption_summary.json with all energies and the top-2
    H* sites for use as NEB IS/FS seeds in Section C.

    Parameters
    ----------
    surface_sites_json : str
        Path to surface_sites.json from Section A Phase 3.
    relaxed_slab_path : str
        Path to relaxed slab LAMMPS data file (Section A Phase 2 output).
    e_clean : float
        Potential energy of the clean relaxed slab (eV).
    e_h2_gas : float
        Potential energy of an isolated H2 molecule (eV).
    outdir : str
        Base output directory. Default 'calculation/adsorption'.
    slurm_opts : dict, optional
        SLURM configuration shared by both phases. If None, each phase uses
        its own defaults (2 h for H2*, 1 h for H*).
    dry_run : bool
        If True, generate all scripts without SLURM submission. Default True.

    Returns
    -------
    dict
        {
          'phase1': phase 1 result dict,
          'phase2': phase 2 result dict,
          'neb_candidates': [[site_id, E_H_eV], ...],
          'summary_json': str,
          'status': str,
        }
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"SECTION B: Adsorption Energy Calculation")
    print(f"{'='*80}\n")

    print(f">>> PHASE 1: H2* Adsorption Energy")
    p1 = run_phase1_h2_adsorption(
        surface_sites_json=surface_sites_json,
        relaxed_slab_path=relaxed_slab_path,
        e_clean=e_clean,
        e_h2_gas=e_h2_gas,
        outdir=outdir,
        slurm_opts=slurm_opts,
        dry_run=dry_run,
    )

    print(f"\n>>> PHASE 2: H* Adsorption Energy")
    p2 = run_phase2_h_adsorption(
        surface_sites_json=surface_sites_json,
        relaxed_slab_path=relaxed_slab_path,
        e_clean=e_clean,
        e_h2_gas=e_h2_gas,
        outdir=outdir,
        slurm_opts=slurm_opts,
        dry_run=dry_run,
    )

    summary_path = Path(outdir) / 'adsorption_summary.json'
    summary = {
        'phase1'     : p1,
        'phase2'     : p2,
        'is_pool'    : p1.get('is_pool', {}),
        'fs_pool'    : p2.get('fs_pool', {}),
        'summary_json': str(summary_path),
        'status'     : p1['status'],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    n_is = len(p1.get('is_pool', {}))
    n_fs = len(p2.get('fs_pool', {}))
    print(f"\n{'='*80}")
    print(f"SECTION B COMPLETE")
    print(f"{'='*80}")
    print(f"  H2* energies : {p1['n_sites_computed']}/{p1['n_sites_total']} sites")
    print(f"  H* energies  : {p2['n_sites_computed']}/{p2['n_sites_total']} sites")
    print(f"  IS pool (H2* intact)  : {n_is} sites  → {Path(outdir)/'phase1_h2'/'H2_site_coords.json'}")
    print(f"  FS pool (E_ads < 0)   : {n_fs} sites  → {Path(outdir)/'phase2_h'/'H_site_coords.json'}")
    print(f"  Summary: {summary_path}")
    return summary




# -----------------------SECTION C: NEB workflow: Dissociation & Surface-subsurface  -----------------------
"""
Section C: NEB Workflow — H2 Dissociation on Surface
=====================================================
Sets up and submits Climbing-Image NEB calculations for every symmetry-unique
IS (H2* intact) → FS (H* + H*) dissociation pathway.

Phases:
    1.   Load IS pool, FS pool, surface graph, and site metadata from Section B.
    1b.  Enumerate all candidate FS pairs with geometric + graph-distance filters.
    1c.  Deduplicate FS pairs by chemical-environment fingerprint (L1 + L2 shell-1).
    1d.  Deduplicate IS sites by the same fingerprint; keep most stable per group.
    2.   IS × FS cross-product — compute delta_E for every combination.
    2b.  Proximity filter (IS centroid ↔ FS midpoint < 5 Å) then label dedup.
    3.   Per-job: build IS/FS LAMMPS files, FS-min script, ASE NEB script, SLURM job.
         Emit job_index.txt + run_neb_array.sh for batch array submission.

Key implementation notes (matching NB06b2 exactly):
    - FS raw structure = IS metal atoms + 2 H at NB06b relaxed XY coords,
      z = max_metal_z + 1.5 Å, then LAMMPS-minimized before NEB.
    - NEB uses ASE CI-NEB: spring_const=2.0 eV/Å², n_images=18.
    - Two-phase NEB: phase-1 fmax=0.15 eV/Å (5000 steps);
                     phase-2 CI fmax=0.05 eV/Å (10000 steps).
    - E_IS is hardcoded in run_neb.py; E_FS is parsed at runtime from fs_min.log.
"""

from itertools import combinations as _combinations

from models.energetics import calc_reaction_energy, summarise_neb
from models.structure import build_fs_raw_structure
from models.ase_neb import run_neb_pipeline


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_site_subgraph(G):
    """Site-only NetworkX graph built from 'site-site' edges in G."""
    import networkx as nx
    sg = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get('edge_type') == 'site-site':
            sg.add_edge(u, v)
    return sg


def _graph_distance(s1: str, s2: str, site_subgraph) -> int:
    """Shortest-path hops between two site nodes; -1 if disconnected."""
    import networkx as nx
    try:
        return nx.shortest_path_length(site_subgraph, s1, s2)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return -1


def _site_signature(sid: str, sites_dict: dict) -> tuple:
    """L1+L2-shell-1 chemical fingerprint for one site (matches NB06b2 Cell A+).

    L1 = sorted element tuple of constituent atoms (level1).
    L2 = for each L1 atom: (element, sorted shell-1 neighbour elements).
    """
    site = sites_dict[sid]
    l1 = tuple(sorted(
        a['element'] for a in site['level1']['constituent_atoms']
    ))
    l2 = tuple(sorted(
        (env['element'],
         tuple(sorted(n['element'] for n in env.get('shell1', []))))
        for env in site.get('level2', {}).values()
    ))
    return (l1, l2)


def _pair_signature(s1: str, s2: str, sites_dict: dict) -> tuple:
    """Order-independent fingerprint for a (s1, s2) FS pair."""
    return tuple(sorted([_site_signature(s1, sites_dict),
                         _site_signature(s2, sites_dict)]))


# ---------------------------------------------------------------------------
# Section C: Phase 1 — Load pools and surface graph
# ---------------------------------------------------------------------------

def load_neb_pools(
    phase1_h2_dir: str,
    phase2_h_dir: str,
    phase3_sites_dir: str,
    outdir: str,
) -> dict:
    """
    Section C Phase 1: Load IS/FS pools, surface graph, and site metadata.

    Reads all Section B output files and pre-loads per-site energies and
    coordinates into memory so downstream phases work from dicts.

    Parameters
    ----------
    phase1_h2_dir : str
        Section B Phase 1 directory (H2_intact_pool.json, results/h2_*).
    phase2_h_dir : str
        Section B Phase 2 directory (H_site_coords.json, results/h_*).
    phase3_sites_dir : str
        Section A Phase 3 directory (surface_sites.json, surface_graph.gml).
    outdir : str
        Output directory; pools_metadata.json written here.

    Returns
    -------
    dict with keys:
        is_pool, is_xy, is_energies, is_true_labels,
        fs_pool, fs_xy, fs_energies, fs_eads, fs_true_labels,
        sites_dict, G, site_subgraph,
        phase1_h2_dir, phase2_h_dir.
    """
    from models.parsers import parse_energy_log

    Path(outdir).mkdir(parents=True, exist_ok=True)
    p1 = Path(phase1_h2_dir)
    p2 = Path(phase2_h_dir)
    p3 = Path(phase3_sites_dir)

    # ── IS pool ──────────────────────────────────────────────────────────────
    with open(p1 / 'H2_intact_pool.json') as f:
        is_pool = json.load(f)

    is_xy = {}
    is_energies = {}
    is_true_labels = {}

    for sid, info in is_pool.items():
        is_true_labels[sid] = info.get('true_label', sid)

        parsed = parse_energy_log(str(p1 / 'results' / f'h2_min_{sid}.log'))
        if parsed and 'pe_final_eV' in parsed:
            is_energies[sid] = parsed['pe_final_eV']

        relaxed = p1 / 'results' / f'h2_{sid}_relaxed.lammps'
        if relaxed.exists():
            try:
                struct = ase_read(str(relaxed), format='lammps-data', style='atomic')
                syms = np.array(struct.get_chemical_symbols())
                pos  = struct.get_positions()
                h_pos = pos[syms == 'H']
                if len(h_pos) == 2:
                    cx, cy = h_pos.mean(axis=0)[:2]
                    is_xy[sid] = (float(cx), float(cy))
            except Exception:
                pass

    # ── FS pool ───────────────────────────────────────────────────────────────
    with open(p2 / 'H_site_coords.json') as f:
        fs_pool = json.load(f)

    fs_xy = {}
    fs_energies = {}
    fs_eads = {}
    fs_true_labels = {}

    for sid, info in fs_pool.items():
        fs_true_labels[sid] = info.get('true_label', sid)
        fs_eads[sid]        = info.get('E_ads', float('nan'))

        parsed = parse_energy_log(str(p2 / 'results' / f'h_min_{sid}.log'))
        if parsed and 'pe_final_eV' in parsed:
            fs_energies[sid] = parsed['pe_final_eV']

        relaxed = p2 / 'results' / f'h_atom_{sid}_relaxed.lammps'
        if relaxed.exists():
            try:
                struct = ase_read(str(relaxed), format='lammps-data', style='atomic')
                syms = np.array(struct.get_chemical_symbols())
                pos  = struct.get_positions()
                h_idx = np.where(syms == 'H')[0]
                if len(h_idx) > 0:
                    x, y = pos[h_idx[-1]][:2]
                    fs_xy[sid] = (float(x), float(y))
            except Exception:
                pass

    # ── Surface graph + site metadata ────────────────────────────────────────
    with open(p3 / 'surface_sites.json') as f:
        sites_data = json.load(f)
    sites_dict = {s['site_id']: s for s in sites_data['sites']}

    G = nx.read_gml(str(p3 / 'surface_graph.gml'))
    site_subgraph = _build_site_subgraph(G)

    print(f'[Section C Phase 1] IS pool: {len(is_pool)} intact  '
          f'({len(is_xy)} xy, {len(is_energies)} energies)')
    print(f'                    FS pool: {len(fs_pool)} (E<0)    '
          f'({len(fs_xy)} xy, {len(fs_energies)} energies)')

    meta = {
        'n_is': len(is_pool), 'n_is_xy': len(is_xy),
        'n_fs': len(fs_pool), 'n_fs_xy': len(fs_xy),
    }
    with open(Path(outdir) / 'pools_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return {
        'is_pool'        : is_pool,
        'is_xy'          : is_xy,
        'is_energies'    : is_energies,
        'is_true_labels' : is_true_labels,
        'fs_pool'        : fs_pool,
        'fs_xy'          : fs_xy,
        'fs_energies'    : fs_energies,
        'fs_eads'        : fs_eads,
        'fs_true_labels' : fs_true_labels,
        'sites_dict'     : sites_dict,
        'G'              : G,
        'site_subgraph'  : site_subgraph,
        'phase1_h2_dir'  : str(p1),
        'phase2_h_dir'   : str(p2),
    }


# ---------------------------------------------------------------------------
# Section C: Phase 1b — FS pair enumeration
# ---------------------------------------------------------------------------

def enumerate_fs_pairs(
    pools: dict,
    e_clean: float,
    outdir: str,
    sep_min: float = 2.5,
    sep_max: float = 6.0,
    graph_dist_min: int = 2,
) -> list:
    """
    Section C Phase 1b: Enumerate all candidate H* + H* FS site pairs.

    Filters applied per pair (s1, s2):
        - Both sites have valid XY coords and E_ads < 0.
        - XY Euclidean separation: sep_min ≤ sep ≤ sep_max  (2.5–6.0 Å).
        - graph_dist ≥ graph_dist_min (2) — rejects nearest-neighbour pairs.
          graph_dist = -1 (disconnected in site graph) is accepted.

    E_FS = fs_energies[s1] + fs_energies[s2] - e_clean
    (slab counted once; each fs_energy is the total slab+H energy).

    Returns fs_pairs sorted by combined_eads (most stable first).
    Saves fs_pairs_raw.json.
    """
    fs_xy    = pools['fs_xy']
    fs_eads  = pools['fs_eads']
    fs_eng   = pools['fs_energies']
    fs_lbls  = pools['fs_true_labels']
    subgraph = pools['site_subgraph']

    sites_pool = [s for s in fs_eng
                  if s in fs_eads and fs_eads[s] < 0 and s in fs_xy]

    fs_pairs = []
    for s1, s2 in _combinations(sites_pool, 2):
        x1, y1 = fs_xy[s1]
        x2, y2 = fs_xy[s2]
        sep = float(np.hypot(x1 - x2, y1 - y2))

        if not (sep_min <= sep <= sep_max):
            continue

        gdist = _graph_distance(s1, s2, subgraph)
        if gdist >= 0 and gdist < graph_dist_min:
            continue

        E_FS = fs_eng[s1] + fs_eng[s2] - e_clean
        fs_pairs.append({
            'site1'        : s1,
            'site2'        : s2,
            'sep'          : sep,
            'graph_dist'   : gdist,
            'combined_eads': fs_eads[s1] + fs_eads[s2],
            'E_FS'         : E_FS,
            'label'        : f'{s1}+{s2}',
            'true_label1'  : fs_lbls.get(s1, s1),
            'true_label2'  : fs_lbls.get(s2, s2),
        })

    fs_pairs.sort(key=lambda x: x['combined_eads'])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir) / 'fs_pairs_raw.json', 'w') as f:
        json.dump(fs_pairs, f, indent=2)

    print(f'[Section C Phase 1b] {len(sites_pool)} FS sites → '
          f'{len(fs_pairs)} raw pairs  (sep {sep_min}–{sep_max} Å, '
          f'graph_dist≥{graph_dist_min})')
    return fs_pairs


# ---------------------------------------------------------------------------
# Section C: Phase 1c — FS pair deduplication
# ---------------------------------------------------------------------------

def deduplicate_fs_pairs(
    fs_pairs: list,
    sites_dict: dict,
    outdir: str,
) -> list:
    """
    Section C Phase 1c: Deduplicate FS pairs by chemical-environment fingerprint.

    Groups fs_pairs (sorted by combined_eads) by _pair_signature.
    Keeps the first (most stable) representative per group.
    Adds 'n_grouped' to each kept pair.
    Saves fs_pairs_unique.json.
    """
    groups = {}
    for p in fs_pairs:
        key = _pair_signature(p['site1'], p['site2'], sites_dict)
        groups.setdefault(key, []).append(p)

    unique_pairs = []
    for grp in groups.values():
        rep = grp[0]
        rep['n_grouped'] = len(grp)
        unique_pairs.append(rep)

    unique_pairs.sort(key=lambda x: x['combined_eads'])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir) / 'fs_pairs_unique.json', 'w') as f:
        json.dump(unique_pairs, f, indent=2)

    print(f'[Section C Phase 1c] {len(fs_pairs)} raw → '
          f'{len(unique_pairs)} unique pairs after fingerprint dedup')
    return unique_pairs


# ---------------------------------------------------------------------------
# Section C: Phase 1d — IS site deduplication
# ---------------------------------------------------------------------------

def deduplicate_is_sites(
    pools: dict,
    sites_dict: dict,
    outdir: str,
) -> list:
    """
    Section C Phase 1d: Deduplicate IS sites by chemical-environment fingerprint.

    Groups valid IS sites (have xy + energies) by _site_signature.
    Keeps the representative with the lowest pe_final_eV per group.
    Saves is_sites_unique.json.
    Returns list of site_id strings.
    """
    is_xy    = pools['is_xy']
    is_eng   = pools['is_energies']
    is_pool  = pools['is_pool']
    is_lbls  = pools['is_true_labels']

    valid = [sid for sid in is_pool if sid in is_xy and sid in is_eng]

    groups = {}
    for sid in valid:
        key = _site_signature(sid, sites_dict)
        groups.setdefault(key, []).append(sid)

    unique_is = []
    for sids in groups.values():
        best = min(sids, key=lambda s: is_eng[s])
        unique_is.append(best)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    out = {
        sid: {
            'true_label': is_lbls.get(sid, sid),
            'E_is'      : is_eng[sid],
            'is_xy'     : is_xy[sid],
            'n_grouped' : len(groups[_site_signature(sid, sites_dict)]),
        }
        for sid in unique_is
    }
    with open(Path(outdir) / 'is_sites_unique.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f'[Section C Phase 1d] {len(valid)} valid IS → '
          f'{len(unique_is)} unique after fingerprint dedup')
    return unique_is


# ---------------------------------------------------------------------------
# Section C: Phase 2 — IS × FS cross-product
# ---------------------------------------------------------------------------

def build_is_fs_cross_product(
    unique_is_sites: list,
    unique_pairs: list,
    pools: dict,
    outdir: str,
) -> list:
    """
    Section C Phase 2: Form all unique_is × unique_pairs combinations.

    Per combination computes:
        is_fs_dist = XY distance between IS centroid and FS pair midpoint
        delta_E    = calc_reaction_energy(E_FS, E_IS)  (eV)
        label      = '{is_sid}__{fs_sid1}+{fs_sid2}'

    Saves all_combinations.json. Returns list sorted by delta_E.
    """
    is_xy    = pools['is_xy']
    is_eng   = pools['is_energies']
    is_lbls  = pools['is_true_labels']
    fs_xy    = pools['fs_xy']
    fs_lbls  = pools['fs_true_labels']

    all_combos = []
    for is_sid in unique_is_sites:
        ix, iy = is_xy[is_sid]
        E_IS   = is_eng[is_sid]

        for fp in unique_pairs:
            s1, s2 = fp['site1'], fp['site2']
            if s1 not in fs_xy or s2 not in fs_xy:
                continue
            fcx = (fs_xy[s1][0] + fs_xy[s2][0]) / 2.0
            fcy = (fs_xy[s1][1] + fs_xy[s2][1]) / 2.0

            all_combos.append({
                'is_site'       : is_sid,
                'is_true_label' : is_lbls.get(is_sid, is_sid),
                'fs_site1'      : s1,
                'fs_site2'      : s2,
                'fs_true_label1': fs_lbls.get(s1, s1),
                'fs_true_label2': fs_lbls.get(s2, s2),
                'is_xy'         : (ix, iy),
                'fs_centroid'   : (fcx, fcy),
                'is_fs_dist'    : float(np.hypot(ix - fcx, iy - fcy)),
                'E_IS'          : E_IS,
                'E_FS'          : fp['E_FS'],
                'delta_E'       : calc_reaction_energy(fp['E_FS'], E_IS),
                'fs_sep'        : fp['sep'],
                'graph_dist'    : fp['graph_dist'],
                'label'         : f'{is_sid}__{s1}+{s2}',
            })

    all_combos.sort(key=lambda x: x['delta_E'])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir) / 'all_combinations.json', 'w') as f:
        json.dump(all_combos, f, indent=2)

    print(f'[Section C Phase 2] {len(unique_is_sites)} IS × '
          f'{len(unique_pairs)} FS = {len(all_combos)} combinations')
    return all_combos


# ---------------------------------------------------------------------------
# Section C: Phase 2b — Proximity filter + label deduplication
# ---------------------------------------------------------------------------

def apply_proximity_and_dedup_filter(
    all_combinations: list,
    outdir: str,
    prox_cutoff: float = 5.0,
) -> list:
    """
    Section C Phase 2b: Proximity filter then true-label + graph-dist dedup.

    Step 1 — Proximity: keep combinations where is_fs_dist < prox_cutoff (Å, XY).
    Step 2 — Label dedup: key = (is_true_label,
                                  sorted([fs_true_label1, fs_true_label2]),
                                  graph_dist).
             Keep the representative with the most negative delta_E per key.

    Saves filtered_combinations.json and deduped_combinations.json.
    Returns deduped_combinations (the final NEB job list).
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    filtered = [c for c in all_combinations if c['is_fs_dist'] < prox_cutoff]
    with open(Path(outdir) / 'filtered_combinations.json', 'w') as f:
        json.dump(filtered, f, indent=2)

    dedup = {}
    for c in filtered:
        key = (
            c['is_true_label'],
            tuple(sorted([c['fs_true_label1'], c['fs_true_label2']])),
            c['graph_dist'],
        )
        if key not in dedup or c['delta_E'] < dedup[key]['delta_E']:
            dedup[key] = c

    deduped = sorted(dedup.values(), key=lambda x: x['delta_E'])
    with open(Path(outdir) / 'deduped_combinations.json', 'w') as f:
        json.dump(deduped, f, indent=2)

    print(f'[Section C Phase 2b] {len(all_combinations)} → '
          f'{len(filtered)} (prox<{prox_cutoff} Å) → '
          f'{len(deduped)} (after label dedup) NEB jobs')
    return deduped


# ---------------------------------------------------------------------------
# Section C: Phase 3 — NEB Orchestrator
# ---------------------------------------------------------------------------

def orchestrate_neb(
    deduped_combinations: list,
    pools: dict,
    e_clean: float,
    outdir: str,
    slurm_opts: dict | None = None,
    neb_slurm_opts: dict | None = None,
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 0.05,
    h_height: float = 1.5,
    dry_run: bool = True,
) -> dict:
    """
    Section C Phase 3: Write per-job NEB files and split GPU/CPU SLURM scripts.

    Per combination:
        1. neb_initial.lammps       — IS relaxed structure (Velocities stripped)
        2. neb_final_raw.lammps     — FS raw (metal + 2 H at relaxed XY, z=max+h_height)
        3. min_fs.lammps            — LAMMPS CG min of FS raw → neb_final_relaxed.lammps
        4. run_neb.py               — ASE CI-NEB script (E_IS hardcoded, E_FS from log)
        5. slurm_fsmin_{label}.sh   — GPU job: LAMMPS FS minimization only
        6. slurm_neb_{label}.sh     — CPU job: ASE NEB only

    After all jobs:
        job_index.txt           — one label per line
        run_fsmin_array.sh      — GPU SLURM array (LAMMPS FS min, one task per label)
        run_neb_array.sh        — CPU SLURM array (ASE NEB, one task per label)
        neb_pairs.json          — full metadata for all jobs

    Parameters
    ----------
    deduped_combinations : list
        Output of apply_proximity_and_dedup_filter().
    pools : dict
        Output of load_neb_pools().
    e_clean : float
        Clean slab potential energy (eV).
    outdir : str
        Base output dir. Per-job dirs go to {outdir}/neb/{label}/.
    slurm_opts : dict, optional
        GPU SLURM config for LAMMPS FS minimization. Default: multigpu, a100:1, 06:00:00.
    neb_slurm_opts : dict, optional
        CPU SLURM config for ASE NEB. Default: short partition, no GPU, 12:00:00.
    n_images : int
        Intermediate NEB images. Default 18.
    spring_const : float
        Spring constant eV/Å². Default 1.0.
    neb_ftol : float
        CINEB force tolerance eV/Å. Default 0.05.
    h_height : float
        FS H placement height above max metal z (Å). Default 1.5.
    dry_run : bool
        Generate files without SLURM submission. Default True.

    Returns
    -------
    dict with keys: neb_jobs, job_index, fsmin_array_script, neb_array_script,
        pairs_json, n_jobs, status.
    """
    from models.config import (
        SLURM_DEFAULTS, LAMMPS_CMD, MACE_MODEL_LAMMPS, MACE_MODEL_ASE,
        PAIR_STYLE, PAIR_SUFFIX, ELEM_STR_7, KOKKOS_FLAGS,
        MASSES_7, E2T_7, Z_FREEZE_CUTOFF, FTOL,
        ADS_MIN_ETOL, ADS_MIN_FTOL, ADS_MIN_MAXITER, ADS_MIN_MAXEVAL,
    )

    if slurm_opts is None:
        slurm_opts = {**SLURM_DEFAULTS, 'partition': 'multigpu', 'time': '06:00:00'}

    if neb_slurm_opts is None:
        neb_slurm_opts = {
            **SLURM_DEFAULTS,
            'partition': 'short',
            'time': '12:00:00',
            'gpu': None,
            'cuda_version': None,
        }

    neb_dir = Path(outdir) / 'neb'
    neb_dir.mkdir(parents=True, exist_ok=True)

    p1_results = Path(pools['phase1_h2_dir']) / 'results'
    fs_xy      = pools['fs_xy']
    is_eng     = pools['is_energies']
    kk         = ' '.join(KOKKOS_FLAGS)

    neb_jobs = []

    for c in deduped_combinations:
        is_sid = c['is_site']
        s1, s2 = c['fs_site1'], c['fs_site2']
        label  = c['label']

        job_dir = neb_dir / label
        job_dir.mkdir(parents=True, exist_ok=True)

        # 1. IS — copy relaxed lammps, strip Velocities
        is_src  = p1_results / f'h2_{is_sid}_relaxed.lammps'
        is_dest = job_dir / 'neb_initial.lammps'
        if is_src.exists():
            txt = is_src.read_text()
            if 'Velocities' in txt:
                txt = txt[:txt.index('Velocities')].rstrip() + '\n'
            is_dest.write_text(txt)

        # 2. FS raw — metal slab from IS + 2 H at relaxed FS XY coords
        fs_raw = job_dir / 'neb_final_raw.lammps'
        if is_dest.exists() and s1 in fs_xy and s2 in fs_xy:
            build_fs_raw_structure(
                is_lammps=str(is_dest),
                fs_xy1=fs_xy[s1],
                fs_xy2=fs_xy[s2],
                masses=MASSES_7,
                e2t=E2T_7,
                out_path=str(fs_raw),
                h_height=h_height,
            )

        # 3. FS minimization script (reuse write_adsorbate_min_script)
        fs_relaxed = job_dir / 'neb_final_relaxed.lammps'
        fs_min_log = job_dir / 'fs_min.log'
        min_script = job_dir / 'min_fs.lammps'
        write_adsorbate_min_script(
            slab_ads_input=str(fs_raw),
            ads_output=str(fs_relaxed),
            out_path=str(min_script),
            pair_style=PAIR_STYLE,
            mace_model=MACE_MODEL_LAMMPS,
            pair_suffix=PAIR_SUFFIX,
            elem_str=ELEM_STR_7,
            z_freeze_cutoff=Z_FREEZE_CUTOFF,
            etol=ADS_MIN_ETOL,
            ftol=ADS_MIN_FTOL,
            maxiter=ADS_MIN_MAXITER,
            maxeval=ADS_MIN_MAXEVAL,
        )

        # 4. ASE NEB script — E_IS hardcoded; E_FS parsed at runtime from fs_min.log
        E_IS        = is_eng.get(is_sid, float('nan'))
        traj_p1     = str(job_dir / 'neb_phase1.traj')
        traj_p2     = str(job_dir / 'neb_phase2.traj')
        neb_script  = run_neb_pipeline(
            is_file=str(is_dest),
            fs_file=str(fs_relaxed),
            e_is=E_IS,
            mace_model_path=MACE_MODEL_ASE,
            barrier_file=str(job_dir / 'neb_barrier.txt'),
            path_file=str(job_dir / 'neb_path.dat'),
            outdir=str(job_dir),
            fs_log_file=str(fs_min_log),
            job_name='neb',
            n_images=n_images,
            spring_const=spring_const,
            neb_ftol=neb_ftol,
            z_freeze_cutoff=Z_FREEZE_CUTOFF,
            device='cpu',
            label_is=f'IS:{is_sid}',
            label_fs=f'FS:{s1}+{s2}',
            traj_phase1=traj_p1,
            traj_phase2=traj_p2,
        )

        # 5a. GPU SLURM: LAMMPS FS minimization only
        fsmin_sh = str(job_dir / f'slurm_fsmin_{label}.sh')
        write_slurm_job(
            job_name=f'fsmin_{label}',
            slurm_config=slurm_opts,
            out_path=fsmin_sh,
            commands=[f'{LAMMPS_CMD} {kk} -in {min_script} -log {fs_min_log}'],
        )

        # 5b. CPU SLURM: ASE NEB (self-chaining via traj checkpoint)
        neb_sh = str(job_dir / f'slurm_neb_{label}.sh')
        _neb_wall = neb_slurm_opts.get('time', '12:00:00')
        _h, _m, _s = (int(x) for x in _neb_wall.split(':'))
        _neb_cutoff_secs = _h * 3600 + _m * 60 + _s - 300
        _neb_cutoff = f'{_neb_cutoff_secs // 3600:02d}:{(_neb_cutoff_secs % 3600) // 60:02d}:{_neb_cutoff_secs % 60:02d}'
        write_chained_slurm_job(
            job_name=f'neb_{label}',
            slurm_config=neb_slurm_opts,
            out_path=neb_sh,
            first_commands=[f'python {neb_script}'],
            restart_commands=[f'python {neb_script}'],
            restart_glob=traj_p2,
            cutoff=_neb_cutoff,
            work_dir=str(job_dir),
        )

        if not dry_run:
            submit_slurm_job(fsmin_sh)
            submit_slurm_job(neb_sh)

        neb_jobs.append({
            'label'       : label,
            'is_site'     : is_sid,
            'fs_site1'    : s1,
            'fs_site2'    : s2,
            'E_IS'        : E_IS,
            'E_FS'        : c['E_FS'],
            'delta_E'     : c['delta_E'],
            'is_fs_dist'  : c['is_fs_dist'],
            'neb_script'  : neb_script,
            'min_script'  : str(min_script),
            'fsmin_sh'    : fsmin_sh,
            'neb_sh'      : neb_sh,
            'barrier_file': str(job_dir / 'neb_barrier.txt'),
            'path_file'   : str(job_dir / 'neb_path.dat'),
            'job_dir'     : str(job_dir),
        })

    # job_index.txt — one label per line (1-indexed for SLURM array)
    job_index_path = neb_dir / 'job_index.txt'
    job_index_path.write_text('\n'.join(j['label'] for j in neb_jobs) + '\n')

    ar = (1, len(neb_jobs))

    # GPU array: LAMMPS FS minimization
    fsmin_array = str(neb_dir / 'run_fsmin_array.sh')
    write_slurm_job(
        job_name='fsmin_array',
        slurm_config=slurm_opts,
        out_path=fsmin_array,
        array_range=ar,
        concurrent=4,
        commands=[
            f'LABEL=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_index_path})',
            f'bash {neb_dir}/${{LABEL}}/slurm_fsmin_${{LABEL}}.sh',
        ],
    )

    # CPU array: ASE NEB
    neb_array = str(neb_dir / 'run_neb_array.sh')
    write_slurm_job(
        job_name='neb_array',
        slurm_config=neb_slurm_opts,
        out_path=neb_array,
        array_range=ar,
        concurrent=50,
        commands=[
            f'LABEL=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_index_path})',
            f'bash {neb_dir}/${{LABEL}}/slurm_neb_${{LABEL}}.sh',
        ],
    )

    pairs_json = str(neb_dir / 'neb_pairs.json')
    with open(pairs_json, 'w') as f:
        json.dump(neb_jobs, f, indent=2)

    status = 'submitted' if not dry_run else 'generated'
    print(f'[Section C Phase 3] {len(neb_jobs)} NEB jobs  ({status})')
    print(f'  job_index      : {job_index_path}')
    print(f'  fsmin array sh : {fsmin_array}')
    print(f'  neb array sh   : {neb_array}')
    print(f'  pairs json     : {pairs_json}')

    return {
        'neb_jobs'           : neb_jobs,
        'job_index'          : str(job_index_path),
        'fsmin_array_script' : fsmin_array,
        'neb_array_script'   : neb_array,
        'pairs_json'         : pairs_json,
        'n_jobs'             : len(neb_jobs),
        'status'             : status,
    }


# ---------------------------------------------------------------------------
# Section C: Phase 4 — Collect results from completed NEB jobs
# ---------------------------------------------------------------------------

def collect_neb_results(
    neb_outdir: str,
    outdir: str | None = None,
) -> dict:
    """
    Section C Phase 4: Parse completed NEB jobs and rank by forward barrier.

    Reads neb_pairs.json written by orchestrate_neb, calls summarise_neb on
    each job's neb_barrier.txt and neb_path.dat, and writes ranked_barriers.json
    sorted by Ea (forward barrier, smallest first).

    Run this after all SLURM NEB jobs have finished.

    Parameters
    ----------
    neb_outdir : str
        The outdir passed to orchestrate_neb (contains the neb/ subdirectory).
    outdir : str, optional
        Directory for ranked_barriers.json.  Defaults to neb_outdir.

    Returns
    -------
    dict
        {
          'results'    : list of per-job dicts (label, Ea, E_des, delta_E, ...),
          'n_converged': int,
          'n_failed'   : int,
          'n_missing'  : int,
          'ranked_json': path to ranked_barriers.json,
        }
    """
    neb_dir   = Path(neb_outdir) / 'neb'
    pairs_json = neb_dir / 'neb_pairs.json'

    with open(pairs_json) as f:
        neb_jobs = json.load(f)

    results     = []
    n_converged = 0
    n_failed    = 0
    n_missing   = 0

    for job in neb_jobs:
        label        = job['label']
        barrier_file = job['barrier_file']
        path_file    = job['path_file']

        if not Path(barrier_file).exists():
            n_missing += 1
            results.append({
                'label'     : label,
                'is_site'   : job['is_site'],
                'fs_site1'  : job['fs_site1'],
                'fs_site2'  : job['fs_site2'],
                'Ea'        : None,
                'E_des'     : None,
                'delta_E'   : None,
                'converged' : None,
                'fmax_final': None,
                'status'    : 'missing',
            })
            continue

        neb_res = summarise_neb(barrier_file, path_file)
        converged = neb_res.get('converged')
        if converged:
            n_converged += 1
        else:
            n_failed += 1

        results.append({
            'label'     : label,
            'is_site'   : job['is_site'],
            'fs_site1'  : job['fs_site1'],
            'fs_site2'  : job['fs_site2'],
            'Ea'        : neb_res.get('Ea'),
            'E_des'     : neb_res.get('E_des'),
            'delta_E'   : neb_res.get('delta_E'),
            'converged' : converged,
            'fmax_final': neb_res.get('fmax_final'),
            'status'    : 'converged' if converged else 'unconverged',
        })

    results.sort(key=lambda x: (x['Ea'] is None, x['Ea'] or float('inf')))

    out_path = Path(outdir) if outdir else Path(neb_outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    ranked_json = str(out_path / 'ranked_barriers.json')
    with open(ranked_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'[Section C Phase 4] {len(results)} NEB jobs parsed:')
    print(f'  converged  : {n_converged}')
    print(f'  unconverged: {n_failed}')
    print(f'  missing    : {n_missing}')
    print(f'  ranked_barriers → {ranked_json}')

    return {
        'results'    : results,
        'n_converged': n_converged,
        'n_failed'   : n_failed,
        'n_missing'  : n_missing,
        'ranked_json': ranked_json,
    }


# ---------------------------------------------------------------------------
# Section C: Full Pipeline Orchestrator
# ---------------------------------------------------------------------------

def orchestrate_neb_pipeline(
    phase1_h2_dir: str,
    phase2_h_dir: str,
    phase3_sites_dir: str,
    e_clean: float,
    outdir: str = 'calculation/neb',
    slurm_opts: dict | None = None,
    neb_slurm_opts: dict | None = None,
    sep_min: float = 2.5,
    sep_max: float = 6.0,
    graph_dist_min: int = 2,
    prox_cutoff: float = 5.0,
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 0.05,
    h_height: float = 1.5,
    dry_run: bool = True,
) -> dict:
    """
    Section C Full Pipeline: pools → pairs → dedup → filter → NEB jobs.

    Chains all Section C phases in order and writes section_c_summary.json.

    Parameters
    ----------
    phase1_h2_dir : str
        Section B Phase 1 output directory.
    phase2_h_dir : str
        Section B Phase 2 output directory.
    phase3_sites_dir : str
        Section A Phase 3 output directory.
    e_clean : float
        Clean slab potential energy (eV).
    outdir : str
        Base output directory. Default 'calculation/neb'.
    slurm_opts : dict, optional
        GPU SLURM config for LAMMPS FS minimization.
    neb_slurm_opts : dict, optional
        CPU SLURM config for ASE NEB. Default: short partition, no GPU.
    sep_min, sep_max : float
        FS pair XY separation bounds (Å). Default 2.5–6.0.
    graph_dist_min : int
        Minimum graph distance for FS pairs. Default 2.
    prox_cutoff : float
        IS ↔ FS midpoint XY cutoff (Å). Default 5.0.
    n_images : int
        NEB intermediate images. Default 18.
    spring_const : float
        NEB spring constant (eV/Å²). Default 1.0.
    neb_ftol : float
        CINEB force tolerance (eV/Å). Default 0.05.
    h_height : float
        FS H height above max metal z (Å). Default 1.5.
    dry_run : bool
        Generate all files without SLURM submission. Default True.

    Returns
    -------
    dict
        pools, fs_pairs, is_sites, combinations, deduped,
        neb_result, summary_json, status.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}\nSECTION C: NEB Workflow\n{'='*80}\n")

    print('>>> Phase 1: Load pools')
    pools = load_neb_pools(
        phase1_h2_dir, phase2_h_dir, phase3_sites_dir,
        outdir=str(Path(outdir) / 'phase1_pools'),
    )

    print('\n>>> Phase 1b: FS pair enumeration')
    fs_pairs_raw = enumerate_fs_pairs(
        pools, e_clean,
        outdir=str(Path(outdir) / 'phase1b_fs_pairs'),
        sep_min=sep_min, sep_max=sep_max, graph_dist_min=graph_dist_min,
    )

    print('\n>>> Phase 1c: FS pair deduplication')
    unique_pairs = deduplicate_fs_pairs(
        fs_pairs_raw, pools['sites_dict'],
        outdir=str(Path(outdir) / 'phase1c_unique_fs_pairs'),
    )

    print('\n>>> Phase 1d: IS site deduplication')
    unique_is = deduplicate_is_sites(
        pools, pools['sites_dict'],
        outdir=str(Path(outdir) / 'phase1d_unique_is_sites'),
    )

    print('\n>>> Phase 2: IS × FS cross-product')
    all_combos = build_is_fs_cross_product(
        unique_is, unique_pairs, pools,
        outdir=str(Path(outdir) / 'phase2_cross_product'),
    )

    print('\n>>> Phase 2b: Proximity filter + label dedup')
    deduped = apply_proximity_and_dedup_filter(
        all_combos,
        outdir=str(Path(outdir) / 'phase2b_filtered'),
        prox_cutoff=prox_cutoff,
    )

    print('\n>>> Phase 3: NEB job generation')
    neb_result = orchestrate_neb(
        deduped, pools, e_clean, outdir=outdir,
        slurm_opts=slurm_opts,
        neb_slurm_opts=neb_slurm_opts,
        n_images=n_images, spring_const=spring_const,
        neb_ftol=neb_ftol, h_height=h_height, dry_run=dry_run,
    )

    summary = {
        'n_is_unique'        : len(unique_is),
        'n_fs_unique'        : len(unique_pairs),
        'n_combinations'     : len(all_combos),
        'n_deduped'          : len(deduped),
        'n_neb_jobs'         : neb_result['n_jobs'],
        'job_index'          : neb_result['job_index'],
        'fsmin_array_script' : neb_result['fsmin_array_script'],
        'neb_array_script'   : neb_result['neb_array_script'],
        'pairs_json'         : neb_result['pairs_json'],
        'status'             : neb_result['status'],
    }
    summary_path = Path(outdir) / 'section_c_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}\nSECTION C COMPLETE\n{'='*80}")
    print(f"  IS unique    : {summary['n_is_unique']}")
    print(f"  FS unique    : {summary['n_fs_unique']}")
    print(f"  Combinations : {summary['n_combinations']}")
    print(f"  After filters: {summary['n_deduped']}")
    print(f"  NEB jobs     : {summary['n_neb_jobs']}")
    print(f"  Summary      : {summary_path}")

    return {
        'pools'       : pools,
        'fs_pairs'    : unique_pairs,
        'is_sites'    : unique_is,
        'combinations': all_combos,
        'deduped'     : deduped,
        'neb_result'  : neb_result,
        'summary_json': str(summary_path),
        'status'      : neb_result['status'],
    }


# ---------------------------------------------------------------------------
# Full pipeline: Sections A → B → C
# ---------------------------------------------------------------------------

def orchestrate_full_neb_workflow(
    bulk_min_path: str,
    e_h2_gas: float,
    slab_dir: str,
    ads_dir: str,
    neb_dir: str,
    miller: tuple = (1, 1, 1),
    layers: int = 12,
    vacuum: float = 15.0,
    lateral_repeat: tuple = (5, 6),
    z_freeze_cutoff: float = 22.115,
    surf_timestep: float = 0.0005,
    sep_min: float = 2.5,
    sep_max: float = 6.0,
    graph_dist_min: int = 2,
    prox_cutoff: float = 5.0,
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 0.05,
    h_height: float = 1.5,
    gpu_slurm_cfg: dict | None = None,
    neb_slurm_cfg: dict | None = None,
    dry_run: bool = True,
) -> dict:
    """Chain Sections A → B → C in a single call.

    Designed for use inside the generated neb_run.py cluster orchestrator.
    Each section submits its SLURM jobs (when dry_run=False); the caller is
    responsible for waiting between sections before results are available.

    E_CLEAN is parsed at runtime from the Phase A LAMMPS log. If the log
    does not yet exist (dry_run=True), e_clean is set to float('nan').

    Returns
    -------
    dict with keys: e_clean, n_sites, n_neb_jobs, fsmin_array_script,
        neb_array_script, status.
    """
    from models.parsers import parse_energy_log

    # ── Section A ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSection A: Slab preparation\n{'='*60}")
    results_a = orchestrate_slab_prep(
        bulk_min_path=bulk_min_path,
        outdir=slab_dir,
        miller=miller,
        layers=layers,
        vacuum=vacuum,
        lateral_repeat=lateral_repeat,
        z_freeze_cutoff=z_freeze_cutoff,
        timestep=surf_timestep,
        slurm_opts=gpu_slurm_cfg,
        dry_run=dry_run,
    )

    # e_clean is available only after the Phase A SLURM job finishes and
    # writes the LAMMPS log. In dry_run mode the log does not exist yet.
    phase2_log = results_a.get('phase2_log', '')
    parsed = parse_energy_log(phase2_log) if phase2_log else {}
    e_clean = parsed.get('pe_final_eV', float('nan'))
    print(f'  E_CLEAN = {e_clean}')

    # ── Section B ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSection B: Adsorption energies\n{'='*60}")
    orchestrate_adsorption_energies(
        surface_sites_json=results_a['phase3_sites'],
        relaxed_slab_path=results_a['phase2_relaxed'],
        e_clean=e_clean,
        e_h2_gas=e_h2_gas,
        outdir=ads_dir,
        slurm_opts=gpu_slurm_cfg,
        dry_run=dry_run,
    )

    # ── Section C ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSection C: NEB enumeration + job generation\n{'='*60}")
    c_result = orchestrate_neb_pipeline(
        phase1_h2_dir=str(Path(ads_dir) / 'phase1_h2'),
        phase2_h_dir=str(Path(ads_dir) / 'phase2_h'),
        phase3_sites_dir=str(Path(slab_dir) / 'phase3_sites'),
        e_clean=e_clean,
        outdir=neb_dir,
        slurm_opts=gpu_slurm_cfg,
        neb_slurm_opts=neb_slurm_cfg,
        sep_min=sep_min,
        sep_max=sep_max,
        graph_dist_min=graph_dist_min,
        prox_cutoff=prox_cutoff,
        n_images=n_images,
        spring_const=spring_const,
        neb_ftol=neb_ftol,
        h_height=h_height,
        dry_run=dry_run,
    )

    return {
        'e_clean'            : e_clean,
        'n_sites'            : results_a.get('n_sites', 0),
        'fsmin_array_script' : c_result['neb_result']['fsmin_array_script'],
        'neb_array_script'   : c_result['neb_result']['neb_array_script'],
        'n_neb_jobs'         : c_result['neb_result']['n_jobs'],
        'status'             : c_result['status'],
    }


# ---------------------------------------------------------------------------
# Script writers — called from neb_calculation.ipynb
# ---------------------------------------------------------------------------

def write_neb_run_script(
    bulk_min_path,
    work_dir,
    e_h2_gas,
    slab_dir,
    ads_dir,
    neb_dir,
    miller,
    layers,
    vacuum,
    lat_repeat,
    sep_min,
    sep_max,
    graph_dist_min,
    prox_cutoff,
    n_images,
    spring_const,
    neb_ftol,
    h_height,
    gpu_slurm_cfg,
    neb_slurm_cfg,
    out_py: str,
    z_freeze_cutoff: float = 22.115,
    surf_timestep: float   = 0.0005,
    vib_slurm_cfg=None,
) -> str:
    """Write neb_run.py with embedded config. Returns the output path."""

    _vib_cfg = vib_slurm_cfg if vib_slurm_cfg is not None else neb_slurm_cfg

    _header = f'''#!/usr/bin/env python3
"""
neb_run.py
==========
Standalone NEB workflow orchestrator for H2 dissociation barriers on Hastelloy N.
Submitted via neb_run.sh (configured by notebook variables).

Phase A : Build slab -> surface relax -> enumerate sites
Phase B : H2* adsorption (IS pool) -> H* adsorption (FS pool)
Phase C : FS pair enumeration -> deduplication -> cross-product -> proximity filter
          -> generate per-job NEB scripts + SLURM arrays
Phase D : Submit FS-min GPU array -> wait -> submit NEB CPU array -> wait
Phase E : Vibrational frequencies for dissociation IS + TS (ZPE corrections)

Generated by calculation/neb_calculation.ipynb -- do not edit by hand.
"""

# -- config (injected by neb_calculation.ipynb) --------------------------------
BULK_MIN_PATH  = {bulk_min_path!r}
WORK_DIR       = {work_dir!r}
E_H2_GAS       = {e_h2_gas!r}
SLAB_DIR       = {slab_dir!r}
ADS_DIR        = {ads_dir!r}
NEB_DIR        = {neb_dir!r}
MILLER         = {miller!r}
LAYERS         = {layers!r}
VACUUM         = {vacuum!r}
LAT_REPEAT     = {lat_repeat!r}
# NEB pair-selection filters
SEP_MIN        = {sep_min!r}
SEP_MAX        = {sep_max!r}
GRAPH_DIST_MIN = {graph_dist_min!r}
PROX_CUTOFF    = {prox_cutoff!r}
# NEB calculation parameters
N_IMAGES       = {n_images!r}
SPRING_CONST   = {spring_const!r}
NEB_FTOL       = {neb_ftol!r}
H_HEIGHT       = {h_height!r}
# SLURM configurations
GPU_SLURM_CFG  = {gpu_slurm_cfg!r}
NEB_SLURM_CFG  = {neb_slurm_cfg!r}
VIB_SLURM_CFG  = {_vib_cfg!r}
# Surface relaxation (Phase A)
Z_FREEZE_CUTOFF = {z_freeze_cutoff!r}
SURF_TIMESTEP   = {surf_timestep!r}

'''

    _body = r"""
import os
import sys
sys.path.insert(0, os.path.dirname(WORK_DIR))

from models.neb_workflow import orchestrate_full_neb_workflow
from models.create_slurm import submit_slurm_job, wait_for_jobs

# -- Phases A-D ----------------------------------------------------------------
_ranked_f = os.path.join(NEB_DIR, 'ranked_barriers.json')
if not os.path.exists(_ranked_f):
    result = orchestrate_full_neb_workflow(
        bulk_min_path=BULK_MIN_PATH,
        e_h2_gas=E_H2_GAS,
        slab_dir=SLAB_DIR,
        ads_dir=ADS_DIR,
        neb_dir=NEB_DIR,
        miller=MILLER,
        layers=LAYERS,
        vacuum=VACUUM,
        lateral_repeat=LAT_REPEAT,
        z_freeze_cutoff=Z_FREEZE_CUTOFF,
        surf_timestep=SURF_TIMESTEP,
        sep_min=SEP_MIN,
        sep_max=SEP_MAX,
        graph_dist_min=GRAPH_DIST_MIN,
        prox_cutoff=PROX_CUTOFF,
        n_images=N_IMAGES,
        spring_const=SPRING_CONST,
        neb_ftol=NEB_FTOL,
        h_height=H_HEIGHT,
        gpu_slurm_cfg=GPU_SLURM_CFG,
        neb_slurm_cfg=NEB_SLURM_CFG,
        dry_run=False,
    )
    print(f'  E_CLEAN  : {result["e_clean"]}')
    print(f'  NEB jobs : {result["n_neb_jobs"]}')

    # -- Phase D: Submit SLURM arrays ------------------------------------------
    print('\n' + '='*60)
    print('  Phase D: Submit and wait')
    print('='*60)

    fsmin_jid = submit_slurm_job(result['fsmin_array_script'])
    print(f'  FS-min array submitted  ->  job {fsmin_jid}')
    wait_for_jobs({'fsmin_array': fsmin_jid})
    print('  All FS minimisations done.')

    neb_jid = submit_slurm_job(result['neb_array_script'])
    print(f'  NEB array submitted  ->  job {neb_jid}')
    wait_for_jobs({'neb_array': neb_jid})
    print('  All NEB calculations done.')
else:
    print(f'  Phases A-D already complete: {_ranked_f} — skipping')

# -- Phase E: Vibrational frequencies for dissociation IS + TS -----------------
print('\n' + '='*60)
print('  Phase E: Dissociation vibrational frequencies (ZPE corrections)')
print('='*60)

import json as _json_e
import glob as _glob_e

from models.config import MACE_MODEL_ASE as _MACE_ASE_E
from models.vibrations import orchestrate_vibrations as _orch_vib_e
from models.tst_rates import (
    split_vib_results as _split_vib_e,
    build_rate_dict as _brd_e,
)

_ranked_f_e = os.path.join(NEB_DIR, 'ranked_barriers.json')
if not os.path.exists(_ranked_f_e):
    print(f'WARNING: {_ranked_f_e} not found — Phase E skipped.')
else:
    with open(_ranked_f_e) as _f_e:
        _ranked_e = _json_e.load(_f_e)

    _vib_pairs_e  = []
    _neb_for_rd_e = {}
    _label_pair_e = {}

    for _job_e in _ranked_e:
        if not _job_e.get('converged', False):
            continue
        _lbl_e     = _job_e['label']
        _job_dir_e = os.path.join(NEB_DIR, _lbl_e)
        _is_e      = os.path.join(_job_dir_e, 'neb_initial.lammps')
        _img_dir_e = os.path.join(_job_dir_e, 'images')
        _path_f_e  = _job_e.get('path_file',
                                 os.path.join(_job_dir_e, 'neb_path.dat'))

        # TS = peak-energy intermediate image from neb_path.dat
        _ts_e = None
        if os.path.exists(_path_f_e):
            _rows_e = []
            with open(_path_f_e) as _pf_e:
                for _line_e in _pf_e:
                    if _line_e.startswith('#') or not _line_e.strip():
                        continue
                    _p_e = _line_e.split()
                    if len(_p_e) >= 3:
                        try:
                            _rows_e.append((float(_p_e[0]), float(_p_e[2])))
                        except ValueError:
                            pass
            if len(_rows_e) >= 3:
                _ts_idx_e = max(range(len(_rows_e)),
                                key=lambda _i: _rows_e[_i][1])
                _cands_e = _glob_e.glob(
                    os.path.join(_img_dir_e,
                                 f'image_{_ts_idx_e:02d}_*.lammps'))
                if _cands_e:
                    _ts_e = _cands_e[0]

        if not (os.path.exists(_is_e) and _ts_e and os.path.exists(_ts_e)):
            print(f'  [{_lbl_e}] IS or TS not found — skipping vib.')
            continue

        _vib_pairs_e.append((f'{_lbl_e}_IS', _is_e))
        _vib_pairs_e.append((f'{_lbl_e}_TS', _ts_e))
        _neb_for_rd_e[_lbl_e] = {
            'E_abs':     _job_e.get('Ea',      0.0),
            'E_des':     _job_e.get('E_des',   0.0),
            'delta_E':   _job_e.get('delta_E', 0.0),
            'converged': True,
        }
        try:
            _fs_e = _lbl_e.split('__')[-1]          # e.g. 'Ni_fcc+Mo_hcp'
            _e1_e = _fs_e.split('+')[0].split('_')[0]
            _e2_e = _fs_e.split('+')[1].split('_')[0]
            _label_pair_e[_lbl_e] = tuple(sorted([_e1_e, _e2_e]))
        except Exception:
            _label_pair_e[_lbl_e] = ('?', '?')

    print(f'  Diss vib pairs: {len(_vib_pairs_e)//2} IS/TS sets')

    if _vib_pairs_e:
        _vib_out_e = _orch_vib_e(
            structure_paths = _vib_pairs_e,
            outdir          = os.path.join(NEB_DIR, 'vibrations_diss'),
            mace_model_path = _MACE_ASE_E,
            slurm_opts      = VIB_SLURM_CFG,
            delta           = 0.01,
            device          = 'cpu',
            dry_run         = True,
        )
        from models.create_slurm import (
            submit_slurm_job as _sub_vib_e,
            wait_for_jobs    as _wait_vib_e,
        )
        _vib_jids_e = {_k: _sub_vib_e(_v['slurm'])
                       for _k, _v in _vib_out_e.items() if _v.get('slurm')}
        print(f'  Submitted {len(_vib_jids_e)} diss vib jobs.')
        _wait_vib_e(_vib_jids_e)
        print('  Diss vibrations complete.')

        _vib_is_e, _vib_ts_e = _split_vib_e(_vib_out_e)
        _rd_e = _brd_e(_neb_for_rd_e, _vib_is_e, _vib_ts_e,
                       T_K=700.0, apply_zpe=True)

        _diss_rates_e = {}
        for _lbl_rd, _r_e in _rd_e.items():
            _pair_e = _label_pair_e.get(_lbl_rd, ('?', '?'))
            _diss_rates_e[_lbl_rd] = {
                'pair':   list(_pair_e),
                'Ea_zpe': _r_e.get('Ea_zpe', _neb_for_rd_e[_lbl_rd]['E_abs']),
                'Ed_zpe': _r_e.get('Ed_zpe', _neb_for_rd_e[_lbl_rd]['E_des']),
                'Ea_raw': _r_e.get('Ea_raw', _neb_for_rd_e[_lbl_rd]['E_abs']),
                'Ed_raw': _r_e.get('Ed_raw', _neb_for_rd_e[_lbl_rd]['E_des']),
                'nu':     _r_e.get('nu',     1e13),
                'label':  _lbl_rd,
            }

        _diss_out_e = os.path.join(NEB_DIR, 'diss_vib_rates.json')
        with open(_diss_out_e, 'w') as _fw_e:
            _json_e.dump(_diss_rates_e, _fw_e, indent=2)
        print(f'  Saved diss_vib_rates.json '
              f'({len(_diss_rates_e)} labels) -> {_diss_out_e}')

print('\n' + '='*60)
print('  NEB workflow complete (incl. diss vibrations).')
print('='*60)
"""

    Path(out_py).parent.mkdir(parents=True, exist_ok=True)
    with open(out_py, 'w') as fh:
        fh.write(_header + _body)
    return out_py


def write_neb_orchestrator_sh(
    orch_job_name,
    orch_partition,
    orch_cpus_per_task,
    orch_time,
    orch_openmpi_ver,
    orch_cuda_version,
    orch_conda_env,
    orch_ld_paths,
    out_py,
    out_sh: str,
) -> str:
    """Write neb_run.sh via write_slurm_job. Returns the output path."""
    slurm_cfg = {
        'partition':     orch_partition,
        'ntasks':        1,
        'cpus_per_task': orch_cpus_per_task,
        'gpu':           None,
        'time':          orch_time or '48:00:00',
        'conda_env':     orch_conda_env,
        'cuda_version':  orch_cuda_version,
        'openmpi_ver':   orch_openmpi_ver,
        'ld_paths':      orch_ld_paths,
    }
    write_slurm_job(
        job_name=orch_job_name,
        slurm_config=slurm_cfg,
        out_path=out_sh,
        runner='python',
        script_path=out_py,
    )
    return out_sh


# ---------------------------------------------------------------------------
# Phase 4: Local analysis — called from neb_calculation.ipynb
# ---------------------------------------------------------------------------

def load_neb_results(neb_dir: str):
    """Load completed NEB barrier results into a DataFrame.

    Reads neb_pairs.json from neb_dir and calls summarise_neb on each job.
    Falls back to scanning neb_dir/*/neb_barrier.txt if the JSON is absent.
    Returns a pandas DataFrame sorted by E_abs (ascending), or an empty
    DataFrame if no completed jobs are found.
    """
    import glob
    import os

    pairs_json = os.path.join(neb_dir, 'neb_pairs.json')
    records = []

    if os.path.exists(pairs_json):
        import json
        with open(pairs_json) as f:
            neb_jobs = json.load(f)
        for job in neb_jobs:
            barrier_file = job.get('barrier_file', '')
            path_file    = job.get('path_file', '')
            if not os.path.exists(barrier_file):
                continue
            try:
                result = summarise_neb(barrier_file, path_file)
            except Exception:
                continue
            records.append({
                'is_label'    : job.get('is_true_label', ''),
                'fs_label1'   : job.get('fs_true_label1', ''),
                'fs_label2'   : job.get('fs_true_label2', ''),
                'graph_dist'  : job.get('graph_dist', -1),
                'n_grouped'   : job.get('n_dedup_group', 1),
                'E_abs'       : result['Ea'],
                'E_des'       : result['E_des'],
                'delta_E'     : result['delta_E'],
                'converged'   : result['converged'],
                'fmax_final'  : result['fmax_final'],
                'barrier_file': barrier_file,
                'path_file'   : path_file,
            })
    else:
        print(f'neb_pairs.json not found at {pairs_json}; scanning directory tree.')
        for bf in sorted(glob.glob(os.path.join(neb_dir, '*', 'neb_barrier.txt'))):
            pf = bf.replace('neb_barrier.txt', 'neb_path.dat')
            try:
                result = summarise_neb(bf, pf)
            except Exception:
                continue
            label = os.path.basename(os.path.dirname(bf))
            records.append({
                'is_label': label, 'fs_label1': '', 'fs_label2': '',
                'graph_dist': -1, 'n_grouped': 1,
                'E_abs': result['Ea'], 'E_des': result['E_des'],
                'delta_E': result['delta_E'], 'converged': result['converged'],
                'fmax_final': result['fmax_final'],
                'barrier_file': bf, 'path_file': pf,
            })

    import pandas as pd
    if records:
        barriers = pd.DataFrame(records).sort_values('E_abs').reset_index(drop=True)
        print(f'Loaded {len(barriers)} completed NEB results.')
        return barriers
    else:
        print('No completed NEB results found. Run the cluster jobs first.')
        return pd.DataFrame()


def plot_barrier_heatmap(barriers, neb_dir: str):
    """Plot mean E_a heatmap (IS x FS-pair chemical environment).

    Saves barrier_heatmap.png to neb_dir. Returns the saved path, or None if
    barriers is empty.
    """
    import matplotlib.pyplot as plt
    import os

    if barriers.empty:
        print('No data to plot.')
        return None

    df = barriers.copy()
    df['fs_pair'] = df.apply(lambda r: f"{r['fs_label1']}+{r['fs_label2']}", axis=1)
    pivot = df.pivot_table(index='is_label', columns='fs_pair', values='E_abs', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(min(0.8 * len(pivot.columns) + 2, 24),
                                    min(0.6 * len(pivot.index)  + 2, 18)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis_r')
    plt.colorbar(im, ax=ax, label='Mean E_a (eV)')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel('FS pair (label1 + label2)')
    ax.set_ylabel('IS label')
    ax.set_title('H2 Dissociation Barrier Heatmap — mean E_a (eV)')

    valid = pivot.values[~np.isnan(pivot.values)]
    mean_val = float(valid.mean()) if len(valid) else 0.0
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=5, color='white' if v < mean_val else 'black')

    fig.tight_layout()
    out_fig = os.path.join(neb_dir, 'barrier_heatmap.png')
    os.makedirs(neb_dir, exist_ok=True)
    plt.savefig(out_fig, dpi=150)
    plt.show()
    print(f'Saved: {out_fig}')
    return out_fig


def plot_mep_overlay(barriers, neb_dir: str):
    """Plot MEP overlay (all NEB paths; top-10 lowest barriers highlighted).

    Saves mep_overlay.png to neb_dir. Returns the saved path, or None if
    barriers is empty.
    """
    import matplotlib.pyplot as plt
    import os
    from models.parsers import parse_neb_path

    if barriers.empty:
        print('No data to plot.')
        return None

    top10_labels = set(barriers.head(10)['is_label'].tolist())
    fig, ax = plt.subplots(figsize=(8, 5))
    n_plotted = 0

    for _, row in barriers.iterrows():
        if not os.path.exists(row['path_file']):
            continue
        try:
            frac, _E_abs, dE_arr = parse_neb_path(row['path_file'])
        except Exception:
            continue

        is_top10 = row['is_label'] in top10_labels
        ax.plot(
            frac, dE_arr,
            color='tab:blue' if is_top10 else 'lightgrey',
            lw=1.5 if is_top10 else 0.5,
            alpha=1.0 if is_top10 else 0.4,
            zorder=5 if is_top10 else 1,
            label=f"{row['is_label']}  E_a={row['E_abs']:.2f} eV" if is_top10 else None,
        )
        n_plotted += 1

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Reaction coordinate')
    ax.set_ylabel('dE from IS (eV)')
    ax.set_title(f'NEB MEP Overlay — {n_plotted} paths  (top-10 lowest barriers highlighted)')
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:10], labels_[:10], fontsize=7, loc='upper right',
                  title='Top-10 lowest Ea')
    fig.tight_layout()
    out_fig = os.path.join(neb_dir, 'mep_overlay.png')
    os.makedirs(neb_dir, exist_ok=True)
    plt.savefig(out_fig, dpi=150)
    plt.show()
    print(f'Saved: {out_fig}')
    return out_fig
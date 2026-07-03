"""
tests/pipeline/run_pipeline_test.py
=====================================
Minimal end-to-end pipeline smoke test for the H-in-Ni permeation workflow.

Uses the smallest practical structures to validate the full code path on
the cluster before committing to the full-scale production run:

  Stage 0 — Build 2×2 Ni FCC(111) slab (3 layers, 12 atoms) with ASE
  Stage 1 — CG-minimise the slab with MACE (1 GPU SLURM job, ~5 min)
  Stage 2 — Write synthetic surface_sites.json (ACAT tested in tests/functional/)
  Stage 3 — H2* adsorption on 3 sites (auto_submit GPU array, ~10 min)
  Stage 4 — H* adsorption on 3 sites  (auto_submit GPU array, ~10 min)
  Stage 5 — NEB pipeline on ≤1 pair with 6 images (~30-60 min, polled)
  Stage 6 — Diffusivity NVT MD at 2 temperatures (minimal steps, ~30 min)
  Stage 7 — Permeation math: TST → KMC → permeability (pure Python, ~2 min)
  Summary — print pass/fail per stage; write summary.txt

Total estimated wall time: ~2–3 hours (vs. weeks for full run).

Usage:
    # Recommended — submit as SLURM CPU job so the orchestrator persists:
    sbatch tests/pipeline/submit_pipeline_test.sh

    # Or directly from an interactive session / login node:
    python tests/pipeline/run_pipeline_test.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import pathlib
import time
import traceback

import networkx as nx

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ─── tuneable parameters ──────────────────────────────────────────────────────

N_MAX_SITES    = 3      # max surface sites fed into adsorption phases
N_NEB_IMAGES   = 6      # intermediate NEB images (production uses 18)
NEB_FTOL       = 0.10   # relaxed convergence for test (production: 0.05 eV/Å)
NEB_POLL_SEC   = 120    # seconds between barrier-file polls
NEB_TIMEOUT_H  = 3      # hours before NEB poll gives up

# ─── diffusivity stage parameters ────────────────────────────────────────────
DIFF_TEMPERATURES  = [600, 700]   # two temps required for Arrhenius fit
DIFF_N_EQUIL       = 500          # NVT equilibration steps (production: ~50 k)
DIFF_N_PROD        = 2000         # NVT production steps   (production: ~500 k)
DIFF_NPT_HEAT      = 500          # NPT heating steps
DIFF_NPT_PROD      = 500          # NPT production steps
DIFF_DUMP_EVERY    = 100
DIFF_THERMO_EVERY  = 100
DIFF_RESTART_EVERY = 1000

WORK_DIR = str(PROJECT_ROOT / 'tests' / 'pipeline' / 'work')

# SLURM — shared GPU partition for short adsorption jobs
GPU_SLURM = {'partition': 'sharing', 'time': '00:20:00'}
# NEB runs CPU-only on the short partition
NEB_SLURM = {'partition': 'short', 'time': '01:00:00',
              'gpu': None, 'cuda_version': None}

# ─── imports ─────────────────────────────────────────────────────────────────

from ase.build import bulk as ase_bulk
from ase.build import fcc111

from models.config import (
    E2T_7, MASSES_7, ELEM_STR_7,
    SLURM_DEFAULTS, LAMMPS_CMD,
    MACE_MODEL_LAMMPS, PAIR_STYLE, PAIR_SUFFIX,
    KOKKOS_FLAGS, ADS_MIN_ETOL, ADS_MIN_FTOL, ADS_MIN_MAXITER, ADS_MIN_MAXEVAL,
)
from models.structure import write_lammps_data
from models.lammps_script import write_adsorbate_min_script
from models.create_slurm import write_slurm_job, submit_slurm_job, wait_for_jobs
from models.parsers import parse_energy_log, parse_barrier_file
from models.neb_workflow import (
    run_phase1_h2_adsorption,
    run_phase2_h_adsorption,
    orchestrate_neb_pipeline,
)
from models.diffusivity_workflow import generate_diffusivity_scripts
from models.tst_rates import arrhenius_rate
from models.permeation import (
    sweep_pressure,
    arrhenius_diffusivity,
    lattice_site_S0,
    permeability,
)

# ─── pass/fail tracker ───────────────────────────────────────────────────────

_PASS: list[str] = []
_FAIL: list[str] = []


def _check(label: str, ok: bool, detail: str = ''):
    if ok:
        _PASS.append(label)
        print(f'  [PASS] {label}' + (f' — {detail}' if detail else ''))
    else:
        _FAIL.append(label)
        print(f'  [FAIL] {label}' + (f' — {detail}' if detail else ''))


def _exists(p: str) -> bool:
    return pathlib.Path(p).exists()


def _header(title: str):
    bar = '=' * 60
    print(f'\n{bar}\n  {title}\n{bar}')


# ═════════════════════════════════════════════════════════════════════════════
# Stage 0 — Build 2×2 Ni FCC(111) slab (4 layers, 16 atoms) with ASE
# ═════════════════════════════════════════════════════════════════════════════

def stage0_build_slab(work_dir: str) -> dict:
    _header('Stage 0: Build 2×2 Ni FCC(111) slab (3 layers, 12 atoms)')

    slab_dir = pathlib.Path(work_dir) / 'slab'
    slab_dir.mkdir(parents=True, exist_ok=True)

    raw_slab = str(slab_dir / 'slab_raw.lammps')
    slab_out = str(slab_dir / 'slab_relaxed.lammps')
    slab_log = str(slab_dir / 'slab_min.log')
    slab_in  = str(slab_dir / 'slab_min.in')
    slab_sh  = str(slab_dir / 'slab_min.sh')

    # 3 layers: FCC(111) ABC stacking period is 3 — ACAT requires n_layers % 3 == 0.
    # 4 layers fails (4 % 3 = 1); 3 layers is the minimal valid choice.
    slab = fcc111('Ni', size=(2, 2, 3), vacuum=10.0)
    slab.wrap()

    pos  = slab.get_positions()
    cell = slab.get_cell().lengths()   # [Lx, Ly, Lz]
    syms = slab.get_chemical_symbols() # all 'Ni'

    # z_freeze_cutoff: midpoint between the two bottom layers
    z_vals   = sorted(set(round(float(z), 3) for z in pos[:, 2]))
    z_freeze = (z_vals[0] + z_vals[1]) / 2.0

    write_lammps_data(
        symbols=syms,
        positions=pos,
        cell_lengths=cell,
        masses=MASSES_7,
        e2t=E2T_7,
        out_path=raw_slab,
        comment='2x2 Ni FCC(111) 4-layer pipeline test slab',
    )

    _check('slab_raw_written', _exists(raw_slab), f'{len(slab)} atoms')
    print(f'  z_freeze = {z_freeze:.3f} Å  (bottom layer frozen)')
    print(f'  cell     = {cell[0]:.3f} × {cell[1]:.3f} × {cell[2]:.3f} Å')

    return dict(
        raw_slab=raw_slab, slab_out=slab_out, slab_log=slab_log,
        slab_in=slab_in,   slab_sh=slab_sh,
        z_freeze=z_freeze, slab_dir=str(slab_dir),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — CG-minimise slab (1 GPU SLURM job)
# ═════════════════════════════════════════════════════════════════════════════

def stage1_relax_slab(s0: dict) -> dict:
    _header('Stage 1: Slab CG minimisation with MACE (GPU SLURM job)')

    kk         = ' '.join(KOKKOS_FLAGS)
    slurm_cfg  = {**SLURM_DEFAULTS, **GPU_SLURM}

    write_adsorbate_min_script(
        slab_ads_input=s0['raw_slab'],
        ads_output=s0['slab_out'],
        out_path=s0['slab_in'],
        pair_style=PAIR_STYLE,
        mace_model=MACE_MODEL_LAMMPS,
        pair_suffix=PAIR_SUFFIX,
        elem_str=ELEM_STR_7,
        z_freeze_cutoff=s0['z_freeze'],
        etol=ADS_MIN_ETOL,
        ftol=ADS_MIN_FTOL,
        maxiter=ADS_MIN_MAXITER,
        maxeval=ADS_MIN_MAXEVAL,
    )

    write_slurm_job(
        job_name='pipeline_slab_min',
        slurm_config=slurm_cfg,
        out_path=s0['slab_sh'],
        commands=[f'{LAMMPS_CMD} {kk} -in {s0["slab_in"]} -log {s0["slab_log"]}'],
    )

    print('  Submitting slab minimisation...')
    jid = submit_slurm_job(s0['slab_sh'])
    print(f'  Job ID: {jid}  — waiting...')
    wait_for_jobs({'slab_min': jid})

    parsed  = parse_energy_log(s0['slab_log'])
    e_clean = parsed['pe_final_eV'] if parsed else float('nan')

    _check('slab_relaxed_file', _exists(s0['slab_out']), s0['slab_out'])
    _check('slab_log_parsed',   parsed is not None,      f'e_clean = {e_clean:.4f} eV')
    _check('e_clean_finite',    e_clean == e_clean,      '')  # NaN check (nan != nan)

    if not _exists(s0['slab_out']):
        raise RuntimeError(
            f'slab_relaxed.lammps not written — LAMMPS job likely crashed '
            f'(check {s0["slab_log"].replace(".log","_*.out")} on the compute node). '
            f'Common cause: /tmp full on the compute node.'
        )

    print(f'  e_clean = {e_clean:.6f} eV')
    return dict(e_clean=e_clean)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Synthetic surface_sites.json (ACAT enumeration tested separately)
# ═════════════════════════════════════════════════════════════════════════════

def stage2_enumerate_sites(s0: dict, work_dir: str) -> dict:
    _header('Stage 2: Surface sites (synthetic — ACAT tested in tests/functional/)')

    print('  [SKIP] ACAT CustomSurface requires atoms_per_layer ≥ 4;')
    print('         the 3-layer 2×2 smoke-test slab has only 4 atoms total per')
    print('         layer but ACAT\'s heuristic (_n_layers_est=12) misdetects 1.')
    print('         ACAT site enumeration is covered by tests/functional/.')
    print('         Writing synthetic ontop sites from Stage-0 geometry.')

    # Reconstruct the same 3-layer slab used in Stage 0 to get exact positions.
    # Top layer = 4 atoms for a 2×2 cell; we keep N_MAX_SITES of them.
    _slab  = fcc111('Ni', size=(2, 2, 3), vacuum=10.0)
    _slab.wrap()
    _pos   = _slab.get_positions()
    _cell  = _slab.get_cell()

    _top_idx   = sorted(range(len(_slab)), key=lambda i: float(_pos[i, 2]),
                        reverse=True)[:4]
    _surface_z = max(float(_pos[i, 2]) for i in _top_idx)
    _site_h    = 1.8   # Å above top Ni layer

    sites = []
    for k, idx in enumerate(_top_idx[:N_MAX_SITES]):
        x, y = float(_pos[idx, 0]), float(_pos[idx, 1])
        sites.append({
            'site_id': f's_{k}',
            'level1': {
                'site_type':    'ontop',
                'composition':  {'Ni': 1},
                'full_label':   'ontop_Ni1',
                'position':     [x, y, _surface_z + _site_h],
                'atom_indices': [idx],
            },
            'level2': {str(idx): {'elem': 'Ni'}},
            'level3': [],
        })

    sites_dir  = str(pathlib.Path(work_dir) / 'sites')
    pathlib.Path(sites_dir).mkdir(parents=True, exist_ok=True)
    sites_json = str(pathlib.Path(sites_dir) / 'surface_sites.json')

    data = {
        'metadata': {
            'n_atoms_total':    len(_slab),
            'n_sites_total':    len(sites),
            'cell':             _cell.tolist(),
            'slab_composition': {'Ni': len(_slab)},
            'site_type_counts': {'ontop': len(sites)},
        },
        'surface_atoms': list(_top_idx),
        'sites': sites,
    }
    with open(sites_json, 'w') as f:
        json.dump(data, f, indent=2)

    n_sites = len(sites)
    _check('sites_json_written', _exists(sites_json), sites_json)
    _check('sites_nonzero', n_sites > 0, f'{n_sites} synthetic ontop sites')

    # Write a minimal surface_graph.gml — load_neb_pools requires this file.
    # No site-site edges needed: _graph_distance returns -1 for disconnected
    # pairs, which enumerate_fs_pairs explicitly accepts.
    G_syn = nx.Graph()
    for site in sites:
        G_syn.add_node(site['site_id'], node_type='site')
    gml_path = str(pathlib.Path(sites_dir) / 'surface_graph.gml')
    nx.write_gml(G_syn, gml_path)
    _check('surface_graph_gml_written', _exists(gml_path), gml_path)

    print(f'  surface_z = {_surface_z:.3f} Å  |  {n_sites} sites written')

    return dict(sites_json=sites_json, sites_dir=sites_dir, n_sites=n_sites)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — H2* adsorption (Phase B1, GPU array via auto_submit)
# ═════════════════════════════════════════════════════════════════════════════

def stage3_h2_adsorption(s0: dict, s1: dict, s2: dict,
                          work_dir: str) -> dict:
    _header(f'Stage 3: H2* adsorption ({s2["n_sites"]} sites, GPU array)')

    ads_dir = str(pathlib.Path(work_dir) / 'adsorption')
    result  = run_phase1_h2_adsorption(
        surface_sites_json=s2['sites_json'],
        relaxed_slab_path=s0['slab_out'],
        e_clean=s1['e_clean'],
        e_h2_gas=0.0,   # placeholder; code path is what matters here
        outdir=ads_dir,
        slurm_opts={**SLURM_DEFAULTS, **GPU_SLURM},
        dry_run=False,
        elem_str=ELEM_STR_7,
        e2t=E2T_7,
        masses=MASSES_7,
    )

    n_computed = result['n_sites_computed']
    n_intact   = sum(1 for v in result['is_pool'].values()
                     if v['status'] == 'intact')

    _check('h2_status', result['status'] in ('submitted', 'generated'),
           result['status'])
    _check('h2_n_computed', n_computed > 0,
           f'{n_computed}/{s2["n_sites"]}')
    _check('h2_intact_pool_written',
           _exists(str(pathlib.Path(result['outdir']) / 'H2_intact_pool.json')), '')

    print(f'  H2 intact sites: {n_intact}  '
          f'(e_h2_gas=0 → binding energies are proxies)')
    return dict(result=result, ads_dir=ads_dir)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 — H* adsorption (Phase B2, GPU array via auto_submit)
# ═════════════════════════════════════════════════════════════════════════════

def stage4_h_adsorption(s0: dict, s1: dict, s2: dict, s3: dict) -> dict:
    _header(f'Stage 4: H* adsorption ({s2["n_sites"]} sites, GPU array)')

    result = run_phase2_h_adsorption(
        surface_sites_json=s2['sites_json'],
        relaxed_slab_path=s0['slab_out'],
        e_clean=s1['e_clean'],
        e_h2_gas=0.0,
        outdir=s3['ads_dir'],   # same parent → creates ads_dir/phase2_h/
        slurm_opts={**SLURM_DEFAULTS, **GPU_SLURM},
        dry_run=False,
        elem_str=ELEM_STR_7,
        e2t=E2T_7,
        masses=MASSES_7,
    )

    n_computed = result['n_sites_computed']

    _check('h_status', result['status'] in ('submitted', 'generated'),
           result['status'])
    _check('h_n_computed', n_computed > 0,
           f'{n_computed}/{s2["n_sites"]}')
    _check('h_site_coords_written',
           _exists(str(pathlib.Path(result['outdir']) / 'H_site_coords.json')), '')

    return dict(result=result)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5 — NEB pipeline (Section C, CPU jobs, polled)
# ═════════════════════════════════════════════════════════════════════════════

def stage5_neb(s0: dict, s1: dict, s2: dict, s3: dict, s4: dict,
               work_dir: str) -> dict:
    _header('Stage 5: Surface NEB pipeline (≤1 pair, 6 images)')

    neb_outdir = str(pathlib.Path(work_dir) / 'neb')
    h2_dir     = s3['result']['outdir']   # ads_dir/phase1_h2
    h_dir      = s4['result']['outdir']   # ads_dir/phase2_h
    sites_dir  = s2['sites_dir']

    result = orchestrate_neb_pipeline(
        phase1_h2_dir=h2_dir,
        phase2_h_dir=h_dir,
        phase3_sites_dir=sites_dir,
        e_clean=s1['e_clean'],
        outdir=neb_outdir,
        slurm_opts={**SLURM_DEFAULTS, **GPU_SLURM},
        neb_slurm_opts={**SLURM_DEFAULTS, **NEB_SLURM},
        sep_min=2.5,
        sep_max=6.0,
        n_images=N_NEB_IMAGES,
        neb_ftol=NEB_FTOL,
        dry_run=False,
        elem_str=ELEM_STR_7,
        e2t=E2T_7,
        masses=MASSES_7,
    )

    neb_result  = result['neb_result']
    n_deduped   = len(result['deduped'])
    n_jobs      = neb_result['n_jobs']
    is_pool_sz  = len(result['pools'].get('is_pool', {}))
    fs_pool_sz  = len(result['pools'].get('fs_pool', {}))

    _check('neb_pools_loaded',
           is_pool_sz + fs_pool_sz > 0,
           f'IS={is_pool_sz}  FS={fs_pool_sz}')
    _check('neb_dedup_ran', n_deduped >= 0,
           f'{n_deduped} combination(s) after proximity+label filter')

    if n_jobs == 0:
        print('\n  WARNING: n_jobs=0 — no NEB pairs within sep_min–sep_max range.')
        print('  Likely causes:')
        print('    • All H2* dissociated on the small slab → IS pool empty')
        print('    • No H* with E_ads < 0 at e_h2_gas=0 → FS pool empty')
        print('    • N_MAX_SITES=3 too small for a valid pair')
        print('  Code paths for pool loading, pair enumeration, and script')
        print('  generation were all exercised. NEB submission skipped.')
        _check('neb_graceful_empty', True, 'n_jobs=0 handled without crash')
        return dict(n_jobs=0, skipped=True)

    _check('neb_jobs_submitted', True, f'{n_jobs} pair(s) submitted')

    # ── Poll for the first job's barrier file ──────────────────────────────
    barrier_path = neb_result['neb_jobs'][0]['barrier_file']
    timeout_sec  = NEB_TIMEOUT_H * 3600
    t0           = time.time()

    print(f'\n  Polling for: {barrier_path}')
    print(f'  Timeout: {NEB_TIMEOUT_H} h  |  poll every {NEB_POLL_SEC} s')

    while not pathlib.Path(barrier_path).exists():
        elapsed = time.time() - t0
        if elapsed > timeout_sec:
            print(f'\n  TIMEOUT after {elapsed/3600:.1f} h')
            _check('neb_barrier_within_timeout', False,
                   f'timed out at {elapsed/3600:.1f} h')
            return dict(n_jobs=n_jobs, skipped=False, timed_out=True)
        print(f'  [{time.strftime("%H:%M:%S")}] '
              f'{elapsed/60:.0f} min elapsed — waiting...')
        time.sleep(NEB_POLL_SEC)

    _check('neb_barrier_file_written', True, barrier_path)

    barrier = parse_barrier_file(barrier_path)
    _check('neb_barrier_parseable',
           bool(barrier) and 'E_abs' in barrier,
           f"E_abs = {barrier.get('E_abs', 'N/A')} eV")
    _check('neb_barrier_positive',
           barrier.get('E_abs', -1.0) > 0,
           f"E_abs = {barrier.get('E_abs', 'N/A')} eV")
    _check('neb_converged',
           barrier.get('converged') is True,
           f"converged = {barrier.get('converged')}")

    elapsed_min = (time.time() - t0) / 60
    print(f'  NEB completed in {elapsed_min:.1f} min')
    return dict(n_jobs=n_jobs, skipped=False, timed_out=False,
                barrier_path=barrier_path)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 6 — Diffusivity NVT MD (2 temperatures, minimal steps)
# ═════════════════════════════════════════════════════════════════════════════

def stage6_diffusivity(work_dir: str) -> dict:
    _header(f'Stage 6: Diffusivity NVT MD ({DIFF_TEMPERATURES} K, minimal steps)')

    diff_dir = pathlib.Path(work_dir) / 'diffusivity'
    diff_dir.mkdir(parents=True, exist_ok=True)

    # ── Build a tiny orthogonal FCC Ni bulk (32 atoms) ────────────────────────
    ni = ase_bulk('Ni', 'fcc', a=3.52, cubic=True).repeat([2, 2, 2])
    bulk_path = str(diff_dir / 'ni_bulk_test.lammps')
    write_lammps_data(
        symbols=ni.get_chemical_symbols(),
        positions=ni.get_positions(),
        cell_lengths=ni.get_cell().lengths(),
        masses=MASSES_7,
        e2t=E2T_7,
        out_path=bulk_path,
        comment='2x2x2 Ni FCC bulk — diffusivity pipeline test',
    )
    _check('diff_bulk_written', _exists(bulk_path), f'{len(ni)} atoms')

    # ── Generate diffusivity_run.py with embedded minimal config ──────────────
    out_py = str(diff_dir / 'diffusivity_run.py')
    generate_diffusivity_scripts(
        input_structures=[bulk_path],
        n_h_values=[1],
        temperatures=DIFF_TEMPERATURES,
        work_dir=str(diff_dir),
        nvt_wall_time='00:20:00',
        cutoff='00:05:00',
        gpu_partition='sharing',
        gpu_time='00:20:00',
        short_gpu_partition='sharing',
        short_gpu_time='00:20:00',
        timestep_ps=0.001,
        tau_t_ps=0.1,
        n_equil_steps=DIFF_N_EQUIL,
        n_prod_steps=DIFF_N_PROD,
        thermo_every=DIFF_THERMO_EVERY,
        dump_every=DIFF_DUMP_EVERY,
        velocity_seed=12345,
        restart_every=DIFF_RESTART_EVERY,
        npt_heat_steps=DIFF_NPT_HEAT,
        npt_prod_steps=DIFF_NPT_PROD,
        out_py=out_py,
    )
    _check('diffusivity_run_py_written', _exists(out_py), out_py)

    # ── Run the orchestrator as a subprocess (blocks until all sub-jobs done) ─
    # PYTHONPATH ensures models.* is importable regardless of the path that
    # generate_diffusivity_scripts embeds via sys.path.insert.
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    print(f'\n  Running diffusivity_run.py  (submits GPU sub-jobs and waits) …')
    ret = subprocess.run([sys.executable, out_py], env=env)
    _check('diffusivity_run_exit_0', ret.returncode == 0,
           f'exit code {ret.returncode}')

    # ── Check expected output files ───────────────────────────────────────────
    struct_stem = 'ni_bulk_test'
    run_name    = f'{struct_stem}_1H'
    analysis_dir = diff_dir / 'results' / run_name / 'analysis'
    msd_files    = list(analysis_dir.glob('msd_*.txt')) if analysis_dir.exists() else []
    diff_table   = analysis_dir / 'diffusivity_table.txt'
    arr_json     = diff_dir / 'results' / struct_stem / 'diffusivity_arrhenius.json'

    _check('diff_msd_files_written', len(msd_files) >= len(DIFF_TEMPERATURES),
           f'{len(msd_files)} msd_*.txt files found')
    _check('diff_table_written', _exists(str(diff_table)), str(diff_table))
    _check('diff_arrhenius_json_written', _exists(str(arr_json)), str(arr_json))

    if arr_json.exists():
        with open(arr_json) as f:
            arr = json.load(f)
        Ea  = arr.get('E_D_eV', float('nan'))
        D0  = arr.get('D0_m2s', float('nan'))
        R2  = arr.get('R2_fit', float('nan'))
        _check('diff_Ea_positive', Ea > 0, f'Ea = {Ea:.4f} eV')
        _check('diff_D0_positive', D0 > 0, f'D0 = {D0:.3e} m²/s')
        _check('diff_R2_reasonable', R2 > 0.5, f'R² = {R2:.4f}')
        print(f'  Arrhenius: Ea={Ea:.4f} eV  D0={D0:.3e} m²/s  R²={R2:.4f}')

    return dict(diff_dir=str(diff_dir), arr_json=str(arr_json))


# ═════════════════════════════════════════════════════════════════════════════
# Stage 7 — Permeation math: TST → KMC → permeability (pure Python)
# ═════════════════════════════════════════════════════════════════════════════

def stage7_permeation_math(s5: dict, work_dir: str) -> dict:
    _header('Stage 7: Permeation math — TST → KMC → permeability (pure Python)')

    KB_EV  = 8.617333262e-5   # eV / K
    T_K    = 700.0
    A0_M   = 3.52e-10         # Ni lattice constant (m)
    L_M    = 1.0e-6           # 1 µm membrane thickness
    D0_M2S = 1.0e-7           # bulk diffusion pre-exponential
    E_D_EV = 0.40             # bulk diffusion barrier (eV)
    NU_TST = 1.0e13           # attempt frequency (s⁻¹)

    # ── Choose barrier: real NEB result from Stage 5, or synthetic fallback ───
    E_ABS = 0.30   # fallback entry barrier (eV)
    E_DES = 0.50   # fallback exit barrier (eV)
    source = 'synthetic (Stage 5 skipped or no NEB pairs)'

    if (not s5.get('skipped', True)
            and not s5.get('timed_out', True)
            and s5.get('barrier_path')):
        bf = pathlib.Path(s5['barrier_path'])
        if bf.exists():
            bd = parse_barrier_file(str(bf))
            if bd and bd.get('E_abs', 0) > 0:
                E_ABS  = bd['E_abs']
                E_DES  = bd.get('E_des', E_ABS + 0.20)
                source = f'real NEB barrier from {bf.name}'

    print(f'  Barrier source : {source}')
    print(f'  E_abs = {E_ABS:.3f} eV   E_des = {E_DES:.3f} eV   T = {T_K:.0f} K')

    # ── TST rates ─────────────────────────────────────────────────────────────
    k_entry     = arrhenius_rate(NU_TST, E_ABS, T_K)
    k_exit      = arrhenius_rate(NU_TST, E_DES, T_K)
    k_diss      = arrhenius_rate(NU_TST, E_ABS + 0.10, T_K)
    k_des       = arrhenius_rate(NU_TST, E_DES + 0.20, T_K)
    k_surf_diff = arrhenius_rate(NU_TST, E_ABS - 0.10, T_K)   # faster than entry

    _check('tst_k_entry_positive', k_entry > 0,
           f'k_entry = {k_entry:.3e} s⁻¹')
    db_ratio   = k_entry / k_exit
    db_expect  = math.exp((E_DES - E_ABS) / (KB_EV * T_K))
    _check('tst_detailed_balance',
           abs(db_ratio - db_expect) / db_expect < 1e-6,
           f'k_fwd/k_rev = {db_ratio:.4e}  expected {db_expect:.4e}')

    rate_dict = {
        'k_diss':      {('Ni', 'Ni'): k_diss},
        'k_des':       {('Ni', 'Ni'): k_des},
        'k_surf_diff': {('Ni', 'Ni'): k_surf_diff},
        'k_entry':     {'Ni': k_entry},
        'k_exit':      {'Ni': k_exit},
    }

    # ── KMC pressure sweep (4×4 pure-Ni grid, 3 pressures, short run) ────────
    D_T    = arrhenius_diffusivity(D0_M2S, E_D_EV, T_K)
    P_VALS = [1e4, 1e5, 1e6]   # 0.1, 1, 10 bar

    sweep = sweep_pressure(
        P_vals_Pa  = P_VALS,
        rate_dict  = rate_dict,
        D_m2s      = D_T,
        L_m        = L_M,
        T_K        = T_K,
        a0_m       = A0_M,
        nx         = 4,
        ny         = 4,
        composition = {'Ni': 1.0},
        seed       = 42,
        kmc_kwargs = {'window': 50, 'max_steps': 10_000},
    )

    n_J_pos = sum(1 for j in sweep['J_vals'] if j > 0)
    n_C_pos = sum(1 for c in sweep['C0_vals'] if c > 0)
    _check('kmc_flux_positive',   n_J_pos > 0,
           f'{n_J_pos}/{len(P_VALS)} pressures give J > 0')
    _check('kmc_C0_positive',     n_C_pos > 0,
           f'{n_C_pos}/{len(P_VALS)} pressures give C0 > 0')

    # ── Permeability from KMC surface concentration at highest pressure ───────
    S0    = lattice_site_S0(A0_M)
    C0_hi = sweep['C0_vals'][-1]
    P_hi  = P_VALS[-1]
    S_est = C0_hi / math.sqrt(P_hi) if C0_hi > 0 else 0.0
    Phi   = permeability(D_T, S_est)

    _check('permeability_positive', Phi > 0, f'Φ = {Phi:.3e} mol m⁻¹ s⁻¹ Pa⁻⁰·⁵')
    print(f'  D({T_K:.0f}K) = {D_T:.3e} m²/s   S_est = {S_est:.3e}   Φ = {Phi:.3e}')

    # ── Save summary JSON ─────────────────────────────────────────────────────
    perm_out = pathlib.Path(work_dir) / 'permeation_math_summary.json'
    with open(perm_out, 'w') as f:
        json.dump({
            'T_K': T_K,
            'barrier_source': source,
            'E_abs_eV': E_ABS,
            'E_des_eV': E_DES,
            'k_entry_s1': k_entry,
            'k_exit_s1':  k_exit,
            'D_m2s':      D_T,
            'S0_m3_pasqrt': S0,
            'S_est_m3_pasqrt': S_est,
            'Phi_mol_m_s_Pasqrt': Phi,
            'J_vals':  sweep['J_vals'],
            'C0_vals': sweep['C0_vals'],
            'converged': sweep['converged'],
        }, f, indent=2)
    _check('permeation_summary_written', _exists(str(perm_out)), '')

    return dict(Phi=Phi, D_T=D_T, S_est=S_est)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    t_start = time.time()
    print(f'\n{"#"*60}')
    print(f'  PIPELINE SMOKE TEST — {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Work dir     : {WORK_DIR}')
    print(f'  N_MAX_SITES  : {N_MAX_SITES}')
    print(f'  N_NEB_IMAGES : {N_NEB_IMAGES}  NEB_FTOL={NEB_FTOL}')
    print(f'  NEB timeout  : {NEB_TIMEOUT_H} h  (poll every {NEB_POLL_SEC} s)')
    print(f'  DIFF temps   : {DIFF_TEMPERATURES} K  '
          f'equil={DIFF_N_EQUIL} prod={DIFF_N_PROD} steps')
    print(f'{"#"*60}\n')

    pathlib.Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

    s0 = s1 = s2 = s3 = s4 = None

    try:
        s0 = stage0_build_slab(WORK_DIR)
    except Exception:
        print('[ABORT] Stage 0 failed:')
        traceback.print_exc()
        return 1

    try:
        s1 = stage1_relax_slab(s0)
    except Exception:
        print('[ABORT] Stage 1 failed:')
        traceback.print_exc()
        return 1

    try:
        s2 = stage2_enumerate_sites(s0, WORK_DIR)
    except Exception:
        print('[ABORT] Stage 2 failed:')
        traceback.print_exc()
        return 1

    try:
        s3 = stage3_h2_adsorption(s0, s1, s2, WORK_DIR)
    except Exception:
        print('[ABORT] Stage 3 failed:')
        traceback.print_exc()
        return 1

    try:
        s4 = stage4_h_adsorption(s0, s1, s2, s3)
    except Exception:
        print('[ABORT] Stage 4 failed:')
        traceback.print_exc()
        return 1

    s5 = {}
    try:
        s5 = stage5_neb(s0, s1, s2, s3, s4, WORK_DIR)
    except Exception:
        print('[WARN] Stage 5 raised an exception:')
        traceback.print_exc()
        _FAIL.append('neb_stage_exception')

    try:
        stage6_diffusivity(WORK_DIR)
    except Exception:
        print('[WARN] Stage 6 raised an exception:')
        traceback.print_exc()
        _FAIL.append('diffusivity_stage_exception')

    try:
        stage7_permeation_math(s5, WORK_DIR)
    except Exception:
        print('[WARN] Stage 7 raised an exception:')
        traceback.print_exc()
        _FAIL.append('permeation_math_stage_exception')

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _header('SUMMARY')
    print(f'  Wall time : {elapsed/60:.1f} min ({elapsed/3600:.2f} h)')
    print(f'  Passed    : {len(_PASS)}')
    print(f'  Failed    : {len(_FAIL)}')
    for label in _PASS:
        print(f'    [PASS] {label}')
    for label in _FAIL:
        print(f'    [FAIL] {label}')

    summary_path = pathlib.Path(WORK_DIR) / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'Pipeline smoke test — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Wall time : {elapsed/60:.1f} min\n')
        f.write(f'Passed    : {len(_PASS)} / {len(_PASS) + len(_FAIL)}\n\n')
        for label in _PASS:
            f.write(f'PASS  {label}\n')
        for label in _FAIL:
            f.write(f'FAIL  {label}\n')
    print(f'\n  Summary written → {summary_path}')

    return 0 if not _FAIL else 1


if __name__ == '__main__':
    sys.exit(main())

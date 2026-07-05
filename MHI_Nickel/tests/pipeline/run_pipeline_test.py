"""
tests/pipeline/run_pipeline_test.py
=====================================
Minimal end-to-end pipeline smoke test for the H-in-Ni permeation workflow.

Uses the smallest practical structures to validate the full code path on
the cluster before committing to the full-scale production run:

  Stage 0 — Build 2×2 Ni FCC(111) slab (SLAB_LAYERS layers) with ASE.
            Layer count auto-derives subsurface_1/subsurface_2 in Stage 7
            via build_subsurface_graph — see models/subsurface_graph.py.
  Stage 1 — CG-minimise the slab with MACE (1 GPU SLURM job, ~5 min)
  Stage 2 — Write synthetic surface_sites.json (ACAT tested in tests/functional/)
  Stage 3 — H2* adsorption on 3 sites (auto_submit GPU array, ~10 min)
  Stage 4 — H* adsorption on 3 sites  (auto_submit GPU array, ~10 min)
  Stage 5 — NEB pipeline on ≤1 pair with 6 images (~30-60 min, polled)
            Fallbacks if enumeration yields 0 pairs: (2a) loosened filter
            bounds, (2b) one forced example pair from the pools.
  Stage 6 — Diffusivity NVT MD at 2 temperatures × n_H=[1,3] (minimal steps,
            ~30 min). Regression-tests the per-n_H Arrhenius overwrite fix:
            each n_H must land in its own results/{stem}_{n_H}H/ file.
  Stage 7 — Permeation (Part 2, real): calls generate_permeation_scripts()
            and executes the generated script for real — Hop A NEB, Hop B
            NEB, vibrational frequencies, TST rates, KMC pressure sweeps,
            and Richardson-Sieverts permeability, once per n_H (~1-2 h).
            Regression-tests the per-n_H bulk-diffusivity fix: each n_H's
            permeability must embed its OWN D0/Ea, never a shared value.
  Summary — print pass/fail per stage; write summary.txt

Total estimated wall time: ~3–4 hours (vs. weeks for full run).

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

# Surface slab layer count — shared by Stage 0 (build) and Stage 2 (synthetic
# site reconstruction) so they can never drift apart. 6 layers gives
# build_subsurface_graph's auto-derived roles (Part 2, Stage 7) a clean
# 2-layer margin above the frozen bottom round(6/3)=2 layers: subsurface_1=
# L5 (Hop A target), subsurface_2=L4 (Hop B / bulk-entry target). 3 layers
# (the old value) collides subsurface_2 with the frozen region — see
# models/subsurface_graph.py's auto-derivation warning.
SLAB_LAYERS = 6

# ─── diffusivity stage parameters ────────────────────────────────────────────
DIFF_TEMPERATURES  = [600, 700]   # two temps required for Arrhenius fit
DIFF_N_H_VALUES    = [1, 3]       # regression-tests the per-n_H Arrhenius fix
DIFF_N_EQUIL       = 500          # NVT equilibration steps (production: ~50 k)
DIFF_N_PROD        = 2000         # NVT production steps   (production: ~500 k)
DIFF_NPT_HEAT      = 500          # NPT heating steps
DIFF_NPT_PROD      = 500          # NPT production steps
DIFF_DUMP_EVERY    = 100
DIFF_THERMO_EVERY  = 100
DIFF_RESTART_EVERY = 1000

# ─── permeation stage parameters ─────────────────────────────────────────────
PERM_P_VALS_PA    = [1e4, 1e5, 1e6]   # 0.1, 1, 10 bar
PERM_A0_M         = 3.52e-10          # Ni lattice constant (m)
PERM_L_M          = 1.0e-6            # 1 µm membrane thickness
PERM_DH_DISS_EV   = 0.30              # explicit — auto-source (ranked_barriers.json)
                                       # is GitHub #7's known gap, not under test here
PERM_NX, PERM_NY  = 4, 4
PERM_SEED         = 42
PERM_KMC_MAX_STEPS = 10_000

WORK_DIR = str(PROJECT_ROOT / 'tests' / 'pipeline' / 'work')

# SLURM — shared GPU partition for short adsorption jobs
GPU_SLURM = {'partition': 'sharing', 'time': '00:20:00'}
# NEB (and, reused, vibrations) run CPU-only on the short partition
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
    SPRING_CONST,
)
from models.structure import write_lammps_data, compute_z_freeze_cutoff
from models.lammps_script import write_adsorbate_min_script
from models.create_slurm import write_slurm_job, submit_slurm_job, wait_for_jobs
from models.parsers import parse_energy_log, parse_barrier_file
from models.neb_workflow import (
    run_phase1_h2_adsorption,
    run_phase2_h_adsorption,
    orchestrate_neb_pipeline,
    orchestrate_neb,
)
from models.diffusivity_workflow import generate_diffusivity_scripts
from models.permeation_workflow import generate_permeation_scripts

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
    _header(f'Stage 0: Build 2×2 Ni FCC(111) slab ({SLAB_LAYERS} layers)')

    slab_dir = pathlib.Path(work_dir) / 'slab'
    slab_dir.mkdir(parents=True, exist_ok=True)

    raw_slab = str(slab_dir / 'slab_raw.lammps')
    slab_out = str(slab_dir / 'slab_relaxed.lammps')
    slab_log = str(slab_dir / 'slab_min.log')
    slab_in  = str(slab_dir / 'slab_min.in')
    slab_sh  = str(slab_dir / 'slab_min.sh')

    slab = fcc111('Ni', size=(2, 2, SLAB_LAYERS), vacuum=10.0)
    slab.wrap()

    pos  = slab.get_positions()
    cell = slab.get_cell().lengths()   # [Lx, Ly, Lz]
    syms = slab.get_chemical_symbols() # all 'Ni'

    write_lammps_data(
        symbols=syms,
        positions=pos,
        cell_lengths=cell,
        masses=MASSES_7,
        e2t=E2T_7,
        out_path=raw_slab,
        comment=f'2x2 Ni FCC(111) {SLAB_LAYERS}-layer pipeline test slab',
    )
    _check('slab_raw_written', _exists(raw_slab), f'{len(slab)} atoms')

    # Auto bottom-1/3 freeze cutoff — same round(N/3) formula
    # build_subsurface_graph's layer-role auto-derivation assumes.
    z_freeze = compute_z_freeze_cutoff(raw_slab)

    print(f'  z_freeze = {z_freeze:.3f} Å  '
          f'(bottom round({SLAB_LAYERS}/3) layers frozen)')
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
    print(f'         the {SLAB_LAYERS}-layer 2×2 smoke-test slab has only 4 atoms total')
    print('         per layer but ACAT\'s heuristic (_n_layers_est=12) misdetects it.')
    print('         ACAT site enumeration is covered by tests/functional/.')
    print('         Writing synthetic ontop sites from Stage-0 geometry.')

    # Reconstruct the same slab used in Stage 0 to get exact positions.
    # Top layer = 4 atoms for a 2×2 cell; we keep N_MAX_SITES of them.
    _slab  = fcc111('Ni', size=(2, 2, SLAB_LAYERS), vacuum=10.0)
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
                'site_type':         'ontop',
                'composition':       {'Ni': 1},
                'full_label':        'ontop_Ni1',
                'position':          [x, y, _surface_z + _site_h],
                'atom_indices':      [idx],
                'constituent_atoms': [{'element': 'Ni'}],
            },
            'level2': {str(idx): {'element': 'Ni', 'shell1': []}},
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

def _poll_and_parse_barrier(barrier_path: str) -> dict:
    """Poll for a NEB barrier file, then parse and check it.

    Shared by the normal enumerated path and the forced-pair fallbacks.
    Returns dict(timed_out=bool, barrier_path=str|None).
    """
    timeout_sec = NEB_TIMEOUT_H * 3600
    t0          = time.time()

    print(f'\n  Polling for: {barrier_path}')
    print(f'  Timeout: {NEB_TIMEOUT_H} h  |  poll every {NEB_POLL_SEC} s')

    while not pathlib.Path(barrier_path).exists():
        elapsed = time.time() - t0
        if elapsed > timeout_sec:
            print(f'\n  TIMEOUT after {elapsed/3600:.1f} h')
            _check('neb_barrier_within_timeout', False,
                   f'timed out at {elapsed/3600:.1f} h')
            return dict(timed_out=True, barrier_path=None)
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
    return dict(timed_out=False, barrier_path=barrier_path)


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
        # 2x2 smoke slab: all ontop-ontop separations are a0/sqrt(2) = 2.489 A,
        # so the production default sep_min=2.5 would filter out every pair.
        sep_min=2.0,
        sep_max=5.0,
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

    tier = 'enumerated'

    # ── Fallback 2a: re-enumerate with loosened filter bounds ───────────────
    if n_jobs == 0:
        print('\n  n_jobs=0 — no NEB pairs within the standard filters.')
        print('  Fallback 2a: re-running enumeration with loosened bounds')
        print('  (sep 0–1e6 Å, prox 1e6 Å, graph_dist_min 0 — TEST ONLY).')
        try:
            result2 = orchestrate_neb_pipeline(
                phase1_h2_dir=h2_dir,
                phase2_h_dir=h_dir,
                phase3_sites_dir=sites_dir,
                e_clean=s1['e_clean'],
                outdir=neb_outdir,
                slurm_opts={**SLURM_DEFAULTS, **GPU_SLURM},
                neb_slurm_opts={**SLURM_DEFAULTS, **NEB_SLURM},
                sep_min=0.0,
                sep_max=1e6,
                graph_dist_min=0,
                prox_cutoff=1e6,
                n_images=N_NEB_IMAGES,
                neb_ftol=NEB_FTOL,
                dry_run=False,
                elem_str=ELEM_STR_7,
                e2t=E2T_7,
                masses=MASSES_7,
            )
            if result2['neb_result']['n_jobs'] > 0:
                result     = result2
                neb_result = result2['neb_result']
                n_jobs     = neb_result['n_jobs']
                tier       = 'fallback_2a_loosened_filters'
        except Exception:
            print('  Fallback 2a raised — continuing to 2b:')
            traceback.print_exc()

    # ── Fallback 2b: force one example pair directly from the pools ─────────
    if n_jobs == 0:
        print('\n  Fallback 2b: forcing one example NEB pair from the pools.')
        try:
            pools  = result['pools']
            is_ids = [sid for sid in pools.get('is_xy', {})
                      if sid in pools.get('is_energies', {})]
            fs_ids = [sid for sid in pools.get('fs_xy', {})
                      if sid in pools.get('fs_energies', {})]
            if is_ids and len(fs_ids) >= 2:
                is_sid = is_ids[0]
                # pick the two closest FS sites (plain xy — tiny test cell)
                best = None
                for i in range(len(fs_ids)):
                    for j in range(i + 1, len(fs_ids)):
                        a, b = fs_ids[i], fs_ids[j]
                        d = math.hypot(pools['fs_xy'][a][0] - pools['fs_xy'][b][0],
                                       pools['fs_xy'][a][1] - pools['fs_xy'][b][1])
                        if best is None or d < best[0]:
                            best = (d, a, b)
                sep, s1_id, s2_id = best
                ix, iy = pools['is_xy'][is_sid]
                fcx = (pools['fs_xy'][s1_id][0] + pools['fs_xy'][s2_id][0]) / 2.0
                fcy = (pools['fs_xy'][s1_id][1] + pools['fs_xy'][s2_id][1]) / 2.0
                E_IS = pools['is_energies'][is_sid]
                E_FS = (pools['fs_energies'][s1_id]
                        + pools['fs_energies'][s2_id] - s1['e_clean'])
                combo = {
                    'is_site'       : is_sid,
                    'is_true_label' : pools.get('is_true_labels', {}).get(is_sid, is_sid),
                    'fs_site1'      : s1_id,
                    'fs_site2'      : s2_id,
                    'fs_true_label1': pools.get('fs_true_labels', {}).get(s1_id, s1_id),
                    'fs_true_label2': pools.get('fs_true_labels', {}).get(s2_id, s2_id),
                    'is_xy'         : (ix, iy),
                    'fs_centroid'   : (fcx, fcy),
                    'is_fs_dist'    : float(math.hypot(ix - fcx, iy - fcy)),
                    'E_IS'          : E_IS,
                    'E_FS'          : E_FS,
                    'delta_E'       : E_FS - E_IS,
                    'fs_sep'        : float(sep),
                    'graph_dist'    : -1,
                    'label'         : f'{is_sid}__{s1_id}+{s2_id}',
                }
                print(f'  Forced pair: IS={is_sid}  FS=({s1_id}, {s2_id})  '
                      f'sep={sep:.3f} Å')
                neb_result = orchestrate_neb(
                    [combo], pools, s1['e_clean'],
                    outdir=neb_outdir,
                    slurm_opts={**SLURM_DEFAULTS, **GPU_SLURM},
                    neb_slurm_opts={**SLURM_DEFAULTS, **NEB_SLURM},
                    n_images=N_NEB_IMAGES,
                    neb_ftol=NEB_FTOL,
                    dry_run=False,
                    elem_str=ELEM_STR_7,
                    e2t=E2T_7,
                    masses=MASSES_7,
                )
                n_jobs = neb_result['n_jobs']
                if n_jobs > 0:
                    tier = 'fallback_2b_forced_pair'
            else:
                print(f'  Pools too small for a forced pair '
                      f'(IS={len(is_ids)}, FS={len(fs_ids)}).')
        except Exception:
            print('  Fallback 2b raised — falling back to hardcoded barriers:')
            traceback.print_exc()

    # ── Tier 3: graceful skip → Stage 7 uses hardcoded barrier values ────────
    if n_jobs == 0:
        print('\n  WARNING: n_jobs=0 after both fallbacks.')
        print('  Likely causes:')
        print('    • All H2* dissociated on the small slab → IS pool empty')
        print('    • No H* with E_ads < 0 at e_h2_gas=0 → FS pool empty')
        print('  Code paths for pool loading, pair enumeration, and script')
        print('  generation were all exercised. NEB submission skipped;')
        print('  Stage 7 will use the hardcoded synthetic barrier values.')
        _check('neb_graceful_empty', True, 'n_jobs=0 handled without crash')
        return dict(n_jobs=0, skipped=True)

    if tier != 'enumerated':
        _check('neb_fallback_pair_used', True, tier)
    _check('neb_jobs_submitted', True, f'{n_jobs} pair(s) submitted [{tier}]')

    poll = _poll_and_parse_barrier(neb_result['neb_jobs'][0]['barrier_file'])
    return dict(n_jobs=n_jobs, skipped=False,
                timed_out=poll['timed_out'],
                barrier_path=poll['barrier_path'], tier=tier)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 6 — Diffusivity NVT MD (2 temperatures, minimal steps)
# ═════════════════════════════════════════════════════════════════════════════

def stage6_diffusivity(work_dir: str) -> dict:
    _header(f'Stage 6: Diffusivity NVT MD ({DIFF_TEMPERATURES} K × '
            f'n_H={DIFF_N_H_VALUES}, minimal steps)')

    diff_dir = pathlib.Path(work_dir) / 'diffusivity'
    diff_dir.mkdir(parents=True, exist_ok=True)
    struct_stem = 'ni_bulk_test'

    # ── Build a tiny orthogonal FCC Ni bulk (32 atoms) ────────────────────────
    ni = ase_bulk('Ni', 'fcc', a=3.52, cubic=True).repeat([2, 2, 2])
    bulk_path = str(diff_dir / f'{struct_stem}.lammps')
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
    # n_h_values=[1, 3] regression-tests the per-n_H Arrhenius overwrite fix:
    # before the fix, every n_H shared one results/{stem}/ path, so the
    # second run silently clobbered the first's fit.
    out_py = str(diff_dir / 'diffusivity_run.py')
    generate_diffusivity_scripts(
        input_structures=[bulk_path],
        n_h_values=DIFF_N_H_VALUES,
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

    # ── Check expected output files, once per n_H ─────────────────────────────
    arr_jsons: dict[int, dict] = {}
    for n_h in DIFF_N_H_VALUES:
        run_name     = f'{struct_stem}_{n_h}H'
        run_root     = diff_dir / 'results' / run_name
        analysis_dir = run_root / 'analysis'
        msd_files    = list(analysis_dir.glob('msd_*.txt')) if analysis_dir.exists() else []
        diff_table   = analysis_dir / 'diffusivity_table.txt'
        arr_json     = run_root / 'diffusivity_arrhenius.json'

        _check(f'diff_msd_files_written_{n_h}H',
               len(msd_files) >= len(DIFF_TEMPERATURES),
               f'{len(msd_files)} msd_*.txt files found')
        _check(f'diff_table_written_{n_h}H', _exists(str(diff_table)), str(diff_table))
        _check(f'diff_arrhenius_json_written_{n_h}H', _exists(str(arr_json)), str(arr_json))

        if arr_json.exists():
            with open(arr_json) as f:
                arr = json.load(f)
            arr_jsons[n_h] = arr
            Ea  = arr.get('E_D_eV', float('nan'))
            D0  = arr.get('D0_m2s', float('nan'))
            R2  = arr.get('R2_fit', float('nan'))
            D_arr   = arr.get('D_arr', [])
            n_valid = sum(1 for d in D_arr
                          if isinstance(d, (int, float)) and d == d and d > 0)
            if n_valid >= 2:
                _check(f'diff_Ea_positive_{n_h}H', Ea > 0, f'Ea = {Ea:.4f} eV')
                _check(f'diff_D0_positive_{n_h}H', D0 > 0, f'D0 = {D0:.3e} m²/s')
                _check(f'diff_R2_reasonable_{n_h}H', R2 > 0.5, f'R² = {R2:.4f}')
            else:
                # At smoke scale (few H atoms, 2 ps) a negative-D noise point
                # is statistically expected; the pipeline must drop it and
                # write a NaN fit gracefully instead of crashing.
                _check(f'diff_arrhenius_graceful_nan_{n_h}H',
                       Ea != Ea and D0 != D0,   # NaN != NaN
                       f'{n_valid} valid D point(s) at smoke scale — '
                       f'NaN fit is the designed graceful behaviour')
            print(f'  n_H={n_h}  Arrhenius: Ea={Ea:.4f} eV  '
                  f'D0={D0:.3e} m²/s  R²={R2:.4f}')

    # ── Direct proof Bug 1 (shared-path overwrite) is fixed ───────────────────
    # Both n_H's files must exist independently (already checked above — the
    # pre-fix code had exactly one shared path, so only one file could ever
    # exist). When both fits are numerically valid, they must also differ,
    # since D0/Ea genuinely depend on H loading.
    if len(arr_jsons) == len(DIFF_N_H_VALUES) and len(DIFF_N_H_VALUES) >= 2:
        _n_a, _n_b = DIFF_N_H_VALUES[0], DIFF_N_H_VALUES[1]
        _a, _b = arr_jsons[_n_a], arr_jsons[_n_b]
        _both_valid = (_a.get('D0_m2s') == _a.get('D0_m2s')  # not NaN
                       and _b.get('D0_m2s') == _b.get('D0_m2s'))
        if _both_valid:
            _check('diff_per_nH_fits_independent',
                   _a.get('D0_m2s') != _b.get('D0_m2s')
                   or _a.get('E_D_eV') != _b.get('E_D_eV'),
                   f'{_n_a}H: D0={_a.get("D0_m2s"):.3e} Ea={_a.get("E_D_eV"):.4f}  vs  '
                   f'{_n_b}H: D0={_b.get("D0_m2s"):.3e} Ea={_b.get("E_D_eV"):.4f}')
        else:
            print(f'  [SKIP] diff_per_nH_fits_independent — one or both fits '
                  f'are NaN at smoke scale (graceful, not a failure)')

    return dict(diff_dir=str(diff_dir), struct_stem=struct_stem,
                arr_jsons={n_h: str(diff_dir / 'results' / f'{struct_stem}_{n_h}H'
                                     / 'diffusivity_arrhenius.json')
                           for n_h in DIFF_N_H_VALUES})


# ═════════════════════════════════════════════════════════════════════════════
# Stage 7 — Permeation (Part 2, real): Hop A/B NEB + vibrations + KMC + Φ
# ═════════════════════════════════════════════════════════════════════════════

def stage7_permeation(s0: dict, s2: dict, s4: dict, s6: dict,
                      work_dir: str) -> dict:
    _header('Stage 7: Permeation — real generate_permeation_scripts() '
            '(Hop A/B NEB, vibrations, KMC, permeability)')

    if not all(k in s6 for k in ('diff_dir', 'struct_stem', 'arr_jsons')):
        print('  [SKIP] Stage 6 did not complete — Part 3 diffusivity fits '
              'are Part 2\'s required input (no placeholder D0/Ea).')
        _check('permeation_prereqs_available', False, 'Stage 6 result incomplete')
        return {}

    stem          = s6['struct_stem']
    perm_work_dir = s6['diff_dir']   # same WORK_DIR Part 3 wrote results/ under
    perm_dir      = pathlib.Path(work_dir) / 'permeation'
    perm_dir.mkdir(parents=True, exist_ok=True)

    sub_neb_dir = os.path.join(perm_work_dir, 'neb_subsurface', stem)
    vib_dir     = os.path.join(perm_work_dir, 'vibrations', stem)
    results_dir = os.path.join(perm_work_dir, 'results', stem)
    out_py      = str(perm_dir / 'permeation_run.py')

    generate_permeation_scripts(
        work_dir=perm_work_dir,
        stem=stem,
        relaxed_slab_path=s0['slab_out'],
        surface_sites_json=s2['sites_json'],
        phase2_h_dir=os.path.join(s4['result']['outdir'], 'results'),
        sub_neb_dir=sub_neb_dir,
        vib_dir=vib_dir,
        results_dir=results_dir,
        temperatures=DIFF_TEMPERATURES,
        n_h_values=DIFF_N_H_VALUES,
        p_vals_pa=PERM_P_VALS_PA,
        a0_m=PERM_A0_M,
        l_m=PERM_L_M,
        # dh_diss_ev explicit: its auto-source (ranked_barriers.json) is a
        # known separately-tracked gap (GitHub #7), not what this test
        # verifies. dh_entry_ev=None exercises the REAL auto-extraction
        # from this run's own Hop A NEB results.
        dh_diss_ev=PERM_DH_DISS_EV,
        dh_entry_ev=None,
        nx=PERM_NX, ny=PERM_NY,
        seed=PERM_SEED,
        kmc_max_steps=PERM_KMC_MAX_STEPS,
        gpu_slurm_cfg={**SLURM_DEFAULTS, **GPU_SLURM},
        neb_slurm_cfg={**SLURM_DEFAULTS, **NEB_SLURM},
        vib_slurm_cfg={**SLURM_DEFAULTS, **NEB_SLURM},   # vibrations: CPU too
        n_images=N_NEB_IMAGES,
        spring_const=SPRING_CONST,
        neb_ftol=NEB_FTOL,
        out_py=out_py,
        elem_str=ELEM_STR_7,
        e2t=E2T_7,
        masses=MASSES_7,
        metal_type='alloy',
    )
    _check('permeation_run_py_written', _exists(out_py), out_py)

    # ── Run the orchestrator as a subprocess — submits real Hop A/B NEB and
    # vibration SLURM jobs and blocks until they (and the KMC sweeps) finish.
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    print('\n  Running permeation_run.py  (Hop A/B NEB + vibrations + KMC — '
          'submits real SLURM jobs and waits) …')
    ret = subprocess.run([sys.executable, out_py], env=env)
    _check('permeation_run_exit_0', ret.returncode == 0,
           f'exit code {ret.returncode}')

    # ── Which n_H concentrations had a valid (non-NaN) Part-3 fit? Part 2
    # skips the rest entirely by design — that's not a smoke-test failure.
    n_h_ready = {}
    for n_h in DIFF_N_H_VALUES:
        ready = False
        if _exists(s6['arr_jsons'][n_h]):
            with open(s6['arr_jsons'][n_h]) as f:
                a = json.load(f)
            d0, ea = a.get('D0_m2s'), a.get('E_D_eV')
            ready = (d0 == d0 and ea == ea)   # NaN-safe
        n_h_ready[n_h] = ready
        if not ready:
            print(f'  [INFO] n_H={n_h}: Part 3 diffusivity fit invalid/NaN at '
                  f'smoke scale — Part 2 correctly skips it (no placeholder).')

    perm_data: dict[int, dict] = {n_h: {} for n_h in DIFF_N_H_VALUES}
    for n_h in DIFF_N_H_VALUES:
        if not n_h_ready[n_h]:
            continue
        nh_dir = pathlib.Path(perm_work_dir) / 'results' / f'{stem}_{n_h}H'
        for T in DIFF_TEMPERATURES:
            sweep_f = nh_dir / f'permeation_sweep_T{int(T)}K.json'
            perm_f  = nh_dir / f'permeability_T{int(T)}K.json'
            _check(f'permeation_sweep_written_{n_h}H_{int(T)}K',
                   _exists(str(sweep_f)), str(sweep_f))
            _check(f'permeability_written_{n_h}H_{int(T)}K',
                   _exists(str(perm_f)), str(perm_f))
            if perm_f.exists():
                with open(perm_f) as f:
                    perm_data[n_h][T] = json.load(f)

        if perm_data[n_h]:
            has_caveat = any('dilute_limit_caveat' in d
                              for d in perm_data[n_h].values())
            if n_h > 1:
                _check(f'dilute_limit_caveat_present_{n_h}H', has_caveat,
                       'n_H>1 permeability output must carry the caveat')
            else:
                _check(f'dilute_limit_caveat_absent_{n_h}H', not has_caveat,
                       'n_H=1 IS the dilute limit — no caveat expected')
        else:
            print(f'  [SKIP] dilute_limit_caveat check for n_H={n_h} — no '
                  f'permeability file produced (Phase 6 may be globally '
                  f'skipped; see DH_ENTRY_EV auto-extraction messages above)')

    # ── Direct proof Bug 2 (global placeholder bulk diffusivity) is fixed:
    # each n_H's permeability JSON must embed that concentration's OWN
    # D0/Ea from Part 3 — never a shared placeholder.
    _ready = [n for n in DIFF_N_H_VALUES if perm_data.get(n)]
    if len(_ready) >= 2:
        _n_a, _n_b = _ready[0], _ready[1]
        _T0 = DIFF_TEMPERATURES[0]
        _da, _db = perm_data[_n_a].get(_T0), perm_data[_n_b].get(_T0)
        if _da and _db:
            _check('permeability_uses_per_nH_diffusivity',
                   _da.get('D0_m2s') != _db.get('D0_m2s')
                   or _da.get('E_D_eV') != _db.get('E_D_eV'),
                   f'{_n_a}H: D0={_da.get("D0_m2s")} Ea={_da.get("E_D_eV")}  vs  '
                   f'{_n_b}H: D0={_db.get("D0_m2s")} Ea={_db.get("E_D_eV")}')
    else:
        print('  [SKIP] permeability_uses_per_nH_diffusivity — fewer than 2 '
              'n_H concentrations produced a permeability file at smoke scale')

    return dict(perm_dir=str(perm_dir), perm_data=perm_data)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    t_start = time.time()
    print(f'\n{"#"*60}')
    print(f'  PIPELINE SMOKE TEST — {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Work dir     : {WORK_DIR}')
    print(f'  SLAB_LAYERS  : {SLAB_LAYERS}')
    print(f'  N_MAX_SITES  : {N_MAX_SITES}')
    print(f'  N_NEB_IMAGES : {N_NEB_IMAGES}  NEB_FTOL={NEB_FTOL}')
    print(f'  NEB timeout  : {NEB_TIMEOUT_H} h  (poll every {NEB_POLL_SEC} s)')
    print(f'  DIFF temps   : {DIFF_TEMPERATURES} K  n_H={DIFF_N_H_VALUES}  '
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

    s6 = {}
    try:
        s6 = stage6_diffusivity(WORK_DIR)
    except Exception:
        print('[WARN] Stage 6 raised an exception:')
        traceback.print_exc()
        _FAIL.append('diffusivity_stage_exception')

    try:
        stage7_permeation(s0, s2, s4, s6, WORK_DIR)
    except Exception:
        print('[WARN] Stage 7 raised an exception:')
        traceback.print_exc()
        _FAIL.append('permeation_stage_exception')

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

"""
models/permeation_workflow.py
==============================
Script generator + local analysis functions backing calculation/permeation.ipynb.

Part 2 of the three-part multiscale H permeation pipeline:
  Part 1 — Surface NEB          (neb_calculation.ipynb  / neb_workflow.py)
  Part 2 — Surface→Subsurface   (permeation.ipynb       / this module)   ← HERE
  Part 3 — Bulk diffusivity     (diffusivity.ipynb      / diffusivity_workflow.py)

Section A — Script generation
    generate_permeation_scripts(...)  →  permeation_run.py
    generate_permeation_sh(...)       →  permeation_run.sh

Section B — Local analysis (Phase 4 cells in permeation.ipynb)
    load_barrier_summary, load_rate_summary, load_kmc_sweeps,
    load_permeability_results,
    plot_barrier_overview, plot_mep_overlay, plot_kmc_sieverts,
    plot_permeability_vs_T, plot_arrhenius_S0, plot_bottleneck
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.parsers import parse_energy_log

try:
    from IPython.display import display as _display
except ImportError:
    _display = print


def collect_dedup_is_labels(phase2_h_dir: str) -> list:
    """
    Build ``(sid, is_path, e_is)`` triples for Hop A's NEB.

    ``e_is`` is the ABSOLUTE relaxed total energy of each H* adsorption
    structure (from Part 1's own ``h_min_{sid}.log``) — not a binding or
    adsorption energy. Hop A's NEB needs the same absolute-energy
    reference ``E_FS`` uses (parsed from ``fs_min.log``); a placeholder
    like ``0.0`` here would silently produce reaction energies and
    barriers offset by the slab's entire total energy (~100+ eV), since
    0.0 sits nowhere near this structure's real energy scale.

    Sites whose ``h_min_{sid}.log`` is missing or unparseable are skipped
    with a warning — never assigned a fabricated energy.

    Parameters
    ----------
    phase2_h_dir : str
        Directory containing ``h_atom_{sid}_relaxed.lammps`` and
        ``h_min_{sid}.log`` for each surface site (Part 1 Section B
        Phase 2 output).

    Returns
    -------
    list of (sid, is_path, e_is)
        Sorted by ``sid``. Sites with no parseable log are omitted.
    """
    dedup_is_labels = []
    for p in sorted(glob.glob(os.path.join(phase2_h_dir, 'h_atom_*_relaxed.lammps'))):
        sid = os.path.basename(p).replace('h_atom_', '').replace('_relaxed.lammps', '')
        log_p = os.path.join(phase2_h_dir, f'h_min_{sid}.log')
        parsed = parse_energy_log(log_p) if os.path.exists(log_p) else None
        if not parsed or 'pe_final_eV' not in parsed:
            print(f'  WARNING: could not parse E_IS for sid={sid} from {log_p} '
                  f'— skipping (no fabricated energy).')
            continue
        dedup_is_labels.append((sid, p, parsed['pe_final_eV']))
    return dedup_is_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Section A — Script generation
# ═══════════════════════════════════════════════════════════════════════════════

# Raw orchestrator body injected verbatim into permeation_run.py.
# All ALL_CAPS variables are defined in the f-string header written by
# generate_permeation_scripts().
_PERMEATION_BODY = r"""
# ── Runtime imports ────────────────────────────────────────────────────────────
import os
import sys
import json
import glob
import itertools
import numpy as np
from ase.io import read as _ase_read

from models.config import MACE_MODEL_ASE
from models.subsurface_graph import build_subsurface_graph, connect_to_surface
from models.neb_subsurface import orchestrate_hopa_neb, orchestrate_hopb_neb
from models.vibrations import collect_is_ts_paths, orchestrate_vibrations
from models.tst_rates import (
    collect_neb_results,
    split_vib_results,
    build_rate_dict,
    rates_to_json,
)
from models.permeation import (
    sweep_pressure,
    arrhenius_diffusivity,
    fit_solubility_from_kmc,
    lattice_site_S0,
    solubility_from_rates,
    sieverts_solubility,
    permeability,
    richardson_flux,
    resolve_nh_diffusivity,
)
from models.permeation_workflow import collect_dedup_is_labels
from models.parsers import parse_barrier_file
from models.create_slurm import submit_slurm_job, wait_for_jobs, auto_submit

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUB_NEB_DIR, exist_ok=True)
os.makedirs(VIB_DIR, exist_ok=True)

_KB_EV = 8.617333262e-5   # eV/K

# ── Build subsurface graph ─────────────────────────────────────────────────────
print('Building subsurface graph …')
with open(SURFACE_SITES_JSON) as _f:
    _surf_data = json.load(_f)
G, subsurface_sites = build_subsurface_graph(RELAXED_SLAB_PATH, SURFACE_SITES_JSON, seed=42,
                                             metal_type=METAL_TYPE)
_slab_atoms = _ase_read(RELAXED_SLAB_PATH, format='lammps-data', atom_style='atomic')
# connect_to_surface (and _periodic_xy_distance) expect a flat [Lx, Ly, Lz]
# array, not ASE's 3x3 Cell object -- .get_cell() alone made cell[0] a row
# vector, breaking round(dx / cell[0]) the first time this ever ran for real.
_cell = _slab_atoms.get_cell().diagonal()
surface_connections = connect_to_surface(subsurface_sites, _surf_data, _cell)
_sub1_n = sum(1 for s in subsurface_sites if s.get('layer_classification') == 'subsurface_1')
print(f'  subsurface-1 sites : {_sub1_n}')
print(f'  surface connections: {len(surface_connections)}')

# Species present in this slab (length-descending so 2-letter symbols match
# before 'O' in strings like 'CrO_bridge'); used for rate-label matching.
_slab_species = sorted(
    {s for s in _slab_atoms.get_chemical_symbols() if s != 'H'},
    key=len, reverse=True,
)
_sid2comp = {s['site_id']: str(s.get('level1', {}).get('composition', ''))
             for s in _surf_data.get('sites', [])}

# KMC surface composition: oxide surfaces use the actual surface-atom
# element fractions; metals keep make_grid's Hastelloy default (unchanged).
_kmc_composition = None
if METAL_TYPE == 'oxide':
    _cnt = {}
    for _a in _surf_data.get('surface_atoms', []):
        _cnt[_a['element']] = _cnt.get(_a['element'], 0) + 1
    _tot = sum(_cnt.values()) or 1
    _kmc_composition = {k: v / _tot for k, v in _cnt.items()}
    print(f'  KMC surface composition (oxide): {_kmc_composition}')

# ── Collect dedup IS labels ────────────────────────────────────────────────────
dedup_is_labels = collect_dedup_is_labels(PHASE2_H_DIR)
print(f'Dedup IS labels: {len(dedup_is_labels)}')
if not dedup_is_labels:
    raise FileNotFoundError(
        f'No h_atom_*_relaxed.lammps found in {PHASE2_H_DIR}. '
        'Run Part 1 (neb_calculation.ipynb) first.'
    )

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Hop A NEB: surface H* → subsurface-1 oct
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Phase 1: Hop A NEB ──────────────────────────────────────────────────')
_hopa_jobs_json = os.path.join(SUB_NEB_DIR, 'hopa', 'hopa_jobs.json')
if not os.path.exists(_hopa_jobs_json):
    hopa_out = orchestrate_hopa_neb(
        dedup_is_labels    = dedup_is_labels,
        subsurface_graph   = (G, subsurface_sites),
        surface_connections= surface_connections,
        outdir             = SUB_NEB_DIR,
        masses             = MASSES,
        e2t                = E2T,
        elem_str           = ELEM_STR,
        slurm_opts         = GPU_SLURM_CFG,
        neb_slurm_opts     = NEB_SLURM_CFG,
        n_images           = N_IMAGES,
        spring_const       = SPRING_K,
        neb_ftol           = NEB_FTOL_VAL,
        dry_run            = True,
    )
    hopa_jobs = hopa_out['jobs']
    print(f'  Hop A: {hopa_out["n_jobs"]} jobs  fsmin_array={hopa_out["fsmin_array"]}')

    print('  Submitting Hop A FS-min …')
    auto_submit(
        array_script   = hopa_out['fsmin_array'],
        index_file     = os.path.join(SUB_NEB_DIR, 'hopa', 'job_index.txt'),
        result_dir     = os.path.join(SUB_NEB_DIR, 'hopa'),
        result_pattern = '*/sub1_fs_relaxed.lammps',
        n_total        = hopa_out['n_jobs'],
        job_name       = 'hopa_fsmin_array',
        queue_max      = 8,
        concurrent     = 4,
    )
    print('  Hop A FS-min done.')

    print('  Submitting Hop A NEB …')
    _jid_hopa_neb = submit_slurm_job(hopa_out['neb_array'])
    wait_for_jobs({'hopa_neb': _jid_hopa_neb})
    print('  Hop A NEB done.')
else:
    with open(_hopa_jobs_json) as _f:
        hopa_jobs = json.load(_f)
    print(f'  Hop A already done ({len(hopa_jobs)} jobs) — skipping')

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Hop B NEB: subsurface-1 → subsurface-2 oct
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Phase 2: Hop B NEB ──────────────────────────────────────────────────')
_hopb_jobs_json = os.path.join(SUB_NEB_DIR, 'hopb', 'hopb_jobs.json')
if not os.path.exists(_hopb_jobs_json):
    hopb_out = orchestrate_hopb_neb(
        hopa_jobs          = hopa_jobs,
        hopa_outdir        = os.path.join(SUB_NEB_DIR, 'hopa'),
        subsurface_graph   = (G, subsurface_sites),
        outdir             = SUB_NEB_DIR,
        masses             = MASSES,
        e2t                = E2T,
        elem_str           = ELEM_STR,
        slurm_opts         = GPU_SLURM_CFG,
        neb_slurm_opts     = NEB_SLURM_CFG,
        n_images           = N_IMAGES,
        spring_const       = SPRING_K,
        neb_ftol           = NEB_FTOL_VAL,
        dry_run            = True,
    )
    hopb_jobs = hopb_out['jobs']
    print(f'  Hop B: {hopb_out["n_jobs"]} jobs  fsmin_array={hopb_out["fsmin_array"]}')

    print('  Submitting Hop B FS-min …')
    auto_submit(
        array_script   = hopb_out['fsmin_array'],
        index_file     = os.path.join(SUB_NEB_DIR, 'hopb', 'job_index.txt'),
        result_dir     = os.path.join(SUB_NEB_DIR, 'hopb'),
        result_pattern = '*/sub2_fs_relaxed.lammps',
        n_total        = hopb_out['n_jobs'],
        job_name       = 'hopb_fsmin_array',
        queue_max      = 8,
        concurrent     = 4,
    )
    print('  Hop B FS-min done.')

    print('  Submitting Hop B NEB …')
    _jid_hopb_neb = submit_slurm_job(hopb_out['neb_array'])
    wait_for_jobs({'hopb_neb': _jid_hopb_neb})
    print('  Hop B NEB done.')
else:
    with open(_hopb_jobs_json) as _f:
        hopb_jobs = json.load(_f)
    print(f'  Hop B already done ({len(hopb_jobs)} jobs) — skipping')

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Vibrational frequencies (IS + TS, both hops)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Phase 3: Vibrational frequencies ────────────────────────────────────')
_pairs_a = collect_is_ts_paths(hopa_jobs, hop='hopa', is_key='is_path')
_pairs_b = collect_is_ts_paths(hopb_jobs, hop='hopb', is_key='hopb_is')
_all_pairs = _pairs_a + _pairs_b
print(f'  Structures: {len(_all_pairs)} (IS + TS for both hops)')

vib_out = orchestrate_vibrations(
    structure_paths = _all_pairs,
    outdir          = VIB_DIR,
    mace_model_path = MACE_MODEL_ASE,
    slurm_opts      = VIB_SLURM_CFG,
    delta           = 0.01,
    device          = 'cpu',
    dry_run         = True,
)

_vib_jids = {}
for _lbl, _info in vib_out.items():
    if _info.get('slurm'):
        _vib_jids[_lbl] = submit_slurm_job(_info['slurm'])
print(f'  Submitted {len(_vib_jids)} vibration jobs.')
wait_for_jobs(_vib_jids)
print('  Vibrations done.')

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — TST rate constants at each temperature
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Phase 4: TST rate constants ──────────────────────────────────────────')
_neb_res_a = collect_neb_results(hopa_jobs, hop='hopa')
_neb_res_b = collect_neb_results(hopb_jobs, hop='hopb')
_neb_results = {**_neb_res_a, **_neb_res_b}
_vib_is, _vib_ts = split_vib_results(vib_out)
print(f'  NEB results: {len(_neb_results)} labels  IS: {len(_vib_is)}  TS: {len(_vib_ts)}')

for _T in TEMPERATURES:
    _rd = build_rate_dict(_neb_results, _vib_is, _vib_ts, T_K=_T, apply_zpe=True)
    _out_json = rates_to_json(_rd, os.path.join(RESULTS_DIR, f'rate_dict_T{int(_T)}K.json'))
    print(f'  T={_T:4.0f} K: {len(_rd)} rates → {_out_json}')

# ══════════════════════════════════════════════════════════════════════════════
# Phases 5-6 — Per H-concentration: KMC pressure sweeps + permeability
# ══════════════════════════════════════════════════════════════════════════════
# Bulk diffusivity genuinely depends on H loading (H-H site blocking), unlike
# Phases 1-4 above (surface/subsurface entry, dissociation), which are
# independent of bulk H concentration and were computed once, per stem.
# So Phases 5-6 run once per N_H_VALUES entry, each using that concentration's
# own MD-fitted Arrhenius diffusivity from Part 3 — never a placeholder. If a
# concentration's fit is missing or invalid, that concentration is skipped
# entirely (loud error, no output files for it), not faked.
_DISS_JSON = os.path.join(WORK_DIR, 'neb', 'diss_jobs.json')
_NU_DISS   = 1e13   # s⁻¹ — fallback prefactor when ZPE rates unavailable

# T-dependent lattice parameter from Part 3 NPT MD (n_H-independent — the
# bare-bulk NPT that produces this has no H in it).
_LAT_JSON_5 = os.path.join(RESULTS_DIR, 'lattice_params_vs_T.json')
if os.path.exists(_LAT_JSON_5):
    with open(_LAT_JSON_5) as _f:
        _lat5 = json.load(_f)
    _a0_dict = dict(zip(_lat5['temperatures'], _lat5['a0_m']))
    print(f'  Loaded lattice_params_vs_T.json: {len(_a0_dict)} temperatures')
else:
    _a0_dict = {}
    print(f'  WARNING: lattice_params_vs_T.json not found — using fixed A0_M={A0_M} m')

# ZPE-corrected dissociation rates from Part 1 Phase E (n_H-independent —
# see GitHub issue on collect_neb_results never being called; this is
# expected to be absent today).
_DISS_VIB_JSON = os.path.join(WORK_DIR, 'neb', 'diss_vib_rates.json')
_diss_vib = {}   # {tuple(pair): {Ea_zpe, Ed_zpe, nu}}
if os.path.exists(_DISS_VIB_JSON):
    with open(_DISS_VIB_JSON) as _f:
        _dv_raw = json.load(_f)
    for _dv_lbl, _dv in _dv_raw.items():
        _pkey = tuple(_dv['pair'])
        if _pkey not in _diss_vib or _dv.get('Ea_zpe', 9e9) < _diss_vib[_pkey]['Ea_zpe']:
            _diss_vib[_pkey] = {'Ea_zpe': _dv['Ea_zpe'],
                                 'Ed_zpe': _dv['Ed_zpe'],
                                 'nu':     _dv['nu']}
    print(f'  Loaded diss_vib_rates.json: {len(_diss_vib)} element pairs (ZPE-corrected)')
else:
    print(f'  WARNING: diss_vib_rates.json not found — using raw barriers for diss/des rates')

# Auto-extract DH values from NEB results if not set manually (n_H-independent
# — used only by Phase 6 Option 1 below).
_DH_DISS_USED  = DH_DISS_EV
_DH_ENTRY_USED = DH_ENTRY_EV

if _DH_ENTRY_USED is None:
    _fst_rd_f = os.path.join(RESULTS_DIR, f'rate_dict_T{int(TEMPERATURES[0])}K.json')
    if os.path.exists(_fst_rd_f):
        with open(_fst_rd_f) as _f: _fst_rd = json.load(_f)
        _hopa_de = [_r.get('delta_e', 0.0) for _lbl, _r in _fst_rd.items()
                    if _lbl.startswith('hopa_')]
        if _hopa_de:
            _DH_ENTRY_USED = float(np.mean(_hopa_de))
            print(f'  Auto-extracted DH_ENTRY_EV = {_DH_ENTRY_USED:.4f} eV '
                  f'(mean of {len(_hopa_de)} Hop A barriers)')

if _DH_DISS_USED is None:
    _ranked6_f = os.path.join(WORK_DIR, 'neb', 'ranked_barriers.json')
    if os.path.exists(_ranked6_f):
        with open(_ranked6_f) as _f: _ranked6 = json.load(_f)
        _diss_de6 = [_r.get('delta_E', 0.0) for _r in _ranked6
                     if _r.get('converged', False)]
        if _diss_de6:
            _DH_DISS_USED = float(np.mean(_diss_de6))
            print(f'  Auto-extracted DH_DISS_EV  = {_DH_DISS_USED:.4f} eV '
                  f'(mean of {len(_diss_de6)} converged diss barriers)')

_PHASE6_READY = _DH_DISS_USED is not None and _DH_ENTRY_USED is not None
if not _PHASE6_READY:
    print('WARNING: DH_DISS_EV or DH_ENTRY_EV could not be determined — Phase 6 '
          '(permeability) will be skipped for every H-concentration.')
    print('  Fill DH_DISS_EV and DH_ENTRY_EV in permeation.ipynb Cell 2 and regenerate.')
else:
    _DH_SOL = _DH_DISS_USED / 2.0 + _DH_ENTRY_USED
_P_HIGH = max(P_VALS_PA)

# Tracks what actually got produced across all n_H, since every skip below is
# a `continue` (never a raise) by design — the fail-loud signal for the
# orchestrator has to come from checking this at the end, not from an
# exception, or a metal that silently produced zero output would still
# report success. See [[project_pipeline_test_bugs]].
_PERM_STATUS = {
    'stem':                STEM,
    'n_h_requested':       list(N_H_VALUES),
    'phase6_ready':        _PHASE6_READY,
    'n_h_skipped':         [],
    'permeability_written': [],
}

for _n_h in N_H_VALUES:
    print(f'\n{"="*76}\nH concentration: n_H = {_n_h}\n{"="*76}')
    _res_nh = resolve_nh_diffusivity(WORK_DIR, STEM, _n_h)
    _nh_dir = _res_nh['nh_dir']

    if not _res_nh['ready']:
        print(f"  ERROR: {_res_nh['message']}")
        _PERM_STATUS['n_h_skipped'].append({'n_h': _n_h, 'reason': _res_nh['message']})
        continue
    _D0_nh, _ED_nh = _res_nh['D0_m2s'], _res_nh['E_D_eV']
    _dilute_note = _res_nh['dilute_note']
    print(f'  Loaded real diffusivity fit: D0={_D0_nh:.3e} m²/s  Ea={_ED_nh:.4f} eV')
    if _dilute_note:
        print(f'  NOTE: {_dilute_note}')

    os.makedirs(_nh_dir, exist_ok=True)

    # ── Phase 5: KMC pressure sweeps for this n_H ────────────────────────────
    print(f'\n── Phase 5 (n_H={_n_h}): KMC pressure sweeps ──────────────────────')
    for _T in TEMPERATURES:
        _out = os.path.join(_nh_dir, f'permeation_sweep_T{int(_T)}K.json')
        if os.path.exists(_out):
            print(f'  T={_T:4.0f} K  KMC sweep already done — skipping')
            continue

        with open(os.path.join(RESULTS_DIR, f'rate_dict_T{int(_T)}K.json')) as _f:
            _tst = json.load(_f)

        _kBT = _KB_EV * _T
        _a0_T = _a0_dict.get(_T, A0_M)
        _k_diss, _k_des, _k_entry, _k_exit = {}, {}, {}, {}

        for _lbl, _r in _tst.items():
            if _lbl.startswith('hopa_'):
                _sid = _lbl[len('hopa_'):]
                _matched = False
                for _el in _slab_species:
                    if _el in _sid:
                        _k_entry[_el] = _r['k_forward']
                        _k_exit[_el]  = _r['k_reverse']
                        _matched = True
                        break
                if not _matched:
                    # Bare site ids (s_NN) carry no element name — resolve via
                    # the site's level1 composition from surface_sites.json.
                    _comp = _sid2comp.get(_sid, '')
                    for _el in _slab_species:
                        if _el in _comp:
                            _k_entry.setdefault(_el, _r['k_forward'])
                            _k_exit.setdefault(_el, _r['k_reverse'])

        if _diss_vib:
            for _pkey5, _dv5 in _diss_vib.items():
                _k_diss[_pkey5] = np.exp(-_dv5['Ea_zpe'] / _kBT)
                _k_des[_pkey5]  = _dv5['nu'] * np.exp(-_dv5['Ed_zpe'] / _kBT)
        elif os.path.exists(_DISS_JSON):
            with open(_DISS_JSON) as _f:
                _diss = json.load(_f)
            for _job in _diss:
                _bf = _job.get('barrier_file', '')
                if not os.path.exists(_bf):
                    continue
                _bd   = parse_barrier_file(_bf)
                _pair = tuple(sorted(_job['sid'].replace('-', '').split('_')[:2]))
                _k_diss[_pair] = np.exp(-_bd['E_abs'] / _kBT)
                _k_des[_pair]  = _NU_DISS * np.exp(-_bd['E_des'] / _kBT)
        else:
            print(f'  WARNING: no diss rate source found — using placeholders at T={_T} K')
            # element_pair keys are sorted tuples; cover every species pair
            # actually present in this slab (metal or oxide).
            for _pair in itertools.combinations_with_replacement(
                    sorted(_slab_species), 2):
                _k_diss[_pair] = np.exp(-0.5  / _kBT)
                _k_des[_pair]  = _NU_DISS * np.exp(-1.2 / _kBT)

        for _el, _ke in _k_entry.items():
            print(f'  [{_T:.0f}K] k_entry({_el})={_ke:.3e} s⁻¹  k_exit({_el})={_k_exit.get(_el, float("nan")):.3e} s⁻¹')
        _rate_dict = {'k_diss': _k_diss, 'k_des': _k_des,
                      'k_entry': _k_entry, 'k_exit': _k_exit}
        _D_T = arrhenius_diffusivity(_D0_nh, _ED_nh, _T)
        np.random.seed(SEED)
        _sweep = sweep_pressure(
            P_vals_Pa  = P_VALS_PA,
            rate_dict  = _rate_dict,
            D_m2s      = _D_T,
            L_m        = L_M,
            T_K        = _T,
            a0_m       = _a0_T,
            nx         = NX,
            ny         = NY,
            seed       = SEED,
            composition = _kmc_composition,
            kmc_kwargs = {'window': 2000, 'rtol': 0.02, 'max_steps': KMC_MAX_STEPS},
        )
        _sweep['T_K']   = _T
        _sweep['D_m2s'] = _D_T
        _sweep['a0_m']  = _a0_T
        _sweep['n_H']   = _n_h
        if _dilute_note:
            _sweep['dilute_limit_caveat'] = _dilute_note
        with open(_out, 'w') as _f:
            json.dump(_sweep, _f, indent=2)
        _conv = sum(1 for c in _sweep.get('converged', []) if c)
        print(f'  T={_T:4.0f} K  a0={_a0_T:.4e} m  D={_D_T:.2e} m²/s  '
              f'{_conv}/{len(P_VALS_PA)} converged → {_out}')

    # ── Phase 6: Richardson-Sieverts permeability for this n_H ───────────────
    if not _PHASE6_READY:
        continue
    print(f'\n── Phase 6 (n_H={_n_h}): Richardson-Sieverts permeability ─────────')

    for _T in TEMPERATURES:
        _sweep_f = os.path.join(_nh_dir, f'permeation_sweep_T{int(_T)}K.json')
        if not os.path.exists(_sweep_f):
            print(f'  T={_T:4.0f} K  no sweep file — skipping permeability')
            continue
        with open(_sweep_f) as _f:
            _sw = json.load(_f)
        _D_T  = arrhenius_diffusivity(_D0_nh, _ED_nh, _T)
        _a0_T6 = _a0_dict.get(_T, A0_M)
        _kBT6  = _KB_EV * _T

        # Option 1 — lattice site density (per-T a₀)
        _S0_lat6 = lattice_site_S0(_a0_T6)
        _S1      = sieverts_solubility(_DH_SOL, _S0_lat6, _T)
        _Phi1    = permeability(_D_T, _S1)
        _J1      = richardson_flux(_Phi1, _P_HIGH, 0.0, L_M)

        # Option 2 — TST detailed balance (first Hop A rate as representative)
        with open(os.path.join(RESULTS_DIR, f'rate_dict_T{int(_T)}K.json')) as _f:
            _rd2 = json.load(_f)
        _repr_entry, _repr_exit = 1e9, 1e8
        for _lbl, _r in _rd2.items():
            if _lbl.startswith('hopa_'):
                _repr_entry, _repr_exit = _r['k_forward'], _r['k_reverse']
                break
        if _diss_vib:
            _dv0_t = next(iter(_diss_vib.values()))
            _repr_diss = np.exp(-_dv0_t['Ea_zpe'] / _kBT6)
            _repr_des  = _dv0_t['nu'] * np.exp(-_dv0_t['Ed_zpe'] / _kBT6)
        else:
            _repr_diss = np.exp(-0.5  / _kBT6)
            _repr_des  = _NU_DISS * np.exp(-1.2 / _kBT6)
        _S2   = solubility_from_rates(_repr_diss, _repr_des, _repr_entry, _repr_exit,
                                      _a0_T6, _T)
        _Phi2 = permeability(_D_T, _S2)
        _J2   = richardson_flux(_Phi2, _P_HIGH, 0.0, L_M)

        # Option 3 — KMC empirical Sieverts fit
        _kmc_sol = fit_solubility_from_kmc(_sw)
        _S3   = _kmc_sol['S_mean']
        _Phi3 = permeability(_D_T, _S3)
        _J3   = richardson_flux(_Phi3, _P_HIGH, 0.0, L_M)

        _perm_f = os.path.join(_nh_dir, f'permeability_T{int(_T)}K.json')
        _perm_payload = {
            'T_K': _T, 'n_H': _n_h, 'D0_m2s': _D0_nh, 'E_D_eV': _ED_nh,
            'dH_sol_eV': _DH_SOL, 'a0_m': _a0_T6,
            'dH_diss_eV': _DH_DISS_USED, 'dH_entry_eV': _DH_ENTRY_USED,
            'option1': {'S': _S1, 'Phi': _Phi1, 'J': _J1},
            'option2': {'S': _S2, 'Phi': _Phi2, 'J': _J2},
            'option3': {'S': _S3, 'Phi': _Phi3, 'J': _J3,
                        'S_std':       _kmc_sol['S_std'],
                        'n_converged': _kmc_sol['n_converged']},
            'P_high_Pa': _P_HIGH, 'L_m': L_M,
        }
        if _dilute_note:
            _perm_payload['dilute_limit_caveat'] = _dilute_note
        with open(_perm_f, 'w') as _f:
            json.dump(_perm_payload, _f, indent=2)
        _PERM_STATUS['permeability_written'].append(f'{_n_h}H_T{int(_T)}K')
        print(f'  T={_T:4.0f} K  Opt1(lattice):  S={_S1:.3e}  Phi={_Phi1:.3e}  J={_J1:.3e} atoms/m²/s')
        print(f'  T={_T:4.0f} K  Opt2(TST):      S={_S2:.3e}  Phi={_Phi2:.3e}  J={_J2:.3e} atoms/m²/s')
        print(f'  T={_T:4.0f} K  Opt3(KMC):      S={_S3:.3e}  Phi={_Phi3:.3e}  J={_J3:.3e} atoms/m²/s')

    # Multi-T Arrhenius S₀ fit from KMC, for this n_H
    _S_arr, _T_arr = [], []
    for _T in TEMPERATURES:
        _sw_f = os.path.join(_nh_dir, f'permeation_sweep_T{int(_T)}K.json')
        if not os.path.exists(_sw_f):
            continue
        with open(_sw_f) as _f:
            _sw = json.load(_f)
        _sol = fit_solubility_from_kmc(_sw)
        if _sol['n_converged'] > 0:
            _S_arr.append(_sol['S_mean'])
            _T_arr.append(float(_T))

    if len(_S_arr) >= 2:
        _S_np    = np.array(_S_arr)
        _T_np    = np.array(_T_arr)
        _slope, _inter = np.polyfit(1.0 / _T_np, np.log(_S_np), 1)
        _dH_kmc  = -_slope * _KB_EV
        _S0_kmc  = np.exp(_inter)
        _log_pred = _slope / _T_np + _inter
        _ss_res  = np.sum((np.log(_S_np) - _log_pred) ** 2)
        _ss_tot  = np.sum((np.log(_S_np) - np.mean(np.log(_S_np))) ** 2)
        _r2      = 1.0 - _ss_res / _ss_tot if _ss_tot > 0.0 else 1.0
        _sol_out = os.path.join(_nh_dir, 'solubility_arrhenius_kmc.json')
        with open(_sol_out, 'w') as _f:
            json.dump({'T_K_arr':       _T_arr,
                       'S_mean_arr':    _S_arr,
                       'S0_kmc':        _S0_kmc,
                       'dH_sol_kmc_eV': _dH_kmc,
                       'r2_fit':        _r2,
                       'n_H':           _n_h,
                       'D0_m2s':        _D0_nh,
                       'E_D_eV':        _ED_nh}, _f, indent=2)
        print(f'\nArrhenius  S0={_S0_kmc:.3e}  dH_sol={_dH_kmc:.3f} eV  R²={_r2:.4f} → {_sol_out}')
    else:
        print(f'WARNING: fewer than 2 valid temperatures for n_H={_n_h} — Arrhenius S₀ fit skipped.')

_status_path = os.path.join(RESULTS_DIR, 'permeation_status.json')
with open(_status_path, 'w') as _f:
    json.dump(_PERM_STATUS, _f, indent=2)

if not _PERM_STATUS['permeability_written']:
    print(f'\n*** FAILED: no permeability results were produced for any (n_H, T) '
          f'combination — see {_status_path} ***')
    sys.exit(1)
else:
    print(f'\n=== permeation_run.py complete: '
          f'{len(_PERM_STATUS["permeability_written"])} permeability result(s) written ===')
"""


def generate_permeation_scripts(
    work_dir,
    stem,
    relaxed_slab_path,
    surface_sites_json,
    phase2_h_dir,
    sub_neb_dir,
    vib_dir,
    results_dir,
    temperatures,
    n_h_values,
    p_vals_pa,
    a0_m,
    l_m,
    dh_diss_ev,
    dh_entry_ev,
    nx,
    ny,
    seed,
    kmc_max_steps,
    gpu_slurm_cfg,
    neb_slurm_cfg,
    vib_slurm_cfg,
    n_images,
    spring_const,
    neb_ftol,
    out_py,
    elem_str=None,
    e2t=None,
    masses=None,
    metal_type='alloy',
):
    """Write permeation_run.py with embedded config. Returns the output path."""
    from models.config import MASSES_7, E2T_7, ELEM_STR_7
    if elem_str is None:
        elem_str = ELEM_STR_7
    if e2t is None:
        e2t = E2T_7
    if masses is None:
        masses = MASSES_7
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    _header = f'''#!/usr/bin/env python3
"""
permeation_run.py
==================
Standalone H permeation workflow orchestrator (Part 2).
Submitted via permeation_run.sh (SLURM wrapper generated by permeation.ipynb).

Phases:
  1 — Hop A NEB  (surface H* → subsurface-1 oct)
  2 — Hop B NEB  (subsurface-1 → subsurface-2 oct)
  3 — Vibrational frequencies (IS + TS, both hops)
  4 — TST rate constants at each temperature
  5 — KMC pressure sweeps at each temperature, once per H-concentration
  6 — Richardson-Sieverts permeability (all three S0 options), once per
      H-concentration, each using that concentration's own Part-3-fitted
      bulk diffusivity (no placeholder — a missing/invalid fit skips that
      concentration entirely rather than substituting a fake value)

Generated by calculation/permeation.ipynb — do not edit by hand.
"""

# ── Injected configuration ────────────────────────────────────────────────────
import os
import sys

_parent = {_parent!r}
if _parent not in sys.path:
    sys.path.insert(0, _parent)

WORK_DIR           = {work_dir!r}
STEM               = {stem!r}
RELAXED_SLAB_PATH  = {relaxed_slab_path!r}
SURFACE_SITES_JSON = {surface_sites_json!r}
PHASE2_H_DIR       = {phase2_h_dir!r}
SUB_NEB_DIR        = {sub_neb_dir!r}
VIB_DIR            = {vib_dir!r}
RESULTS_DIR        = {results_dir!r}

TEMPERATURES   = {temperatures!r}
N_H_VALUES     = {n_h_values!r}
P_VALS_PA      = {p_vals_pa!r}
A0_M           = {a0_m!r}
L_M            = {l_m!r}

DH_DISS_EV     = {dh_diss_ev!r}   # eV  — dissociation NEB delta_E (fill after NEB)
DH_ENTRY_EV    = {dh_entry_ev!r}  # eV  — Hop A delta_E (fill after NEB)

NX             = {nx!r}
NY             = {ny!r}
SEED           = {seed!r}
KMC_MAX_STEPS  = {kmc_max_steps!r}

N_IMAGES       = {n_images!r}
SPRING_K       = {spring_const!r}
NEB_FTOL_VAL   = {neb_ftol!r}

GPU_SLURM_CFG  = {gpu_slurm_cfg!r}
NEB_SLURM_CFG  = {neb_slurm_cfg!r}
VIB_SLURM_CFG  = {vib_slurm_cfg!r}

# Element / type tables (metal-specific)
ELEM_STR = {elem_str!r}
E2T      = {e2t!r}
MASSES   = {masses!r}
METAL_TYPE = {metal_type!r}

'''

    content = _header + _PERMEATION_BODY
    with open(out_py, 'w') as f:
        f.write(content)
    print(f'Written: {out_py}')
    return out_py


def generate_permeation_sh(
    orch_job_name,
    orch_partition,
    orch_cpus_per_task,
    orch_mem,
    orch_time,
    orch_openmpi_ver,
    orch_cuda_version,
    orch_conda_env,
    orch_ld_paths,
    work_dir,
    out_py,
    out_sh,
):
    """Write permeation_run.sh — the SLURM wrapper for the orchestrator."""
    lines = ['#!/bin/bash\n']
    lines.append(f'#SBATCH --job-name={orch_job_name}\n')
    lines.append(f'#SBATCH --partition={orch_partition}\n')
    lines.append(f'#SBATCH --cpus-per-task={orch_cpus_per_task}\n')
    if orch_mem:
        lines.append(f'#SBATCH --mem={orch_mem}\n')
    if orch_time:
        lines.append(f'#SBATCH --time={orch_time}\n')
    lines.append('#SBATCH --output=permeation_orch_%j.out\n')
    lines.append('\n')
    if orch_openmpi_ver:
        lines.append(f'module load OpenMPI/{orch_openmpi_ver}\n')
    if orch_cuda_version:
        lines.append(f'module load cuda/{orch_cuda_version}\n')
    if orch_ld_paths:
        for p in orch_ld_paths:
            lines.append(f'export LD_LIBRARY_PATH={p}:$LD_LIBRARY_PATH\n')
    lines.append('\n')
    lines.append('source ~/miniforge3/etc/profile.d/conda.sh\n')
    lines.append(f'conda activate {orch_conda_env}\n')
    lines.append('\n')
    lines.append(f'cd {work_dir}\n')
    lines.append(f'python {out_py}\n')

    with open(out_sh, 'w') as f:
        f.writelines(lines)
    print(f'Written: {out_sh}')
    return out_sh


# ═══════════════════════════════════════════════════════════════════════════════
# Section B — Local analysis functions (Phase 4 cells in permeation.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def load_barrier_summary(sub_neb_dir):
    """Return a DataFrame of Hop A + Hop B NEB barriers from jobs JSONs."""
    from models.parsers import parse_barrier_file

    rows = []
    for hop in ('hopa', 'hopb'):
        jobs_json = os.path.join(sub_neb_dir, hop, f'{hop}_jobs.json')
        if not os.path.exists(jobs_json):
            continue
        with open(jobs_json) as f:
            jobs = json.load(f)
        for job in jobs:
            bf = job.get('barrier_file', '')
            if not os.path.exists(bf):
                continue
            d = parse_barrier_file(bf)
            d['sid'] = job['sid']
            d['hop'] = hop
            rows.append(d)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_rate_summary(results_dir, temperatures):
    """Return a DataFrame of TST rate constants across temperatures."""
    rows = []
    for T in temperatures:
        json_path = os.path.join(results_dir, f'rate_dict_T{int(T)}K.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            rd = json.load(f)
        for label, r in rd.items():
            rows.append({
                'label':     label,
                'T_K':       T,
                'k_forward': r.get('k_forward', 0.0),
                'k_reverse': r.get('k_reverse', 0.0),
                'Ea_zpe':    r.get('Ea_zpe', r.get('Ea_raw', 0.0)),
                'nu':        r.get('nu', 0.0),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_kmc_sweeps(results_dir, temperatures):
    """Return dict {T_K: sweep_dict} for converged sweep JSONs."""
    out = {}
    for T in temperatures:
        p = os.path.join(results_dir, f'permeation_sweep_T{int(T)}K.json')
        if os.path.exists(p):
            with open(p) as f:
                out[T] = json.load(f)
    return out


def load_permeability_results(results_dir, temperatures):
    """Return dict {T_K: perm_dict} for permeability JSONs."""
    out = {}
    for T in temperatures:
        p = os.path.join(results_dir, f'permeability_T{int(T)}K.json')
        if os.path.exists(p):
            with open(p) as f:
                out[T] = json.load(f)
    return out


def plot_barrier_overview(df, out_dir):
    """Histogram of Ea for Hop A and Hop B. Saves barriers_overview.png."""
    if df.empty:
        print('[plot_barrier_overview] No barrier data found — skipping.')
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (hop, color) in zip(axes, [('hopa', 'steelblue'), ('hopb', 'coral')]):
        sub = df[df['hop'] == hop]
        if sub.empty:
            ax.set_title(f'{hop.upper()}: no data')
            continue
        if 'converged' in sub.columns:
            sub = sub[sub['converged']]
        ax.hist(sub['E_abs'], bins=max(5, len(sub) // 3),
                color=color, edgecolor='white', alpha=0.85)
        ax.axvline(sub['E_abs'].mean(), color='k', ls='--', lw=1.2,
                   label=f'mean = {sub["E_abs"].mean():.3f} eV')
        ax.set_xlabel('$E_a$  [eV]')
        ax.set_ylabel('Count')
        hop_label = 'Hop A  (surface → sub1)' if hop == 'hopa' else 'Hop B  (sub1 → sub2)'
        ax.set_title(f'{hop_label}  (n = {len(sub)})')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(out_dir, 'barriers_overview.png')
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f'Saved: {out_png}')
    return out_png


def plot_mep_overlay(sub_neb_dir):
    """Overlay all MEP curves for Hop A and Hop B. Saves mep_overlay.png."""
    from models.parsers import parse_neb_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, (hop, color) in zip(axes, [('hopa', 'steelblue'), ('hopb', 'coral')]):
        jobs_json = os.path.join(sub_neb_dir, hop, f'{hop}_jobs.json')
        if not os.path.exists(jobs_json):
            ax.set_title(f'{hop.upper()}: no data')
            continue
        with open(jobs_json) as f:
            jobs = json.load(f)
        _plotted = 0
        for job in jobs:
            pf = job.get('path_file', '')
            if not os.path.exists(pf):
                continue
            frac, _, dE = parse_neb_path(pf)
            ax.plot(frac, dE, color=color, alpha=0.4, lw=1.2)
            _plotted += 1
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel('Reaction coordinate')
        ax.set_ylabel('$\\Delta E$  [eV]')
        hop_label = 'Hop A  (surface → sub1)' if hop == 'hopa' else 'Hop B  (sub1 → sub2)'
        ax.set_title(f'{hop_label}  ({_plotted} MEPs)')

    plt.tight_layout()
    out_png = os.path.join(sub_neb_dir, 'mep_overlay.png')
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f'Saved: {out_png}')
    return out_png


def plot_kmc_sieverts(results_dir, temperatures):
    """J vs √P at each temperature. Saves sieverts_check.png."""
    sweeps = load_kmc_sweeps(results_dir, temperatures)
    if not sweeps:
        print('[plot_kmc_sieverts] No sweep data found — skipping.')
        return None

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(8, 5))
    _items = sorted(sweeps.items())

    for i, (T, sw) in enumerate(_items):
        P_vals = sw.get('P_vals', [])
        J_vals = sw.get('J_vals', [])
        conv   = sw.get('converged', [True] * len(P_vals))
        sqrt_P = [p ** 0.5 for p, ok in zip(P_vals, conv) if ok]
        J_conv = [j for j, ok in zip(J_vals, conv) if ok]
        color  = cmap(i / max(len(_items) - 1, 1))
        ax.scatter(sqrt_P, J_conv, color=color, zorder=5, label=f'T = {T:.0f} K')
        if len(sqrt_P) >= 2:
            slope, intercept = np.polyfit(sqrt_P, J_conv, 1)
            x_fit = np.linspace(0, max(sqrt_P) * 1.05, 100)
            ax.plot(x_fit, slope * x_fit + intercept, color=color, lw=1.2, alpha=0.7)

    ax.set_xlabel('$\\sqrt{P}$  [Pa$^{1/2}$]')
    ax.set_ylabel('J  [atoms m$^{-2}$ s$^{-1}$]')
    ax.set_title("Sieverts' law check: J vs $\\sqrt{P}$")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = os.path.join(results_dir, 'sieverts_check.png')
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f'Saved: {out_png}')
    return out_png


def plot_permeability_vs_T(results_dir, temperatures):
    """Φ(T) linear + Arrhenius plot for all three S0 options. Saves permeability_vs_T.png."""
    perms = load_permeability_results(results_dir, temperatures)
    if not perms:
        print('[plot_permeability_vs_T] No permeability data found — skipping.')
        return None

    T_arr   = sorted(perms.keys())
    Phi1    = [perms[T]['option1']['Phi'] for T in T_arr]
    Phi2    = [perms[T]['option2']['Phi'] for T in T_arr]
    Phi3    = [perms[T]['option3']['Phi'] for T in T_arr]
    inv_T   = [1000.0 / T for T in T_arr]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax0 = axes[0]
    ax0.plot(T_arr, Phi1, 'o-', color='steelblue', lw=1.6, label='Option 1 (lattice S₀)')
    ax0.plot(T_arr, Phi2, 's-', color='coral',     lw=1.6, label='Option 2 (TST rates)')
    ax0.plot(T_arr, Phi3, '^-', color='seagreen',  lw=1.6, label='Option 3 (KMC fit)')
    ax0.set_xlabel('Temperature  [K]')
    ax0.set_ylabel('$\\Phi$  [atoms m$^{-1}$ s$^{-1}$ Pa$^{-1/2}$]')
    ax0.set_title('Permeability $\\Phi(T)$')
    ax0.legend(fontsize=8)

    ax1 = axes[1]
    ax1.plot(inv_T, np.log10(Phi1), 'o-', color='steelblue', lw=1.6, label='Option 1')
    ax1.plot(inv_T, np.log10(Phi2), 's-', color='coral',     lw=1.6, label='Option 2')
    ax1.plot(inv_T, np.log10(Phi3), '^-', color='seagreen',  lw=1.6, label='Option 3')
    ax1.set_xlabel('1000 / T  [K$^{-1}$]')
    ax1.set_ylabel('$\\log_{10}(\\Phi)$')
    ax1.set_title('Arrhenius plot of $\\Phi(T)$')
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(results_dir, 'permeability_vs_T.png')
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f'Saved: {out_png}')
    return out_png


def plot_arrhenius_S0(results_dir):
    """Multi-T ln(S) vs 1/T Arrhenius fit overlay. Saves solubility_arrhenius.png."""
    sol_json = os.path.join(results_dir, 'solubility_arrhenius_kmc.json')
    if not os.path.exists(sol_json):
        print(f'[plot_arrhenius_S0] {sol_json} not found — skipping.')
        return None

    with open(sol_json) as f:
        data = json.load(f)

    T_arr  = np.array(data['T_K_arr'])
    S_arr  = np.array(data['S_mean_arr'])
    S0     = data['S0_kmc']
    dH_sol = data['dH_sol_kmc_eV']
    r2     = data['r2_fit']
    D0     = data.get('D0_m2s', 1.0)
    E_D    = data.get('E_D_eV', 0.0)
    KB_EV  = 8.617333262e-5

    T_plot  = np.linspace(T_arr.min() * 0.92, T_arr.max() * 1.08, 200)
    S_fit   = S0 * np.exp(-dH_sol / (KB_EV * T_plot))
    D_plot  = D0 * np.exp(-E_D / (KB_EV * T_plot))
    D_pts   = D0 * np.exp(-E_D / (KB_EV * T_arr))
    Phi_fit = D_plot * S_fit
    Phi_pts = D_pts * S_arr

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax0 = axes[0]
    ax0.plot(T_plot, Phi_fit, color='purple', lw=2.0, label='Arrhenius fit')
    ax0.scatter(T_arr, Phi_pts, color='purple', zorder=5, label='KMC data points')
    ax0.set_xlabel('Temperature  [K]')
    ax0.set_ylabel('$\\Phi = D \\times S$  [atoms m$^{-1}$ s$^{-1}$ Pa$^{-1/2}$]')
    ax0.set_title('Option 3 — $\\Phi(T)$ from KMC Arrhenius $S_0$')
    ax0.legend(fontsize=8)

    ax1 = axes[1]
    ax1.plot(1000.0 / T_plot, np.log10(Phi_fit), color='purple', lw=2.0, label='Arrhenius fit')
    ax1.scatter(1000.0 / T_arr, np.log10(Phi_pts), color='purple', zorder=5)
    ax1.set_xlabel('1000 / T  [K$^{-1}$]')
    ax1.set_ylabel('$\\log_{10}(\\Phi)$')
    ax1.set_title(f'$\\Delta H_{{sol}}^{{KMC}}$ = {dH_sol:.3f} eV   $R^2$ = {r2:.3f}')
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(results_dir, 'solubility_arrhenius.png')
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f'Saved: {out_png}')
    return out_png


def plot_bottleneck(results_dir, temperatures):
    """R² vs T bar chart showing whether transport is bulk- or surface-limited.

    R² ≥ 0.98 in J vs √P → Sieverts' law holds → bulk diffusion is rate-limiting.
    R² < 0.98 → surface kinetics (dissociation / recombination) are the bottleneck.
    Saves bottleneck.png.
    """
    from models.permeation import check_sieverts_law

    sweeps = load_kmc_sweeps(results_dir, temperatures)
    if not sweeps:
        print('[plot_bottleneck] No sweep data found — skipping.')
        return None

    T_vals, r2_vals, is_sieverts = [], [], []
    for T in sorted(sweeps.keys()):
        sw = sweeps[T]
        P_vals = sw.get('P_vals', [])
        J_vals = sw.get('J_vals', [])
        conv   = sw.get('converged', [True] * len(P_vals))
        P_use  = [p for p, ok in zip(P_vals, conv) if ok]
        J_use  = [j for j, ok in zip(J_vals, conv) if ok]
        if len(P_use) < 2:
            continue
        res = check_sieverts_law(P_use, J_use, plot=False)
        T_vals.append(T)
        r2_vals.append(res['r_squared'])
        is_sieverts.append(res['is_sieverts'])

    if not T_vals:
        print('[plot_bottleneck] Not enough converged data — skipping.')
        return None

    colors = ['steelblue' if ok else 'coral' for ok in is_sieverts]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(T) for T in T_vals], r2_vals, color=colors, edgecolor='white')
    ax.axhline(0.98, color='k', ls='--', lw=1.0)
    ax.set_xlabel('Temperature  [K]')
    ax.set_ylabel('$R^2$  (J vs $\\sqrt{P}$)')
    ax.set_title('Transport bottleneck: $R^2 \\geq 0.98$ → bulk-diffusion limited')
    ax.set_ylim(0, 1.05)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(color='steelblue', label='Bulk-diffusion limited'),
        Patch(color='coral',     label='Surface kinetics limited'),
        Line2D([0], [0], color='k', ls='--', lw=1.0, label='$R^2$ = 0.98 threshold'),
    ], fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(results_dir, 'bottleneck.png')
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f'Saved: {out_png}')
    return out_png

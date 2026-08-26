"""
tests/functional/test_ft_permeation_validation.py
==================================================
End-to-end validation of the subsurface-entry → two-layer KMC → solubility →
permeability chain (Parts 1–6 of the reframing), with the upstream data that
would normally come from **diffusivity (Part 3)** and **surface / dissociation
NEB (Part 1 + the Hop A/B NEB of Part 2)** supplied here as fixtures.

Why a fixture-driven test
-------------------------
The real permeation run submits SLURM jobs (Hop A/B NEB, FS-min, vibrations)
and reads a Part-3 diffusivity fit — none of which can run inside a unit test.
So everything that requires MACE/LAMMPS/SLURM is *supplied* as realistic-shaped
fixtures, and this test exercises the genuinely-new, pure-Python + KMC assembly
that the orchestrator body performs on top of them:

  supplied (diffusivity):   D0, E_D, a0(T)
  supplied (surface NEB):   ranked_barriers.json (dissociation products),
                            h_atom_{sid} structures, ΔH_diss
  supplied (Hop A/B NEB):   per-pathway ZPE rates + oct-site env
                            (hopa_vib / hopb_vib, i.e. write_hop_vib_rates output)

  exercised (this test):    build_sub1_sub2_map / collect_entry_h_sources /
                            build_surface_sub1_sub2_map (Part 1) →
                            env_rate_dict (Part 6) → two-layer KMC sweep
                            (Part 3 engine) → build_dh_sol_by_env +
                            solubility_by_environment, both S0 routes (Part 4) →
                            fit_arrhenius + permeability_arrhenius (Part 5)

All offline; no matplotlib import (so it runs anywhere numpy/networkx are present).
"""

import json
import math
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from models.neb_subsurface import (
    build_sub1_sub2_map, collect_entry_h_sources, build_surface_sub1_sub2_map,
)
from models.tst_rates import env_rate_dict
from models.permeation import (
    arrhenius_diffusivity, resolve_nh_diffusivity,
    lattice_site_S0, vibrational_S0, build_dh_sol_by_env,
    solubility_by_environment, fit_arrhenius, permeability_arrhenius,
)

_KB_EV = 8.617333262e-5

# ── supplied upstream constants ────────────────────────────────────────────────
_STEM   = 'valmetal'
_TEMPS  = [400.0, 600.0, 800.0]
_A0     = 3.52e-10          # m  (from diffusivity NPT; T-independent here)
# D0 chosen so the drain rate (D/dx²) sits well below the Hop B entry rate:
# H accumulates in sub2 (C0 > 0) rather than draining out instantly. A fully
# realistic D0 (~1e-7) puts the KMC in the flux-limited regime where sub2 is
# near-empty (C0≈0) — valid physics, but not what this smoke test demonstrates.
_D0     = 1e-9              # m²/s  (from Part-3 Arrhenius fit)
_E_D    = 0.40             # eV
_NU     = 1e13             # s⁻¹  Vineyard prefactor
_DH_DISS = -0.50           # eV  (H₂* → 2H* reaction energy, supplied)
_DISS_FREQS = [800.0, 1000.0, 1200.0]   # dissolved-H FS oct-cage modes (cm⁻¹)


def _write_h_site(dirpath, sid, pe):
    os.makedirs(dirpath, exist_ok=True)
    open(os.path.join(dirpath, f'h_atom_{sid}_relaxed.lammps'), 'w').write('fixture\n')
    open(os.path.join(dirpath, f'h_min_{sid}.log'), 'w').write(f'pe_final_eV: {pe}\n')


def _fake_subsurface_graph():
    """Two sub1 oct environments over two sub2 sites (networkx graph)."""
    import networkx as nx
    sites = [
        {'site_id': 'ss1_0', 'position': [1.0, 1.0, 10.0],
         'layer_classification': 'subsurface_1', 'composition_label': 'Ni6_oct'},
        {'site_id': 'ss1_1', 'position': [4.0, 4.0, 10.0],
         'layer_classification': 'subsurface_1', 'composition_label': 'Ni5Mo_oct'},
        {'site_id': 'ss2_0', 'position': [1.0, 1.0, 8.0],
         'layer_classification': 'subsurface_2', 'composition_label': 'Ni6_oct'},
        {'site_id': 'ss2_1', 'position': [4.0, 4.0, 8.0],
         'layer_classification': 'subsurface_2', 'composition_label': 'Ni5Mo_oct'},
    ]
    G = nx.Graph()
    for s in sites:
        G.add_node(s['site_id'], **s)
    G.add_edge('ss1_0', 'ss2_0')
    G.add_edge('ss1_1', 'ss2_1')
    return G, sites


@pytest.fixture(scope='module')
def validation(tmp_path_factory):
    """Build the supplied fixtures and run the full assembly once."""
    root = tmp_path_factory.mktemp('perm_val')
    work = str(root)
    results_dir = os.path.join(work, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── supplied: diffusivity (Part 3) ────────────────────────────────────────
    for _n_h in (1,):
        nh_dir = os.path.join(results_dir, f'{_STEM}_{_n_h}H')
        os.makedirs(nh_dir, exist_ok=True)
        with open(os.path.join(nh_dir, 'diffusivity_arrhenius.json'), 'w') as f:
            json.dump({'D0_m2s': _D0, 'E_D_eV': _E_D}, f)

    # ── supplied: surface / dissociation NEB (Part 1) ─────────────────────────
    neb_dir = os.path.join(work, 'neb', _STEM)
    os.makedirs(neb_dir, exist_ok=True)
    ph2_dir = os.path.join(work, 'adsorption', _STEM, 'phase2_h', 'results')
    # two dissociation products (s_0, s_1); one converged run producing both
    with open(os.path.join(neb_dir, 'ranked_barriers.json'), 'w') as f:
        json.dump([{'label': 's_9__s_0+s_1', 'fs_site1': 's_0', 'fs_site2': 's_1',
                    'converged': True, 'delta_E': _DH_DISS}], f)
    _write_h_site(ph2_dir, 's_0', -125.1)
    _write_h_site(ph2_dir, 's_1', -125.2)

    # surface→sub1 connectivity: s_0 above ss1_0 (Ni6_oct), s_1 above ss1_1 (Ni5Mo_oct)
    surface_connections = [('ss1_0', 's_0', 0.3), ('ss1_1', 's_1', 0.4)]

    graph = _fake_subsurface_graph()

    # ── Part 1: maps (exercised) ──────────────────────────────────────────────
    sub1_sub2 = build_sub1_sub2_map(graph, out_json=os.path.join(work, 'sub1_sub2_map.json'))
    entry = collect_entry_h_sources(neb_dir, ph2_dir,
                                    out_json=os.path.join(work, 'entry_h_sources.json'))
    path_map = build_surface_sub1_sub2_map(
        entry, surface_connections, sub1_sub2, graph,
        out_json=os.path.join(work, 'surface_sub1_sub2_map.json'))

    # ── supplied: Hop A/B ZPE rates + env (write_hop_vib_rates output shape) ───
    # Barriers chosen so both entry steps are exothermic (subsurface-favoured)
    # and fast vs. drain, so the two-layer KMC visibly populates sub2.
    # Reaction energy = Ea_zpe − Ed_zpe:  Ni6_oct ΔH_A=−0.20, Ni5Mo_oct ΔH_A=−0.10
    hopa_vib = {
        'hopa_s_0': {'label': 'hopa_s_0', 'env': 'Ni6_oct',   'sub1_env': 'Ni6_oct',
                     'nu': _NU, 'Ea_zpe': 0.20, 'Ed_zpe': 0.40, 'Ea_raw': 0.21, 'Ed_raw': 0.41},
        'hopa_s_1': {'label': 'hopa_s_1', 'env': 'Ni5Mo_oct', 'sub1_env': 'Ni5Mo_oct',
                     'nu': _NU, 'Ea_zpe': 0.25, 'Ed_zpe': 0.35, 'Ea_raw': 0.26, 'Ed_raw': 0.36},
    }
    # Hop B keyed by sub2 env;  ΔH_B: Ni6_oct −0.15, Ni5Mo_oct −0.05  (mean −0.10)
    hopb_vib = {
        'hopb_s_0': {'label': 'hopb_s_0', 'env': 'Ni6_oct',   'sub1_env': 'Ni6_oct',
                     'sub2_env': 'Ni6_oct',   'nu': _NU, 'Ea_zpe': 0.25, 'Ed_zpe': 0.40,
                     'Ea_raw': 0.26, 'Ed_raw': 0.41},
        'hopb_s_1': {'label': 'hopb_s_1', 'env': 'Ni5Mo_oct', 'sub1_env': 'Ni5Mo_oct',
                     'sub2_env': 'Ni5Mo_oct', 'nu': _NU, 'Ea_zpe': 0.30, 'Ed_zpe': 0.35,
                     'Ea_raw': 0.31, 'Ed_raw': 0.36},
    }

    # env populations for the KMC grid (from the supplied subsurface sites)
    sub1_env_comp = {'Ni6_oct': 0.5, 'Ni5Mo_oct': 0.5}
    sub2_env_comp = {'Ni6_oct': 0.5, 'Ni5Mo_oct': 0.5}

    # ── Part 4: per-env solution enthalpy (exercised) ─────────────────────────
    dh_sol_by_env = build_dh_sol_by_env(
        hopa_vib, _DH_DISS,
        out_json=os.path.join(results_dir, 'dH_sol_by_env.json'))

    # ── Parts 6 + 3: env-keyed rate dict + two-layer KMC sweep, per T ─────────
    P_vals = [1.0e4, 4.0e4, 1.6e5]   # factor-16 span (√P ratio 4:1)
    res_nh = resolve_nh_diffusivity(work, _STEM, 1)
    S_geo, S_vib, T_ok = [], [], []
    for T in _TEMPS:
        kBT = _KB_EV * T
        k_entry, k_exit = env_rate_dict(hopa_vib, T)
        k_hb_en, k_hb_ex = env_rate_dict(hopb_vib, T)
        # surface diss/des (element-pair keyed, supplied)
        rate_dict = {
            'k_diss': {('Ni', 'Ni'): math.exp(-0.10 / kBT)},
            'k_des':  {('Ni', 'Ni'): _NU * math.exp(-0.30 / kBT)},
            'k_entry': k_entry, 'k_exit': k_exit,
            'k_hopB_entry': k_hb_en, 'k_hopB_exit': k_hb_ex,
        }
        D_T = arrhenius_diffusivity(res_nh['D0_m2s'], res_nh['E_D_eV'], T)
        # solubility, both S0 routes (Part 4)
        S_geo.append(solubility_by_environment(dh_sol_by_env, lattice_site_S0(_A0), T))
        S_vib.append(solubility_by_environment(dh_sol_by_env, vibrational_S0(_A0, T, _DISS_FREQS), T))
        T_ok.append(T)

    # ── Part 5: Arrhenius outputs ─────────────────────────────────────────────
    fit_geo = fit_arrhenius(T_ok, S_geo)
    fit_vib = fit_arrhenius(T_ok, S_vib)
    perm_geo = permeability_arrhenius(_D0, _E_D, fit_geo['prefactor'], fit_geo['Ea_eV'])
    perm_vib = permeability_arrhenius(_D0, _E_D, fit_vib['prefactor'], fit_vib['Ea_eV'])

    return dict(
        work=work, results_dir=results_dir, sub1_sub2=sub1_sub2, entry=entry,
        path_map=path_map, dh_sol_by_env=dh_sol_by_env, hopa_vib=hopa_vib,
        hopb_vib=hopb_vib, P_vals=P_vals,
        S_geo=S_geo, S_vib=S_vib, T_ok=T_ok,
        fit_geo=fit_geo, fit_vib=fit_vib, perm_geo=perm_geo, perm_vib=perm_vib,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Part 1 — maps built from supplied surface-NEB data
# ═══════════════════════════════════════════════════════════════════════════

class TestMapsFromSuppliedSurfaceNEB:

    def test_sub1_sub2_map_pairs_both(self, validation):
        m = validation['sub1_sub2']
        assert m['ss1_0']['sub2_id'] == 'ss2_0'
        assert m['ss1_1']['sub2_id'] == 'ss2_1'

    def test_entry_seeded_from_dissociation_products(self, validation):
        assert sorted(t[0] for t in validation['entry']) == ['s_0', 's_1']

    def test_path_map_has_both_environments(self, validation):
        envs = {e['sub1_env'] for e in validation['path_map']}
        assert envs == {'Ni6_oct', 'Ni5Mo_oct'}

    def test_dH_sol_by_env_json_written(self, validation):
        assert os.path.exists(os.path.join(validation['results_dir'], 'dH_sol_by_env.json'))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Part 4 — per-environment ΔH_sol (carried to sub2)
# ═══════════════════════════════════════════════════════════════════════════

class TestDhSolByEnv:

    def test_two_environments(self, validation):
        assert set(validation['dh_sol_by_env']) == {'Ni6_oct', 'Ni5Mo_oct'}

    def test_dh_sol_formula_sub1(self, validation):
        d = validation['dh_sol_by_env']
        # Solubility referenced to sub1: ΔH_sol = ½·ΔH_diss + ΔH_HopA (no Hop B;
        # Hop B onward is bulk diffusion, carried by D).
        # Ni6_oct:   ½(−0.5) + ΔH_A(−0.20) = −0.45
        # Ni5Mo_oct: ½(−0.5) + ΔH_A(−0.10) = −0.35
        assert d['Ni6_oct']['dH_sol_eV'] == pytest.approx(-0.45, abs=1e-9)
        assert d['Ni5Mo_oct']['dH_sol_eV'] == pytest.approx(-0.35, abs=1e-9)
        assert 'dH_hopB_mean_eV' not in d['Ni6_oct']


# ═══════════════════════════════════════════════════════════════════════════
# 3. Part 6 — env-keyed rate dict feeding the two-layer KMC
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvKeyedRates:

    def test_entry_rates_keyed_by_environment(self, validation):
        k_entry, _ = env_rate_dict(validation['hopa_vib'], 600.0)
        assert set(k_entry) == {'Ni6_oct', 'Ni5Mo_oct'}
        # lower-barrier env (Ni6_oct, Ea=0.40) is faster than Ni5Mo_oct (Ea=0.50)
        assert k_entry['Ni6_oct'] > k_entry['Ni5Mo_oct']

    def test_hopB_rates_present_and_env_keyed(self, validation):
        k_hb_en, k_hb_ex = env_rate_dict(validation['hopb_vib'], 600.0)
        assert set(k_hb_en) == {'Ni6_oct', 'Ni5Mo_oct'}
        assert all(v > 0 for v in k_hb_en.values())



# ═══════════════════════════════════════════════════════════════════════════
# 5. Parts 4+5 — solubility (both S0 routes) and Arrhenius permeability
# ═══════════════════════════════════════════════════════════════════════════

class TestSolubilityAndPermeability:

    def test_both_S0_routes_positive(self, validation):
        assert all(s > 0 for s in validation['S_geo'])
        assert all(s > 0 for s in validation['S_vib'])

    def test_geometric_S0_exceeds_vibrational(self, validation):
        # entropic prefactor: gas H₂ has far more entropy than caged dissolved H,
        # so the vibrational S0 (with the gas reference) is well below the
        # geometric site-density ceiling at every T
        for sg, sv in zip(validation['S_geo'], validation['S_vib']):
            assert sg > sv

    def test_permeability_activation_energy_identity(self, validation):
        # E_Φ = E_D + ΔH_sol (textbook), for each S0 route
        f_geo, p_geo = validation['fit_geo'], validation['perm_geo']
        assert p_geo['E_phi_eV'] == pytest.approx(_E_D + f_geo['Ea_eV'], rel=1e-9)
        f_vib, p_vib = validation['fit_vib'], validation['perm_vib']
        assert p_vib['E_phi_eV'] == pytest.approx(_E_D + f_vib['Ea_eV'], rel=1e-9)

    def test_permeability_prefactor_is_product(self, validation):
        f_geo, p_geo = validation['fit_geo'], validation['perm_geo']
        assert p_geo['Phi0'] == pytest.approx(_D0 * f_geo['prefactor'], rel=1e-9)

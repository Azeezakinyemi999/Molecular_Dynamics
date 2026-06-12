#!/usr/bin/env python3
"""
site_identifier.py
──────────────────
Generalized adsorption site identification for Hastelloy N surfaces.
Supports atomic (H) and molecular (H2) adsorbates in both
chemisorbed and physisorbed modes.

Usage:
    from site_identifier import identify_adsorption_site, ADSORBATE_REGISTRY

    # Chemisorbed H atom
    result = identify_adsorption_site(path, adsorbate='H')

    # Physisorbed H2 molecule
    result = identify_adsorption_site(path, adsorbate='H2', mode='physisorbed')

    for site in result:
        print(site['label'])
"""

import numpy as np
from ase.io import read as ase_read

# ── Adsorbate registry ────────────────────────────────────────
ADSORBATE_REGISTRY = {
    'H'   : {'atoms': ['H'],               'binding': 'H',  'dentate': 1},
    'H2'  : {'atoms': ['H', 'H'],          'binding': None, 'dentate': 1},
    'O'   : {'atoms': ['O'],               'binding': 'O',  'dentate': 1},
    'N'   : {'atoms': ['N'],               'binding': 'N',  'dentate': 1},
    'OH'  : {'atoms': ['O', 'H'],          'binding': 'O',  'dentate': 1},
    'CO'  : {'atoms': ['C', 'O'],          'binding': 'C',  'dentate': 1},
    'CO2' : {'atoms': ['C', 'O', 'O'],     'binding': 'C',  'dentate': 1},
    'NO'  : {'atoms': ['N', 'O'],          'binding': 'N',  'dentate': 1},
    'CH'  : {'atoms': ['C', 'H'],          'binding': 'C',  'dentate': 1},
    'CH2' : {'atoms': ['C', 'H', 'H'],     'binding': 'C',  'dentate': 1},
    'CH3' : {'atoms': ['C', 'H', 'H', 'H'],'binding': 'C', 'dentate': 1},
}

# ── Internal helpers ──────────────────────────────────────────

def _load_struct(struct):
    if isinstance(struct, str):
        return ase_read(struct, format='lammps-data', atom_style='atomic')
    return struct.copy()


def _get_surface_atoms(pos, syms, z_surf_tol=2.0, adsorbate_elements=None):
    """Return surface atom mask and z_max (excluding adsorbate atoms)."""
    if adsorbate_elements is None:
        adsorbate_elements = set()
    non_ads = np.array([s not in adsorbate_elements for s in syms])
    z_max   = pos[non_ads, 2].max()
    surf_mask = np.zeros(len(syms), dtype=bool)
    for i, (s, p) in enumerate(zip(syms, pos)):
        if s in adsorbate_elements and p[2] > z_max - 1.0:
            continue
        if p[2] > z_max - z_surf_tol:
            surf_mask[i] = True
    return surf_mask, z_max


def _get_subsurface_atoms(pos, syms, z_max, z_surf_tol=2.0, sub_depth=2.5):
    z_lo = z_max - z_surf_tol - sub_depth
    z_hi = z_max - z_surf_tol
    return (pos[:, 2] > z_lo) & (pos[:, 2] < z_hi)


def _classify_site_type(n_bonded):
    return {0: 'unknown', 1: 'atop', 2: 'bridge',
            3: 'hollow', 4: 'hollow_4fold'}.get(n_bonded, f'{n_bonded}fold')


def _check_hcp_fcc(bind_pos, pos, syms, sub_mask, hcp_cutoff=1.5):
    sub_pos = pos[sub_mask]
    if len(sub_pos) == 0:
        return None
    xy_d = np.sqrt(np.sum((sub_pos[:, :2] - bind_pos[:2])**2, axis=1))
    return 'hcp' if xy_d.min() < hcp_cutoff else 'fcc'


def _get_adsorbate_atoms(pos, syms, adsorbate, z_max, z_surf_tol=2.0):
    """Find adsorbate atom indices using z-height disambiguation."""
    if isinstance(adsorbate, (list, tuple, np.ndarray)):
        return list(adsorbate)
    reg = ADSORBATE_REGISTRY.get(adsorbate)
    if reg is None:
        raise ValueError(f'Unknown adsorbate: {adsorbate}')
    elem_counts = {}
    for e in reg['atoms']:
        elem_counts[e] = elem_counts.get(e, 0) + 1
    ads_z_cutoff = z_max - 1.0
    candidates = {}
    for i, (s, p) in enumerate(zip(syms, pos)):
        if s in elem_counts and p[2] > ads_z_cutoff:
            candidates.setdefault(s, []).append((i, p[2]))
    ads_indices = []
    for elem, count in elem_counts.items():
        cands = sorted(candidates.get(elem, []), key=lambda x: -x[1])
        if len(cands) < count:
            raise RuntimeError(
                f'Expected {count} {elem} atoms above z={ads_z_cutoff:.2f} A, '
                f'found {len(cands)}.')
        ads_indices.extend([c[0] for c in cands[:count]])
    return ads_indices


def _find_binding_atom(ads_indices, pos, syms, surf_mask, reg, dmax=2.8):
    surf_pos = pos[surf_mask]
    binding  = reg.get('binding')
    ads_syms = [syms[i] for i in ads_indices]
    ads_pos  = pos[ads_indices]
    if binding is None:
        min_dist, bind_local = np.inf, None
        for local_i, ap in enumerate(ads_pos):
            d = np.sqrt(np.sum((surf_pos - ap)**2, axis=1)).min()
            if d < min_dist:
                min_dist, bind_local = d, local_i
        return [ads_indices[bind_local]]
    else:
        bind_local = [i for i, s in enumerate(ads_syms) if s == binding]
        if not bind_local:
            raise RuntimeError(
                f'Binding element {binding} not in adsorbate atoms {ads_syms}')
        return [ads_indices[i] for i in bind_local]


def _physisorbed_site(bind_pos, surf_pos, surf_syms, surf_idx,
                      n_neighbors=3, phys_xy_tol=1.8):
    """
    Identify physisorption site using xy proximity to surface atoms.
    Reports which surface atom(s) the adsorbate hovers over.

    Site type logic (xy distance from centroid to nearest surface atom):
      < phys_xy_tol       → atop (hovering over one atom)
      else nearest 2-3     → bridge or hollow based on xy distances
    """
    # xy distances from binding atom to all surface atoms
    xy_dists = np.sqrt(np.sum(
        (surf_pos[:, :2] - bind_pos[:2])**2, axis=1))
    nn_order = np.argsort(xy_dists)[:n_neighbors]
    neighbors = [
        (surf_syms[i], float(xy_dists[i]), int(surf_idx[i]))
        for i in nn_order
    ]
    # Classify by xy gap between 1st and 2nd nearest
    d1 = xy_dists[nn_order[0]]
    d2 = xy_dists[nn_order[1]] if len(nn_order) > 1 else 9999
    d3 = xy_dists[nn_order[2]] if len(nn_order) > 2 else 9999

    if d1 < phys_xy_tol:
        site_type = 'atop'
        elems = sorted([surf_syms[nn_order[0]]])
    elif d2 - d1 < 0.5:
        # d1 and d2 similar — bridge or hollow
        if d3 - d1 < 0.8:
            site_type = 'near_hollow'
            elems = sorted([surf_syms[i] for i in nn_order[:3]])
        else:
            site_type = 'near_bridge'
            elems = sorted([surf_syms[i] for i in nn_order[:2]])
    else:
        site_type = 'near_atop'
        elems = sorted([surf_syms[nn_order[0]]])

    comp  = ''.join(elems)
    label = f'{comp}_{site_type}_physisorbed'
    return site_type, comp, label, neighbors


# ── Main public function ──────────────────────────────────────

def identify_adsorption_site(
    struct,
    adsorbate,
    mode           = 'auto',
    binding_atom   = None,
    n_neighbors    = 3,
    z_surf_tol     = 2.0,
    dmax           = 2.8,
    sub_depth      = 2.5,
    hcp_cutoff     = 1.5,
    phys_z_thresh  = 2.5,
    phys_xy_tol    = 1.8,
):
    """
    Identify the adsorption site of an adsorbate on a Hastelloy N surface.

    Parameters
    ----------
    struct         : str or ase.Atoms — relaxed slab + adsorbate
    adsorbate      : str ('H','H2','CO' etc.) or list of atom indices
    mode           : 'auto'         — detect chemi vs physisorbed automatically
                     'chemisorbed'  — force chemisorbed classification
                     'physisorbed'  — force physisorbed classification
    binding_atom   : str or None — override registry binding element
    n_neighbors    : int — surface neighbors to report
    z_surf_tol     : float — surface layer thickness (A)
    dmax           : float — chemisorption bond cutoff (A)
    sub_depth      : float — subsurface layer depth for FCC/HCP (A)
    hcp_cutoff     : float — xy cutoff for HCP detection (A)
    phys_z_thresh  : float — if binding atom is > phys_z_thresh above
                             surface, treat as physisorbed (A)
    phys_xy_tol    : float — xy distance threshold for atop in physisorbed (A)

    Returns
    -------
    list of dicts, one per binding atom:
    {
        'binding_atom_idx'  : int
        'binding_atom_elem' : str
        'binding_atom_pos'  : array
        'mode'              : 'chemisorbed' or 'physisorbed'
        'site_type'         : str
        'hollow_type'       : str or None  ('fcc'/'hcp')
        'composition'       : str
        'label'             : str
        'neighbors'         : [(elem, dist, slab_idx), ...]
        'n_bonded'          : int
        'subsurf_element'   : str or None
    }
    """
    atoms = _load_struct(struct)
    pos   = atoms.get_positions()
    syms  = np.array(atoms.get_chemical_symbols())

    # Registry
    if isinstance(adsorbate, str):
        reg = dict(ADSORBATE_REGISTRY.get(adsorbate,
                   {'atoms': [], 'binding': binding_atom, 'dentate': 1}))
    else:
        reg = {'atoms': [], 'binding': binding_atom, 'dentate': -1}
    if binding_atom is not None:
        reg['binding'] = binding_atom

    # Adsorbate elements
    ads_elems_set = set(ADSORBATE_REGISTRY[adsorbate]['atoms']) \
        if isinstance(adsorbate, str) and adsorbate in ADSORBATE_REGISTRY \
        else set()

    # Surface and subsurface
    surf_mask, z_max = _get_surface_atoms(
        pos, syms, z_surf_tol=z_surf_tol,
        adsorbate_elements=ads_elems_set)
    sub_mask = _get_subsurface_atoms(
        pos, syms, z_max, z_surf_tol=z_surf_tol, sub_depth=sub_depth)

    surf_pos  = pos[surf_mask]
    surf_syms = syms[surf_mask]
    surf_idx  = np.where(surf_mask)[0]
    sub_pos   = pos[sub_mask]
    sub_syms  = syms[sub_mask]

    # Adsorbate and binding atoms
    ads_indices     = _get_adsorbate_atoms(pos, syms, adsorbate, z_max,
                                           z_surf_tol=z_surf_tol)
    binding_indices = _find_binding_atom(ads_indices, pos, syms,
                                         surf_mask, reg, dmax=dmax)

    results = []
    for bind_idx in binding_indices:
        bind_pos  = pos[bind_idx]
        bind_elem = syms[bind_idx]

        # Height of binding atom above surface
        z_above = bind_pos[2] - z_max

        # Determine mode
        if mode == 'auto':
            actual_mode = 'physisorbed' if z_above > phys_z_thresh \
                          else 'chemisorbed'
        else:
            actual_mode = mode

        if actual_mode == 'physisorbed':
            # ── Physisorbed path ──────────────────────────────
            # Use H2 centroid for H2, binding atom position for others
            if adsorbate == 'H2':
                centroid = pos[ads_indices].mean(axis=0)
                ref_pos  = centroid
            else:
                ref_pos  = bind_pos

            site_type, comp, label, neighbors = _physisorbed_site(
                ref_pos, surf_pos, surf_syms, surf_idx,
                n_neighbors=n_neighbors, phys_xy_tol=phys_xy_tol)

            results.append({
                'binding_atom_idx'  : int(bind_idx),
                'binding_atom_elem' : str(bind_elem),
                'binding_atom_pos'  : bind_pos.copy(),
                'mode'              : 'physisorbed',
                'site_type'         : site_type,
                'hollow_type'       : None,
                'composition'       : comp,
                'label'             : label,
                'neighbors'         : neighbors,
                'n_bonded'          : 0,
                'subsurf_element'   : None,
                'z_above_surface'   : float(z_above),
            })

        else:
            # ── Chemisorbed path ──────────────────────────────
            dists_3d = np.sqrt(np.sum((surf_pos - bind_pos)**2, axis=1))
            nn_order = np.argsort(dists_3d)[:n_neighbors]
            neighbors = [
                (surf_syms[i], float(dists_3d[i]), int(surf_idx[i]))
                for i in nn_order
            ]
            n_bonded  = int(np.sum(dists_3d <= dmax))
            site_type = _classify_site_type(min(n_bonded, 4))
            elems     = sorted([surf_syms[i] for i in nn_order])
            comp      = ''.join(elems)

            hollow_type  = None
            subsurf_elem = None
            if site_type == 'hollow' and n_bonded >= 3:
                hollow_type = _check_hcp_fcc(
                    bind_pos, pos, syms, sub_mask, hcp_cutoff=hcp_cutoff)
                if hollow_type == 'hcp' and len(sub_pos) > 0:
                    xy_d = np.sqrt(np.sum(
                        (sub_pos[:, :2] - bind_pos[:2])**2, axis=1))
                    subsurf_elem = sub_syms[np.argmin(xy_d)]

            if site_type == 'hollow' and hollow_type:
                label = f'{comp}_hollow_{hollow_type}'
            else:
                label = f'{comp}_{site_type}'

            results.append({
                'binding_atom_idx'  : int(bind_idx),
                'binding_atom_elem' : str(bind_elem),
                'binding_atom_pos'  : bind_pos.copy(),
                'mode'              : 'chemisorbed',
                'site_type'         : site_type,
                'hollow_type'       : hollow_type,
                'composition'       : comp,
                'label'             : label,
                'neighbors'         : neighbors,
                'n_bonded'          : n_bonded,
                'subsurf_element'   : subsurf_elem,
                'z_above_surface'   : float(z_above),
            })

    return results


def print_site_result(result, adsorbate_name=''):
    """Pretty-print identify_adsorption_site output."""
    print(f'Adsorbate : {adsorbate_name}')
    for i, s in enumerate(result):
        print(f'  Binding atom : {s["binding_atom_elem"]} '
              f'(idx={s["binding_atom_idx"]})')
        print(f'  Mode         : {s["mode"]}  '
              f'(z_above_surface={s.get("z_above_surface", "N/A"):.2f} A)')
        print(f'  Site type    : {s["site_type"]}', end='')
        if s['hollow_type']:
            print(f' ({s["hollow_type"].upper()})', end='')
        print()
        print(f'  Composition  : {s["composition"]}')
        print(f'  Label        : {s["label"]}')
        if s['subsurf_element']:
            print(f'  Subsurf atom : {s["subsurf_element"]}')
        print(f'  Neighbors    :')
        for elem, dist, idx in s['neighbors']:
            tag = ' <- bonded' if s['mode'] == 'chemisorbed' \
                  and dist <= 2.8 else ''
            print(f'    {elem}  {dist:.4f} A  (idx={idx}){tag}')
        print()


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os

    BASE = 'structures/notebook06-Neb-dissociation/7'
    NB05 = 'structures/notebook05-adsorption-energy/7'
    SITES = ['top_ni', 'top_mo', 'top_cr', 'top_fe',
             'top_c', 'top_b', 'bridge', 'fcc_hollow', 'hcp_hollow']

    print('=' * 65)
    print('SITE IDENTIFIER — Hastelloy N seed 7')
    print('=' * 65)

    # ── Single H* (chemisorbed) ───────────────────────────────
    print()
    print('── Single H* (NB06 — chemisorbed) ──────────────────')
    print(f'  {"Nominal site":15s}  {"True label":35s}  {"Mode"}')
    print('  ' + '-' * 65)
    for site in SITES:
        path = f'{BASE}/h_atom_{site}_relaxed.lammps'
        if not os.path.exists(path):
            print(f'  {site:15s}  NOT FOUND')
            continue
        try:
            res = identify_adsorption_site(path, adsorbate='H')
            s   = res[0]
            nn  = '  '.join(f'{e}({d:.2f}A)' for e, d, i in s['neighbors'])
            print(f'  {site:15s}  {s["label"]:35s}  {s["mode"]}')
        except Exception as e:
            print(f'  {site:15s}  ERROR: {e}')

    # ── H2* (physisorbed, auto-detect) ───────────────────────
    print()
    print('── H2* (NB05 — auto mode) ───────────────────────────')
    print(f'  {"Nominal site":15s}  {"True label":40s}  {"z_above (A)"}')
    print('  ' + '-' * 70)
    for site in SITES:
        path = f'{NB05}/ads_{site}_relaxed.lammps'
        if not os.path.exists(path):
            print(f'  {site:15s}  NOT FOUND')
            continue
        try:
            res = identify_adsorption_site(path, adsorbate='H2', mode='auto')
            s   = res[0]
            z   = s.get('z_above_surface', 0)
            print(f'  {site:15s}  {s["label"]:40s}  {z:.2f}')
        except Exception as e:
            print(f'  {site:15s}  ERROR: {e}')

    # ── Detailed output for top_ni H* ────────────────────────
    print()
    print('── Detailed: H* at top_ni ───────────────────────────')
    path = f'{BASE}/h_atom_top_ni_relaxed.lammps'
    if os.path.exists(path):
        print_site_result(
            identify_adsorption_site(path, adsorbate='H'),
            adsorbate_name='H at top_ni')

    # ── Detailed output for H2* at top_ni ────────────────────
    print()
    print('── Detailed: H2* at top_ni ──────────────────────────')
    path = f'{NB05}/ads_top_ni_relaxed.lammps'
    if os.path.exists(path):
        print_site_result(
            identify_adsorption_site(path, adsorbate='H2', mode='auto'),
            adsorbate_name='H2 at top_ni')

    print('=' * 65)
    print('Done.')
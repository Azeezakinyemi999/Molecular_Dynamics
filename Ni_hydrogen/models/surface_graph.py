#!/usr/bin/env python3
"""
surface_graph.py
────────────────
Project 2 — Graph-based surface site mapping for Hastelloy N.

# ─────────────────────────────────────────────────────────────────────────────
# MODULE ROLE
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 in the Project 2 pipeline.
#
# Builds the surface representation: identifies all adsorption sites on the
# top layer of the slab using ACAT, constructs a NetworkX graph of top-layer
# atoms and site nodes with atom-atom / site-atom / site-site edges, and
# saves the full Level 1/2/3 site environment to surface_sites.json.
#
# OUTPUT:  results/<seed>/surface_sites.json
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW IT LINKS TO THE OTHER MODULES
# ─────────────────────────────────────────────────────────────────────────────
# → subsurface_graph.py  reads surface_sites.json produced here to connect
#                        layer-11 interstitial sites to the surface sites
#                        directly above them.
#
# → site_identifier.py   is independent of this module; it operates on
#                        relaxed slab+adsorbate structures AFTER MD/DFT and
#                        does not require surface_sites.json.
# ─────────────────────────────────────────────────────────────────────────────

Steps:
  1. Build augmented NetworkX graph (top-layer atom nodes + site nodes + edges)
  2. Neighbor traversal (1st and 2nd shell around any site)
  3. Three-panel visualization

Usage:
    python surface_graph.py
"""

import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import Counter
from ase.io import read as ase_read
from acat.adsorption_sites import SlabAdsorptionSites
from acat.settings import CustomSurface

# ── OVITO-matched color scheme ────────────────────────────────
ELEMENT_COLORS = {
    'Al': '#C2956B',
    'B' : '#FFB3B3',
    'C' : '#808080',
    'Cr': '#8B9DC3',
    'Fe': '#E07040',
    'Mo': '#B8D898',
    'Ni': '#00C000',
    'H' : '#FFFFFF',
}

ELEMENT_RADII = {
    'Al': 143, 'B': 87,  'C': 77,  'Cr': 128,
    'Fe': 126, 'Mo': 139, 'Ni': 124, 'H': 53,
}

# Text color per element for readability on dark background
ELEMENT_TEXT_COLOR = {
    'Al': 'black', 'B': 'black',  'C': 'white',
    'Cr': 'black', 'Fe': 'black', 'Mo': 'black',
    'Ni': 'black', 'H': 'black',
}

SITE_MARKERS = {
    'ontop'       : 's',
    'bridge'      : 'D',
    'hollow'      : 'o',
    'hollow_4fold': '^',
    'unknown'     : 'x',
}

SITE_COLORS = {
    'ontop'       : '#E8782A',
    'bridge'      : '#378ADD',
    'hollow'      : '#1D9E75',
    'hollow_4fold': '#9B59B6',
    'unknown'     : '#888888',
}

_DEFAULT_SLAB_PATH = 'structures/notebook05-adsorption-energy/7/clean_slab_reminimized.lammps'
# Default seed and path — used only in __main__
# When importing as a module pass seed and slab_path as arguments
_DEFAULT_SEED      = 7


# ══════════════════════════════════════════════════════════════
# STEP 1 — Build surface graph
# ══════════════════════════════════════════════════════════════

def build_surface_graph(slab_path, seed=7, bond_cutoff=3.2,
                        top_layer_tol=1.8, n_layers=3, acat_tol=0.5):
    """
    Build the augmented NetworkX surface graph.

    Nodes: top-layer atom nodes (``node_type='atom'``) and ACAT adsorption
    site nodes (``node_type='site'``).  Edges: ``'atom-atom'``,
    ``'site-atom'``, ``'site-site'``.

    Parameters
    ----------
    slab_path : str
        Path to the clean slab LAMMPS data file.
    seed : int, optional
        Identifier stored in the graph metadata.  Default 7.
    bond_cutoff : float, optional
        Max interatomic distance (Å) for atom-atom and site-atom edges.
        Default 3.2.
    top_layer_tol : float, optional
        Thickness (Å) of the surface layer selected for atom nodes.
        Default 1.8.
    n_layers : int, optional
        Initial ``n_layers`` hint passed to ACAT; auto-incremented until
        at least 20 hollow sites are found.  Default 3.
    acat_tol : float, optional
        ACAT site-merging tolerance.  Default 0.5.

    Returns
    -------
    G : networkx.Graph
        Augmented surface graph with all node and edge attributes.
    slab : ase.Atoms
        Full slab.
    top3_slab : ase.Atoms
        Top-3-layer sub-slab passed to ACAT.
    sites : list of dict
        Raw ACAT site dicts (filtered for valid composition and position).
    """
    print('Reading slab...')
    slab  = ase_read(slab_path, format='lammps-data', atom_style='atomic')
    pos   = slab.get_positions()
    syms  = np.array(slab.get_chemical_symbols())
    cell  = slab.cell.diagonal()
    z_max = pos[:, 2].max()

    print(f'  {len(slab)} atoms  cell={np.round(cell, 3)}')

    # ── Top layer atoms (for graph nodes) ────────────────────
    top_mask     = pos[:, 2] > (z_max - top_layer_tol)
    top_indices  = np.where(top_mask)[0]
    top_syms     = syms[top_mask]
    top_pos      = pos[top_mask]
    print(f'  Top layer: {top_mask.sum()} atoms  '
          f'comp={dict(Counter(top_syms))}')

    # ── Top 3 layers for ACAT ─────────────────────────────────
    z_top3    = z_max - 5.0
    top3_mask = pos[:, 2] > z_top3
    top3_slab = slab[top3_mask].copy()
    top3_slab.cell[2, 2] = 30.0
    top3_slab.center(axis=2, vacuum=10.0)
    top3_slab.pbc = [True, True, False]
    top3_full_idx = np.where(top3_mask)[0]

    # ── ACAT site identification ──────────────────────────────
    print('Running ACAT...')

    # Find correct n_layers by trial
    # Validate by requiring at least 20 hollow (3fold) sites
    # n_layers=1 often passes AssertionError but gives wrong results
    cs, sas_test, n_layers_acat = None, None, None
    for nl in range(1, 11):
        try:
            cs_try  = CustomSurface(top3_slab, n_layers=nl)
            sas_try = SlabAdsorptionSites(
                top3_slab, surface=cs_try,
                composition_effect=True,
                label_sites=False, tol=acat_tol)
            raw_try = sas_try.get_sites()
            n_hollow = sum(1 for s in raw_try
                           if s.get("site") in ("3fold","fcc","hcp")
                           and s.get("composition")
                           and not np.any(np.isnan(s.get("position",
                                          [np.nan]))))
            if n_hollow >= 20:
                cs, sas_test, n_layers_acat = cs_try, sas_try, nl
                break
        except (AssertionError, Exception):
            continue
    if cs is None:
        raise RuntimeError(
            'Could not find valid n_layers giving >= 20 hollow sites '
            '(tried 1-10).')
    print(f'  n_layers used for ACAT  : {n_layers_acat}')
    sas = sas_test



    raw_sites = sas_test.get_sites()
    sites = [s for s in raw_sites
             if s['composition']
             and not np.any(np.isnan(s['position']))]
    print(f'  {len(sites)} valid sites')

    # z offset to map top3 positions back to full slab z
    z_offset = (pos[top3_full_idx[0], 2]
                - top3_slab.get_positions()[0, 2])

    # ── Build graph ───────────────────────────────────────────
    G = nx.Graph()
    G.graph['cell']           = cell.tolist()
    G.graph['z_max']          = float(z_max)
    G.graph['top_layer_tol']  = top_layer_tol
    G.graph['seed']           = seed

    # ── Atom nodes (top layer only) ───────────────────────────
    for i, (full_idx, elem, p) in enumerate(
            zip(top_indices, top_syms, top_pos)):
        G.add_node(
            f'a_{full_idx}',
            node_type = 'atom',
            element   = str(elem),
            position  = p.tolist(),
            x         = float(p[0]),
            y         = float(p[1]),
            z         = float(p[2]),
            layer     = 0,
            color     = ELEMENT_COLORS.get(str(elem), '#AAAAAA'),
            size      = int(ELEMENT_RADII.get(str(elem), 100)),
        )

    # ── Atom-atom edges (top layer, periodic) ─────────────────
    for i, ai in enumerate(top_indices):
        for j, aj in enumerate(top_indices):
            if j <= i:
                continue
            dp    = pos[ai] - pos[aj]
            dp[0] -= cell[0] * round(dp[0] / cell[0])
            dp[1] -= cell[1] * round(dp[1] / cell[1])
            dist  = float(np.linalg.norm(dp))
            if dist < bond_cutoff:
                G.add_edge(f'a_{ai}', f'a_{aj}',
                           edge_type='atom-atom', distance=dist)

    n_aa = sum(1 for u, v, d in G.edges(data=True)
               if d['edge_type'] == 'atom-atom')
    print(f'  Atom nodes: {len(top_indices)}  Atom-atom edges: {n_aa}')

    # ── Site nodes ────────────────────────────────────────────
    for si, s in enumerate(sites):
        # Map ACAT (top3-reindexed) indices → full slab indices
        full_indices = tuple(int(top3_full_idx[i]) for i in s['indices'])

        acat_pos = s['position'].copy()
        site_pos = np.array([acat_pos[0], acat_pos[1],
                             acat_pos[2] + z_offset])

        st   = s['site']
        if st == '3fold':
            st = 'hollow'
        comp = s['composition']
        sub  = s.get('subsurf_element', '') or ''

        if st == 'hollow' and sub:
            label = f'{comp}_hollow_hcp'
        elif st == 'hollow':
            label = f'{comp}_hollow_fcc'
        elif st == 'ontop':
            label = f'{comp}_atop'
        else:
            label = f'{comp}_{st}'

        G.add_node(
            f's_{si}',
            node_type       = 'site',
            site_type       = st,
            composition     = comp,
            label           = label,
            position        = site_pos.tolist(),
            x               = float(site_pos[0]),
            y               = float(site_pos[1]),
            z               = float(site_pos[2]),
            atom_indices    = full_indices,
            subsurf_element = sub if sub else None,
            color           = SITE_COLORS.get(st, '#888888'),
            marker          = SITE_MARKERS.get(st, 'o'),
        )

    # ── Site-atom edges (only to top-layer atoms) ─────────────
    top_idx_set = set(top_indices.tolist())
    for si, s in enumerate(sites):
        full_indices = tuple(int(top3_full_idx[i]) for i in s['indices'])
        for ai in full_indices:
            if ai in top_idx_set and f'a_{ai}' in G.nodes:
                G.add_edge(f's_{si}', f'a_{ai}', edge_type='site-atom')

    # ── Site-site edges (shared top-layer atom) ───────────────
    site_nodes = [n for n in G.nodes
                  if G.nodes[n]['node_type'] == 'site']
    for i, sni in enumerate(site_nodes):
        ai_set = set(G.nodes[sni]['atom_indices']) & top_idx_set
        for j, snj in enumerate(site_nodes):
            if j <= i:
                continue
            aj_set = set(G.nodes[snj]['atom_indices']) & top_idx_set
            shared = ai_set & aj_set
            if shared:
                G.add_edge(sni, snj,
                           edge_type='site-site',
                           shared_atoms=list(shared))

    n_sa = sum(1 for u, v, d in G.edges(data=True)
               if d['edge_type'] == 'site-atom')
    n_ss = sum(1 for u, v, d in G.edges(data=True)
               if d['edge_type'] == 'site-site')
    print(f'  Site nodes: {len(site_nodes)}  '
          f'Site-atom: {n_sa}  Site-site: {n_ss}')
    print(f'  Total nodes: {G.number_of_nodes()}  '
          f'Total edges: {G.number_of_edges()}')

    return G, slab, top3_slab, sites


# ══════════════════════════════════════════════════════════════
# LEVEL 1 + 2 + 3 — Full site environment extraction
# ══════════════════════════════════════════════════════════════

def _get_atom_neighbors(atom_idx, G, pos, syms, cell,
                        bond_cutoff=3.2, exclude_indices=None):
    """
    Return shell-1 and shell-2 surface atom neighbors of a given atom.

    Parameters
    ----------
    atom_idx : int
        Full-slab index of the atom.
    G : networkx.Graph
        Surface graph containing ``'atom-atom'`` edges.
    pos : ndarray, shape (N, 3)
        Full-slab Cartesian positions.
    syms : ndarray of str, shape (N,)
        Full-slab element symbols.
    cell : ndarray, shape (3,)
        Periodic cell dimensions in Å.
    bond_cutoff : float, optional
        Bond distance cutoff in Å.  Default 3.2.
    exclude_indices : set of int, optional
        Atom indices to exclude from neighbor lists (e.g. other constituent
        atoms of the same adsorption site).  Default ``None``.

    Returns
    -------
    shell1 : list of dict
        Immediate neighbors; each dict has keys ``index``, ``element``,
        ``distance``, ``shell``.
    shell2 : list of dict
        Neighbors of neighbors (excluding shell 1); same keys as above.
    """
    if exclude_indices is None:
        exclude_indices = set()

    node_key = f'a_{atom_idx}'

    # Shell 1 — direct atom-atom neighbors in graph
    shell1_idx = set()
    if node_key in G.nodes:
        for u, v, d in G.edges(node_key, data=True):
            if d['edge_type'] == 'atom-atom':
                other_key = v if u == node_key else u
                other_idx = int(other_key.split('_')[1])
                if other_idx not in exclude_indices:
                    shell1_idx.add(other_idx)

    shell1 = []
    for idx in shell1_idx:
        dp   = pos[atom_idx] - pos[idx]
        dp[0] -= cell[0] * round(dp[0] / cell[0])
        dp[1] -= cell[1] * round(dp[1] / cell[1])
        dist = float(np.linalg.norm(dp))
        shell1.append({
            'index'   : int(idx),
            'element' : str(syms[idx]),
            'distance': round(dist, 4),
            'shell'   : 1,
        })
    shell1.sort(key=lambda x: x['distance'])

    # Shell 2 — neighbors of shell1 (excluding shell1 and exclude_indices)
    shell2_idx = set()
    for s1 in shell1:
        s1_key = f'a_{s1["index"]}'
        if s1_key in G.nodes:
            for u, v, d in G.edges(s1_key, data=True):
                if d['edge_type'] == 'atom-atom':
                    other_key = v if u == s1_key else u
                    other_idx = int(other_key.split('_')[1])
                    if (other_idx not in exclude_indices
                            and other_idx not in shell1_idx
                            and other_idx != atom_idx):
                        shell2_idx.add(other_idx)

    shell2 = []
    for idx in shell2_idx:
        dp   = pos[atom_idx] - pos[idx]
        dp[0] -= cell[0] * round(dp[0] / cell[0])
        dp[1] -= cell[1] * round(dp[1] / cell[1])
        dist = float(np.linalg.norm(dp))
        shell2.append({
            'index'   : int(idx),
            'element' : str(syms[idx]),
            'distance': round(dist, 4),
            'shell'   : 2,
        })
    shell2.sort(key=lambda x: x['distance'])

    return shell1, shell2


def build_site_environment(G, slab, bond_cutoff=3.2):
    """
    Build the Level 1 / 2 / 3 chemical environment for every site node.

    Level 1 — site metadata (type, composition, position, constituent atoms).
    Level 2 — for each constituent atom: shell-1 and shell-2 surface
               neighbors, excluding the other site atoms.
    Level 3 — neighboring sites that share at least one constituent atom.

    Parameters
    ----------
    G : networkx.Graph
        Surface graph from :func:`build_surface_graph`.
    slab : ase.Atoms
        Full slab.
    bond_cutoff : float, optional
        Bond distance cutoff in Å for shell traversal.  Default 3.2.

    Returns
    -------
    environments : dict
        Keyed by site node ID (e.g. ``'s_0'``).  Each value is a dict with
        keys ``'level1'``, ``'level2'``, and ``'level3'``.

        ``level1`` : dict
            Site identity: ``site_id``, ``site_type``, ``composition``,
            ``full_label``, ``position``, ``constituent_atoms``, etc.
        ``level2`` : dict of dict
            Keyed by constituent atom index (str).  Each sub-dict has
            ``element``, ``shell1``, ``shell2``, ``n_shell1``, ``n_shell2``.
        ``level3`` : list of dict
            Neighboring sites; each dict has ``site_id``, ``full_label``,
            ``site_type``, ``composition``, ``position``, ``shared_atoms``.
    """
    pos  = slab.get_positions()
    syms = np.array(slab.get_chemical_symbols())
    cell = slab.cell.diagonal()

    site_nodes = [(n, d) for n, d in G.nodes(data=True)
                  if d['node_type'] == 'site']

    environments = {}

    for site_node, site_data in site_nodes:

        # ── Level 1 ───────────────────────────────────────────
        level1 = {
            'site_id'        : site_node,
            'site_type'      : site_data['site_type'],
            'hollow_type'    : site_data.get('hollow_type') or '',
            'composition'    : site_data['composition'],
            'full_label'     : site_data['label'],
            'position'       : site_data['position'],
            'atom_indices'   : list(site_data['atom_indices']),
            'constituent_atoms': [],
            'subsurf_element': site_data.get('subsurf_element') or '',
        }

        # Constituent atoms detail
        for ai in site_data['atom_indices']:
            level1['constituent_atoms'].append({
                'index'   : int(ai),
                'element' : str(syms[ai]),
                'position': pos[ai].tolist(),
            })

        # ── Level 2 ───────────────────────────────────────────
        # For each constituent atom, get shell1 and shell2
        # Exclude the other constituent atoms from neighbor lists
        site_atom_set = set(site_data['atom_indices'])
        level2 = {}

        for ai in site_data['atom_indices']:
            exclude = site_atom_set - {ai}
            shell1, shell2 = _get_atom_neighbors(
                ai, G, pos, syms, cell,
                bond_cutoff=bond_cutoff,
                exclude_indices=exclude,
            )
            level2[str(ai)] = {
                'element'    : str(syms[ai]),
                'shell1'     : shell1,
                'shell2'     : shell2,
                'n_shell1'   : len(shell1),
                'n_shell2'   : len(shell2),
            }

        # ── Level 3 ───────────────────────────────────────────
        # Neighboring sites — sites that share at least one atom
        level3 = []
        for u, v, edge_data in G.edges(site_node, data=True):
            if edge_data['edge_type'] == 'site-site':
                neighbor_node = v if u == site_node else u
                nd = G.nodes[neighbor_node]
                shared_atom_indices = edge_data.get('shared_atoms', [])
                shared_atoms_detail = [
                    {'index'  : int(idx),
                     'element': str(syms[idx])}
                    for idx in shared_atom_indices
                ]
                level3.append({
                    'site_id'    : neighbor_node,
                    'full_label' : nd['label'],
                    'site_type'  : nd['site_type'],
                    'composition': nd['composition'],
                    'position'   : nd['position'],
                    'shared_atoms': shared_atoms_detail,
                })

        # Sort level3 by site_id for consistency
        level3.sort(key=lambda x: int(x['site_id'].split('_')[1]))

        environments[site_node] = {
            'level1': level1,
            'level2': level2,
            'level3': level3,
        }

    print(f'Built environments for {len(environments)} sites')
    print(f'  Level 2: shell1 + shell2 per constituent atom')
    print(f'  Level 3: neighboring sites per site')

    # Sample printout for first hollow site
    hollow = next((n for n, d in site_nodes
                   if d['site_type'] == 'hollow'), None)
    if hollow:
        env = environments[hollow]
        print(f'\n  Example — {hollow} ({env["level1"]["full_label"]}):')
        print(f'    Level 1: {len(env["level1"]["constituent_atoms"])} '
              f'constituent atoms')
        for ca in env['level1']['constituent_atoms']:
            l2 = env['level2'][str(ca['index'])]
            print(f'      {ca["element"]:2s} idx={ca["index"]:4d}  '
                  f'shell1={l2["n_shell1"]} neighbors  '
                  f'shell2={l2["n_shell2"]} neighbors')
        print(f'    Level 3: {len(env["level3"])} neighboring sites')
        type_count = Counter(x['site_type'] for x in env['level3'])
        for st, cnt in sorted(type_count.items()):
            print(f'      {st:10s}: {cnt}')

    return environments


def save_surface_sites(G, environments, slab, save_path, seed):
    """
    Save the complete surface site list (Levels 1–3) to a JSON file.

    Parameters
    ----------
    G : networkx.Graph
        Surface graph from :func:`build_surface_graph`.
    environments : dict
        Site environments from :func:`build_site_environment`.
    slab : ase.Atoms
        Full slab.
    save_path : str
        Output JSON file path.
    seed : int
        Seed identifier stored in the file metadata.

    Returns
    -------
    output : dict
        The serialised JSON payload (also written to ``save_path``).
    """
    import json
    from collections import Counter

    syms = np.array(slab.get_chemical_symbols())
    pos  = slab.get_positions()
    cell = slab.cell.diagonal()

    site_nodes = [(n, d) for n, d in G.nodes(data=True)
                  if d['node_type'] == 'site']
    atom_nodes = [(n, d) for n, d in G.nodes(data=True)
                  if d['node_type'] == 'atom']

    # Surface atom summary
    surface_atoms = []
    for n, d in atom_nodes:
        surface_atoms.append({
            'node_id' : n,
            'index'   : int(n.split('_')[1]),
            'element' : d['element'],
            'position': d['position'],
        })

    # Site list with all levels
    site_list = []
    for site_node, env in environments.items():
        site_list.append({
            'site_id'    : site_node,
            'level1'     : env['level1'],
            'level2'     : env['level2'],
            'level3'     : env['level3'],
        })

    # Sort by site_id number
    site_list.sort(key=lambda x: int(x['site_id'].split('_')[1]))

    # Summary statistics
    type_counts = Counter(
        env['level1']['site_type'] for env in environments.values())
    comp_counts = Counter(
        env['level1']['composition'] for env in environments.values())

    output = {
        'metadata': {
            'seed'             : int(seed),
            'slab_path'        : '',
            'n_atoms_total'    : int(len(slab)),
            'n_atoms_surface'  : len(surface_atoms),
            'n_sites_total'    : len(site_list),
            'cell'             : cell.tolist(),
            'z_max'            : float(pos[:, 2].max()),
            'slab_composition' : dict(Counter(syms)),
            'site_type_counts' : dict(type_counts),
            'top_20_compositions': dict(comp_counts.most_common(20)),
        },
        'surface_atoms': surface_atoms,
        'sites'        : site_list,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Saved: {save_path}')
    print(f'  Total sites     : {len(site_list)}')
    print(f'  Surface atoms   : {len(surface_atoms)}')
    print(f'  Site types      : {dict(type_counts)}')
    print(f'  Top 5 comps     : '
          f'{dict(comp_counts.most_common(5))}')

    return output


# ══════════════════════════════════════════════════════════════
# STEP 2 — Neighbor traversal
# ══════════════════════════════════════════════════════════════

def get_site_neighbors(G, site_node, shell=1):
    """
    Return site nodes within ``shell`` hops via ``'site-site'`` edges.

    Parameters
    ----------
    G : networkx.Graph
        Surface graph from :func:`build_surface_graph`.
    site_node : str
        Node ID of the query site (e.g. ``'s_42'``).
    shell : int, optional
        Number of hops to traverse.  Default 1.

    Returns
    -------
    list of str
        Node IDs of neighboring sites within ``shell`` hops, excluding the
        query site itself.
    """
    site_graph = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d['edge_type'] == 'site-site':
            site_graph.add_edge(u, v)
    if site_node not in site_graph:
        return []
    neighbors, frontier = set(), {site_node}
    for _ in range(shell):
        nxt = set()
        for node in frontier:
            for nb in site_graph.neighbors(node):
                if nb not in neighbors and nb != site_node:
                    nxt.add(nb)
        neighbors |= nxt
        frontier   = nxt
    return list(neighbors)


def describe_local_environment(G, site_node, shell=2):
    """
    Print the local chemical environment around a site node.

    Parameters
    ----------
    G : networkx.Graph
        Surface graph from :func:`build_surface_graph`.
    site_node : str
        Node ID of the query site (e.g. ``'s_0'``).
    shell : int, optional
        Number of neighbor shells to report.  Default 2.
    """
    d = G.nodes[site_node]
    print(f'Site : {site_node}')
    print(f'  Label      : {d["label"]}')
    print(f'  Type       : {d["site_type"]}')
    print(f'  Composition: {d["composition"]}')
    print(f'  Position   : ({d["x"]:.3f}, {d["y"]:.3f})')
    print()

    atom_nbrs = [v for u, v, dd in G.edges(site_node, data=True)
                 if dd['edge_type'] == 'site-atom']
    print(f'  Constituent atoms ({len(atom_nbrs)}):')
    for an in atom_nbrs:
        ad = G.nodes[an]
        print(f'    {an}  {ad["element"]:2s}  '
              f'({ad["x"]:.3f}, {ad["y"]:.3f})')
    print()

    prev = set()
    for sh in range(1, shell + 1):
        all_sh  = set(get_site_neighbors(G, site_node, shell=sh))
        new_sh  = all_sh - prev
        prev    = all_sh
        counts  = Counter(G.nodes[n]['label'] for n in new_sh)
        print(f'  Shell {sh} ({len(new_sh)} sites):')
        for label, cnt in sorted(counts.items()):
            print(f'    {label:35s} x{cnt}')
        print()


# ══════════════════════════════════════════════════════════════
# STEP 3 — Visualization
# ══════════════════════════════════════════════════════════════

def visualize_surface_graph(G, slab, selected_site=None,
                            save_path='surface_graph.png',
                            seed=None):
    """
    Generate a three-panel surface graph figure and save to disk.

    Panel 1 — Top-layer atoms colored by element with labels.
    Panel 2 — Full graph overlay: atoms, sites, and site-site edges.
    Panel 3 — Local environment of ``selected_site`` (2 shells).

    Parameters
    ----------
    G : networkx.Graph
        Surface graph from :func:`build_surface_graph`.
    slab : ase.Atoms
        Full slab (used for cell dimensions).
    selected_site : str or None, optional
        Node ID of the site to highlight in Panel 3.  Auto-selects the
        first hollow site when ``None``.  Default ``None``.
    save_path : str, optional
        Output image file path.  Default ``'surface_graph.png'``.
    seed : int or None, optional
        Seed label shown in the figure title.  Falls back to
        ``G.graph['seed']`` when ``None``.  Default ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The three-panel figure object.
    """
    cell  = np.array(G.graph['cell'])
    z_max = G.graph['z_max']

    atom_nodes = [(n, d) for n, d in G.nodes(data=True)
                  if d['node_type'] == 'atom']
    site_nodes = [(n, d) for n, d in G.nodes(data=True)
                  if d['node_type'] == 'site']

    # Auto-select first hollow site if none given
    if selected_site is None:
        for n, d in site_nodes:
            if d['site_type'] == 'hollow':
                selected_site = n
                break

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#1A1A2E')
    _seed = seed if seed is not None else G.graph.get('seed', '?')
    fig.suptitle(f'Hastelloy N Surface Graph — Seed {_seed}',
                 fontsize=13, fontweight='bold', color='white')

    def draw_cell(ax):
        rect = plt.Polygon(
            [[0,0],[cell[0],0],[cell[0],cell[1]],[0,cell[1]]],
            fill=False, edgecolor='white',
            linewidth=1.5, linestyle='--')
        ax.add_patch(rect)

    def setup_ax(ax, title):
        ax.set_facecolor('#1A1A2E')
        ax.set_aspect('equal')
        ax.set_xlim(-0.8, cell[0] + 0.8)
        ax.set_ylim(-0.8, cell[1] + 0.8)
        ax.set_title(title, fontsize=11, color='white', pad=8)
        ax.set_xlabel('x (Å)', fontsize=9, color='white')
        ax.set_ylabel('y (Å)', fontsize=9, color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # ── Panel 1: Surface atoms ────────────────────────────────
    ax = axes[0]
    setup_ax(ax, 'Surface atoms — top layer (30 atoms)')
    draw_cell(ax)

    # Atom-atom bonds
    top_set = {n for n, d in atom_nodes}
    for u, v, d in G.edges(data=True):
        if d['edge_type'] != 'atom-atom':
            continue
        xu, yu = G.nodes[u]['x'], G.nodes[u]['y']
        xv, yv = G.nodes[v]['x'], G.nodes[v]['y']
        if abs(xu - xv) < cell[0]/2 and abs(yu - yv) < cell[1]/2:
            ax.plot([xu, xv], [yu, yv],
                    color='#555577', linewidth=1.2, zorder=1)

    # Atoms
    for n, d in atom_nodes:
        s = (d['size'] / 50) ** 2 * 350
        ax.scatter(d['x'], d['y'], s=s,
                   c=d['color'],
                   edgecolors='white' if d['element'] == 'H' else '#222244',
                   linewidths=0.8, zorder=3)
        tc = ELEMENT_TEXT_COLOR.get(d['element'], 'black')
        ax.text(d['x'], d['y'], d['element'],
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                color=tc, zorder=4)

    # Element legend
    seen = sorted(set(d['element'] for n, d in atom_nodes))
    patches = [mpatches.Patch(
                   color=ELEMENT_COLORS.get(e, '#AAAAAA'), label=e)
               for e in seen]
    ax.legend(handles=patches, fontsize=7, loc='upper right',
              framealpha=0.8, facecolor='#1A1A2E',
              labelcolor='white', edgecolor='#444466')

    # ── Panel 2: Graph overlay ────────────────────────────────
    ax = axes[1]
    setup_ax(ax, 'Graph overlay\n(atoms · sites · site-site edges)')
    draw_cell(ax)

    # Atom-atom bonds (thin, background)
    for u, v, d in G.edges(data=True):
        if d['edge_type'] != 'atom-atom':
            continue
        xu, yu = G.nodes[u]['x'], G.nodes[u]['y']
        xv, yv = G.nodes[v]['x'], G.nodes[v]['y']
        if abs(xu - xv) < cell[0]/2 and abs(yu - yv) < cell[1]/2:
            ax.plot([xu, xv], [yu, yv],
                    color='#333355', linewidth=0.8, zorder=1)

    # Site-site edges (only within cell to avoid visual clutter)
    for u, v, d in G.edges(data=True):
        if d['edge_type'] != 'site-site':
            continue
        xu, yu = G.nodes[u]['x'], G.nodes[u]['y']
        xv, yv = G.nodes[v]['x'], G.nodes[v]['y']
        if abs(xu - xv) < cell[0]/2 and abs(yu - yv) < cell[1]/2:
            ax.plot([xu, xv], [yu, yv],
                    color='#FFFF88', linewidth=0.4,
                    alpha=0.25, zorder=2)

    # Atoms (smaller in this panel)
    for n, d in atom_nodes:
        s = (d['size'] / 60) ** 2 * 200
        ax.scatter(d['x'], d['y'], s=s,
                   c=d['color'], edgecolors='none', zorder=3)
        ax.text(d['x'], d['y'], d['element'],
                ha='center', va='center',
                fontsize=5, color='black', zorder=4)

    # Site markers
    for n, d in site_nodes:
        hl  = (n == selected_site)
        s   = 140 if hl else 55
        ec  = 'white' if hl else 'none'
        lw  = 2.0 if hl else 0
        ax.scatter(d['x'], d['y'],
                   s=s, marker=d['marker'],
                   c=d['color'], edgecolors=ec,
                   linewidths=lw, zorder=5, alpha=0.9)

    # Site type legend
    leg = [Line2D([0],[0], marker=SITE_MARKERS[st], color='w',
                  markerfacecolor=SITE_COLORS[st],
                  markersize=7, label=st, linestyle='None')
           for st in ['ontop','bridge','hollow']]
    ax.legend(handles=leg, fontsize=7, loc='upper right',
              framealpha=0.8, facecolor='#1A1A2E',
              labelcolor='white', edgecolor='#444466')

    # ── Panel 3: Local environment ────────────────────────────
    # Shows selected site + 1st shell only (no site-site edges)
    # 2nd shell shown as faint markers only — no labels
    ax = axes[2]
    sd = G.nodes[selected_site]
    setup_ax(ax, f'Local environment\n{sd["label"]}')

    shell1 = set(get_site_neighbors(G, selected_site, shell=1))
    shell2 = set(get_site_neighbors(G, selected_site, shell=2)) - shell1

    # Show selected + 1st shell sites and their atoms
    # 2nd shell sites shown as faint markers only
    show_sites_main  = {selected_site} | shell1
    show_sites_faint = shell2
    all_show_sites   = show_sites_main | show_sites_faint

    # Atoms constituent to selected + 1st shell only
    show_atoms = set()
    for sn in show_sites_main:
        for u, v, d in G.edges(sn, data=True):
            if d['edge_type'] == 'site-atom':
                other = v if u == sn else u
                show_atoms.add(other)

    # Axis limits — based on selected + 1st shell atoms only
    pad = 2.5
    xs  = [G.nodes[n]['x'] for n in show_sites_main | show_atoms]
    ys  = [G.nodes[n]['y'] for n in show_sites_main | show_atoms]
    xlo = max(min(xs) - pad, -1.0)
    xhi = min(max(xs) + pad, cell[0] + 1.0)
    ylo = max(min(ys) - pad, -1.0)
    yhi = min(max(ys) + pad, cell[1] + 1.0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    draw_cell(ax)

    # Atom-atom bonds (only within show_atoms)
    for u, v, d in G.edges(data=True):
        if d['edge_type'] == 'atom-atom' \
                and u in show_atoms and v in show_atoms:
            xu, yu = G.nodes[u]['x'], G.nodes[u]['y']
            xv, yv = G.nodes[v]['x'], G.nodes[v]['y']
            if abs(xu-xv) < cell[0]/2 and abs(yu-yv) < cell[1]/2:
                ax.plot([xu, xv], [yu, yv],
                        color='#555577', linewidth=1.2, zorder=2)

    # Atom nodes
    for an in show_atoms:
        ad = G.nodes[an]
        s  = (ad['size'] / 45) ** 2 * 320
        ax.scatter(ad['x'], ad['y'], s=s,
                   c=ad['color'],
                   edgecolors='#AAAACC', linewidths=0.8, zorder=3)
        tc = ELEMENT_TEXT_COLOR.get(ad['element'], 'black')
        ax.text(ad['x'], ad['y'], ad['element'],
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=tc, zorder=4)

    # 2nd shell sites — faint, no labels, no edges
    for sn in show_sites_faint:
        sd_n = G.nodes[sn]
        # Only draw if within current axis range
        if xlo <= sd_n['x'] <= xhi and ylo <= sd_n['y'] <= yhi:
            ax.scatter(sd_n['x'], sd_n['y'],
                       s=35, marker=sd_n['marker'],
                       c=sd_n['color'], edgecolors='none',
                       linewidths=0, zorder=4, alpha=0.35)

    # 1st shell sites — yellow border, label below marker
    for sn in shell1:
        sd_n = G.nodes[sn]
        ax.scatter(sd_n['x'], sd_n['y'],
                   s=130, marker=sd_n['marker'],
                   c=sd_n['color'], edgecolors='#FFFF88',
                   linewidths=1.8, zorder=5, alpha=0.95)
        # Label — short form below marker
        short = sd_n['label'].replace('_hollow_fcc','_hol_fcc') \
                             .replace('_hollow_hcp','_hol_hcp') \
                             .replace('_bridge','_br') \
                             .replace('_atop','_at')
        ax.text(sd_n['x'], sd_n['y'] - 0.45,
                short, ha='center', va='top',
                fontsize=5.5, color='#FFFF88', zorder=6,
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='#1A1A2E',
                          edgecolor='none', alpha=0.75))

    # Selected site — white border, bold label above
    sd_n = G.nodes[selected_site]
    ax.scatter(sd_n['x'], sd_n['y'],
               s=300, marker=sd_n['marker'],
               c=sd_n['color'], edgecolors='white',
               linewidths=2.8, zorder=7, alpha=1.0)
    ax.text(sd_n['x'], sd_n['y'] + 0.55,
            sd_n['label'],
            ha='center', va='bottom',
            fontsize=8.5, color='white',
            fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='#1A1A2E',
                      edgecolor='white',
                      linewidth=0.8, alpha=0.9))

    # Shell legend
    leg3 = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=SITE_COLORS['hollow'],
               markeredgecolor='white', markersize=10,
               linestyle='None', label='Selected'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=SITE_COLORS['hollow'],
               markeredgecolor='#FFFF88', markersize=8,
               linestyle='None', label='1st shell'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=SITE_COLORS['hollow'],
               markeredgecolor='none', markersize=5,
               alpha=0.4,
               linestyle='None', label='2nd shell'),
    ]
    ax.legend(handles=leg3, fontsize=7, loc='upper right',
              framealpha=0.8, facecolor='#1A1A2E',
              labelcolor='white', edgecolor='#444466')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#1A1A2E')
    print(f'Saved: {save_path}')
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os

    print('=' * 60)
    print('STEP 1 — Building surface graph')
    print('=' * 60)
    SEED      = _DEFAULT_SEED
    SLAB_PATH = _DEFAULT_SLAB_PATH
    G, slab, top3, sites = build_surface_graph(SLAB_PATH, seed=SEED)

    print()
    print('=' * 60)
    print('STEP 2 — Local environment traversal')
    print('=' * 60)

    hollow_sites = [(n, d) for n, d in G.nodes(data=True)
                    if d.get('node_type') == 'site'
                    and d.get('site_type') == 'hollow']

    demo_site = hollow_sites[0][0] if hollow_sites else None

    if demo_site:
        describe_local_environment(G, demo_site, shell=2)

        print('Site type summary:')
        labels = [(d['site_type'], d['label'], n)
                  for n, d in G.nodes(data=True)
                  if d.get('node_type') == 'site']
        for st, cnt in sorted(Counter(l[0] for l in labels).items()):
            print(f'  {st:12s}: {cnt} sites')

    print()
    print('=' * 60)
    print('STEP 2b — Build Level 1 + 2 + 3 site environments')
    print('=' * 60)
    environments = build_site_environment(G, slab)

    print()
    print('=' * 60)
    print('STEP 2c — Save surface_sites.json')
    print('=' * 60)
    sites_json_path = (f'results/notebook04b-surface-relaxation/'
                       f'{SEED}/surface_sites.json')
    os.makedirs(os.path.dirname(sites_json_path), exist_ok=True)
    save_surface_sites(G, environments, slab,
                       save_path=sites_json_path, seed=SEED)

    print()
    print('=' * 60)
    print('STEP 3 — Visualization')
    print('=' * 60)

    save_path = (f'results/notebook06-Neb-dissociation/'
                 f'{SEED}/surface_graph.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    visualize_surface_graph(G, slab,
                            selected_site=demo_site,
                            save_path=save_path)

    print()
    print('Done.')
    print(f'  Atom nodes      : {sum(1 for n,d in G.nodes(data=True) if d["node_type"]=="atom")}')
    print(f'  Site nodes      : {sum(1 for n,d in G.nodes(data=True) if d["node_type"]=="site")}')
    print(f'  Total edges     : {G.number_of_edges()}')
    print(f'  surface_sites   : {sites_json_path}')
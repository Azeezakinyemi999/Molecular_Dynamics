"""
subsurface_graph.py
====================

# ─────────────────────────────────────────────────────────────────────────────
# MODULE ROLE
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 in the Project 2 pipeline.
#
# Builds the subsurface representation: locates octahedral and tetrahedral
# interstitial sites in subsurface layers (10, 11) via scipy Voronoi
# tessellation, randomly samples bulk-reference sites from interior layers
# (5-9), classifies each site by coordination geometry and local composition,
# connects layer-11 sites to the surface sites above them, and saves the
# result to subsurface_sites.json.
#
# INPUT:   clean slab LAMMPS data file
#          surface_sites.json  (produced by surface_graph.py)
# OUTPUT:  results/<seed>/subsurface_sites.json
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW IT LINKS TO THE OTHER MODULES
# ─────────────────────────────────────────────────────────────────────────────
# ← surface_graph.py     must run first; its surface_sites.json is loaded
#                        here to build surface-subsurface edges.
#
# → site_identifier.py   is independent of this module; it runs after
#                        relaxation and does not use subsurface_sites.json.
# ─────────────────────────────────────────────────────────────────────────────

Module for identifying and labeling subsurface octahedral and tetrahedral
interstitial sites in a Hastelloy N slab.

Used by Notebook 08 (subsurface H absorption energy) to:
  1. Find oct/tet interstitial sites in subsurface layers (10, 11)
  2. Sample bulk-like oct/tet sites in layers 5-9 for E_abs(bulk) reference
  3. Classify each site (oct vs tet, composition, distortion)
  4. Connect subsurface sites (layer 11) to surface sites above
  5. Output a subsurface_sites.json and an augmented NetworkX graph

Design: Path C (standalone). Does not import from surface_graph.py.
Layer identification is duplicated as a private helper.

Dependencies: scipy, networkx, numpy, ase
"""

import json
import os
import warnings
from collections import Counter

import numpy as np
import networkx as nx
from ase.io import read as ase_read

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Coordination cutoff for classifying a Voronoi vertex as oct/tet
# 2.2 A captures both oct first NN (~1.76 A for Ni) and tet first NN (~1.52 A)
COORD_CUTOFF_DEFAULT = 2.2

# Pymatgen Voronoi parameters
CLUSTERING_TOL_DEFAULT = 0.75   # merge sites within this distance
MIN_DIST_DEFAULT       = 0.5    # discard sites within this distance of an atom

# Surface-subsurface connection tolerance (xy projection)
XY_TOL_DEFAULT = 1.5            # max xy distance for "directly above/below"


# ─────────────────────────────────────────────────────────────────────────────
# LAYER IDENTIFICATION (duplicated from surface_graph.py, Path C)
# ─────────────────────────────────────────────────────────────────────────────

def _identify_layers(positions, n_layers=12, z_tol=None):
    """
    Bin atoms into z-layers using rank-based assignment.

    Sorts atoms by z-coordinate, then assigns the bottom 1/n_layers fraction
    to layer 1, next fraction to layer 2, etc. This is exact when each layer
    has the same number of atoms (typical for constructed slabs like
    FCC(111) where every layer has N atoms by design).

    For 360 atoms in 12 layers, gives exactly 30 atoms per layer.

    Buckled atoms (atoms whose z position is in another layer's range due
    to relaxation) are still assigned to the layer that matches their z-rank,
    which is what we want for spatial layer identification.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian atom positions.
    n_layers : int
        Expected number of layers in the slab.
    z_tol : float or None, optional
        Kept for backward compatibility; ignored.  Default ``None``.

    Returns
    -------
    layer_map : dict
        Maps atom index (int) → layer number (int, 1-indexed;
        1 = bottom, ``n_layers`` = top).
    layer_z : dict
        Maps layer number (int) → mean z-coordinate (float) in Å.
    """
    positions = np.asarray(positions)
    z = positions[:, 2]
    n_atoms = len(z)

    if n_atoms < n_layers:
        raise ValueError(
            f"_identify_layers: only {n_atoms} atoms but {n_layers} layers requested"
        )

    # Sort atoms by z (ascending: bottom layer first)
    sorted_idx = np.argsort(z)

    # Compute layer sizes (uniform when N is divisible, extras go to bottom)
    n_per_layer = n_atoms // n_layers
    remainder   = n_atoms - n_per_layer * n_layers

    if remainder != 0:
        warnings.warn(
            f"_identify_layers: {n_atoms} atoms not divisible by {n_layers} "
            f"layers. Will distribute {remainder} extra atom(s) to bottom layers."
        )

    # Assign each atom to a layer by rank
    layer_map = {}
    layer_z   = {}
    start = 0
    for L in range(n_layers):
        # Bottom `remainder` layers get one extra atom
        extra = 1 if L < remainder else 0
        end = start + n_per_layer + extra
        atoms_in_layer = sorted_idx[start:end]
        layer_z[L + 1] = float(np.mean(z[atoms_in_layer]))
        for a in atoms_in_layer:
            layer_map[int(a)] = L + 1
        start = end

    return layer_map, layer_z


# ─────────────────────────────────────────────────────────────────────────────
# VORONOI SITE FINDING (scipy-based, with PBC 3x3 replication)
# ─────────────────────────────────────────────────────────────────────────────
#
# We use scipy.spatial.Voronoi directly rather than pymatgen because pymatgen's
# VoronoiInterstitialGenerator runs symmetry analysis that is prohibitively
# slow on disordered alloy slabs (360 atoms, 7 elements in Hastelloy N).

from scipy.spatial import Voronoi


def _slab_to_atoms(slab_path):
    """Read a LAMMPS data file into ASE Atoms."""
    atoms = ase_read(slab_path, format="lammps-data", atom_style="atomic")
    return atoms


def find_voronoi_sites(slab_path,
                      clustering_tol=CLUSTERING_TOL_DEFAULT,
                      min_dist=MIN_DIST_DEFAULT):
    """
    Find interstitial sites using ``scipy.spatial.Voronoi`` with PBC handling.

    The slab is replicated 3 × 3 in xy (not z) before tessellation.
    Voronoi vertices are filtered to the central cell, deduplicated, and
    pruned of any vertex too close to an atom.

    Parameters
    ----------
    slab_path : str
        Path to the LAMMPS data file.
    clustering_tol : float, optional
        Merge Voronoi vertices within this distance in Å.
        Default ``CLUSTERING_TOL_DEFAULT`` (0.75).
    min_dist : float, optional
        Discard vertices within this distance of any atom in Å.
        Default ``MIN_DIST_DEFAULT`` (0.5).

    Returns
    -------
    sites_cart : ndarray, shape (M, 3)
        Cartesian coordinates of unique interstitial sites in the central cell.
    structure : None
        Returned for API compatibility (not used in the scipy path).
    atoms : ase.Atoms
        The slab as ASE Atoms.
    """
    atoms = _slab_to_atoms(slab_path)
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())

    # Filter out H if any (slab should be clean)
    is_metal = symbols != "H"
    metal_positions = positions[is_metal]

    cell = atoms.cell.diagonal()  # assumes orthorhombic
    Lx, Ly = cell[0], cell[1]

    # Step 1: Replicate atoms 3x3 in xy (NOT in z)
    replicated = []
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            shifted = metal_positions.copy()
            shifted[:, 0] += ix * Lx
            shifted[:, 1] += iy * Ly
            replicated.append(shifted)
    replicated_positions = np.vstack(replicated)

    # Step 2: Run scipy Voronoi
    vor = Voronoi(replicated_positions)
    vertices = vor.vertices

    # Step 3: Filter to vertices inside the central cell (xy)
    # For z: keep vertices that are between the bottom and top of the slab
    z_min = metal_positions[:, 2].min() - 1.0
    z_max = metal_positions[:, 2].max() + 1.0

    in_cell = (
        (vertices[:, 0] >= 0.0) & (vertices[:, 0] < Lx) &
        (vertices[:, 1] >= 0.0) & (vertices[:, 1] < Ly) &
        (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
    )
    vertices = vertices[in_cell]

    if len(vertices) == 0:
        return np.empty((0, 3)), None, atoms

    # Step 4: Filter out vertices too close to ANY atom (min_dist)
    keep_mask = np.ones(len(vertices), dtype=bool)
    for i, v in enumerate(vertices):
        diffs = replicated_positions - v
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        if dists.min() < min_dist:
            keep_mask[i] = False
    vertices = vertices[keep_mask]

    if len(vertices) == 0:
        return np.empty((0, 3)), None, atoms

    # Step 5: Cluster nearby vertices (clustering_tol)
    kept = []
    for v in vertices:
        merge = False
        for k in kept:
            if np.linalg.norm(v - k) < clustering_tol:
                merge = True
                break
        if not merge:
            kept.append(v)

    sites_cart = np.array(kept)

    return sites_cart, None, atoms


# ─────────────────────────────────────────────────────────────────────────────
# OCT/TET CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _find_coordinating_atoms(site_pos, atom_positions, atom_elements,
                              cell, cutoff=COORD_CUTOFF_DEFAULT):
    """
    Find all metal atoms within ``cutoff`` of ``site_pos`` using xy PBC.

    Parameters
    ----------
    site_pos : ndarray, shape (3,)
        Cartesian position of the interstitial site.
    atom_positions : ndarray, shape (N, 3)
        Cartesian positions of all metal atoms in the slab.
    atom_elements : ndarray of str, shape (N,)
        Element symbols.
    cell : ndarray, shape (3,)
        Orthorhombic cell dimensions ``[Lx, Ly, Lz]`` in Å.
    cutoff : float, optional
        Distance cutoff in Å.  Default ``COORD_CUTOFF_DEFAULT`` (2.2).

    Returns
    -------
    coord_list : list of dict
        Sorted by distance ascending.  Each dict has keys
        ``atom_index`` (int), ``element`` (str), ``distance`` (float).
    """
    coord_list = []
    for i, (pos, el) in enumerate(zip(atom_positions, atom_elements)):
        # Periodic image in xy only (slab is not periodic in z)
        dx = site_pos[0] - pos[0]
        dy = site_pos[1] - pos[1]
        dz = site_pos[2] - pos[2]

        # Apply PBC in xy
        dx -= cell[0] * round(dx / cell[0])
        dy -= cell[1] * round(dy / cell[1])

        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < cutoff:
            coord_list.append({
                "atom_index": int(i),
                "element": str(el),
                "distance": float(dist),
            })

    # Sort by distance
    coord_list.sort(key=lambda x: x["distance"])
    return coord_list


def _composition_label(coord_list, site_type):
    """
    Generate a composition label such as ``'Ni3MoCrFe_oct'``.

    Elements are sorted by count descending, then alphabetically for ties.

    Parameters
    ----------
    coord_list : list of dict
        Output from :func:`_find_coordinating_atoms`.
    site_type : str
        ``'oct'`` or ``'tet'``.

    Returns
    -------
    label : str
        Composition label, e.g. ``'Ni3MoCrFe_oct'`` or ``'Ni2Mo2_tet'``.
    """
    elements = [c["element"] for c in coord_list]
    counts = Counter(elements)

    # Sort: count desc, then element name asc
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    parts = []
    for elem, count in sorted_items:
        if count == 1:
            parts.append(elem)
        else:
            parts.append(f"{elem}{count}")

    return "".join(parts) + "_" + site_type


def classify_site(site_pos, atom_positions, atom_elements, cell,
                  cutoff=COORD_CUTOFF_DEFAULT):
    """
    Classify a Voronoi vertex as octahedral, tetrahedral, or unknown.

    Coordination-number rules:

    - 6 NN → ``'oct'`` (perfect)
    - 5 or 7 NN → ``'oct'`` (distorted)
    - 4 NN → ``'tet'`` (perfect)
    - 3 NN → ``'tet'`` (distorted)
    - other → ``'unknown'``

    Parameters
    ----------
    site_pos : ndarray, shape (3,)
        Cartesian position of the site.
    atom_positions : ndarray, shape (N, 3)
        Metal atom positions.
    atom_elements : ndarray of str, shape (N,)
        Element symbols.
    cell : ndarray, shape (3,)
        Orthorhombic cell dimensions in Å.
    cutoff : float, optional
        Coordination cutoff in Å.  Default ``COORD_CUTOFF_DEFAULT`` (2.2).

    Returns
    -------
    dict
        Keys:

        ``site_type`` : str
            ``'oct'``, ``'tet'``, or ``'unknown'``.
        ``coord_count`` : int
            Number of coordinating atoms.
        ``coord_list`` : list of dict
            Coordinating atom details (see :func:`_find_coordinating_atoms`).
        ``composition_label`` : str
            E.g. ``'Ni3MoCrFe_oct'``.
        ``distortion_score`` : float
            ``(max_d − min_d) / mean_d`` among the coordinating atoms.
        ``is_distorted`` : bool
            ``True`` when coordination number is not exactly 4 or 6.
    """
    coord_list = _find_coordinating_atoms(
        site_pos, atom_positions, atom_elements, cell, cutoff=cutoff
    )

    n = len(coord_list)

    # Classify
    if n == 6:
        site_type = "oct"
        is_distorted = False
    elif n in (5, 7):
        site_type = "oct"
        is_distorted = True
    elif n == 4:
        site_type = "tet"
        is_distorted = False
    elif n == 3:
        site_type = "tet"
        is_distorted = True
    else:
        site_type = "unknown"
        is_distorted = True

    # Distortion score: spread in nearest-neighbor distances
    if n >= 2:
        dists = [c["distance"] for c in coord_list]
        mean_d = float(np.mean(dists))
        spread = float(max(dists) - min(dists))
        distortion_score = spread / mean_d if mean_d > 0 else 0.0
    else:
        distortion_score = 0.0

    # Composition label (only meaningful for oct/tet)
    if site_type in ("oct", "tet"):
        composition_label = _composition_label(coord_list, site_type)
    else:
        composition_label = f"unknown_n{n}"

    return {
        "site_type": site_type,
        "coord_count": n,
        "coord_list": coord_list,
        "composition_label": composition_label,
        "distortion_score": float(distortion_score),
        "is_distorted": is_distorted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE - SUBSURFACE CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def _periodic_xy_distance(pos1, pos2, cell):
    """xy distance with periodic boundary conditions."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dx -= cell[0] * round(dx / cell[0])
    dy -= cell[1] * round(dy / cell[1])
    return np.sqrt(dx * dx + dy * dy)


def _get_surface_site_position(surf_site):
    """
    Extract the [x, y, z] position from a surface site dict.

    Handles the nested level1.position structure produced by NB04b's
    surface_graph.py, and falls back to flat 'position' or 'xy' keys
    for compatibility with older formats.
    """
    # Primary: nested level1.position (NB04b format)
    level1 = surf_site.get("level1")
    if isinstance(level1, dict) and "position" in level1:
        return level1["position"]
    # Fallbacks
    if "position" in surf_site:
        return surf_site["position"]
    if "xy" in surf_site:
        return surf_site["xy"]
    return None


def connect_to_surface(subsurface_sites, surface_sites_data, cell,
                        xy_tol=XY_TOL_DEFAULT):
    """
    Connect each layer-11 subsurface site to the surface site directly above.

    Parameters
    ----------
    subsurface_sites : list of dict
        Classified subsurface sites; each dict must have ``'site_id'``,
        ``'layer_classification'``, and ``'position'``.
    surface_sites_data : dict
        Parsed ``surface_sites.json``; must have a ``'sites'`` key, with
        positions at ``site['level1']['position']``.
    cell : ndarray, shape (3,)
        Orthorhombic cell dimensions in Å.
    xy_tol : float, optional
        Max xy distance (Å) for a site to be considered directly above/below.
        Default ``XY_TOL_DEFAULT`` (1.5).

    Returns
    -------
    connections : list of tuple
        Each entry is ``(subsurf_site_id, surface_site_id, xy_dist)`` where
        ``xy_dist`` is the periodic xy distance in Å.
    """
    surface_sites_list = surface_sites_data["sites"]

    connections = []
    for sub_site in subsurface_sites:
        # Only connect layer 11 (immediate subsurface) to surface
        if sub_site["layer_classification"] != "subsurface_1":
            continue

        sub_pos = sub_site["position"]
        for surf in surface_sites_list:
            surf_pos = _get_surface_site_position(surf)
            if surf_pos is None:
                continue
            surf_pos = np.array(surf_pos)
            surf_xy = surf_pos[:2] if len(surf_pos) >= 2 else surf_pos

            xy_dist = _periodic_xy_distance(
                np.array(sub_pos), np.append(surf_xy, 0.0), cell
            )

            if xy_dist < xy_tol:
                connections.append((
                    sub_site["site_id"],
                    surf["site_id"],
                    float(xy_dist),
                ))

    return connections


# ─────────────────────────────────────────────────────────────────────────────
# BULK SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_bulk_sites(all_sites, bulk_layer_range, n_samples, seed):
    """
    Randomly sample ``n_samples`` sites from bulk-like layers.

    Parameters
    ----------
    all_sites : list of dict
        All classified sites (typically from layers 5–11).
    bulk_layer_range : iterable of int
        Layer numbers to sample from, e.g. ``[5, 6, 7, 8, 9]``.
    n_samples : int
        Number of sites to sample.  Capped at available candidates.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sampled : list of dict
        Shallow copies of the sampled sites with
        ``layer_classification`` set to ``'bulk_sample'``.
    """
    bulk_layers_set = set(bulk_layer_range)
    bulk_candidates = [
        s for s in all_sites
        if s["layer_number"] in bulk_layers_set
    ]

    if len(bulk_candidates) == 0:
        return []

    rng = np.random.default_rng(seed)
    n_sample_actual = min(n_samples, len(bulk_candidates))
    indices = rng.choice(
        len(bulk_candidates), size=n_sample_actual, replace=False
    )

    sampled = []
    for i in indices:
        site = dict(bulk_candidates[i])  # shallow copy
        site["layer_classification"] = "bulk_sample"
        sampled.append(site)

    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def build_subsurface_graph(
    slab_path,
    surface_sites_json_path,
    seed,
    n_layers_total=12,
    subsurface_layers=(10, 11),
    bulk_sample_layers=(5, 6, 7, 8, 9),
    n_bulk_samples=15,
    clustering_tol=CLUSTERING_TOL_DEFAULT,
    min_dist=MIN_DIST_DEFAULT,
    coord_cutoff=COORD_CUTOFF_DEFAULT,
    xy_tol=XY_TOL_DEFAULT,
    z_tol=0.5,
):
    """
    Top-level orchestrator. Returns the augmented graph and site list.

    Steps:
      1. Read slab, identify 12 layers by z-binning.
      2. Run scipy.spatial.Voronoi to find all interstitial sites.
      3. Filter to sites in target layers (5-11).
      4. Classify each as oct/tet with composition label.
      5. Subsurface (10, 11): keep all.
         Bulk (5-9): randomly sample n_bulk_samples.
      6. Load surface_sites.json, connect layer-11 sites to surface sites.
      7. Build NetworkX graph and return it with the sites list.

    Parameters
    ----------
    slab_path : str
        Path to ``clean_slab_reminimized.lammps``.
    surface_sites_json_path : str
        Path to ``surface_sites.json`` produced by NB04b.
    seed : int
        Random seed for bulk-site sampling reproducibility.
    n_layers_total : int, optional
        Expected total number of layers in the slab.  Default 12.
    subsurface_layers : tuple of int, optional
        Layers to enumerate fully.  Default ``(10, 11)``.
    bulk_sample_layers : tuple of int, optional
        Layers to randomly sample from.  Default ``(5, 6, 7, 8, 9)``.
    n_bulk_samples : int, optional
        Number of bulk reference sites to sample.  Default 15.
    clustering_tol : float, optional
        Voronoi vertex clustering tolerance in Å.
        Default ``CLUSTERING_TOL_DEFAULT`` (0.75).
    min_dist : float, optional
        Minimum allowed distance from a Voronoi vertex to any atom in Å.
        Default ``MIN_DIST_DEFAULT`` (0.5).
    coord_cutoff : float, optional
        Coordination cutoff for oct/tet classification in Å.
        Default ``COORD_CUTOFF_DEFAULT`` (2.2).
    xy_tol : float, optional
        Tolerance for surface–subsurface xy connection in Å.
        Default ``XY_TOL_DEFAULT`` (1.5).
    z_tol : float, optional
        Tolerance for layer identification (kept for compatibility).
        Default 0.5.

    Returns
    -------
    G : networkx.Graph
        Augmented graph containing surface site nodes (from JSON), subsurface
        site nodes, and ``'surface-subsurface'`` edges.
    subsurface_sites : list of dict
        All classified subsurface and bulk-sample sites with full metadata.
    """
    print(f"=" * 70)
    print(f"  Building subsurface graph for seed {seed}")
    print(f"=" * 70)

    # Step 1: Find all Voronoi sites
    print(f"\nStep 1: Voronoi tessellation via scipy...")
    sites_cart, structure, atoms = find_voronoi_sites(
        slab_path, clustering_tol=clustering_tol, min_dist=min_dist
    )
    print(f"  Found {len(sites_cart)} unique interstitial sites in the cell")

    # Step 2: Identify layers
    print(f"\nStep 2: Identifying {n_layers_total} z-layers...")
    metal_positions = atoms.get_positions()
    metal_elements  = np.array(atoms.get_chemical_symbols())

    # Filter out H atoms if present (slab should be clean but just in case)
    is_metal = metal_elements != "H"
    metal_positions = metal_positions[is_metal]
    metal_elements  = metal_elements[is_metal]

    layer_map, layer_z = _identify_layers(
        metal_positions, n_layers=n_layers_total
    )
    print(f"  Layer z-centers: " +
          ", ".join(f"L{k}={v:.2f}" for k, v in sorted(layer_z.items())))

    cell = atoms.cell.diagonal()

    # Step 3: Assign each Voronoi site to a layer (by nearest atomic plane)
    print(f"\nStep 3: Assigning sites to layers...")
    target_layers = set(subsurface_layers) | set(bulk_sample_layers)

    all_classified_sites = []
    for site_idx, site_pos in enumerate(sites_cart):
        # Layer assignment: nearest atomic plane by z
        z = site_pos[2]
        nearest_layer = min(layer_z.keys(), key=lambda L: abs(layer_z[L] - z))

        if nearest_layer not in target_layers:
            continue

        # Step 4: Classify the site
        clf = classify_site(
            site_pos, metal_positions, metal_elements, cell,
            cutoff=coord_cutoff
        )

        if clf["site_type"] == "unknown":
            continue  # skip sites we cannot classify cleanly

        # Tag with layer + initial site_id
        site_dict = {
            "site_id": f"ss_{site_idx}",
            "position": [float(x) for x in site_pos],
            "layer_number": nearest_layer,
            "layer_classification": "tbd",  # set below
            **clf,
        }
        # Convert coord_list dicts to JSON-safe
        site_dict["coord_list"] = [
            {"atom_index": c["atom_index"],
             "element": c["element"],
             "distance": c["distance"]}
            for c in clf["coord_list"]
        ]
        all_classified_sites.append(site_dict)

    print(f"  Classified {len(all_classified_sites)} sites in target layers")

    # Step 4: Assign final layer_classification
    subsurface_sites_final = []
    for s in all_classified_sites:
        L = s["layer_number"]
        if L == max(subsurface_layers):     # typically layer 11
            s["layer_classification"] = "subsurface_1"
            subsurface_sites_final.append(s)
        elif L in subsurface_layers:        # typically layer 10
            s["layer_classification"] = "subsurface_2"
            subsurface_sites_final.append(s)
        # bulk layers handled below by sampling

    # Step 5: Bulk sampling
    print(f"\nStep 5: Sampling {n_bulk_samples} bulk-like sites from layers "
          f"{list(bulk_sample_layers)}...")
    bulk_samples = sample_bulk_sites(
        all_classified_sites, bulk_sample_layers, n_bulk_samples, seed
    )
    subsurface_sites_final.extend(bulk_samples)
    print(f"  Total sites for NB08: {len(subsurface_sites_final)} "
          f"(subsurface + bulk samples)")

    # Reassign site_ids cleanly: ss_0, ss_1, ...
    for new_idx, s in enumerate(subsurface_sites_final):
        s["site_id"] = f"ss_{new_idx}"

    # Step 6: Build graph
    print(f"\nStep 6: Building NetworkX graph...")
    G = nx.Graph()

    # Add subsurface site nodes
    for s in subsurface_sites_final:
        G.add_node(
            s["site_id"],
            node_type="subsurface_site",
            site_type=s["site_type"],
            composition_label=s["composition_label"],
            layer_classification=s["layer_classification"],
            layer_number=s["layer_number"],
            position=s["position"],
            distortion_score=s["distortion_score"],
        )

    # Step 7: Connect to surface
    print(f"\nStep 7: Connecting to surface_sites.json...")
    with open(surface_sites_json_path) as f:
        surface_data = json.load(f)

    # Add surface site nodes (we only need their IDs and positions for edges)
    for surf in surface_data["sites"]:
        if surf["site_id"] not in G.nodes:
            surf_pos = _get_surface_site_position(surf) or [0.0, 0.0, 0.0]
            G.add_node(
                surf["site_id"],
                node_type="surface_site",
                position=surf_pos,
                site_type=surf.get("level1", {}).get("site_type", "unknown"),
                composition_label=surf.get("level1", {}).get("full_label", "unknown"),
            )

    # Add surface-subsurface edges
    connections = connect_to_surface(
        subsurface_sites_final, surface_data, cell, xy_tol=xy_tol
    )
    for sub_id, surf_id, dist in connections:
        G.add_edge(sub_id, surf_id,
                   edge_type="surface-subsurface", distance=dist)

    print(f"  Surface-subsurface connections: {len(connections)}")
    print(f"  Total graph nodes: {G.number_of_nodes()}")
    print(f"  Total graph edges: {G.number_of_edges()}")

    print(f"\n{'=' * 70}")
    print(f"  subsurface_graph build complete")
    print(f"{'=' * 70}\n")

    return G, subsurface_sites_final


# ─────────────────────────────────────────────────────────────────────────────
# JSON SAVE/LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_subsurface_sites(subsurface_sites, output_path, seed,
                           metadata=None):
    """
    Save subsurface sites to a JSON file mirroring the ``surface_sites.json`` schema.

    Parameters
    ----------
    subsurface_sites : list of dict
        Output from :func:`build_subsurface_graph`.
    output_path : str
        Destination path for ``subsurface_sites.json``.  Parent directories
        are created if they do not exist.
    seed : int
        Seed used for the run; stored in the file metadata.
    metadata : dict, optional
        Additional key-value pairs to include in the metadata section.
        Default ``None``.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "seed": seed,
        "n_sites": len(subsurface_sites),
        "metadata": metadata or {},
        "sites": subsurface_sites,
    }

    # Tally for the metadata summary
    by_layer = {}
    by_type  = {"oct": 0, "tet": 0}
    by_composition = {}
    for s in subsurface_sites:
        L = s["layer_classification"]
        by_layer[L] = by_layer.get(L, 0) + 1
        if s["site_type"] in by_type:
            by_type[s["site_type"]] += 1
        c = s["composition_label"]
        by_composition[c] = by_composition.get(c, 0) + 1

    payload["summary"] = {
        "by_layer_classification": by_layer,
        "by_site_type": by_type,
        "by_composition_top10": dict(
            sorted(by_composition.items(),
                   key=lambda x: -x[1])[:10]
        ),
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved subsurface_sites.json -> {output_path}")
    print(f"  Total sites      : {payload['n_sites']}")
    print(f"  By layer         : {by_layer}")
    print(f"  By site type     : {by_type}")
    print(f"  Top compositions :")
    for k, v in payload["summary"]["by_composition_top10"].items():
        print(f"    {k}: {v}")
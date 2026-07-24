"""
subsurface_graph.py
====================

# ─────────────────────────────────────────────────────────────────────────────
# MODULE ROLE
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 in the Project 2 pipeline.
#
# Builds the subsurface representation: auto-detects the slab's total layer
# count, derives subsurface_1 (Hop A target) and subsurface_2 (Hop B /
# bulk-entry target) as the two layers immediately above the frozen bottom
# fraction, locates octahedral and tetrahedral interstitial sites in those
# layers via scipy Voronoi tessellation, classifies each site by coordination
# geometry and local composition, connects subsurface_1 sites to the surface
# sites above them, and saves the result to subsurface_sites.json.
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
  1. Auto-derive subsurface_1/subsurface_2 layer numbers from the slab's
     real detected layer count (bottom round(N/3) frozen, matching
     models.structure.compute_z_freeze_cutoff)
  2. Find oct/tet interstitial sites in those two layers
  3. Classify each site (oct vs tet, composition, distortion)
  4. Connect subsurface_1 sites to surface sites above
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


def _n_layers_from_metadata(surface_sites_json_path):
    """Derive the slab's construction layer count from surface_sites.json metadata.

    ``n_layers = n_atoms_total // n_atoms_surface`` — the total atom count divided
    by the atoms in one (111) plane (the surface layer). This is the number of
    z-layers the slab was *built* with, recovered from the slab's own recorded
    atom counts rather than from the relaxed geometry (which rumpling blurs).

    Valid only for metals/alloys, where every plane has the same atom count; the
    caller must not use it for oxides.

    Returns
    -------
    int or None
        The layer count, or ``None`` if the metadata is missing/unreadable or
        ``n_atoms_total`` is not an exact multiple of ``n_atoms_surface`` (in
        which case the surface count is not one clean plane — fall back).
    """
    try:
        with open(surface_sites_json_path) as fh:
            meta = json.load(fh).get('metadata', {})
        n_total   = int(meta.get('n_atoms_total', 0))
        n_surface = int(meta.get('n_atoms_surface', 0))
    except (OSError, ValueError, TypeError):
        return None
    if n_surface > 0 and n_total > 0 and n_total % n_surface == 0:
        return n_total // n_surface
    return None


def _identify_layers_by_gaps(positions, gap_tol=0.5):
    """
    Bin atoms into z-layers by gap detection (no equal-count assumption).

    Sorts atoms by z and starts a new layer wherever the gap between
    consecutive z values exceeds ``gap_tol``.  Required for oxide slabs,
    whose atomic planes have unequal atom counts (e.g. O3 vs Cr planes in
    corundum) and whose plane count differs from the repeat-unit count.

    Same contract as :func:`_identify_layers`:

    Returns
    -------
    layer_map : dict
        Maps atom index (int) → layer number (int, 1-indexed;
        1 = bottom, N = top).
    layer_z : dict
        Maps layer number (int) → mean z-coordinate (float) in Å.
    """
    positions = np.asarray(positions)
    z = positions[:, 2]
    order = np.argsort(z)

    layer_map = {}
    layer_members = [[order[0]]]
    for idx in order[1:]:
        if z[idx] - z[layer_members[-1][-1]] > gap_tol:
            layer_members.append([idx])
        else:
            layer_members[-1].append(idx)

    layer_z = {}
    for L, members in enumerate(layer_members, start=1):
        layer_z[L] = float(np.mean(z[members]))
        for a in members:
            layer_map[int(a)] = L

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
    atoms.wrap()
    return atoms


def find_voronoi_sites(slab_path,
                      clustering_tol=CLUSTERING_TOL_DEFAULT,
                      min_dist=MIN_DIST_DEFAULT,
                      layer_mode='rank',
                      n_layers=12):
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
    layer_mode : str, optional
        ``'rank'`` (default) — equal-count rank binning into ``n_layers``
        layers (metal slabs).  ``'gaps'`` — gap-based plane detection for
        slabs with unequal plane populations (oxides).
    n_layers : int, optional
        Total layer count for the ``'rank'`` z-smoothing pass.  Must match
        the slab's real detected plane count — a mismatch here would smooth
        atoms into the wrong layer means. Ignored when ``layer_mode='gaps'``.
        Default 12.

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

    # Z-layer smoothing: project each atom's z to its layer mean before Voronoi.
    # Thermal displacements shift Voronoi vertices enough to create spurious
    # sites or misclassify oct/tet. Smoothing z removes that noise while
    # preserving x/y chemical diversity. Raw positions are kept for the
    # atom-proximity filter (Step 4) so real atom locations are respected.
    if layer_mode == 'gaps':
        _layer_map, _layer_z = _identify_layers_by_gaps(metal_positions)
    else:
        _layer_map, _layer_z = _identify_layers(metal_positions, n_layers=n_layers)
    smoothed_positions = metal_positions.copy()
    for _i in range(len(smoothed_positions)):
        smoothed_positions[_i, 2] = _layer_z[_layer_map[_i]]

    # Step 1: Replicate z-smoothed atoms 3x3 in xy (NOT in z)
    replicated_smoothed = []
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            shifted = smoothed_positions.copy()
            shifted[:, 0] += ix * Lx
            shifted[:, 1] += iy * Ly
            replicated_smoothed.append(shifted)
    replicated_smoothed = np.vstack(replicated_smoothed)

    # Also replicate raw positions for proximity filtering
    replicated_raw = []
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            shifted = metal_positions.copy()
            shifted[:, 0] += ix * Lx
            shifted[:, 1] += iy * Ly
            replicated_raw.append(shifted)
    replicated_raw = np.vstack(replicated_raw)

    # Step 2: Run scipy Voronoi on smoothed positions
    vor = Voronoi(replicated_smoothed)
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

    # Step 4: Filter out vertices too close to ANY raw atom (min_dist)
    # Use raw positions here — real atom locations, not smoothed ones
    keep_mask = np.ones(len(vertices), dtype=bool)
    for i, v in enumerate(vertices):
        diffs = replicated_raw - v
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
                  cutoff=COORD_CUTOFF_DEFAULT, keep_unclassified=False):
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
    keep_unclassified : bool, optional
        When True, coordination counts outside the oct/tet rules yield
        ``site_type='interstitial'`` with a real composition label instead
        of ``'unknown'`` (used for oxides, whose interstitial geometry does
        not follow FCC oct/tet coordination counts).  Default False.

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
        site_type = "interstitial" if keep_unclassified else "unknown"
        is_distorted = True

    # Distortion score: spread in nearest-neighbor distances
    if n >= 2:
        dists = [c["distance"] for c in coord_list]
        mean_d = float(np.mean(dists))
        spread = float(max(dists) - min(dists))
        distortion_score = spread / mean_d if mean_d > 0 else 0.0
    else:
        distortion_score = 0.0

    # Composition label (only meaningful for classified sites)
    if site_type in ("oct", "tet", "interstitial"):
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
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def build_subsurface_graph(
    slab_path,
    surface_sites_json_path,
    seed,
    n_layers_total=None,
    subsurface_layers=None,
    clustering_tol=CLUSTERING_TOL_DEFAULT,
    min_dist=MIN_DIST_DEFAULT,
    coord_cutoff=COORD_CUTOFF_DEFAULT,
    xy_tol=XY_TOL_DEFAULT,
    z_tol=0.5,
    metal_type='alloy',
):
    """
    Top-level orchestrator. Returns the augmented graph and site list.

    Steps:
      1. Auto-detect the slab's total layer count and derive layer roles:
         bottom ``round(N/3)`` layers are frozen (same fraction as
         :func:`models.structure.compute_z_freeze_cutoff`), the next layer
         up is ``subsurface_1`` (Hop A target), and the layer above that is
         ``subsurface_2`` — the bulk-entry layer (Hop B target; once H
         reaches it, ``ΔH_entry`` from Hop B is what Sieverts' law uses,
         and long-range bulk diffusion is handled separately by
         ``diffusivity_workflow.py``).
      2. Run scipy.spatial.Voronoi to find all interstitial sites.
      3. Filter to sites in ``{subsurface_1, subsurface_2}``.
      4. Classify each as oct/tet with composition label.
      5. Load surface_sites.json, connect subsurface_1 sites to surface sites.
      6. Build NetworkX graph and return it with the sites list.

    Parameters
    ----------
    slab_path : str
        Path to ``clean_slab_reminimized.lammps``.
    surface_sites_json_path : str
        Path to ``surface_sites.json`` produced by NB04b.
    seed : int
        Random seed (kept for API/reproducibility compatibility; no longer
        consumes randomness now that bulk sampling has been removed).
    n_layers_total : int or None, optional
        Total number of layers in the slab.  When ``None`` (default),
        auto-detected via gap-based z-plane clustering
        (:func:`_identify_layers_by_gaps`) — works for any slab height,
        metal or oxide, without a hand-tuned constant.
    subsurface_layers : tuple of int or None, optional
        ``(subsurface_2_layer, subsurface_1_layer)``.  When ``None``
        (default), auto-derived as ``(N-2, N-1)`` where ``N`` is the
        detected/given total layer count.
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
    metal_type : str, optional
        ``'alloy'``/``'pure'`` — rank-based layer binning (immune to
        thermal/alloy z-scatter), oct/tet-only sites.  ``'oxide'`` —
        gap-based plane detection end-to-end, and ALL Voronoi interstitials
        retained (labelled ``'interstitial'`` when outside the FCC oct/tet
        coordination rules).  Default ``'alloy'``.

    Returns
    -------
    G : networkx.Graph
        Augmented graph containing surface site nodes (from JSON), subsurface
        site nodes, ``'surface-subsurface'`` edges, and
        ``'subsurface-subsurface'`` (Hop B) edges.
    subsurface_sites : list of dict
        All classified subsurface_1/subsurface_2 sites with full metadata.
    """
    print(f"=" * 70)
    print(f"  Building subsurface graph for seed {seed}")
    print(f"=" * 70)

    _is_oxide = (metal_type == 'oxide')

    # Step 1: determine the total layer count up front (needed before Voronoi
    # so the rank-path z-smoothing bins atoms into the right number of layers).
    #
    # Priority for metals/alloys:
    #   1. explicit n_layers_total from the caller;
    #   2. the CONSTRUCTION count derived from the slab's own metadata,
    #      n_layers = n_atoms_total // n_atoms_surface (atoms per (111) plane) —
    #      robust to relaxation/rumpling, which blurs adjacent planes and makes
    #      gap-based z-clustering over-count (e.g. a 12-layer Ni slab read as 17);
    #   3. gap-based auto-detect as a last resort (with a warning).
    # Oxide planes have unequal atom counts, so total/surface is meaningless
    # there — oxides keep the gap-based path.
    _layer_src = 'caller (n_layers_total)'
    if n_layers_total is not None:
        _N = n_layers_total
    else:
        _N = None
        if not _is_oxide:
            _N = _n_layers_from_metadata(surface_sites_json_path)
            if _N is not None:
                _layer_src = 'slab metadata (n_atoms_total / n_atoms_surface)'
        if _N is None:
            _probe_atoms = _slab_to_atoms(slab_path)
            _probe_elem = np.array(_probe_atoms.get_chemical_symbols())
            _probe_pos = _probe_atoms.get_positions()[_probe_elem != 'H']
            _, _probe_layer_z = _identify_layers_by_gaps(_probe_pos)
            _N = len(_probe_layer_z)
            _layer_src = ('gap-based auto-detect (fallback — may over-count on '
                          'rumpled/relaxed slabs)')

    n_frozen = round(_N / 3)
    if subsurface_layers is not None:
        _sub2_L, _sub1_L = min(subsurface_layers), max(subsurface_layers)
    else:
        _sub1_L, _sub2_L = _N - 1, _N - 2

    print(f"\nStep 1: {_N} total layers [{_layer_src}] -> frozen: bottom "
          f"{n_frozen}, subsurface_1: L{_sub1_L}, "
          f"subsurface_2 (bulk entry): L{_sub2_L}")

    target_layers = set()
    for _name, _L in (('subsurface_1', _sub1_L), ('subsurface_2', _sub2_L)):
        if 1 <= _L <= _N and _L > n_frozen:
            target_layers.add(_L)
        else:
            print(f"  WARNING: {_name} (layer {_L}) falls inside the frozen "
                  f"bottom {n_frozen} of {_N} layers (or outside the slab) — "
                  f"increase the slab's z-layer count to separate {_name} "
                  f"from the frozen region. Proceeding without it.")

    # Step 2: Find all Voronoi sites
    print(f"\nStep 2: Voronoi tessellation via scipy...")
    sites_cart, structure, atoms = find_voronoi_sites(
        slab_path, clustering_tol=clustering_tol, min_dist=min_dist,
        layer_mode='gaps' if _is_oxide else 'rank', n_layers=_N,
    )
    print(f"  Found {len(sites_cart)} unique interstitial sites in the cell")

    # Step 3: Identify layers
    metal_positions = atoms.get_positions()
    metal_elements  = np.array(atoms.get_chemical_symbols())

    # Filter out H atoms if present (slab should be clean but just in case)
    is_metal = metal_elements != "H"
    metal_positions = metal_positions[is_metal]
    metal_elements  = metal_elements[is_metal]

    if _is_oxide:
        # Gap-based plane detection — oxide planes have unequal atom counts
        # and the plane count is unrelated to the repeat-unit count.
        print(f"\nStep 3: Identifying z-layers by gap detection (oxide)...")
        layer_map, layer_z = _identify_layers_by_gaps(metal_positions)
    else:
        print(f"\nStep 3: Identifying {_N} z-layers...")
        layer_map, layer_z = _identify_layers(metal_positions, n_layers=_N)
    print(f"  Layer z-centers: " +
          ", ".join(f"L{k}={v:.2f}" for k, v in sorted(layer_z.items())))

    cell = atoms.cell.diagonal()

    # Step 4: Assign each Voronoi site to a layer (by nearest atomic plane)
    print(f"\nStep 4: Assigning sites to layers...")

    all_classified_sites = []
    for site_idx, site_pos in enumerate(sites_cart):
        # Layer assignment: nearest atomic plane by z
        z = site_pos[2]
        nearest_layer = min(layer_z.keys(), key=lambda L: abs(layer_z[L] - z))

        if nearest_layer not in target_layers:
            continue

        # Classify the site.  On the oxide path unmatched
        # coordination counts become 'interstitial' (kept) instead of
        # 'unknown' (discarded) — FCC oct/tet rules don't apply there.
        clf = classify_site(
            site_pos, metal_positions, metal_elements, cell,
            cutoff=coord_cutoff, keep_unclassified=_is_oxide,
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

    # Step 5: Assign final layer_classification
    subsurface_sites_final = []
    for s in all_classified_sites:
        L = s["layer_number"]
        if L == _sub1_L:
            s["layer_classification"] = "subsurface_1"
            subsurface_sites_final.append(s)
        elif L == _sub2_L:
            s["layer_classification"] = "subsurface_2"
            subsurface_sites_final.append(s)

    print(f"  Total sites for NB08: {len(subsurface_sites_final)}")

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

    # Sub1–sub2 edges — required by Hop B (find_sub2_neighbor walks
    # G.neighbors of a subsurface_1 node looking for subsurface_2).
    # Connect by periodic xy proximity; if a sub1 site has no sub2 within
    # xy_tol, fall back to its nearest sub2 so Hop B always has a path.
    _sub1 = [s for s in subsurface_sites_final
             if s["layer_classification"] == "subsurface_1"]
    _sub2 = [s for s in subsurface_sites_final
             if s["layer_classification"] == "subsurface_2"]
    n_ss_edges = 0
    for s1 in _sub1:
        added = 0
        best = None   # (distance, site_id)
        for s2 in _sub2:
            d = float(_periodic_xy_distance(s1["position"], s2["position"],
                                            cell))
            if best is None or d < best[0]:
                best = (d, s2["site_id"])
            if d < xy_tol:
                G.add_edge(s1["site_id"], s2["site_id"],
                           edge_type="subsurface-subsurface", distance=d)
                added += 1
        if added == 0 and best is not None:
            G.add_edge(s1["site_id"], best[1],
                       edge_type="subsurface-subsurface", distance=best[0])
            added = 1
        n_ss_edges += added

    print(f"  Surface-subsurface connections: {len(connections)}")
    print(f"  Subsurface1-subsurface2 edges : {n_ss_edges}  (Hop B)")
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
        by_type[s["site_type"]] = by_type.get(s["site_type"], 0) + 1
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
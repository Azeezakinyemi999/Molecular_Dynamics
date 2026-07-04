# Oxide-Slab Support for the H-Permeation Pipeline (Parts 1–2)

## Context

The pipeline computes H transport per material: Part 1 surface NEB (H₂ dissociative
adsorption via ACAT site enumeration + MACE minimizations + NEB), Part 2 permeation
(subsurface entry Hops A/B, vibrations, TST, KMC), Part 3 bulk diffusivity. Parts 1–2
are metal-only today: ACAT can't describe oxide surfaces, `surface_graph.py` hardcodes
12 equal-count layers, and `subsurface_graph.py` uses FCC-tuned classification. The user
needs Cr₂O₃/NiO treated like metals: **dissociative-entry barriers + bulk D(T) + solubility**.

Exploration established the seams are narrow: ACAT's site list enters `surface_graph.py`
at one block and everything downstream reads generic node attributes; the subsurface
finder is already Voronoi-based (structure-agnostic).

**User decisions:** oxide sites = ontop (every top-plane atom) + M–O bridge midpoints;
oxide slab thickness auto-matched to ~22 Å; keep ALL oxide Voronoi interstitials;
FIX the Part-2 path mismatch in pipeline.ipynb (broken for metals too); FIX the dead
Hop B path (missing sub1↔sub2 edges — currently every Hop B job silently skips, metals too).

**Hard constraint:** metal (`alloy`/`pure`) behavior must stay byte-identical — all oxide
behavior behind `metal_type == 'oxide'` routing. `calculation/*.py` are generated; edit
notebooks + `models/` templates only. Commits: no Co-Authored-By tag.

## Implementation

### 1. `models/structure.py` — oxide auto-thickness
In `build_slab()` oxide branch (~line 501): add `oxide_target_thickness=22.0` param.
Build 1-unit and 2-unit probe slabs → `d_unit = t2 - t1` (guard `d_unit<=0.1` → use `t1`);
`n_units = max(1, 1 + round((target - t1)/d_unit))`; build final slab with `n_units`;
log that incoming `layers` is ignored for oxides + chosen n_units + final thickness.
`compute_z_freeze_cutoff` already handles any thickness — no change.

### 2. `models/surface_graph.py` — gap planes + oxide site enumerator
- **2a.** Wrap module-level matplotlib imports (~line 44) in try/except → `plt=None`;
  `visualize_surface_graph` raises clearly if missing (enables local testing).
- **2b.** New `_z_plane_clusters(z, gap_tol=0.5)` — sort z, split at gaps > tol.
- **2c.** New `_enumerate_oxide_sites(pos, syms, cell, top_indices, oxide_bond_cutoff=2.6)`
  returning ACAT-shaped dicts with FULL-slab indices:
  - ontop per top-plane atom: `{'site':'ontop','composition':syms[i],'position':pos[i],'indices':(i,)}`
  - bridge per top-plane pair (min-image xy dist < 2.6 Å, exactly one atom is O):
    midpoint (periodic-aware, wrapped), z = mean; composition metal-first (`'CrO'`); `indices=(i,j)`.
- **2d.** `build_surface_graph(..., metal_type='alloy', n_layers_total=12, oxide_gap_tol=0.5, oxide_exposure_tol=1.0, oxide_bond_cutoff=2.6)`:
  - metal: replace hardcode `_n_layers_est = 12` → `n_layers_total` (default 12, identical).
  - oxide top layer: last z-cluster; merge next-lower cluster while mean-z gap < 1.0 Å
    (composite terminations like Cr₂O₃(0001) O₃/Cr). `z_max` = mean z of selection.
  - oxide: skip top3/z-snap/ACAT block entirely; `sites = _enumerate_oxide_sites(...)`;
    in the site-node loop use identity index map + `z_offset=0`; labels `'{comp}_atop'`/`'{comp}_bridge'`
    (existing label logic covers these); `subsurf_element=''`. Return `(G, slab, None, sites)`.
  - Atom nodes/edges, site-atom edges, shared-atom site-site edges, `build_site_environment`,
    `save_surface_sites`: **unchanged** (verified generic). Ontop–bridge sharing gives
    ontop↔ontop graph-distance 2, matching `graph_dist_min=2` in `enumerate_fs_pairs`.
  - Warn if an oxide surface yields zero bridge sites (pure-O termination beyond tol).

### 3. `models/neb_workflow.py` — thread routing
- `run_phase3_site_enumeration(..., metal_type='alloy', n_layers_total=12)` → forward to
  `build_surface_graph`.
- `orchestrate_slab_prep`: pass `metal_type=metal_type, n_layers_total=layers` at the
  phase-3 call (~line 514). Template already embeds/threads `METAL_TYPE` — no other change.
- Sections B/C and NEB pools: no changes (schema-generic; verified).

### 4. `models/subsurface_graph.py` — oxide layers, keep-all, Hop B edges
- New `_identify_layers_by_gaps(positions, gap_tol=0.5)` mirroring `_identify_layers`
  contract (`{atom_idx: layer}, {layer: mean_z}`); duplicate clustering logic here
  (module is deliberately standalone).
- `find_voronoi_sites(..., layer_mode='rank')`: `'gaps'` uses the new helper for z-smoothing.
- `classify_site(..., keep_unclassified=False)`: True → unmatched coordination gives
  `site_type='interstitial'` + composition label, instead of `'unknown'`.
- `build_subsurface_graph(..., metal_type='alloy')`:
  - oxide: gap layers; `N=len(layer_z)`; `subsurface_layers=(N-2, N-1)`;
    `bulk_sample_layers=range(max(2,N-8), N-2)`; `layer_mode='gaps'`;
    `keep_unclassified=True`; do NOT discard `'unknown'` sites on this path.
  - metal: identical to today.
- **Hop B fix (metals + oxides):** in the edge-building step (~line 802), also add
  `sub1 ↔ sub2` edges by periodic xy proximity (reuse `_periodic_xy_distance`, same
  1.5 Å default as `connect_to_surface`; fall back to nearest-sub2-per-sub1 if a sub1
  has no match within tol) with `edge_type='subsurface-subsurface'` so
  `find_sub2_neighbor` (neb_subsurface.py:34) finds neighbors and Hop B jobs generate.

### 5. `models/permeation_workflow.py` — metal_type + species generalization
- `generate_permeation_scripts(..., metal_type='alloy')`; emit `METAL_TYPE = {metal_type!r}`
  in `_header`.
- In `_PERMEATION_BODY`:
  - `build_subsurface_graph(..., metal_type=METAL_TYPE)` (~line 84).
  - `_slab_species = sorted({s != 'H'}, key=len, reverse=True)` from the already-read slab;
    replace hardcoded `('Ni','Mo','Cr','Fe')` scan (~line 288) with `_slab_species`, plus
    fallback: resolve bare `s_NN` sids via `surface_sites.json` level1 composition
    (hardens the metal path too — bare sids currently yield empty k_entry).
  - Placeholder diss pairs (~311–313): `itertools.combinations_with_replacement(sorted(_slab_species), 2)`
    (confirm `kmc.element_pair` uses sorted-tuple keys).
  - KMC grid: when `METAL_TYPE=='oxide'`, pass `composition=` from normalized
    `surface_atoms` element counts to `sweep_pressure`; metals keep default (unchanged).

### 6. `calculation/pipeline.ipynb` (JSON-edit cells; regenerate `calculation/*.py` after)
- Cell 12 (permeation loop): add `metal_type = cfg['type'],` AND fix the path mismatch:
  - `relaxed_slab_path` → `slabs/{stem}/phase2_relax/relaxed_slab.lammps`
  - `surface_sites_json` → `slabs/{stem}/phase3_sites/surface_sites.json`
  - `phase2_h_dir` → `adsorption/{stem}/phase2_h` (verify exact dir Part 1 writes:
    `adsorption/{stem}/phase2_h/results` holds logs; check what permeation_run reads —
    hopa needs `dedup_is_labels` is_paths from Section B phase-2 results)
- Cell 8 already passes `metal_type=cfg['type']` — no change.
- `calculation/permeation.ipynb` / `neb_calculation.ipynb`: add `metal_type` only where used.

## Edge cases to honor
- Bridge midpoints across PBC: min-image displacement, wrap into cell.
- Oxide `atom_indices` all reference top-plane atom nodes → site-atom/site-site edges never drop.
- O-top NN spacing ~2.7–3.0 Å is inside the sep 2.5–5.0 Å window — verify pair count > 0 in smoke.
- `LAT_REPEAT=(5,6)` on an oxide primitive may explode atom count — check in smoke test;
  add per-type lateral repeat only if needed.

## Verification
Local (`/Users/akinyemi.az/anaconda3/bin/python3`; ase/scipy/networkx OK, spglib absent,
matplotlib/pandas broken → 2a required first):
1. `import models.surface_graph` succeeds after lazy-matplotlib.
2. Synthetic NiO rocksalt slab via `ase.build.bulk('NiO','rocksalt',a=4.17)` + `surface()`
   (no spglib) → `build_surface_graph(metal_type='oxide')` + environments + save:
   assert ontop+bridge counts > 0, site-site edges > 0, every site has constituent_atoms
   + level2 shell1; `_site_signature` dedup collapses symmetry-equivalent sites;
   `load_neb_pools`-critical keys present.
3. `build_subsurface_graph(metal_type='oxide')` on same slab: gap plane count correct,
   sub1/sub2 = first/second planes below surface, interstitials retained,
   `connect_to_surface` > 0, **sub1–sub2 edges > 0** (Hop B fix).
4. Metal regression: run both builders with defaults on the existing relaxed metal slab
   (tests/pipeline/work/slab/slab_relaxed.lammps) — outputs identical to pre-change,
   EXCEPT new sub1–sub2 edges (intended Hop B fix).
5. Generated-file diffs: regenerate a metal `permeation_run_*.py` — only intended additions
   (METAL_TYPE line, species generalization); metal `neb_run_*.py` unchanged.
Cluster:
6. `build_slab` oxide: log shows n_units + final thickness ≈ 22 Å; z-freeze in a gap.
7. Part 1 Section A on small oxide → surface_sites.json/gml → pools load, pairs > 0.
8. `permeation_run_Cr_oxide_supercell.py` header `METAL_TYPE='oxide'`; k_entry species show Cr/O.

## Records
Update memory `project_pipeline_test_bugs.md`: Hop B dead-path bug (fixed), Part-2 path
mismatch (fixed), oxide support implemented (resolves the `_n_layers_est` open item).

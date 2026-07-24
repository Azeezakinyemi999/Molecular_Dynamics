"""
models/neb_subsurface.py
========================
NEB calculations for H permeation from the surface into the subsurface.

Two hop types, following the same FS-min → NEB pattern as neb_workflow.py Section C:

  Hop A: H* at surface adsorption site  →  H at subsurface-1 oct site (between layers 11-12)
  Hop B: H at subsurface-1 oct site     →  H at subsurface-2 oct site (between layers 10-11)

Relies on models.subsurface_graph for interstitial site geometry.

Execution order (from the calling notebook):
  1. orchestrate_hopa_neb()   — generate Hop A FS-min + NEB scripts
  2. [submit hopa_fsmin_array.sh on cluster, wait for completion]
  3. orchestrate_hopb_neb()   — generate Hop B FS-min + NEB scripts
     (uses Hop A relaxed structures as IS; parses E_IS from Hop A fs_min.log)
  4. [submit hopb_fsmin_array.sh + hopa_neb_array.sh in parallel, then hopb_neb_array.sh]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ase.io import read as ase_read


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_sub2_neighbor(G, ss1_id: str, subsurface_sites: list) -> str:
    """Return the subsurface_2 node in G that is directly below ss1_id.

    Selects the neighbor with layer_classification='subsurface_2' that has
    the smallest z-distance from ss1 (going deeper into the slab).

    Raises
    ------
    ValueError
        If no subsurface_2 neighbor is found for ss1_id.
    """
    site_lookup = {s['site_id']: s for s in subsurface_sites}
    z1 = site_lookup[ss1_id]['position'][2]

    best_id = None
    best_dz = float('inf')
    for nbr in G.neighbors(ss1_id):
        nbr_data = G.nodes[nbr]
        if nbr_data.get('layer_classification') != 'subsurface_2':
            continue
        z2 = nbr_data.get('position', [0, 0, 0])[2]
        dz = abs(z1 - z2)
        if dz < best_dz:
            best_dz = dz
            best_id = nbr

    if best_id is None:
        raise ValueError(
            f'No subsurface_2 neighbor found for {ss1_id}. '
            f'build_subsurface_graph may have warned that subsurface_2 falls '
            f'inside the frozen region — increase the slab z-layer count.'
        )
    print(f'[sub2 neighbor] {ss1_id} → {best_id}  dz={best_dz:.3f} Å')
    return best_id


def _closest_sub2(G, ss1_id: str, site_lookup: dict) -> tuple:
    """Non-raising variant of ``find_sub2_neighbor``.

    Returns ``(ss2_id, dz)`` for the closest-below subsurface_2 neighbour of
    ``ss1_id``, or ``(None, None)`` when the graph has no subsurface_2 neighbour
    for it (e.g. sub2 fell inside the frozen region). Used by
    :func:`build_sub1_sub2_map`, which must record the unmatched case rather
    than abort the whole map — ``find_sub2_neighbor`` is kept as-is for the
    single-site call sites that legitimately want the hard error.
    """
    z1 = site_lookup[ss1_id]['position'][2]
    best_id, best_dz = None, float('inf')
    for nbr in G.neighbors(ss1_id):
        if G.nodes[nbr].get('layer_classification') != 'subsurface_2':
            continue
        z2 = G.nodes[nbr].get('position', [0, 0, 0])[2]
        dz = abs(z1 - z2)
        if dz < best_dz:
            best_dz, best_id = dz, nbr
    return (best_id, best_dz if best_id is not None else None)


def oct_env_label(site: dict) -> str:
    """Environment label for a subsurface octahedral site.

    Uses the site's ``composition_label`` (already a deterministic
    coordination fingerprint like ``'Ni3MoCrFe_oct'``, produced by
    ``subsurface_graph._composition_label``) as the per-environment key that
    Hop A/B rates and the KMC grid are keyed on. Falls back to
    ``'unknown_env'`` if the label is absent, never a silent empty string.
    """
    return str(site.get('composition_label') or 'unknown_env')


# ─────────────────────────────────────────────────────────────────────────────
# Section — Subsurface entry maps (surface → sub1 → sub2)
#
# These build and persist the physically-anchored path maps that drive Hop A/B:
# the H that enters the bulk is the H that dissociated on the surface, threaded
# through the octahedral interstitial sites. See the approved plan for the full
# rationale.
# ─────────────────────────────────────────────────────────────────────────────

def build_sub1_sub2_map(
    subsurface_graph: tuple,
    out_json: str | None = None,
) -> dict:
    """Map each subsurface-1 oct site to its closest-below subsurface-2 oct site.

    Enumerated independently of the surface (per the reframing): the sub1↔sub2
    connectivity is a property of the lattice, not of which H happens to be
    entering. Unequal sub1/sub2 counts are handled explicitly — a sub1 with no
    sub2 neighbour is recorded with ``sub2_id: None`` and a reason, never dropped.

    Parameters
    ----------
    subsurface_graph : (G, subsurface_sites)
        Output of ``models.subsurface_graph.build_subsurface_graph``.
    out_json : str, optional
        If given, the map is written here as ``sub1_sub2_map.json``.

    Returns
    -------
    dict
        ``{ss1_id: {sub2_id, z_dist, sub1_env, sub2_env}}`` for matched sub1
        sites, and ``{ss1_id: {sub2_id: None, reason}}`` for unmatched ones.
    """
    G, subsurface_sites = subsurface_graph
    site_lookup = {s['site_id']: s for s in subsurface_sites}
    sub1_ids = [s['site_id'] for s in subsurface_sites
                if s.get('layer_classification') == 'subsurface_1']

    mapping: dict = {}
    n_matched = 0
    for ss1_id in sub1_ids:
        ss2_id, dz = _closest_sub2(G, ss1_id, site_lookup)
        if ss2_id is None:
            mapping[ss1_id] = {
                'sub2_id': None,
                'reason': 'no subsurface_2 neighbour in graph '
                          '(sub2 may fall inside the frozen region)',
                'sub1_env': oct_env_label(site_lookup[ss1_id]),
            }
            continue
        mapping[ss1_id] = {
            'sub2_id': ss2_id,
            'z_dist': float(dz),
            'sub1_env': oct_env_label(site_lookup[ss1_id]),
            'sub2_env': oct_env_label(site_lookup[ss2_id]),
        }
        n_matched += 1

    print(f'[sub1→sub2 map] {n_matched}/{len(sub1_ids)} sub1 sites mapped to a sub2 '
          f'({len(sub1_ids) - n_matched} unmatched)')
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, 'w') as fh:
            json.dump(mapping, fh, indent=2)
        print(f'  → {out_json}')
    return mapping


def collect_entry_h_sources(
    neb_dir: str,
    phase2_h_dir: str,
    out_json: str | None = None,
) -> list:
    """Collect the surface H\\* atoms that seed Hop A, from dissociation products.

    The H that enters the subsurface is the H that dissociated on the surface,
    so Hop A is seeded from the 2H\\* final states of each *converged* surface
    dissociation NEB run — not the full standalone adsorption-site enumeration
    (that decoupling was the over-enumeration bug this replaces).

    Each converged run in ``ranked_barriers.json`` records ``fs_site1``/
    ``fs_site2`` (the two H\\* surface site ids). Those ids map directly to the
    already-relaxed single-H\\* structures ``h_atom_{sid}_relaxed.lammps`` and
    their ``h_min_{sid}.log`` energies in ``phase2_h_dir`` — copied, not rebuilt.

    A given surface site can be an FS product of more than one dissociation run;
    it is emitted once, with every source run recorded for provenance.

    Parameters
    ----------
    neb_dir : str
        Directory containing the dissociation ``ranked_barriers.json``.
    phase2_h_dir : str
        Directory with ``h_atom_{sid}_relaxed.lammps`` / ``h_min_{sid}.log``.
    out_json : str, optional
        If given, the provenance record is written here as
        ``entry_h_sources.json``.

    Returns
    -------
    list of (sid, is_path, e_is)
        Same triple shape as ``collect_dedup_is_labels`` (drop-in for
        ``orchestrate_hopa_neb``), sorted by ``sid``. Sites whose structure or
        energy log is missing/unparseable are skipped with a warning — never a
        fabricated energy.
    """
    from models.parsers import parse_energy_log

    ranked_json = os.path.join(neb_dir, 'ranked_barriers.json')
    if not os.path.exists(ranked_json):
        raise FileNotFoundError(
            f'{ranked_json} not found — run Part 1 surface dissociation NEB '
            f'(collect_neb_results) before collecting entry H* sources.'
        )
    with open(ranked_json) as fh:
        ranked = json.load(fh)

    # sid -> set of source dissociation-run labels (order-preserving via dict)
    sources: dict = {}
    for job in ranked:
        if not job.get('converged', False):
            continue
        run_label = job.get('label', '')
        for key in ('fs_site1', 'fs_site2'):
            sid = job.get(key)
            if sid is None:
                continue
            sources.setdefault(str(sid), [])
            if run_label and run_label not in sources[str(sid)]:
                sources[str(sid)].append(run_label)

    triples: list = []
    records: list = []
    for sid in sorted(sources):
        is_path = os.path.join(phase2_h_dir, f'h_atom_{sid}_relaxed.lammps')
        log_p   = os.path.join(phase2_h_dir, f'h_min_{sid}.log')
        if not os.path.exists(is_path):
            print(f'  WARNING: entry H* sid={sid}: {is_path} missing — skipping.')
            continue
        parsed = parse_energy_log(log_p) if os.path.exists(log_p) else None
        if not parsed or 'pe_final_eV' not in parsed:
            print(f'  WARNING: entry H* sid={sid}: could not parse E_IS from '
                  f'{log_p} — skipping (no fabricated energy).')
            continue
        e_is = parsed['pe_final_eV']
        triples.append((sid, is_path, e_is))
        records.append({
            'surface_sid':      sid,
            'source_diss_runs': sources[sid],
            'is_structure':     is_path,
            'e_is_eV':          e_is,
        })

    print(f'[entry H* sources] {len(triples)} surface H* from '
          f'{sum(1 for j in ranked if j.get("converged"))} converged dissociation runs')
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, 'w') as fh:
            json.dump(records, fh, indent=2)
        print(f'  → {out_json}')
    return triples


def build_surface_sub1_sub2_map(
    entry_sources: list,
    surface_connections: list,
    sub1_sub2_map: dict,
    subsurface_graph: tuple,
    out_json: str | None = None,
) -> list:
    """Join entry H\\* → closest sub1 oct → mapped sub2 oct into the full path map.

    For each entry H\\* surface site, picks the closest subsurface-1 oct site
    (smallest xy-distance, from ``connect_to_surface``) and looks up that sub1's
    mapped sub2 (from :func:`build_sub1_sub2_map`), attaching both environment
    labels.

    Two-H\\*-to-same-sub1 collapse: if several entry H\\* map to the *same* sub1
    oct site, only the lowest-energy IS is kept as the Hop A pathway (they share
    the same FS); the dropped alternates are recorded on the survivor.

    Parameters
    ----------
    entry_sources : list of (sid, is_path, e_is)
        Output of :func:`collect_entry_h_sources`.
    surface_connections : list of (ss1_id, surface_sid, xy_dist)
        Output of ``models.subsurface_graph.connect_to_surface``.
    sub1_sub2_map : dict
        Output of :func:`build_sub1_sub2_map`.
    subsurface_graph : (G, subsurface_sites)
        Used only for the sub1 position/env lookup.
    out_json : str, optional
        If given, written here as ``surface_sub1_sub2_map.json``.

    Returns
    -------
    list of dict
        One entry per surviving Hop A pathway, each with keys:
        ``surface_sid, is_path, e_is, sub1_id, sub1_env, sub1_xyz,
        sub2_id, sub2_env, collapsed_from``.
    """
    _G, subsurface_sites = subsurface_graph
    site_lookup = {s['site_id']: s for s in subsurface_sites}

    # surface_sid -> (ss1_id, xy_dist), keeping the closest sub1 if several match
    surf_to_ss1: dict = {}
    for ss1_id, surf_id, xy_dist in surface_connections:
        if surf_id not in surf_to_ss1 or xy_dist < surf_to_ss1[surf_id][1]:
            surf_to_ss1[surf_id] = (ss1_id, float(xy_dist))

    # Build one candidate per entry H* that has a sub1 above it
    candidates: list = []
    for sid, is_path, e_is in entry_sources:
        if sid not in surf_to_ss1:
            print(f'[surface→sub1→sub2] no sub1 above surface sid={sid} — skipping.')
            continue
        ss1_id, xy_dist = surf_to_ss1[sid]
        s1_entry = sub1_sub2_map.get(ss1_id, {})
        candidates.append({
            'surface_sid': sid,
            'is_path':     is_path,
            'e_is':        e_is,
            'sub1_id':     ss1_id,
            'sub1_env':    oct_env_label(site_lookup[ss1_id]) if ss1_id in site_lookup else 'unknown_env',
            'sub1_xyz':    site_lookup[ss1_id]['position'] if ss1_id in site_lookup else None,
            'sub2_id':     s1_entry.get('sub2_id'),
            'sub2_env':    s1_entry.get('sub2_env'),
            'xy_dist':     xy_dist,
        })

    # Collapse two-H*-to-same-sub1: keep the lowest-e_is IS per sub1 oct
    by_sub1: dict = {}
    for c in candidates:
        key = c['sub1_id']
        if key not in by_sub1 or c['e_is'] < by_sub1[key]['e_is']:
            by_sub1[key] = c
    # record dropped alternates on the survivor
    dropped: dict = {}
    for c in candidates:
        key = c['sub1_id']
        if by_sub1[key]['surface_sid'] != c['surface_sid']:
            dropped.setdefault(key, []).append(c['surface_sid'])

    path_map: list = []
    for key, c in by_sub1.items():
        path_map.append({
            'surface_sid':   c['surface_sid'],
            'is_path':       c['is_path'],
            'e_is':          c['e_is'],
            'sub1_id':       c['sub1_id'],
            'sub1_env':      c['sub1_env'],
            'sub1_xyz':      c['sub1_xyz'],
            'sub2_id':       c['sub2_id'],
            'sub2_env':      c['sub2_env'],
            'collapsed_from': dropped.get(key, []),
        })
    path_map.sort(key=lambda x: str(x['surface_sid']))

    n_collapsed = sum(len(v) for v in dropped.values())
    print(f'[surface→sub1→sub2] {len(path_map)} Hop A pathways '
          f'({n_collapsed} surface H* collapsed onto a shared sub1)')
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, 'w') as fh:
            json.dump(path_map, fh, indent=2)
        print(f'  → {out_json}')
    return path_map


# ─────────────────────────────────────────────────────────────────────────────
# Structure builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_single_h_fs(
    source_lammps: str,
    new_h_xyz: list | np.ndarray,
    masses: dict,
    e2t: dict,
    out_path: str,
    comment: str = '',
) -> str:
    """Read source LAMMPS (slab + 1 H), strip the H, place H at new_h_xyz.

    Shared implementation for build_hopa_fs and build_hopb_fs.  The metal
    positions are taken from source_lammps so that IS and FS share the same
    underlying slab configuration.
    """
    from models.structure import write_lammps_data

    atoms = ase_read(source_lammps, format='lammps-data', atom_style='atomic')
    atoms.wrap()
    pos   = atoms.get_positions()
    syms  = np.array(atoms.get_chemical_symbols())
    cell  = atoms.cell.lengths()

    metal_mask = syms != 'H'
    metal_pos  = pos[metal_mask]
    metal_syms = list(syms[metal_mask])

    h_pos = np.array(new_h_xyz, dtype=float).reshape(1, 3)
    return write_lammps_data(
        symbols=metal_syms + ['H'],
        positions=np.vstack([metal_pos, h_pos]),
        cell_lengths=cell,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment=comment or f'FS raw: H at oct site {list(new_h_xyz)}',
    )


def build_hopa_fs(
    is_path: str,
    sub1_xyz: list | np.ndarray,
    masses: dict,
    e2t: dict,
    out_path: str,
) -> str:
    """Build raw FS for Hop A: strip surface H from IS, place H at sub1 oct site.

    Metal positions come from is_path (h_atom_{sid}_relaxed.lammps) so the
    IS and FS share the same slab configuration — only H moves.
    """
    return _build_single_h_fs(
        source_lammps=is_path,
        new_h_xyz=sub1_xyz,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment='Hop A FS raw: H at subsurface-1 oct site',
    )


def build_hopb_fs(
    hopb_is_path: str,
    sub2_xyz: list | np.ndarray,
    masses: dict,
    e2t: dict,
    out_path: str,
) -> str:
    """Build raw FS for Hop B: strip sub1 H from the Hop-A-relaxed IS, place H at sub2 oct site."""
    return _build_single_h_fs(
        source_lammps=hopb_is_path,
        new_h_xyz=sub2_xyz,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment='Hop B FS raw: H at subsurface-2 oct site',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hop A: Surface → Subsurface-1
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate_hopa_neb(
    dedup_is_labels: list,
    subsurface_graph: tuple,
    surface_connections: list,
    outdir: str,
    masses: dict | None = None,
    e2t: dict | None = None,
    elem_str: str | None = None,
    slurm_opts: dict | None = None,
    neb_slurm_opts: dict | None = None,
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 0.05,
    dry_run: bool = True,
    z_freeze_cutoff: float | None = None,
) -> dict:
    """Write Hop A NEB scripts (surface H* → subsurface-1 oct site).

    Parameters
    ----------
    dedup_is_labels : list of (sid, is_path, e_is)
        sid      — surface site id (str), same labels used in Section B dedup.
        is_path  — path to h_atom_{sid}_relaxed.lammps.
        e_is     — PE of IS in eV, from Section B Phase 2 minimisation log.
    subsurface_graph : (G, subsurface_sites)
        Output of models.subsurface_graph.build_subsurface_graph.
    surface_connections : list of (ss_id, surface_sid, xy_dist)
        Output of models.subsurface_graph.connect_to_surface.
    outdir : str
        Base output directory; job dirs go to {outdir}/hopa/{sid}/.
    masses : dict, optional  (defaults to MASSES_7)
    e2t    : dict, optional  (defaults to E2T_7)
    slurm_opts     : GPU SLURM config for LAMMPS FS-min. Defaults to multigpu.
    neb_slurm_opts : CPU SLURM config for ASE NEB. Defaults to short, no GPU.
    n_images, spring_const, neb_ftol : NEB parameters.
    dry_run : bool  — write scripts without submitting to SLURM.

    Returns
    -------
    dict
        {jobs, fsmin_array, neb_array, jobs_json, n_jobs, status}

        Each job dict:
          sid, is_path, e_is, ss1_id, sub1_xyz,
          fs_raw, fs_relaxed, fsmin_script, neb_script,
          fsmin_sh, neb_sh, barrier_file, path_file, job_dir
    """
    from models.config import (
        SLURM_DEFAULTS, LAMMPS_CMD, MACE_MODEL_LAMMPS, MACE_MODEL_ASE,
        PAIR_STYLE, PAIR_SUFFIX, ELEM_STR_7, KOKKOS_FLAGS,
        MASSES_7, E2T_7, Z_FREEZE_CUTOFF, FTOL,
    )
    from models.lammps_script import write_adsorbate_min_script
    from models.ase_neb import run_neb_pipeline
    from models.create_slurm import write_slurm_job, write_chained_slurm_job

    if masses is None:
        masses = MASSES_7
    if e2t is None:
        e2t = E2T_7
    if elem_str is None:
        elem_str = ELEM_STR_7
    if slurm_opts is None:
        slurm_opts = {**SLURM_DEFAULTS, 'partition': 'sharing', 'time': '01:00:00'}
    if neb_slurm_opts is None:
        neb_slurm_opts = {
            **SLURM_DEFAULTS,
            'partition': 'short',
            'time': '12:00:00',
            'gpu': None,
            'cuda_version': None,
        }

    if z_freeze_cutoff is None:
        # Auto: bottom 1/3 of the slab the IS structures were built from;
        # falls back to the config constant if no IS structure is available.
        if dedup_is_labels and os.path.exists(dedup_is_labels[0][1]):
            from models.structure import compute_z_freeze_cutoff
            z_freeze_cutoff = compute_z_freeze_cutoff(dedup_is_labels[0][1])
        else:
            z_freeze_cutoff = Z_FREEZE_CUTOFF

    G, subsurface_sites = subsurface_graph
    site_lookup = {s['site_id']: s for s in subsurface_sites}

    # surface_sid → (ss1_id, xy_dist) — keep smallest xy_dist if multiple matches
    surf_to_ss1: dict = {}
    for ss_id, surf_id, xy_dist in surface_connections:
        if surf_id not in surf_to_ss1 or xy_dist < surf_to_ss1[surf_id][1]:
            surf_to_ss1[surf_id] = (ss_id, float(xy_dist))

    hopa_dir = Path(outdir) / 'hopa'
    hopa_dir.mkdir(parents=True, exist_ok=True)
    kk = ' '.join(KOKKOS_FLAGS)

    jobs = []

    for entry in dedup_is_labels:
        sid, is_path, e_is = entry

        if sid not in surf_to_ss1:
            print(f'[Hop A] No sub1 site paired to surface sid={sid} — skipping.')
            continue

        ss1_id, _ = surf_to_ss1[sid]
        sub1_xyz   = site_lookup[ss1_id]['position']

        job_dir = hopa_dir / str(sid)
        job_dir.mkdir(parents=True, exist_ok=True)

        fs_raw       = str(job_dir / 'sub1_fs_raw.lammps')
        fs_relaxed   = str(job_dir / 'sub1_fs_relaxed.lammps')
        fsmin_log    = str(job_dir / 'fs_min.log')
        min_script   = str(job_dir / 'min_fs.lammps')
        barrier_file = str(job_dir / 'neb_barrier.txt')
        path_file    = str(job_dir / 'neb_path.dat')

        # Skip regenerating already-completed work's SLURM commands -- matches
        # Section C's FS-min skip convention in neb_workflow.py (existence-only
        # check against the real output; write_data is always the last command
        # its writer emits, so a killed job never leaves a partial output file
        # here). Structure/script generation below stays unconditional (cheap
        # local Python, not the compute resource being protected) -- only the
        # SLURM command embedded in each per-site .sh becomes a no-op stub, so
        # the array still submits its full range but wastes no real cluster
        # compute on already-done sites.
        fs_already_done  = os.path.exists(fs_relaxed)
        neb_already_done = os.path.exists(barrier_file)

        # 1. FS raw
        build_hopa_fs(
            is_path=is_path,
            sub1_xyz=sub1_xyz,
            masses=masses,
            e2t=e2t,
            out_path=fs_raw,
        )

        # 2. LAMMPS FS-min script (same CG pattern as Section C)
        write_adsorbate_min_script(
            slab_ads_input=fs_raw,
            ads_output=fs_relaxed,
            out_path=min_script,
            pair_style=PAIR_STYLE,
            mace_model=MACE_MODEL_LAMMPS,
            pair_suffix=PAIR_SUFFIX,
            elem_str=elem_str,
            z_freeze_cutoff=z_freeze_cutoff,
            ftol=FTOL,
        )

        # 3. ASE NEB script — E_IS hardcoded, E_FS parsed at runtime from fs_min.log
        traj_p1    = str(job_dir / 'hopa_phase1.traj')
        traj_p2    = str(job_dir / 'hopa_phase2.traj')
        neb_script = run_neb_pipeline(
            is_file=is_path,
            fs_file=fs_relaxed,
            e_is=float(e_is),
            mace_model_path=MACE_MODEL_ASE,
            barrier_file=barrier_file,
            path_file=path_file,
            outdir=str(job_dir),
            fs_log_file=fsmin_log,
            job_name='hopa',
            n_images=n_images,
            spring_const=spring_const,
            neb_ftol=neb_ftol,
            z_freeze_cutoff=z_freeze_cutoff,
            device='cpu',
            label_is=f'surface:{sid}',
            label_fs=f'sub1:{ss1_id}',
            traj_phase1=traj_p1,
            traj_phase2=traj_p2,
            masses=masses,
            e2t=e2t,
        )

        # 4a. GPU SLURM: FS-min (no-op stub if already done -- job_index.txt/the
        # array range are shared between fsmin_array and neb_array (both index
        # by array-task-ID), so an already-done site's sid must stay in the
        # index; a no-op stub keeps array indexing intact while skipping the
        # wasted resubmission.
        fsmin_sh = str(job_dir / f'slurm_fsmin_{sid}.sh')
        _fsmin_commands = (
            [f'echo "[fsmin skip] {sid}: {fs_relaxed} already exists — skipping FS-min"']
            if fs_already_done else
            [f'{LAMMPS_CMD} {kk} -in {min_script} -log {fsmin_log}']
        )
        write_slurm_job(
            job_name=f'fsmin_hopa_{sid}',
            slurm_config=slurm_opts,
            out_path=fsmin_sh,
            commands=_fsmin_commands,
        )

        # 4b. CPU SLURM: NEB (self-chaining via traj checkpoint; no-op stub if
        # this pair's NEB already converged).
        neb_sh = str(job_dir / f'slurm_neb_{sid}.sh')
        if neb_already_done:
            write_slurm_job(
                job_name=f'neb_hopa_{sid}',
                slurm_config=neb_slurm_opts,
                out_path=neb_sh,
                commands=[f'echo "[neb skip] {sid}: {barrier_file} already exists — skipping NEB"'],
            )
        else:
            _neb_wall = neb_slurm_opts.get('time', '12:00:00')
            _h, _m, _s = (int(x) for x in _neb_wall.split(':'))
            _neb_cutoff_secs = _h * 3600 + _m * 60 + _s - 300
            _neb_cutoff = f'{_neb_cutoff_secs // 3600:02d}:{(_neb_cutoff_secs % 3600) // 60:02d}:{_neb_cutoff_secs % 60:02d}'
            write_chained_slurm_job(
                job_name=f'neb_hopa_{sid}',
                slurm_config=neb_slurm_opts,
                out_path=neb_sh,
                first_commands=[f'python {neb_script}'],
                restart_commands=[f'python {neb_script}'],
                restart_glob=traj_p2,
                cutoff=_neb_cutoff,
                work_dir=str(job_dir),
            )

        if not dry_run:
            from models.create_slurm import submit_with_retry
            if neb_already_done:
                print(f"  Hop A {sid}: already done ({barrier_file}) — skipping submission")
            else:
                if fs_already_done:
                    submit_with_retry(neb_sh)
                else:
                    fsmin_jid = submit_with_retry(fsmin_sh)
                    dep = f'afterok:{fsmin_jid}' if fsmin_jid else None
                    submit_with_retry(neb_sh, dependency=dep)

        jobs.append({
            'sid'         : sid,
            'is_path'     : is_path,
            'e_is'        : float(e_is),
            'ss1_id'      : ss1_id,
            'sub1_xyz'    : sub1_xyz,
            # Oct-site coordination environment of the sub1 destination — the
            # per-environment key Part 6 uses to build env-resolved k_entry/
            # k_exit (instead of the old collapse to one rate per element).
            'sub1_env'    : oct_env_label(site_lookup[ss1_id]),
            'fs_raw'      : fs_raw,
            'fs_relaxed'  : fs_relaxed,
            'fsmin_script': min_script,
            'neb_script'  : neb_script,
            'fsmin_sh'    : fsmin_sh,
            'neb_sh'      : neb_sh,
            'barrier_file': barrier_file,
            'path_file'   : path_file,
            'job_dir'     : str(job_dir),
        })

    if not jobs:
        return {'jobs': [], 'n_jobs': 0, 'status': 'no_jobs'}

    # job_index.txt — one sid per line (1-indexed for SLURM array)
    job_index_path = hopa_dir / 'job_index.txt'
    job_index_path.write_text('\n'.join(str(j['sid']) for j in jobs) + '\n')

    ar = (0, len(jobs) - 1)

    # GPU array: FS-min
    fsmin_array = str(hopa_dir / 'hopa_fsmin_array.sh')
    write_slurm_job(
        job_name='hopa_fsmin_array',
        slurm_config=slurm_opts,
        out_path=fsmin_array,
        array_range=ar,
        concurrent=2,
        commands=[
            f'SID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {job_index_path})',
            f'bash {hopa_dir}/${{SID}}/slurm_fsmin_${{SID}}.sh',
        ],
    )

    # CPU array: NEB
    neb_array = str(hopa_dir / 'hopa_neb_array.sh')
    write_slurm_job(
        job_name='hopa_neb_array',
        slurm_config=neb_slurm_opts,
        out_path=neb_array,
        array_range=ar,
        concurrent=50,
        commands=[
            f'SID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {job_index_path})',
            f'bash {hopa_dir}/${{SID}}/slurm_neb_${{SID}}.sh',
        ],
    )

    jobs_json = str(hopa_dir / 'hopa_jobs.json')
    with open(jobs_json, 'w') as fh:
        json.dump(jobs, fh, indent=2)

    status = 'submitted' if not dry_run else 'generated'
    print(f'[Hop A] {len(jobs)} jobs ({status})')
    print(f'  job_index   : {job_index_path}')
    print(f'  fsmin array : {fsmin_array}')
    print(f'  neb array   : {neb_array}')
    print(f'  jobs json   : {jobs_json}')

    return {
        'jobs'       : jobs,
        'fsmin_array': fsmin_array,
        'neb_array'  : neb_array,
        'jobs_json'  : jobs_json,
        'n_jobs'     : len(jobs),
        'status'     : status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hop B: Subsurface-1 → Subsurface-2 (one-to-one with Hop A)
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate_hopb_neb(
    hopa_jobs: list,
    hopa_outdir: str,
    subsurface_graph: tuple,
    outdir: str,
    masses: dict | None = None,
    e2t: dict | None = None,
    elem_str: str | None = None,
    slurm_opts: dict | None = None,
    neb_slurm_opts: dict | None = None,
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 0.05,
    dry_run: bool = True,
    z_freeze_cutoff: float | None = None,
) -> dict:
    """Write Hop B NEB scripts (subsurface-1 → subsurface-2 oct site).

    Call this AFTER Hop A FS-min jobs have completed on the cluster.
    The relaxed sub1 structures and their energy logs must be present
    in hopa_outdir/{sid}/ before this function is called.

    Parameters
    ----------
    hopa_jobs : list of dict
        orchestrate_hopa_neb()['jobs'] — provides sid and ss1_id for each job.
    hopa_outdir : str
        Directory where Hop A wrote its outputs.
        Expects {hopa_outdir}/{sid}/sub1_fs_relaxed.lammps and fs_min.log.
    subsurface_graph : (G, subsurface_sites)
        Same output passed to orchestrate_hopa_neb.
    outdir : str
        Base output directory; job dirs go to {outdir}/hopb/{sid}/.
    masses, e2t, slurm_opts, neb_slurm_opts, n_images, spring_const, neb_ftol, dry_run
        Same semantics as orchestrate_hopa_neb.

    Returns
    -------
    dict
        {jobs, fsmin_array, neb_array, jobs_json, n_jobs, status}

        Each job dict:
          sid, ss1_id, ss2_id, hopb_is, sub2_xyz,
          fs_raw, fs_relaxed, fsmin_script, neb_script,
          fsmin_sh, neb_sh, barrier_file, path_file, job_dir
    """
    from models.config import (
        SLURM_DEFAULTS, LAMMPS_CMD, MACE_MODEL_LAMMPS, MACE_MODEL_ASE,
        PAIR_STYLE, PAIR_SUFFIX, ELEM_STR_7, KOKKOS_FLAGS,
        MASSES_7, E2T_7, Z_FREEZE_CUTOFF, FTOL,
    )
    from models.lammps_script import write_adsorbate_min_script
    from models.ase_neb import run_neb_pipeline
    from models.create_slurm import write_slurm_job, write_chained_slurm_job
    from models.parsers import parse_energy_log

    if masses is None:
        masses = MASSES_7
    if e2t is None:
        e2t = E2T_7
    if elem_str is None:
        elem_str = ELEM_STR_7
    if slurm_opts is None:
        slurm_opts = {**SLURM_DEFAULTS, 'partition': 'sharing', 'time': '01:00:00'}
    if neb_slurm_opts is None:
        neb_slurm_opts = {
            **SLURM_DEFAULTS,
            'partition': 'short',
            'time': '12:00:00',
            'gpu': None,
            'cuda_version': None,
        }

    if z_freeze_cutoff is None:
        # Auto: bottom 1/3 of the relaxed Hop A slab; falls back to the
        # config constant if the structure is not yet available.
        _first_sub1 = (
            os.path.join(hopa_outdir, hopa_jobs[0]['sid'], 'sub1_fs_relaxed.lammps')
            if hopa_jobs else ''
        )
        if _first_sub1 and os.path.exists(_first_sub1):
            from models.structure import compute_z_freeze_cutoff
            z_freeze_cutoff = compute_z_freeze_cutoff(_first_sub1)
        else:
            z_freeze_cutoff = Z_FREEZE_CUTOFF

    G, subsurface_sites = subsurface_graph
    site_lookup = {s['site_id']: s for s in subsurface_sites}

    hopb_dir = Path(outdir) / 'hopb'
    hopb_dir.mkdir(parents=True, exist_ok=True)
    kk = ' '.join(KOKKOS_FLAGS)

    jobs = []

    for ha_job in hopa_jobs:
        sid    = ha_job['sid']
        ss1_id = ha_job['ss1_id']

        # IS = sub1_fs_relaxed from Hop A FS-min (must exist before calling this)
        hopb_is = Path(hopa_outdir) / str(sid) / 'sub1_fs_relaxed.lammps'
        if not hopb_is.exists():
            print(f'[Hop B] {hopb_is} not found — skipping sid={sid}. '
                  f'Run Hop A FS-min on cluster first.')
            continue

        # Find sub2 neighbor in graph
        try:
            ss2_id = find_sub2_neighbor(G, ss1_id, subsurface_sites)
        except ValueError as exc:
            print(f'[Hop B] {exc}  Skipping sid={sid}.')
            continue

        sub2_xyz = site_lookup[ss2_id]['position']

        # E_IS = PE of sub1_fs_relaxed, parsed from Hop A fs_min.log
        hopa_log = Path(hopa_outdir) / str(sid) / 'fs_min.log'
        energy_data = parse_energy_log(str(hopa_log))
        e_is_hopb = (
            energy_data.get('pe_final_eV', float('nan'))
            if energy_data else float('nan')
        )
        if e_is_hopb != e_is_hopb:  # nan check
            print(f'[Hop B] Warning: could not parse E_IS from {hopa_log}; '
                  f'E_IS will be nan for sid={sid}.')

        job_dir = hopb_dir / str(sid)
        job_dir.mkdir(parents=True, exist_ok=True)

        fs_raw       = str(job_dir / 'sub2_fs_raw.lammps')
        fs_relaxed   = str(job_dir / 'sub2_fs_relaxed.lammps')
        fsmin_log    = str(job_dir / 'fs_min.log')
        min_script   = str(job_dir / 'min_fs.lammps')
        barrier_file = str(job_dir / 'neb_barrier.txt')
        path_file    = str(job_dir / 'neb_path.dat')

        # Skip regenerating already-completed work's SLURM commands -- see
        # the matching comment in orchestrate_hopa_neb above.
        fs_already_done  = os.path.exists(fs_relaxed)
        neb_already_done = os.path.exists(barrier_file)

        # 1. FS raw
        build_hopb_fs(
            hopb_is_path=str(hopb_is),
            sub2_xyz=sub2_xyz,
            masses=masses,
            e2t=e2t,
            out_path=fs_raw,
        )

        # 2. LAMMPS FS-min script
        write_adsorbate_min_script(
            slab_ads_input=fs_raw,
            ads_output=fs_relaxed,
            out_path=min_script,
            pair_style=PAIR_STYLE,
            mace_model=MACE_MODEL_LAMMPS,
            pair_suffix=PAIR_SUFFIX,
            elem_str=elem_str,
            z_freeze_cutoff=z_freeze_cutoff,
            ftol=FTOL,
        )

        # 3. ASE NEB script — E_IS from Hop A fs_min.log, E_FS from this job's log
        traj_p1    = str(job_dir / 'hopb_phase1.traj')
        traj_p2    = str(job_dir / 'hopb_phase2.traj')
        neb_script = run_neb_pipeline(
            is_file=str(hopb_is),
            fs_file=fs_relaxed,
            e_is=e_is_hopb,
            mace_model_path=MACE_MODEL_ASE,
            barrier_file=barrier_file,
            path_file=path_file,
            outdir=str(job_dir),
            fs_log_file=fsmin_log,
            job_name='hopb',
            n_images=n_images,
            spring_const=spring_const,
            neb_ftol=neb_ftol,
            z_freeze_cutoff=z_freeze_cutoff,
            device='cpu',
            label_is=f'sub1:{ss1_id}',
            label_fs=f'sub2:{ss2_id}',
            traj_phase1=traj_p1,
            traj_phase2=traj_p2,
            masses=masses,
            e2t=e2t,
        )

        # 4a. GPU SLURM: FS-min (no-op stub if already done -- see the matching
        # comment in orchestrate_hopa_neb above).
        fsmin_sh = str(job_dir / f'slurm_fsmin_{sid}.sh')
        _fsmin_commands = (
            [f'echo "[fsmin skip] {sid}: {fs_relaxed} already exists — skipping FS-min"']
            if fs_already_done else
            [f'{LAMMPS_CMD} {kk} -in {min_script} -log {fsmin_log}']
        )
        write_slurm_job(
            job_name=f'fsmin_hopb_{sid}',
            slurm_config=slurm_opts,
            out_path=fsmin_sh,
            commands=_fsmin_commands,
        )

        # 4b. CPU SLURM: NEB (self-chaining via traj checkpoint; no-op stub if
        # this pair's NEB already converged).
        neb_sh = str(job_dir / f'slurm_neb_{sid}.sh')
        if neb_already_done:
            write_slurm_job(
                job_name=f'neb_hopb_{sid}',
                slurm_config=neb_slurm_opts,
                out_path=neb_sh,
                commands=[f'echo "[neb skip] {sid}: {barrier_file} already exists — skipping NEB"'],
            )
        else:
            _neb_wall = neb_slurm_opts.get('time', '12:00:00')
            _h, _m, _s = (int(x) for x in _neb_wall.split(':'))
            _neb_cutoff_secs = _h * 3600 + _m * 60 + _s - 300
            _neb_cutoff = f'{_neb_cutoff_secs // 3600:02d}:{(_neb_cutoff_secs % 3600) // 60:02d}:{_neb_cutoff_secs % 60:02d}'
            write_chained_slurm_job(
                job_name=f'neb_hopb_{sid}',
                slurm_config=neb_slurm_opts,
                out_path=neb_sh,
                first_commands=[f'python {neb_script}'],
                restart_commands=[f'python {neb_script}'],
                restart_glob=traj_p2,
                cutoff=_neb_cutoff,
                work_dir=str(job_dir),
            )

        if not dry_run:
            from models.create_slurm import submit_with_retry
            if neb_already_done:
                print(f"  Hop B {sid}: already done ({barrier_file}) — skipping submission")
            else:
                if fs_already_done:
                    submit_with_retry(neb_sh)
                else:
                    fsmin_jid = submit_with_retry(fsmin_sh)
                    dep = f'afterok:{fsmin_jid}' if fsmin_jid else None
                    submit_with_retry(neb_sh, dependency=dep)

        jobs.append({
            'sid'         : sid,
            'ss1_id'      : ss1_id,
            'ss2_id'      : ss2_id,
            'hopb_is'     : str(hopb_is),
            'e_is'        : e_is_hopb,
            'sub2_xyz'    : sub2_xyz,
            # Oct-site coordination environments of the sub1 origin and sub2
            # destination — sub2_env is the per-environment key Part 6 uses for
            # the deeper (sub1→sub2) entry/exit rates in the two-layer KMC.
            'sub1_env'    : oct_env_label(site_lookup[ss1_id]) if ss1_id in site_lookup else 'unknown_env',
            'sub2_env'    : oct_env_label(site_lookup[ss2_id]),
            'fs_raw'      : fs_raw,
            'fs_relaxed'  : fs_relaxed,
            'fsmin_script': min_script,
            'neb_script'  : neb_script,
            'fsmin_sh'    : fsmin_sh,
            'neb_sh'      : neb_sh,
            'barrier_file': barrier_file,
            'path_file'   : path_file,
            'job_dir'     : str(job_dir),
        })

    if not jobs:
        return {'jobs': [], 'n_jobs': 0, 'status': 'no_jobs'}

    # job_index.txt
    job_index_path = hopb_dir / 'job_index.txt'
    job_index_path.write_text('\n'.join(str(j['sid']) for j in jobs) + '\n')

    ar = (0, len(jobs) - 1)

    # GPU array: FS-min
    fsmin_array = str(hopb_dir / 'hopb_fsmin_array.sh')
    write_slurm_job(
        job_name='hopb_fsmin_array',
        slurm_config=slurm_opts,
        out_path=fsmin_array,
        array_range=ar,
        concurrent=2,
        commands=[
            f'SID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {job_index_path})',
            f'bash {hopb_dir}/${{SID}}/slurm_fsmin_${{SID}}.sh',
        ],
    )

    # CPU array: NEB
    neb_array = str(hopb_dir / 'hopb_neb_array.sh')
    write_slurm_job(
        job_name='hopb_neb_array',
        slurm_config=neb_slurm_opts,
        out_path=neb_array,
        array_range=ar,
        concurrent=50,
        commands=[
            f'SID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {job_index_path})',
            f'bash {hopb_dir}/${{SID}}/slurm_neb_${{SID}}.sh',
        ],
    )

    jobs_json = str(hopb_dir / 'hopb_jobs.json')
    with open(jobs_json, 'w') as fh:
        json.dump(jobs, fh, indent=2)

    status = 'submitted' if not dry_run else 'generated'
    print(f'[Hop B] {len(jobs)} jobs ({status})')
    print(f'  job_index   : {job_index_path}')
    print(f'  fsmin array : {fsmin_array}')
    print(f'  neb array   : {neb_array}')
    print(f'  jobs json   : {jobs_json}')

    return {
        'jobs'       : jobs,
        'fsmin_array': fsmin_array,
        'neb_array'  : neb_array,
        'jobs_json'  : jobs_json,
        'n_jobs'     : len(jobs),
        'status'     : status,
    }

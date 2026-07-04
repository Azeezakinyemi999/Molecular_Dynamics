"""
models/structure.py
====================
Atomic structure builders for Hastelloy N MD workflows.

Functions
---------
write_lammps_data
    Write a LAMMPS atomic-style data file from Python arrays.
get_lattice_parameter
    Extract a₀ from a minimized LAMMPS data file.
build_alloy_bulk
    Build a random FCC Hastelloy N supercell from a composition dict.
insert_hydrogen
    Place N hydrogen atoms at FCC octahedral interstitial sites.
build_slab
    Construct a surface slab from a minimized bulk structure.
add_adsorbate
    Place a single H or H₂ molecule above a surface site.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atom, Atoms
from ase.build import bulk, make_supercell, surface
from ase.io import read, write as ase_write


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Low-level LAMMPS data file writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_lammps_data(
    symbols: list[str],
    positions: np.ndarray,
    cell_lengths: np.ndarray | list[float],
    masses: dict,
    e2t: dict,
    out_path: str,
    comment: str = '',
) -> str:
    """Write a LAMMPS atomic-style data file from Python arrays.

    This is the single canonical writer used by all other structure-building
    functions in this module.  It writes the minimal header required by
    ``read_data`` in a LAMMPS input script with ``atom_style atomic``.

    Parameters
    ----------
    symbols : list of str
        Element symbol for each atom (length N).
    positions : ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    cell_lengths : array-like, shape (3,)
        Orthogonal box edge lengths [Lx, Ly, Lz] in Å.
        The box origin is always placed at (0, 0, 0).
    masses : dict
        ``{atom_type: (mass_amu, element_symbol)}`` — every type that
        appears in ``e2t`` must have an entry.
    e2t : dict
        ``{element_symbol: atom_type}`` — integer type IDs used in the
        Atoms block.
    out_path : str
        Destination file path.  Parent directories are created if needed.
    comment : str, optional
        First-line comment written at the top of the file.

    Returns
    -------
    str
        Absolute path to the written file (same as ``out_path``).
    """
    Lx, Ly, Lz = cell_lengths
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, 'w') as f:
        f.write(f'# {comment}\n\n' if comment else '\n')
        f.write(f'{len(symbols)} atoms\n')
        f.write(f'{len(masses)} atom types\n\n')
        f.write(f'0.0  {Lx:.10f}  xlo xhi\n')
        f.write(f'0.0  {Ly:.10f}  ylo yhi\n')
        f.write(f'0.0  {Lz:.10f}  zlo zhi\n\n')
        f.write('Masses\n\n')
        for t, (m, el) in sorted(masses.items()):
            f.write(f'{t}  {m}  # {el}\n')
        f.write('\nAtoms # atomic\n\n')
        for i, (sym, p) in enumerate(zip(symbols, positions), 1):
            f.write(f'{i}  {e2t[sym]}  {p[0]:.10f}  {p[1]:.10f}  {p[2]:.10f}\n')

    return os.path.abspath(out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Lattice parameter extraction
# ═══════════════════════════════════════════════════════════════════════════════

def get_lattice_parameter(
    bulk_min_path: str,
    supercell_reps: tuple[int, int, int] = (5, 5, 5),
) -> float:
    """Extract the FCC lattice parameter a₀ from a minimized bulk LAMMPS file.

    Reads the cell lengths from the ASE Atoms object and divides by the
    supercell repetitions.  Assumes an orthogonal, cubic supercell (all
    three axes give the same a₀ up to numerical noise; the x-axis value
    is returned).

    Parameters
    ----------
    bulk_min_path : str
        Path to the minimized LAMMPS data file (``lammps-data`` atomic
        style), typically produced by NB02.
    supercell_reps : tuple of int, optional
        (nx, ny, nz) repetitions used when the supercell was built.
        Default ``(5, 5, 5)``.

    Returns
    -------
    float
        Lattice parameter a₀ in Å (= Lx / nx).

    Raises
    ------
    FileNotFoundError
        If ``bulk_min_path`` does not exist.
    """
    if not os.path.exists(bulk_min_path):
        raise FileNotFoundError(f'{bulk_min_path} not found.')
    atoms = read(bulk_min_path, format='lammps-data', style='atomic')
    atoms.wrap()
    Lx = atoms.cell.lengths()[0]
    a0 = Lx / supercell_reps[0]
    print(f'[a0] {a0:.6f} Å  ({os.path.basename(bulk_min_path)})')
    return float(a0)


def get_lattice_parameter_from_dump(
    dump_file: str,
    n_last: int = 50,
    supercell_reps: tuple[int, int, int] = (5, 5, 5),
) -> float:
    """Extract a₀ by averaging the last n_last Lx values from an NPT box-dim dump.

    The dump file is written by ``fix ave/time`` in the NPT script with columns:
    Step  Lx  Ly  Lz.  Averaging over the tail of the production run gives a
    thermally-averaged lattice parameter rather than a single potentially
    fluctuating snapshot.

    Parameters
    ----------
    dump_file : str
        Path to the LAMMPS fix ave/time output (``npt_boxdims_{T}K.dat``).
    n_last : int
        Number of trailing rows to average. Default 50 = last 5000 steps at
        dump_every=100.
    supercell_reps : tuple of int
        (nx, ny, nz) repetitions used when the supercell was built.
        Default (5, 5, 5).

    Returns
    -------
    float
        Lattice parameter a₀ in Å.
    """
    if not os.path.exists(dump_file):
        raise FileNotFoundError(f'{dump_file} not found.')
    lx_vals = []
    with open(dump_file) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.split()
            if len(cols) >= 2:
                try:
                    lx_vals.append(float(cols[1]))
                except ValueError:
                    continue
    if not lx_vals:
        raise ValueError(f'No data rows found in {dump_file}')
    tail = lx_vals[-n_last:]
    lx_mean = sum(tail) / len(tail)
    a0 = lx_mean / supercell_reps[0]
    print(f'[a0] {a0:.6f} Å  (mean of last {len(tail)} frames, {os.path.basename(dump_file)})')
    return float(a0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Random alloy bulk builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_alloy_bulk(
    composition: dict[str, float],
    a0: float,
    supercell_reps: tuple[int, int, int],
    masses: dict,
    e2t: dict,
    out_path: str,
    seed: int = 42,
) -> str:
    """Build a random FCC Hastelloy N bulk supercell and write a LAMMPS file.

    Creates an FCC unit cell using Ni as the template (same crystal
    structure), expands it to the requested supercell size, then randomly
    assigns element symbols according to ``composition``.

    Parameters
    ----------
    composition : dict
        ``{element_symbol: mole_fraction}`` — fractions must sum to 1.
        Example: ``{'Ni': 0.70, 'Mo': 0.16, 'Cr': 0.07, 'Fe': 0.04,
        'Al': 0.01, 'B': 0.01, 'C': 0.01}``.
    a0 : float
        FCC lattice parameter in Å.
    supercell_reps : tuple of int
        (nx, ny, nz) supercell repetitions.
    masses : dict
        ``{atom_type: (mass_amu, element_symbol)}`` passed to
        :func:`write_lammps_data`.
    e2t : dict
        ``{element_symbol: atom_type}`` passed to :func:`write_lammps_data`.
    out_path : str
        Destination LAMMPS data file path.
    seed : int, optional
        NumPy random seed for reproducible composition assignment.
        Default ``42``.

    Returns
    -------
    str
        Absolute path to the written LAMMPS data file.

    Raises
    ------
    ValueError
        If composition fractions do not sum to approximately 1.
    """
    total = sum(composition.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f'Composition fractions sum to {total:.4f}, expected 1.0. '
            f'Check your composition dict.'
        )

    nx, ny, nz = supercell_reps

    # ── Build FCC template ────────────────────────────────────────────────────
    # cubic=True gives the conventional 4-atom orthogonal cell so that a
    # (nx,ny,nz) supercell contains 4·nx·ny·nz atoms with an orthogonal box.
    unit_cell = bulk('Ni', crystalstructure='fcc', a=a0, cubic=True)
    P = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
    super_cell = make_supercell(unit_cell, P)
    n_atoms = len(super_cell)

    # ── Randomly assign element symbols ──────────────────────────────────────
    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    counts = np.round(fractions * n_atoms).astype(int)
    # Correct rounding errors so counts sum exactly to n_atoms
    diff = n_atoms - counts.sum()
    counts[np.argmax(fractions)] += diff

    symbols_arr = np.concatenate([
        np.full(c, el) for el, c in zip(elements, counts)
    ])
    rng = np.random.default_rng(seed)
    rng.shuffle(symbols_arr)

    positions = super_cell.get_positions()
    cell_lengths = super_cell.cell.lengths()

    print(f'Built FCC supercell: {n_atoms} atoms  ({nx}×{ny}×{nz})')
    print(f'Cell : {cell_lengths[0]:.4f} × {cell_lengths[1]:.4f} × {cell_lengths[2]:.4f} Å')
    for el, c in zip(elements, counts):
        print(f'  {el:>2s}  {c:4d}  ({c/n_atoms*100:.1f} at.%)')

    return write_lammps_data(
        symbols=list(symbols_arr),
        positions=positions,
        cell_lengths=cell_lengths,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment=f'Hastelloy N FCC {nx}x{ny}x{nz} random alloy  a0={a0:.6f} A',
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — H insertion (refactored to use write_lammps_data)
# ═══════════════════════════════════════════════════════════════════════════════

def insert_hydrogen(
    bulk_min_path: str,
    n_h: int,
    masses: dict,
    e2t: dict,
    out_dir: str,
    a0: float,
    supercell_reps: tuple[int, int, int] = (5, 5, 5),
    d_min_metal: float = 1.5,
    d_min_hh: float = 2.5,
    seed: int = 42,
) -> tuple[str, np.ndarray, float]:
    """Insert H atoms at FCC octahedral interstitial sites in a bulk supercell.

    Octahedral sites in FCC lie at the body-centre of each unit cell,
    i.e. fractional coordinate (½, ½, ½).  All candidate sites are
    enumerated from the lattice parameter, randomly shuffled, and
    greedily accepted subject to two distance thresholds.

    Parameters
    ----------
    bulk_min_path : str
        Path to the energy-minimized LAMMPS data file (``lammps-data``
        atomic style) produced by NB02.
    n_h : int
        Number of H atoms to insert.
    masses : dict
        ``{atom_type: (mass_amu, element_symbol)}`` — all element types
        must be present so the LAMMPS header is complete.
    e2t : dict
        ``{element_symbol: atom_type}`` used to assign integer type IDs.
    out_dir : str
        Directory in which the output LAMMPS data file is written.
    a0 : float
        Lattice parameter in Å, used to compute octahedral site coordinates.
    supercell_reps : tuple of int, optional
        Repetitions along (x, y, z).  Default ``(5, 5, 5)``.
    d_min_metal : float, optional
        Minimum H-to-metal distance in Å.  Default ``1.5``.
    d_min_hh : float, optional
        Minimum H-to-H distance in Å.  Default ``2.5``.
    seed : int, optional
        Random seed.  Default ``42``.

    Returns
    -------
    out_path : str
        Absolute path to the written LAMMPS data file.
    h_positions : ndarray, shape (n_h, 3)
        Cartesian coordinates (Å) of the accepted H sites.
    min_hm_dist : float
        Smallest H-to-metal distance (Å) — sanity check value.

    Raises
    ------
    FileNotFoundError
        If ``bulk_min_path`` does not exist.
    RuntimeError
        If fewer than ``n_h`` valid octahedral sites can be found.
    """
    if not os.path.exists(bulk_min_path):
        raise FileNotFoundError(f'{bulk_min_path} not found.')

    # ── Load minimized bulk ───────────────────────────────────────────────────
    bulk_atoms = read(bulk_min_path, format='lammps-data', style='atomic')
    bulk_atoms.wrap()
    bulk_pos   = bulk_atoms.get_positions()
    bulk_syms  = np.array(bulk_atoms.get_chemical_symbols())
    L          = bulk_atoms.cell.lengths()

    nx, ny, nz = supercell_reps
    print(f'Bulk supercell : {len(bulk_atoms)} atoms')
    print(f'Cell           : {L[0]:.4f} x {L[1]:.4f} x {L[2]:.4f} Å')
    print(f'a₀             : {a0:.6f} Å')

    # ── Enumerate all octahedral sites ────────────────────────────────────────
    candidates = np.array([
        [(i + 0.5) * a0, (j + 0.5) * a0, (k + 0.5) * a0]
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
    ])
    print(f'Octahedral candidates : {len(candidates)}')

    # ── Shuffle and greedy accept ─────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    accepted: list[np.ndarray] = []

    for site in candidates:
        if len(accepted) == n_h:
            break
        if np.linalg.norm(bulk_pos - site, axis=1).min() < d_min_metal:
            continue
        if accepted and np.linalg.norm(np.array(accepted) - site, axis=1).min() < d_min_hh:
            continue
        accepted.append(site)

    if len(accepted) < n_h:
        raise RuntimeError(
            f'Only {len(accepted)}/{n_h} valid sites found. '
            f'Try reducing n_h, d_min_metal ({d_min_metal} Å) or d_min_hh ({d_min_hh} Å).'
        )

    h_positions = np.array(accepted)

    # ── Nearest metal–H distance ──────────────────────────────────────────────
    all_hm      = np.linalg.norm(
        bulk_pos[:, None, :] - h_positions[None, :, :], axis=2)
    min_hm_dist = float(all_hm.min())

    # ── Write via shared writer ───────────────────────────────────────────────
    all_syms = list(bulk_syms) + ['H'] * n_h
    all_pos  = np.vstack([bulk_pos, h_positions])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'bulk_{n_h}H_initial.lammps')

    write_lammps_data(
        symbols=all_syms,
        positions=all_pos,
        cell_lengths=L,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment=f'Hastelloy N {nx}x{ny}x{nz} + {n_h}H at octahedral sites',
    )

    return out_path, h_positions, min_hm_dist


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Surface slab builder
# ═══════════════════════════════════════════════════════════════════════════════

# Crystal structure map for pure metals: element → (ase_structure, template_element)
_CRYSTAL_STRUCT_MAP: dict[str, tuple[str, str]] = {
    'Ni': ('fcc', 'Ni'), 'Al': ('fcc', 'Al'), 'Cu': ('fcc', 'Cu'),
    'Fe': ('bcc', 'Fe'), 'Cr': ('bcc', 'Cr'), 'Mo': ('bcc', 'Mo'),
    'W':  ('bcc', 'W'),  'V':  ('bcc', 'V'),
}


def build_slab(
    bulk_min_path: str,
    miller: tuple[int, int, int],
    layers: int,
    vacuum: float,
    masses: dict,
    e2t: dict,
    out_path: str,
    supercell_reps: tuple[int, int, int] = (5, 5, 5),
    lateral_repeat: tuple[int, int] = (1, 1),
    seed: int = 42,
    metal_type: str = 'alloy',
) -> tuple[str, float]:
    """Construct a surface slab from a minimized bulk structure.

    Routes slab construction based on ``metal_type``:

    * ``'alloy'``  — Ni-FCC geometry template; element symbols randomly
      shuffled from bulk composition fractions.  Suitable for Ni-based
      alloys such as Hastelloy N.
    * ``'pure'``   — correct FCC or BCC geometry from ``_CRYSTAL_STRUCT_MAP``;
      all sites set to the single non-H element.  No shuffle.
    * ``'oxide'``  — primitive cell extracted via spglib from the minimized
      bulk supercell; stoichiometry preserved exactly.  No shuffle.

    Parameters
    ----------
    bulk_min_path : str
        Path to the minimized bulk LAMMPS data file.
    miller : tuple of int
        Miller indices, e.g. ``(1, 1, 1)``.
    layers : int
        Number of atomic layers in the slab.
    vacuum : float
        Vacuum thickness in Å added above the top surface.
    masses : dict
        ``{atom_type: (mass_amu, element_symbol)}``.
    e2t : dict
        ``{element_symbol: atom_type}``.
    out_path : str
        Destination LAMMPS data file path.
    supercell_reps : tuple of int, optional
        Repetitions used by ``get_lattice_parameter`` (alloy/pure only).
        Default ``(5, 5, 5)``.
    lateral_repeat : tuple of int, optional
        ``(p, q)`` lateral tiling.  Default ``(1, 1)``.
    seed : int, optional
        Random seed for composition shuffle (alloy path only).  Default ``42``.
    metal_type : str, optional
        One of ``'alloy'``, ``'pure'``, ``'oxide'``.  Default ``'alloy'``.

    Returns
    -------
    out_path : str
        Absolute path to the written LAMMPS data file.
    a0 : float
        Lattice parameter used for slab construction (Å).
    """
    from collections import Counter

    bulk_atoms = read(bulk_min_path, format='lammps-data', style='atomic')
    bulk_atoms.wrap()
    bulk_syms = bulk_atoms.get_chemical_symbols()
    hkl_str   = ''.join(map(str, miller))

    if metal_type == 'oxide':
        # ── Oxide: extract primitive cell via spglib, preserve stoichiometry ─
        import spglib
        cell_data = (
            bulk_atoms.get_cell(),
            bulk_atoms.get_scaled_positions(),
            bulk_atoms.get_atomic_numbers(),
        )
        prim_data = spglib.find_primitive(cell_data, symprec=1e-2)
        if prim_data is None:
            raise RuntimeError(
                f'spglib could not find a primitive cell for {bulk_min_path}. '
                'Try increasing symprec or check the structure symmetry.'
            )
        lattice, scaled_pos, numbers = prim_data
        primitive_atoms = Atoms(
            numbers=numbers, scaled_positions=scaled_pos, cell=lattice, pbc=True
        )
        unit_slab = surface(primitive_atoms, miller, layers, vacuum=vacuum)
        p, q = lateral_repeat
        if p != 1 or q != 1:
            unit_slab = make_supercell(unit_slab, [[p, 0, 0], [0, q, 0], [0, 0, 1]])
        symbols_arr = np.array(unit_slab.get_chemical_symbols())
        a0 = float(primitive_atoms.cell.lengths()[0])

    elif metal_type == 'pure':
        # ── Pure metal: look up correct FCC/BCC template, no shuffle ─────────
        non_h = [s for s in set(bulk_syms) if s != 'H']
        if len(non_h) != 1:
            raise ValueError(
                f"metal_type='pure' expects exactly one non-H element; "
                f"found {non_h} in {bulk_min_path}"
            )
        elem = non_h[0]
        if elem not in _CRYSTAL_STRUCT_MAP:
            raise KeyError(
                f"Element '{elem}' not in _CRYSTAL_STRUCT_MAP. "
                f"Add it to the map or use metal_type='alloy'."
            )
        struct_type, template_elem = _CRYSTAL_STRUCT_MAP[elem]
        a0 = get_lattice_parameter(bulk_min_path, supercell_reps)
        unit_slab = surface(bulk(template_elem, struct_type, a=a0), miller, layers, vacuum=vacuum)
        p, q = lateral_repeat
        if p != 1 or q != 1:
            unit_slab = make_supercell(unit_slab, [[p, 0, 0], [0, q, 0], [0, 0, 1]])
        symbols_arr = np.array([elem] * len(unit_slab))

    else:
        # ── Alloy (default): Ni-FCC template + random composition shuffle ─────
        a0 = get_lattice_parameter(bulk_min_path, supercell_reps)
        unit_slab = surface(bulk('Ni', 'fcc', a=a0), miller, layers, vacuum=vacuum)
        p, q = lateral_repeat
        if p != 1 or q != 1:
            unit_slab = make_supercell(unit_slab, [[p, 0, 0], [0, q, 0], [0, 0, 1]])
        n_slab       = len(unit_slab)
        bulk_counts  = Counter(bulk_syms)
        n_bulk       = len(bulk_syms)
        elements     = [el for el in bulk_counts if el != 'H']
        fractions    = np.array([bulk_counts[el] / n_bulk for el in elements])
        counts       = np.round(fractions * n_slab).astype(int)
        diff         = n_slab - counts.sum()
        counts[np.argmax(fractions)] += diff
        symbols_arr  = np.concatenate(
            [np.full(c, el) for el, c in zip(elements, counts)]
        )
        rng = np.random.default_rng(seed)
        rng.shuffle(symbols_arr)

    positions    = unit_slab.get_positions()
    cell_lengths = unit_slab.cell.lengths()
    n_slab       = len(unit_slab)

    print(f'Slab ({hkl_str}): {n_slab} atoms, {layers} layers, {vacuum:.1f} Å vacuum  [{metal_type}]')
    print(f'Cell : {cell_lengths[0]:.4f} × {cell_lengths[1]:.4f} × {cell_lengths[2]:.4f} Å')

    out_path = write_lammps_data(
        symbols=list(symbols_arr),
        positions=positions,
        cell_lengths=cell_lengths,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment=f'{metal_type} ({hkl_str}) slab  {layers} layers  {vacuum:.1f} A vacuum',
    )
    return out_path, a0


def compute_z_freeze_cutoff(slab_path: str, fraction: float = 1.0 / 3.0) -> float:
    """
    Z threshold that freezes the bottom ``fraction`` of the slab thickness.

    Geometry-based replacement for a hardcoded cutoff: works for any layer
    count and for oxide slabs whose atomic-plane count differs from the
    ``layers`` argument passed to ``build_slab`` (one primitive repeat unit
    can contain several planes).

    Parameters
    ----------
    slab_path : str
        Path to the slab LAMMPS data file.
    fraction : float
        Fraction of the slab thickness (from the bottom) to freeze.
        Default 1/3 — for a 12-layer metal slab this freezes the bottom
        4 layers, matching the historical 22.115 Å cutoff.

    Notes
    -----
    For N equally spaced layers this freezes exactly round(N/3) of them
    (8 → 3, 12 → 4, 13 → 4). The returned cutoff is snapped to the midpoint
    of the interlayer gap it falls in, so it can never coincide with an
    atomic plane — a raw thickness/3 cutoff lands exactly ON a plane
    whenever N ≡ 1 (mod 3), and float noise could then split that layer.

    Returns
    -------
    float
        Z coordinate (Å) below which atoms should be frozen.
    """
    atoms = read(slab_path, format='lammps-data', atom_style='atomic')
    zs = np.sort(atoms.get_positions()[:, 2])
    raw = float(zs[0] + (zs[-1] - zs[0]) * fraction)

    # Atoms within `pad` below the raw cutoff belong to a plane the cutoff
    # grazes; count them as free (nearest-whole rule) and snap the cutoff to
    # the midpoint of the gap between the frozen and free sides.
    pad = 0.25  # Å — below thermal displacement, above float noise
    frozen = zs[zs < raw - pad]
    free   = zs[zs >= raw - pad]
    if frozen.size and free.size:
        return float((frozen[-1] + free[0]) / 2.0)
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Adsorbate placement
# ═══════════════════════════════════════════════════════════════════════════════

def add_adsorbate(
    slab_path: str,
    site_position: np.ndarray | list[float],
    species: str,
    masses: dict,
    e2t: dict,
    out_path: str,
    height: float | None = None,
    h2_bond: float | None = None,
    h2_orientation: str = 'parallel',
) -> str:
    """Place a single H atom or H₂ molecule above a surface site.

    The adsorbate is placed at ``[site_x, site_y, z_top + height]`` where
    ``z_top`` is the maximum z-coordinate of the slab surface atoms.

    For H₂, two orientations are supported:
    - ``'parallel'`` (default): both H atoms at the same height above the
      surface, offset ±h2_bond/2 along x.  Matches the standard initial
      geometry used in adsorption energy surveys (NB05b).
    - ``'vertical'``: H atoms placed symmetrically about z_ads along z,
      i.e. stacked perpendicular to the surface.

    Parameters
    ----------
    slab_path : str
        Path to the surface slab LAMMPS data file.
    site_position : array-like, shape (2,) or (3,)
        (x, y) or (x, y, z) of the adsorption site.  Only x and y are
        used; z is determined from the slab surface + ``height``.
    species : str
        ``'H'`` for a single hydrogen atom or ``'H2'`` for a molecule.
    masses : dict
        ``{atom_type: (mass_amu, element_symbol)}``.
    e2t : dict
        ``{element_symbol: atom_type}``.
    out_path : str
        Destination LAMMPS data file path.
    height : float, optional
        Placement height above the top surface atom in Å.
        Defaults to ``config.H2_HEIGHT`` (2.5 Å).
    h2_bond : float, optional
        H–H bond length in Å for H₂ placement.
        Defaults to ``config.H2_BOND`` (0.741 Å).
    h2_orientation : {'parallel', 'vertical'}, optional
        Orientation of the H₂ molecule relative to the surface.
        Default ``'parallel'``.

    Returns
    -------
    str
        Absolute path to the written LAMMPS data file.

    Raises
    ------
    FileNotFoundError
        If ``slab_path`` does not exist.
    ValueError
        If ``species`` is not ``'H'`` or ``'H2'``, or ``h2_orientation``
        is not ``'parallel'`` or ``'vertical'``.
    """
    from models.config import H2_HEIGHT, H2_BOND

    if height is None:
        height = H2_HEIGHT
    if h2_bond is None:
        h2_bond = H2_BOND

    if species not in ('H', 'H2'):
        raise ValueError(f"species must be 'H' or 'H2', got '{species}'.")
    if h2_orientation not in ('parallel', 'vertical'):
        raise ValueError(
            f"h2_orientation must be 'parallel' or 'vertical', got '{h2_orientation}'."
        )

    if not os.path.exists(slab_path):
        raise FileNotFoundError(f'{slab_path} not found.')

    slab = read(slab_path, format='lammps-data', style='atomic')
    slab.wrap()
    slab_pos  = slab.get_positions()
    slab_syms = list(slab.get_chemical_symbols())
    L         = slab.cell.lengths()

    z_top  = slab_pos[:, 2].max()
    site   = np.asarray(site_position, dtype=float)
    x_site = float(site[0])
    y_site = float(site[1])
    z_ads  = z_top + height

    if species == 'H':
        new_syms = ['H']
        new_pos  = np.array([[x_site, y_site, z_ads]])
    elif h2_orientation == 'parallel':
        # Both H atoms at the same z, offset ±h2_bond/2 along x — parallel to surface
        dxy = h2_bond / 2.0
        new_syms = ['H', 'H']
        new_pos  = np.array([
            [x_site - dxy, y_site, z_ads],
            [x_site + dxy, y_site, z_ads],
        ])
    else:  # vertical — symmetric about z_ads along z
        dz = h2_bond / 2.0
        new_syms = ['H', 'H']
        new_pos  = np.array([
            [x_site, y_site, z_ads - dz],
            [x_site, y_site, z_ads + dz],
        ])

    all_syms = slab_syms + new_syms
    all_pos  = np.vstack([slab_pos, new_pos])

    print(f'Added {species} above site ({x_site:.3f}, {y_site:.3f}) at z={z_ads:.3f} Å')
    print(f'  z_top slab = {z_top:.3f} Å   height = {height:.3f} Å')

    return write_lammps_data(
        symbols=all_syms,
        positions=all_pos,
        cell_lengths=L,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment=f'{species} adsorbed at ({x_site:.3f}, {y_site:.3f})',
    )


def build_fs_raw_structure(
    is_lammps: str,
    fs_xy1: tuple[float, float],
    fs_xy2: tuple[float, float],
    masses: dict,
    e2t: dict,
    out_path: str,
    h_height: float = 1.5,
) -> str:
    """Build the NEB final-state (FS) raw structure for H2 dissociation.

    Takes the IS slab (slab + intact H2), strips both H atoms, then places
    two independent H atoms at the relaxed XY coordinates of the two chosen
    FS sites at z = max_metal_z + h_height.  XY coordinates are PBC-wrapped
    to lie within the simulation cell before writing.

    This matches the NB06b2 Cell D1 construction exactly:
        metal_pos  = IS positions where sym != 'H'
        z_fs       = metal_pos[:, 2].max() + h_height   (1.5 Å default)
        h1 / h2    = (x % cell_x, y % cell_y, z_fs)

    The returned structure is a *raw* FS that must be LAMMPS-minimized before
    use as a NEB endpoint.

    Parameters
    ----------
    is_lammps : str
        Path to the IS LAMMPS data file (slab + H2 molecule, i.e. neb_initial.lammps).
    fs_xy1 : (float, float)
        (x, y) of first FS H atom — from NB06b relaxed h_atom structure.
    fs_xy2 : (float, float)
        (x, y) of second FS H atom.
    masses : dict
        ``{atom_type: (mass_amu, element_symbol)}``.
    e2t : dict
        ``{element_symbol: atom_type}``.
    out_path : str
        Destination path for the raw FS LAMMPS data file.
    h_height : float
        Height above top metal atom in Å. Default 1.5 Å (matches NB06b2).

    Returns
    -------
    str
        Absolute path to the written LAMMPS data file.
    """
    if not os.path.exists(is_lammps):
        raise FileNotFoundError(f'{is_lammps} not found.')

    slab     = read(is_lammps, format='lammps-data', style='atomic')
    slab.wrap()
    pos      = slab.get_positions()
    syms     = np.array(slab.get_chemical_symbols())
    cell     = slab.cell.lengths()

    metal_mask = syms != 'H'
    metal_pos  = pos[metal_mask]
    metal_syms = list(syms[metal_mask])

    z_fs = float(metal_pos[:, 2].max()) + h_height

    x1, y1 = float(fs_xy1[0]) % cell[0], float(fs_xy1[1]) % cell[1]
    x2, y2 = float(fs_xy2[0]) % cell[0], float(fs_xy2[1]) % cell[1]

    h1_pos = np.array([[x1, y1, z_fs]])
    h2_pos = np.array([[x2, y2, z_fs]])

    all_syms = metal_syms + ['H', 'H']
    all_pos  = np.vstack([metal_pos, h1_pos, h2_pos])

    print(f'[build_fs_raw_structure] z_fs = {z_fs:.3f} Å  (metal max + {h_height} Å)')
    print(f'  H1: ({x1:.3f}, {y1:.3f}, {z_fs:.3f})')
    print(f'  H2: ({x2:.3f}, {y2:.3f}, {z_fs:.3f})')

    return write_lammps_data(
        symbols=all_syms,
        positions=all_pos,
        cell_lengths=cell,
        masses=masses,
        e2t=e2t,
        out_path=out_path,
        comment='NEB FS raw: metal slab + 2 H at relaxed site coords',
    )

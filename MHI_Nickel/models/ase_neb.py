"""
models/ase_neb.py
=================
ASE Climbing-Image NEB utilities for H₂ dissociation barrier calculations.

All NEB notebooks (NB06, NB06a, NB06b, NB06b2) use the same two-solver hybrid:
  1. LAMMPS (CG minimisation) → relax IS and FS structures
  2. ASE CINEB + MACECalculator (.model, device-agnostic) → path optimisation

This module centralises the repeated inline code across those notebooks.

Typical usage
-------------
    from models.ase_neb import (
        make_mace_calc, make_frozen_calc,
        build_neb_images, run_cineb, extract_mep,
        write_neb_results, write_ase_neb_script,
    )
    from models.config import (
        MACE_MODEL_ASE, N_REPLICAS, SPRING_CONST, NEB_FTOL, Z_FREEZE_CUTOFF,
    )

    images = build_neb_images(is_file, fs_file, n_images=N_REPLICAS)
    neb, converged, fmax = run_cineb(
        images,
        calc_fn=lambda: make_frozen_calc(MACE_MODEL_ASE, Z_FREEZE_CUTOFF),
        spring_const=SPRING_CONST,
        neb_ftol=NEB_FTOL,
    )
    mep = extract_mep(images, E_IS=e_is, E_FS=e_fs)
    write_neb_results(mep, fmax, converged, barrier_file, path_file)
"""

from __future__ import annotations

import os
import sys
import textwrap

import numpy as np


# ---------------------------------------------------------------------------
# Section 1 — Calculator helpers
# ---------------------------------------------------------------------------

def make_mace_calc(model_path: str, device: str = 'cpu',
                   dtype: str = 'float64', head: str = 'omat_pbe'):
    """Return a fresh ``MACECalculator`` instance.

    Parameters
    ----------
    model_path : str
        Path to the ``.model`` file (ASE-compatible MACE variant).
    device : str
        PyTorch device string — ``'cpu'`` or ``'cpu'``.
        Defaults to ``'cpu'``; pass ``'cpu'`` for CPU-only partitions.
    dtype : str
        Floating-point precision.  ``'float64'`` (default) is required for
        NEB accuracy.

    Returns
    -------
    MACECalculator
    """
    from mace.calculators import MACECalculator
    return MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype=dtype,
        head=head,
    )


def make_frozen_calc(model_path: str, z_freeze_cutoff: float,
                     device: str = 'cpu', dtype: str = 'float64',
                     head: str = 'omat_pbe'):
    """Return a ``MACECalculator`` subclass that zeroes forces on frozen atoms.

    Atoms with z-coordinate < ``z_freeze_cutoff`` receive zero force,
    mimicking the frozen-layer constraint used in LAMMPS minimisations.

    Parameters
    ----------
    model_path : str
        Path to the ``.model`` file.
    z_freeze_cutoff : float
        z-coordinate threshold in Å.  Atoms below this value are frozen.
    device : str
        PyTorch device string (``'cpu'`` or ``'cpu'``).
    dtype : str
        Floating-point precision.

    Returns
    -------
    MACEFrozenCalc instance (MACECalculator subclass)
    """
    from mace.calculators import MACECalculator

    _cutoff = z_freeze_cutoff

    class MACEFrozenCalc(MACECalculator):
        def get_forces(self, atoms=None):
            forces = super().get_forces(atoms).copy()
            a = self.atoms if atoms is None else atoms
            forces[a.get_positions()[:, 2] < _cutoff] = 0.0
            return forces

    return MACEFrozenCalc(
        model_paths=model_path,
        device=device,
        default_dtype=dtype,
        head=head,
    )


# ---------------------------------------------------------------------------
# Section 2 — NEB image chain
# ---------------------------------------------------------------------------

def build_neb_images(
    is_file: str,
    fs_file: str,
    n_images: int = 18,
    e_is: float | None = None,
    e_fs: float | None = None,
    interpolation: str = 'idpp',
) -> list:
    """Build and interpolate a NEB image chain between IS and FS.

    Endpoint energies are pinned via ``SinglePointCalculator`` so that
    MACE does not re-evaluate them during NEB (energies come from the
    prior LAMMPS minimisations).

    Parameters
    ----------
    is_file : str
        Path to the LAMMPS data file for the initial state.
    fs_file : str
        Path to the LAMMPS data file for the final state.
    n_images : int
        Number of *intermediate* images (excluding IS and FS).
    e_is : float, optional
        Potential energy of IS in eV (from LAMMPS log).  If provided the
        endpoint is pinned; otherwise it will be evaluated by MACE.
    e_fs : float, optional
        Potential energy of FS in eV (from LAMMPS log).
    interpolation : str
        Interpolation method passed to ``NEB.interpolate()``.
        ``'idpp'`` (default) avoids atom clashes; ``'linear'`` is faster.

    Returns
    -------
    list[Atoms]
        Full image list: ``[IS, img_1, …, img_n, FS]`` (length n_images + 2).
    """
    from ase.io import read
    from ase.mep import NEB
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.geometry import find_mic
    import warnings

    # NOTE: the ASE LAMMPS-data reader below always defaults atoms.pbc to
    # (True, True, True) -- these files carry no periodicity metadata --
    # even though these are slabs with a vacuum gap, not periodic in z.
    # Investigated whether this could let the MACE
    # calculator's periodic neighbour search spuriously "see" an atom's own
    # image across the vacuum during force evaluation: with this pipeline's
    # real production model (r_max=6.0 A, 2 interaction layers) and the
    # real vacuum geometry (~31 A total gap -- VACUUM=15.0 A applied on
    # both sides of the slab, not just one), no atom pair can ever be
    # within r_max of its periodic image across z. Confirmed harmless;
    # left as-is rather than fixed.
    is_raw = read(is_file, format='lammps-data', atom_style='atomic')
    is_raw.wrap()
    fs_raw = read(fs_file, format='lammps-data', atom_style='atomic')

    if not np.allclose(is_raw.cell.array, fs_raw.cell.array):
        raise RuntimeError(
            f'IS and FS cells differ ({is_file} vs {fs_file}) -- find_mic() '
            f'assumes a shared cell; check both LAMMPS outputs.'
        )

    # Align FS to IS's frame via minimum-image displacement per atom, instead
    # of wrapping independently -- an atom near a cell boundary in one
    # structure but not the other otherwise shows a spurious ~cell-length
    # "jump" that NEB.interpolate() takes as a literal path to traverse
    # (confirmed on a real pair: metal atoms with ~cell-length naive
    # displacement but ~0.01 Å true displacement). pbc=(True, True, False)
    # matches every slab in this pipeline -- periodic in-plane (x, y), fixed
    # in z (surface normal, topped with vacuum). This is passed to find_mic
    # only; it deliberately does NOT change is_raw/fs_raw's own .pbc, so
    # calculator behaviour (periodic neighbour lists) elsewhere in the NEB
    # run is unaffected by this alignment step.
    _slab_pbc = (True, True, False)
    _diff = fs_raw.get_positions() - is_raw.get_positions()
    # Gate the ambiguity check by the same pbc mask used below -- z is not
    # periodic (vacuum gap), so any raw z-displacement, however large, is
    # the true motion with zero wrap ambiguity. Checking all 3 axes
    # uniformly would false-positive on a real, unambiguous desorption-like
    # z-motion that's well over half the (typically huge, vacuum-inflated)
    # z cell length but was never a candidate for wrapping in the first place.
    _half_cell = 0.5 * is_raw.cell.lengths()
    if np.any((np.abs(_diff) > _half_cell) & np.array(_slab_pbc)):
        warnings.warn(
            f"Some atom's raw IS->FS displacement in {fs_file} exceeds half "
            f'the cell length along a periodic axis -- find_mic() cannot '
            f'distinguish a genuine large hop from wrapping around the box '
            f'in that case; inspect manually.'
        )
    _mic_vec, _ = find_mic(_diff, is_raw.cell, pbc=_slab_pbc)
    fs_raw.set_positions(is_raw.get_positions() + _mic_vec)

    # Pin endpoint energies if supplied
    def _pin(atoms, energy):
        atoms = atoms.copy()
        if energy is not None:
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results = {
                'energy': energy,
                'forces': np.zeros((len(atoms), 3)),
            }
        return atoms

    is_end = _pin(is_raw, e_is)
    fs_end = _pin(fs_raw, e_fs)

    images = [is_end] + [is_raw.copy() for _ in range(n_images)] + [fs_end]

    neb = NEB(images, climb=False, k=1.0, method='aseneb')
    neb.interpolate(method=interpolation)

    # Report H-H distances along interpolated path
    for i, img in enumerate(images):
        lbl = 'IS' if i == 0 else ('FS' if i == len(images) - 1 else f'img_{i}')
        sym = np.array(img.get_chemical_symbols())
        pos = img.get_positions()
        h_pos = pos[sym == 'H']
        if len(h_pos) == 2:
            dist = np.linalg.norm(h_pos[0] - h_pos[1])
            print(f'  {lbl:6s}: H-H = {dist:.4f} Å')
    sys.stdout.flush()

    return images


# ---------------------------------------------------------------------------
# Section 3 — Two-phase CINEB runner
# ---------------------------------------------------------------------------

def run_cineb(
    images: list,
    calc_fn,
    spring_const: float = 1.0,
    neb_ftol: float = 2.5,
    phase1_steps: int = 5000,
    phase2_steps: int = 10000,
    logfile_phase1: str = 'neb_phase1.log',
    logfile_phase2: str = 'neb_phase2.log',
) -> tuple:
    """Run a two-phase climbing-image NEB optimisation with FIRE.

    Phase 1 relaxes images onto the MEP (regular NEB, fmax = 3× ``neb_ftol``).
    Phase 2 enables the climbing image to locate the true saddle point
    (CINEB, fmax = ``neb_ftol``).

    Parameters
    ----------
    images : list[Atoms]
        Full image list from :func:`build_neb_images`.
    calc_fn : callable
        Zero-argument callable returning a fresh ``Calculator`` for each
        intermediate image.  Example::

            lambda: make_frozen_calc(MACE_MODEL_ASE, Z_FREEZE_CUTOFF)

    spring_const : float
        NEB elastic-band spring constant in eV/Å².
    neb_ftol : float
        Force convergence threshold for CINEB (phase 2) in eV/Å.
    phase1_steps : int
        Maximum FIRE steps for phase 1.
    phase2_steps : int
        Maximum FIRE steps for phase 2 (CINEB).
    logfile_phase1 : str
        FIRE log file path for phase 1.
    logfile_phase2 : str
        FIRE log file path for phase 2.

    Returns
    -------
    neb : NEB
        Converged (or best-effort) NEB object.
    converged : bool
        ``True`` if phase 2 reached ``fmax ≤ neb_ftol``.
    fmax_final : float
        Maximum force across all images at end of phase 2 in eV/Å.
    """
    from ase.mep import NEB
    from ase.optimize import FIRE

    # Attach a fresh calculator to each intermediate image
    for img in images[1:-1]:
        img.calc = calc_fn()

    neb = NEB(images, climb=False, k=spring_const, method='aseneb')

    # Phase 1 — regular NEB
    phase1_fmax = neb_ftol * 1.5
    print(f'\nPhase 1: regular NEB  ({phase1_steps} steps, fmax={phase1_fmax:.3f} eV/Å)')
    sys.stdout.flush()
    opt1 = FIRE(neb, logfile=logfile_phase1, dt=0.05)
    opt1.run(fmax=phase1_fmax, steps=phase1_steps)
    print(f'Phase 1 done: {opt1.nsteps} steps')
    sys.stdout.flush()

    # Reset momenta — prevents phase-1 velocity carryover into CINEB
    for img in images[1:-1]:
        img.set_momenta(np.zeros_like(img.get_momenta()))

    # Phase 2 — CINEB
    neb.climb = True
    print(f'\nPhase 2: CINEB  ({phase2_steps} steps, fmax={neb_ftol:.3f} eV/Å)')
    sys.stdout.flush()
    opt2 = FIRE(neb, logfile=logfile_phase2, dt=0.02)
    converged = False
    try:
        converged = opt2.run(fmax=neb_ftol, steps=phase2_steps)
    except Exception as exc:
        print(f'WARNING: phase 2 raised: {exc}')
        sys.stdout.flush()
    print(f'Phase 2 done: {opt2.nsteps} steps')
    sys.stdout.flush()

    # Final fmax: use neb.get_forces() (the perpendicular/spring-projected
    # force the optimizer actually converges on), not each image's raw
    # calculator force -- the raw force also contains the along-path
    # tangential component NEB deliberately never drives to zero, and is
    # therefore always >= the true NEB fmax (the two components are
    # orthogonal).
    try:
        _f = neb.get_forces()
        fmax_final = float(np.sqrt((_f ** 2).sum(axis=1).max()))
    except Exception:
        fmax_final = float('nan')

    return neb, bool(converged), fmax_final


# ---------------------------------------------------------------------------
# Section 4 — MEP extraction
# ---------------------------------------------------------------------------

def extract_mep(
    images: list,
    E_IS: float | None = None,
    E_FS: float | None = None,
) -> dict:
    """Compute MEP energetics from a converged NEB image chain.

    Parameters
    ----------
    images : list[Atoms]
        Full image list (including IS and FS endpoints).
    E_IS : float, optional
        Pinned IS energy in eV.  If ``None``, read from ``images[0]``.
    E_FS : float, optional
        Pinned FS energy in eV.  If ``None``, read from ``images[-1]``.

    Returns
    -------
    dict with keys
        ``energies``  – list of per-image energies (eV), ``None`` if unavailable
        ``frac``      – reaction coordinate array (0 → 1)
        ``E_IS``      – initial state energy (eV)
        ``E_FS``      – final state energy (eV)
        ``E_abs``     – absorption / forward barrier = E_TS − E_IS (eV)
        ``E_des``     – desorption / reverse barrier = E_TS − E_FS (eV)
        ``delta_E``   – reaction energy = E_FS − E_IS (eV)
        ``ts_index``  – image index of the transition state
    """
    energies = []
    for img in images:
        try:
            energies.append(img.get_potential_energy())
        except Exception:
            energies.append(None)

    valid = [e for e in energies if e is not None]
    e_is = E_IS if E_IS is not None else energies[0]
    e_fs = E_FS if E_FS is not None else energies[-1]

    e_ts = max(valid)
    ts_index = next(i for i, e in enumerate(energies) if e == e_ts)

    n = len(energies)
    frac = np.array([i / (n - 1) for i in range(n)])

    return {
        'energies': energies,
        'frac':     frac,
        'E_IS':     e_is,
        'E_FS':     e_fs,
        'E_abs':    e_ts - e_is,
        'E_des':    e_ts - e_fs,
        'delta_E':  e_fs - e_is,
        'ts_index': ts_index,
    }


# ---------------------------------------------------------------------------
# Section 5 — Write result files
# ---------------------------------------------------------------------------

def write_neb_results(
    mep: dict,
    fmax_final: float,
    converged: bool,
    barrier_file: str,
    path_file: str,
    label_is: str = 'IS',
    label_fs: str = 'FS',
) -> None:
    """Write ``neb_barrier.txt`` and ``neb_path.dat`` from MEP dict.

    Output format is compatible with :func:`models.parsers.parse_barrier_file`
    and :func:`models.parsers.parse_neb_path`.

    Parameters
    ----------
    mep : dict
        Output of :func:`extract_mep`.
    fmax_final : float
        Final maximum force across intermediate images (eV/Å).
    converged : bool
        Whether phase 2 reached the force tolerance.
    barrier_file : str
        Path to write the human-readable barrier summary.
    path_file : str
        Path to write the 3-column MEP data file.
    label_is : str
        Descriptive label for IS printed in barrier_file.
    label_fs : str
        Descriptive label for FS printed in barrier_file.
    """
    os.makedirs(os.path.dirname(barrier_file) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(path_file) or '.', exist_ok=True)

    energies = mep['energies']
    e_is     = mep['E_IS']

    with open(barrier_file, 'w', encoding='utf-8') as f:
        f.write(f'IS          : {label_is}\n')
        f.write(f'FS          : {label_fs}\n')
        f.write(f'N images    : {len(energies)} ({len(energies)-2} intermediate + IS + FS)\n')
        f.write(f'fmax_final  : {fmax_final:.4f} eV/A\n')
        f.write(f'Converged   : {converged}\n\n')
        f.write(f'E_IS    = {mep["E_IS"]:.6f} eV\n')
        f.write(f'E_FS    = {mep["E_FS"]:.6f} eV\n')
        f.write(f'E_abs   = {mep["E_abs"]:.4f} eV\n')
        f.write(f'E_des   = {mep["E_des"]:.4f} eV\n')
        f.write(f'delta_E = {mep["delta_E"]:.4f} eV\n\n')
        f.write('Image energies:\n')
        n = len(energies)
        for i, e in enumerate(energies):
            lbl = 'IS' if i == 0 else ('FS' if i == n - 1 else f'img_{i}')
            if e is not None:
                f.write(f'  {lbl:6s}: {e:.6f} eV  ({e - e_is:+.4f} eV)\n')

    with open(path_file, 'w', encoding='utf-8') as f:
        f.write('# image_frac  E_eV  dE_from_IS_eV\n')
        n = len(energies)
        for i, e in enumerate(energies):
            if e is not None:
                f.write(f'{i/(n-1):.4f}  {e:.6f}  {e - e_is:+.4f}\n')

    print(f'Wrote: {barrier_file}')
    print(f'Wrote: {path_file}')


# ---------------------------------------------------------------------------
# Section 6 — Write standalone ASE NEB job script
# ---------------------------------------------------------------------------

def write_ase_neb_script(
    is_file: str,
    fs_file: str,
    e_is: float,
    mace_model_path: str,
    barrier_file: str,
    path_file: str,
    logfile_phase1: str,
    logfile_phase2: str,
    e_fs: float | None = None,
    fs_log_file: str | None = None,
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 2.5,
    phase1_steps: int = 5000,
    phase2_steps: int = 10000,
    z_freeze_cutoff: float = 22.115,
    device: str = 'cpu',
    dtype: str = 'float32',
    parallel: bool = True,
    label_is: str = 'IS',
    label_fs: str = 'FS',
    out_path: str = 'run_neb.py',
    images_dir: str | None = None,
    traj_phase1: str | None = None,
    traj_phase2: str | None = None,
    masses: dict | None = None,
    e2t: dict | None = None,
) -> str:
    """Write a self-contained Python script that runs ASE CINEB on a cluster node.

    The generated script is submitted via SLURM and does not depend on the
    notebook kernel.  All parameters are injected at write time.

    Parameters
    ----------
    is_file : str
        Path (on the cluster) to the LAMMPS data file for IS.
    fs_file : str
        Path (on the cluster) to the LAMMPS data file for FS.
    e_is : float
        IS potential energy in eV (from LAMMPS minimisation log).
    mace_model_path : str
        Path (on the cluster) to the ``.model`` MACE file.
    barrier_file : str
        Output path for ``neb_barrier.txt``.
    path_file : str
        Output path for ``neb_path.dat``.
    logfile_phase1 : str
        FIRE log path for phase 1.
    logfile_phase2 : str
        FIRE log path for phase 2.
    e_fs : float, optional
        FS potential energy in eV (from LAMMPS minimisation log).
        Mutually exclusive with ``fs_log_file``.
    fs_log_file : str, optional
        Path to the FS LAMMPS minimisation log (``fs_min.log``).  When
        provided, the generated script parses ``E_FS`` at runtime from
        ``pe_final_eV : <value>`` lines written by
        :func:`models.lammps_script.write_adsorbate_min_script`.
        Use this when E_FS is not known at script-generation time.
        Mutually exclusive with ``e_fs``.
    n_images : int
        Number of intermediate NEB images.
    spring_const : float
        Elastic-band spring constant in eV/Å².
    neb_ftol : float
        CINEB force convergence tolerance in eV/Å.
    phase1_steps : int
        Phase 1 FIRE step limit.
    phase2_steps : int
        Phase 2 FIRE step limit.
    z_freeze_cutoff : float
        Frozen-layer z threshold in Å.
    device : str
        PyTorch device (``'cpu'`` or ``'cpu'``).
    dtype : str
        MACE ``default_dtype`` — ``'float32'`` (default) or ``'float64'``.
    parallel : bool
        Pass ``parallel=True`` (default) to ASE's ``NEB`` so intermediate
        images are force-evaluated on separate Python threads instead of
        sequentially. No MPI is required — this uses ASE's built-in
        threading fallback when ``ase.parallel.world.size == 1``.
    label_is : str
        Descriptive label for IS (written to barrier file header).
    label_fs : str
        Descriptive label for FS (written to barrier file header).
    out_path : str
        Path where the generated ``.py`` script is written.
    masses : dict, optional
        ``{atom_type: (mass_amu, element_symbol)}`` used to write each
        per-image structure with :func:`models.structure.write_lammps_data`
        (the same canonical writer used everywhere else in this codebase),
        so the files carry a real ``Masses`` section and round-trip
        correctly when read later (e.g. by the vibration scripts). Default
        ``None`` resolves to ``MASSES_7``.
    e2t : dict, optional
        ``{element_symbol: atom_type}`` paired with ``masses``. Default
        ``None`` resolves to ``E2T_7``.

    Returns
    -------
    str
        Path to the written script (``out_path``).
    """
    from models.config import MASSES_7, E2T_7
    if masses is None:
        masses = MASSES_7
    if e2t is None:
        e2t = E2T_7

    if e_fs is None and fs_log_file is None:
        raise ValueError(
            'write_ase_neb_script: provide either e_fs (float) or '
            'fs_log_file (path to fs_min.log).'
        )
    if images_dir is None:
        images_dir = os.path.dirname(os.path.abspath(out_path))

    # Project root, so the generated script (run standalone on a cluster
    # node) can import models.structure.write_lammps_data.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if fs_log_file is not None:
        # Accumulate last valid float — LAMMPS echoes the `print` command itself
        # (containing `${pe_final}`) on the line immediately before the actual
        # output value. Returning on the first match hits the echo line and raises
        # ValueError. Overwriting `_val` on each successful parse picks the last
        # valid line instead.
        efs_block = (
            'def _parse_pe_final(log_path):\n'
            '    _val = None\n'
            '    with open(log_path) as _f:\n'
            '        for _line in _f:\n'
            "            if 'pe_final_eV' in _line and ':' in _line:\n"
            '                try:\n'
            "                    _val = float(_line.split(':')[1].strip())\n"
            '                except ValueError:\n'
            '                    pass\n'
            '    if _val is None:\n'
            '        raise ValueError(f"pe_final_eV not found in {log_path}")\n'
            '    return _val\n'
            f'\nE_FS = _parse_pe_final("{fs_log_file}")  # eV — from fs_min.log at runtime'
        )
    else:
        efs_block = f'E_FS            = {e_fs}               # eV — from LAMMPS FS minimisation'

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ASE CINEB — generated by models.ase_neb.write_ase_neb_script
        # IS: {label_is}
        # FS: {label_fs}
        # Interpolation: IDPP
        # Two-phase: regular NEB (phase 1) → CINEB (phase 2)

        import os, sys
        import warnings
        import numpy as np

        # Strip cpu stubs from LD_LIBRARY_PATH — causes segfault on some nodes
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            p for p in _ld.split(":") if "stubs" not in p)

        _parent = {_project_root!r}
        if _parent not in sys.path:
            sys.path.insert(0, _parent)

        from pathlib import Path as _Path
        from ase.io import read, Trajectory as _Trajectory
        from ase.mep import NEB
        from ase.optimize import FIRE
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.geometry import find_mic
        from mace.calculators import MACECalculator
        from models.structure import write_lammps_data

        import torch
        torch.set_num_threads(1)  # NEB(parallel=True) below spawns one Python
        # thread per intermediate image; without this each thread's MACE
        # forward pass would also try to use multiple intra-op threads,
        # oversubscribing the allocated CPUs.

        # ── Parameters injected from notebook ─────────────────────────────
        MACE_MODEL      = "{mace_model_path}"
        NEB_IS_FILE     = "{is_file}"
        FS_RELAXED_DATA = "{fs_file}"
        Z_FREEZE_CUTOFF = {z_freeze_cutoff}    # Å — frozen layer threshold
        E_IS            = {e_is}               # eV — from LAMMPS IS minimisation
        MASSES          = {masses!r}
        E2T             = {e2t!r}
        %%EFS_BLOCK%%
        N_IMAGES        = {n_images}           # intermediate images
        SPRING_CONST    = {spring_const}       # eV/Å²
        NEB_FMAX        = {neb_ftol}           # eV/Å — CINEB convergence
        N1_FMAX         = NEB_FMAX * 1.5       # eV/Å — phase 1 (looser)
        N1_STEPS        = {phase1_steps}
        NEB_STEPS       = {phase2_steps}
        DEVICE          = "{device}"
        DTYPE           = "{dtype}"
        PARALLEL        = {parallel!r}
        BARRIER_FILE    = "{barrier_file}"
        PATH_FILE       = "{path_file}"
        LOG_PHASE1      = "{logfile_phase1}"
        LOG_PHASE2      = "{logfile_phase2}"
        LABEL_IS        = "{label_is}"
        LABEL_FS        = "{label_fs}"
        IMAGES_DIR      = "{images_dir}"
        TRAJ_PHASE1     = {repr(traj_phase1)}
        TRAJ_PHASE2     = {repr(traj_phase2)}

        # ── Calculator with frozen layers ─────────────────────────────────
        class MACEFrozenCalc(MACECalculator):
            def get_forces(self, atoms=None):
                forces = super().get_forces(atoms).copy()
                a = self.atoms if atoms is None else atoms
                forces[a.get_positions()[:, 2] < Z_FREEZE_CUTOFF] = 0.0
                return forces

        def make_calc():
            return MACEFrozenCalc(
                model_paths=MACE_MODEL, device=DEVICE, default_dtype=DTYPE, head="omat_pbe")

        # ── Load structures and pin endpoint energies ─────────────────────
        # NOTE: atoms.pbc defaults to (True, True, True) from this read()
        # despite the vacuum gap in z -- investigated and confirmed harmless
        # for the actual MACE calculator's periodic neighbour search given
        # this pipeline's real r_max/vacuum geometry; see the matching note
        # in build_neb_images() in models/ase_neb.py for the full reasoning.
        is_raw = read(NEB_IS_FILE, format="lammps-data", atom_style="atomic")
        is_raw.wrap()
        fs_raw = read(FS_RELAXED_DATA, format="lammps-data", atom_style="atomic")

        if not np.allclose(is_raw.cell.array, fs_raw.cell.array):
            raise RuntimeError(
                f"IS and FS cells differ ({{NEB_IS_FILE}} vs {{FS_RELAXED_DATA}}) -- "
                f"find_mic() assumes a shared cell; check both LAMMPS outputs."
            )

        # Align FS to IS's frame via minimum-image displacement per atom,
        # instead of wrapping independently -- an atom near a cell boundary
        # in one structure but not the other otherwise shows a spurious
        # ~cell-length "jump" that NEB.interpolate() takes as a literal path
        # to traverse (confirmed on a real pair: metal atoms with
        # ~cell-length naive displacement but ~0.01 Å true displacement).
        # pbc=(True, True, False) matches every slab in this pipeline --
        # periodic in-plane (x, y), fixed in z (surface normal, topped with
        # vacuum). This is passed to find_mic only; it deliberately does NOT
        # change is_raw/fs_raw's own .pbc, so calculator behaviour (periodic
        # neighbour lists) elsewhere in the NEB run is unaffected.
        _slab_pbc = (True, True, False)
        _diff = fs_raw.get_positions() - is_raw.get_positions()
        # Gate the ambiguity check by the same pbc mask used below -- z is
        # not periodic (vacuum gap), so any raw z-displacement, however
        # large, is the true motion with zero wrap ambiguity. Checking all
        # 3 axes uniformly would false-positive on a real, unambiguous
        # desorption-like z-motion that's well over half the (typically
        # huge, vacuum-inflated) z cell length but was never a candidate
        # for wrapping in the first place.
        _half_cell = 0.5 * is_raw.cell.lengths()
        if np.any((np.abs(_diff) > _half_cell) & np.array(_slab_pbc)):
            warnings.warn(
                f"Some atom's raw IS->FS displacement in {{FS_RELAXED_DATA}} "
                f"exceeds half the cell length along a periodic axis -- "
                f"find_mic() cannot distinguish a genuine large hop from "
                f"wrapping around the box in that case; inspect manually."
            )
        _mic_vec, _ = find_mic(_diff, is_raw.cell, pbc=_slab_pbc)
        fs_raw.set_positions(is_raw.get_positions() + _mic_vec)

        def pin(atoms, energy):
            atoms = atoms.copy()
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results = {{"energy": energy,
                                   "forces": np.zeros((len(atoms), 3))}}
            return atoms

        is_end = pin(is_raw, E_IS)
        fs_end = pin(fs_raw, E_FS)

        # ── Restart detection / IDPP interpolation ───────────────────────
        def _load_last_band(traj_path):
            _t = _Trajectory(traj_path)
            _n = len(_t)
            # A timeout can land mid-write of one step's N_IMAGES+2-frame
            # batch, leaving a partial trailing batch -- this is a normal,
            # expected consequence of the chain-resubmit mechanism, not an
            # error. Truncate to the last COMPLETE batch and resume from
            # there rather than failing the whole job over a few incomplete
            # trailing frames.
            _n_complete = (_n // (N_IMAGES + 2)) * (N_IMAGES + 2)
            if _n_complete == 0:
                raise RuntimeError(
                    f"{{traj_path}} has only {{_n}} frames, fewer than one "
                    f"full band (N_IMAGES+2={{N_IMAGES + 2}}). Nothing usable "
                    f"to resume from -- delete this file, its .log file, and "
                    f"any .extxyz file derived from it, then resubmit for a "
                    f"clean restart."
                )
            if _n_complete != _n:
                print(f"WARNING: {{traj_path}} has {{_n}} frames, not a clean "
                      f"multiple of N_IMAGES+2 ({{N_IMAGES + 2}}) -- likely an "
                      f"interrupted write mid-batch. Using the last "
                      f"{{_n_complete}} frames, discarding {{_n - _n_complete}} "
                      f"incomplete trailing frame(s).")
                sys.stdout.flush()
            _last = max(0, _n_complete - (N_IMAGES + 2))
            return [_t[_last + _i] for _i in range(min(N_IMAGES + 2, _n_complete - _last))]

        def _neb_fmax(neb_obj):
            # neb.get_forces() returns the perpendicular/spring-projected
            # NEB force -- what the optimizer actually converges on -- not
            # each image's raw calculator force (which also contains the
            # along-path tangential component NEB deliberately never drives
            # to zero, and is therefore always >= the true NEB fmax since
            # the two components are orthogonal).
            try:
                _f = neb_obj.get_forces()
                return float(np.sqrt((_f**2).sum(axis=1).max()))
            except Exception:
                return float("nan")

        def _steps_done(traj_path):
            # Each completed FIRE step writes N_IMAGES+2 frames; one extra
            # pre-optimization batch is written on the very first leg only.
            _n = len(_Trajectory(traj_path))
            return max(0, _n // (N_IMAGES + 2) - 1)

        _p1_fmax_final = float("nan")
        _p1_converged = None

        if TRAJ_PHASE2 and _Path(TRAJ_PHASE2).exists():
            print("Restart: Phase 2 checkpoint found — resuming CINEB")
            sys.stdout.flush()
            images = _load_last_band(TRAJ_PHASE2)
            for img in images[1:-1]:
                img.calc = make_calc()
            neb = NEB(images, climb=True, k=SPRING_CONST, method="aseneb", parallel=PARALLEL)
            _restart_phase = 2
        elif TRAJ_PHASE1 and _Path(TRAJ_PHASE1).exists():
            images = _load_last_band(TRAJ_PHASE1)
            for img in images[1:-1]:
                img.calc = make_calc()
            # climb=False here matches Phase 1's own optimizer -- evaluating
            # the loaded band's fmax against N1_FMAX must use the same
            # (non-climbing) force definition Phase 1 itself converges on.
            neb = NEB(images, climb=False, k=SPRING_CONST, method="aseneb", parallel=PARALLEL)
            _loaded_fmax = _neb_fmax(neb)
            _p1_steps_done = _steps_done(TRAJ_PHASE1)
            _p1_remaining = N1_STEPS - _p1_steps_done
            if _loaded_fmax > N1_FMAX and _p1_remaining > 0:
                print(f"Restart: Phase 1 checkpoint found — fmax={{_loaded_fmax:.4f}} > "
                      f"{{N1_FMAX:.4f}} ({{_p1_steps_done}}/{{N1_STEPS}} steps done) — resuming Phase 1")
                _restart_phase = 1
            else:
                _reason = "converged" if _loaded_fmax <= N1_FMAX else "step budget exhausted"
                print(f"Restart: Phase 1 checkpoint found — {{_reason}} "
                      f"(fmax={{_loaded_fmax:.4f}}, {{_p1_steps_done}}/{{N1_STEPS}} steps) — starting Phase 2")
                neb.climb = True
                _p1_fmax_final = _loaded_fmax
                _p1_converged = _loaded_fmax <= N1_FMAX
                _restart_phase = 2
            sys.stdout.flush()
        else:
            images = [is_end] + [is_raw.copy() for _ in range(N_IMAGES)] + [fs_end]
            neb = NEB(images, climb=False, k=SPRING_CONST, method="aseneb", parallel=PARALLEL)
            neb.interpolate(method="idpp")
            print("IDPP interpolation done")
            sys.stdout.flush()
            for img in images[1:-1]:
                img.calc = make_calc()
            _restart_phase = 1
            _p1_remaining = N1_STEPS

        # ── Phase 1: regular NEB (skipped once converged / budget exhausted) ──
        if _restart_phase == 1:
            print(f"\\nPhase 1: regular NEB  ({{_p1_remaining}} steps remaining of "
                  f"{{N1_STEPS}} total, fmax={{N1_FMAX:.3f}} eV/Å)")
            sys.stdout.flush()
            _traj1_kw = {{"trajectory": TRAJ_PHASE1, "append_trajectory": True}} if TRAJ_PHASE1 else {{}}
            opt1 = FIRE(neb, logfile=LOG_PHASE1, dt=0.05, **_traj1_kw)
            opt1.run(fmax=N1_FMAX, steps=_p1_remaining)
            print(f"Phase 1 done: {{opt1.nsteps}} steps (this leg)")
            sys.stdout.flush()

            if TRAJ_PHASE1 and os.path.exists(TRAJ_PHASE1):
                from ase.io import read as _r1, write as _w1
                _lmp1 = TRAJ_PHASE1.replace(".traj", ".extxyz")
                _w1(_lmp1, _r1(TRAJ_PHASE1, index=":"), format="extxyz")
                print(f"Wrote OVITO trajectory: {{_lmp1}}")
                sys.stdout.flush()

            _p1_fmax_final = _neb_fmax(neb)
            _p1_converged = _p1_fmax_final <= N1_FMAX

            for img in images[1:-1]:
                img.set_momenta(np.zeros_like(img.get_momenta()))

        # ── Phase 2: CINEB ────────────────────────────────────────────────
        neb.climb = True
        _p2_done = _steps_done(TRAJ_PHASE2) if TRAJ_PHASE2 and _Path(TRAJ_PHASE2).exists() else 0
        _p2_remaining = NEB_STEPS - _p2_done
        print(f"\\nPhase 2: CINEB  ({{_p2_remaining}} steps remaining of "
              f"{{NEB_STEPS}} total, fmax={{NEB_FMAX:.3f}} eV/Å)")
        sys.stdout.flush()
        _traj2_kw = {{"trajectory": TRAJ_PHASE2, "append_trajectory": True}} if TRAJ_PHASE2 else {{}}
        opt2 = FIRE(neb, logfile=LOG_PHASE2, dt=0.02, **_traj2_kw)
        converged = False
        try:
            converged = opt2.run(fmax=NEB_FMAX, steps=_p2_remaining)
        except Exception as exc:
            print(f"WARNING: phase 2 raised: {{exc}}")
            sys.stdout.flush()
        print(f"Phase 2 done: {{opt2.nsteps}} steps (this leg)")
        sys.stdout.flush()

        if TRAJ_PHASE2 and os.path.exists(TRAJ_PHASE2):
            from ase.io import read as _r2, write as _w2
            _lmp2 = TRAJ_PHASE2.replace(".traj", ".extxyz")
            _w2(_lmp2, _r2(TRAJ_PHASE2, index=":"), format="extxyz")
            print(f"Wrote OVITO trajectory: {{_lmp2}}")
            sys.stdout.flush()

        # ── Extract MEP results ───────────────────────────────────────────
        energies = []
        for img in images:
            try:    energies.append(img.get_potential_energy())
            except: energies.append(None)

        valid  = [e for e in energies if e is not None]
        e_ts   = max(valid)
        E_abs  = e_ts  - E_IS
        E_des  = e_ts  - E_FS
        dE     = E_FS  - E_IS

        fmax_final = _neb_fmax(neb)

        print(f"\\nResults")
        print(f"  E_IS    : {{E_IS:.6f}} eV")
        print(f"  E_FS    : {{E_FS:.6f}} eV")
        print(f"  E_abs   : {{E_abs:.4f}} eV")
        print(f"  E_des   : {{E_des:.4f}} eV")
        print(f"  delta_E : {{dE:.4f}} eV")
        print(f"  fmax    : {{fmax_final:.4f}} eV/Å")
        print(f"  Converged: {{converged}}")
        sys.stdout.flush()

        # ── Write output files ────────────────────────────────────────────
        os.makedirs(os.path.dirname(BARRIER_FILE) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(PATH_FILE) or ".", exist_ok=True)

        n = len(energies)
        with open(BARRIER_FILE, "w", encoding="utf-8") as f:
            f.write(f"IS          : {{LABEL_IS}}\\n")
            f.write(f"FS          : {{LABEL_FS}}\\n")
            f.write(f"N images    : {{n}} ({{n-2}} intermediate + IS + FS)\\n")
            f.write(f"fmax_final  : {{fmax_final:.4f}} eV/A\\n")
            f.write(f"Converged   : {{converged}}\\n")
            f.write(f"phase1_fmax_final : {{_p1_fmax_final:.4f}} eV/A\\n")
            f.write(f"phase1_converged  : {{_p1_converged}}\\n\\n")
            f.write(f"E_IS    = {{E_IS:.6f}} eV\\n")
            f.write(f"E_FS    = {{E_FS:.6f}} eV\\n")
            f.write(f"E_abs   = {{E_abs:.4f}} eV\\n")
            f.write(f"E_des   = {{E_des:.4f}} eV\\n")
            f.write(f"delta_E = {{dE:.4f}} eV\\n\\n")
            f.write("Image energies:\\n")
            for i, e in enumerate(energies):
                lbl = "IS" if i == 0 else ("FS" if i == n-1 else f"img_{{i}}")
                if e is not None:
                    f.write(f"  {{lbl:6s}}: {{e:.6f}} eV  ({{e-E_IS:+.4f}} eV)\\n")

        with open(PATH_FILE, "w", encoding="utf-8") as f:
            f.write("# image_frac  E_eV  dE_from_IS_eV\\n")
            for i, e in enumerate(energies):
                if e is not None:
                    f.write(f"{{i/(n-1):.4f}}  {{e:.6f}}  {{e-E_IS:+.4f}}\\n")

        print(f"Wrote: {{BARRIER_FILE}}")
        print(f"Wrote: {{PATH_FILE}}")

        # ── Write per-image LAMMPS data files ─────────────────────────────
        # write_lammps_data (this project's canonical writer, used
        # everywhere else) always includes a real Masses section. A bare
        # ase.io.write with no masses does not, and a later ase.io.read of
        # such a file, with no explicit type-to-element override, then
        # falls back to treating the raw atom type id as the atomic
        # number -- with this project's E2T_7 mapping (Ni=7, H=8) that
        # silently collides with real elements (N=7, O=8) instead of
        # erroring, and vib_run.py found "0 H atoms" on a structure that
        # actually has exactly one.
        os.makedirs(IMAGES_DIR, exist_ok=True)
        _n = len(images)
        for _i, _img in enumerate(images):
            _lbl  = "IS" if _i == 0 else ("FS" if _i == _n - 1 else f"img_{{_i:02d}}")
            _path = os.path.join(IMAGES_DIR, f"image_{{_i:02d}}_{{_lbl}}.lammps")
            write_lammps_data(
                symbols=_img.get_chemical_symbols(),
                positions=_img.get_positions(),
                cell_lengths=_img.cell.diagonal(),
                masses=MASSES,
                e2t=E2T,
                out_path=_path,
                comment=f"NEB image {{_i}} ({{_lbl}})",
            )
        print(f"Wrote {{_n}} image structures → {{IMAGES_DIR}}/")
    """).replace('%%EFS_BLOCK%%', efs_block)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(script)
    os.chmod(out_path, 0o755)
    print(f'Written: {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Section 7 — Orchestrator
# ---------------------------------------------------------------------------

def run_neb_pipeline(
    is_file: str,
    fs_file: str,
    e_is: float,
    mace_model_path: str,
    barrier_file: str,
    path_file: str,
    outdir: str,
    e_fs: float | None = None,
    fs_log_file: str | None = None,
    job_name: str = 'neb',
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 2.5,
    phase1_steps: int = 5000,
    phase2_steps: int = 10000,
    z_freeze_cutoff: float = 22.115,
    device: str = 'cpu',
    dtype: str = 'float32',
    parallel: bool = True,
    label_is: str = 'IS',
    label_fs: str = 'FS',
    images_dir: str | None = None,
    traj_phase1: str | None = None,
    traj_phase2: str | None = None,
    masses: dict | None = None,
    e2t: dict | None = None,
) -> str:
    """Write a complete, ready-to-submit ASE CINEB job script.

    Single entry point — calls all other functions in order:

    1. Creates ``outdir`` and derives log file paths from ``job_name``.
    2. Calls :func:`write_ase_neb_script` to generate
       ``<outdir>/run_<job_name>.py`` with all parameters injected.

    The generated script handles everything on the cluster node:
    IDPP interpolation → Phase 1 (regular NEB) → Phase 2 (CINEB) →
    writes ``neb_barrier.txt`` and ``neb_path.dat``.

    Parameters
    ----------
    is_file : str
        Path (on cluster) to the LAMMPS data file for the initial state.
    fs_file : str
        Path (on cluster) to the LAMMPS data file for the final state.
    e_is : float
        IS potential energy in eV (from LAMMPS minimisation log).
    mace_model_path : str
        Path (on cluster) to the ``.model`` MACE file.
    barrier_file : str
        Output path the script will write ``neb_barrier.txt`` to.
    path_file : str
        Output path the script will write ``neb_path.dat`` to.
    outdir : str
        Directory where the script and log files will be written.
    e_fs : float, optional
        FS potential energy in eV. Mutually exclusive with ``fs_log_file``.
    fs_log_file : str, optional
        Path to the FS minimisation log (``fs_min.log``).  Passed through
        to :func:`write_ase_neb_script` so E_FS is parsed at cluster runtime.
        Use this when E_FS is not known at script-generation time.
    job_name : str
        Base name for the script and log files (default ``'neb'``).
    n_images : int
        Number of intermediate NEB images (default 18).
    spring_const : float
        Elastic-band spring constant in eV/Å² (default 2.0).
    neb_ftol : float
        CINEB force convergence tolerance in eV/Å (default 2.5).
    phase1_steps : int
        Phase 1 FIRE step limit (default 5000).
    phase2_steps : int
        Phase 2 FIRE step limit (default 10000).
    z_freeze_cutoff : float
        Frozen-layer z threshold in Å (default 22.115).
    device : str
        PyTorch device — ``'cpu'`` (default) or ``'cpu'`` for CPU partitions.
    dtype : str
        MACE ``default_dtype`` — ``'float32'`` (default) or ``'float64'``.
    parallel : bool
        Thread-parallel NEB image evaluation (default ``True``); see
        :func:`write_ase_neb_script`.
    label_is : str
        Human-readable label for IS written to the barrier file header.
    label_fs : str
        Human-readable label for FS written to the barrier file header.
    masses : dict, optional
        ``{atom_type: (mass_amu, element_symbol)}`` passed through to
        :func:`write_ase_neb_script` for writing per-image structures with
        :func:`models.structure.write_lammps_data`. Default ``None``
        resolves to ``MASSES_7``.
    e2t : dict, optional
        ``{element_symbol: atom_type}`` paired with ``masses``. Default
        ``None`` resolves to ``E2T_7``.

    Returns
    -------
    str
        Path to the generated ``.py`` script, ready for SLURM submission.

    Examples
    --------
    >>> from models.ase_neb import run_neb_pipeline
    >>> from models.config import (
    ...     MACE_MODEL_ASE, N_REPLICAS, SPRING_CONST, NEB_FTOL, Z_FREEZE_CUTOFF)
    >>> script = run_neb_pipeline(
    ...     is_file='structures/neb/neb_initial.lammps',
    ...     fs_file='structures/neb/neb_final.lammps',
    ...     e_is=-1234.56, e_fs=-1234.12,
    ...     mace_model_path=MACE_MODEL_ASE,
    ...     barrier_file='results/neb/neb_barrier.txt',
    ...     path_file='results/neb/neb_path.dat',
    ...     outdir='lammps_scripts/neb',
    ...     job_name='neb06',
    ...     n_images=N_REPLICAS, spring_const=SPRING_CONST,
    ...     neb_ftol=NEB_FTOL, z_freeze_cutoff=Z_FREEZE_CUTOFF,
    ...     device='cpu',
    ... )
    >>> # Then submit: submit_slurm_job(write_slurm_job(..., script_body=f'python {script}'))
    """
    os.makedirs(outdir, exist_ok=True)
    if images_dir is None:
        images_dir = os.path.join(outdir, 'images')

    return write_ase_neb_script(
        is_file=is_file,
        fs_file=fs_file,
        e_is=e_is,
        mace_model_path=mace_model_path,
        barrier_file=barrier_file,
        path_file=path_file,
        logfile_phase1=os.path.join(outdir, f'{job_name}_phase1.log'),
        logfile_phase2=os.path.join(outdir, f'{job_name}_phase2.log'),
        e_fs=e_fs,
        fs_log_file=fs_log_file,
        n_images=n_images,
        spring_const=spring_const,
        neb_ftol=neb_ftol,
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        z_freeze_cutoff=z_freeze_cutoff,
        device=device,
        dtype=dtype,
        parallel=parallel,
        label_is=label_is,
        label_fs=label_fs,
        out_path=os.path.join(outdir, f'run_{job_name}.py'),
        images_dir=images_dir,
        traj_phase1=traj_phase1,
        traj_phase2=traj_phase2,
        masses=masses,
        e2t=e2t,
    )

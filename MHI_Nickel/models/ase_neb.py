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
                   dtype: str = 'float64'):
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
    )


def make_frozen_calc(model_path: str, z_freeze_cutoff: float,
                     device: str = 'cpu', dtype: str = 'float64'):
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

    is_raw = read(is_file, format='lammps-data', atom_style='atomic')
    fs_raw = read(fs_file, format='lammps-data', atom_style='atomic')

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
    neb_ftol: float = 0.05,
    phase1_steps: int = 5000,
    phase2_steps: int = 10000,
    logfile_phase1: str = 'neb_phase1.log',
    logfile_phase2: str = 'neb_phase2.log',
) -> tuple:
    """Run a two-phase climbing-image NEB optimisation with MDMin.

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
        Maximum MDMin steps for phase 1.
    phase2_steps : int
        Maximum MDMin steps for phase 2 (CINEB).
    logfile_phase1 : str
        MDMin log file path for phase 1.
    logfile_phase2 : str
        MDMin log file path for phase 2.

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
    from ase.optimize import MDMin

    # Attach a fresh calculator to each intermediate image
    for img in images[1:-1]:
        img.calc = calc_fn()

    neb = NEB(images, climb=False, k=spring_const, method='aseneb')

    # Phase 1 — regular NEB
    phase1_fmax = neb_ftol * 3.0
    print(f'\nPhase 1: regular NEB  ({phase1_steps} steps, fmax={phase1_fmax:.3f} eV/Å)')
    sys.stdout.flush()
    opt1 = MDMin(neb, logfile=logfile_phase1, dt=0.05)
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
    opt2 = MDMin(neb, logfile=logfile_phase2, dt=0.02)
    converged = False
    try:
        converged = opt2.run(fmax=neb_ftol, steps=phase2_steps)
    except Exception as exc:
        print(f'WARNING: phase 2 raised: {exc}')
        sys.stdout.flush()
    print(f'Phase 2 done: {opt2.nsteps} steps')
    sys.stdout.flush()

    # Compute final fmax across intermediate images
    fmax_vals = []
    for img in images[1:-1]:
        try:
            f = img.get_forces()
            fmax_vals.append(np.sqrt((f ** 2).sum(axis=1).max()))
        except Exception:
            pass
    fmax_final = float(np.max(fmax_vals)) if fmax_vals else float('nan')

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
    e_fs: float,
    mace_model_path: str,
    barrier_file: str,
    path_file: str,
    logfile_phase1: str,
    logfile_phase2: str,
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 0.05,
    phase1_steps: int = 5000,
    phase2_steps: int = 10000,
    z_freeze_cutoff: float = 22.115,
    device: str = 'cpu',
    label_is: str = 'IS',
    label_fs: str = 'FS',
    out_path: str = 'run_neb.py',
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
    e_fs : float
        FS potential energy in eV (from LAMMPS minimisation log).
    mace_model_path : str
        Path (on the cluster) to the ``.model`` MACE file.
    barrier_file : str
        Output path for ``neb_barrier.txt``.
    path_file : str
        Output path for ``neb_path.dat``.
    logfile_phase1 : str
        MDMin log path for phase 1.
    logfile_phase2 : str
        MDMin log path for phase 2.
    n_images : int
        Number of intermediate NEB images.
    spring_const : float
        Elastic-band spring constant in eV/Å².
    neb_ftol : float
        CINEB force convergence tolerance in eV/Å.
    phase1_steps : int
        Phase 1 MDMin step limit.
    phase2_steps : int
        Phase 2 MDMin step limit.
    z_freeze_cutoff : float
        Frozen-layer z threshold in Å.
    device : str
        PyTorch device (``'cpu'`` or ``'cpu'``).
    label_is : str
        Descriptive label for IS (written to barrier file header).
    label_fs : str
        Descriptive label for FS (written to barrier file header).
    out_path : str
        Path where the generated ``.py`` script is written.

    Returns
    -------
    str
        Path to the written script (``out_path``).
    """
    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ASE CINEB — generated by models.ase_neb.write_ase_neb_script
        # IS: {label_is}
        # FS: {label_fs}
        # Interpolation: IDPP
        # Two-phase: regular NEB (phase 1) → CINEB (phase 2)

        import os, sys
        import numpy as np

        # Strip cpu stubs from LD_LIBRARY_PATH — causes segfault on some nodes
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            p for p in _ld.split(":") if "stubs" not in p)

        from ase.io import read
        from ase.mep import NEB
        from ase.optimize import MDMin
        from ase.calculators.singlepoint import SinglePointCalculator
        from mace.calculators import MACECalculator

        # ── Parameters injected from notebook ─────────────────────────────
        MACE_MODEL      = "{mace_model_path}"
        NEB_IS_FILE     = "{is_file}"
        FS_RELAXED_DATA = "{fs_file}"
        Z_FREEZE_CUTOFF = {z_freeze_cutoff}    # Å — frozen layer threshold
        E_IS            = {e_is}               # eV — from LAMMPS IS minimisation
        E_FS            = {e_fs}               # eV — from LAMMPS FS minimisation
        N_IMAGES        = {n_images}           # intermediate images
        SPRING_CONST    = {spring_const}       # eV/Å²
        NEB_FMAX        = {neb_ftol}           # eV/Å — CINEB convergence
        N1_FMAX         = NEB_FMAX * 3         # eV/Å — phase 1 (looser)
        N1_STEPS        = {phase1_steps}
        NEB_STEPS       = {phase2_steps}
        DEVICE          = "{device}"
        BARRIER_FILE    = "{barrier_file}"
        PATH_FILE       = "{path_file}"
        LOG_PHASE1      = "{logfile_phase1}"
        LOG_PHASE2      = "{logfile_phase2}"
        LABEL_IS        = "{label_is}"
        LABEL_FS        = "{label_fs}"

        # ── Calculator with frozen layers ─────────────────────────────────
        class MACEFrozenCalc(MACECalculator):
            def get_forces(self, atoms=None):
                forces = super().get_forces(atoms).copy()
                a = self.atoms if atoms is None else atoms
                forces[a.get_positions()[:, 2] < Z_FREEZE_CUTOFF] = 0.0
                return forces

        def make_calc():
            return MACEFrozenCalc(
                model_paths=MACE_MODEL, device=DEVICE, default_dtype="float64")

        # ── Load structures and pin endpoint energies ─────────────────────
        is_raw = read(NEB_IS_FILE, format="lammps-data", atom_style="atomic")
        fs_raw = read(FS_RELAXED_DATA, format="lammps-data", atom_style="atomic")

        def pin(atoms, energy):
            atoms = atoms.copy()
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results = {{"energy": energy,
                                   "forces": np.zeros((len(atoms), 3))}}
            return atoms

        is_end = pin(is_raw, E_IS)
        fs_end = pin(fs_raw, E_FS)

        # ── Build image chain and IDPP interpolation ──────────────────────
        images = [is_end] + [is_raw.copy() for _ in range(N_IMAGES)] + [fs_end]
        neb = NEB(images, climb=False, k=SPRING_CONST, method="aseneb")
        neb.interpolate(method="idpp")
        print("IDPP interpolation done")
        sys.stdout.flush()

        for img in images[1:-1]:
            img.calc = make_calc()

        # ── Phase 1: regular NEB ──────────────────────────────────────────
        print(f"\\nPhase 1: regular NEB  ({{N1_STEPS}} steps, fmax={{N1_FMAX:.3f}} eV/Å)")
        sys.stdout.flush()
        opt1 = MDMin(neb, logfile=LOG_PHASE1, dt=0.05)
        opt1.run(fmax=N1_FMAX, steps=N1_STEPS)
        print(f"Phase 1 done: {{opt1.nsteps}} steps")
        sys.stdout.flush()

        for img in images[1:-1]:
            img.set_momenta(np.zeros_like(img.get_momenta()))

        # ── Phase 2: CINEB ────────────────────────────────────────────────
        neb.climb = True
        print(f"\\nPhase 2: CINEB  ({{NEB_STEPS}} steps, fmax={{NEB_FMAX:.3f}} eV/Å)")
        sys.stdout.flush()
        opt2 = MDMin(neb, logfile=LOG_PHASE2, dt=0.02)
        converged = False
        try:
            converged = opt2.run(fmax=NEB_FMAX, steps=NEB_STEPS)
        except Exception as exc:
            print(f"WARNING: phase 2 raised: {{exc}}")
            sys.stdout.flush()
        print(f"Phase 2 done: {{opt2.nsteps}} steps")
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

        fmax_vals = []
        for img in images[1:-1]:
            try:
                f = img.get_forces()
                fmax_vals.append(float(np.sqrt((f**2).sum(axis=1).max())))
            except Exception:
                pass
        fmax_final = max(fmax_vals) if fmax_vals else float("nan")

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
            f.write(f"Converged   : {{converged}}\\n\\n")
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
    """)

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
    e_fs: float,
    mace_model_path: str,
    barrier_file: str,
    path_file: str,
    outdir: str,
    job_name: str = 'neb',
    n_images: int = 18,
    spring_const: float = 1.0,
    neb_ftol: float = 0.05,
    phase1_steps: int = 5000,
    phase2_steps: int = 10000,
    z_freeze_cutoff: float = 22.115,
    device: str = 'cpu',
    label_is: str = 'IS',
    label_fs: str = 'FS',
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
    e_fs : float
        FS potential energy in eV (from LAMMPS minimisation log).
    mace_model_path : str
        Path (on cluster) to the ``.model`` MACE file.
    barrier_file : str
        Output path the script will write ``neb_barrier.txt`` to.
    path_file : str
        Output path the script will write ``neb_path.dat`` to.
    outdir : str
        Directory where the script and log files will be written.
    job_name : str
        Base name for the script and log files (default ``'neb'``).
    n_images : int
        Number of intermediate NEB images (default 18).
    spring_const : float
        Elastic-band spring constant in eV/Å² (default 2.0).
    neb_ftol : float
        CINEB force convergence tolerance in eV/Å (default 0.05).
    phase1_steps : int
        Phase 1 MDMin step limit (default 5000).
    phase2_steps : int
        Phase 2 MDMin step limit (default 10000).
    z_freeze_cutoff : float
        Frozen-layer z threshold in Å (default 22.115).
    device : str
        PyTorch device — ``'cpu'`` (default) or ``'cpu'`` for CPU partitions.
    label_is : str
        Human-readable label for IS written to the barrier file header.
    label_fs : str
        Human-readable label for FS written to the barrier file header.

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

    return write_ase_neb_script(
        is_file=is_file,
        fs_file=fs_file,
        e_is=e_is,
        e_fs=e_fs,
        mace_model_path=mace_model_path,
        barrier_file=barrier_file,
        path_file=path_file,
        logfile_phase1=os.path.join(outdir, f'{job_name}_phase1.log'),
        logfile_phase2=os.path.join(outdir, f'{job_name}_phase2.log'),
        n_images=n_images,
        spring_const=spring_const,
        neb_ftol=neb_ftol,
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        z_freeze_cutoff=z_freeze_cutoff,
        device=device,
        label_is=label_is,
        label_fs=label_fs,
        out_path=os.path.join(outdir, f'run_{job_name}.py'),
    )

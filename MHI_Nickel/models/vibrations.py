"""
models/vibrations.py
====================
Frozen-phonon vibrational frequency calculations using ASE Vibrations + MACE.

Applies a partial Hessian — H atom + 6 nearest metal neighbours, 42 single-points
per structure — to compute IS and TS frequencies for Vineyard prefactor evaluation.

Workflow
--------
1. ``write_vibration_script``  — generate standalone ``vib_run.py`` for one structure
2. ``extract_ts_structure``    — locate the TS image from a converged NEB job directory
3. ``collect_is_ts_paths``     — build (label, path) list from Hop A / Hop B job dicts
4. ``load_vibration_results``  — read ``vib_frequencies.json`` after the cluster run
5. ``orchestrate_vibrations``  — write + (optionally submit) SLURM jobs for a batch
"""

from __future__ import annotations

import json
import os
import textwrap
import warnings

from models.create_slurm import write_slurm_job, submit_slurm_job


# ---------------------------------------------------------------------------
# Section 1 — TS structure extraction
# ---------------------------------------------------------------------------

def extract_ts_structure(job_dir: str) -> str:
    """Return path to the TS image LAMMPS file from a converged NEB job directory.

    Parses the ``Image energies:`` block in ``neb_barrier.txt``, identifies
    the highest-energy intermediate image (IS and FS indices are excluded),
    and returns the path to the corresponding file in ``images/``.

    Parameters
    ----------
    job_dir : str
        NEB job directory containing ``neb_barrier.txt`` and ``images/``.

    Returns
    -------
    str
        Absolute path to ``images/image_{ts_index:02d}_img_{ts_index:02d}.lammps``.

    Raises
    ------
    FileNotFoundError
        If ``neb_barrier.txt`` does not exist, or the TS image file is missing.
    ValueError
        If no intermediate image energies can be parsed from the barrier file.
    """
    barrier_file = os.path.join(job_dir, 'neb_barrier.txt')
    if not os.path.exists(barrier_file):
        raise FileNotFoundError(f'neb_barrier.txt not found in {job_dir}')

    image_energies: dict[int, float] = {}
    in_block = False
    with open(barrier_file) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == 'Image energies:':
                in_block = True
                continue
            if not in_block:
                continue
            if not stripped or stripped.startswith('#'):
                continue
            if ':' not in stripped:
                continue
            lbl, _, rest = stripped.partition(':')
            lbl = lbl.strip()
            if lbl in ('IS', 'FS'):
                continue
            if lbl.startswith('img_'):
                try:
                    idx = int(lbl.split('_')[1])
                    e   = float(rest.strip().split()[0])
                    image_energies[idx] = e
                except (IndexError, ValueError):
                    pass

    if not image_energies:
        raise ValueError(
            f'No intermediate image energies found in {barrier_file}. '
            'Verify the NEB has converged and the barrier file is complete.'
        )

    ts_index = max(image_energies, key=image_energies.__getitem__)
    ts_file  = os.path.join(
        job_dir, 'images', f'image_{ts_index:02d}_img_{ts_index:02d}.lammps'
    )
    if not os.path.exists(ts_file):
        raise FileNotFoundError(
            f'TS image file not found: {ts_file}\n'
            f'  ts_index={ts_index} from barrier file — check images/ directory.'
        )
    print(f'[TS] image {ts_index}/{len(image_energies)+1}  E_max={image_energies[ts_index]:.4f} eV  → {ts_file}')
    return os.path.abspath(ts_file)


# ---------------------------------------------------------------------------
# Section 2 — Script generator
# ---------------------------------------------------------------------------

def write_vibration_script(
    structure_path: str,
    mace_model_path: str,
    out_path: str,
    outdir: str,
    delta: float = 0.01,
    device: str = 'cpu',
) -> str:
    """Write a standalone Python script that computes partial-Hessian frequencies.

    The generated script:

    1. Loads the structure from a LAMMPS data file via ASE.
    2. Attaches ``MACECalculator``.
    3. Identifies the single H atom and its 6 nearest metal neighbours.
    4. Runs ``ASE Vibrations`` on those 7 atoms (42 single-point calculations).
    5. Saves ``{outdir}/vib_frequencies.json``.

    Parameters
    ----------
    structure_path : str
        LAMMPS data file (IS or TS geometry).
    mace_model_path : str
        Path to the MACE ``.model`` file.
    out_path : str
        Destination path for the generated ``vib_run.py``.
    outdir : str
        Directory for ASE cache files and output ``vib_frequencies.json``.
    delta : float
        Finite-difference displacement magnitude (Å).  Default 0.01.
    device : str
        MACE device string — ``'cpu'`` or ``'cuda'``.  Default ``'cpu'``.
        Vibration runs on CPU because 42 single-points do not saturate a GPU.

    Returns
    -------
    str
        Path to the written script.
    """
    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Partial-Hessian vibrational frequency calculation via ASE + MACE."""
        import json, os
        import numpy as np
        from ase.io import read
        from ase.vibrations import Vibrations
        from mace.calculators import MACECalculator

        STRUCTURE  = {structure_path!r}
        MACE_MODEL = {mace_model_path!r}
        OUTDIR     = {outdir!r}
        DELTA      = {delta}
        DEVICE     = {device!r}

        os.makedirs(OUTDIR, exist_ok=True)

        # ── Load structure ────────────────────────────────────────────────────
        atoms = read(STRUCTURE, format="lammps-data", atom_style="atomic")
        calc  = MACECalculator(
            model_paths=MACE_MODEL, device=DEVICE, default_dtype="float64", head="omat_pbe"
        )
        atoms.calc = calc

        # ── Identify H and 6 nearest metal neighbours ─────────────────────────
        syms      = np.array(atoms.get_chemical_symbols())
        pos       = atoms.get_positions()
        h_indices = np.where(syms == "H")[0]
        if len(h_indices) != 1:
            raise RuntimeError(
                "Expected exactly 1 H atom, found " + str(len(h_indices))
            )
        h_idx     = int(h_indices[0])
        metal_idx = np.where(syms != "H")[0]
        dists     = np.linalg.norm(pos[metal_idx] - pos[h_idx], axis=1)
        nearest6  = metal_idx[np.argsort(dists)[:6]].tolist()
        indices   = [h_idx] + nearest6
        print("Displacing", len(indices), "atoms: H @", h_idx, "metals @", nearest6)

        # ── Finite-difference vibrations ──────────────────────────────────────
        vib_name = os.path.join(OUTDIR, "vib")
        vib = Vibrations(atoms, indices=indices, delta=DELTA, name=vib_name)
        vib.run()
        vib.summary()

        # ── Extract real and imaginary frequencies ────────────────────────────
        # ASE returns complex cm^-1: imaginary modes have f.imag != 0
        freqs_raw      = vib.get_frequencies()
        freqs_real_cm1 = []
        freqs_imag_cm1 = []
        for f in freqs_raw:
            if abs(f.imag) > 1e-6:
                freqs_imag_cm1.append(float(abs(f.imag)))
            elif f.real > 0:
                freqs_real_cm1.append(float(f.real))
            else:
                # near-zero or slightly negative — treat as imaginary
                freqs_imag_cm1.append(float(abs(f.real)))

        result = {{
            "structure":            STRUCTURE,
            "n_atoms_displaced":    len(indices),
            "h_index":              h_idx,
            "metal_indices":        nearest6,
            "delta_ang":            DELTA,
            "frequencies_real_cm1": freqs_real_cm1,
            "frequencies_imag_cm1": freqs_imag_cm1,
        }}

        out_json = os.path.join(OUTDIR, "vib_frequencies.json")
        with open(out_json, "w") as fh:
            json.dump(result, fh, indent=2)
        print("Wrote:", out_json)
        print("Real modes (", len(freqs_real_cm1), "):", freqs_real_cm1[:5], "...")
        print("Imag modes (", len(freqs_imag_cm1), "):", freqs_imag_cm1)
    ''')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(script)
    os.chmod(out_path, 0o755)
    print(f'Written: {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Section 3 — Collect IS / TS paths from job dicts
# ---------------------------------------------------------------------------

def collect_is_ts_paths(
    neb_jobs: list,
    hop: str = 'hopa',
    is_key: str = 'is_path',
) -> list[tuple[str, str]]:
    """Build a ``(label, lammps_path)`` list for IS and TS of each NEB job.

    The TS path is extracted at call time by parsing ``neb_barrier.txt`` inside
    each job's ``job_dir``.  Jobs whose NEB has not yet converged (missing barrier
    file or image files) emit a warning and are skipped for the TS entry only;
    the IS entry is always included when ``is_key`` resolves to an existing file.

    Parameters
    ----------
    neb_jobs : list of dict
        Job dicts from ``orchestrate_hopa_neb`` or ``orchestrate_hopb_neb``.
        Required keys: ``sid``, ``job_dir``, and the key named by ``is_key``.
    hop : str
        Prefix used in output labels — ``'hopa'`` or ``'hopb'``.
    is_key : str
        Key in each job dict that holds the IS LAMMPS path.
        Use ``'is_path'`` for Hop A jobs, ``'hopb_is'`` for Hop B jobs.

    Returns
    -------
    list of (label, lammps_path)
        Labels follow the pattern ``'{hop}_{sid}_IS'`` and ``'{hop}_{sid}_TS'``.
    """
    pairs: list[tuple[str, str]] = []
    for job in neb_jobs:
        sid       = job['sid']
        is_path   = job.get(is_key, '')
        job_dir   = job.get('job_dir', '')

        if is_path and os.path.exists(is_path):
            pairs.append((f'{hop}_{sid}_IS', is_path))
        else:
            warnings.warn(
                f'[{hop}_{sid}] IS file not found at {is_path!r}; skipping IS.'
            )

        if job_dir:
            try:
                ts_path = extract_ts_structure(job_dir)
                pairs.append((f'{hop}_{sid}_TS', ts_path))
            except (FileNotFoundError, ValueError) as exc:
                warnings.warn(
                    f'[{hop}_{sid}] TS extraction failed: {exc}; '
                    'run NEB first or check images/ directory.'
                )

    return pairs


# ---------------------------------------------------------------------------
# Section 4 — Result loader
# ---------------------------------------------------------------------------

def load_vibration_results(vib_json_path: str) -> dict:
    """Load frequency results from ``vib_frequencies.json``.

    Parameters
    ----------
    vib_json_path : str
        Path to ``vib_frequencies.json`` written by the vib_run.py script.

    Returns
    -------
    dict
        Keys: ``frequencies_real_cm1`` (list[float]), ``frequencies_imag_cm1``
        (list[float]), ``h_index`` (int), ``metal_indices`` (list[int]),
        ``n_atoms_displaced`` (int), ``delta_ang`` (float), ``structure`` (str).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(vib_json_path):
        raise FileNotFoundError(f'vib_frequencies.json not found: {vib_json_path}')
    with open(vib_json_path) as fh:
        _data = json.load(fh)
    _n_real = len(_data.get('frequencies_real_cm1', []))
    _n_imag = len(_data.get('frequencies_imag_cm1', []))
    print(f'[vib] {_n_real} real modes  {_n_imag} imag modes  ({os.path.basename(vib_json_path)})')
    return _data


# ---------------------------------------------------------------------------
# Section 5 — Orchestrator
# ---------------------------------------------------------------------------

def orchestrate_vibrations(
    structure_paths: list,
    outdir: str,
    mace_model_path: str,
    slurm_opts: dict | None = None,
    delta: float = 0.01,
    device: str = 'cpu',
    dry_run: bool = True,
) -> dict:
    """Write and optionally submit one ``vib_run.py`` + SLURM job per structure.

    Parameters
    ----------
    structure_paths : list of (label, lammps_path)
        Structures to compute frequencies for.  Labels should be unique, e.g.
        ``[('hopa_Ni3Mo_IS', '...'), ('hopa_Ni3Mo_TS', '...'), ...]``.
        Build this list with :func:`collect_is_ts_paths`.
    outdir : str
        Root output directory.  Each label gets its own sub-directory
        ``{outdir}/{label}/``.
    mace_model_path : str
        Path to the MACE ``.model`` file.
    slurm_opts : dict, optional
        SLURM cluster configuration passed to ``write_slurm_job``.
        If ``None``, only Python scripts are written (no SLURM scripts).
    delta : float
        Finite-difference displacement magnitude (Å).
    device : str
        MACE device string.  Vibration runs are CPU-bound; default ``'cpu'``.
    dry_run : bool
        If ``True``, write scripts without calling ``sbatch``.

    Returns
    -------
    dict
        ``{label: {'vib_script': str, 'vib_json': str, 'slurm': str | None}}``
    """
    os.makedirs(outdir, exist_ok=True)
    results: dict = {}

    for label, lammps_path in structure_paths:
        job_outdir  = os.path.join(outdir, label)
        os.makedirs(job_outdir, exist_ok=True)
        script_path = os.path.join(job_outdir, 'vib_run.py')
        vib_json    = os.path.join(job_outdir, 'vib_frequencies.json')

        write_vibration_script(
            structure_path  = lammps_path,
            mace_model_path = mace_model_path,
            out_path        = script_path,
            outdir          = job_outdir,
            delta           = delta,
            device          = device,
        )

        slurm_path = None
        if slurm_opts is not None:
            slurm_path = os.path.join(job_outdir, 'vib_job.sh')
            write_slurm_job(
                job_name   = f'vib_{label}',
                slurm_config = slurm_opts,
                out_path   = slurm_path,
                script_path= script_path,
                runner     = 'python',
                output_log = os.path.join(job_outdir, f'vib_{label}_%j.out'),
            )
            if not dry_run:
                submit_slurm_job(slurm_path)

        results[label] = {
            'vib_script': script_path,
            'vib_json'  : vib_json,
            'slurm'     : slurm_path,
        }

    n      = len(structure_paths)
    status = 'scripts written (dry_run)' if dry_run else f'{n} jobs submitted'
    print(f'[orchestrate_vibrations] {n} structures → {outdir}  ({status})')
    return results

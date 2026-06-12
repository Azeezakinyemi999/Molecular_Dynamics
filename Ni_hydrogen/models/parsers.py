"""
models/parsers.py
=================
Parsers for LAMMPS output files produced by this project's workflow.

Public API
----------
parse_minimization_log      NB02 — .log from energy minimization
parse_surface_relaxation_log NB04 — .log from surface relaxation
parse_equil_log             NB10 — .log from NVT bulk equilibration
parse_thermo_series         NB10 — thermo columns from any LAMMPS log
parse_lammps_dump           NB11 — .lammpstrj trajectory
parse_diffusivity_file      NB12 — tabular D(T) results file
"""

#from __future__ import annotations

import os
import re
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# Private base
# ════════════════════════════════════════════════════════════════════════════

def _parse_lammps_log(
    logfile: str,
    result_block_tag: str,
    result_keys: list[str] | None = None,
) -> tuple[dict, dict]:
    """
    Generic LAMMPS log parser — shared base for minimization, relaxation,
    and equilibration logs.

    Parameters
    ----------
    logfile : str
        Path to the LAMMPS log file.
    result_block_tag : str
        Sentinel prefix used in the log, e.g. ``'MINIMIZATION'`` expects
        ``MINIMIZATION_RESULTS_START`` / ``MINIMIZATION_RESULTS_END``.
    result_keys : list of str, optional
        Extra keys to scan for outside the results block via ``key : value``
        inline patterns (backward-compatibility fallback).

    Returns
    -------
    thermo : dict
        Lists keyed by ``step``, ``pe``, ``fmax``, ``lx``, ``press``,
        ``time``, ``temp``.  Only columns present in the log are populated.
    meta : dict
        Scalar results parsed from the ``*_RESULTS`` block and/or inline
        print statements.
    """
    col_map = {
        'step':  'Step',
        'pe':    'PotEng',
        'fmax':  'Fmax',
        'lx':    'Lx',
        'press': 'Press',
        'time':  'Time',
        'temp':  'Temp',
    }
    thermo = {k: [] for k in col_map}
    meta: dict = {}

    start_tag = f'{result_block_tag}_RESULTS_START'
    end_tag   = f'{result_block_tag}_RESULTS_END'

    in_thermo  = False
    in_results = False
    headers: list[str] = []

    _end_tokens = ('Loop time', 'Minimization stats', 'WARNING', 'CITE')

    with open(logfile) as f:
        for raw in f:
            line = raw.strip()

            # ── results block ─────────────────────────────────────────
            if start_tag in line:
                in_results = True
                continue
            if end_tag in line:
                in_results = False
                continue
            if in_results and ':' in line:
                try:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = float(v.strip().split()[0])
                except Exception:
                    pass
                continue

            # ── thermo header ─────────────────────────────────────────
            if line.startswith('Step'):
                headers   = line.split()
                in_thermo = True
                continue

            # ── thermo end ────────────────────────────────────────────
            if in_thermo and any(line.startswith(t) for t in _end_tokens):
                in_thermo = False

            # ── thermo data ───────────────────────────────────────────
            if in_thermo and line and line[0].isdigit():
                vals = line.split()
                if len(vals) == len(headers):
                    row = dict(zip(headers, vals))
                    for dest, src in col_map.items():
                        if src in row:
                            try:
                                thermo[dest].append(float(row[src]))
                            except Exception:
                                pass

            # ── inline fallback keys (outside results block) ──────────
            if result_keys and not in_results and ':' in line:
                for key in result_keys:
                    if key in line:
                        try:
                            meta[key] = float(line.split(':', 1)[1].strip().split()[0])
                        except Exception:
                            pass

    return thermo, meta


# ════════════════════════════════════════════════════════════════════════════
# Public wrappers — log files
# ════════════════════════════════════════════════════════════════════════════

def parse_minimization_log(logfile: str) -> tuple[dict, dict]:
    """
    Parse a LAMMPS energy-minimization log (NB02).

    Parameters
    ----------
    logfile : str
        Path to the LAMMPS ``.log`` file.

    Returns
    -------
    thermo : dict
        Lists — ``step``, ``pe``, ``fmax``, ``lx``, ``press``.
    meta : dict
        Scalars — ``Total_energy_eV``, ``Natoms``, ``Ecoh_eV_per_atom``,
        ``a0_Angstrom``, ``Fmax_eV_per_Ang``, ``stop_criterion`` (str).
    """
    result_keys = [
        'Total_energy_eV', 'Natoms', 'Ecoh_eV_per_atom',
        'a0_Angstrom', 'Fmax_eV_per_Ang',
    ]
    thermo, meta = _parse_lammps_log(logfile, 'MINIMIZATION', result_keys)

    # capture stopping criterion line (string, not float)
    with open(logfile) as f:
        for line in f:
            if 'Stopping criterion' in line:
                meta['stop_criterion'] = line.strip()
                break

    return thermo, meta


def parse_surface_relaxation_log(logfile: str) -> tuple[dict, dict]:
    """
    Parse a LAMMPS surface-relaxation log (NB04).

    Parameters
    ----------
    logfile : str
        Path to the LAMMPS ``.log`` file.

    Returns
    -------
    thermo : dict
        Lists — ``step``, ``time``, ``temp``, ``pe``, ``press``.
    meta : dict
        Scalars — ``z_top_before_Ang``, ``z_top_after_min_Ang``,
        ``z_top_after_nvt_Ang``, ``surface_contraction``, ``pe_final_eV``,
        ``frozen_count`` (int), ``free_count`` (int).
    """
    result_keys = [
        'z_top_before_Ang', 'z_top_after_min_Ang', 'z_top_after_nvt_Ang',
        'surface_contraction', 'pe_final_eV',
    ]
    thermo, meta = _parse_lammps_log(logfile, 'RELAXATION', result_keys)

    # regex for inline group count: "GROUPS: frozen=60  free=300"
    with open(logfile) as f:
        for line in f:
            if 'GROUPS: frozen=' in line:
                m = re.search(r'frozen=(\d+)', line)
                if m:
                    meta['frozen_count'] = int(m.group(1))
                m = re.search(r'free=(\d+)', line)
                if m:
                    meta['free_count'] = int(m.group(1))
                break

    return thermo, meta


def parse_equil_log(logfile: str) -> dict | None:
    """
    Extract final scalar results from a NVT equilibration log (NB10).

    Parameters
    ----------
    logfile : str
        Path to the LAMMPS ``.log`` file.

    Returns
    -------
    meta : dict or None
        Keys — ``T_K``, ``pe_final_eV``, ``temp_final_K``, ``press_final``.
        Returns ``None`` if the file does not exist or no keys were found.
    """
    if not os.path.exists(logfile):
        return None
    result_keys = ['T_K', 'pe_final_eV', 'temp_final_K', 'press_final']
    _, meta = _parse_lammps_log(logfile, 'EQUIL', result_keys)
    return meta if meta else None


# ════════════════════════════════════════════════════════════════════════════
# Standalone — thermo column reader
# ════════════════════════════════════════════════════════════════════════════

def parse_thermo_series(
    logfile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Extract step / temperature / potential-energy columns from any LAMMPS log.

    Uses positional parsing (columns 0, 1, 2) after a ``Step Temp …`` header,
    so it works for both heating and production NVT blocks.

    Parameters
    ----------
    logfile : str
        Path to the LAMMPS ``.log`` file.

    Returns
    -------
    steps : ndarray
    temps : ndarray
    pes   : ndarray
        All shape ``(N,)``.  Returns ``None`` if the file is absent or empty.
    """
    if not os.path.exists(logfile):
        return None

    steps, temps, pes = [], [], []
    in_thermo = False

    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Step') and 'Temp' in line:
                in_thermo = True
                continue
            if in_thermo:
                if line.startswith('Loop') or line.startswith('WARNING'):
                    in_thermo = False
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        steps.append(float(parts[0]))
                        temps.append(float(parts[1]))
                        pes.append(float(parts[2]))
                    except ValueError:
                        pass

    if steps:
        return np.array(steps), np.array(temps), np.array(pes)
    return None


# ════════════════════════════════════════════════════════════════════════════
# Standalone — LAMMPS dump trajectory
# ════════════════════════════════════════════════════════════════════════════

def parse_lammps_dump(
    traj_file: str,
    h_type: int = 9,
    timestep: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Parse a LAMMPS custom dump file and extract H-atom positions (NB11).

    Expects columns ``id type x y z`` (additional columns are ignored).

    Parameters
    ----------
    traj_file : str
        Path to the ``.lammpstrj`` dump file.
    h_type : int, optional
        LAMMPS atom type index for hydrogen.  Default ``9``.
    timestep : float, optional
        MD timestep in ps used to convert step numbers to time.
        Default ``0.0005`` ps (0.5 fs), matching the NVT bulk scripts.

    Returns
    -------
    t_arr   : ndarray, shape (N_frames,)
        Frame times in ps.
    pos_arr : ndarray, shape (N_frames, 3)
        H-atom x, y, z coordinates in Å.  ``NaN`` for frames where no H
        atom of ``h_type`` was found.
    box_arr : ndarray, shape (N_frames, 3)
        Box edge lengths Lx, Ly, Lz in Å.
    
    Returns ``(None, None, None)`` if the file does not exist or is empty.
    """
    if not os.path.exists(traj_file):
        return None, None, None

    with open(traj_file) as f:
        lines = f.readlines()

    timesteps: list[int]         = []
    h_positions: list[list[float]] = []
    box_lengths: list[list[float]] = []

    i = 0
    while i < len(lines):
        if lines[i].strip() != 'ITEM: TIMESTEP':
            i += 1
            continue

        ts = int(lines[i + 1].strip())
        timesteps.append(ts)

        # ── find NUMBER OF ATOMS ──────────────────────────────────────
        n_atoms = 0
        j = i + 2
        while j < len(lines) and 'NUMBER OF ATOMS' not in lines[j]:
            j += 1
        if j < len(lines):
            n_atoms = int(lines[j + 1].strip())

        # ── find BOX BOUNDS ───────────────────────────────────────────
        while j < len(lines) and 'BOX BOUNDS' not in lines[j]:
            j += 1
        if j < len(lines):
            xlo, xhi = map(float, lines[j + 1].split()[:2])
            ylo, yhi = map(float, lines[j + 2].split()[:2])
            zlo, zhi = map(float, lines[j + 3].split()[:2])
            box_lengths.append([xhi - xlo, yhi - ylo, zhi - zlo])
            j += 4

        # ── find ATOMS header ─────────────────────────────────────────
        while j < len(lines) and 'ITEM: ATOMS' not in lines[j]:
            j += 1
        if j >= len(lines):
            h_positions.append([np.nan, np.nan, np.nan])
            i = j
            continue

        header = lines[j].split()[2:]   # drop 'ITEM:' 'ATOMS'
        try:
            type_col = header.index('type')
            x_col    = header.index('x')
            y_col    = header.index('y')
            z_col    = header.index('z')
        except ValueError:
            type_col, x_col, y_col, z_col = 1, 2, 3, 4

        j += 1
        h_found = False
        for k in range(j, j + n_atoms):
            if k >= len(lines):
                break
            parts = lines[k].split()
            if len(parts) > max(type_col, x_col, y_col, z_col):
                if int(parts[type_col]) == h_type:
                    h_positions.append([
                        float(parts[x_col]),
                        float(parts[y_col]),
                        float(parts[z_col]),
                    ])
                    h_found = True
                    break
        if not h_found:
            h_positions.append([np.nan, np.nan, np.nan])

        i = j + n_atoms

    if not timesteps:
        return None, None, None

    t_arr   = np.array(timesteps, dtype=float) * timestep
    pos_arr = np.array(h_positions)
    box_arr = np.array(box_lengths)
    return t_arr, pos_arr, box_arr


# ════════════════════════════════════════════════════════════════════════════
# Standalone — diffusivity results file
# ════════════════════════════════════════════════════════════════════════════

def parse_diffusivity_file(
    diff_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the tabular diffusivity results file written by NB11 (NB12 input).

    Expected column order: ``T_K  D  sigma_D  R2``

    Parameters
    ----------
    diff_file : str
        Path to ``diffusivity.txt``.

    Returns
    -------
    T_arr    : ndarray   Temperature in K
    D_arr    : ndarray   Diffusivity in m²/s
    Derr_arr : ndarray   Standard error of D in m²/s
    R2_arr   : ndarray   R² of the MSD linear fit

    Raises
    ------
    FileNotFoundError
        If ``diff_file`` does not exist.
    """
    if not os.path.exists(diff_file):
        raise FileNotFoundError(
            f'{diff_file} not found. Complete NB11 first.'
        )

    _SKIP = ('#', '=', '-', 'T_K')

    T_vals, D_vals, Derr_vals, R2_vals = [], [], [], []

    with open(diff_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(s) for s in _SKIP):
                continue
            if ':' in line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    T_vals.append(float(parts[0]))
                    D_vals.append(float(parts[1]))
                    Derr_vals.append(float(parts[2]))
                    R2_vals.append(float(parts[3]))
                except ValueError:
                    pass

    return (
        np.array(T_vals),
        np.array(D_vals),
        np.array(Derr_vals),
        np.array(R2_vals),
    )
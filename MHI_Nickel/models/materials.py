"""
models/materials.py
===================
Single source of truth for *which* materials the pipeline processes, and how
each one is classified.

Why this module exists
----------------------
This list used to be typed out four times -- once in each of the three
``calculation/tools/regenerate_*_scripts.py`` and once in
``calculation/pipeline.ipynb`` -- along with a copy of ``classify_metal`` and
the surface-skip rules. Nothing kept the copies in agreement, so adding a
material meant four correct edits and missing one failed *silently*: the part
whose list you forgot simply never generated a script for your metal, and you
would only notice when its results were absent.

Add a material in ONE place: append to :data:`INPUT_STRUCTURE_FILES`.

Scope note: the skip rules below apply to the SURFACE work (Part 1 surface NEB
and Part 2 permeation). Part 3 (bulk diffusivity) runs for every structure,
including the oxides -- it never builds a surface, so the terminations and
crystal-structure caveats do not apply to it.
"""

from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# The materials
# ---------------------------------------------------------------------------
# File names as they appear in calculation/input_structure/. Add new materials
# here and nowhere else.
INPUT_STRUCTURE_FILES = [
    'Hastelloy_N_7_supercell.lammps',
    'Cr_oxide_supercell.lammps',
    'Hastelloy_N_42_supercell.lammps',
    'Hastelloy_N_111_supercell.lammps',
    'Hastelloy_N_1234_supercell.lammps',
    'Hastelloy_N_12345_supercell.lammps',
    'Al_supercell.lammps',
    'Fe_supercell.lammps',
    'Ni_supercell.lammps',
    'bestsqs3.lmp',
    'Ni_oxide_supercell.lammps',
]


def input_structures(work_dir: str) -> list:
    """Absolute paths to every input structure, under ``work_dir``.

    Parameters
    ----------
    work_dir : str
        The ``calculation/`` directory (normally ``os.path.join(BASE_DIR,
        'calculation')``).
    """
    return [os.path.join(work_dir, 'input_structure', f)
            for f in INPUT_STRUCTURE_FILES]


def stem_of(path: str) -> str:
    """``.../Al_supercell.lammps`` -> ``Al_supercell``."""
    return os.path.splitext(os.path.basename(path))[0]


def classify_metal(path: str) -> str:
    """Return ``'oxide'``, ``'alloy'`` or ``'pure'`` for a structure path.

    Drives which element/mass table the generated scripts embed: oxides need
    the O-inclusive 10-type table, everything else the 7-type one.
    """
    stem = stem_of(path).lower()
    if 'oxide' in stem:
        return 'oxide'
    if any(k in stem for k in ('hastelloy', 'bestsqs', 'sqs', 'alloy')):
        return 'alloy'
    return 'pure'


# ---------------------------------------------------------------------------
# Surface-work skip rules (Part 1 surface NEB + Part 2 permeation only)
# ---------------------------------------------------------------------------
SKIP_SURFACE_STEMS = {
    # GitHub #5: NiO(111) primitive-cell Miller indices give the polar
    # Tasker-III termination; needs a per-structure Miller + termination check.
    'Ni_oxide_supercell': 'polar oxide termination (GitHub #5)',
}


def skip_surface_reason(path: str) -> str | None:
    """Why this structure is excluded from the surface workflow, or ``None``.

    Two rules, both validated for FCC(111) only:
      * a named polar-oxide termination, and
      * any pure BCC structure (GitHub #6 -- surface/subsurface untested).

    The BCC test reads the file, so it is done last and tolerates failure: an
    unreadable structure is not silently skipped, it is left in and allowed to
    fail loudly downstream.
    """
    stem = stem_of(path)
    if stem in SKIP_SURFACE_STEMS:
        return SKIP_SURFACE_STEMS[stem]
    if classify_metal(path) == 'pure':
        try:
            from models.structure import is_pure_bcc_structure
            if is_pure_bcc_structure(path):
                return 'BCC surface/subsurface untested (GitHub #6)'
        except Exception:
            return None
    return None

# ---------------------------------------------------------------------------
# Temperature ceilings for molecular dynamics
# ---------------------------------------------------------------------------
# A molecular-dynamics run above a material's melting point measures diffusion
# in a LIQUID. That is a different transport mechanism with a different
# activation energy, and feeding it into a solid-state Arrhenius fit corrupts
# both D0 and E_D -- silently, because the run completes and produces numbers.
#
# The temperature grid is global (one list for every material in the
# regenerators), but the materials are not interchangeable: Al melts at 933 K
# while Ni melts at 1728 K, so a grid that suits Ni runs Al as a liquid. These
# ceilings let one global list stay safe -- each material takes only the
# temperatures it can physically sustain.
#
# VALUES NEED CHECKING against your own sources before being relied on. The
# pure metals are standard; the alloys are approximate, and an alloy melts over
# a RANGE, so the figure below is meant as the solidus -- the temperature above
# which the solid is no longer a solid everywhere.
MELTING_POINT_K = {
    # pure metals -- standard values
    'Al_supercell':               933.5,
    'Ni_supercell':              1728.0,
    'Fe_supercell':              1811.0,
    # oxides -- far above any grid used here
    'Cr_oxide_supercell':        2708.0,
    'Ni_oxide_supercell':        2228.0,
    # Ni-based alloys -- APPROXIMATE solidus, verify before relying on these
    'Hastelloy_N_7_supercell':   1570.0,
    'Hastelloy_N_42_supercell':  1570.0,
    'Hastelloy_N_111_supercell': 1570.0,
    'Hastelloy_N_1234_supercell':1570.0,
    'Hastelloy_N_12345_supercell': 1570.0,
    'bestsqs3':                  1570.0,
}

# Highest homologous temperature T/Tm a production run may use. 0.90 keeps the
# existing Al grid intact (800 K is 0.86 Tm) while excluding 1000 K, which is
# above Al's melting point outright.
MAX_HOMOLOGOUS_T = 0.90


def melting_point_K(path: str) -> float | None:
    """Melting point for a structure, or ``None`` if not recorded."""
    return MELTING_POINT_K.get(stem_of(path))


def max_md_temperature_K(path: str) -> float | None:
    """Highest temperature this material may be run at, or ``None`` if unknown."""
    tm = melting_point_K(path)
    return None if tm is None else MAX_HOMOLOGOUS_T * tm


def usable_temperatures(path: str, temperatures) -> list:
    """Those of `temperatures` this material can physically sustain.

    A material with no recorded melting point is passed through unchanged --
    refusing would break structures that run correctly today -- but the caller
    is expected to surface that, since an unchecked grid is exactly how a
    liquid-phase run reaches an Arrhenius fit.
    """
    cap = max_md_temperature_K(path)
    if cap is None:
        return list(temperatures)
    return [T for T in temperatures if float(T) <= cap]


def dropped_temperatures(path: str, temperatures) -> list:
    """Those of `temperatures` excluded by the ceiling, for reporting."""
    keep = set(usable_temperatures(path, temperatures))
    return [T for T in temperatures if T not in keep]


def narrow_temperatures(path: str, temperatures, results_dir: str = None) -> tuple:
    """`temperatures` reduced to what this material can actually be quoted at.

    Returns ``(kept, notes)``, where `notes` are human-readable reasons for
    every temperature removed — empty when nothing was dropped. Callers are
    expected to print them: a grid that silently shrinks is as bad as one that
    silently melts a slab.

    The caller's list is treated as a *request*, not an instruction. Enforcing
    it here rather than in each regenerator means the notebook, the developer
    tools and any future driver all inherit the same guard, instead of each
    having to remember a rule none of them is in a good position to check.

    Two filters, the second optional:

    1. **Melting ceiling** (always). Drops anything above
       ``MAX_HOMOLOGOUS_T x Tm``. A material with no recorded melting point is
       passed through, and that is itself reported.

    2. **Measured lattice parameter** (only when `results_dir` is given).
       Phase 6 of the permeation workflow reads ``a0(T)`` from
       ``lattice_params_vs_T.json`` with a *silent* fallback to a fixed
       constant, so a temperature with no NPT point is computed with the wrong
       lattice constant and nothing in the output says so. Pass `results_dir`
       wherever that fallback applies; omit it on the diffusivity side, which
       is what *produces* that file — filtering there would be circular, since
       a temperature could never be run for the first time.

    When the lattice file is absent entirely, nothing is dropped and the
    absence is reported instead: with no measured ``a0(T)`` at all, every
    temperature falls back to the same fixed constant, so there is no ground
    for preferring some over others. Removing a caller's temperature needs a
    reason specific to that temperature.
    """
    kept, notes = list(temperatures), []

    cap = max_md_temperature_K(path)
    if cap is None:
        notes.append(f'{stem_of(path)}: no melting point recorded — '
                     f'temperature ceiling NOT enforced')
    else:
        by_melt = dropped_temperatures(path, kept)
        kept = usable_temperatures(path, kept)
        if by_melt:
            notes.append(f'{stem_of(path)}: above the {cap:.0f} K ceiling '
                         f'({MAX_HOMOLOGOUS_T:g} x Tm), dropped {by_melt}')

    if results_dir is not None:
        lat = os.path.join(results_dir, 'lattice_params_vs_T.json')
        if os.path.exists(lat):
            try:
                with open(lat) as f:
                    have = {int(T) for T in json.load(f)['temperatures']}
            except (OSError, ValueError, KeyError, TypeError):
                have = None
                notes.append(f'{stem_of(path)}: lattice_params_vs_T.json is '
                             f'unreadable — lattice filter NOT applied')
            if have is not None:
                by_lat = [T for T in kept if int(T) not in have]
                kept = [T for T in kept if int(T) in have]
                if by_lat:
                    notes.append(f'{stem_of(path)}: no measured a0(T), '
                                 f'dropped {by_lat}')
        else:
            notes.append(f'{stem_of(path)}: no lattice_params_vs_T.json — '
                         f'a0(T) falls back to a fixed constant at every '
                         f'temperature')

    return kept, notes

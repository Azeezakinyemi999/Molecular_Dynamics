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

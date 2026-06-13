"""
models/utils.py
================
Utility helpers for the Hastelloy N MD workflow.
"""

from __future__ import annotations

import os


def make_run_dirs(
    name: str,
    temperatures: list[int | float],
    base_dir: str = '.',
) -> dict:
    """Create the standard directory layout for one diffusivity run.

    Layout
    ------
    ::

        {base_dir}/{name}/
            structures/
            lammps_scripts/{T}K/   (one per temperature)
            slurm_scripts/{T}K/
            results/{T}K/

    Parameters
    ----------
    name : str
        Run name — used as the top-level directory (e.g. ``'seed1'``,
        ``'N_7_10H'``).
    temperatures : list
        Temperature values in K (e.g. ``[300, 400, 600, 800]``).
    base_dir : str, optional
        Root directory.  Default is the current working directory.

    Returns
    -------
    dict
        Nested dict of all created paths::

            {
                'root':  '{base_dir}/{name}',
                'structures': '{base_dir}/{name}/structures',
                300: {
                    'lammps_scripts': '.../{name}/lammps_scripts/300K',
                    'slurm_scripts':  '.../{name}/slurm_scripts/300K',
                    'results':        '.../{name}/results/300K',
                },
                400: { ... },
                ...
            }
    """
    root = os.path.join(base_dir, name)

    subdirs_flat = ['structures']
    subdirs_T    = ['lammps_scripts', 'slurm_scripts', 'results']

    paths: dict = {'root': root}

    # Layer 2 — flat dirs
    for d in subdirs_flat:
        p = os.path.join(root, d)
        os.makedirs(p, exist_ok=True)
        paths[d] = p

    # Layer 2+3 — T-split dirs
    for T in temperatures:
        T_key = int(T)
        paths[T_key] = {}
        for d in subdirs_T:
            p = os.path.join(root, d, f'{T_key}K')
            os.makedirs(p, exist_ok=True)
            paths[T_key][d] = p

    return paths

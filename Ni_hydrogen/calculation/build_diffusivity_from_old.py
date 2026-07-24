#!/usr/bin/env python3
"""
build_diffusivity_from_old.py
=============================
Generalize ``build_ni_diffusivity_from_old.py`` to EVERY material under
``old_notebooks/results/notebook10-bulk-equilibration/`` (except pure Ni, which
is already set up as ``Ni_supercell_1H`` and left untouched). For each material
it converts the old ``fix msd`` output into the ``diffusivity_arrhenius.json``
the permeation orchestrator (Part 2) reads for Part 3 — so a material's
permeation chain can be run *now*, reusing pre-existing diffusivity data instead
of waiting on a fresh Part-3 MD run.

For each ``{prefix}_msd_H_{T}K.txt`` (raw LAMMPS ``fix msd`` output: columns
``TimeStep, MSD[Å²]``), converts steps→ps (dt = 0.0005 ps, metal units), fits
D(T) with the pipeline's own ``fit_diffusivity`` (Einstein D = slope/6), then
Arrhenius-fits D(T) with ``fit_arrhenius`` and writes
``results/{stem}_01H/diffusivity_arrhenius.json``.

Output naming — deliberately zero-padded ``{stem}_01H``
-------------------------------------------------------
``resolve_nh_diffusivity`` (models/permeation.py) looks up ``{stem}_{n_h}H``
*unpadded* (i.e. ``{stem}_1H``), so a ``{stem}_01H`` directory is NOT auto-found
by a permeation run. That makes ``_01H`` an inert "staging" name: the fits sit
ready but dormant. **Rename ``{stem}_01H`` → ``{stem}_1H`` manually** when you
want to run that material's permeation. (Pure Ni is already active as
``Ni_supercell_1H`` and is intentionally excluded here.)

CAVEAT — validation-grade, not production
-----------------------------------------
Same single-origin ``fix msd`` limitation as the Ni converter: at low T the H
barely diffuses in the production window, so the per-T slope is noise-dominated.
Worse here — every Hastelloy variant is capped at 800 K with no high-T anchor —
so these Arrhenius fits are validation-grade only (prove the end-to-end chain /
check magnitudes are physically plausible), not production diffusivities.

Usage
-----
    python build_diffusivity_from_old.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import BASE_DIR
from models.diffusivity_post_processing import fit_diffusivity, fit_arrhenius

WORK_DIR   = os.path.join(BASE_DIR, 'calculation')
OLD_ROOT   = os.path.join(BASE_DIR, 'old_notebooks', 'results',
                          'notebook10-bulk-equilibration')

N_H        = 1                     # old runs: 500 metal + 1 H (dilute)
DT_PS      = 0.0005                # metal-units timestep (0.5 fs), from the equil logs
FIT_WINDOW = (0.2, 0.8)            # skip ballistic start / noisy tail

# (old_folder, msd_prefix, pipeline_stem). Folder name != MSD file prefix.
# Pure_Nickel is intentionally omitted — it is already set up as Ni_supercell_1H
# and left as-is. The 5 pipeline stems with no old MSD data (Al_supercell,
# Fe_supercell, Cr_oxide_supercell, Ni_oxide_supercell, bestsqs3) are not here.
MATERIALS = [
    ('N_7',     'Hastelloy_N_7',     'Hastelloy_N_7_supercell'),
    ('N_42',    'Hastelloy_N_42',    'Hastelloy_N_42_supercell'),
    ('N_111',   'Hastelloy_N_111',   'Hastelloy_N_111_supercell'),
    ('N_1234',  'Hastelloy_N_1234',  'Hastelloy_N_1234_supercell'),
    ('N_12345', 'Hastelloy_N_12345', 'Hastelloy_N_12345_supercell'),
]

_T_RE = re.compile(r'_msd_H_(\d+)K\.txt$')


def _detect_temps(folder, prefix):
    """Temperatures (K, sorted) for which an MSD file exists in ``folder``."""
    temps = []
    for path in glob.glob(os.path.join(folder, f'{prefix}_msd_H_*K.txt')):
        m = _T_RE.search(os.path.basename(path))
        if m:
            temps.append(int(m.group(1)))
    return sorted(temps)


def _d_of_t(msd_path):
    """(D, sigma_D, R2) from one raw fix-msd file, or None if unreadable."""
    if not os.path.exists(msd_path):
        return None
    data = np.loadtxt(msd_path, comments='#')
    if data.ndim != 2 or data.shape[0] < 5:
        return None
    t_ps = (data[:, 0] - data[0, 0]) * DT_PS
    msd  = data[:, 1]
    return fit_diffusivity(t_ps, msd, fit_window=FIT_WINDOW)


def _convert_one(old_folder, prefix, stem):
    """Convert one material. Returns the output path, or None if skipped."""
    folder = os.path.join(OLD_ROOT, old_folder)
    if not os.path.isdir(folder):
        print(f'[{stem}] old folder missing: {folder} — skipping')
        return None
    temps = _detect_temps(folder, prefix)
    if not temps:
        print(f'[{stem}] no MSD files ({prefix}_msd_H_*K.txt) in {folder} — skipping')
        return None
    print(f'[{stem}] {old_folder}: temperatures {temps}')

    T_list, D_list, E_list, R2_list = [], [], [], []
    for T in temps:
        res = _d_of_t(os.path.join(folder, f'{prefix}_msd_H_{T}K.txt'))
        if res is None:
            print(f'    T={T:>5} K: MSD file missing/unreadable — skipping')
            continue
        D, sig, R2 = res
        print(f'    T={T:>5} K: D={D:.4e} m²/s  σ={sig:.2e}  R²(MSD fit)={R2:.3f}')
        T_list.append(float(T)); D_list.append(D); E_list.append(sig); R2_list.append(R2)

    T_arr = np.array(T_list); D_arr = np.array(D_list); E_arr = np.array(E_list)
    valid = np.isfinite(D_arr) & (D_arr > 0) & np.isfinite(E_arr) & (E_arr > 0)
    if int(valid.sum()) < 2:
        print(f'[{stem}] fewer than 2 valid D(T) points ({int(valid.sum())}) — '
              f'cannot Arrhenius-fit; SKIPPING (no file written).')
        return None

    _valid_temps = [int(t) for t in T_arr[valid]]
    Ea, Ea_err, D0, D0_err, R2_arr = fit_arrhenius(T_arr[valid], D_arr[valid], E_arr[valid])
    print(f'[{stem}] Arrhenius over {int(valid.sum())} valid T ({_valid_temps}): '
          f'D0={D0:.4e} m²/s  Ea={Ea:.4f} eV  R²={R2_arr:.4f}')

    out_dir = os.path.join(WORK_DIR, 'results', f'{stem}_{N_H:02d}H')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'diffusivity_arrhenius.json')
    payload = {
        'D0_m2s':      float(D0),
        'E_D_eV':      float(Ea),
        'D0_err_m2s':  float(D0_err),
        'E_D_err_eV':  float(Ea_err),
        'R2':          float(R2_arr),
        'n_H':         N_H,
        'stem':        stem,
        'source':      f'old notebook10 {old_folder} MSD (fix-msd, single-origin)',
        'validation_grade': True,
        'note': ('Single-origin fix-msd data; low-T D dominated by noise. Fit uses '
                 f'T={_valid_temps} K (capped at 800 K, no high-T anchor), so it is even '
                 'lower-confidence than the pure-Ni conversion. Validation-grade, not a '
                 'production diffusivity.'),
        'temperatures_K':   [int(t) for t in T_arr.tolist()],
        'D_per_T_m2s':      {int(t): float(d) for t, d in zip(T_arr, D_arr)},
        'msd_fit_R2_per_T': {int(t): float(r) for t, r in zip(T_arr, R2_list)},
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[{stem}] wrote {out_json}')
    return out_json


def main():
    print(f'Reading old MSD data from {OLD_ROOT}')
    print(f'Output dirs: results/{{stem}}_{N_H:02d}H/  '
          f'(rename 01->1 to activate a material for permeation)\n')

    written, skipped = [], []
    for old_folder, prefix, stem in MATERIALS:
        out = _convert_one(old_folder, prefix, stem)
        (written if out else skipped).append(stem)
        print()

    print('=' * 76)
    print(f'Wrote diffusivity_arrhenius.json for {len(written)} material(s): {written}')
    if skipped:
        print(f'Skipped {len(skipped)} material(s): {skipped}')
    print('\nPure Ni is intentionally excluded (already set up as Ni_supercell_1H).')
    print('Pipeline stems with NO old MSD data (not produced): '
          'Al_supercell, Fe_supercell, Cr_oxide_supercell, Ni_oxide_supercell, bestsqs3.')
    print('\nTo run a material\'s permeation, rename results/{stem}_01H -> {stem}_1H first.')
    if not written:
        sys.exit(1)


if __name__ == '__main__':
    main()

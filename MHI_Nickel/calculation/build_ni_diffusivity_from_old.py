#!/usr/bin/env python3
"""
build_ni_diffusivity_from_old.py
================================
Convert the OLD notebook-10 bulk-equilibration MSD dataset for pure Ni into the
``diffusivity_arrhenius.json`` the permeation orchestrator (Part 2) reads for
Part 3 — so the Ni permeation pipeline can be run end-to-end *now*, reusing
pre-existing diffusivity data instead of waiting on a fresh Part-3 MD run.

What it does
------------
For each ``Pure_Nickel_msd_H_{T}K.txt`` (raw LAMMPS ``fix msd`` output:
columns ``TimeStep, MSD[Å²]``), converts steps→ps (dt = 0.0005 ps, metal units),
fits D(T) with the pipeline's own ``fit_diffusivity`` (Einstein D = slope/6),
then Arrhenius-fits D(T) with ``fit_arrhenius`` and writes
``results/{STEM}_1H/diffusivity_arrhenius.json`` with ``D0_m2s`` / ``E_D_eV``
(the keys ``resolve_nh_diffusivity`` requires).

The old run is 500 Ni + 1 H → the DILUTE (n_H=1) case, so it maps to
``{STEM}_1H`` only; the permeation run processes n_H=1 and skips 3/5/10 (no
diffusivity supplied for those). ``lattice_params_vs_T.json`` is NOT written —
it is optional, and the orchestrator falls back to the fixed A0_M lattice
constant when it is absent (thermal expansion is a second-order effect here).

CAVEAT — validation-grade, not production
-----------------------------------------
The old MSD is single-origin (``fix msd``), so at low T (300–600 K), where H
barely diffuses in 0.75 ns, the per-T slope is dominated by noise (R² ≈ 0);
only ~1000–1200 K give a reliable slope. The Arrhenius fit (inverse-variance
weighted) is therefore driven by the high-T points, and D(400–800 K) used by
the permeation run is an EXTRAPOLATION. Use the resulting permeability to prove
the end-to-end chain and check magnitudes are physically plausible — not as a
production diffusivity/permeability number.

Usage
-----
    python build_ni_diffusivity_from_old.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import BASE_DIR
from models.diffusivity_post_processing import fit_diffusivity, fit_arrhenius

WORK_DIR = os.path.join(BASE_DIR, 'calculation')

STEM     = 'Ni_supercell'
N_H      = 1                       # old run: 500 Ni + 1 H (dilute)
DT_PS    = 0.0005                  # metal-units timestep (0.5 fs), from the equil log
OLD_DIR  = os.path.join(BASE_DIR, 'old_notebooks', 'results',
                        'notebook10-bulk-equilibration', 'Pure_Nickel')
MSD_TEMPLATE = 'Pure_Nickel_msd_H_{T}K.txt'
TEMPS_K  = [300, 400, 600, 800, 1000, 1200]
FIT_WINDOW = (0.2, 0.8)            # skip ballistic start / noisy tail


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


def main():
    T_list, D_list, E_list, R2_list = [], [], [], []
    print(f'Reading old MSD data from {OLD_DIR}')
    for T in TEMPS_K:
        res = _d_of_t(os.path.join(OLD_DIR, MSD_TEMPLATE.format(T=T)))
        if res is None:
            print(f'  T={T:>5} K: MSD file missing/unreadable — skipping')
            continue
        D, sig, R2 = res
        print(f'  T={T:>5} K: D={D:.4e} m²/s  σ={sig:.2e}  R²(MSD fit)={R2:.3f}')
        T_list.append(float(T)); D_list.append(D); E_list.append(sig); R2_list.append(R2)

    T_arr = np.array(T_list); D_arr = np.array(D_list); E_arr = np.array(E_list)
    valid = np.isfinite(D_arr) & (D_arr > 0) & np.isfinite(E_arr) & (E_arr > 0)
    if int(valid.sum()) < 2:
        print('FATAL: fewer than 2 valid D(T) points — cannot Arrhenius-fit.')
        sys.exit(1)

    Ea, Ea_err, D0, D0_err, R2_arr = fit_arrhenius(T_arr[valid], D_arr[valid], E_arr[valid])
    print(f'\nArrhenius fit over {int(valid.sum())} valid T '
          f'({[int(t) for t in T_arr[valid]]}):')
    print(f'  D0 = {D0:.4e} ± {D0_err:.2e} m²/s')
    print(f'  Ea = {Ea:.4f} ± {Ea_err:.4f} eV')
    print(f'  R² = {R2_arr:.4f}')

    out_dir = os.path.join(WORK_DIR, 'results', f'{STEM}_{N_H}H')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'diffusivity_arrhenius.json')
    payload = {
        'D0_m2s':      float(D0),
        'E_D_eV':      float(Ea),
        'D0_err_m2s':  float(D0_err),
        'E_D_err_eV':  float(Ea_err),
        'R2':          float(R2_arr),
        'n_H':         N_H,
        'stem':        STEM,
        'source':      'old notebook10 Pure_Nickel MSD (fix-msd, single-origin)',
        'validation_grade': True,
        'note': ('Single-origin fix-msd data; low-T D dominated by noise, '
                 'Arrhenius fit driven by ~1000-1200 K, D(400-800 K) is an '
                 'extrapolation. Validation-grade, not a production diffusivity.'),
        'temperatures_K':  [int(t) for t in T_arr.tolist()],
        'D_per_T_m2s':     {int(t): float(d) for t, d in zip(T_arr, D_arr)},
        'msd_fit_R2_per_T': {int(t): float(r) for t, r in zip(T_arr, R2_list)},
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nWrote: {out_json}')
    print(f'Ni now has Part 3 (n_H=1). Regenerate + run the permeation script:')
    print(f'  python regenerate_permeation_scripts.py')
    print(f'  python wrap_permeation_runs_west.py && sbatch slurm_permeation_run_{STEM}.sh')


if __name__ == '__main__':
    main()

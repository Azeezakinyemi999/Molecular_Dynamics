"""
models/tst_rates.py
===================
Transition-state theory rate constants from NEB barriers + vibrational frequencies.

Pipeline
--------
1. ``collect_neb_results``  — load parse_barrier_file output for a batch of jobs
2. ``split_vib_results``    — split orchestrate_vibrations() output into IS / TS dicts
3. ``apply_zpe_correction`` — correct a barrier for zero-point energy
4. ``vineyard_prefactor``   — Vineyard (1957) attempt frequency from mode products
5. ``arrhenius_rate``       — k = ν × exp(−ΔE / kB T)
6. ``build_rate_dict``      — assemble all rates keyed by NEB label
7. ``rates_to_json``        — serialise to JSON for the KMC engine

Label convention
----------------
NEB labels take the form ``'{hop}_{sid}'``, e.g. ``'hopa_Ni3Mo'``.
Vibration labels returned by ``orchestrate_vibrations`` use the same base with
``_IS`` / ``_TS`` appended.  ``split_vib_results`` strips those suffixes.

ZPE approximation for reverse barrier
--------------------------------------
The reverse ZPE correction uses ZPE_FS ≈ ZPE_IS (same local host environment).
This means ΔZPE_rev = ZPE_TS − ZPE_IS = ΔZPE_fwd.  The approximation is
reasonable for hops between similar octahedral sites; it breaks down for large
IS/FS asymmetry.  Pass ``apply_zpe=False`` to disable.
"""

from __future__ import annotations

import json
import math
import os
import warnings

from models.vibrations import load_vibration_results

# ── Physical constants ─────────────────────────────────────────────────────────
BOLTZMANN_eV     = 8.617333262e-5   # eV / K
CM1_TO_EV        = 1.2398419843e-4  # eV per cm^-1
SPEED_LIGHT_CM_S = 2.99792458e10    # cm / s  (c in CGS units)


# ---------------------------------------------------------------------------
# Section 1 — Collect NEB results from job dicts
# ---------------------------------------------------------------------------

def collect_neb_results(neb_jobs: list, hop: str = 'hopa') -> dict:
    """Load ``parse_barrier_file`` output for a batch of NEB job dicts.

    Parameters
    ----------
    neb_jobs : list of dict
        Job dicts from ``orchestrate_hopa_neb`` or ``orchestrate_hopb_neb``.
        Required key: ``sid``, ``barrier_file``.
    hop : str
        Hop prefix used to form labels — ``'hopa'`` or ``'hopb'``.

    Returns
    -------
    dict
        ``{'{hop}_{sid}': barrier_dict}`` for every job with a readable
        barrier file.  Jobs with missing or unreadable files emit warnings.
    """
    from models.parsers import parse_barrier_file

    results: dict = {}
    for job in neb_jobs:
        sid    = job['sid']
        label  = f'{hop}_{sid}'
        bf     = job.get('barrier_file', '')
        if not bf or not os.path.exists(bf):
            warnings.warn(f'[{label}] barrier file not found: {bf!r}; skipping.')
            continue
        try:
            bd = parse_barrier_file(bf)
            if not bd.get('converged', False):
                warnings.warn(f'[{label}] NEB not converged; including results anyway.')
            results[label] = bd
        except Exception as exc:
            warnings.warn(f'[{label}] parse_barrier_file failed: {exc}; skipping.')

    print(f'[NEB results] loaded {len(results)}/{len(neb_jobs)} jobs  (hop={hop})')
    return results


# ---------------------------------------------------------------------------
# Section 2 — Split vib results into IS / TS dicts
# ---------------------------------------------------------------------------

def split_vib_results(vib_dict: dict) -> tuple[dict, dict]:
    """Split ``orchestrate_vibrations()`` output into IS and TS dicts.

    ``orchestrate_vibrations`` returns keys like ``'hopa_Ni3Mo_IS'`` and
    ``'hopa_Ni3Mo_TS'``.  This function strips the suffix and returns two
    separate dicts keyed by the base label.

    Parameters
    ----------
    vib_dict : dict
        Output of ``orchestrate_vibrations`` — ``{label_IS/TS: info_dict}``.
        Each info dict must have a ``'vib_json'`` key.

    Returns
    -------
    vib_is : dict
        ``{base_label: vib_json_path}`` for IS structures.
    vib_ts : dict
        ``{base_label: vib_json_path}`` for TS structures.
    """
    vib_is: dict = {}
    vib_ts: dict = {}
    for key, info in vib_dict.items():
        path = info['vib_json']
        if key.endswith('_IS'):
            vib_is[key[:-3]] = path
        elif key.endswith('_TS'):
            vib_ts[key[:-3]] = path
        elif key.endswith('_FS'):
            # Dissolved-H FS modes are consumed by the solubility prefactor
            # (see split_vib_fs), not by the IS->TS barrier ZPE correction.
            continue
        else:
            warnings.warn(
                f'split_vib_results: unexpected key {key!r} (expected _IS/_TS/_FS suffix); '
                'skipping.'
            )
    return vib_is, vib_ts


def split_vib_fs(vib_dict: dict) -> dict:
    """Extract the FS (dissolved-H) vibration results from ``orchestrate_vibrations`` output.

    Companion to :func:`split_vib_results`. Returns ``{base_label: vib_json_path}``
    for every ``'..._FS'`` entry — the dissolved-H octahedral-cage modes that the
    vibrational solubility prefactor (Part 4) needs. Entries without an ``_FS``
    suffix are ignored.
    """
    vib_fs: dict = {}
    for key, info in vib_dict.items():
        if key.endswith('_FS'):
            vib_fs[key[:-3]] = info['vib_json']
    return vib_fs


# ---------------------------------------------------------------------------
# Section 3 — ZPE correction
# ---------------------------------------------------------------------------

def apply_zpe_correction(
    E_barrier_eV: float,
    freqs_is_cm1: list,
    freqs_ts_cm1: list,
    min_freq_cm1: float = 50.0,
) -> float:
    """Return a ZPE-corrected barrier height.

    .. math::

        \\Delta E_{\\text{zpe}} = E_{\\text{barrier}} + (\\text{ZPE}_{\\text{TS}} - \\text{ZPE}_{\\text{IS}})

        \\text{ZPE} = \\tfrac{1}{2} \\sum_{i} \\nu_i \\times C_{\\text{cm}^{-1}\\to\\text{eV}}

    Only real modes above ``min_freq_cm1`` contribute.  The imaginary TS mode
    must already be absent from ``freqs_ts_cm1`` (``vib_frequencies.json``'s
    ``frequencies_real_cm1`` key excludes it automatically).

    Parameters
    ----------
    E_barrier_eV : float
        Raw NEB barrier (E_abs or E_des), in eV.
    freqs_is_cm1 : list of float
        Real-mode frequencies of the IS structure (cm^-1).
    freqs_ts_cm1 : list of float
        Real-mode frequencies of the TS structure (cm^-1).
    min_freq_cm1 : float
        Modes below this value are excluded from ZPE sums.

    Returns
    -------
    float
        ZPE-corrected barrier in eV.
    """
    zpe_is = 0.5 * sum(f * CM1_TO_EV for f in freqs_is_cm1 if f >= min_freq_cm1)
    zpe_ts = 0.5 * sum(f * CM1_TO_EV for f in freqs_ts_cm1 if f >= min_freq_cm1)
    return E_barrier_eV + (zpe_ts - zpe_is)


# ---------------------------------------------------------------------------
# Section 4 — Vineyard prefactor
# ---------------------------------------------------------------------------

def vineyard_prefactor(
    freqs_is_cm1: list,
    freqs_ts_cm1: list,
    min_freq_cm1: float = 50.0,
) -> float:
    """Compute the Vineyard (1957) attempt frequency in s⁻¹.

    .. math::

        \\nu^* = c \\cdot \\frac{\\prod_i \\nu^{\\text{IS}}_i}{\\prod_j \\nu^{\\text{TS}}_j}

    where *c* = 2.998 × 10¹⁰ cm/s converts the remaining unpaired cm⁻¹ to s⁻¹.
    The extra factor of *c* arises because the IS product has one more frequency
    than the TS product (the imaginary mode is excluded from ``freqs_ts_cm1``).

    The product is evaluated in log-space for numerical stability.

    Parameters
    ----------
    freqs_is_cm1 : list of float
        Real IS frequencies (cm^-1).  Pass ``frequencies_real_cm1`` from
        ``vib_frequencies.json``.
    freqs_ts_cm1 : list of float
        Real TS frequencies (cm^-1), imaginary mode excluded.  Pass
        ``frequencies_real_cm1`` from the TS ``vib_frequencies.json``.
    min_freq_cm1 : float
        Modes below this threshold are excluded from both products with a
        warning.  Prevents numerical issues from near-zero modes introduced
        by the partial Hessian.  Default 50 cm^-1.

    Returns
    -------
    float
        Attempt frequency in s⁻¹.  Typical values: 10¹² – 10¹³ s⁻¹.

    Raises
    ------
    ValueError
        If no valid frequencies remain after applying the threshold.
    """
    is_valid = [f for f in freqs_is_cm1 if f >= min_freq_cm1]
    ts_valid = [f for f in freqs_ts_cm1 if f >= min_freq_cm1]

    n_skip_is = len(freqs_is_cm1) - len(is_valid)
    n_skip_ts = len(freqs_ts_cm1) - len(ts_valid)
    if n_skip_is:
        warnings.warn(
            f'vineyard_prefactor: excluded {n_skip_is} IS mode(s) below '
            f'{min_freq_cm1} cm^-1.'
        )
    if n_skip_ts:
        warnings.warn(
            f'vineyard_prefactor: excluded {n_skip_ts} TS mode(s) below '
            f'{min_freq_cm1} cm^-1.'
        )
    if not is_valid:
        raise ValueError('No valid IS frequencies above threshold.')
    if not ts_valid:
        raise ValueError('No valid TS frequencies above threshold.')

    log_nu = (
        math.log(SPEED_LIGHT_CM_S)
        + sum(math.log(f) for f in is_valid)
        - sum(math.log(f) for f in ts_valid)
    )
    return math.exp(log_nu)


# ---------------------------------------------------------------------------
# Section 5 — Arrhenius rate
# ---------------------------------------------------------------------------

def arrhenius_rate(nu_s1: float, delta_e_eV: float, T_K: float) -> float:
    """Compute k = ν × exp(−ΔE / k_B T).

    Parameters
    ----------
    nu_s1 : float
        Attempt frequency in s⁻¹ (e.g. from :func:`vineyard_prefactor`).
    delta_e_eV : float
        Activation barrier in eV.
    T_K : float
        Temperature in K.

    Returns
    -------
    float
        Rate constant in s⁻¹.
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    return nu_s1 * math.exp(-delta_e_eV / (BOLTZMANN_eV * T_K))


# ---------------------------------------------------------------------------
# Section 6 — Rate dict assembler
# ---------------------------------------------------------------------------

def build_rate_dict(
    neb_results: dict,
    vib_results_is: dict,
    vib_results_ts: dict,
    T_K: float,
    apply_zpe: bool = True,
    min_freq_cm1: float = 50.0,
) -> dict:
    """Assemble rate constants for all NEB labels.

    For each label present in *neb_results*:

    1. Load IS and TS ``vib_frequencies.json`` files.
    2. Compute the Vineyard prefactor ν.
    3. Optionally apply ZPE corrections to E_abs (forward) and E_des (reverse).
    4. Compute k_forward and k_reverse via the Arrhenius expression.

    The reverse ZPE correction uses the approximation ZPE_FS ≈ ZPE_IS (see
    module docstring for details).

    Parameters
    ----------
    neb_results : dict
        ``{label: barrier_dict}`` from :func:`collect_neb_results`.
        ``barrier_dict`` must contain ``'E_abs'`` and ``'E_des'`` (eV).
    vib_results_is : dict
        ``{label: vib_json_path}`` for IS structures.
        Build with :func:`split_vib_results`.
    vib_results_ts : dict
        ``{label: vib_json_path}`` for TS structures.
    T_K : float
        Temperature in K.
    apply_zpe : bool
        Apply ZPE correction to barriers.  Default ``True``.
    min_freq_cm1 : float
        Frequency threshold passed to :func:`vineyard_prefactor` and
        :func:`apply_zpe_correction`.

    Returns
    -------
    dict
        ``{label: {k_forward, k_reverse, Ea_raw, Ea_zpe, Ed_raw, Ed_zpe,
                   nu, delta_e, T_K}}``
        Rates in s⁻¹, barriers in eV.
    """
    rate_dict: dict = {}
    skipped: list  = []

    for label, neb in neb_results.items():
        if label not in vib_results_is:
            skipped.append(f'{label} (IS vib missing)')
            continue
        if label not in vib_results_ts:
            skipped.append(f'{label} (TS vib missing)')
            continue

        try:
            is_vib = load_vibration_results(vib_results_is[label])
            ts_vib = load_vibration_results(vib_results_ts[label])
        except FileNotFoundError as exc:
            skipped.append(f'{label} ({exc})')
            continue

        is_imag = is_vib.get('frequencies_imag_cm1', [])
        ts_imag = ts_vib.get('frequencies_imag_cm1', [])
        if len(is_imag) != 0:
            warnings.warn(f'[{label}] IS has {len(is_imag)} imaginary mode(s) — expected 0.')
        if len(ts_imag) != 1:
            warnings.warn(f'[{label}] TS has {len(ts_imag)} imaginary mode(s) — expected 1.')

        is_freqs = is_vib['frequencies_real_cm1']
        ts_freqs = ts_vib['frequencies_real_cm1']  # imaginary mode already excluded

        Ea_raw = float(neb['E_abs'])
        Ed_raw = float(neb.get('E_des', 0.0))

        try:
            nu = vineyard_prefactor(is_freqs, ts_freqs, min_freq_cm1=min_freq_cm1)
        except (ValueError, Exception) as exc:
            skipped.append(f'{label} (Vineyard failed: {exc})')
            continue

        kw = dict(min_freq_cm1=min_freq_cm1)
        if apply_zpe:
            Ea_use = apply_zpe_correction(Ea_raw, is_freqs, ts_freqs, **kw)
            Ed_use = apply_zpe_correction(Ed_raw, is_freqs, ts_freqs, **kw)
        else:
            Ea_use = Ea_raw
            Ed_use = Ed_raw

        _k_fwd = arrhenius_rate(nu, Ea_use, T_K)
        _k_rev = arrhenius_rate(nu, Ed_use, T_K)
        rate_dict[label] = {
            'k_forward': _k_fwd,
            'k_reverse': _k_rev,
            'Ea_raw':    Ea_raw,
            'Ea_zpe':    Ea_use,
            'Ed_raw':    Ed_raw,
            'Ed_zpe':    Ed_use,
            'nu':        nu,
            'delta_e':   float(neb.get('delta_E', float('nan'))),
            'T_K':       T_K,
        }
        print(f'  {label:<24s}  k_fwd={_k_fwd:.3e}  k_rev={_k_rev:.3e}  Ea_zpe={Ea_use:.3f} eV  ν={nu:.2e} s⁻¹')

    if skipped:
        warnings.warn(
            f'build_rate_dict: skipped {len(skipped)} label(s):\n  '
            + '\n  '.join(skipped)
        )

    print(f'[rate_dict] {len(rate_dict)} labels at T={T_K:.0f} K  (skipped={len(skipped)})')
    return rate_dict


# ---------------------------------------------------------------------------
# Section 7 — Serialisation
# ---------------------------------------------------------------------------

def rates_to_json(rate_dict: dict, out_path: str) -> str:
    """Write ``rate_dict`` to a JSON file for consumption by the KMC engine.

    Parameters
    ----------
    rate_dict : dict
        Output of :func:`build_rate_dict`.
    out_path : str
        Destination path for the JSON file.

    Returns
    -------
    str
        Path to the written file.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(rate_dict, fh, indent=2)
    print(f'Wrote: {out_path}  ({len(rate_dict)} labels)')
    return out_path

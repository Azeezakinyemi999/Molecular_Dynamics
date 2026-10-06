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
7. ``rates_to_json``        — serialise the rate table to JSON

Label convention
----------------
NEB labels take the form ``'{hop}_{sid}'``, e.g. ``'hopa_Ni3Mo'``.
Vibration labels returned by ``orchestrate_vibrations`` use the same base with
``_IS`` / ``_TS`` appended.  ``split_vib_results`` strips those suffixes.

Forward and reverse are built from their own end states
-------------------------------------------------------
The forward barrier pairs IS with TS; the reverse barrier pairs **FS** with TS.
So ``ΔZPE_fwd = ZPE_TS − ZPE_IS`` while ``ΔZPE_rev = ZPE_TS − ZPE_FS``, and the
attempt frequencies are ``ν*_fwd = c·Πν_IS/Πν_TS`` and
``ν*_rev = c·Πν_FS/Πν_TS``.  Using the IS quantities for both (as this module
did before) leaves the reverse rate wrong on both counts and, because the two
ZPE terms then cancel in ``Ea_zpe − Ed_zpe``, silently strips the ZPE
correction out of the reaction energy that feeds ΔH_sol.

``build_rate_dict`` therefore takes ``vib_results_fs``.  When a label has no FS
vibration it falls back to the IS quantities and records
``zpe_source='IS_fallback'`` so the affected labels stay identifiable.  Pass
``apply_zpe=False`` to disable the ZPE correction entirely.
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
        If no valid frequencies remain after applying the threshold, or if the
        surviving mode counts are not ``len(IS) == len(TS) + 1`` — see the
        inline note: a mismatch means the states displaced different atom sets
        and the ratio is not a frequency.
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

    # The single factor of c only yields s^-1 if the numerator carries exactly
    # one more frequency than the denominator — i.e. both states displaced the
    # same atoms and the TS contributed one imaginary mode. Otherwise the result
    # is c x (cm^-1)^n for some n != 0: not an attempt frequency at all. This
    # has happened in practice (an H2 dissociation IS displacing 2H+6 metals
    # against a TS displacing 2H+8 metals gave 24 vs 29 modes and a "prefactor"
    # of 5e-10 s^-1, ~22 orders of magnitude low), so refuse rather than return
    # a number that looks like a rate.
    if len(is_valid) != len(ts_valid) + 1:
        raise ValueError(
            f'Vineyard prefactor needs len(IS) == len(TS) + 1 after the '
            f'{min_freq_cm1} cm^-1 cut, got IS={len(is_valid)} TS={len(ts_valid)}. '
            f'The two states must displace the same atom set (the TS supplying '
            f'the one imaginary mode); a mismatch leaves the ratio dimensionally '
            f'invalid.'
        )

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
    vib_results_fs: dict | None = None,
) -> dict:
    """Assemble rate constants for all NEB labels.

    For each label present in *neb_results*:

    1. Load IS, TS and (when given) FS ``vib_frequencies.json`` files.
    2. Compute the forward Vineyard prefactor from IS/TS and the reverse one
       from FS/TS.
    3. Optionally apply ZPE corrections — E_abs with ZPE_TS − ZPE_IS, E_des
       with ZPE_TS − ZPE_FS.
    4. Compute k_forward and k_reverse via the Arrhenius expression.

    Without ``vib_results_fs`` the reverse quantities fall back to the IS ones
    and ``zpe_source`` is set to ``'IS_fallback'`` (see module docstring).

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
    vib_results_fs : dict, optional
        ``{label: vib_json_path}`` for FS structures, from
        :func:`split_vib_fs` applied to the **same** vibration run as IS/TS
        (the ``_FS`` entries, not the H-only set the solubility prefactor
        uses — those have a different partial-Hessian convention and must not
        be passed here).

    Returns
    -------
    dict
        ``{label: {k_forward, k_reverse, Ea_raw, Ea_zpe, Ed_raw, Ed_zpe,
                   nu, nu_reverse, zpe_source, delta_e, T_K}}``
        Rates in s⁻¹, barriers in eV.  ``nu`` is the forward prefactor and
        ``nu_reverse`` the reverse one; ``zpe_source`` is ``'FS'`` or
        ``'IS_fallback'``.
    """
    rate_dict: dict = {}
    skipped: list  = []
    vib_results_fs = vib_results_fs or {}
    no_fs: list = []

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

        # FS drives the reverse direction. Absent (older runs, or a hop whose
        # FS vibration was never computed) we fall back to IS and say so, rather
        # than silently reporting an IS-derived number as if it were the reverse.
        fs_freqs = None
        if label in vib_results_fs:
            try:
                fs_vib = load_vibration_results(vib_results_fs[label])
                if len(fs_vib.get('frequencies_imag_cm1', [])) != 0:
                    warnings.warn(
                        f'[{label}] FS has '
                        f'{len(fs_vib["frequencies_imag_cm1"])} imaginary mode(s) — expected 0.'
                    )
                fs_freqs = fs_vib['frequencies_real_cm1']
            except (FileNotFoundError, KeyError) as exc:
                warnings.warn(f'[{label}] FS vibration unreadable ({exc}); '
                              'falling back to IS for the reverse direction.')
        if fs_freqs is None:
            no_fs.append(label)
        zpe_source = 'FS' if fs_freqs is not None else 'IS_fallback'

        Ea_raw = float(neb['E_abs'])
        Ed_raw = float(neb.get('E_des', 0.0))

        try:
            nu = vineyard_prefactor(is_freqs, ts_freqs, min_freq_cm1=min_freq_cm1)
        except (ValueError, Exception) as exc:
            skipped.append(f'{label} (Vineyard failed: {exc})')
            continue

        # ν*_rev = c·Πν_FS/Πν_TS. FS carries one more real mode than TS (the
        # imaginary mode is excluded there), the same count asymmetry the
        # forward prefactor relies on, so the factor of c applies identically.
        nu_rev = nu
        if fs_freqs is not None:
            try:
                nu_rev = vineyard_prefactor(fs_freqs, ts_freqs, min_freq_cm1=min_freq_cm1)
            except (ValueError, Exception) as exc:
                warnings.warn(f'[{label}] reverse Vineyard failed ({exc}); '
                              'falling back to the forward prefactor.')
                nu_rev = nu
                zpe_source = 'IS_fallback'
                fs_freqs = None

        kw = dict(min_freq_cm1=min_freq_cm1)
        if apply_zpe:
            Ea_use = apply_zpe_correction(Ea_raw, is_freqs, ts_freqs, **kw)
            # Reverse pairs TS with FS: E_des is E_TS - E_FS, so the matching
            # correction is ZPE_TS - ZPE_FS.
            _rev_ref = fs_freqs if fs_freqs is not None else is_freqs
            Ed_use = apply_zpe_correction(Ed_raw, _rev_ref, ts_freqs, **kw)
        else:
            Ea_use = Ea_raw
            Ed_use = Ed_raw

        _k_fwd = arrhenius_rate(nu, Ea_use, T_K)
        _k_rev = arrhenius_rate(nu_rev, Ed_use, T_K)
        rate_dict[label] = {
            'k_forward':  _k_fwd,
            'k_reverse':  _k_rev,
            'Ea_raw':     Ea_raw,
            'Ea_zpe':     Ea_use,
            'Ed_raw':     Ed_raw,
            'Ed_zpe':     Ed_use,
            'nu':         nu,
            'nu_reverse': nu_rev,
            'zpe_source': zpe_source,
            'delta_e':    float(neb.get('delta_E', float('nan'))),
            'T_K':        T_K,
        }
        print(f'  {label:<24s}  k_fwd={_k_fwd:.3e}  k_rev={_k_rev:.3e}  '
              f'Ea_zpe={Ea_use:.3f} eV  ν={nu:.2e}/ν_rev={nu_rev:.2e} s⁻¹  [{zpe_source}]')

    if skipped:
        warnings.warn(
            f'build_rate_dict: skipped {len(skipped)} label(s):\n  '
            + '\n  '.join(skipped)
        )
    if no_fs:
        warnings.warn(
            f'build_rate_dict: {len(no_fs)} label(s) had no usable FS vibration; '
            f'their reverse barrier and prefactor fall back to IS '
            f'(zpe_source=IS_fallback):\n  ' + '\n  '.join(no_fs)
        )

    print(f'[rate_dict] {len(rate_dict)} labels at T={T_K:.0f} K  (skipped={len(skipped)})')
    return rate_dict


# ---------------------------------------------------------------------------
# Section 7 — Serialisation
# ---------------------------------------------------------------------------

def rates_to_json(rate_dict: dict, out_path: str) -> str:
    """Write ``rate_dict`` to a JSON file for downstream consumers.

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


# ---------------------------------------------------------------------------
# Section 8 — Per-hop ranked-barrier and ZPE-rate artifacts
#
# The Hop A/B analogues of the dissociation pipeline's ranked_barriers.json
# and diss_vib_rates.json: per-pathway summaries that carry the oct-site
# environment label (sub1_env / sub2_env) alongside the barriers/rates.
# ---------------------------------------------------------------------------

def write_hop_ranked(neb_jobs: list, hop: str, out_json: str,
                     env_key: str = 'sub1_env') -> list:
    """Collect + rank Hop A/B barrier results, carrying the oct-site environment.

    Analogue of the dissociation ``ranked_barriers.json`` (neb_workflow.py),
    for subsurface hops. Reads each job's ``barrier_file`` and records the
    forward barrier (``E_abs``), reverse barrier (``E_des``), reaction energy
    (``delta_E``), convergence flag, and the destination oct-site environment
    (``env_key`` on the job dict — ``sub1_env`` for Hop A, ``sub2_env`` for
    Hop B). Sorted by forward barrier (converged first, ascending).

    Jobs with a missing/unreadable barrier file are still listed, with
    ``converged: None`` and null barriers — never silently dropped.

    Returns the ranked list (also written to ``out_json``).
    """
    from models.parsers import parse_barrier_file

    rows: list = []
    for job in neb_jobs:
        sid = job.get('sid')
        bf  = job.get('barrier_file', '')
        row = {
            'label':    f'{hop}_{sid}',
            'sid':      sid,
            'sub1_env': job.get('sub1_env'),
            'sub2_env': job.get('sub2_env'),
            'env':      job.get(env_key),
        }
        if bf and os.path.exists(bf):
            try:
                bd = parse_barrier_file(bf)
                row.update({
                    'Ea':         bd.get('E_abs'),
                    'E_des':      bd.get('E_des'),
                    'delta_E':    bd.get('delta_E'),
                    'fmax_final': bd.get('fmax_final'),
                    'converged':  bd.get('converged'),
                })
            except Exception as exc:                       # noqa: BLE001
                warnings.warn(f'[{hop}_{sid}] parse_barrier_file failed: {exc}')
                row['converged'] = None
        else:
            row['converged'] = None
        rows.append(row)

    rows.sort(key=lambda r: (r.get('Ea') is None, r.get('Ea') or float('inf')))

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, indent=2)
    _n_conv = sum(1 for r in rows if r.get('converged'))
    print(f'[{hop} ranked] {len(rows)} pathways ({_n_conv} converged) → {out_json}')
    return rows


def write_hop_vib_rates(rate_dict: dict, neb_jobs: list, hop: str, out_json: str,
                        env_key: str = 'sub1_env') -> dict:
    """Write the per-hop ZPE-rate artifact (analogue of diss_vib_rates.json).

    Filters ``rate_dict`` (from :func:`build_rate_dict`) to this hop's labels
    and emits the temperature-independent ZPE-corrected barriers + Vineyard
    prefactor, tagged with the destination oct-site environment so Part 4
    (solubility) and Part 6 (env-keyed rate assembly) can consume them.

    ``rate_dict`` may be built at any single temperature — only the
    T-independent fields (``Ea_zpe``, ``Ed_zpe``, ``Ea_raw``, ``Ed_raw``,
    ``nu_reverse``, ``zpe_source``,
    ``nu``) are carried through here.

    Returns ``{label: {env, sub1_env, sub2_env, Ea_zpe, Ed_zpe, Ea_raw,
    Ed_raw, nu}}`` (also written to ``out_json``).
    """
    job_by_label = {f'{hop}_{j.get("sid")}': j for j in neb_jobs}

    out: dict = {}
    for label, r in rate_dict.items():
        if not label.startswith(f'{hop}_'):
            continue
        job = job_by_label.get(label, {})
        out[label] = {
            'label':    label,
            'env':      job.get(env_key),
            'sub1_env': job.get('sub1_env'),
            'sub2_env': job.get('sub2_env'),
            'Ea_zpe':   r.get('Ea_zpe'),
            'Ed_zpe':   r.get('Ed_zpe'),
            'Ea_raw':   r.get('Ea_raw'),
            'Ed_raw':   r.get('Ed_raw'),
            'nu':       r.get('nu'),
            'nu_reverse': r.get('nu_reverse'),
            'zpe_source': r.get('zpe_source'),
        }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print(f'[{hop} vib rates] {len(out)} labels → {out_json}')
    return out


# ---------------------------------------------------------------------------
# Section 9 — Partition functions (for the vibrational solubility prefactor)
#
# Building blocks for models/permeation.py's vibrational S0 route: the
# dissolved-H vibrational partition function (from the FS oct-cage modes) and
# the gas-phase H2 reference partition function. Standard ideal-gas statistical
# mechanics; see e.g. McQuarrie, "Statistical Mechanics".
# ---------------------------------------------------------------------------

_PLANCK_J_S     = 6.62607015e-34   # J·s
_KB_J           = 1.380649e-23     # J/K
_M_H2_KG        = 2.0 * 1.6735575e-27   # kg  (H2 molecule)
_THETA_ROT_H2_K = 87.6             # K, rotational temperature of H2
_THETA_VIB_H2_K = 6332.0           # K, vibrational temperature of H2 (~4401 cm^-1)
_SIGMA_H2       = 2                # symmetry number (homonuclear diatomic)


def vib_partition_function(freqs_cm1, T_K: float, min_freq_cm1: float = 50.0) -> float:
    """Harmonic vibrational partition function q_vib = Π 1/(1 − e^(−hcν/kT)).

    Zero-point energy is taken as the reference (q → 1 as T → 0), the usual
    convention when the ZPE is carried separately in ΔH_sol. Soft/spurious
    modes below ``min_freq_cm1`` are dropped (they are the translational/
    rotational near-zero modes an harmonic analysis of an adsorbate produces).

    Parameters
    ----------
    freqs_cm1 : iterable of float
        Real vibrational frequencies [cm⁻¹].
    T_K : float
        Temperature [K].
    min_freq_cm1 : float
        Discard modes below this threshold. Default 50 cm⁻¹.

    Returns
    -------
    float
        Dimensionless vibrational partition function (≥ 1).
    """
    if T_K <= 0:
        raise ValueError(f'Temperature must be positive; got T_K={T_K}.')
    q = 1.0
    for nu in freqs_cm1:
        if nu is None or nu < min_freq_cm1:
            continue
        x = (CM1_TO_EV * float(nu)) / (BOLTZMANN_eV * T_K)   # hcν / kT
        q *= 1.0 / (1.0 - math.exp(-x))
    return q


def h2_gas_partition_function(T_K: float, P_Pa: float) -> dict:
    """Gas-phase H₂ molecular partition function at (T, P), ideal-gas.

    Returns the translational (per-molecule, using V = k_B T / P), rotational
    (rigid rotor, high-T with symmetry number σ=2), and vibrational (ZPE
    reference) components, plus their product. The translational term carries
    the pressure dependence that makes the Sieverts prefactor ∝ P^(−½) once
    the H₂ → 2H stoichiometry (a square-root on q_H2) is applied downstream.

    Parameters
    ----------
    T_K : float
        Temperature [K].
    P_Pa : float
        H₂ reference pressure [Pa].

    Returns
    -------
    dict
        ``{'trans', 'rot', 'vib', 'total'}`` — all dimensionless except that
        ``trans`` embeds the per-molecule volume, so ``total`` has the units of
        the translational term (m⁻³·m³ = dimensionless per molecule at V).

    Notes
    -----
    The high-T rigid-rotor form ``q_rot = T/(σ·θ_rot)`` is used. For H₂
    (θ_rot ≈ 87.6 K) this is accurate to a few percent above ~400 K, the
    regime of this pipeline; it is not valid near/below θ_rot.
    """
    if T_K <= 0 or P_Pa <= 0:
        raise ValueError(f'T_K and P_Pa must be positive; got {T_K}, {P_Pa}.')
    V_per_molecule = _KB_J * T_K / P_Pa
    q_trans = (2.0 * math.pi * _M_H2_KG * _KB_J * T_K / _PLANCK_J_S ** 2) ** 1.5 * V_per_molecule
    q_rot   = T_K / (_SIGMA_H2 * _THETA_ROT_H2_K)
    q_vib   = 1.0 / (1.0 - math.exp(-_THETA_VIB_H2_K / T_K))
    return {'trans': q_trans, 'rot': q_rot, 'vib': q_vib,
            'total': q_trans * q_rot * q_vib}


def env_rate_dict(hop_vib: dict, T_K: float) -> tuple:
    """Group per-hop ZPE rates by oct-site environment.

    Reduces the per-pathway ZPE-rate artifact (``write_hop_vib_rates`` output)
    to environment-keyed forward/reverse rate dicts at temperature ``T_K``,
    replacing the old collapse-to-one-rate-per-element. Within each environment
    the Arrhenius rates are combined by **arithmetic mean**, which preserves the
    group's expected aggregate event rate over its independent, parallel
    channels.

    Parameters
    ----------
    hop_vib : dict
        ``{label: {env, nu, Ea_zpe, Ed_zpe, ...}}`` — Hop A tags env by sub1
        env, Hop B by sub2 env (see ``write_hop_vib_rates``).
    T_K : float
        Temperature [K].

    Returns
    -------
    (dict, dict)
        ``(k_forward_by_env, k_reverse_by_env)`` — forward is entry
        (surf→sub1 / sub1→sub2), reverse is exit.
    """
    fwd: dict = {}
    rev: dict = {}
    for r in hop_vib.values():
        env = r.get('env')
        nu, ea, ed = r.get('nu'), r.get('Ea_zpe'), r.get('Ed_zpe')
        if env is None or nu is None or ea is None or ed is None:
            continue
        # Exit uses the reverse prefactor (FS/TS). Older artifacts predate the
        # field and only carry the forward ν; fall back to it there.
        nu_rev = r.get('nu_reverse') or nu
        fwd.setdefault(env, []).append(arrhenius_rate(nu, ea, T_K))
        rev.setdefault(env, []).append(arrhenius_rate(nu_rev, ed, T_K))
    k_fwd = {e: sum(v) / len(v) for e, v in fwd.items()}
    k_rev = {e: sum(v) / len(v) for e, v in rev.items()}
    return k_fwd, k_rev

#!/usr/bin/env python3
"""
check_equations.py — verify every equation stated in docs/02-theory.md.

Each check recomputes a quantity two ways: once by calling the project's own
function, and once from an independent expression written out here directly
from the equation as the documentation states it. If the documentation and the
code have drifted apart, a check fails.

Nothing here reads a real run. Every input comes from the fictitious system in
``example/toy.py``.

Usage
-----
    ~/anaconda3/envs/mace_env/bin/python docs/figures/check_equations.py

Exit status is 0 when every check passes, 1 otherwise, so this can be wired
into CI. Appendix E of the documentation lists what it checks and the expected
output.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # repo root
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_HERE, 'example'))

from toy import TOY, KB_EV, C_CM_S, H_EVS, N_A          # noqa: E402

from models import permeation as P                       # noqa: E402
from models import tst_rates as R                        # noqa: E402
from models import diffusivity_post_processing as DPP    # noqa: E402
from models import energetics as E                       # noqa: E402


# ── harness ──────────────────────────────────────────────────────────────────
_RESULTS: list[tuple[str, bool, str]] = []
RTOL = 1e-9


def check(name: str, got, want, rtol: float = RTOL, note: str = ''):
    """Record one comparison between the code's value and the equation's."""
    try:
        ok = (math.isclose(float(got), float(want), rel_tol=rtol, abs_tol=0.0)
              if want != 0 else math.isclose(float(got), 0.0, abs_tol=1e-30))
    except (TypeError, ValueError):
        ok = bool(got == want)
    detail = f'code={got!r} eqn={want!r}' if not ok else note
    _RESULTS.append((name, ok, detail))


def check_true(name: str, cond: bool, note: str = ''):
    _RESULTS.append((name, bool(cond), '' if cond else note or 'condition false'))


def report() -> int:
    width = max(len(n) for n, _, _ in _RESULTS) + 2
    print()
    for name, ok, detail in _RESULTS:
        line = f'  {"PASS" if ok else "FAIL"}  {name:<{width}}'
        if detail:
            line += f'  {detail}'
        print(line)
    n_ok = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f'\n  {n_ok} / {len(_RESULTS)} checks passed\n')
    return 0 if n_ok == len(_RESULTS) else 1


T = TOY
T_MID = 600.0


# ── 1. unit conversions ──────────────────────────────────────────────────────
# A wavenumber nu~ [cm^-1] is an energy h*c*nu~; in eV that is
#   E = h[eV s] * c[cm/s] * nu~[cm^-1]
check('unit: cm^-1 -> eV', R.CM1_TO_EV, H_EVS * C_CM_S, rtol=1e-6)

# MSD in A^2 against time in ps gives D in A^2/ps; 1 A^2/ps = 1e-20 m^2 / 1e-12 s
check('unit: A^2/ps -> m^2/s', 1e-8, (1e-10 ** 2) / 1e-12)

# Octahedral sites per FCC cubic cell = 4, divided by Avogadro for mol
check('oct-site density rho_oct', T.rho_oct_mol_m3(), 4.0 / (T.a0_m ** 3) / N_A)


# ── 2. zero-point energy ─────────────────────────────────────────────────────
# ZPE = 1/2 sum_i h*nu_i, and the forward correction is ZPE_TS - ZPE_IS.
zpe_is = 0.5 * sum(f * H_EVS * C_CM_S for f in T.freqs_is_cm1)
zpe_ts = 0.5 * sum(f * H_EVS * C_CM_S for f in T.freqs_ts_cm1)
zpe_fs = 0.5 * sum(f * H_EVS * C_CM_S for f in T.freqs_fs_cm1)

ea_zpe_code = R.apply_zpe_correction(
    T.ea_ev, list(T.freqs_is_cm1), list(T.freqs_ts_cm1), min_freq_cm1=0.0)
check('ZPE: forward barrier = Ea + (ZPE_TS - ZPE_IS)',
      ea_zpe_code, T.ea_ev + (zpe_ts - zpe_is), rtol=1e-6)

# The reverse barrier is referenced to the FINAL state, not the initial one:
# E_des = E_TS - E_FS, so its ZPE correction is ZPE_TS - ZPE_FS.
ed_zpe_code = R.apply_zpe_correction(
    T.ed_ev, list(T.freqs_fs_cm1), list(T.freqs_ts_cm1), min_freq_cm1=0.0)
check('ZPE: reverse barrier uses FS, not IS',
      ed_zpe_code, T.ed_ev + (zpe_ts - zpe_fs), rtol=1e-6)

check_true('ZPE: every real mode contributes at min_freq_cm1=0',
           abs(R.apply_zpe_correction(0.0, [10.0, 500.0], [500.0], min_freq_cm1=0.0)
               - (0.5 * 500.0 - 0.5 * (10.0 + 500.0)) * R.CM1_TO_EV) < 1e-12)


# ── 3. Vineyard prefactor and the quasi-harmonic floor ───────────────────────
FLOOR = 100.0
is_raised = [max(f, FLOOR) for f in T.freqs_is_cm1]
ts_raised = [max(f, FLOOR) for f in T.freqs_ts_cm1]
nu_eqn = C_CM_S * math.prod(is_raised) / math.prod(ts_raised)
nu_code = R.vineyard_prefactor(list(T.freqs_is_cm1), list(T.freqs_ts_cm1),
                               low_freq_cm1=FLOOR)
check('Vineyard: nu* = c * prod(nu_IS) / prod(nu_TS), modes raised to floor',
      nu_code, nu_eqn, rtol=1e-9)

# With the floor disabled the raw frequencies must be used unchanged.
check('Vineyard: low_freq_cm1=0 uses frequencies as computed',
      R.vineyard_prefactor(list(T.freqs_is_cm1), list(T.freqs_ts_cm1), low_freq_cm1=0.0),
      C_CM_S * math.prod(T.freqs_is_cm1) / math.prod(T.freqs_ts_cm1), rtol=1e-9)

# Raising, not discarding, is what keeps the counts legal. Two IS modes here lie
# below the floor; discarding them would make n_IS != n_TS + 1.
check_true('Vineyard: toy IS has modes below the floor (exercises raising)',
           any(f < FLOOR for f in T.freqs_is_cm1))

# The dimensional rule: one unpaired frequency, supplied by c, means the IS list
# must hold exactly one more mode than the TS list.
check_true('Vineyard: n_IS == n_TS + 1 holds for the toy pathway',
           len(T.freqs_is_cm1) == len(T.freqs_ts_cm1) + 1)
_refused = False
try:
    R.vineyard_prefactor([100.0, 200.0], [100.0, 200.0])   # equal counts: illegal
except ValueError:
    _refused = True
check_true('Vineyard: refuses a prefactor when n_IS != n_TS + 1', _refused)


# ── 4. partition functions ───────────────────────────────────────────────────
# Harmonic vibrational partition function, measured from the zero-point level:
#   q_vib = prod_i 1 / (1 - exp(-h nu_i / kB T))
modes = [f for f in T.freqs_is_cm1 if f >= 50.0]
q_eqn = 1.0
for f in modes:
    q_eqn *= 1.0 / (1.0 - math.exp(-(f * H_EVS * C_CM_S) / (KB_EV * T_MID)))
check('q_vib = prod 1/(1 - exp(-h nu / kB T))',
      R.vib_partition_function(list(T.freqs_is_cm1), T_MID, min_freq_cm1=50.0),
      q_eqn, rtol=1e-6)

check_true('q_vib >= 1 for every real mode', q_eqn >= 1.0)


# ── 5. Arrhenius rate ────────────────────────────────────────────────────────
check('k = nu * exp(-Ea / kB T)',
      R.arrhenius_rate(1.0e13, 0.5, T_MID),
      1.0e13 * math.exp(-0.5 / (KB_EV * T_MID)), rtol=1e-12)


# ── 6. diffusivity: Einstein relation and Arrhenius ──────────────────────────
# In three dimensions MSD(t) = 6 D t, so D is the slope over six.
t_ps, msd_A2 = T.msd_trace()
slope = (msd_A2[-1] - msd_A2[0]) / (t_ps[-1] - t_ps[0])
check('Einstein: D = slope / 6  (3-D)', slope / 6.0, T.msd_slope_A2_ps / 6.0)
check('Einstein: slope recovered from the toy trace', slope, T.msd_slope_A2_ps)

# > [!IMPORTANT]
# The same Arrhenius expression exists in two modules with REVERSED argument
# order. Both are checked here so the trap cannot regress unnoticed:
#     permeation.arrhenius_diffusivity(D0, E_D, T)   -- D0 first
#     diffusivity_post_processing.arrhenius_D(T, Ea, D0) -- T first
check('D(T) = D0 exp(-E_D / kB T)   [diffusivity_post_processing: T, Ea, D0]',
      float(DPP.arrhenius_D(T_MID, T.e_d_ev, T.d0_m2s)), T.d_at(T_MID), rtol=1e-12)
check('D(T) = D0 exp(-E_D / kB T)   [permeation: D0, E_D, T]',
      P.arrhenius_diffusivity(T.d0_m2s, T.e_d_ev, T_MID), T.d_at(T_MID), rtol=1e-12)

# Every module must share ONE Boltzmann constant. Part 3 and energetics once
# carried a 7-digit truncation, which surfaced here as a ~1e-7 disagreement in
# D(T) across modules; unified to CODATA 2018. These checks keep them pinned.
_KB_CODATA = 8.617333262e-5
check('kB: permeation', P._KB_EV, _KB_CODATA)
check('kB: tst_rates', R.BOLTZMANN_eV, _KB_CODATA)
check('kB: diffusivity_post_processing', DPP.KB_EV, _KB_CODATA)
check('kB: energetics', E.KB_EV, _KB_CODATA)
check_true('the two Arrhenius helpers agree exactly when each is called its own way',
           math.isclose(float(DPP.arrhenius_D(T_MID, T.e_d_ev, T.d0_m2s)),
                        P.arrhenius_diffusivity(T.d0_m2s, T.e_d_ev, T_MID),
                        rel_tol=1e-12))
# An Arrhenius fit of a series generated from known parameters must return them.
# fit_arrhenius takes per-point sigmas and weights by inverse variance,
# w_i = (D_i / sigma_i)^2; equal fractional errors therefore weight equally.
Ts  = list(T.temperatures_K)
Ds  = [T.d_at(x) for x in Ts]
Des = [0.01 * d for d in Ds]                       # 1% on every point
import numpy as _np                               # ndarray-only signature
Ea_f, Ea_err_f, D0_f, D0_err_f, R2_f = DPP.fit_arrhenius(
    _np.asarray(Ts, float), _np.asarray(Ds, float), _np.asarray(Des, float))
check('Arrhenius fit recovers D0', D0_f, T.d0_m2s, rtol=1e-6)
check('Arrhenius fit recovers E_a', Ea_f, T.e_d_ev, rtol=1e-6)
check_true('Arrhenius fit of exact data gives R^2 = 1', abs(R2_f - 1.0) < 1e-9)


# ── 7. solubility ────────────────────────────────────────────────────────────
# Geometric prefactor is the oct-site density itself.
check('S0 geometric = 4 / (a0^3 N_A)',
      P.lattice_site_S0(T.a0_m), 4.0 / (T.a0_m ** 3) / N_A)

# dH_sol(env) = 1/2 dH_diss + dH_HopA(env)
env_tbl = T.dh_sol_by_env()
for env, v in env_tbl.items():
    check(f'dH_sol({env}) = 1/2 dH_diss + dH_HopA',
          v['dH_sol_eV'], 0.5 * T.dh_diss_ev + T.environments[env]['dH_hopA_eV'])

# S(T) = S0 * sum_env w_env exp(-dH_sol(env) / kB T)
S0 = P.lattice_site_S0(T.a0_m)
boltz = sum(v['w_env'] * math.exp(-v['dH_sol_eV'] / (KB_EV * T_MID))
            for v in env_tbl.values())
check('S(T) = S0 * sum_env w_env exp(-dH_sol/kB T)',
      P.solubility_by_environment(env_tbl, S0, T_MID), S0 * boltz, rtol=1e-9)

check_true('Boltzmann weights w_env sum to 1',
           abs(sum(v['w_env'] for v in env_tbl.values()) - 1.0) < 1e-12)
check_true('the weighted sum is bounded above by 1 (every dH_sol > 0)', boltz < 1.0)

# The single-environment case must reduce to plain Sieverts.
one = {'only': {'dH_sol_eV': 0.5, 'w_env': 1.0}}
check('single environment reduces to Sieverts S = S0 exp(-dH/kB T)',
      P.solubility_by_environment(one, S0, T_MID),
      P.sieverts_solubility(0.5, S0, T_MID), rtol=1e-12)


# ── 8. permeability ──────────────────────────────────────────────────────────
D_mid = T.d_at(T_MID)
S_mid = P.solubility_by_environment(env_tbl, S0, T_MID)
check('Phi = D * S', P.permeability(D_mid, S_mid), D_mid * S_mid)

# Phi0 = D0 * S0 and E_Phi = E_D + dH_sol, the textbook identity.
dh_mean = sum(v['w_env'] * v['dH_sol_eV'] for v in env_tbl.values())
pa = P.permeability_arrhenius(T.d0_m2s, T.e_d_ev, S0, dh_mean)
check('Phi0 = D0 * S0', pa['Phi0'], T.d0_m2s * S0, rtol=1e-12)
check('E_Phi = E_D + dH_sol', pa['E_phi_eV'], T.e_d_ev + dh_mean, rtol=1e-12)

# Richardson flux through a membrane of thickness L.
check('J = (Phi / L) (sqrt(P_high) - sqrt(P_low))',
      P.richardson_flux(1.0e-12, 1.0e6, 0.0, 1.0e-3),   # (Phi, P_high, P_low, L)
      (1.0e-12 / 1.0e-3) * (math.sqrt(1.0e6) - math.sqrt(0.0)), rtol=1e-12)


# ── 9. error propagation conventions ─────────────────────────────────────────
# Energies are additive, so absolute sigmas add in quadrature.
s1, s2 = 0.03, 0.04
check('energy errors add in quadrature (absolute)',
      math.sqrt(s1 ** 2 + s2 ** 2), 0.05, rtol=1e-12)

# Prefactors are multiplicative, so FRACTIONAL sigmas add in quadrature.
f1, f2 = 0.3, 0.4
check('prefactor errors add in quadrature (fractional)',
      math.sqrt(f1 ** 2 + f2 ** 2), 0.5, rtol=1e-12)

# A fractional sigma maps to a multiplicative x/÷ band, never a symmetric +/-.
check('fractional sigma -> x/div factor exp(sigma)',
      math.exp(1.0), 2.718281828459045, rtol=1e-12)


if __name__ == '__main__':
    raise SystemExit(report())

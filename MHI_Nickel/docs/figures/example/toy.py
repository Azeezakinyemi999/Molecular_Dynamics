"""
toy.py — a fictitious material used by every figure and equation check.

Nothing here comes from a real calculation. "Xx" is an invented FCC metal with
round-numbered properties, chosen so that each quantity in the documentation can
be computed, checked and plotted in under a second, with no MACE model, no
LAMMPS run and no cluster.

The numbers are deliberately unphysical-but-plausible: they exercise every code
path (two interstitial environments, a transition state with exactly one
imaginary mode, modes below the quasi-harmonic floor, a three-temperature
Arrhenius series) without resembling any real metal.

Import it as::

    from toy import TOY

Every field is documented in the TOY docstring below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# Physical constants, repeated here rather than imported, so the checks compare
# the project's constants against an independent copy instead of themselves.
KB_EV   = 8.617333262e-5        # Boltzmann constant [eV/K]
C_CM_S  = 2.99792458e10         # speed of light [cm/s]
H_EVS   = 4.135667696e-15       # Planck constant [eV·s]
N_A     = 6.02214076e23         # Avogadro [1/mol]


@dataclass(frozen=True)
class ToySystem:
    """A fictitious FCC metal and one H pathway through it.

    Attributes
    ----------
    name, symbol
        Invented identifiers. Never a real element.
    a0_m
        Cubic lattice parameter [m]. 4.0 Å exactly, so the octahedral site
        density is a round number and arithmetic slips are obvious.
    temperatures_K
        The three temperatures the toy Arrhenius series is defined on.
    freqs_is_cm1, freqs_ts_cm1, freqs_fs_cm1
        Real vibrational frequencies [cm^-1] for the initial, transition and
        final states of one hop. The TS list has exactly one fewer entry than
        the IS list, which is what a saddle looks like after its single
        imaginary mode is removed -- the condition ``vineyard_prefactor``
        enforces. Two IS modes sit below the 100 cm^-1 quasi-harmonic floor so
        the raising behaviour is exercised.
    freqs_ts_imag_cm1
        The TS imaginary mode [cm^-1], reported as a positive magnitude, which
        is the convention the vibration JSON uses.
    ea_ev, ed_ev
        Forward and reverse electronic barriers [eV] for that hop.
    dh_diss_ev
        H2 dissociation reaction energy [eV], per H2.
    environments
        Two interstitial environments, each with a Hop A reaction energy [eV]
        and a population weight. The weights sum to 1 and the enthalpies differ
        by several k_B T at 400 K, so the Boltzmann collapse is visible.
    d0_m2s, e_d_ev
        Diffusivity Arrhenius parameters [m^2/s] and [eV].
    msd_slope_A2_ps
        Slope of a synthetic linear MSD trace [A^2/ps], used to check the
        Einstein relation and the unit conversion.
    """

    name:    str = 'Toymetal'
    symbol:  str = 'Xx'
    a0_m:    float = 4.0e-10

    temperatures_K: tuple = (400.0, 600.0, 800.0)

    freqs_is_cm1: tuple = (60.0, 85.0, 140.0, 410.0, 630.0, 900.0)
    freqs_ts_cm1: tuple = (95.0, 150.0, 430.0, 640.0, 880.0)
    freqs_fs_cm1: tuple = (70.0, 110.0, 155.0, 400.0, 620.0, 910.0)
    freqs_ts_imag_cm1: tuple = (520.0,)

    ea_ev: float = 0.4000
    ed_ev: float = 0.2500
    dh_diss_ev: float = 0.9000

    environments: dict = field(default_factory=lambda: {
        'Xx6_oct': {'dH_hopA_eV': 0.1000, 'w_env': 0.25, 'n_sites': 2},
        'Xx4_tet': {'dH_hopA_eV': 0.3500, 'w_env': 0.75, 'n_sites': 6},
    })

    d0_m2s: float = 1.0e-7
    e_d_ev: float = 0.2000

    msd_slope_A2_ps: float = 6.0

    # ── derived helpers, so checks and figures agree by construction ────────
    def dh_sol_by_env(self) -> dict:
        """ΔH_sol(env) = ½·ΔH_diss + ΔH_HopA(env), the Stage 9 construction."""
        return {
            env: {
                'dH_sol_eV': 0.5 * self.dh_diss_ev + v['dH_hopA_eV'],
                'dH_sol_err_eV': 0.0,
                'w_env': v['w_env'],
                'n_sites': v['n_sites'],
                'dH_hopA_eV': v['dH_hopA_eV'],
                'dH_hopA_err_eV': 0.0,
            }
            for env, v in self.environments.items()
        }

    def rho_oct_mol_m3(self) -> float:
        """Octahedral site density [mol/m^3]: 4 sites per cubic cell."""
        return 4.0 / (self.a0_m ** 3) / N_A

    def d_at(self, T_K: float) -> float:
        """D(T) [m^2/s] from the toy Arrhenius parameters."""
        return self.d0_m2s * math.exp(-self.e_d_ev / (KB_EV * T_K))

    def msd_trace(self, n: int = 200, t_max_ps: float = 100.0):
        """A noiseless linear MSD trace: MSD = slope · t, in A^2 against ps."""
        dt = t_max_ps / (n - 1)
        t = [i * dt for i in range(n)]
        return t, [self.msd_slope_A2_ps * x for x in t]


TOY = ToySystem()


if __name__ == '__main__':
    t = TOY
    print(f'{t.name} ({t.symbol})  a0 = {t.a0_m * 1e10:.2f} A')
    print(f'  oct-site density : {t.rho_oct_mol_m3():.6e} mol/m^3')
    print(f'  IS/TS/FS modes   : {len(t.freqs_is_cm1)}/{len(t.freqs_ts_cm1)}'
          f'(+{len(t.freqs_ts_imag_cm1)} imag)/{len(t.freqs_fs_cm1)}')
    for env, v in t.dh_sol_by_env().items():
        print(f'  {env:<10} dH_sol = {v["dH_sol_eV"]:+.4f} eV   w = {v["w_env"]:.2f}')

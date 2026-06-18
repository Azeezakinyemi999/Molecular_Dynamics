"""
models/diffusivity_post_processing.py
======================================
Post-processing pipeline for H diffusivity from LAMMPS MD trajectories.

Workflow (NB10 → NB11 → NB12):
  1. LAMMPS NVT MD runs produce dump trajectory files  (NB10)
  2. Unwrap coordinates → compute MSD → fit D(T)       (NB11)
  3. Fit Arrhenius: D(T) → Ea, D0                      (NB12)

Typical usage
-------------
NB11::

    from models.diffusivity_post_processing import run_diffusivity_pipeline

    results = {}
    for T in [300, 400, 600, 800]:
        results[T] = run_diffusivity_pipeline(
            dump_file=f'results/notebook10-bulk-equilibration/N_7/traj_{T}K.dump',
            temperature=T,
            h_type=9,
            timestep=0.0005,
            outdir='results/notebook11/N_7',
        )
    save_diffusivity_table(
        temperatures=list(results.keys()),
        D_vals=[r['D'] for r in results.values()],
        D_errs=[r['sigma_D'] for r in results.values()],
        R2_vals=[r['R2'] for r in results.values()],
        outfile='results/notebook11/N_7/diffusivity.txt',
    )

NB12::

    from models.diffusivity_post_processing import fit_arrhenius, arrhenius_D, plot_arrhenius
    from models.parsers import parse_diffusivity_file

    T, D, Derr, R2 = parse_diffusivity_file('results/notebook11/N_7/diffusivity.txt')
    Ea, Ea_err, D0, D0_err, R2_fit = fit_arrhenius(T, D, Derr)
    fig = plot_arrhenius(T, D, Derr, Ea, D0, outfile='results/notebook12/N_7/arrhenius.png')
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Section 1 — Physical constants
# ---------------------------------------------------------------------------

KB_EV          = 8.617333e-5   # eV K⁻¹  (Boltzmann constant)
ANG2_PS_TO_M2S = 1e-8          # Å² ps⁻¹ → m² s⁻¹


# ---------------------------------------------------------------------------
# Section 2 — Coordinate unwrapping
# ---------------------------------------------------------------------------

def unwrap_trajectory(
    positions: np.ndarray,
    box_lengths: np.ndarray,
) -> np.ndarray:
    """Unwrap periodic boundary condition jumps in a trajectory.

    Converts wrapped (in-box) coordinates to continuous Cartesian paths
    using the minimum image convention applied frame-to-frame.

    Parameters
    ----------
    positions : ndarray, shape (n_frames, n_atoms, 3)
        Wrapped atomic coordinates in Å.
    box_lengths : ndarray, shape (3,)
        Simulation box edge lengths [Lx, Ly, Lz] in Å.

    Returns
    -------
    ndarray, shape (n_frames, n_atoms, 3)
        Unwrapped (continuous) atomic coordinates in Å.
    """
    n_frames, n_atoms, _ = positions.shape
    unwrapped = positions.copy()

    for t in range(1, n_frames):
        dr = positions[t] - positions[t - 1]
        # Minimum image: shift by ±L to minimise |dr|
        dr -= box_lengths * np.round(dr / box_lengths)
        unwrapped[t] = unwrapped[t - 1] + dr

    print(f'[unwrap] {n_atoms} atoms  {n_frames} frames  box={box_lengths[0]:.2f}×{box_lengths[1]:.2f}×{box_lengths[2]:.2f} Å')
    return unwrapped


# ---------------------------------------------------------------------------
# Section 3 — MSD calculation
# ---------------------------------------------------------------------------

def compute_msd(
    positions_unwrapped: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean-squared displacement using multi-origin averaging.

    For each lag time τ the MSD is averaged over all valid starting frames:

    .. math::

        \\text{MSD}(\\tau) = \\left\\langle |\\mathbf{r}(t+\\tau) -
        \\mathbf{r}(t)|^2 \\right\\rangle_{t}

    Parameters
    ----------
    positions_unwrapped : ndarray, shape (n_frames, n_atoms, 3)
        Unwrapped coordinates in Å from :func:`unwrap_trajectory`.
        For a single-H trajectory ``n_atoms=1``; for multi-H runs the MSD
        is averaged over all atoms.
    max_lag : int, optional
        Maximum lag in frames.  Defaults to ``n_frames // 2``.

    Returns
    -------
    lag_frames : ndarray, shape (max_lag,)
        Lag times in units of **frames** (multiply by dump interval to get ps).
    msd_arr : ndarray, shape (max_lag,)
        MSD values in Å².
    """
    pos = positions_unwrapped
    n_frames = pos.shape[0]
    if max_lag is None:
        max_lag = n_frames // 2

    lag_frames = np.arange(1, max_lag + 1)
    msd_arr    = np.zeros(max_lag)

    for lag in lag_frames:
        disp = pos[lag:] - pos[:-lag]               # (n_origins, ..., 3)
        sq   = np.sum(disp ** 2, axis=-1)           # (n_origins,) or (n_origins, n_atoms)
        msd_arr[lag - 1] = float(np.mean(sq))

    n_origins = n_frames - max_lag
    print(f'[MSD] {n_origins} origins  {max_lag} lags  MSD_max={msd_arr[-1]:.3f} Å²')
    return lag_frames, msd_arr


# ---------------------------------------------------------------------------
# Section 4 — Diffusivity fit
# ---------------------------------------------------------------------------

def fit_diffusivity(
    t_arr: np.ndarray,
    msd_arr: np.ndarray,
    fit_window: tuple[float, float] = (0.2, 0.8),
) -> tuple[float, float, float]:
    """Fit a linear slope to MSD(t) and extract the self-diffusion coefficient.

    Uses the Einstein relation in 3D:

    .. math::

        D = \\frac{\\text{slope}}{6}, \\quad \\text{MSD}(t) = 6Dt + c

    Parameters
    ----------
    t_arr : ndarray
        Time array in ps.
    msd_arr : ndarray
        MSD array in Å².
    fit_window : tuple of float
        ``(start_frac, end_frac)`` — fraction of ``t_arr`` range to use for
        the linear fit.  Default ``(0.2, 0.8)`` skips ballistic and
        anomalous-diffusion regimes at the ends.

    Returns
    -------
    D : float
        Self-diffusion coefficient in m²/s.
    sigma_D : float
        Standard error of D in m²/s (propagated from regression stderr).
    R2 : float
        Coefficient of determination of the linear fit.
    """
    from scipy.stats import linregress

    t_min = t_arr[0] + fit_window[0] * (t_arr[-1] - t_arr[0])
    t_max = t_arr[0] + fit_window[1] * (t_arr[-1] - t_arr[0])
    mask  = (t_arr >= t_min) & (t_arr <= t_max)

    t_fit   = t_arr[mask]
    msd_fit = msd_arr[mask]

    res = linregress(t_fit, msd_fit)

    slope    = res.slope      # Å² / ps
    stderr   = res.stderr     # Å² / ps
    ss_res   = np.sum((msd_fit - (res.slope * t_fit + res.intercept)) ** 2)
    ss_tot   = np.sum((msd_fit - msd_fit.mean()) ** 2)
    R2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    D       = (slope  / 6.0) * ANG2_PS_TO_M2S
    sigma_D = (stderr / 6.0) * ANG2_PS_TO_M2S

    print(f'[D fit] D={D:.4e} m²/s  σ={sigma_D:.2e} m²/s  R²={R2:.4f}')
    return D, sigma_D, R2


# ---------------------------------------------------------------------------
# Section 5 — Save D(T) table
# ---------------------------------------------------------------------------

def save_diffusivity_table(
    temperatures: list[float],
    D_vals: list[float],
    D_errs: list[float],
    R2_vals: list[float],
    outfile: str,
) -> None:
    """Write the diffusivity results table consumed by NB12 and parsers.py.

    The output format is compatible with
    :func:`models.parsers.parse_diffusivity_file`.

    Parameters
    ----------
    temperatures : list of float
        Temperatures in K.
    D_vals : list of float
        Diffusivities in m²/s.
    D_errs : list of float
        Standard errors in m²/s.
    R2_vals : list of float
        R² values from the MSD linear fit.
    outfile : str
        Path to write ``diffusivity.txt``.
    """
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)

    header = (
        '# Hydrogen self-diffusivity from MSD linear fit\n'
        '# D = slope/6  (Einstein relation, 3D)\n'
        '# Fit window: 20-80% of trajectory\n'
        '#\n'
        f'# {"T_K":>8}  {"D (m2/s)":>14}  {"sigma_D (m2/s)":>16}  {"R2":>8}\n'
        f'# {"="*8}  {"="*14}  {"="*16}  {"="*8}\n'
    )

    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(header)
        for T, D, Derr, R2 in zip(temperatures, D_vals, D_errs, R2_vals):
            f.write(f'  {T:8.1f}  {D:14.6e}  {Derr:16.6e}  {R2:8.4f}\n')

    print(f'Wrote: {outfile}')


# ---------------------------------------------------------------------------
# Section 6 — Pipeline orchestrator  (NB11 entry point)
# ---------------------------------------------------------------------------

def run_diffusivity_pipeline(
    dump_file: str,
    temperature: float,
    h_type: int = 8,
    timestep: float = 0.0005,
    outdir: str = '.',
    fit_window: tuple[float, float] = (0.2, 0.8),
) -> dict:
    """Full MSD → D pipeline for a single temperature trajectory.

    Calls :func:`unwrap_trajectory` → :func:`compute_msd` →
    :func:`fit_diffusivity` and saves the MSD data to ``<outdir>/msd_<T>K.txt``.

    Parameters
    ----------
    dump_file : str
        Path to LAMMPS dump file from NB10.
    temperature : float
        Temperature label in K (used for output file naming only).
    h_type : int
        LAMMPS atom type index for H.  Use ``8`` for the 8-type system
        (no O) or ``9`` for the 9-type system (with O).
    timestep : float
        MD timestep in ps (default 0.0005 ps = 0.5 fs).
    outdir : str
        Directory to write ``msd_<T>K.txt``.
    fit_window : tuple of float
        Fit window fraction passed to :func:`fit_diffusivity`.

    Returns
    -------
    dict with keys
        ``D``        – diffusivity in m²/s
        ``sigma_D``  – standard error in m²/s
        ``R2``       – fit quality
        ``msd_file`` – path to the saved MSD text file
    """
    from models.parsers import parse_lammps_dump

    # Load trajectory
    # parse_lammps_dump returns pos shape (n_frames, n_H, 3)
    t_raw, pos, box = parse_lammps_dump(
        dump_file, h_type=h_type, timestep=timestep)

    if pos is None:
        raise FileNotFoundError(f'No trajectory data found in {dump_file}')

    # box: (n_frames, 3) — use first frame for unwrapping
    box_lengths = box[0]

    unwrapped  = unwrap_trajectory(pos, box_lengths)
    lag_frames, msd_arr = compute_msd(unwrapped)

    # Convert lag frames → time in ps
    dump_interval = t_raw[1] - t_raw[0] if len(t_raw) > 1 else 1.0
    t_lag = lag_frames * dump_interval

    D, sigma_D, R2 = fit_diffusivity(t_lag, msd_arr, fit_window=fit_window)

    # Save MSD data
    os.makedirs(outdir, exist_ok=True)
    msd_file = os.path.join(outdir, f'msd_{int(temperature)}K.txt')
    with open(msd_file, 'w', encoding='utf-8') as f:
        f.write('# t_ps  MSD_ang2\n')
        for t, m in zip(t_lag, msd_arr):
            f.write(f'{t:.6f}  {m:.6f}\n')

    print(f'T={temperature:.0f} K  D={D:.4e} m²/s  σ_D={sigma_D:.2e}  R²={R2:.4f}')
    return {'D': D, 'sigma_D': sigma_D, 'R2': R2, 'msd_file': msd_file,
            't_lag': t_lag, 'msd_arr': msd_arr}


# ---------------------------------------------------------------------------
# Section 7 — Arrhenius fit
# ---------------------------------------------------------------------------

def _weighted_linregress(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Weighted least-squares linear regression: minimise Σ wᵢ(yᵢ - (ax+b))².

    Parameters
    ----------
    x, y : ndarray
        Data arrays.
    weights : ndarray
        Per-point weights (larger = more influence).

    Returns
    -------
    slope, intercept, slope_err, intercept_err, R2 : float
    """
    w  = weights
    sw = w.sum()
    sx = (w * x).sum()
    sy = (w * y).sum()
    sxx = (w * x * x).sum()
    sxy = (w * x * y).sum()
    syy = (w * y * y).sum()

    denom     = sw * sxx - sx ** 2
    slope     = (sw * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / sw

    y_pred = slope * x + intercept
    ss_res = (w * (y - y_pred) ** 2).sum()
    ss_tot = (w * (y - sy / sw) ** 2).sum()
    R2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Variance of coefficients (assuming uniform residual variance)
    sigma2    = ss_res / max(len(x) - 2, 1)
    slope_err     = np.sqrt(sigma2 * sw / denom)
    intercept_err = np.sqrt(sigma2 * sxx / denom)

    return float(slope), float(intercept), float(slope_err), float(intercept_err), float(R2)


def fit_arrhenius(
    T_arr: np.ndarray,
    D_arr: np.ndarray,
    D_err_arr: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Fit Arrhenius parameters from D(T) data using weighted linear regression.

    Linearises :math:`D = D_0 \\exp(-E_a / k_B T)` as:

    .. math::

        \\ln D = \\ln D_0 - \\frac{E_a}{k_B} \\cdot \\frac{1}{T}

    Weights are set to inverse-variance: :math:`w_i = (D_i / \\sigma_{D_i})^2`.

    Parameters
    ----------
    T_arr : ndarray
        Temperatures in K.
    D_arr : ndarray
        Diffusivities in m²/s.
    D_err_arr : ndarray
        Standard errors of D in m²/s.

    Returns
    -------
    Ea : float
        Activation energy in eV.
    Ea_err : float
        Uncertainty in Ea in eV.
    D0 : float
        Pre-exponential factor in m²/s.
    D0_err : float
        Uncertainty in D0 in m²/s.
    R2 : float
        R² of the weighted fit.
    """
    x = 1.0 / T_arr
    y = np.log(D_arr)
    w = (D_arr / D_err_arr) ** 2   # inverse-variance weights

    slope, intercept, slope_err, intercept_err, R2 = _weighted_linregress(x, y, w)

    # slope = -Ea / kB
    Ea     = -slope      * KB_EV
    Ea_err =  slope_err  * KB_EV

    # intercept = ln(D0)
    D0     = float(np.exp(intercept))
    D0_err = D0 * intercept_err    # error propagation: σ_D0 = D0 * σ_ln(D0)

    return Ea, Ea_err, D0, D0_err, R2


# ---------------------------------------------------------------------------
# Section 8 — Arrhenius extrapolation
# ---------------------------------------------------------------------------

def arrhenius_D(
    T: float | np.ndarray,
    Ea: float,
    D0: float,
) -> float | np.ndarray:
    """Evaluate the Arrhenius expression at temperature(s) T.

    .. math::

        D(T) = D_0 \\, \\exp\\!\\left(-\\frac{E_a}{k_B T}\\right)

    Parameters
    ----------
    T : float or ndarray
        Temperature(s) in K.
    Ea : float
        Activation energy in eV.
    D0 : float
        Pre-exponential factor in m²/s.

    Returns
    -------
    float or ndarray
        Diffusivity in m²/s.
    """
    return D0 * np.exp(-Ea / (KB_EV * np.asarray(T, dtype=float)))


# ---------------------------------------------------------------------------
# Section 9 — Arrhenius plot
# ---------------------------------------------------------------------------

def plot_arrhenius(
    T_arr: np.ndarray,
    D_arr: np.ndarray,
    D_err_arr: np.ndarray,
    Ea: float,
    D0: float,
    outfile: str | None = None,
    lit_Ea: float | None = None,
    lit_D0: float | None = None,
    lit_label: str = 'Pure Ni (lit.)',
) -> 'matplotlib.figure.Figure':
    """Plot log₁₀(D) vs 1000/T with error bars and fitted Arrhenius line.

    Parameters
    ----------
    T_arr : ndarray
        MD temperatures in K.
    D_arr : ndarray
        MD diffusivities in m²/s.
    D_err_arr : ndarray
        Standard errors of D in m²/s.
    Ea : float
        Fitted activation energy in eV.
    D0 : float
        Fitted pre-exponential in m²/s.
    outfile : str, optional
        If given, save figure to this path.
    lit_Ea : float, optional
        Literature activation energy in eV for comparison overlay.
    lit_D0 : float, optional
        Literature pre-exponential in m²/s for comparison overlay.
    lit_label : str
        Legend label for the literature line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # x-axis: 1000/T for readability
    x_data = 1000.0 / T_arr
    y_data = np.log10(D_arr)
    y_err  = D_err_arr / (D_arr * np.log(10))   # σ in log10 units

    # Smooth fit line over extended range
    T_fine  = np.linspace(T_arr.min() * 0.9, T_arr.max() * 1.1, 200)
    x_fine  = 1000.0 / T_fine
    y_fine  = np.log10(arrhenius_D(T_fine, Ea, D0))

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.errorbar(x_data, y_data, yerr=y_err,
                fmt='o', color='tab:blue', capsize=4, zorder=5,
                label='MD data')
    ax.plot(x_fine, y_fine, 'k-', lw=1.5,
            label=f'Fit: $E_a$={Ea:.3f} eV,  $D_0$={D0:.2e} m²/s')

    if lit_Ea is not None and lit_D0 is not None:
        y_lit = np.log10(arrhenius_D(T_fine, lit_Ea, lit_D0))
        ax.plot(x_fine, y_lit, '--', color='tab:orange', lw=1.5,
                label=f'{lit_label}: $E_a$={lit_Ea:.3f} eV')

    # Secondary x-axis: temperature in K
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    t_ticks = np.array([300, 400, 500, 600, 700, 800])
    t_ticks = t_ticks[(t_ticks >= T_arr.min() * 0.9) & (t_ticks <= T_arr.max() * 1.1)]
    ax2.set_xticks(1000.0 / t_ticks)
    ax2.set_xticklabels([str(t) for t in t_ticks])
    ax2.set_xlabel('Temperature (K)')

    ax.set_xlabel('1000 / T  (K⁻¹)')
    ax.set_ylabel('log₁₀( D / m² s⁻¹ )')
    ax.set_title('Arrhenius Plot — H Diffusivity in Hastelloy N')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if outfile:
        os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved: {outfile}')

    return fig


# ---------------------------------------------------------------------------
# Section 10 — Arrhenius pipeline orchestrator  (NB12 entry point)
# ---------------------------------------------------------------------------

def run_arrhenius_pipeline(
    diffusivity_file: str,
    outdir: str = '.',
    plot_filename: str = 'arrhenius.png',
    lit_Ea: float | None = None,
    lit_D0: float | None = None,
    lit_label: str = 'Literature',
) -> dict:
    """Load D(T) table → Arrhenius fit → save plot.

    One-call orchestrator for NB12 that wraps
    :func:`fit_arrhenius` and :func:`plot_arrhenius`.

    Parameters
    ----------
    diffusivity_file : str
        Path to ``diffusivity.txt`` written by :func:`save_diffusivity_table`
        (or the equivalent NB11 output).
    outdir : str
        Directory to save ``arrhenius.png`` (and the text summary).
    plot_filename : str
        Filename for the Arrhenius plot inside ``outdir``.
    lit_Ea : float, optional
        Literature activation energy in eV for an overlay comparison line.
    lit_D0 : float, optional
        Literature pre-exponential in m²/s for the overlay line.
    lit_label : str
        Legend label for the literature overlay.

    Returns
    -------
    dict with keys
        ``Ea``        – activation energy in eV
        ``Ea_err``    – uncertainty in Ea (eV)
        ``D0``        – pre-exponential factor in m²/s
        ``D0_err``    – uncertainty in D0 (m²/s)
        ``R2``        – R² of the Arrhenius fit
        ``T_arr``     – temperature array (K)
        ``D_arr``     – diffusivity array (m²/s)
        ``D_err_arr`` – diffusivity error array (m²/s)
        ``fig``       – matplotlib Figure
        ``plot_file`` – path to the saved plot
    """
    from models.parsers import parse_diffusivity_file

    T_arr, D_arr, D_err_arr, _ = parse_diffusivity_file(diffusivity_file)

    Ea, Ea_err, D0, D0_err, R2 = fit_arrhenius(T_arr, D_arr, D_err_arr)

    print(f'Arrhenius fit:  Ea = {Ea:.4f} ± {Ea_err:.4f} eV  |  '
          f'D0 = {D0:.4e} ± {D0_err:.2e} m²/s  |  R² = {R2:.4f}')

    os.makedirs(outdir, exist_ok=True)
    plot_file = os.path.join(outdir, plot_filename)

    fig = plot_arrhenius(
        T_arr, D_arr, D_err_arr, Ea, D0,
        outfile=plot_file,
        lit_Ea=lit_Ea, lit_D0=lit_D0, lit_label=lit_label,
    )

    return {
        'Ea': Ea, 'Ea_err': Ea_err,
        'D0': D0, 'D0_err': D0_err,
        'R2': R2,
        'T_arr': T_arr, 'D_arr': D_arr, 'D_err_arr': D_err_arr,
        'fig': fig, 'plot_file': plot_file,
    }

"""
models/lammps_scripts.py
========================
Writer functions for LAMMPS input scripts used across MHI_Nickel notebooks.

Covers:
- Bulk CG minimization + box/relax        (NB02)
- NPT lattice parameter scan               (NB02)
- Surface relaxation (freeze → min → NVT) (NB04, NB04b)
- NVT bulk equilibration + H MSD tracking (NB10)

All functions support optional periodic restart checkpointing via
``restart_dir`` and ``restart_every``.  The surface relaxation function
additionally writes both ``write_restart`` and ``write_data`` at the end
of each phase.
"""

import os


# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _pair_block(pair_style, mace_model, pair_suffix, elem_str):
    return (
        f'pair_style     {pair_style} {mace_model} {pair_suffix}\n'
        f'pair_coeff     * * {elem_str}'
    )


def _neighbor_block():
    return (
        'neighbor       2.0 bin\n'
        'neigh_modify   every 1 delay 0 check yes'
    )


def _restart_line(restart_dir, restart_every, label='run'):
    """Return a LAMMPS restart directive, or empty string if restart_dir is None."""
    if restart_dir is None:
        return ''
    return f'restart        {restart_every}  {restart_dir}/{label}.*.restart'


def _stem(path):
    """Return path without extension, used to derive per-phase filenames."""
    base, _ = os.path.splitext(path)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# 1. BULK CG MINIMIZATION  (NB02)
# ═══════════════════════════════════════════════════════════════════════════

def write_minimization_script(
    bulk_input,
    min_output,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    etol=0.0,
    ftol=1e-8,
    maxiter=50000,
    maxeval=500000,
    supercell_reps=5,
    thermo_every=10,
    restart_dir=None,
    restart_every=10000,
    traj_file=None,
    dump_every=10,
    masses=None,
):
    """
    Write a LAMMPS conjugate-gradient minimization script with isotropic
    box relaxation (``fix box/relax iso 0.0``).

    Used in NB02 to find the MACE equilibrium lattice parameter at 0 K.

    Parameters
    ----------
    bulk_input : str
        Path to the LAMMPS data file produced by NB01.
    min_output : str
        Path where the minimized structure is written (``write_data``).
    out_path : str
        Destination path for the ``.lammps`` input script.
    pair_style : str
        LAMMPS ``pair_style`` keyword, e.g. ``'mliap unified'``.
    mace_model : str
        Full path to the MACE ``.pt`` model file.
    pair_suffix : str
        Suffix appended to the model path in ``pair_style``, e.g. ``'0'``.
    elem_str : str
        Space-separated element string for ``pair_coeff * *``.
    etol : float, optional
        Energy tolerance for ``minimize``. Default ``0.0``.
    ftol : float, optional
        Force tolerance (eV/Å) for ``minimize``. Default ``1e-8``.
    maxiter : int, optional
        Maximum minimization iterations. Default ``50000``.
    maxeval : int, optional
        Maximum force evaluations. Default ``500000``.
    supercell_reps : int, optional
        Unit cell repetitions along one axis for ``a0`` calculation.
        Default ``5``.
    thermo_every : int, optional
        Frequency of thermo output lines. Default ``10``.
    restart_dir : str or None, optional
        Directory for periodic binary restart files.  If ``None``, no
        restart directive is written. Default ``None``.
    restart_every : int, optional
        Steps between restart file writes. Default ``10000``.

    Returns
    -------
    out_path : str
        Path to the written ``.lammps`` script.
    """
    pair    = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh   = _neighbor_block()
    restart = _restart_line(restart_dir, restart_every, label='min')
    if masses:
        mass_lines = '\n'.join(
            f'mass           {t}  {m[0]}  # {m[1]}'
            for t, m in sorted(masses.items())
        ) + '\n'
    else:
        mass_lines = ''
    if traj_file:
        dump_block   = (f'dump           min_traj  all  custom  {dump_every}  {traj_file}'
                        f'  id type x y z\ndump_modify    min_traj  sort id\n\n')
        undump_block = 'undump         min_traj\n\n'
    else:
        dump_block = undump_block = ''

    script = f"""# ════════════════════════════════════════════════════
# LAMMPS Energy Minimization — Hastelloy N Bulk
# Notebook 02 | pair_style: mliap unified
# ════════════════════════════════════════════════════

units          metal
atom_style     atomic
newton         on
boundary       p p p

read_data      {bulk_input}
{mass_lines}
{restart}

{pair}

{neigh}

thermo         {thermo_every}
thermo_style   custom step pe fmax fnorm press vol lx ly lz

fix            boxrelax all box/relax iso 0.0 vmax 0.001

{dump_block}print "### Minimization ###"

minimize       {etol} {ftol} {maxiter} {maxeval}

print "### Minimization complete ###"
{undump_block}
unfix          boxrelax

variable       pe_final  equal  pe
variable       natoms    equal  atoms
variable       ecoh      equal  -v_pe_final/v_natoms
variable       a0_min    equal  lx/{supercell_reps}
variable       fmax_f    equal  fmax

print "MINIMIZATION_RESULTS_START"
print "  Total_energy_eV     : ${{pe_final}}"
print "  Natoms              : ${{natoms}}"
print "  Ecoh_eV_per_atom    : ${{ecoh}}"
print "  a0_Angstrom         : ${{a0_min}}"
print "  Fmax_eV_per_Ang     : ${{fmax_f}}"
print "MINIMIZATION_RESULTS_END"

write_restart  {_stem(min_output)}_final.restart
write_data     {min_output}
print "Minimized structure written to {min_output}"
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 2. NPT LATTICE PARAMETER SCAN  (NB02)
# ═══════════════════════════════════════════════════════════════════════════

def write_npt_script(
    min_output,
    npt_dump,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    target_t=300,
    timestep=0.001,
    thermo_damp=0.1,
    baro_damp=1.0,
    heat_steps=20000,
    npt_steps=200000,
    dump_every=100,
    supercell_reps=5,
    restart_dir=None,
    restart_every=10000,
    traj_file=None,
):
    """
    Write a two-stage NPT script: temperature ramp followed by
    constant-T/P production to extract the equilibrium lattice parameter.

    Used in NB02 immediately after ``write_minimization_script``.

    Parameters
    ----------
    min_output : str
        Path to the minimized LAMMPS data file (NB02 output).
    npt_dump : str
        Path for the ``ave/time`` box-dimension output file.
    out_path : str
        Destination path for the ``.lammps`` input script.
    pair_style : str
        LAMMPS ``pair_style`` keyword.
    mace_model : str
        Full path to the MACE ``.pt`` model file.
    pair_suffix : str
        Suffix for ``pair_style``.
    elem_str : str
        Element string for ``pair_coeff * *``.
    target_t : float, optional
        Target temperature in K. Default ``300``.
    timestep : float, optional
        MD timestep in ps. Default ``0.001`` (1 fs).
    thermo_damp : float, optional
        Nosé-Hoover thermostat damping time in ps. Default ``0.1``.
    baro_damp : float, optional
        Parrinello-Rahman barostat damping time in ps. Default ``1.0``.
    heat_steps : int, optional
        Steps for the 10 K → target_t heating ramp. Default ``20000``.
    npt_steps : int, optional
        Steps for constant-T NPT production. Default ``200000``.
    dump_every : int, optional
        Frequency for box-dimension output. Default ``100``.
    supercell_reps : int, optional
        Unit cell repetitions along one axis for ``a0`` calculation.
        Default ``5``.
    restart_dir : str or None, optional
        Directory for periodic binary restart files.  If ``None``, no
        restart directive is written. Default ``None``.
    restart_every : int, optional
        Steps between restart file writes. Default ``10000``.

    Returns
    -------
    out_path : str
        Path to the written ``.lammps`` script.
    """
    pair          = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh         = _neighbor_block()
    restart       = _restart_line(restart_dir, restart_every, label=f'npt_{target_t}K')
    if traj_file:
        traj_open  = (f'dump           npt_traj  all  custom  {dump_every}  {traj_file}'
                      f'  id type x y z\ndump_modify    npt_traj  sort id\n\n')
        traj_close = 'undump         npt_traj\n'
    else:
        traj_open = traj_close = ''
    thermo_damp_fs = thermo_damp * 1000
    baro_damp_fs   = baro_damp  * 1000
    heat_ps        = heat_steps * timestep
    npt_ps         = npt_steps  * timestep

    script = f"""# ════════════════════════════════════════════════════
# LAMMPS NPT — Lattice Parameter at {target_t} K
# Notebook 02
# ════════════════════════════════════════════════════

units          metal
atom_style     atomic
newton         on
boundary       p p p

read_data      {min_output}

{restart}

{pair}

{neigh}

timestep       {timestep}

velocity       all create 10.0 12345 mom yes rot yes dist gaussian

thermo         {dump_every}
thermo_style   custom step time temp pe ke press vol lx ly lz

{traj_open}# ── Stage 1: Heat 10 K → {target_t} K over {heat_ps:.0f} ps ──────────────
fix            heat all npt temp 10.0 {target_t}.0 {thermo_damp_fs:.1f} &
               iso 0.0 0.0 {baro_damp_fs:.1f}
run            {heat_steps}
unfix          heat
print "### Heating complete ###"

write_restart  {_stem(npt_dump)}_after_heat.restart
write_data     {_stem(min_output)}_npt_after_heat.lammps

# ── Stage 2: NPT production at {target_t} K ({npt_ps:.0f} ps) ────────────
fix            npt_run all npt temp {target_t}.0 {target_t}.0 {thermo_damp_fs:.1f} &
               iso 0.0 0.0 {baro_damp_fs:.1f}

variable       lx_val  equal  lx
variable       ly_val  equal  ly
variable       lz_val  equal  lz
fix            boxdump all ave/time 1 1 {dump_every} v_lx_val v_ly_val v_lz_val &
               file {npt_dump}

run            {npt_steps}
{traj_close}
unfix          npt_run
unfix          boxdump
print "### NPT complete at {target_t} K ###"

variable       a0_npt  equal  lx/{supercell_reps}.0
print "NPT_RESULT: a0_at_{target_t}K = ${{a0_npt}} Angstrom"

write_restart  {_stem(npt_dump)}_final.restart
write_data     {_stem(min_output)}_npt_final_{target_t}K.lammps
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


def write_npt_restart_script(
    restart_file,
    npt_dump,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    target_t=300,
    timestep=0.001,
    thermo_damp=0.1,
    baro_damp=1.0,
    npt_steps=200000,
    dump_every=100,
    supercell_reps=5,
    restart_dir=None,
    restart_every=10000,
    min_output=None,
):
    """Write a LAMMPS NPT continuation script (Stage 2 production only).

    Reads ``*_after_heat.restart`` written by :func:`write_npt_script` and
    runs NPT production at *target_t*, appending box dimensions to the
    existing *npt_dump* file.

    Mirrors :func:`write_npt_script` Stage 2 exactly with three differences:

    1. ``read_restart`` replaces ``read_data`` — velocities and thermostat
       state are restored from the checkpoint; no ``velocity`` command.
    2. ``fix boxdump`` uses ``append yes`` — box dims are appended to the
       existing ``npt_dump`` rather than overwriting it.
    3. Only Stage 2 (production) runs — Stage 1 heating is skipped.

    Intended as the ``restart_commands`` LAMMPS input for
    :func:`~models.create_slurm.write_chained_slurm_job`.

    Parameters
    ----------
    restart_file : str
        Path to ``*_after_heat.restart`` produced by :func:`write_npt_script`.
    npt_dump : str
        Path to the ``ave/time`` box-dimension dump file
        (``npt_boxdims_{T}K.dat``).  Box dims are appended to this file.
    out_path : str
        Destination path for the ``.lammps`` input script.
    pair_style : str
        LAMMPS ``pair_style`` keyword.
    mace_model : str
        Full path to the MACE ``.pt`` model file.
    pair_suffix : str
        Suffix for ``pair_style``.
    elem_str : str
        Element string for ``pair_coeff * *``.
    target_t : float, optional
        Target temperature in K. Default ``300``.
    timestep : float, optional
        MD timestep in ps. Default ``0.001`` (1 fs).
    thermo_damp : float, optional
        Nosé-Hoover thermostat damping time in ps. Default ``0.1``.
    baro_damp : float, optional
        Parrinello-Rahman barostat damping time in ps. Default ``1.0``.
    npt_steps : int, optional
        Steps for constant-T NPT production. Default ``200000``.
    dump_every : int, optional
        Frequency for box-dimension output. Default ``100``.
    supercell_reps : int, optional
        Unit cell repetitions along one axis for ``a0`` calculation.
        Default ``5``.
    restart_dir : str or None, optional
        Directory for periodic binary restart files.  If ``None``, no
        restart directive is written. Default ``None``.
    restart_every : int, optional
        Steps between restart file writes. Default ``10000``.

    Returns
    -------
    out_path : str
        Path to the written ``.lammps`` script.
    """
    pair           = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh          = _neighbor_block()
    restart        = _restart_line(restart_dir, restart_every,
                                   label=f'npt_{target_t}K_rst')
    thermo_damp_fs = thermo_damp * 1000
    baro_damp_fs   = baro_damp   * 1000
    npt_ps         = npt_steps   * timestep
    _final_stem    = _stem(min_output) if min_output else _stem(npt_dump)

    script = f"""# ════════════════════════════════════════════════════
# LAMMPS NPT RESTART — Stage 2 only at {target_t} K
# Notebook 02 (continuation leg)
# Reads *_after_heat.restart; appends box dims to existing dump
# ════════════════════════════════════════════════════

units          metal
atom_style     atomic
newton         on
boundary       p p p

# Velocities and thermostat state are restored from the checkpoint.
# No velocity command needed.
read_restart   {restart_file}

{restart}

{pair}

{neigh}

timestep       {timestep}

thermo         {dump_every}
thermo_style   custom step time temp pe ke press vol lx ly lz

# ── Stage 2: NPT production at {target_t} K ({npt_ps:.0f} ps) ────────────
fix            npt_run all npt temp {target_t}.0 {target_t}.0 {thermo_damp_fs:.1f} &
               iso 0.0 0.0 {baro_damp_fs:.1f}

variable       lx_val  equal  lx
variable       ly_val  equal  ly
variable       lz_val  equal  lz
fix            boxdump all ave/time 1 1 {dump_every} v_lx_val v_ly_val v_lz_val &
               file {npt_dump}  append yes

run            {npt_steps}
unfix          npt_run
unfix          boxdump
print "### NPT restart complete at {target_t} K ###"

variable       a0_npt  equal  lx/{supercell_reps}.0
print "NPT_RESULT: a0_at_{target_t}K = ${{a0_npt}} Angstrom"

write_restart  {_stem(npt_dump)}_final.restart
write_data     {_final_stem}_npt_final_{target_t}K.lammps
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 3. SURFACE RELAXATION  (NB04, NB04b)
# ═══════════════════════════════════════════════════════════════════════════

def write_surface_relaxation_script(
    slab_input,
    slab_relaxed,
    relax_thermo,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    target_t=300,
    z_freeze_cutoff=None,
    timestep=0.0005,
    thermo_damp=0.05,
    etol=0.0,
    ftol=1e-6,
    maxiter=10000,
    maxeval=100000,
    heat_steps=10000,
    nvt_steps=100000,
    thermo_freq=1000,
    restart_dir=None,
    restart_every=10000,
):
    """
    Write a three-phase surface relaxation script:
    frozen-layer setup → CG minimization → velocity-rescale heating → NVT.

    Each phase ends with both ``write_restart`` and ``write_data`` so the
    structure is checkpointed incrementally.  A periodic ``restart``
    directive is also written at the top of the script when
    ``restart_dir`` is provided.

    Used in NB04 and NB04b.

    Parameters
    ----------
    slab_input : str
        Path to the LAMMPS slab data file from NB03.
    slab_relaxed : str
        Base path for the relaxed slab output.  Phase outputs are
        derived by appending ``_phase1_min``, ``_phase2_heat``, and the
        final structure uses this path directly.
    relax_thermo : str
        Path for the ``ave/time`` temperature output file.
    out_path : str
        Destination path for the ``.lammps`` input script.
    pair_style : str
        LAMMPS ``pair_style`` keyword.
    mace_model : str
        Full path to the MACE ``.pt`` model file.
    pair_suffix : str
        Suffix for ``pair_style``.
    elem_str : str
        Element string for ``pair_coeff * *``.
    target_t : float, optional
        NVT target temperature in K. Default ``300``.
    z_freeze_cutoff : float or None, optional
        Atoms with z ≤ this value (Å) are frozen with
        ``fix setforce 0.0 0.0 0.0``.  If ``None``, a placeholder is
        written and must be filled manually. Default ``None``.
    timestep : float, optional
        MD timestep in ps. Default ``0.0005`` (0.5 fs).
    thermo_damp : float, optional
        Nosé-Hoover thermostat damping time in ps. Default ``0.05``.
    etol : float, optional
        Energy tolerance for ``minimize``. Default ``0.0``.
    ftol : float, optional
        Force tolerance (eV/Å) for ``minimize``. Default ``1e-6``.
    maxiter : int, optional
        Maximum minimization iterations. Default ``10000``.
    maxeval : int, optional
        Maximum force evaluations. Default ``100000``.
    heat_steps : int, optional
        Steps for velocity-rescale heating. Default ``10000``.
    nvt_steps : int, optional
        Steps for NVT equilibration. Default ``100000``.
    thermo_freq : int, optional
        Thermo output frequency (steps). Default ``1000``.
    restart_dir : str or None, optional
        Directory for periodic binary restart files.  If ``None``, no
        periodic restart directive is written. Default ``None``.
    restart_every : int, optional
        Steps between periodic restart file writes. Default ``10000``.

    Returns
    -------
    out_path : str
        Path to the written ``.lammps`` script.
    """
    pair           = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh          = _neighbor_block()
    restart        = _restart_line(restart_dir, restart_every,
                                   label=f'surf_{target_t}K')
    thermo_damp_fs = thermo_damp * 1000
    z_cut          = z_freeze_cutoff if z_freeze_cutoff is not None else 'FILL_IN_Z_CUTOFF'
    stem           = _stem(slab_relaxed)

    # Per-phase output paths
    phase1_data      = f'{stem}_phase1_min.lammps'
    phase1_restart   = f'{stem}_phase1_min.restart'
    phase2_data      = f'{stem}_phase2_heat.lammps'
    phase2_restart   = f'{stem}_phase2_heat.restart'
    phase3_data      = f'{stem}_phase3_nvt.lammps'
    phase3_restart   = f'{stem}_phase3_nvt.restart'
    phase4_restart   = f'{stem}_phase4_quench.restart'
    traj_file        = f'{stem}_nvt_traj.lammpstrj'
    min_traj_file    = f'{stem}_min_traj.lammpstrj'
    heat_traj_file   = f'{stem}_heat_traj.lammpstrj'
    quench_traj_file = f'{stem}_phase4_quench_traj.lammpstrj'

    script = f"""# ═══════════════════════════════════════════════════════
# LAMMPS Surface Relaxation — Hastelloy N
# Protocol: freeze bottom layers → minimize → heat → NVT
# T = {target_t} K | boundary p p f
# ═══════════════════════════════════════════════════════

units          metal
atom_style     atomic
newton         on
boundary       p p f

read_data      {slab_input}

{restart}

{pair}

{neigh}

# ── Frozen layers (z ≤ {z_cut} Å) ────────────────────────────────
region         frozen_reg    block  INF INF  INF INF  EDGE  {z_cut}
group          frozen_atoms  region  frozen_reg
group          free_atoms    subtract  all  frozen_atoms

variable       frozen_count  equal  count(frozen_atoms)
variable       free_count    equal  count(free_atoms)
print "GROUPS: frozen=${{frozen_count}}  free=${{free_count}}"

fix            freeze  frozen_atoms  setforce  0.0  0.0  0.0

thermo         {thermo_freq}
thermo_style   custom  step  time  temp  pe  ke  press  lx  ly  lz

# ═══ PHASE 1: Minimization ═══════════════════════════════════════
print "### Phase 1: Surface minimization ###"

variable       z_top_before  equal  bound(free_atoms,zmax)
print "  Top layer z BEFORE minimize: ${{z_top_before}} Ang"

dump           min_traj  free_atoms  custom  {thermo_freq}  {min_traj_file}  id type x y z
dump_modify    min_traj  sort id

minimize       {etol}  {ftol}  {maxiter}  {maxeval}

undump         min_traj

variable       z_top_after_min  equal  bound(free_atoms,zmax)
variable       dz_min           equal  v_z_top_before - v_z_top_after_min
print "  Top layer z AFTER minimize : ${{z_top_after_min}} Ang"
print "  Surface contraction (min)  : ${{dz_min}} Ang"
print "### Phase 1 complete ###"

write_restart  {phase1_restart}
write_data     {phase1_data}

# ═══ PHASE 2: Heat to {target_t} K ═══════════════════════════════
print "### Phase 2: Heating to {target_t} K ###"

velocity       free_atoms  create  {target_t}.0  12345  mom yes  rot yes  dist gaussian
velocity       free_atoms  scale   {target_t}.0

timestep       {timestep}

compute        surf_temp_heat  free_atoms  temp
thermo_modify  temp  surf_temp_heat

fix            nve_heat   free_atoms  nve
fix            rescale    free_atoms  temp/rescale  10  {target_t}.0  {target_t}.0  10.0  1.0

dump           heat_traj  free_atoms  custom  {thermo_freq}  {heat_traj_file}  id type x y z
dump_modify    heat_traj  sort id

run            {heat_steps}
undump         heat_traj
unfix          rescale
unfix          nve_heat
thermo_modify  temp  thermo_temp
uncompute      surf_temp_heat
print "### Phase 2 complete ###"

write_restart  {phase2_restart}
write_data     {phase2_data}

# ═══ PHASE 3: NVT Equilibration at {target_t} K ══════════════════
print "### Phase 3: NVT equilibration at {target_t} K ###"

compute        surf_temp  free_atoms  temp
thermo_modify  temp  surf_temp

fix            thermo_out  all  ave/time  1  1  {thermo_freq}  &
               c_surf_temp  &
               file  {relax_thermo}  mode scalar

dump           surf_traj  free_atoms  custom  {thermo_freq}  {traj_file}  id type x y z
dump_modify    surf_traj  sort id

fix            nvt_equil  free_atoms  nvt  temp  {target_t}.0  {target_t}.0  {thermo_damp_fs:.1f}
run            {nvt_steps}
unfix          nvt_equil
undump         surf_traj
unfix          thermo_out
print "### Phase 3 complete ###"

write_restart  {phase3_restart}
write_data     {phase3_data}

# ═══ PHASE 4: Quench (CG minimization of NVT snapshot) ══════════
print "### Phase 4: Quench minimization ###"

variable       z_top_after_nvt  equal  bound(free_atoms,zmax)
print "  z top BEFORE quench: ${{z_top_after_nvt}} Ang"

dump           quench_traj  free_atoms  custom  {thermo_freq}  {quench_traj_file}  id type x y z
dump_modify    quench_traj  sort id

minimize       {etol}  {ftol}  {maxiter}  {maxeval}

undump         quench_traj
print "### Phase 4 complete ###"

# ═══ Results ═════════════════════════════════════════════════════
variable       z_top_final   equal  bound(free_atoms,zmax)
variable       pe_final_val  equal  pe

print "RELAXATION_RESULTS_START"
print "  z_top_before_Ang       : ${{z_top_before}}"
print "  z_top_after_min_Ang    : ${{z_top_after_min}}"
print "  surface_contraction    : ${{dz_min}}"
print "  z_top_after_nvt_Ang    : ${{z_top_after_nvt}}"
print "  z_top_after_quench_Ang : ${{z_top_final}}"
print "  pe_final_eV            : ${{pe_final_val}}"
print "RELAXATION_RESULTS_END"

unfix          freeze
write_restart  {phase4_restart}
write_data     {slab_relaxed}
print "Relaxed slab written: {slab_relaxed}"
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


def write_surface_relaxation_restart_script(
    restart_file,
    slab_relaxed,
    relax_thermo,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    target_t=300,
    z_freeze_cutoff=None,
    timestep=0.0005,
    thermo_damp=0.05,
    etol=0.0,
    ftol=1e-6,
    maxiter=10000,
    maxeval=100000,
    nvt_steps=100000,
    thermo_freq=1000,
    restart_dir=None,
    restart_every=10000,
):
    """Write a LAMMPS surface relaxation continuation script (Phase 3 NVT only).

    Reads the latest periodic restart from *restart_file* (a glob pattern
    matching ``restarts/surf_300K.*.restart``) and re-runs Phase 3 NVT,
    appending to the existing trajectory and thermo files.

    Mirrors :func:`write_surface_relaxation_script` Phase 3 exactly with
    three differences:

    1. ``read_restart`` replaces all three phases' setup — velocities and
       thermostat state are restored; no ``velocity`` or ``minimize`` needed.
    2. The NVT dump uses ``append yes``.
    3. The ``fix ave/time`` thermo file uses ``append yes``.

    Intended as the ``restart_commands`` LAMMPS input for
    :func:`~models.create_slurm.write_chained_slurm_job`.

    Parameters
    ----------
    restart_file : str
        Glob-style path to the periodic LAMMPS restart file, e.g.
        ``'restarts/surf_300K.*.restart'``.  LAMMPS picks the most recent.
    slab_relaxed : str
        Path for the final relaxed slab output (``write_data``).
    relax_thermo : str
        Path to the existing ``ave/time`` temperature file. Appended to.
    out_path : str
        Destination path for the ``.lammps`` input script.
    pair_style : str
        LAMMPS ``pair_style`` keyword.
    mace_model : str
        Full path to the MACE ``.pt`` model file.
    pair_suffix : str
        Suffix for ``pair_style``.
    elem_str : str
        Element string for ``pair_coeff * *``.
    target_t : float, optional
        NVT target temperature in K. Default ``300``.
    z_freeze_cutoff : float or None, optional
        Z-coordinate threshold for frozen-layer group. Default ``None``.
    timestep : float, optional
        MD timestep in ps. Default ``0.0005`` (0.5 fs).
    thermo_damp : float, optional
        Nosé-Hoover thermostat damping time in ps. Default ``0.05``.
    nvt_steps : int, optional
        Steps for Phase 3 NVT. Default ``100000``.
    thermo_freq : int, optional
        Thermo output frequency. Default ``1000``.
    restart_dir : str or None, optional
        Directory for periodic binary restart files. Default ``None``.
    restart_every : int, optional
        Steps between periodic restart writes. Default ``10000``.

    Returns
    -------
    out_path : str
        Path to the written ``.lammps`` script.
    """
    pair             = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh            = _neighbor_block()
    restart          = _restart_line(restart_dir, restart_every,
                                     label=f'surf_{target_t}K')
    thermo_damp_fs   = thermo_damp * 1000
    z_cut            = z_freeze_cutoff if z_freeze_cutoff is not None else 'FILL_IN_Z_CUTOFF'
    stem             = _stem(slab_relaxed)

    phase3_data      = f'{stem}_phase3_nvt.lammps'
    phase3_restart   = f'{stem}_phase3_nvt.restart'
    phase4_restart   = f'{stem}_phase4_quench.restart'
    traj_file        = f'{stem}_nvt_traj.lammpstrj'
    quench_traj_file = f'{stem}_phase4_quench_traj.lammpstrj'

    script = f"""# ═══════════════════════════════════════════════════════
# LAMMPS Surface Relaxation RESTART — Phase 3 NVT only
# T = {target_t} K | boundary p p f (continuation leg)
# Reads latest periodic restart; appends to existing trajectory
# ═══════════════════════════════════════════════════════

units          metal
atom_style     atomic
newton         on
boundary       p p f

# Velocities and thermostat state are restored from the checkpoint.
# No velocity command needed.
read_restart   {restart_file}

{restart}

{pair}

{neigh}

# ── Frozen layers (z ≤ {z_cut} Å) ────────────────────────────────
region         frozen_reg    block  INF INF  INF INF  EDGE  {z_cut}
group          frozen_atoms  region  frozen_reg
group          free_atoms    subtract  all  frozen_atoms

fix            freeze  frozen_atoms  setforce  0.0  0.0  0.0

thermo         {thermo_freq}
thermo_style   custom  step  time  temp  pe  ke  press  lx  ly  lz

timestep       {timestep}

# ═══ PHASE 3: NVT Equilibration at {target_t} K (restart) ════════
print "### Phase 3 NVT restart at {target_t} K ###"

compute        surf_temp  free_atoms  temp
thermo_modify  temp  surf_temp

# append yes — rows continue from where the previous leg stopped
fix            thermo_out  all  ave/time  1  1  {thermo_freq}  &
               c_surf_temp  &
               file  {relax_thermo}  mode scalar  append yes

# append yes — frames added after those from the previous leg
dump           surf_traj  free_atoms  custom  {thermo_freq}  {traj_file}  id type x y z
dump_modify    surf_traj  sort id  append yes

fix            nvt_equil  free_atoms  nvt  temp  {target_t}.0  {target_t}.0  {thermo_damp_fs:.1f}
run            {nvt_steps}
unfix          nvt_equil
undump         surf_traj
unfix          thermo_out
print "### Phase 3 restart complete ###"

write_restart  {phase3_restart}
write_data     {phase3_data}

# ═══ PHASE 4: Quench (CG minimization of NVT snapshot) ══════════
print "### Phase 4: Quench minimization ###"

variable       z_top_after_nvt  equal  bound(free_atoms,zmax)
print "  z top BEFORE quench: ${{z_top_after_nvt}} Ang"

dump           quench_traj  free_atoms  custom  {thermo_freq}  {quench_traj_file}  id type x y z
dump_modify    quench_traj  sort id

minimize       {etol}  {ftol}  {maxiter}  {maxeval}

undump         quench_traj
print "### Phase 4 complete ###"

# ═══ Results ═════════════════════════════════════════════════════
variable       z_top_final   equal  bound(free_atoms,zmax)
variable       pe_final_val  equal  pe

print "RELAXATION_RESULTS_START"
print "  z_top_after_nvt_Ang    : ${{z_top_after_nvt}}"
print "  z_top_after_quench_Ang : ${{z_top_final}}"
print "  pe_final_eV            : ${{pe_final_val}}"
print "RELAXATION_RESULTS_END"

unfix          freeze
write_restart  {phase4_restart}
write_data     {slab_relaxed}
print "Relaxed slab written: {slab_relaxed}"
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 4. NVT BULK EQUILIBRATION + H MSD TRACKING  (NB10)
# ═══════════════════════════════════════════════════════════════════════════

def write_nvt_bulk_script(
    bulk_h_file,
    traj_file,
    out_file,
    msd_file,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    temperature,
    h_type=9,
    timestep=0.0005,
    tau_t=0.1,
    n_equil=500000,
    n_prod=1500000,
    thermo_every=1000,
    dump_every=1000,
    velocity_seed=42,
    restart_dir=None,
    restart_every=10000,
):
    """
    Write a two-phase NVT bulk equilibration script with H-atom MSD tracking.

    Phase 1 equilibrates the system; Phase 2 runs production and dumps
    the full trajectory for NB11 MSD analysis.  Both phases end with
    ``write_restart`` and ``write_data``.

    Used in NB10 for each temperature in the Arrhenius series.

    Parameters
    ----------
    bulk_h_file : str
        Path to the bulk + H LAMMPS data file from the H insertion cell.
    traj_file : str
        Path for the production trajectory dump (``.lammpstrj``).
    out_file : str
        Path where the final structure is written (``write_data``).
    msd_file : str
        Path for the H-atom MSD time series output.
    out_path : str
        Destination path for the ``.lammps`` input script.
    pair_style : str
        LAMMPS ``pair_style`` keyword.
    mace_model : str
        Full path to the MACE ``.pt`` model file.
    pair_suffix : str
        Suffix for ``pair_style``.
    elem_str : str
        Element string for ``pair_coeff * *``.
    temperature : float
        NVT temperature in K.
    h_type : int, optional
        LAMMPS atom type number for H. Default ``9``.
    timestep : float, optional
        MD timestep in ps. Default ``0.0005`` (0.5 fs).
    tau_t : float, optional
        Nosé-Hoover thermostat damping time in ps. Default ``0.1``.
    n_equil : int, optional
        Number of equilibration steps (Phase 1). Default ``500000``.
    n_prod : int, optional
        Number of production steps (Phase 2). Default ``1500000``.
    thermo_every : int, optional
        Thermo output frequency. Default ``1000``.
    dump_every : int, optional
        Trajectory dump frequency. Default ``1000``.
    velocity_seed : int, optional
        Random seed for velocity initialisation. Default ``42``.
    restart_dir : str or None, optional
        Directory for periodic binary restart files.  If ``None``, no
        restart directive is written. Default ``None``.
    restart_every : int, optional
        Steps between periodic restart file writes. Default ``10000``.

    Returns
    -------
    out_path : str
        Path to the written ``.lammps`` script.
    """
    pair      = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh     = _neighbor_block()
    restart   = _restart_line(restart_dir, restart_every,
                               label=f'nvt_{temperature}K')
    tau_t_fs  = tau_t * 1000
    equil_ps  = n_equil * timestep
    prod_ps   = n_prod  * timestep
    stem      = _stem(out_file)

    phase1_data    = f'{stem}_phase1_equil.lammps'
    phase1_restart = f'{stem}_phase1_equil.restart'
    phase2_restart = f'{stem}_phase2_prod.restart'
    equil_traj     = f'{stem}_phase1_equil.lammpstrj'

    script = f"""# LAMMPS NVT bulk equilibration — Hastelloy N + H
# T = {temperature} K | Notebook 10
# Phase 1: {equil_ps:.0f} ps equilibration
# Phase 2: {prod_ps:.0f} ps production (trajectory for NB11 MSD)

units          metal
atom_style     atomic
newton         on
boundary       p p p

read_data      {bulk_h_file}

{restart}

{pair}

{neigh}

group          H_atom  type {h_type}
group          metal   subtract all H_atom

thermo         {thermo_every}
thermo_style   custom step temp pe ke etotal press vol

timestep       {timestep}

velocity       all create {temperature}.0 {velocity_seed} mom yes rot yes dist gaussian

# ═══ PHASE 1: Equilibration ({equil_ps:.0f} ps) ═══════════════════════════
fix            nvt_equil  all  nvt  temp  {temperature}.0  {temperature}.0  {tau_t_fs:.1f}

dump           equil_dump  all  custom  {dump_every}  {equil_traj}  id type x y z
dump_modify    equil_dump  sort id

print "### Phase 1: Equilibration {equil_ps:.0f} ps at {temperature} K ###"
run            {n_equil}
undump         equil_dump
unfix          nvt_equil
print "### Phase 1 complete ###"

write_restart  {phase1_restart}
write_data     {phase1_data}

# ═══ PHASE 2: Production ({prod_ps:.0f} ps) ════════════════════════════════
dump           prod_dump  all  custom  {dump_every}  {traj_file} &
               id type x y z
dump_modify    prod_dump  sort id

fix            nvt_prod  all  nvt  temp  {temperature}.0  {temperature}.0  {tau_t_fs:.1f}

compute        msd_H    H_atom  msd
fix            msd_out  all  ave/time  1  1  {thermo_every} &
               c_msd_H[4]  file  {msd_file}  mode scalar

print "### Phase 2: Production {prod_ps:.0f} ps at {temperature} K ###"
run            {n_prod}
print "### Phase 2 complete ###"

variable  pe_final   equal  pe
variable  temp_fin   equal  temp
variable  press_fin  equal  press

print "EQUIL_RESULTS_START"
print "  T_K          : {temperature}"
print "  pe_final_eV  : ${{pe_final}}"
print "  temp_final_K : ${{temp_fin}}"
print "  press_final  : ${{press_fin}}"
print "EQUIL_RESULTS_END"

write_restart  {phase2_restart}
write_data     {out_file}
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 5. NVT BULK RESTART  (NB10 — continuation legs for chained jobs)
# ═══════════════════════════════════════════════════════════════════════════

def write_nvt_bulk_restart_script(
    restart_file,
    traj_file,
    out_file,
    msd_file,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    temperature,
    h_type=9,
    timestep=0.0005,
    tau_t=0.1,
    n_equil=500000,
    n_prod=1500000,
    thermo_every=1000,
    dump_every=1000,
    restart_dir=None,
    restart_every=10000,
):
    """
    Write a NVT bulk continuation script that resumes from a LAMMPS
    binary restart file.

    Mirrors write_nvt_bulk_script exactly, with three differences:
    1. read_restart instead of read_data (velocities restored from checkpoint).
    2. dump_modify append yes  — trajectory frames appended, not overwritten.
    3. fix ave/time append yes — MSD rows appended, not overwritten.

    Use this as the restart_commands leg in write_chained_slurm_job.

    Parameters
    ----------
    restart_file : str
        Glob path to the LAMMPS binary restart, e.g.
        'results/500K/nvt_500K.*.restart'.
    traj_file : str
        Path to the existing .lammpstrj; opened with append yes.
    out_file : str
        Path where the final structure is written (write_data).
    msd_file : str
        Path to the existing MSD time series; opened with append yes.
    out_path : str
        Destination path for the .lammps script.
    pair_style, mace_model, pair_suffix, elem_str : str
        Same as write_nvt_bulk_script.
    temperature : float
        NVT temperature in K.
    h_type : int, optional
        LAMMPS atom type number for H. Default 9.
    timestep : float, optional
        MD timestep in ps. Default 0.0005.
    tau_t : float, optional
        Nose-Hoover damping time in ps. Default 0.1.
    n_equil : int, optional
        Equilibration steps. Default 500000.
    n_prod : int, optional
        Production steps. Default 1500000.
    thermo_every : int, optional
        Thermo output frequency. Default 1000.
    dump_every : int, optional
        Trajectory dump frequency. Default 1000.
    restart_dir : str or None, optional
        Directory for periodic restart files. Default None.
    restart_every : int, optional
        Steps between periodic restart writes. Default 10000.

    Returns
    -------
    out_path : str
    """
    pair      = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh     = _neighbor_block()
    restart   = _restart_line(restart_dir, restart_every,
                               label=f'nvt_{temperature}K')
    tau_t_fs  = tau_t * 1000
    equil_ps  = n_equil * timestep
    prod_ps   = n_prod  * timestep
    stem      = _stem(out_file)

    phase1_data    = f'{stem}_phase1_equil.lammps'
    phase1_restart = f'{stem}_phase1_equil.restart'
    phase2_restart = f'{stem}_phase2_prod.restart'
    equil_traj     = f'{stem}_phase1_equil.lammpstrj'

    script = (
        "# LAMMPS NVT bulk RESTART — Hastelloy N + H\n"
        f"# T = {temperature} K | Notebook 10 (continuation leg)\n"
        f"# Phase 1: {equil_ps:.0f} ps equilibration (resumed from checkpoint)\n"
        f"# Phase 2: {prod_ps:.0f} ps production   (trajectory appended)\n"
        "\n"
        "units          metal\n"
        "atom_style     atomic\n"
        "newton         on\n"
        "boundary       p p p\n"
        "\n"
        "# Resume from most recent checkpoint.\n"
        "# Velocities and thermostat state restored; no velocity command needed.\n"
        f"read_restart   {restart_file}\n"
        "\n"
        f"{restart}\n"
        "\n"
        f"{pair}\n"
        "\n"
        f"{neigh}\n"
        "\n"
        f"group          H_atom  type {h_type}\n"
        "group          metal   subtract all H_atom\n"
        "\n"
        f"thermo         {thermo_every}\n"
        "thermo_style   custom step temp pe ke etotal press vol\n"
        "\n"
        f"timestep       {timestep}\n"
        "\n"
        f"# === PHASE 1: Equilibration ({equil_ps:.0f} ps) ===\n"
        f"fix            nvt_equil  all  nvt  temp  {temperature}.0  {temperature}.0  {tau_t_fs:.1f}\n"
        "\n"
        f"dump           equil_dump  all  custom  {dump_every}  {equil_traj}  id type x y z\n"
        "dump_modify    equil_dump  sort id  append yes\n"
        "\n"
        f'print "### Phase 1: Equilibration {equil_ps:.0f} ps at {temperature} K (restart) ###"\n'
        f"run            {n_equil}\n"
        "undump         equil_dump\n"
        "unfix          nvt_equil\n"
        'print "### Phase 1 complete ###"\n'
        "\n"
        f"write_restart  {phase1_restart}\n"
        f"write_data     {phase1_data}\n"
        "\n"
        f"# === PHASE 2: Production ({prod_ps:.0f} ps) ===\n"
        "# append yes — frames added after those from the previous leg\n"
        f"dump           prod_dump  all  custom  {dump_every}  {traj_file} &\n"
        "               id type x y z\n"
        "dump_modify    prod_dump  sort id  append yes\n"
        "\n"
        f"fix            nvt_prod  all  nvt  temp  {temperature}.0  {temperature}.0  {tau_t_fs:.1f}\n"
        "\n"
        "compute        msd_H    H_atom  msd\n"
        "# append yes — MSD rows continue from where the previous leg left off\n"
        f"fix            msd_out  all  ave/time  1  1  {thermo_every} &\n"
        f"               c_msd_H[4]  file  {msd_file}  mode scalar  append yes\n"
        "\n"
        f'print "### Phase 2: Production {prod_ps:.0f} ps at {temperature} K (restart) ###"\n'
        f"run            {n_prod}\n"
        'print "### Phase 2 complete ###"\n'
        "\n"
        "variable  pe_final   equal  pe\n"
        "variable  temp_fin   equal  temp\n"
        "variable  press_fin  equal  press\n"
        "\n"
        'print "EQUIL_RESULTS_START"\n'
        f'print "  T_K          : {temperature}"\n'
        'print "  pe_final_eV  : ${pe_final}"\n'
        'print "  temp_final_K : ${temp_fin}"\n'
        'print "  press_final  : ${press_fin}"\n'
        'print "EQUIL_RESULTS_END"\n'
        "\n"
        f"write_restart  {phase2_restart}\n"
        f"write_data     {out_file}\n"
    )

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 6. ADSORBATE CG MINIMIZATION  (NB05, NB06)
# ═══════════════════════════════════════════════════════════════════════════

def write_adsorbate_min_script(
    slab_ads_input,
    ads_output,
    out_path,
    pair_style,
    mace_model,
    pair_suffix,
    elem_str,
    z_freeze_cutoff,
    etol=0.0,
    ftol=1e-6,
    maxiter=10000,
    maxeval=100000,
    thermo_freq=100,
    traj_file=None,
):
    """
    Write a fixed-box CG minimization script for a slab + adsorbate system.

    Unlike ``write_minimization_script`` (which relaxes the box with
    ``fix box/relax``) and ``write_surface_relaxation_script`` (which runs
    3-phase MD), this function minimizes atomic positions only in a fixed
    periodic-in-xy, free-in-z slab geometry.  Bottom layers are frozen
    via ``fix setforce 0.0 0.0 0.0``.

    Used in NB05 (H2*) and NB06 (H*) to compute adsorption energies for
    every site enumerated in Section A Phase 3.

    Parameters
    ----------
    slab_ads_input : str
        LAMMPS data file containing the slab with the adsorbate placed above
        a site (written by ``structure.add_adsorbate``).
    ads_output : str
        Path where the CG-relaxed slab+adsorbate structure is written.
    out_path : str
        Destination path for the ``.in`` LAMMPS input script.
    pair_style : str
        LAMMPS ``pair_style`` keyword, e.g. ``'mliap unified'``.
    mace_model : str
        Full path to the MACE ``.pt`` model file.
    pair_suffix : str
        Suffix for ``pair_style``, e.g. ``'0'``.
    elem_str : str
        Space-separated element string for ``pair_coeff * *``.
    z_freeze_cutoff : float
        Atoms with z ≤ this value (Å) are frozen with
        ``fix setforce 0.0 0.0 0.0``.  Should match the value used in
        surface relaxation (e.g. ``Z_FREEZE_CUTOFF = 22.115`` Å).
    etol : float, optional
        Energy tolerance for ``minimize``. Default ``0.0``.
    ftol : float, optional
        Force tolerance (eV/Å) for ``minimize``. Default ``1e-6``.
    maxiter : int, optional
        Maximum minimization iterations. Default ``10000``.
    maxeval : int, optional
        Maximum force evaluations. Default ``100000``.
    thermo_freq : int, optional
        Thermo output frequency (steps). Default ``100``.

    Returns
    -------
    out_path : str
        Path to the written ``.in`` script.
    """
    pair  = _pair_block(pair_style, mace_model, pair_suffix, elem_str)
    neigh = _neighbor_block()
    z_cut = z_freeze_cutoff
    if traj_file:
        dump_block   = (f'dump           ads_traj  all  custom  {thermo_freq}  {traj_file}'
                        f'  id type x y z\ndump_modify    ads_traj  sort id\n\n')
        undump_block = 'undump         ads_traj\n\n'
    else:
        dump_block = undump_block = ''

    script = f"""# ═══════════════════════════════════════════════════════
# LAMMPS Adsorbate CG Minimization — Hastelloy N Slab
# Section B (NB05/NB06) | fixed box | boundary p p f
# pair_style: mliap unified
# ═══════════════════════════════════════════════════════

units          metal
atom_style     atomic
newton         on
boundary       p p f

read_data      {slab_ads_input}

{pair}

{neigh}

# ── Freeze bottom layers (z ≤ {z_cut} Å) ─────────────────────────────
region         frozen_reg   block  INF INF  INF INF  EDGE  {z_cut}
group          frozen_atoms region frozen_reg
group          free_atoms   subtract all frozen_atoms

fix            freeze  frozen_atoms  setforce  0.0  0.0  0.0

thermo         {thermo_freq}
thermo_style   custom  step  pe  fmax  fnorm  press

print "### Adsorbate minimization ###"

{dump_block}minimize       {etol}  {ftol}  {maxiter}  {maxeval}

print "### Minimization complete ###"
{undump_block}

variable       pe_final  equal  pe
variable       fmax_f    equal  fmax
variable       natoms    equal  atoms

print "  pe_final_eV     : ${{pe_final}}"
print "  fmax_eV_per_Ang : ${{fmax_f}}"
print "  natoms          : ${{natoms}}"

unfix          freeze
write_data     {ads_output}
print "Relaxed adsorbate structure written to {ads_output}"
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script)
    return out_path
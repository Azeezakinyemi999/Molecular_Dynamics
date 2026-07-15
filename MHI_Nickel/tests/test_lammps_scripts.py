"""
tests/test_lammps_scripts.py
============================
Tests for models/lammps_script.py — offline, no LAMMPS required.

All functions write text files; tests read the content back and assert
on specific LAMMPS directives.
"""

import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.lammps_script import (
    _pair_block,
    _neighbor_block,
    _restart_line,
    _stem,
    write_minimization_script,
    write_minimization_restart_script,
    write_npt_script,
    write_npt_restart_script,
    write_surface_relaxation_script,
    write_surface_relaxation_restart_script,
    write_nvt_bulk_script,
    write_nvt_bulk_restart_script,
    write_nvt_equil_restart_script,
    write_nvt_prod_restart_script,
    write_adsorbate_min_script,
)


_PS   = 'mliap unified'
_MOD  = '/models/mace.pt'
_SUFF = '0'
_ELEM = 'Ni Cr Mo Fe'


def _read(path):
    return pathlib.Path(path).read_text()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Private helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_pair_block_pair_style_line(self):
        block = _pair_block(_PS, _MOD, _SUFF, _ELEM)
        assert f'pair_style     {_PS} {_MOD} {_SUFF}' in block

    def test_pair_block_pair_coeff_line(self):
        block = _pair_block(_PS, _MOD, _SUFF, _ELEM)
        assert f'pair_coeff     * * {_ELEM}' in block

    def test_neighbor_block_contains_bin(self):
        assert 'neighbor       2.0 bin' in _neighbor_block()

    def test_neighbor_block_contains_neigh_modify(self):
        assert 'neigh_modify   every 1 delay 0 check yes' in _neighbor_block()

    def test_restart_line_none_returns_empty(self):
        assert _restart_line(None, 5000) == ''

    def test_restart_line_contains_dir_and_steps(self):
        line = _restart_line('/restarts', 5000, label='run')
        assert '/restarts/run.*.restart' in line
        assert '5000' in line

    def test_restart_line_custom_label(self):
        line = _restart_line('/r', 1000, label='npt_300K')
        assert 'npt_300K' in line

    def test_stem_strips_extension(self):
        assert _stem('/out/file.lammps') == '/out/file'

    def test_stem_no_extension_unchanged(self):
        assert _stem('/out/file') == '/out/file'


# ═══════════════════════════════════════════════════════════════════════════
# 2. write_minimization_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteMinimizationScript:

    @pytest.fixture()
    def min_script(self, tmp_path):
        p = str(tmp_path / 'min.lammps')
        write_minimization_script(
            bulk_input='/data/bulk.lammps',
            min_output='/data/bulk_min.lammps',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'min.lammps')
        ret = write_minimization_script(
            '/d/b', '/d/bm', p, _PS, _MOD, _SUFF, _ELEM,
        )
        assert ret == p

    def test_file_created(self, tmp_path):
        p = str(tmp_path / 'min2.lammps')
        write_minimization_script('/d/b', '/d/o', p, _PS, _MOD, _SUFF, _ELEM)
        assert pathlib.Path(p).exists()

    def test_units_metal(self, min_script):
        assert 'units          metal' in min_script

    def test_boundary_p_p_p(self, min_script):
        assert 'boundary       p p p' in min_script

    def test_read_data(self, min_script):
        assert 'read_data      /data/bulk.lammps' in min_script

    def test_box_relax_fix(self, min_script):
        assert 'fix            boxrelax all box/relax iso 0.0' in min_script

    def test_minimize_command_defaults(self, min_script):
        assert 'minimize       0.0 1e-08 50000 500000' in min_script

    def test_write_data(self, min_script):
        assert 'write_data     /data/bulk_min.lammps' in min_script

    def test_a0_variable_default_reps(self, min_script):
        assert 'a0_min    equal  lx/5' in min_script

    def test_no_periodic_restart_by_default(self, min_script):
        # write_restart appears but the periodic `restart        ` directive should not
        assert 'restart        ' not in min_script

    def test_restart_line_written_when_dir_given(self, tmp_path):
        p = str(tmp_path / 'min_r.lammps')
        write_minimization_script(
            '/d/b', '/d/o', p, _PS, _MOD, _SUFF, _ELEM,
            restart_dir='/restarts', restart_every=5000,
        )
        content = _read(p)
        assert '/restarts/min.*.restart' in content
        assert '5000' in content

    def test_masses_written_when_given(self, tmp_path):
        p = str(tmp_path / 'min_m.lammps')
        masses = {1: (58.69, 'Ni'), 2: (51.996, 'Cr')}
        write_minimization_script(
            '/d/b', '/d/o', p, _PS, _MOD, _SUFF, _ELEM, masses=masses,
        )
        content = _read(p)
        assert 'mass           1  58.69  # Ni' in content
        assert 'mass           2  51.996  # Cr' in content

    def test_no_mass_lines_when_masses_none(self, min_script):
        assert 'mass           ' not in min_script

    def test_traj_dump_written_when_traj_file_given(self, tmp_path):
        p = str(tmp_path / 'min_d.lammps')
        write_minimization_script(
            '/d/b', '/d/o', p, _PS, _MOD, _SUFF, _ELEM,
            traj_file='/traj/min.lammpstrj', dump_every=20,
        )
        content = _read(p)
        assert 'dump           min_traj' in content
        assert '/traj/min.lammpstrj' in content
        assert 'undump         min_traj' in content

    def test_no_traj_dump_by_default(self, min_script):
        assert 'dump           min_traj' not in min_script

    def test_custom_supercell_reps(self, tmp_path):
        p = str(tmp_path / 'min_reps.lammps')
        write_minimization_script(
            '/d/b', '/d/o', p, _PS, _MOD, _SUFF, _ELEM, supercell_reps=4,
        )
        assert 'a0_min    equal  lx/4' in _read(p)

    def test_minimization_results_block(self, min_script):
        assert 'MINIMIZATION_RESULTS_START' in min_script
        assert 'MINIMIZATION_RESULTS_END' in min_script


class TestWriteMinimizationRestartScript:

    @pytest.fixture()
    def min_rst_script(self, tmp_path):
        p = str(tmp_path / 'min_rst.lammps')
        write_minimization_restart_script(
            restart_file='/restarts/min.*.restart',
            min_output='/data/bulk_min.lammps',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'min_rst.lammps')
        ret = write_minimization_restart_script(
            '/r/f', '/d/o', p, _PS, _MOD, _SUFF, _ELEM,
        )
        assert ret == p

    def test_read_restart(self, min_rst_script):
        assert 'read_restart   /restarts/min.*.restart' in min_rst_script

    def test_no_read_data(self, min_rst_script):
        assert 'read_data' not in min_rst_script

    def test_no_mass_lines(self, min_rst_script):
        assert 'mass           ' not in min_rst_script

    def test_box_relax_fix_redeclared(self, min_rst_script):
        assert 'fix            boxrelax all box/relax iso 0.0' in min_rst_script

    def test_minimize_command_uses_full_budget(self, min_rst_script):
        # Same default budget as a fresh-start leg -- CG minimization has
        # no persistent state across separate `minimize` calls, so a
        # restart leg re-runs the full tolerance/iteration budget, not a
        # reduced "remaining" one.
        assert 'minimize       0.0 1e-08 50000 500000' in min_rst_script

    def test_write_data(self, min_rst_script):
        assert 'write_data     /data/bulk_min.lammps' in min_rst_script

    def test_write_restart_final(self, min_rst_script):
        assert 'write_restart  /data/bulk_min_final.restart' in min_rst_script

    def test_minimization_results_block(self, min_rst_script):
        assert 'MINIMIZATION_RESULTS_START' in min_rst_script
        assert 'MINIMIZATION_RESULTS_END' in min_rst_script

    def test_restart_line_written_when_dir_given(self, tmp_path):
        p = str(tmp_path / 'min_rst_r.lammps')
        write_minimization_restart_script(
            '/r/f', '/d/o', p, _PS, _MOD, _SUFF, _ELEM,
            restart_dir='/restarts', restart_every=2000,
        )
        content = _read(p)
        assert '/restarts/min.*.restart' in content
        assert '2000' in content

    def test_no_periodic_restart_by_default(self, min_rst_script):
        assert 'restart        ' not in min_rst_script

    def test_traj_dump_append_yes(self, tmp_path):
        p = str(tmp_path / 'min_rst_d.lammps')
        write_minimization_restart_script(
            '/r/f', '/d/o', p, _PS, _MOD, _SUFF, _ELEM,
            traj_file='/traj/min.lammpstrj', dump_every=20,
        )
        content = _read(p)
        assert 'dump           min_traj' in content
        assert 'dump_modify    min_traj  sort id  append yes' in content

    def test_no_traj_dump_by_default(self, min_rst_script):
        assert 'dump           min_traj' not in min_rst_script


# ═══════════════════════════════════════════════════════════════════════════
# 3. write_npt_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNptScript:

    @pytest.fixture()
    def npt_script(self, tmp_path):
        p = str(tmp_path / 'npt.lammps')
        write_npt_script(
            min_output='/data/bulk_min.lammps',
            npt_dump='/data/npt_boxdims.dat',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, target_t=600,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'npt.lammps')
        ret = write_npt_script('/d/m', '/d/n', p, _PS, _MOD, _SUFF, _ELEM)
        assert ret == p

    def test_read_data(self, npt_script):
        assert 'read_data      /data/bulk_min.lammps' in npt_script

    def test_velocity_command_present(self, npt_script):
        assert 'velocity       all create 10.0' in npt_script

    def test_stage1_heat_fix(self, npt_script):
        assert 'fix            heat all npt temp 10.0 600.0' in npt_script

    def test_stage2_npt_fix(self, npt_script):
        assert 'fix            npt_run all npt temp 600.0 600.0' in npt_script

    def test_boxdump_ave_time(self, npt_script):
        assert 'fix            boxdump all ave/time' in npt_script

    def test_boxdump_file_path(self, npt_script):
        assert 'file /data/npt_boxdims.dat' in npt_script

    def test_heat_run_steps(self, tmp_path):
        p = str(tmp_path / 'npt_hs.lammps')
        write_npt_script('/d/m', '/d/n', p, _PS, _MOD, _SUFF, _ELEM,
                         heat_steps=15000)
        assert 'run            15000' in _read(p)

    def test_npt_run_steps(self, tmp_path):
        p = str(tmp_path / 'npt_ns.lammps')
        write_npt_script('/d/m', '/d/n', p, _PS, _MOD, _SUFF, _ELEM,
                         npt_steps=100000)
        assert 'run            100000' in _read(p)

    def test_npt_result_print(self, npt_script):
        assert 'NPT_RESULT: a0_at_600K' in npt_script

    def test_restart_label_contains_temperature(self, tmp_path):
        p = str(tmp_path / 'npt_r.lammps')
        write_npt_script('/d/m', '/d/n', p, _PS, _MOD, _SUFF, _ELEM,
                         target_t=300, restart_dir='/restarts')
        assert 'npt_300K' in _read(p)

    def test_traj_dump_written_when_given(self, tmp_path):
        p = str(tmp_path / 'npt_t.lammps')
        write_npt_script('/d/m', '/d/n', p, _PS, _MOD, _SUFF, _ELEM,
                         traj_file='/traj/npt.lammpstrj')
        content = _read(p)
        assert 'dump           npt_traj' in content
        assert '/traj/npt.lammpstrj' in content

    def test_no_traj_dump_by_default(self, npt_script):
        assert 'dump           npt_traj' not in npt_script


# ═══════════════════════════════════════════════════════════════════════════
# 4. write_npt_restart_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNptRestartScript:

    @pytest.fixture()
    def npt_rst_script(self, tmp_path):
        p = str(tmp_path / 'npt_rst.lammps')
        write_npt_restart_script(
            restart_file='/restarts/npt_300K_after_heat.restart',
            npt_dump='/data/npt_boxdims.dat',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, target_t=300,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'npt_rst.lammps')
        ret = write_npt_restart_script(
            '/r/f', '/d/n', p, _PS, _MOD, _SUFF, _ELEM,
        )
        assert ret == p

    def test_read_restart(self, npt_rst_script):
        assert 'read_restart   /restarts/npt_300K_after_heat.restart' in npt_rst_script

    def test_no_read_data(self, npt_rst_script):
        assert 'read_data' not in npt_rst_script

    def test_no_velocity_command(self, npt_rst_script):
        assert 'velocity       ' not in npt_rst_script

    def test_no_heating_stage(self, npt_rst_script):
        assert 'fix            heat' not in npt_rst_script

    def test_npt_run_fix_present(self, npt_rst_script):
        assert 'fix            npt_run all npt temp 300.0 300.0' in npt_rst_script

    def test_boxdump_append_yes(self, npt_rst_script):
        assert 'append yes' in npt_rst_script

    def test_npt_result_print(self, npt_rst_script):
        assert 'NPT_RESULT: a0_at_300K' in npt_rst_script

    def test_restart_label_includes_rst_suffix(self, tmp_path):
        p = str(tmp_path / 'npt_rst2.lammps')
        write_npt_restart_script(
            '/r/f', '/d/n', p, _PS, _MOD, _SUFF, _ELEM,
            restart_dir='/r', target_t=500,
        )
        assert 'npt_500K_rst' in _read(p)


# ═══════════════════════════════════════════════════════════════════════════
# 5. write_surface_relaxation_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteSurfaceRelaxationScript:

    @pytest.fixture()
    def surf_script(self, tmp_path):
        p = str(tmp_path / 'surf_relax.lammps')
        write_surface_relaxation_script(
            slab_input='/data/slab.lammps',
            slab_relaxed='/data/slab_relaxed.lammps',
            relax_thermo='/data/relax_thermo.dat',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, target_t=300, z_freeze_cutoff=5.0,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'surf_relax.lammps')
        ret = write_surface_relaxation_script(
            '/d/s', '/d/r', '/d/t', p, _PS, _MOD, _SUFF, _ELEM,
        )
        assert ret == p

    def test_boundary_p_p_f(self, surf_script):
        assert 'boundary       p p f' in surf_script

    def test_read_data(self, surf_script):
        assert 'read_data      /data/slab.lammps' in surf_script

    def test_freeze_fix_present(self, surf_script):
        assert 'fix            freeze  frozen_atoms  setforce  0.0  0.0  0.0' in surf_script

    def test_phase1_minimization(self, surf_script):
        assert 'Phase 1: Surface minimization' in surf_script
        assert 'minimize' in surf_script

    def test_phase2_heating(self, surf_script):
        assert 'Phase 2: Heating to 300 K' in surf_script
        assert 'fix            nve_heat' in surf_script
        assert 'fix            rescale' in surf_script

    def test_phase3_nvt(self, surf_script):
        assert 'Phase 3: NVT equilibration at 300 K' in surf_script
        assert 'fix            nvt_equil' in surf_script

    def test_phase4_quench(self, surf_script):
        assert 'Phase 4: Quench minimization' in surf_script

    def test_z_cutoff_value_in_region(self, surf_script):
        assert 'EDGE  5.0' in surf_script

    def test_z_cutoff_placeholder_when_none(self, tmp_path):
        p = str(tmp_path / 'surf_ph.lammps')
        write_surface_relaxation_script(
            '/d/s', '/d/r', '/d/t', p, _PS, _MOD, _SUFF, _ELEM,
            z_freeze_cutoff=None,
        )
        assert 'FILL_IN_Z_CUTOFF' in _read(p)

    def test_phase_output_filenames_derived_from_stem(self, surf_script):
        assert '_phase1_min.lammps' in surf_script
        assert '_phase2_heat.lammps' in surf_script
        assert '_phase3_nvt.lammps' in surf_script

    def test_final_write_data(self, surf_script):
        assert 'write_data     /data/slab_relaxed.lammps' in surf_script

    def test_relax_thermo_ave_time_file(self, surf_script):
        assert 'file  /data/relax_thermo.dat' in surf_script

    def test_restart_label_contains_temperature(self, tmp_path):
        p = str(tmp_path / 'surf_r.lammps')
        write_surface_relaxation_script(
            '/d/s', '/d/r', '/d/t', p, _PS, _MOD, _SUFF, _ELEM,
            target_t=300, restart_dir='/restarts',
        )
        assert 'surf_300K' in _read(p)

    def test_relaxation_results_block(self, surf_script):
        assert 'RELAXATION_RESULTS_START' in surf_script
        assert 'RELAXATION_RESULTS_END' in surf_script


# ═══════════════════════════════════════════════════════════════════════════
# 6. write_surface_relaxation_restart_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteSurfaceRelaxationRestartScript:

    @pytest.fixture()
    def surf_rst_script(self, tmp_path):
        p = str(tmp_path / 'surf_rst.lammps')
        write_surface_relaxation_restart_script(
            restart_file='/restarts/surf_300K.*.restart',
            slab_relaxed='/data/slab_relaxed.lammps',
            relax_thermo='/data/relax_thermo.dat',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, target_t=300, z_freeze_cutoff=5.0,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'surf_rst.lammps')
        ret = write_surface_relaxation_restart_script(
            '/r/f', '/d/r', '/d/t', p, _PS, _MOD, _SUFF, _ELEM,
        )
        assert ret == p

    def test_read_restart(self, surf_rst_script):
        assert 'read_restart   /restarts/surf_300K.*.restart' in surf_rst_script

    def test_no_read_data(self, surf_rst_script):
        assert 'read_data' not in surf_rst_script

    def test_no_velocity_command(self, surf_rst_script):
        assert 'velocity       ' not in surf_rst_script

    def test_no_phase1_minimization(self, surf_rst_script):
        assert 'Phase 1' not in surf_rst_script

    def test_no_phase2_heating(self, surf_rst_script):
        assert 'Phase 2' not in surf_rst_script

    def test_phase3_nvt_present(self, surf_rst_script):
        assert 'Phase 3 NVT restart' in surf_rst_script
        assert 'fix            nvt_equil' in surf_rst_script

    def test_nvt_traj_dump_append_yes(self, surf_rst_script):
        assert 'dump_modify    surf_traj  sort id  append yes' in surf_rst_script

    def test_thermo_out_append_yes(self, surf_rst_script):
        assert 'mode scalar  append yes' in surf_rst_script

    def test_phase4_quench_present(self, surf_rst_script):
        assert 'Phase 4: Quench minimization' in surf_rst_script

    def test_final_write_data(self, surf_rst_script):
        assert 'write_data     /data/slab_relaxed.lammps' in surf_rst_script


# ═══════════════════════════════════════════════════════════════════════════
# 7. write_nvt_bulk_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNvtBulkScript:

    @pytest.fixture()
    def nvt_script(self, tmp_path):
        p = str(tmp_path / 'nvt.lammps')
        write_nvt_bulk_script(
            bulk_h_file='/data/bulk_h.lammps',
            traj_file='/data/traj.lammpstrj',
            out_file='/data/nvt_out.lammps',
            msd_equil_file='/data/msd_equil.dat',
            msd_prod_file='/data/msd_prod.dat',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, temperature=500,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'nvt.lammps')
        ret = write_nvt_bulk_script(
            '/d/h', '/d/t', '/d/o', '/d/me', '/d/mp', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=300,
        )
        assert ret == p

    def test_read_data(self, nvt_script):
        assert 'read_data      /data/bulk_h.lammps' in nvt_script

    def test_h_atom_group_default_type(self, nvt_script):
        assert 'group          H_atom  type 9' in nvt_script

    def test_velocity_command_present(self, nvt_script):
        assert 'velocity       all create 500.0' in nvt_script

    def test_phase1_equil_fix(self, nvt_script):
        assert 'fix            nvt_equil  all  nvt  temp  500.0  500.0' in nvt_script

    def test_msd_compute_present(self, nvt_script):
        assert 'compute        msd_H      H_atom  msd' in nvt_script

    def test_msd_equil_file_referenced(self, nvt_script):
        assert '/data/msd_equil.dat' in nvt_script

    def test_phase2_prod_fix(self, nvt_script):
        assert 'fix            nvt_prod  all  nvt  temp  500.0  500.0' in nvt_script

    def test_msd_prod_file_referenced(self, nvt_script):
        assert '/data/msd_prod.dat' in nvt_script

    def test_prod_traj_dump(self, nvt_script):
        assert 'dump           prod_dump  all  custom' in nvt_script
        assert '/data/traj.lammpstrj' in nvt_script

    def test_equil_results_block(self, nvt_script):
        assert 'EQUIL_RESULTS_START' in nvt_script
        assert 'EQUIL_RESULTS_END' in nvt_script

    def test_custom_h_type(self, tmp_path):
        p = str(tmp_path / 'nvt_ht.lammps')
        write_nvt_bulk_script(
            '/d/h', '/d/t', '/d/o', '/d/me', '/d/mp', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=300, h_type=3,
        )
        assert 'group          H_atom  type 3' in _read(p)

    def test_phase_filenames_derived_from_out_file(self, nvt_script):
        assert '/data/nvt_out_phase1_equil.lammps' in nvt_script

    def test_restart_label_contains_temperature(self, tmp_path):
        p = str(tmp_path / 'nvt_r.lammps')
        write_nvt_bulk_script(
            '/d/h', '/d/t', '/d/o', '/d/me', '/d/mp', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=400,
            restart_dir='/restarts',
        )
        assert 'nvt_400K' in _read(p)


# ═══════════════════════════════════════════════════════════════════════════
# 8. write_nvt_bulk_restart_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNvtBulkRestartScript:

    @pytest.fixture()
    def nvt_rst_script(self, tmp_path):
        p = str(tmp_path / 'nvt_rst.lammps')
        write_nvt_bulk_restart_script(
            restart_file='/restarts/nvt_500K.*.restart',
            traj_file='/data/traj.lammpstrj',
            out_file='/data/nvt_out.lammps',
            msd_file='/data/msd_prod.dat',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, temperature=500,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'nvt_rst.lammps')
        ret = write_nvt_bulk_restart_script(
            '/r/f', '/d/t', '/d/o', '/d/m', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=300,
        )
        assert ret == p

    def test_read_restart(self, nvt_rst_script):
        assert 'read_restart   /restarts/nvt_500K.*.restart' in nvt_rst_script

    def test_no_read_data(self, nvt_rst_script):
        assert 'read_data' not in nvt_rst_script

    def test_no_velocity_command(self, nvt_rst_script):
        assert 'velocity       ' not in nvt_rst_script

    def test_equil_dump_append_yes(self, nvt_rst_script):
        assert 'equil_dump  sort id  append yes' in nvt_rst_script

    def test_prod_dump_append_yes(self, nvt_rst_script):
        assert 'prod_dump  sort id  append yes' in nvt_rst_script

    def test_msd_append_yes(self, nvt_rst_script):
        assert 'mode scalar  append yes' in nvt_rst_script

    def test_custom_equil_run_count(self, tmp_path):
        p = str(tmp_path / 'nvt_rst2.lammps')
        write_nvt_bulk_restart_script(
            '/r/f', '/d/t', '/d/o', '/d/m', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=300, n_equil=250000,
        )
        assert 'run            250000' in _read(p)

    def test_custom_prod_run_count(self, tmp_path):
        p = str(tmp_path / 'nvt_rst3.lammps')
        write_nvt_bulk_restart_script(
            '/r/f', '/d/t', '/d/o', '/d/m', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=300, n_prod=750000,
        )
        assert 'run            750000' in _read(p)


# ═══════════════════════════════════════════════════════════════════════════
# 9. write_nvt_equil_restart_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNvtEquilRestartScript:

    @pytest.fixture()
    def equil_rst_script(self, tmp_path):
        p = str(tmp_path / 'equil_rst.lammps')
        write_nvt_equil_restart_script(
            restart_file='/r/nvt_300K.*.restart',
            traj_file='/d/traj.lammpstrj',
            out_file='/d/nvt_out.lammps',
            msd_equil_file='/d/msd_equil.dat',
            msd_prod_file='/d/msd_prod.dat',
            log_file='/d/nvt.log',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, temperature=300, n_equil=500000, n_prod=1500000,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'equil_rst.lammps')
        ret = write_nvt_equil_restart_script(
            '/r/f', '/d/t', '/d/o', '/d/me', '/d/mp', '/d/l', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=300,
        )
        assert ret == p

    def test_log_append(self, equil_rst_script):
        assert 'log            /d/nvt.log  append' in equil_rst_script

    def test_read_restart(self, equil_rst_script):
        assert 'read_restart   /r/nvt_300K.*.restart' in equil_rst_script

    def test_equil_run_upto(self, equil_rst_script):
        # 'run N upto' runs to absolute step N (0 steps if already past it);
        # replaces the remaining_equil variable, which could go negative.
        assert 'run            500000  upto' in equil_rst_script
        assert 'remaining_equil' not in equil_rst_script

    def test_equil_no_ternary(self, equil_rst_script):
        # LAMMPS variable formulas have no C-style ternary — a '?' in a
        # formula crashes every restart leg (2026-07-03 smoke-test bug).
        assert '?' not in equil_rst_script

    def test_equil_msd_append_yes(self, equil_rst_script):
        # equil MSD appends to msd_equil_file
        assert 'mode scalar  append yes' in equil_rst_script

    def test_prod_dump_no_append(self, equil_rst_script):
        # production dump is fresh — no append yes
        assert 'dump_modify    prod_dump  sort id\n' in equil_rst_script

    def test_prod_run_full_count(self, equil_rst_script):
        assert 'run            1500000' in equil_rst_script

    def test_prod_msd_no_append(self, equil_rst_script):
        # production MSD is fresh — ends with mode scalar, not mode scalar append yes
        assert 'file  /d/msd_prod.dat  mode scalar\n' in equil_rst_script


# ═══════════════════════════════════════════════════════════════════════════
# 10. write_nvt_prod_restart_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteNvtProdRestartScript:

    @pytest.fixture()
    def prod_rst_script(self, tmp_path):
        p = str(tmp_path / 'prod_rst.lammps')
        write_nvt_prod_restart_script(
            restart_file='/r/nvt_300K.*.restart',
            traj_file='/d/traj.lammpstrj',
            out_file='/d/nvt_out.lammps',
            msd_prod_file='/d/msd_prod.dat',
            log_file='/d/nvt.log',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, temperature=300, n_equil=500000, n_prod=1500000,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'prod_rst.lammps')
        ret = write_nvt_prod_restart_script(
            '/r/f', '/d/t', '/d/o', '/d/m', '/d/l', p,
            _PS, _MOD, _SUFF, _ELEM, temperature=300,
        )
        assert ret == p

    def test_log_append(self, prod_rst_script):
        assert 'log            /d/nvt.log  append' in prod_rst_script

    def test_read_restart(self, prod_rst_script):
        assert 'read_restart   /r/nvt_300K.*.restart' in prod_rst_script

    def test_prod_run_upto_total(self, prod_rst_script):
        # total = n_equil + n_prod = 500000 + 1500000 = 2000000
        assert 'run            2000000  upto' in prod_rst_script
        assert 'remaining_prod' not in prod_rst_script

    def test_prod_no_ternary(self, prod_rst_script):
        # The old '(N - step) > 0 ? (N - step) : 0' ternary is invalid
        # LAMMPS syntax — it crashed every prod restart leg and truncated
        # the production MSD files (2026-07-03 smoke-test bug).
        assert '?' not in prod_rst_script

    def test_prod_dump_append_yes(self, prod_rst_script):
        assert 'dump_modify    prod_dump  sort id  append yes' in prod_rst_script

    def test_msd_prod_append_yes(self, prod_rst_script):
        assert 'mode scalar  append yes' in prod_rst_script

    def test_no_equil_phase(self, prod_rst_script):
        assert 'nvt_equil' not in prod_rst_script

    def test_equil_results_block(self, prod_rst_script):
        assert 'EQUIL_RESULTS_START' in prod_rst_script


# ═══════════════════════════════════════════════════════════════════════════
# 11. write_adsorbate_min_script
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteAdsorbateMinScript:

    @pytest.fixture()
    def ads_script(self, tmp_path):
        p = str(tmp_path / 'ads_min.lammps')
        write_adsorbate_min_script(
            slab_ads_input='/data/slab_h.lammps',
            ads_output='/data/slab_h_min.lammps',
            out_path=p,
            pair_style=_PS, mace_model=_MOD, pair_suffix=_SUFF,
            elem_str=_ELEM, z_freeze_cutoff=22.115,
        )
        return _read(p)

    def test_returns_out_path(self, tmp_path):
        p = str(tmp_path / 'ads_min.lammps')
        ret = write_adsorbate_min_script(
            '/d/s', '/d/o', p, _PS, _MOD, _SUFF, _ELEM, z_freeze_cutoff=5.0,
        )
        assert ret == p

    def test_boundary_p_p_f(self, ads_script):
        assert 'boundary       p p f' in ads_script

    def test_read_data(self, ads_script):
        assert 'read_data      /data/slab_h.lammps' in ads_script

    def test_z_freeze_in_region(self, ads_script):
        assert 'EDGE  22.115' in ads_script

    def test_freeze_fix(self, ads_script):
        assert 'fix            freeze  frozen_atoms  setforce  0.0  0.0  0.0' in ads_script

    def test_minimize_command_defaults(self, ads_script):
        assert 'minimize       0.0  1e-06  10000  100000' in ads_script

    def test_write_data(self, ads_script):
        assert 'write_data     /data/slab_h_min.lammps' in ads_script

    def test_no_periodic_restart_directive(self, ads_script):
        assert 'restart        ' not in ads_script

    def test_traj_dump_written_when_given(self, tmp_path):
        p = str(tmp_path / 'ads_t.lammps')
        write_adsorbate_min_script(
            '/d/s', '/d/o', p, _PS, _MOD, _SUFF, _ELEM,
            z_freeze_cutoff=5.0, traj_file='/traj/ads.lammpstrj',
        )
        content = _read(p)
        assert 'dump           ads_traj' in content
        assert '/traj/ads.lammpstrj' in content
        assert 'undump         ads_traj' in content

    def test_no_dump_by_default(self, ads_script):
        assert 'dump           ads_traj' not in ads_script

    def test_custom_ftol(self, tmp_path):
        p = str(tmp_path / 'ads_ft.lammps')
        write_adsorbate_min_script(
            '/d/s', '/d/o', p, _PS, _MOD, _SUFF, _ELEM,
            z_freeze_cutoff=5.0, ftol=1e-8,
        )
        assert '1e-08' in _read(p)

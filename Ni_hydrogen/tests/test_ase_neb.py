"""
tests/test_ase_neb.py
======================
Tests for models/ase_neb.py — offline, no LAMMPS/SLURM/cluster required.

Covers:
  write_ase_neb_script / run_neb_pipeline — embedded config, the
  per-image LAMMPS-data write regression (Bug 20), and the Phase 1
  restart-fmax-check regression (Bug 21).
"""

import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.ase_neb import run_neb_pipeline, build_neb_images
from models.config import MASSES_7, E2T_7


def _generate(tmp_path, **overrides):
    kwargs = dict(
        is_file='/x/is.lammps',
        fs_file='/x/fs.lammps',
        e_is=-100.0,
        mace_model_path='/x/model.model',
        barrier_file='/x/barrier.txt',
        path_file='/x/path.dat',
        outdir=str(tmp_path),
        fs_log_file='/x/fs_min.log',
        job_name='test',
        n_images=6,
        spring_const=1.0,
        neb_ftol=0.1,
        z_freeze_cutoff=10.0,
        device='cpu',
        label_is='IS',
        label_fs='FS',
    )
    kwargs.update(overrides)
    out_py = run_neb_pipeline(**kwargs)
    return out_py, pathlib.Path(out_py).read_text()


class TestRunNebPipeline:

    def test_file_created_and_compiles(self, tmp_path):
        out_py, content = _generate(tmp_path)
        assert pathlib.Path(out_py).exists()
        compile(content, out_py, 'exec')

    def test_per_image_write_uses_write_lammps_data(self, tmp_path):
        """Regression test for Bug 20: a bare ase.io.write of NEB image
        structures omits the Masses section, so a later read() (no
        explicit type-to-element override) falls back to treating the
        raw LAMMPS atom type id as the atomic number -- with this
        project's E2T_7 mapping (Ni=7, H=8) that silently collides with
        real elements (N=7, O=8) instead of erroring. vib_run.py found
        "0 H atoms" on a structure that actually has exactly one.
        write_lammps_data (this project's canonical writer) always
        writes a real Masses section, so the round trip is correct
        regardless of what later reads the file."""
        _, content = _generate(tmp_path, masses=MASSES_7, e2t=E2T_7)
        assert 'write_lammps_data(' in content
        assert 'from models.structure import write_lammps_data' in content
        assert '_ase_write' not in content

    def test_masses_and_e2t_embedded(self, tmp_path):
        _, content = _generate(tmp_path, masses=MASSES_7, e2t=E2T_7)
        assert 'MASSES          = ' in content
        assert 'E2T             = ' in content
        assert "'Ni': 7" in content
        assert "'H': 8" in content

    def test_masses_and_e2t_default_when_not_provided(self, tmp_path):
        # Callers that don't pass masses/e2t must still get a valid,
        # non-empty mapping (MASSES_7/E2T_7), not an empty/None embed.
        _, content = _generate(tmp_path)
        assert 'MASSES          = {' in content
        assert 'E2T             = {' in content

    def test_generated_script_can_import_models_package(self, tmp_path):
        """The generated script runs standalone on a cluster node --
        it must add the project root to sys.path before importing
        models.structure, or the import will fail there."""
        _, content = _generate(tmp_path)
        assert 'sys.path.insert(0, _parent)' in content
        import_idx = content.index('from models.structure import write_lammps_data')
        syspath_idx = content.index('sys.path.insert(0, _parent)')
        assert syspath_idx < import_idx

    def test_default_dtype_is_float32(self, tmp_path):
        _, content = _generate(tmp_path)
        assert 'DTYPE           = "float32"' in content
        assert 'default_dtype=DTYPE' in content
        assert 'default_dtype="float64"' not in content

    def test_custom_dtype_override(self, tmp_path):
        _, content = _generate(tmp_path, dtype='float64')
        assert 'DTYPE           = "float64"' in content

    def test_parallel_true_by_default(self, tmp_path):
        _, content = _generate(tmp_path)
        assert 'PARALLEL        = True' in content
        # 3 NEB(...) construction sites: fresh-start, Phase-2-checkpoint
        # restart, and Phase-1-checkpoint restart (a single NEB object is
        # now built once with climb=False and flipped to climb=True in
        # place if already converged, rather than being reconstructed --
        # see the _neb_fmax fix, which needs the same NEB object Phase 1
        # itself would optimize with to evaluate the correct convergence
        # criterion).
        assert content.count('parallel=PARALLEL') == 3

    def test_parallel_false_override(self, tmp_path):
        _, content = _generate(tmp_path, parallel=False)
        assert 'PARALLEL        = False' in content

    def test_torch_set_num_threads_present(self, tmp_path):
        """NEB(parallel=True) spawns one thread per image; without pinning
        PyTorch's own intra-op thread count to 1, each thread would also
        try to multithread its MACE forward pass, oversubscribing CPUs."""
        _, content = _generate(tmp_path)
        assert 'import torch' in content
        assert 'torch.set_num_threads(1)' in content
        assert content.index('import torch') < content.index('torch.set_num_threads(1)')

    def test_append_trajectory_true_for_both_phases(self, tmp_path):
        """Regression test for Bug 21: without append_trajectory=True,
        every chain-resubmitted leg's MDMin(trajectory=...) truncates the
        existing .traj file back to empty on construction."""
        _, content = _generate(tmp_path)
        assert content.count('"append_trajectory": True') == 2

    def test_phase1_barrier_fields_written(self, tmp_path):
        _, content = _generate(tmp_path)
        assert 'phase1_fmax_final' in content
        assert 'phase1_converged' in content

    def test_uses_fire_optimizer_not_mdmin(self, tmp_path):
        """FIRE typically needs fewer iterations than MDMin to reach the
        same fmax target, with no accuracy tradeoff (same convergence
        criterion) -- confirmed a mechanically simple drop-in via ASE's
        shared Optimizer/Dynamics interface."""
        _, content = _generate(tmp_path)
        assert 'from ase.optimize import FIRE' in content
        assert content.count('FIRE(neb,') == 2
        assert 'MDMin' not in content


# ═══════════════════════════════════════════════════════════════════════════
# IS/FS periodic alignment -- independent wrap() calls used to let an atom
# near a cell boundary in one endpoint but not the other get wrapped to
# opposite sides of the cell, so NEB.interpolate() (raw Cartesian, no
# periodic awareness) would treat that as a literal ~cell-length path to
# traverse. Confirmed on a real pair (Hastelloy_N_1234_supercell,
# s_145__s_73+s_144): 3 metal atoms showed ~12.6 Å naive displacement
# (essentially the full 12.619 Å cell x-length) vs ~0.01 Å true
# (minimum-image) displacement.

class TestFsAlignedViaMinimumImage:

    def test_find_mic_import_present(self, tmp_path):
        _, content = _generate(tmp_path)
        assert 'from ase.geometry import find_mic' in content

    def test_fs_no_longer_wrapped_independently(self, tmp_path):
        _, content = _generate(tmp_path)
        assert 'fs_raw.wrap()' not in content

    def test_is_still_wrapped_as_the_anchor_frame(self, tmp_path):
        _, content = _generate(tmp_path)
        assert 'is_raw.wrap()' in content

    def test_fs_positions_set_from_minimum_image_vector(self, tmp_path):
        _, content = _generate(tmp_path)
        assert 'find_mic(_diff, is_raw.cell, pbc=_slab_pbc)' in content
        assert 'fs_raw.set_positions(is_raw.get_positions() + _mic_vec)' in content
        # The alignment must happen after IS is loaded/wrapped (it aligns
        # FS *to* IS's frame) and before FS's positions are used elsewhere.
        wrap_idx  = content.index('is_raw.wrap()')
        align_idx = content.index('fs_raw.set_positions(')
        assert wrap_idx < align_idx

    def test_slab_pbc_is_xy_periodic_z_fixed(self, tmp_path):
        """find_mic must NOT treat z as periodic -- these are slabs with a
        vacuum gap, not bulk-periodic in z. A pbc=True blanket would let
        find_mic "shortcut" a real z-motion (e.g. toward desorption) through
        the vacuum as if it wrapped around the box, which is physically
        nonsensical for a non-periodic direction."""
        _, content = _generate(tmp_path)
        assert '_slab_pbc = (True, True, False)' in content

    def test_slab_pbc_passed_to_find_mic_not_atoms_pbc(self, tmp_path):
        """The fix must only affect the find_mic() call, not is_raw/fs_raw's
        own .pbc attribute -- changing that would alter calculator behaviour
        (periodic neighbour lists) for every force evaluation in the NEB
        run, not just this alignment step."""
        _, content = _generate(tmp_path)
        assert 'is_raw.pbc = ' not in content
        assert 'fs_raw.pbc = ' not in content

    def test_cell_mismatch_raises_before_find_mic(self, tmp_path):
        """find_mic() uses is_raw.cell only -- if FS's own minimisation ever
        produced a different cell, aligning FS's positions against the
        wrong periodic geometry would silently produce garbage."""
        _, content = _generate(tmp_path)
        assert 'np.allclose(is_raw.cell.array, fs_raw.cell.array)' in content
        assert 'IS and FS cells differ' in content
        cell_check_idx = content.index('np.allclose(is_raw.cell.array')
        find_mic_call_idx = content.index('_mic_vec, _ = find_mic(')
        assert cell_check_idx < find_mic_call_idx

    def test_half_cell_length_warning_present(self, tmp_path):
        """find_mic() structurally cannot distinguish a genuine displacement
        > L/2 from a shorter wrap-around candidate -- flag it rather than
        silently returning a possibly-ambiguous answer."""
        _, content = _generate(tmp_path)
        assert 'import warnings' in content
        assert '0.5 * is_raw.cell.lengths()' in content
        assert 'warnings.warn(' in content
        assert 'find_mic() cannot distinguish a genuine large hop' in content


# ═══════════════════════════════════════════════════════════════════════════
# Bug 21 — Phase 1 restart must check convergence, not just file existence
# ═══════════════════════════════════════════════════════════════════════════
#
# Before this fix, the elif TRAJ_PHASE1-exists branch unconditionally set
# _restart_phase = 2 -- so a job that timed out mid-Phase-1 (never reaching
# its own fmax target) would, on the very next resubmitted leg, load the
# under-converged band and jump straight into the tight Phase 2 CI-NEB
# optimization, while neb_phase1.traj/.log silently stopped being written
# to. These tests exec() just the restart-detection slice of the generated
# script (the same pattern tests/test_neb_workflow.py's
# TestNebRunH2CacheBehavior uses) against lightweight fakes, so the
# decision logic is verified without needing a real ASE/MACE calculation.

class _FakeImage:
    def __init__(self, force_val=0.0):
        self._f = force_val
        self.calc = None

    def get_forces(self):
        return np.array([[self._f, 0.0, 0.0]])


class _FakeBandTrajectory:
    def __init__(self, frames):
        self._frames = frames

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


class _FakeNEB:
    def __init__(self, images, climb, k, method, parallel):
        self.images = images
        self.climb = climb
        self.k = k
        self.method = method
        self.parallel = parallel

    def get_forces(self):
        # Mimics ASE's flattened (n_intermediate_images * natoms, 3) shape
        # returned by the real NEB.get_forces().
        return np.concatenate([img.get_forces() for img in self.images[1:-1]], axis=0)


class TestPhase1RestartFmaxCheck:

    def _restart_slice(self, content):
        start = content.index('# ── Restart detection / IDPP interpolation')
        end = content.index('# ── Phase 1: regular NEB')
        return content[start:end]

    def _run(self, tmp_path, *, n_images, n1_fmax, n1_steps, n_frames, force_val):
        _, content = _generate(tmp_path, n_images=n_images)
        traj1 = tmp_path / 'neb_phase1.traj'
        traj1.write_text('placeholder')  # only needs to exist on disk

        frames = [_FakeImage(force_val) for _ in range(n_frames)]
        ns = {
            'np': np,
            'sys': sys,
            '_Path': pathlib.Path,
            '_Trajectory': lambda path: _FakeBandTrajectory(frames),
            'NEB': _FakeNEB,
            'make_calc': lambda: object(),
            'N_IMAGES': n_images,
            'N1_FMAX': n1_fmax,
            'N1_STEPS': n1_steps,
            'SPRING_CONST': 1.0,
            'PARALLEL': True,
            'TRAJ_PHASE1': str(traj1),
            'TRAJ_PHASE2': None,
        }
        exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)
        return ns

    def test_not_converged_budget_remains_stays_in_phase1(self, tmp_path):
        # 3 completed steps recorded (18 frames / 6 per step - 1 initial
        # batch = 2 done), budget of 10 -> 8 remaining.
        ns = self._run(tmp_path, n_images=4, n1_fmax=0.05, n1_steps=10,
                        n_frames=18, force_val=0.5)
        assert ns['_restart_phase'] == 1
        assert ns['_p1_remaining'] == 8
        assert ns['neb'].climb is False

    def test_converged_band_advances_to_phase2(self, tmp_path):
        ns = self._run(tmp_path, n_images=4, n1_fmax=0.05, n1_steps=10,
                        n_frames=18, force_val=0.01)
        assert ns['_restart_phase'] == 2
        assert ns['_p1_converged'] is True
        assert ns['_p1_fmax_final'] == pytest.approx(0.01)
        assert ns['neb'].climb is True

    def test_budget_exhausted_advances_to_phase2_but_not_converged(self, tmp_path):
        # steps_done (2) >= n1_steps (2) -> remaining <= 0, even though
        # fmax is still far above the target.
        ns = self._run(tmp_path, n_images=4, n1_fmax=0.05, n1_steps=2,
                        n_frames=18, force_val=0.5)
        assert ns['_restart_phase'] == 2
        assert ns['_p1_converged'] is False
        assert ns['neb'].climb is True


# ═══════════════════════════════════════════════════════════════════════════
# N_IMAGES-mismatch guard -- regenerating a pair's script with a different
# n_images while an old .traj (written under a different N_IMAGES) still
# exists on disk would otherwise silently misinterpret the frame-slicing
# math in _load_last_band, producing a structurally-plausible but wrong
# restart band with no error. _load_last_band now checks that the frame
# count divides evenly by N_IMAGES+2 before trusting it.
# ═══════════════════════════════════════════════════════════════════════════

class TestNImagesMismatchGuard:

    def _restart_slice(self, content):
        start = content.index('# ── Restart detection / IDPP interpolation')
        end = content.index('# ── Phase 1: regular NEB')
        return content[start:end]

    def _ns(self, tmp_path, *, n_images, traj1_exists, traj2_exists, n_frames):
        _, content = _generate(tmp_path, n_images=n_images)
        frames = [_FakeImage(0.01) for _ in range(n_frames)]

        traj1 = tmp_path / 'neb_phase1.traj'
        traj2 = tmp_path / 'neb_phase2.traj'
        if traj1_exists:
            traj1.write_text('placeholder')
        if traj2_exists:
            traj2.write_text('placeholder')

        return content, {
            'np': np,
            'sys': sys,
            '_Path': pathlib.Path,
            '_Trajectory': lambda path: _FakeBandTrajectory(frames),
            'NEB': _FakeNEB,
            'make_calc': lambda: object(),
            'N_IMAGES': n_images,
            'N1_FMAX': 0.05,
            'N1_STEPS': 10,
            'SPRING_CONST': 1.0,
            'PARALLEL': True,
            'TRAJ_PHASE1': str(traj1) if traj1_exists else None,
            'TRAJ_PHASE2': str(traj2) if traj2_exists else None,
            '_test_frames': frames,   # exposed so tests can assert frame identity
        }

    def test_partial_trailing_batch_truncates_not_raises(self, tmp_path):
        """Regression test for the real production failure this guard
        introduced: a SLURM timeout landing mid-write of one step's frame
        batch leaves a partial trailing batch (31 frames for N_IMAGES=6,
        N_IMAGES+2=8, observed for real on the cluster) -- this is a normal
        consequence of the chain-resubmit mechanism, not an operator error,
        and must NOT hard-fail the job. It should truncate to the last
        complete batch (24 frames = 3 complete steps) and continue."""
        content, ns = self._ns(tmp_path, n_images=6, traj1_exists=True,
                                traj2_exists=False, n_frames=31)
        exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)
        assert ns['_restart_phase'] in (1, 2)
        assert len(ns['images']) == 8   # one full band (N_IMAGES+2), not 7

    def test_partial_trailing_batch_uses_last_complete_batch_frames(self, tmp_path):
        """Confirms truncation drops the incomplete tail specifically --
        not just any N_IMAGES+2-sized slice of the raw frame list."""
        content, ns = self._ns(tmp_path, n_images=4, traj1_exists=True,
                                traj2_exists=False, n_frames=17)
        exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)
        # 17 frames, N_IMAGES+2=6 -> 2 complete batches (12 frames); the
        # last complete batch is frames[6:12], NOT a naive frames[11:17]
        # (which would wrongly include the 5 incomplete trailing frames).
        raw_frames = ns['_test_frames']
        assert ns['images'] == raw_frames[6:12]
        assert ns['_p1_steps_done'] == 1   # 12 // 6 - 1 == 1 completed step

    def test_zero_complete_batches_still_raises(self, tmp_path):
        """A genuine N_IMAGES mismatch (or real corruption) leaving fewer
        frames than a single complete band has nothing usable to resume
        from -- this must still fail loudly, not silently proceed with a
        garbage band."""
        content, ns = self._ns(tmp_path, n_images=4, traj1_exists=True,
                                traj2_exists=False, n_frames=3)
        with pytest.raises(RuntimeError, match='Nothing usable to resume from'):
            exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)

    def test_phase2_partial_trailing_batch_truncates_not_raises(self, tmp_path):
        # Covers the other _load_last_band call site (TRAJ_PHASE2 branch).
        content, ns = self._ns(tmp_path, n_images=6, traj1_exists=False,
                                traj2_exists=True, n_frames=31)
        exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)
        assert ns['_restart_phase'] == 2
        assert len(ns['images']) == 8

    def test_divisible_frame_count_does_not_raise(self, tmp_path):
        # 18 frames / 6 per step == 3 -- valid, must not false-positive.
        content, ns = self._ns(tmp_path, n_images=4, traj1_exists=True,
                                traj2_exists=False, n_frames=18)
        exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)
        assert ns['_restart_phase'] in (1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# fmax formula correctness -- neb.get_forces() vs raw img.get_forces()
# ═══════════════════════════════════════════════════════════════════════════
#
# fmax_final / phase1_fmax_final used to be computed from each intermediate
# image's raw calculator forces (img.get_forces()). NEB's optimizer does not
# converge on that quantity -- it converges on the perpendicular/spring-
# projected force returned by neb.get_forces(). The raw force also contains
# the along-path tangential component NEB deliberately never drives to
# zero, and since the two components are orthogonal, raw fmax is always >=
# the true NEB fmax. This is untestable with simple fakes (a fake NEB can
# only mimic whichever formula it's told to), so this test uses ASE's real,
# always-available EMT calculator on a genuine NEB problem (classic Al(100)
# adatom hop) to prove neb.get_forces()-based fmax actually matches what
# the optimizer converges on, while the old raw-force formula would not.

class TestNebFmaxMatchesOptimizerConvergence:

    def test_neb_projected_fmax_matches_convergence_raw_does_not(self):
        from ase.build import fcc100, add_adsorbate
        from ase.constraints import FixAtoms
        from ase.calculators.emt import EMT
        from ase.mep import NEB
        from ase.optimize import FIRE, QuasiNewton

        def make_slab():
            slab = fcc100('Al', size=(2, 2, 3), vacuum=10.0)
            slab.calc = EMT()
            slab.set_constraint(FixAtoms(mask=[a.tag > 1 for a in slab]))
            return slab

        slab1 = make_slab()
        add_adsorbate(slab1, 'Al', 1.7, 'hollow')
        slab1.calc = EMT()
        QuasiNewton(slab1, logfile=None).run(fmax=0.01)
        initial = slab1.copy()

        slab2 = make_slab()
        add_adsorbate(slab2, 'Al', 1.7, 'hollow', offset=(1, 0))
        slab2.calc = EMT()
        QuasiNewton(slab2, logfile=None).run(fmax=0.01)
        final = slab2.copy()

        n_images = 6
        neb_fmax_target = 0.05
        images = [initial.copy()] + [initial.copy() for _ in range(n_images)] + [final.copy()]
        for img in images:
            img.calc = EMT()
        neb = NEB(images, climb=False, k=1.0, method='aseneb', parallel=True)
        neb.interpolate(method='idpp')

        FIRE(neb, logfile=None, dt=0.05).run(fmax=0.15, steps=300)
        neb.climb = True
        converged = FIRE(neb, logfile=None, dt=0.02).run(fmax=neb_fmax_target, steps=300)
        assert converged, "sanity check: this real NEB problem must converge"

        # The fixed formula: matches what the optimizer actually used.
        proj_forces = neb.get_forces()
        neb_fmax = float(np.sqrt((proj_forces ** 2).sum(axis=1).max()))

        # The old (Bug: fmax formula) approach: raw per-image forces.
        raw_fmax_vals = [
            float(np.sqrt((img.get_forces() ** 2).sum(axis=1).max()))
            for img in images[1:-1]
        ]
        raw_fmax = max(raw_fmax_vals)

        assert neb_fmax <= neb_fmax_target, (
            f"neb.get_forces()-based fmax ({neb_fmax:.4f}) should match the "
            f"optimizer's own converged criterion (<= {neb_fmax_target})"
        )
        assert raw_fmax > neb_fmax_target, (
            "this real NEB problem should demonstrate the raw-force formula "
            "overstating the residual force past the target -- if this "
            "assertion fails, the test problem no longer exercises the "
            "regression it's meant to guard against"
        )
        assert raw_fmax > neb_fmax, (
            "raw per-image force must be >= the true NEB-projected force "
            "(the two components are orthogonal) -- confirms the old "
            "formula could never be an UNDER-estimate, i.e. Bug 21's phase1 "
            "convergence gate was always safe-direction, just imprecise"
        )


# ═══════════════════════════════════════════════════════════════════════════
# build_neb_images -- numeric confirmation of the periodic-alignment fix
# ═══════════════════════════════════════════════════════════════════════════
# Synthetic reproduction of the real-pair bug: a metal atom sitting near a
# cell face in IS, whose true physical relaxation is tiny (< 1 Å) but crosses
# the boundary in FS. Independently wrapping each endpoint used to map that
# atom to opposite sides of the cell -- a ~cell-length apparent jump that
# NEB.interpolate() would take as a literal path. A control atom (safely
# mid-cell) and the 2 H atoms (which genuinely move several Å -- the real
# reaction coordinate) must be unaffected by the fix.

class TestBuildNebImagesPeriodicAlignment:

    _CELL = [10.0, 10.0, 10.0]
    _SYMBOLS = ['Ni', 'Ni', 'H', 'H']

    def _write_structure(self, path, positions):
        from ase import Atoms
        from ase.io import write as ase_write
        atoms = Atoms(self._SYMBOLS, positions=positions, cell=self._CELL, pbc=True)
        ase_write(path, atoms, format='lammps-data', masses=True,
                  specorder=['Ni', 'H'])

    @pytest.fixture()
    def images(self, tmp_path):
        # IS: "problem" Ni atom near the x=0 face; control Ni safely mid-cell;
        # 2 H atoms mid-cell (intact H2-like separation).
        is_positions = [
            [0.2, 5.0, 5.0],    # problem atom
            [5.0, 5.0, 5.0],    # control atom
            [3.0, 3.0, 7.0],    # H
            [3.0, 3.0, 7.5],    # H
        ]
        # FS: problem atom's TRUE displacement is 0.3 Å (0.2 -> -0.1) --
        # independent wrapping would map -0.1 to 9.9, a ~9.7 Å apparent jump.
        # Control atom moves a trivial 0.05 Å. H atoms move several Å (the
        # real reaction coordinate) and must be left alone by the fix.
        fs_positions = [
            [-0.1, 5.0, 5.0],   # problem atom -- true displacement 0.3 Å
            [5.05, 5.0, 5.0],   # control atom -- true displacement 0.05 Å
            [4.0, 4.0, 7.0],    # H -- moved several Å (real reaction coordinate)
            [4.0, 4.0, 7.5],    # H -- moved several Å
        ]

        is_file = str(tmp_path / 'is.lammps')
        fs_file = str(tmp_path / 'fs.lammps')
        self._write_structure(is_file, is_positions)
        self._write_structure(fs_file, fs_positions)

        return build_neb_images(is_file, fs_file, n_images=3)

    def test_boundary_atom_does_not_jump_full_cell_length(self, images):
        is_img, fs_img = images[0], images[-1]
        problem_disp = np.linalg.norm(
            fs_img.get_positions()[0] - is_img.get_positions()[0])
        assert problem_disp < 1.0, (
            f'boundary atom displaced {problem_disp:.2f} Å end-to-end -- '
            f'looks like a wrap-boundary artifact, not the true ~0.3 Å '
            f'relaxation'
        )

    def test_intermediate_images_never_show_a_large_jump_for_that_atom(self, images):
        """Even if the endpoints happened to line up, IDPP could still route
        an intermediate image far away -- check every image, not just the
        two ends."""
        disps = [
            np.linalg.norm(img.get_positions()[0] - images[0].get_positions()[0])
            for img in images
        ]
        assert max(disps) < 1.0, (
            f'largest intermediate-image displacement for the boundary atom '
            f'was {max(disps):.2f} Å across the path'
        )

    def test_control_atom_displacement_unaffected(self, images):
        is_img, fs_img = images[0], images[-1]
        control_disp = np.linalg.norm(
            fs_img.get_positions()[1] - is_img.get_positions()[1])
        assert control_disp == pytest.approx(0.05, abs=1e-6)

    def test_h_atoms_still_move_the_real_reaction_distance(self, images):
        """The fix must not clip or distort the genuine, large H
        displacement -- only correct the spurious metal-atom wrap jump."""
        is_img, fs_img = images[0], images[-1]
        is_h = is_img.get_positions()[2:4]
        fs_h = fs_img.get_positions()[2:4]
        h_disps = np.linalg.norm(fs_h - is_h, axis=1)
        assert all(d > 1.0 for d in h_disps), (
            f'H displacements {h_disps} were clipped -- the fix must only '
            f'correct metal-atom wrap artifacts, not the real reaction '
            f'coordinate'
        )


# ═══════════════════════════════════════════════════════════════════════════
# build_neb_images -- hardening: pbc=(True,True,False), half-cell warning,
# cell-mismatch guard
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildNebImagesHardening:

    _CELL = [10.0, 10.0, 10.0]
    _SYMBOLS = ['Ni', 'Ni', 'H', 'H']

    def _write_structure(self, path, positions, cell=None):
        from ase import Atoms
        from ase.io import write as ase_write
        atoms = Atoms(self._SYMBOLS, positions=positions,
                       cell=cell or self._CELL, pbc=True)
        ase_write(path, atoms, format='lammps-data', masses=True,
                  specorder=['Ni', 'H'])

    def test_large_z_motion_not_shortcut_through_vacuum(self, tmp_path):
        """z must NOT be treated as periodic -- these are slabs with a
        vacuum gap. A real, large z-displacement (e.g. toward desorption)
        must be preserved as-is, not "wrapped" through the non-existent
        periodic boundary in z the way it legitimately can in x/y."""
        from models.ase_neb import build_neb_images

        # z-displacement of 7 Å is > half of Lz=10 -- if z were (wrongly)
        # treated as periodic, find_mic would shortcut this to ~-3 Å
        # (7 - 10) instead of preserving the real 7 Å motion.
        is_positions = [
            [5.0, 5.0, 1.0],   # atom moving a lot in z (e.g. toward vacuum)
            [5.0, 5.0, 5.0],   # control atom
            [3.0, 3.0, 3.0],   # H
            [3.0, 3.0, 3.5],   # H
        ]
        fs_positions = [
            [5.0, 5.0, 8.0],   # true z-displacement: 7 Å
            [5.05, 5.0, 5.0],
            [4.0, 4.0, 3.0],
            [4.0, 4.0, 3.5],
        ]
        is_file = str(tmp_path / 'is.lammps')
        fs_file = str(tmp_path / 'fs.lammps')
        self._write_structure(is_file, is_positions)
        self._write_structure(fs_file, fs_positions)

        images = build_neb_images(is_file, fs_file, n_images=3)
        is_img, fs_img = images[0], images[-1]
        z_disp = fs_img.get_positions()[0, 2] - is_img.get_positions()[0, 2]
        assert z_disp == pytest.approx(7.0, abs=1e-6), (
            f'z-displacement came back {z_disp:.3f} Å instead of the true '
            f'7.0 Å -- z is being treated as periodic when it should not be'
        )

    def test_no_warning_for_large_z_only_displacement(self, tmp_path):
        """The half-cell warning must be gated by the same pbc mask as
        find_mic itself -- z is non-periodic, so a large z-motion (even one
        that exceeds half the z cell length) is never ambiguous and must
        not trigger the "find_mic() cannot distinguish..." warning. Same
        scenario as test_large_z_motion_not_shortcut_through_vacuum, this
        time asserting on the warning behaviour rather than the position."""
        import warnings
        from models.ase_neb import build_neb_images

        is_positions = [
            [5.0, 5.0, 1.0],
            [5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.5],
        ]
        fs_positions = [
            [5.0, 5.0, 8.0],   # z-displacement 7 Å, > half of Lz=10 -- but
                               # z isn't periodic, so this is unambiguous.
            [5.05, 5.0, 5.0],
            [4.0, 4.0, 3.0],
            [4.0, 4.0, 3.5],
        ]
        is_file = str(tmp_path / 'is.lammps')
        fs_file = str(tmp_path / 'fs.lammps')
        self._write_structure(is_file, is_positions)
        self._write_structure(fs_file, fs_positions)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            build_neb_images(is_file, fs_file, n_images=3)
            assert len(w) == 0, (
                f'unexpected warning(s) for an unambiguous, non-periodic '
                f'z-only displacement: {[str(x.message) for x in w]}'
            )

    def test_warns_when_displacement_exceeds_half_cell_length(self, tmp_path):
        from models.ase_neb import build_neb_images

        # x-displacement of 6 Å is > half of Lx=10 -- find_mic cannot tell
        # this apart from a genuine -4 Å (6 - 10) motion; must warn.
        is_positions = [
            [1.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [3.0, 3.0, 7.0],
            [3.0, 3.0, 7.5],
        ]
        fs_positions = [
            [7.0, 5.0, 5.0],   # true x-displacement: 6 Å (> half of Lx=10)
            [5.05, 5.0, 5.0],
            [4.0, 4.0, 7.0],
            [4.0, 4.0, 7.5],
        ]
        is_file = str(tmp_path / 'is.lammps')
        fs_file = str(tmp_path / 'fs.lammps')
        self._write_structure(is_file, is_positions)
        self._write_structure(fs_file, fs_positions)

        with pytest.warns(UserWarning, match='exceeds half the cell length'):
            build_neb_images(is_file, fs_file, n_images=3)

    def test_no_warning_when_all_displacements_are_small(self, tmp_path):
        from models.ase_neb import build_neb_images
        import warnings

        is_positions = [
            [5.0, 5.0, 5.0], [4.0, 4.0, 5.0], [3.0, 3.0, 7.0], [3.0, 3.0, 7.5],
        ]
        fs_positions = [
            [5.05, 5.0, 5.0], [4.05, 4.0, 5.0], [3.5, 3.5, 7.0], [3.5, 3.5, 7.5],
        ]
        is_file = str(tmp_path / 'is.lammps')
        fs_file = str(tmp_path / 'fs.lammps')
        self._write_structure(is_file, is_positions)
        self._write_structure(fs_file, fs_positions)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            build_neb_images(is_file, fs_file, n_images=3)
            assert len(w) == 0, f'unexpected warning(s): {[str(x.message) for x in w]}'

    def test_mismatched_cells_raise_before_find_mic(self, tmp_path):
        from models.ase_neb import build_neb_images

        positions = [
            [5.0, 5.0, 5.0], [4.0, 4.0, 5.0], [3.0, 3.0, 7.0], [3.0, 3.0, 7.5],
        ]
        is_file = str(tmp_path / 'is.lammps')
        fs_file = str(tmp_path / 'fs.lammps')
        self._write_structure(is_file, positions, cell=[10.0, 10.0, 10.0])
        self._write_structure(fs_file, positions, cell=[10.5, 10.0, 10.0])

        with pytest.raises(RuntimeError, match='IS and FS cells differ'):
            build_neb_images(is_file, fs_file, n_images=3)

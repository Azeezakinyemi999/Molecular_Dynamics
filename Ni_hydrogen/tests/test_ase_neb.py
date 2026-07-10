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

from models.ase_neb import run_neb_pipeline
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
        }

    def test_phase1_mismatched_frame_count_raises(self, tmp_path):
        # n_images=4 -> N_IMAGES+2=6; 17 frames is not a multiple of 6.
        content, ns = self._ns(tmp_path, n_images=4, traj1_exists=True,
                                traj2_exists=False, n_frames=17)
        with pytest.raises(RuntimeError, match='not a multiple'):
            exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)

    def test_phase2_mismatched_frame_count_raises(self, tmp_path):
        # Covers the other _load_last_band call site (TRAJ_PHASE2 branch).
        content, ns = self._ns(tmp_path, n_images=4, traj1_exists=False,
                                traj2_exists=True, n_frames=17)
        with pytest.raises(RuntimeError, match='not a multiple'):
            exec(compile(self._restart_slice(content), 'restart_slice', 'exec'), ns)

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

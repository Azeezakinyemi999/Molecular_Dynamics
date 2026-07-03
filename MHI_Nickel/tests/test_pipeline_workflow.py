"""
tests/test_pipeline_workflow.py
================================
Tests for models/pipeline_workflow.py — offline, no LAMMPS or SLURM.

Covers:
  generate_pipeline_scripts — config embedding, parallel launch, sequential
                              permeation, failure handling
  generate_pipeline_sh      — SBATCH directives, optional time, module loads,
                              conda activate, python run line
"""

import os
import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.pipeline_workflow import (
    generate_pipeline_scripts,
    generate_pipeline_sh,
)


# ── shared config ────────────────────────────────────────────────────────────

_METALS = [
    {
        'stem': 'Ni3Mo',
        'neb_run_py': '/work/neb_run_Ni3Mo.py',
        'permeation_run_py': '/work/permeation_run_Ni3Mo.py',
    },
    {
        'stem': 'Ni',
        'neb_run_py': '/work/neb_run_Ni.py',
        'permeation_run_py': '/work/permeation_run_Ni.py',
    },
]

_DIFF_RUN = '/work/diffusivity_run.py'
_WORK_DIR = '/work/calculation'

_SH_CFG = dict(
    orch_job_name='pipeline_orch',
    orch_partition='cpu',
    orch_cpus_per_task=4,
    orch_mem='16G',
    orch_time='24:00:00',
    orch_openmpi_ver='4.1.4',
    orch_cuda_version='12.0',
    orch_conda_env='ase-env',
    orch_ld_paths=['/usr/local/lib'],
    out_py='/work/pipeline_run.py',
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. generate_pipeline_scripts
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratePipelineScripts:

    @pytest.fixture()
    def gen_result(self, tmp_path):
        out_py = str(tmp_path / 'scripts' / 'pipeline_run.py')
        ret = generate_pipeline_scripts(
            metals=_METALS,
            diffusivity_run_py=_DIFF_RUN,
            work_dir=_WORK_DIR,
            out_py=out_py,
        )
        content = pathlib.Path(out_py).read_text()
        return ret, out_py, content

    def test_file_created(self, gen_result):
        _, out_py, _ = gen_result
        assert pathlib.Path(out_py).exists()

    def test_returns_out_py_path(self, gen_result):
        ret, out_py, _ = gen_result
        assert ret == out_py

    def test_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / 'a' / 'b' / 'c' / 'pipeline_run.py')
        generate_pipeline_scripts(
            metals=_METALS,
            diffusivity_run_py=_DIFF_RUN,
            work_dir=_WORK_DIR,
            out_py=nested,
        )
        assert pathlib.Path(nested).exists()

    def test_metals_config_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'METALS' in content
        assert 'Ni3Mo' in content
        assert 'Ni' in content

    def test_diffusivity_run_py_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'DIFFUSIVITY_RUN_PY' in content
        assert _DIFF_RUN in content

    def test_work_dir_embedded(self, gen_result):
        _, _, content = gen_result
        assert 'WORK_DIR' in content
        assert _WORK_DIR in content

    def test_parallel_subprocess_popen_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'subprocess.Popen' in content

    def test_diffusivity_launched_in_parallel_block(self, gen_result):
        _, _, content = gen_result
        assert 'DIFFUSIVITY_RUN_PY' in content
        assert 'diffusivity' in content

    def test_neb_run_launched_per_metal(self, gen_result):
        _, _, content = gen_result
        assert "neb_run_py" in content

    def test_sequential_permeation_uses_subprocess_run(self, gen_result):
        _, _, content = gen_result
        assert 'subprocess.run' in content
        assert 'permeation_run_py' in content

    def test_failure_check_exits_with_1(self, gen_result):
        _, _, content = gen_result
        assert 'sys.exit(1)' in content

    def test_error_message_when_parallel_jobs_fail(self, gen_result):
        _, _, content = gen_result
        assert 'permeation will NOT run' in content

    def test_wait_for_parallel_jobs(self, gen_result):
        _, _, content = gen_result
        assert '_p.wait()' in content or '.wait()' in content

    def test_single_metal_list_works(self, tmp_path):
        out_py = str(tmp_path / 'single.py')
        single = [_METALS[0]]
        ret = generate_pipeline_scripts(
            metals=single,
            diffusivity_run_py=_DIFF_RUN,
            work_dir=_WORK_DIR,
            out_py=out_py,
        )
        content = pathlib.Path(out_py).read_text()
        assert 'Ni3Mo' in content
        assert ret == out_py

    def test_neb_log_path_uses_stem(self, gen_result):
        _, _, content = gen_result
        assert "neb_run_" in content

    def test_permeation_complete_message_in_body(self, gen_result):
        _, _, content = gen_result
        assert 'Pipeline complete' in content


# ═══════════════════════════════════════════════════════════════════════════
# 2. generate_pipeline_sh
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratePipelineSh:

    @pytest.fixture()
    def sh_result(self, tmp_path):
        out_sh = str(tmp_path / 'scripts' / 'pipeline_run.sh')
        ret = generate_pipeline_sh(**_SH_CFG, out_sh=out_sh)
        content = pathlib.Path(out_sh).read_text()
        return ret, out_sh, content

    def test_file_created(self, sh_result):
        _, out_sh, _ = sh_result
        assert pathlib.Path(out_sh).exists()

    def test_returns_out_sh_path(self, sh_result):
        ret, out_sh, _ = sh_result
        assert ret == out_sh

    def test_sbatch_job_name_uses_arg(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --job-name=pipeline_orch' in content

    def test_sbatch_partition_present(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --partition=cpu' in content

    def test_sbatch_mem_present(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --mem=16G' in content

    def test_sbatch_time_present_when_set(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --time=24:00:00' in content

    def test_sbatch_time_absent_when_none(self, tmp_path):
        out_sh = str(tmp_path / 'no_time.sh')
        generate_pipeline_sh(**{**_SH_CFG, 'orch_time': None}, out_sh=out_sh)
        assert '#SBATCH --time=' not in pathlib.Path(out_sh).read_text()

    def test_output_log_uses_job_name(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --output=pipeline_orch_%j.out' in content

    def test_error_log_uses_job_name(self, sh_result):
        _, _, content = sh_result
        assert '#SBATCH --error=pipeline_orch_%j.err' in content

    def test_module_load_openmpi(self, sh_result):
        _, _, content = sh_result
        assert 'module load OpenMPI/4.1.4' in content

    def test_module_load_cuda(self, sh_result):
        _, _, content = sh_result
        assert 'module load cuda/12.0' in content

    def test_conda_activate_present(self, sh_result):
        _, _, content = sh_result
        assert 'conda activate ase-env' in content

    def test_python_run_line_uses_out_py(self, sh_result):
        _, _, content = sh_result
        assert 'python /work/pipeline_run.py' in content

    def test_ld_library_path_present(self, sh_result):
        _, _, content = sh_result
        assert '/usr/local/lib' in content

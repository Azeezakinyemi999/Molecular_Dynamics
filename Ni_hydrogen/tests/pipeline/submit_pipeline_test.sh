#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --output=tests/pipeline/work/pipeline_test_%j.out

# Pipeline smoke test orchestrator.
# Runs on a CPU-only node (no GPU) and orchestrates GPU SLURM sub-jobs
# for MACE minimisations and the NEB array.
#
# Submit from the repo root:
#   sbatch tests/pipeline/submit_pipeline_test.sh
#
# Output:
#   tests/pipeline/work/pipeline_test_<jobid>.out  — full log
#   tests/pipeline/work/summary.txt                — pass/fail summary

module load OpenMPI/4.1.6
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps

# Navigate to repo root. $0 resolves to a SLURM temp file in /var/spool so
# realpath "$0" cannot be used. Walk up from $SLURM_SUBMIT_DIR instead.
repo_root="$SLURM_SUBMIT_DIR"
while [[ "$repo_root" != "/" && ! -d "$repo_root/models" ]]; do
    repo_root="$(dirname "$repo_root")"
done
if [[ ! -d "$repo_root/models" ]]; then
    echo "ERROR: could not locate repo root (models/ not found above $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
cd "$repo_root"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)  Start: $(date)"

python tests/pipeline/run_pipeline_test.py

EXIT_CODE=$?
echo "End: $(date)  Exit code: $EXIT_CODE"
exit $EXIT_CODE

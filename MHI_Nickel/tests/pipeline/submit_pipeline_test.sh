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

cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)  Start: $(date)"

python tests/pipeline/run_pipeline_test.py

EXIT_CODE=$?
echo "End: $(date)  Exit code: $EXIT_CODE"
exit $EXIT_CODE

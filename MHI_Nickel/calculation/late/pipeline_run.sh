#!/bin/bash
#SBATCH --job-name=pipeline_orch
#SBATCH --partition=west
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30-00:00:00
#SBATCH --output=pipeline_orch_%j.out
#SBATCH --error=pipeline_orch_%j.err

# ── environment ──────────────────────────────────────────────────────────
module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH
cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation

python /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/pipeline_run.py

#!/bin/bash
#SBATCH --job-name=bulk_min_pre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=multigpu
#SBATCH --time=01:00:00
#SBATCH --output=/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/slurm_scripts/bulk_min_pre_%j.out


module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib/libjemalloc.so

cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation

echo "Node: $(hostname)  Start: $(date)"
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/slurm_scripts/bulk_min_pre.lammps
echo "End: $(date)"

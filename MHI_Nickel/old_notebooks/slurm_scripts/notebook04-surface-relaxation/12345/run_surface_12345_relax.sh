#!/bin/bash
#SBATCH --job-name=hastelloyN_12345_surf_relax
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --output=results/notebook04-surface-relaxation/12345/surface_relax_12345_%j.log

module load OpenMPI/4.1.6
module load cuda/12.3.0

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps

export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH

cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in lammps_scripts/notebook04-surface-relaxation/12345/surface_relax.lammps \
    -log results/notebook04-surface-relaxation/12345/surface_relaxation.log

echo "Surface relaxation finished: $(date)"

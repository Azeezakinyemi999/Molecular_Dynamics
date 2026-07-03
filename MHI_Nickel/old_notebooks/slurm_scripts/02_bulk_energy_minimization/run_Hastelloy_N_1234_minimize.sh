#!/bin/bash
#SBATCH --job-name=Hastelloy_N_1234_min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=''gpu
#SBATCH --time=08:00:00
#SBATCH --output=results/02_bulk_energy_minimization/Hastelloy_N_1234_minimize_%j.log

module load OpenMPI/4.1.6
module load cuda/12.3.0

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps

export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH

cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

# Run LAMMPS minimization with Kokkos GPU backend
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in lammps_scripts/02_bulk_energy_minimization/bulk_N_1234_minimize.lammps \
    -log results/02_bulk_energy_minimization/Hastelloy_N_1234_minimization.log

echo "Minimization finished at: $(date)"

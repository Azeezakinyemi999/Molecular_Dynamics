#!/bin/bash
#SBATCH --job-name=nb06_h_top_fe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=multigpu
#SBATCH --time=01:00:00
#SBATCH --output=results/notebook06-Neb-dissociation/111/slurm_h_atom_top_fe_%j.out
module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH
cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in lammps_scripts/notebook06-Neb-dissociation/111/min_h_atom_top_fe.lammps -log results/notebook06-Neb-dissociation/111/h_atom_top_fe.log
echo "h_atom_top_fe done: $(date)"

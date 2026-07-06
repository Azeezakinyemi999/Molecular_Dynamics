#!/bin/bash
#SBATCH --job-name=Hads_s_113
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --output=/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/adsorption/phase2_h/slurm/h_slurm_s_113_%j.out


module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH


cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation

echo "Node: $(hostname)  Start: $(date)"
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/adsorption/phase2_h/scripts/h_min_s_113.in -log /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/adsorption/phase2_h/results/h_min_s_113.log
echo "End: $(date)"

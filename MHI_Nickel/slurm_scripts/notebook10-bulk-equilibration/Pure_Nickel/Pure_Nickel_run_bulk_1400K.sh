#!/bin/bash
#SBATCH --job-name=Pure_Nickelequilibration_1400K
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=multigpu
#SBATCH --time=24:00:00
#SBATCH --output=results/notebook10-bulk-equilibration/Pure_Nickel/Pure_Nickel_slurm_bulk_1400K_%j.out

module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH
cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

echo "T=1400K Start: $(date)"
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in lammps_scripts/notebook10-bulk-equilibration/Pure_Nickel/Pure_Nickel_nvt_1400K.lammps -log results/notebook10-bulk-equilibration/Pure_Nickel/Pure_Nickel_equil_1400K.log
echo "T=1400K End: $(date)"

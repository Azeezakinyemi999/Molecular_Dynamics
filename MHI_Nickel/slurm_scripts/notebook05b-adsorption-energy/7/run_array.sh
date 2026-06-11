#!/bin/bash
#SBATCH --job-name=nb05b_ads
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=multigpu
#SBATCH --time=02:00:00
#SBATCH --array=0-170%8
#SBATCH --output=results/notebook05b-adsorption-energy/7/slurm_%A_%a.out

module load OpenMPI/4.1.6
module load cuda/12.3.0

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps

export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH

cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

SITE_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm_scripts/notebook05b-adsorption-energy/7/site_index.txt)
if [ -z "$SITE_ID" ]; then
    echo "ERROR: empty SITE_ID for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID: $SITE_ID"

/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in lammps_scripts/notebook05b-adsorption-energy/7/min_${SITE_ID}.lammps \
    -log results/notebook05b-adsorption-energy/7/ads_${SITE_ID}.log

if [ $? -ne 0 ]; then
    echo "LAMMPS failed for $SITE_ID"
    exit 1
fi
echo "Done: $SITE_ID $(date)"

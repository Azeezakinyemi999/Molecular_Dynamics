#!/bin/bash
#SBATCH --job-name=nb06a_neb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=multigpu
#SBATCH --time=06:00:00
#SBATCH --array=0-123%8
#SBATCH --output=results/notebook06-Neb-dissociation/7/multi/slurm_%A_%a.out

module load OpenMPI/4.1.6
module load cuda/12.3.0

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps

export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH

cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

# Get label from index file using task ID
LABEL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm_scripts/notebook06-Neb-dissociation/7/multi/job_index.txt)
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL"

RESULT_DIR=results/notebook06-Neb-dissociation/7/multi/$LABEL
SCRIPT_DIR=lammps_scripts/notebook06-Neb-dissociation/7/multi/$LABEL

echo "Step 1: FS minimization"
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in $SCRIPT_DIR/min_fs.lammps \
    -log $RESULT_DIR/fs_min.log

echo "Step 2: NEB"
python3 $SCRIPT_DIR/run_neb.py

echo "Done: $LABEL $(date)"

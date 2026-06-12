#!/bin/bash
#SBATCH --job-name=nb06b2_neb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=multigpu
#SBATCH --time=06:00:00
#SBATCH --array=0-847%8
#SBATCH --output=logs/slurm_%A_%a.out

mkdir -p logs

module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mace-lammps

export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH

cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

TASKS_PER_JOB=400
N_TOTAL=338830
START_TASK=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
END_TASK=$(( START_TASK + TASKS_PER_JOB - 1 ))
if [ $END_TASK -ge $N_TOTAL ]; then END_TASK=$(( N_TOTAL - 1 )); fi

echo "Array element $SLURM_ARRAY_TASK_ID: tasks $START_TASK–$END_TASK"

for (( id=START_TASK; id<=END_TASK; id++ )); do
    python3 run_worker.py --task_id $id
done

echo "Done: element $SLURM_ARRAY_TASK_ID at $(date)"

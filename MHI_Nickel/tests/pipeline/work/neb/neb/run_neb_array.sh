#!/bin/bash
#SBATCH --job-name=neb_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --output=/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/neb/neb/run_neb_array_%j.out
#SBATCH --exclude=d3204
#SBATCH --array=1-1%50

module load OpenMPI/4.1.6

source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH


cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

echo "Node: $(hostname)  Start: $(date)"
LABEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/neb/neb/job_index.txt)
bash /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/neb/neb/${LABEL}/slurm_neb_${LABEL}.sh
echo "End: $(date)"

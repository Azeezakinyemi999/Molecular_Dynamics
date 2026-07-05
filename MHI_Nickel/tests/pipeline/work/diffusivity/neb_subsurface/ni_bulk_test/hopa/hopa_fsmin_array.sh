#!/bin/bash
#SBATCH --job-name=hopa_fsmin_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=sharing
#SBATCH --time=00:20:00
#SBATCH --output=/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/neb_subsurface/ni_bulk_test/hopa/hopa_fsmin_array_%j.out
#SBATCH --exclude=d3204
#SBATCH --array=1-3%2

module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH


cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

echo "Node: $(hostname)  Start: $(date)"
SID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/neb_subsurface/ni_bulk_test/hopa/job_index.txt)
bash /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/neb_subsurface/ni_bulk_test/hopa/${SID}/slurm_fsmin_${SID}.sh
echo "End: $(date)"

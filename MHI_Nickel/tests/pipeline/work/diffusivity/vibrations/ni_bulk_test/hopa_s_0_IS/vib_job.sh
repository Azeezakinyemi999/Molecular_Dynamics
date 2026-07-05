#!/bin/bash
#SBATCH --job-name=vib_hopa_s_0_IS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --output=/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/vibrations/ni_bulk_test/hopa_s_0_IS/vib_hopa_s_0_IS_%j.out
#SBATCH --exclude=d3204


module load OpenMPI/4.1.6

source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH


cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel

echo "Node: $(hostname)  Start: $(date)"
python /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/vibrations/ni_bulk_test/hopa_s_0_IS/vib_run.py
echo "End: $(date)"

#!/bin/bash
#SBATCH --job-name=npt_400K_Hastelloy_N_7_supercell_1H
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --output=/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/results/Hastelloy_N_7_supercell_1H/slurm_scripts/npt_400K_%j.out

# ── Environment ─────────────────────────────────────────
module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH


cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation

# ── Paths and config ─────────────────────────────────────
SCRIPT_PATH="$(realpath "$0")"
RESTART_GLOB="/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/results/Hastelloy_N_7_supercell_1H/structures/npt_boxdims_400K_after_heat.restart"
CUTOFF_SEC=28500
FLUSH_WAIT=30

echo "Job: $SLURM_JOB_ID  |  Node: $(hostname)  |  $(date)"

# ── Select command block ─────────────────────────────────
# ls -t glob | head -1 returns the most recent restart file.
# If the glob matches nothing, RESTART_FILE is empty.
RESTART_FILE=$(ls -t ${RESTART_GLOB} 2>/dev/null | head -1)

if [ -n "$RESTART_FILE" ]; then
    echo "Restart file found: $RESTART_FILE -- using restart commands."
    timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<'LAMMPS_BLOCK'
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/results/Hastelloy_N_7_supercell_1H/lammps_scripts/npt_restart_400K.lammps
LAMMPS_BLOCK
else
    echo "No restart file found -- starting fresh."
    timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<'LAMMPS_BLOCK'
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/calculation/results/Hastelloy_N_7_supercell_1H/lammps_scripts/npt_400K.lammps
LAMMPS_BLOCK
fi

EXIT_CODE=$?
echo "Command block exited with code $EXIT_CODE at $(date)"

# ── Handle exit code ─────────────────────────────────────
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "LAMMPS converged (exit 0). Simulation complete. No resubmission."
    exit 0

elif [ "$EXIT_CODE" -eq 124 ]; then
    echo "Timeout fired. Waiting ${FLUSH_WAIT}s for LAMMPS to flush restart files..."
    sleep "$FLUSH_WAIT"
    echo "Resubmitting: sbatch $SCRIPT_PATH"
    NEW_JOB=$(sbatch "$SCRIPT_PATH")
    echo "  --> $NEW_JOB"
    exit 0

else
    echo "Job failed with exit code $EXIT_CODE. Not resubmitting."
    exit $EXIT_CODE
fi

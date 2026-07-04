#!/bin/bash
#SBATCH --job-name=nvt_600K_ni_bulk_test_1H
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=sharing
#SBATCH --time=00:20:00
#SBATCH --output=/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/results/ni_bulk_test_1H/slurm_scripts/600K/nvt_600K_chain_%j.out
#SBATCH --exclude=d3204

# ── Environment ─────────────────────────────────────────
module load OpenMPI/4.1.6
module load cuda/12.3.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/akinyemi.az/miniforge3/envs/mace-lammps
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/cuda/12.3.0/lib64/stubs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akinyemi.az/miniforge3/envs/mace-lammps/lib:$LD_LIBRARY_PATH


cd /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity

# ── Paths and config ─────────────────────────────────────
SCRIPT_PATH="/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/results/ni_bulk_test_1H/slurm_scripts/600K/nvt_600K_chain.sh"
RESTART_GLOB="/projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/results/ni_bulk_test_1H/results/600K/checkpoints/nvt_600K.*.restart"
CUTOFF_SEC=300
FLUSH_WAIT=30

echo "Job: $SLURM_JOB_ID  |  Node: $(hostname)  |  $(date)"

# Clear any failure sentinel from a previous attempt of this chain.
rm -f "${SCRIPT_PATH}.failed"

# ── Select command block ─────────────────────────────────
# ls -t glob | head -1 returns the most recent restart file.
# If the glob matches nothing, RESTART_FILE is empty.
RESTART_FILE=$(ls -t ${RESTART_GLOB} 2>/dev/null | head -1)

N_EQUIL=500

if [ -n "$RESTART_FILE" ]; then
    # Extract the step number from the checkpoint filename.
    LAST_STEP=$(basename "$RESTART_FILE" .restart | grep -oE '[0-9]+$')
    if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -lt "$N_EQUIL" ]; then
        echo "Equil-phase restart (step=$LAST_STEP < n_equil=$N_EQUIL)."
        timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<'EQUIL_RST_BLOCK'
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/results/ni_bulk_test_1H/lammps_scripts/600K/nvt_600K_rst_equil.lammps -log /dev/null
EQUIL_RST_BLOCK
    else
        echo "Prod-phase restart (step=$LAST_STEP >= n_equil=$N_EQUIL)."
        timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<'PROD_RST_BLOCK'
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/results/ni_bulk_test_1H/lammps_scripts/600K/nvt_600K_rst_prod.lammps -log /dev/null
PROD_RST_BLOCK
    fi
else
    echo "No restart file found -- starting fresh."
    timeout --signal=SIGTERM --kill-after=60 "$CUTOFF_SEC" bash <<'FIRST_BLOCK'
/projects/westgroup/akinyemi.az/mace_lammps/lammps/build-mliap/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/results/ni_bulk_test_1H/lammps_scripts/600K/nvt_600K.lammps -log /projects/westgroup/akinyemi.az/mace_lammps/MHI_Nickel/tests/pipeline/work/diffusivity/results/ni_bulk_test_1H/results/600K/nvt_600K.log
FIRST_BLOCK
fi

EXIT_CODE=$?
echo "Command block exited with code $EXIT_CODE at $(date)"

# ── Handle exit code ─────────────────────────────────────
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "LAMMPS converged (exit 0). Simulation complete."
    touch "${SCRIPT_PATH}.done"
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
    touch "${SCRIPT_PATH}.failed"
    exit $EXIT_CODE
fi

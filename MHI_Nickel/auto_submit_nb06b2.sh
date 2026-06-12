#!/bin/bash
# auto_submit_nb06b2.sh
# Submits the NB06b2 array in chunks under QOS limit

ARRAY_SCRIPT="lammps_scripts/notebook06b2-Neb-dissociation/${SEED}/multi/array/run_neb_array.sh"
N_TOTAL=338830          # set to len(filtered_combinations)
CHUNK_SIZE=4         # tasks per submission
CHECK_INTERVAL=120   # seconds between queue checks
JOB_PREFIX="nb06b2_neb"

for start in $(seq 0 $CHUNK_SIZE $((N_TOTAL - 1))); do
    end=$((start + CHUNK_SIZE - 1))
    if [ $end -ge $N_TOTAL ]; then
        end=$((N_TOTAL - 1))
    fi
    
    # Wait until queue has room
    while true; do
        QUEUED=$(squeue -u $USER -h -o "%j" | grep -c "$JOB_PREFIX")
        if [ "$QUEUED" -lt 8 ]; then
            break
        fi
        echo "$(date): $QUEUED jobs in queue, waiting..."
        sleep $CHECK_INTERVAL
    done
    
    echo "Submitting tasks $start-$end"
    sbatch --array=${start}-${end} "$ARRAY_SCRIPT"
    sleep 10
done

echo "All chunks submitted"
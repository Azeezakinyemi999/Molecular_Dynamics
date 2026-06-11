#!/bin/bash
# auto_submit_nb06b2.sh — keeps queue at max capacity
SCRIPT=slurm_scripts/notebook06b2-Neb-dissociation/7/multi/run_neb_array.sh
INDEX_FILE=slurm_scripts/notebook06b2-Neb-dissociation/7/multi/job_index.txt
RESULT_DIR=results/notebook06b2-Neb-dissociation/7/multi
START=85
END=338829
QUEUE_MAX=8
CONCURRENT=4
INTERVAL=60

echo "=================================================="
echo "Auto-submit NB06b2 (tasks $START to $END)"
echo "Strategy: keep queue at $QUEUE_MAX tasks at all times"
echo "Started: $(date)"
echo "=================================================="

next=$START
while [ $next -le $END ]; do
    N=$(squeue -u $USER -h -t pending,running -r -n nb06b2_neb | wc -l)
    room=$((QUEUE_MAX - N))
    if [ $room -gt 0 ]; then
        remaining=$((END - next + 1))
        n_submit=$((room < remaining ? room : remaining))
        last=$((next + n_submit - 1))
        echo "$(date '+%H:%M:%S')  Queue has $N tasks, room for $room. Submitting $next-$last"
        sbatch --array=${next}-${last}%${CONCURRENT} $SCRIPT
        next=$((last + 1))
        sleep 5
    else
        echo "$(date '+%H:%M:%S')  Queue full ($N/$QUEUE_MAX), waiting..."
        sleep $INTERVAL
    fi
done

echo ""
echo "All tasks submitted at $(date). Waiting for queue to drain..."
while true; do
    N=$(squeue -u $USER -h -t pending,running -r -n nb06b2_neb | wc -l)
    [ $N -eq 0 ] && break
    echo "$(date '+%H:%M:%S')  $N tasks remaining..."
    sleep $INTERVAL
done

echo ""
echo "=================================================="
echo "All tasks finished at $(date)"
echo "=================================================="

# ── Completion check ─────────────────────────────────
TOTAL_LABELS=$(wc -l < $INDEX_FILE)
COMPLETED=$(find $RESULT_DIR -name "neb_barrier.txt" | wc -l)
MISSING=$((TOTAL_LABELS - COMPLETED))
echo ""
echo "Final status:"
echo "  Total   : $TOTAL_LABELS"
echo "  Completed: $COMPLETED"
echo "  Missing : $MISSING"
if [ $MISSING -gt 0 ]; then
    echo ""
    python3 - << PYEOF
import os
with open("slurm_scripts/notebook06b2-Neb-dissociation/7/multi/job_index.txt") as f:
    labels = [l.strip() for l in f if l.strip()]
result_dir = "results/notebook06b2-Neb-dissociation/7/multi"
missing_ids = [i for i, l in enumerate(labels)
    if not os.path.exists(f"{result_dir}/{l}/neb_barrier.txt")]
if missing_ids:
    id_str = ",".join(map(str, missing_ids))
    print(f"Missing task IDs: {id_str}")
    print(f"Resubmit: sbatch --array={id_str}%4 slurm_scripts/notebook06b2-Neb-dissociation/7/multi/run_neb_array.sh")
PYEOF
fi

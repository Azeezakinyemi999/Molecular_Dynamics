#!/bin/bash
# auto_submit_neb.sh — Keeps the queue at max capacity at all times

SCRIPT=slurm_scripts/notebook06-Neb-dissociation/7/multi/run_neb_array.sh
INDEX_FILE=slurm_scripts/notebook06-Neb-dissociation/7/multi/job_index.txt
RESULT_DIR=results/notebook06-Neb-dissociation/7/multi

START=0
END=123
QUEUE_MAX=8        # multigpu partition: max 8 submitted
CONCURRENT=48       # multigpu partition: max 4 running
INTERVAL=60

echo "=================================================="
echo "Auto-submit NEB sweep (tasks $START to $END)"
echo "Strategy: keep queue at $QUEUE_MAX tasks at all times"
echo "Started: $(date)"
echo "=================================================="

next=$START

while [ $next -le $END ]; do
    # Count tasks currently in queue
    N=$(squeue -u $USER -h -t pending,running -r -n nb06a_neb | wc -l)
    room=$((QUEUE_MAX - N))

    if [ $room -gt 0 ]; then
        # Submit as many as fit, up to remaining work
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
    N=$(squeue -u $USER -h -t pending,running -r -n nb06a_neb | wc -l)
    [ $N -eq 0 ] && break
    echo "$(date '+%H:%M:%S')  $N tasks remaining..."
    sleep $INTERVAL
done

echo ""
echo "=================================================="
echo "All tasks finished at $(date)"
echo "=================================================="

# Completion check
TOTAL_LABELS=$(wc -l < $INDEX_FILE)
COMPLETED=$(ls $RESULT_DIR/*/neb_barrier.txt 2>/dev/null | wc -l)
CONVERGED=$(grep -l "Converged   : True" $RESULT_DIR/*/neb_barrier.txt 2>/dev/null | wc -l)
NOT_CONVERGED=$(grep -l "Converged   : False" $RESULT_DIR/*/neb_barrier.txt 2>/dev/null | wc -l)
MISSING=$((TOTAL_LABELS - COMPLETED))

echo ""
echo "Final status:"
echo "  Total      : $TOTAL_LABELS"
echo "  Completed  : $COMPLETED"
echo "  Converged  : $CONVERGED"
echo "  Failed     : $NOT_CONVERGED"
echo "  Missing    : $MISSING"

if [ $MISSING -gt 0 ]; then
    echo ""
    python3 - << PYEOF
import os
with open("$INDEX_FILE") as f:
    labels = [l.strip() for l in f if l.strip()]
missing = [i for i, l in enumerate(labels)
           if not os.path.exists(f"$RESULT_DIR/{l}/neb_barrier.txt")]
if missing:
    print(f'Missing task IDs: {",".join(map(str, missing))}')
    print(f'Resubmit: sbatch --array={",".join(map(str, missing))}%4 $SCRIPT')
PYEOF
fi
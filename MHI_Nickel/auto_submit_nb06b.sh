#!/bin/bash
# auto_submit_nb06b.sh — Keeps the queue at max capacity at all times

SCRIPT=slurm_scripts/notebook06b-Neb-dissociation/7/run_array.sh
INDEX_FILE=slurm_scripts/notebook06b-Neb-dissociation/7/site_index.txt
RESULT_DIR=results/notebook06b-Neb-dissociation/7

START=0
END=170
QUEUE_MAX=8        # multigpu partition: max 8 submitted
CONCURRENT=4       # multigpu partition: max 4 running
INTERVAL=60

echo "=================================================="
echo "Auto-submit NB06b (tasks $START to $END)"
echo "Strategy: keep queue at $QUEUE_MAX tasks at all times"
echo "Started: $(date)"
echo "=================================================="

next=$START

while [ $next -le $END ]; do
    # Count NB06b tasks currently in queue
    N=$(squeue -u $USER -h -t pending,running -r -n nb06b_h | wc -l)
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
    N=$(squeue -u $USER -h -t pending,running -r -n nb06b_h | wc -l)
    [ $N -eq 0 ] && break
    echo "$(date '+%H:%M:%S')  $N tasks remaining..."
    sleep $INTERVAL
done

echo ""
echo "=================================================="
echo "All tasks finished at $(date)"
echo "=================================================="

# ── Completion check ─────────────────────────────────────────
TOTAL_LABELS=$(wc -l < $INDEX_FILE)
COMPLETED=$(ls $RESULT_DIR/h_atom_s_*.log 2>/dev/null | wc -l)
MISSING=$((TOTAL_LABELS - COMPLETED))

echo ""
echo "Final status:"
echo "  Total array  : $TOTAL_LABELS"
echo "  Completed    : $COMPLETED"
echo "  Missing      : $MISSING"

if [ $MISSING -gt 0 ]; then
    echo ""
    python3 - << PYEOF
import os
with open("$INDEX_FILE") as f:
    labels = [l.strip() for l in f if l.strip()]
missing = [i for i, l in enumerate(labels)
           if not os.path.exists(f"$RESULT_DIR/h_atom_{l}.log")]
if missing:
    print(f'Missing task IDs: {",".join(map(str, missing))}')
    print(f'Resubmit: sbatch --array={",".join(map(str, missing))}%4 $SCRIPT')
PYEOF
fi
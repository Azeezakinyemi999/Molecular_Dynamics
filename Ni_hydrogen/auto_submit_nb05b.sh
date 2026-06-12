#!/bin/bash
# auto_submit_nb05b.sh — Keeps the queue at max capacity at all times

SCRIPT=slurm_scripts/notebook05b-adsorption-energy/7/run_array.sh
CLEAN_SCRIPT=slurm_scripts/notebook05b-adsorption-energy/7/run_clean_slab.sh
H2_SCRIPT=slurm_scripts/notebook05b-adsorption-energy/7/run_h2_gas.sh
INDEX_FILE=slurm_scripts/notebook05b-adsorption-energy/7/site_index.txt
RESULT_DIR=results/notebook05b-adsorption-energy/7

START=0
END=170
QUEUE_MAX=8        # multigpu partition: max 8 submitted
CONCURRENT=4      # multigpu partition: max 4 running
INTERVAL=60

echo "=================================================="
echo "Auto-submit NB05b (tasks $START to $END + 2 refs)"
echo "Strategy: keep queue at $QUEUE_MAX tasks at all times"
echo "Started: $(date)"
echo "=================================================="

# ── Submit 2 reference jobs first ─────────────────────────────
echo ""
echo "Submitting reference jobs (clean_slab + h2_gas)..."
sbatch $CLEAN_SCRIPT
sleep 2
sbatch $H2_SCRIPT
sleep 5
echo ""

next=$START

while [ $next -le $END ]; do
    # Count NB05b tasks currently in queue (array + refs)
    N=$(squeue -u $USER -h -t pending,running -r \
        -n nb05b_ads,nb05b_clean_slab,nb05b_h2_gas | wc -l)
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
    N=$(squeue -u $USER -h -t pending,running -r \
        -n nb05b_ads,nb05b_clean_slab,nb05b_h2_gas | wc -l)
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
COMPLETED=$(ls $RESULT_DIR/ads_s_*.log 2>/dev/null | wc -l)
MISSING=$((TOTAL_LABELS - COMPLETED))

REF_OK=0
[ -f $RESULT_DIR/ads_clean_slab.log ] && REF_OK=$((REF_OK+1))
[ -f $RESULT_DIR/ads_h2_gas.log ]     && REF_OK=$((REF_OK+1))

echo ""
echo "Final status:"
echo "  Total array  : $TOTAL_LABELS"
echo "  Completed    : $COMPLETED"
echo "  Missing      : $MISSING"
echo "  References   : $REF_OK / 2"

if [ $MISSING -gt 0 ]; then
    echo ""
    python3 - << PYEOF
import os
with open("$INDEX_FILE") as f:
    labels = [l.strip() for l in f if l.strip()]
missing = [i for i, l in enumerate(labels)
           if not os.path.exists(f"$RESULT_DIR/ads_{l}.log")]
if missing:
    print(f'Missing task IDs: {",".join(map(str, missing))}')
    print(f'Resubmit: sbatch --array={",".join(map(str, missing))}%4 $SCRIPT')
PYEOF
fi

if [ $REF_OK -lt 2 ]; then
    echo ""
    echo "Missing references:"
    [ ! -f $RESULT_DIR/ads_clean_slab.log ] && echo "  sbatch $CLEAN_SCRIPT"
    [ ! -f $RESULT_DIR/ads_h2_gas.log ]     && echo "  sbatch $H2_SCRIPT"
fi
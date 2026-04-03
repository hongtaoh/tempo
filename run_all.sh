#!/bin/bash
# run_all.sh - Train lowdim and highdim models back-to-back.
# Starts the lowdim jobs (GPU-parallel via run_parallel.sh), waits for them
# to finish, then starts the highdim jobs.  No editing of config.yaml needed.
#
# GPU assignments are still read from the `gpus:` block in config.yaml.
# Adjust that block to reflect how many GPUs you have.

set -e

mkdir -p logs
TIMING_FILE="logs/training_time.txt"
TOTAL_START=$SECONDS
echo "Training started: $(date)" | tee "$TIMING_FILE"

wait_for_jobs() {
    echo "Waiting for training jobs to finish..."
    while true; do
        all_done=true
        for pid_file in logs/.pid_gpu*; do
            [ -f "$pid_file" ] || continue
            PID=$(cat "$pid_file")
            if ps -p "$PID" > /dev/null 2>&1; then
                all_done=false
                break
            fi
        done
        if [ "$all_done" = "true" ]; then break; fi
        sleep 30
    done
    echo "All jobs finished."
}

# ── Step 1: lowdim ────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " STEP 1 / 2 : Training lowdim models"
echo "========================================"
if [ -d "./models" ] && [ "$(ls -A ./models 2>/dev/null)" ]; then
    echo "Skipping: ./models already exists and is non-empty."
else
    STEP1_START=$SECONDS
    rm -f logs/.pid_gpu* && mkdir -p logs
    bash kill.sh 2>/dev/null || true
    bash run_parallel.sh --train_data_dir ./train --models_dir ./models
    wait_for_jobs
    STEP1_ELAPSED=$(( SECONDS - STEP1_START ))
    {
      printf "Lowdim training time:  %02d:%02d:%02d\n" $((STEP1_ELAPSED/3600)) $((STEP1_ELAPSED%3600/60)) $((STEP1_ELAPSED%60))
      [ -f logs/per_exp_lowdim.txt ] && cat logs/per_exp_lowdim.txt
    } | tee -a "$TIMING_FILE"
fi

# ── Step 2: highdim ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " STEP 2 / 2 : Training highdim models"
echo "========================================"
if [ -d "./models_highdim" ] && [ "$(ls -A ./models_highdim 2>/dev/null)" ]; then
    echo "Skipping: ./models_highdim already exists and is non-empty."
else
    STEP2_START=$SECONDS
    rm -f logs/.pid_gpu* && mkdir -p logs
    bash run_parallel.sh --train_data_dir ./train_highdim --models_dir ./models_highdim
    wait_for_jobs
    STEP2_ELAPSED=$(( SECONDS - STEP2_START ))
    {
      printf "Highdim training time: %02d:%02d:%02d\n" $((STEP2_ELAPSED/3600)) $((STEP2_ELAPSED%3600/60)) $((STEP2_ELAPSED%60))
      [ -f logs/per_exp_highdim.txt ] && cat logs/per_exp_highdim.txt
    } | tee -a "$TIMING_FILE"
fi

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
echo "Total training time:   $(printf '%02d:%02d:%02d' $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)) $((TOTAL_ELAPSED%60)))" | tee -a "$TIMING_FILE"
echo "Training ended:  $(date)" | tee -a "$TIMING_FILE"

echo ""
echo "All training complete."
echo "Next: bash postprocess.sh"

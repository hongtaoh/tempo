#!/bin/bash
# kill.sh - Kill all running training jobs

echo "Killing all training jobs..."

# Kill by saved PIDs (finds all .pid_gpu* files)
for pid_file in logs/.pid_gpu*; do
    if [ -f "$pid_file" ]; then
        gpu_name=$(basename $pid_file | sed 's/.pid_//')
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            # Kill the entire process group (shell wrapper + Python child).
            # start_new_session=True in run_parallel.sh makes the shell the
            # process group leader, so PGID == PID. kill -- -$PID kills both.
            kill -- -$PID 2>/dev/null && echo "Killed $gpu_name job (PID/PGID: $PID)"
        else
            echo "$gpu_name job not running (PID: $PID)"
        fi
    fi
done

# Also kill any remaining python tempo.py processes (backup)
pkill -f "python.*tempo.py" 2>/dev/null

echo ""
echo "Done. Check with: nvidia-smi"
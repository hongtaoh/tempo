#!/bin/bash
# run_parallel.sh - Run training on multiple GPUs in parallel
# Reads GPU assignments from config.yaml
# Usage:
#   ./run_parallel.sh                                              # uses dirs from config.yaml
#   ./run_parallel.sh --train_data_dir ./train --models_dir ./models

CONFIG_FILE="config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found"
    exit 1
fi

# Parse optional overrides
TRAIN_DATA_DIR=""
MODELS_DIR=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train_data_dir) TRAIN_DATA_DIR="$2"; shift ;;
        --models_dir) MODELS_DIR="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo $TIMESTAMP > logs/.latest_timestamp

echo "Starting parallel training..."
echo "Reading GPU config from $CONFIG_FILE"
echo "Timestamp: $TIMESTAMP"
[ -n "$TRAIN_DATA_DIR" ] && echo "train_data_dir override: $TRAIN_DATA_DIR"
[ -n "$MODELS_DIR" ] && echo "models_dir override: $MODELS_DIR"
echo ""

python3 << EOF
import yaml
import subprocess
import os

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

gpus = config.get('gpus', {})

if not gpus:
    print("No GPU config found in config.yaml")
    print("Add something like:")
    print("gpus:")
    print("  0: [exp1, exp2, exp3]")
    print("  1: [exp4, exp5, exp6]")
    exit(1)

extra_args = ""
train_data_dir = "$TRAIN_DATA_DIR"
models_dir = "$MODELS_DIR"
if train_data_dir:
    extra_args += f" --train_data_dir {train_data_dir}"
if models_dir:
    extra_args += f" --models_dir {models_dir}"

pids = []
for gpu_id, experiments in gpus.items():
    exp_str = ' '.join(experiments)
    log_file = f"logs/gpu{gpu_id}_$TIMESTAMP.log"

    print(f"Starting GPU {gpu_id}: {exp_str}")

    proc = subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES={gpu_id} python -u tempo.py --experiments {exp_str}{extra_args}",
        shell=True,
        stdout=open(log_file, 'w'),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

    pid_file = f"logs/.pid_gpu{gpu_id}"
    with open(pid_file, 'w') as pf:
        pf.write(str(proc.pid))

    print(f"  PID: {proc.pid}, Log: {log_file}")
    pids.append((gpu_id, proc.pid, log_file))

print("")
print("All jobs started in background (will survive terminal close)!")
print("")
print("Monitor progress with:")
for gpu_id, pid, log_file in pids:
    print(f"  tail -f {log_file}")
print("")
print(f"  tail -f logs/gpu*_$TIMESTAMP.log")
print("")
print("To kill all jobs: ./kill.sh")
print("You can safely close this terminal now.")
EOF
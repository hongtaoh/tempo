# TEMPO

This repository contains codes, data and models to reproduce the results reported in the submission of *TEMPO: Transformers for Temporal Disease Progression from Cross-Sectional Data*.

Additionally,

- [`https://github.com/hongtaoh/TEMPO_lowdim`](https://github.com/hongtaoh/TEMPO_lowdim) contains the reproducible codes for the low dimensional experiments for the benchmarking algorithms. 
- [`https://github.com/hongtaoh/TEMPO_highdim`](https://github.com/hongtaoh/TEMPO_lowdim) contains the reproducible codes for the high dimensional experiments for the benchmarking algorithms. 

## Table of Contents

- [Installation](#installation)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Running on ADNI Data](#running-on-adni-data)

## Installation

First, git clone this repository. Then create a virtual environment and install dependences:

```sh
uv venv --python python3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Hardware Requirements

The scripts are optimized for GPU execution. While CPU execution is supported, expect significantly longer runtimes.

## Quick Start

From the root directory, run these three commands in order:

```sh
bash gen.sh           # Generate all synthetic data (lowdim + highdim, train + test)
bash run_all.sh       # Train all models (lowdim then highdim)
bash postprocess.sh   # Run inference, flatten results, and generate plots
```

Before training, set the `gpus:` block in `config.yaml` to match your available GPUs:

```yaml
gpus:
  0: [exp1, exp2, exp3]
  1: [exp4, exp5, exp6, exp7, exp8, exp9]
```

That's it. No manual commenting/uncommenting of config files required.

## Step-by-Step Guide

### 1. Generating Synthetic Data

```sh
bash gen.sh
```

This generates all four data directories (`train`, `test`, `train_highdim`, `test_highdim`).
Already-existing directories are skipped automatically.

To regenerate from scratch, delete the directories first:

```sh
rm -rf train test train_highdim test_highdim
bash gen.sh
```

To generate only one dimension:

```sh
python3 gen.py --mode lowdim
python3 gen.py --mode highdim
```

Additional parameters (number of datasets, participants, experiments, etc.) can be modified in `config.yaml`.

### 2. Training

Set the `gpus:` block in `config.yaml` to assign experiments to GPUs, then run:

```sh
bash run_all.sh
```

This trains lowdim models first (saving to `models/`), waits for completion, then trains highdim models (saving to `models_highdim/`). Training logs are saved to the `logs/` folder. If a models directory already exists and is non-empty, that step is skipped automatically.

If running on a remote server or want training to survive terminal close, use:

```sh
mkdir -p logs && nohup bash run_all.sh > logs/run_all.log 2>&1 &
```

To retrain from scratch:

```sh
rm -rf models models_highdim
bash run_all.sh
```

To train only one dimension:

```sh
bash run_parallel.sh --train_data_dir ./train --models_dir ./models
bash run_parallel.sh --train_data_dir ./train_highdim --models_dir ./models_highdim
```

To stop running jobs:

```sh
bash kill.sh
```

### 3. Inference, Results Processing, and Visualization

```sh
bash postprocess.sh
```

This runs three steps automatically:

1. **Inference** — evaluates trained models on all test experiments, saving results to:
   - `results/cross_experiment_results_12bio.json`
   - `results/cross_experiment_results_100bio.json`

2. **Flatten results** — extracts diagonal (same-experiment) results into:
   - `results/12bio_tempo.csv`
   - `results/100bio_tempo.csv`

3. **Plots** — merges TEMPO results with benchmark algorithm results and generates figures in `plots/12bio/` and `plots/100bio/`.

Benchmark algorithm results are available in the `benchmark_algo_results/` folder.

## Running on ADNI Data

You'll need to request access to the ADNI data through [https://adni.loni.usc.edu/data-samples/adni-data/](https://adni.loni.usc.edu/data-samples/adni-data/). Download the `ADNIMERGE.csv`. Put it in the root directory of this repository. Then

```sh
python process_adni.py --raw ADNIMERGE.csv
```

You will see `adni.csv` and also `id_dx.json`. Then, run

```sh
python3 run_adni.py
```

Results will be saved to the `adni_results` folder.

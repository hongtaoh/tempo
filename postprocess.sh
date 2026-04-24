#!/bin/bash
# postprocess.sh - Run inference, flatten results, and generate plots.
# Run this after training is complete (bash run_all.sh).
#
# Usage:
#   bash postprocess.sh

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# # ── Step 1: Inference ─────────────────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " STEP 1 / 3 : Inference"
# echo "========================================"

# mkdir -p "$ROOT_DIR/results/12bio" "$ROOT_DIR/results/100bio"

# echo "Running inference on lowdim models..."
# python3 "$ROOT_DIR/inference.py" \
#     --models_dir models \
#     --test_dir test \
#     --output results/12bio/cross_experiment_results.json

# echo ""
# echo "Running inference on highdim models..."
# python3 "$ROOT_DIR/inference.py" \
#     --models_dir models_highdim \
#     --test_dir test_highdim \
#     --output results/100bio/cross_experiment_results.json

# # ── Step 2: Flatten results ───────────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " STEP 2 / 3 : Flatten results"
# echo "========================================"

# python3 "$ROOT_DIR/postprocess_scripts/flatten_diagonal_results.py" \
#     --input "$ROOT_DIR/results/12bio/cross_experiment_results.json" \
#     --output "$ROOT_DIR/results/12bio_tempo.csv"

# python3 "$ROOT_DIR/postprocess_scripts/flatten_diagonal_results.py" \
#     --input "$ROOT_DIR/results/100bio/cross_experiment_results.json" \
#     --output "$ROOT_DIR/results/100bio_tempo.csv"

# ── Step 3: Plots ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " STEP 3 / 3 : Generate plots"
echo "========================================"

mkdir -p "$ROOT_DIR/plots"
rm -rf "$ROOT_DIR/plots/12bio" "$ROOT_DIR/plots/100bio"

python3 "$ROOT_DIR/postprocess_scripts/merge_results.py"

python3 "$ROOT_DIR/postprocess_scripts/plot.py" \
    --input "$ROOT_DIR/plots/12bio.csv" \
    --output "$ROOT_DIR/plots/12bio/"

python3 "$ROOT_DIR/postprocess_scripts/plot.py" \
    --input "$ROOT_DIR/plots/100bio.csv" \
    --output "$ROOT_DIR/plots/100bio/"

python3 "$ROOT_DIR/postprocess_scripts/plot_adni_heat.py"
python3 "$ROOT_DIR/postprocess_scripts/plot_design_matrix.py"
python3 "$ROOT_DIR/postprocess_scripts/plot_sequence_mae_highdim.py"
python3 "$ROOT_DIR/postprocess_scripts/plot_sequence_mae_lowdim.py"

echo ""
echo "All done. Outputs:"
echo "  results/12bio/cross_experiment_results.json  (+ pivot CSVs)"
echo "  results/100bio/cross_experiment_results.json (+ pivot CSVs)"
echo "  results/12bio_tempo.csv"
echo "  results/100bio_tempo.csv"
echo "  plots/12bio.csv  plots/100bio.csv"
echo "  plots/12bio/     plots/100bio/"
echo "  plots/*.pdf      plots/*.png"

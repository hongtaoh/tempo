"""
Merge TEMPO results with benchmark algorithm results
"""

import pandas as pd
import os

if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

    # -----------------------------
    # Benchmark algorithm results
    # -----------------------------
    bench_12bio_path = os.path.join(ROOT_DIR, "benchmark_algo_results", "12bio_bench.csv")
    bench_100bio_path = os.path.join(ROOT_DIR, "benchmark_algo_results", "100bio_bench.csv")

    bench_12bio = pd.read_csv(bench_12bio_path)
    bench_100bio = pd.read_csv(bench_100bio_path)

    # Normalize algorithm names
    bench_12bio['algo'] = bench_12bio['algo'].replace('Conjugate Priors', 'SA-EBM')
    bench_100bio['algo'] = bench_100bio['algo'].replace('Conjugate Priors', 'SA-EBM')

    # -----------------------------
    # TEMPO results
    # -----------------------------
    tempo_12bio_path = os.path.join(ROOT_DIR, "results", "12bio_tempo.csv")
    tempo_100bio_path = os.path.join(ROOT_DIR, "results", "100bio_tempo.csv")

    tempo_12bio = pd.read_csv(tempo_12bio_path)
    tempo_100bio = pd.read_csv(tempo_100bio_path)

    assert list(bench_12bio.columns) == list(tempo_12bio.columns)
    assert list(bench_100bio.columns) == list(tempo_100bio.columns)

    # -----------------------------
    # Output paths (plots/)
    # -----------------------------
    output_12bio_path = os.path.join(PLOTS_DIR, "12bio.csv")
    output_100bio_path = os.path.join(PLOTS_DIR, "100bio.csv")

    # -----------------------------
    # Concatenate (row-wise)
    # -----------------------------
    merged_12bio = pd.concat([bench_12bio, tempo_12bio], ignore_index=True)
    merged_100bio = pd.concat([bench_100bio, tempo_100bio], ignore_index=True)

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs(PLOTS_DIR, exist_ok=True)
    merged_12bio.to_csv(output_12bio_path, index=False)
    merged_100bio.to_csv(output_100bio_path, index=False)

    print("Saved:")
    print(" -", output_12bio_path, merged_12bio.shape)
    print(" -", output_100bio_path, merged_100bio.shape)
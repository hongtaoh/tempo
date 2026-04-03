"""
python flatten_diagonal_results.py \
  --input cross_experiment_results_12bio.json \
  --output 12bio_tempo.csv

python flatten_diagonal_results.py \
  --input cross_experiment_results_100bio.json \
  --output 100bio_tempo.csv

"""

import json
import re
import argparse
import pandas as pd
import numpy as np


EXPERIMENTS = [
    "sn_kjOrdinalDM_xnjNormal",
    "sn_kjOrdinalDM_xnjNonNormal",
    "sn_kjOrdinalUniform_xnjNormal",
    "sn_kjOrdinalUniform_xnjNonNormal",
    "sn_kjContinuousBeta_sigmoid",
    "sn_kjContinuousBeta_xnjNormal",
    "sn_kjContinuousBeta_xnjNonNormal",
    "xiNearNormalWithNoise_kjContinuousBeta_sigmoid",
    "xiNearNormalWithNoise_kjContinuousBeta_xnjNormal",
]


titles = [
    "Exp 1: Ordinal kj (DM) + EBM + X (Normal)",
    "Exp 2: Ordinal kj (DM) + EBM + X (Non-Normal)",
    "Exp 3: Ordinal kj (Uniform) + EBM + X (Normal)",
    "Exp 4: Ordinal kj (Uniform) + EBM + X (Non-Normal)",
    "Exp 5: Continuous kj (Beta) + Sigmoid",
    "Exp 6: Continuous kj (Beta) + EBM + X (Normal)",
    "Exp 7: Continuous kj (Beta) + EBM + X (Non-Normal)",
    "Exp 8: xi (Noise) + Continuous kj (Beta) + Sigmoid",
    "Exp 9: xi (Noise) + Continuous kj (Beta) + EBM + X (Normal)",
]

def extract_components(exp_key: str):
    """
    Parse keys like:
    j726_r0.21_Esn_kjOrdinalDM_xnjNormal_m0
    -> (J, R, E, M)
    """
    pattern = r'^j(\d+)_r([\d.]+)_E(.*?)_m(\d+)$'
    m = re.match(pattern, exp_key)
    if not m:
        return None
    J, R, E, M = m.groups()
    return int(J), float(R), E, int(M)


def main(input_path: str, output_path: str):
    code2title = dict(zip(EXPERIMENTS, titles))
    title2code = {v: k for k, v in code2title.items()}

    with open(input_path, "r") as f:
        data = json.load(f)

    rows = []
    ALGO_NAME = "TEMPO"

    # iterate over exp1, exp2, ..., exp12
    for train_exp, payload in data.items():
        tested_on = payload.get("tested_on", {})
        same_exp_block = tested_on.get(train_exp)

        if same_exp_block is None:
            continue

        for exp_key, metrics in same_exp_block.items():
            parsed = extract_components(exp_key)
            if parsed is None:
                continue

            J, R, E, M = parsed

            rows.append({
                "J": J,
                "R": R,
                "E": code2title[E],
                "M": M,
                "algo": ALGO_NAME,
                'runtime': np.nan,
                "kendalls_tau": metrics["tau"],
                "mae": metrics["mae"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"✓ Wrote {len(df)} rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten diagonal (same-exp) results from cross-experiment JSON"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to cross_experiment_results_*.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output CSV",
    )

    args = parser.parse_args()
    main(args.input, args.output)

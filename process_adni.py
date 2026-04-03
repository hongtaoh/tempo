#!/usr/bin/env python3
"""
Standalone ADNI preprocessing script.

Creates:
  - adni.csv     (long-format dataframe for modeling)
  - id_dx.json   (participant_id -> diagnosis label)

Only requires ADNIMERGE.csv as input.
"""

import os
import json
import copy
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from collections import Counter


# ============================================================
# Functions (standalone)
# ============================================================

def get_adni_filtered(
    raw: str,
    meta_data: List[str],
    select_biomarkers: List[str],
    diagnosis_list: List[str],
) -> pd.DataFrame:
    """Filter ADNIMERGE.csv to baseline subjects with complete biomarkers."""
    # df = pd.read_csv(raw, usecols=meta_data + select_biomarkers)
    df = pd.read_csv(raw)[meta_data + select_biomarkers]

    # Baseline + valid diagnosis
    df = df[df["VISCODE"] == "bl"]
    df = df[df["DX_bl"].isin(diagnosis_list)]

    # Numeric conversion
    for col in select_biomarkers:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing biomarkers
    df = df.dropna(subset=select_biomarkers).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    print(len(df))
    if len(df.PTID.unique()) == len(df):
        print("No duplicates!")
    else:
        print("Data has duplicates!")

    # Diagnosis distribution
    counts = Counter(df["DX_bl"])
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"{k}: {v} ({100*v/total:.1f}%)")

    print("----------------------------------------------------")

    # Cohort distribution
    counts = Counter(df["COLPROT"])
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"{k}: {v} ({100*v/total:.1f}%)")

    return df


def process_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Produce long-format dataframe (adni.csv)
    and participant -> diagnosis mapping.
    """
    df = df.copy()

    # Assign participant IDs
    df["PTID"] = range(len(df))

    # Remove _bl suffix
    df.columns = df.columns.str.replace("_bl", "", regex=False)

    # Diagnosis mapping
    participant_dx_dict = dict(zip(df.PTID, df.DX))

    # ICV normalization
    df["VentricleNorm"] = df["Ventricles"] / df["ICV"]
    df["HippocampusNorm"] = df["Hippocampus"] / df["ICV"]
    df["WholeBrainNorm"] = df["WholeBrain"] / df["ICV"]
    df["EntorhinalNorm"] = df["Entorhinal"] / df["ICV"]
    df["FusiformNorm"] = df["Fusiform"] / df["ICV"]
    df["MidTempNorm"] = df["MidTemp"] / df["ICV"]

    # Drop unused raw columns
    df.drop(
        [
            "VISCODE",
            "COLPROT",
            "DX",
            "ICV",
            "Ventricles",
            "Hippocampus",
            "WholeBrain",
            "Entorhinal",
            "Fusiform",
            "MidTemp",
        ],
        axis=1,
        inplace=True,
    )

    # Diseased indicator
    df["diseased"] = [bool(dx != "CN") for dx in participant_dx_dict.values()]
    df["participant"] = df["PTID"]

    # Long format
    df_long = pd.melt(
        df,
        id_vars=["participant", "diseased"],
        var_name="biomarker",
        value_name="measurement",
    )

    return df_long, participant_dx_dict


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adni.csv and id_dx.json")
    parser.add_argument(
        "--raw",
        required=True,
        help="Path to ADNIMERGE.csv",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Configuration
    meta_data = ["PTID", "DX_bl", "VISCODE", "COLPROT"]

    select_biomarkers = [
        "MMSE_bl",
        "Ventricles_bl",
        "WholeBrain_bl",
        "MidTemp_bl",
        "Fusiform_bl",
        "Entorhinal_bl",
        "Hippocampus_bl",
        "ADAS13_bl",
        "PTAU_bl",
        "TAU_bl",
        "ABETA_bl",
        "RAVLT_immediate_bl",
        "ICV_bl",
    ]

    diagnosis_list = ["CN", "EMCI", "LMCI", "AD"]

    # Pipeline
    adni_filtered = get_adni_filtered(
        args.raw, meta_data, select_biomarkers, diagnosis_list
    )
    df_long, participant_dx_dict = process_data(adni_filtered)

    # Save outputs
    adni_csv = os.path.join(out_dir, "adni.csv")
    df_long.to_csv(adni_csv, index=False)

    id_dx_json = os.path.join(out_dir, "id_dx.json")
    with open(id_dx_json, "w") as f:
        json.dump(
            {str(k): v for k, v in participant_dx_dict.items()},
            f,
            indent=2,
        )

    print(f"Wrote:\n  {adni_csv}\n  {id_dx_json}")


if __name__ == "__main__":
    main()

import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Load rank data from adni_results JSON files
ADNI_DIR = os.path.join(ROOT_DIR, "adni_results")
exp_orders = {}
for exp_json in sorted(glob.glob(os.path.join(ADNI_DIR, "exp*", "exp*_results.json"))):
    with open(exp_json) as f:
        result = json.load(f)
    exp_orders[result["experiment"]] = result["true_order"]

experiments = sorted(exp_orders.keys())
n_exps = len(experiments)
all_biomarkers = list(exp_orders[experiments[0]].keys())

# 2. Build DataFrame with per-experiment ranks and mean rank
rank_data = {"Biomarker": all_biomarkers}
for exp in experiments:
    rank_data[exp] = [exp_orders[exp][b] for b in all_biomarkers]
df = pd.DataFrame(rank_data)
df["MeanRank"] = df[experiments].mean(axis=1)
df = df.sort_values("MeanRank").reset_index(drop=True)

biomarkers_sorted = df["Biomarker"].tolist()
n_biomarkers = len(all_biomarkers)
y_labels = [f"{name} ({i+1})" for i, name in enumerate(biomarkers_sorted)]

# 3. Build probability matrix
df_ranks = df.set_index("Biomarker")[experiments]
freq_matrix = pd.DataFrame(0.0, index=biomarkers_sorted, columns=range(1, n_biomarkers + 1))
for b in biomarkers_sorted:
    for r in df_ranks.loc[b]:
        freq_matrix.loc[b, r] += 1
freq_matrix = freq_matrix / n_exps

# 4. Plot
plt.figure(figsize=(5.5, 6))
sns.set_style("white")

ax = sns.heatmap(
    freq_matrix,
    annot=False,
    cmap="mako_r",
    vmin=0,
    vmax=1,
    cbar=True,
    cbar_kws={"label": "Probability", "shrink": 0.5},
    xticklabels=range(1, n_biomarkers + 1),
    yticklabels=y_labels,
    square=False
)

plt.title("ADNI Consensus Ordering", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Stage Position", fontsize=12, fontweight="bold")
plt.ylabel("Biomarker", fontsize=12, fontweight="bold")
plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ADNI_Professional_Heatmap.png"), bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(PLOTS_DIR, "ADNI_Professional_Heatmap.pdf"), bbox_inches="tight", dpi=300)
plt.show()

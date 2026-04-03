import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Load Data
csv_path = os.path.join(ROOT_DIR, "results", "12bio", "seq_mae_pivot.csv")
raw = pd.read_csv(csv_path, index_col=0)

# Drop summary row/column
df = raw.drop(index="col_mean", errors="ignore").drop(columns="row_mean", errors="ignore")

# Rename exp1 -> Exp 1, etc.
n = len(df)
rename = {f"exp{i}": f"Exp {i}" for i in range(1, n + 1)}
df = df.rename(index=rename, columns=rename)

# 2. Setup Plot
plt.figure(figsize=(10, 8))
sns.set_theme(style="white")

# 3. Draw Heatmap
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues",
                 linewidths=.5, cbar_kws={'label': 'Sequence MAE'},
                 annot_kws={"size": 10})

# 4. Labeling
experiments = list(df.index)
plt.title('Generalization Matrix: Sequence MAE Across Conditions (Low Dimensional)', fontsize=14, pad=20)
plt.ylabel('Training Condition', fontsize=12, fontweight='bold')
plt.xlabel('Testing Condition', fontsize=12, fontweight='bold')

for i in range(len(experiments)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2, clip_on=False))

plt.axvline(x=4, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=4, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'sequence_mae_lowdim.pdf'), dpi=300)
plt.savefig(os.path.join(PLOTS_DIR, 'sequence_mae_lowdim.png'), dpi=300)
plt.show()

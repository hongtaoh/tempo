import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Data Setup
data = [
    ["1", "EBM",     "Disc",  "Disc", "N", "N"],
    ["2", "EBM",     "Disc",  "Disc", "N", "I"],
    ["3", "EBM",     "Disc",  "Disc", "U", "N"],
    ["4", "EBM",     "Disc",  "Disc", "U", "I"],
    ["5", "Sigmoid", "Disc",  "Cont", "B", "N->S"],
    ["6", "EBM",     "Disc",  "Cont", "B", "N"],
    ["7", "EBM",     "Disc",  "Cont", "B", "I"],
    ["8", "Sigmoid", "Cont*", "Cont", "B", "N->S"],
    ["9", "EBM",     "Cont*", "Cont", "B", "N"],
]

columns = ["Exp", "Framework", "Event Seq", "Stage Type", "Stage Dist", "Bio Dist"]
df = pd.DataFrame(data, columns=columns)

# 2. Plot Setup
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.set_xlim(0, 6)
ax.set_ylim(-2.5, 10)

headers = ["Exp", "Framework", "Event\nSequence", "Stage\nType", "Stage\nDist.", "Biomarker\nDist."]
for i, h in enumerate(headers):
    ax.text(i + 0.5, 9.5, h, ha='center', va='center', fontweight='bold', fontsize=11)

# 3. Drawing Logic
for i, row in df.iterrows():
    y = 8.5 - i
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((0, y-0.5), 6, 1, color='gray', alpha=0.06, zorder=0))

    ax.text(0.5, y, row["Exp"], ha='center', va='center', fontsize=10.5)

    f_color = "#D1E8FF" if row["Framework"] == "EBM" else "#FFE0B2"
    rect = patches.FancyBboxPatch((1.1, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.02",
                                  color=f_color, ec='gray', lw=0.5)
    ax.add_patch(rect)
    ax.text(1.5, y, row["Framework"], ha='center', va='center', fontsize=10, fontweight='bold')

    for col_idx, col_name in zip([2.5, 3.5], ["Event Seq", "Stage Type"]):
        val = row["Event Seq"] if col_idx == 2.5 else row["Stage Type"]
        if "Disc" in val:
            ax.add_patch(plt.Rectangle((col_idx-0.18, y-0.18), 0.36, 0.36, fill=False, edgecolor='#34495e', lw=1.6))
        elif val == "Cont*":
            ax.add_patch(patches.Ellipse((col_idx, y), 0.5, 0.3, fill=False, edgecolor='#34495e', lw=1.6, linestyle='dashed'))
        else:
            ax.add_patch(patches.Ellipse((col_idx, y), 0.5, 0.3, color='#34495e'))

    if row["Stage Dist"] == "N":
        s_dist = r"$\mathcal{N}$"
    elif row["Stage Dist"] == "B":
        s_dist = r"$\mathcal{B}$"
    else:
        s_dist = r"$\mathcal{U}$"
    ax.text(4.5, y, s_dist, ha='center', va='center', fontsize=13)

    if row["Bio Dist"] == "N": b_dist = r"$\mathcal{N}$"
    elif row["Bio Dist"] == "I": b_dist = r"$\mathcal{I}$"
    else: b_dist = r"$\mathcal{N} \rightarrow S$"
    ax.text(5.5, y, b_dist, ha='center', va='center', fontsize=12)

# 4. Legend
leg_y_row1 = -0.6
leg_y_row2 = -1.3
leg_y_row3 = -2.0
leg_fs = 10

ax.add_patch(plt.Rectangle((0.2, leg_y_row1-0.15), 0.25, 0.25, fill=False, edgecolor='#34495e', lw=1.5))
ax.text(0.5, leg_y_row1, "Discrete", fontsize=leg_fs, va='center')

ax.add_patch(patches.Ellipse((1.6, leg_y_row1), 0.35, 0.2, color='#34495e'))
ax.text(1.85, leg_y_row1, "Continuous", fontsize=leg_fs, va='center')

ax.text(3.3, leg_y_row1, r"$\mathcal{N}$:", fontsize=13, va='center', fontweight='bold')
ax.text(3.55, leg_y_row1, "Normal", fontsize=leg_fs, va='center')

ax.text(4.7, leg_y_row1, r"$\mathcal{U}$:", fontsize=13, va='center', fontweight='bold')
ax.text(4.95, leg_y_row1, "Uniform", fontsize=leg_fs, va='center')

ax.text(0.2, leg_y_row2, r"$\mathcal{I}$:", fontsize=13, va='center', fontweight='bold')
ax.text(0.45, leg_y_row2, "Irregular (Non-normal)", fontsize=leg_fs, va='center')

ax.text(2.8, leg_y_row2, r"$\mathcal{N} \to S$:", fontsize=13, va='center', fontweight='bold')
ax.text(3.5, leg_y_row2, r"Healthy Normal $\rightarrow$ Sigmoid Abnormality", fontsize=leg_fs, va='center')

ax.add_patch(patches.Ellipse((0.45, leg_y_row3), 0.35, 0.2, fill=False, edgecolor='#34495e', lw=1.5, linestyle='dashed'))
ax.text(0.7, leg_y_row3, r"Cont* (Continuous + $\xi$ noise)", fontsize=leg_fs, va='center')

ax.text(3.3, leg_y_row3, r"$\mathcal{B}$:", fontsize=13, va='center', fontweight='bold')
ax.text(3.55, leg_y_row3, r"Beta$(5,2)$ (skewed, continuous $k_j$)", fontsize=leg_fs, va='center')

ax.axhline(9, color='black', lw=1.5)
ax.set_axis_off()

# 5. Output
plt.savefig(os.path.join(PLOTS_DIR, 'design_matrix.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(PLOTS_DIR, 'design_matrix.png'), bbox_inches='tight', dpi=300)
plt.show()

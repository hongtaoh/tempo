"""
Publication-Quality Algorithm Performance Comparison
=====================================================
Compares Normalized Tau Distance and MAE across different algorithms with 95% CI.

Updates:
- Figure 2: Plots error bars for 95% CI.
- Figure 4: X-axis limit set to 0.6.
- Figures 5 & 6: Legends STRICTLY sorted by mean values (Ascending).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os
import sys

# ============================================================
# Argument Parsing
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate publication-quality algorithm performance plots.")
    parser.add_argument('-i', '--input', type=str, default='12bio.csv',
                        help='Path to the input CSV file (default: 12bio.csv)')
    parser.add_argument('-o', '--output', type=str, default='./',
                        help='Directory to save output figures (default: current dir)')
    return parser.parse_args()

args = parse_arguments()

# Validate input
if not os.path.exists(args.input):
    print(f"Error: Input file '{args.input}' not found.")
    sys.exit(1)

# Ensure output directory exists
if not os.path.exists(args.output):
    os.makedirs(args.output)
    print(f"Created output directory: {args.output}")

INPUT_FILE = args.input
OUTPUT_DIR = os.path.join(args.output, '')
TAU_YLIM = 0.6 if '100bio' in INPUT_FILE else 0.5

# ============================================================
# Configuration
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'errorbar.capsize': 3,
})

# STRICLY DESIGNATED COLORS
COLORS = {
    'TEMPO': '#1f77b4',              # Blue
    'SA-EBM': '#ff7f0e',             # Orange
    'UCL GMM': '#2ca02c',            # Green
    'DEBM GMM': '#d62728',           # Red
    'DEBM': '#9467bd',               # Purple
    'UCL KDE': '#8c564b',            # Brown
}

RANDOM_COLOR = '#808080'  

# ============================================================
# Helper Functions
# ============================================================

def compute_ci_95(data):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, n - 1)
    return pd.Series({
        'mean': mean,
        'std': np.std(data, ddof=1),
        'sem': sem,
        'ci_95': ci,
        'ci_lower': mean - ci,
        'ci_upper': mean + ci,
        'n': n
    })

def shorten_exp_title(exp):
    parts = exp.split(': ', 1)
    if len(parts) == 2:
        exp_num = parts[0]
        desc = parts[1]
        desc_short = desc.replace('S + ', '').replace('xi ', 'ξ ')
        desc_short = desc_short.replace('kj ', 'k ')
        desc_short = desc_short.replace('Ordinal', 'Ord.')
        desc_short = desc_short.replace('Continuous', 'Cont.')
        desc_short = desc_short.replace('Non-Normal', 'Non-Norm.')
        desc_short = desc_short.replace('Normal', 'Norm.')
        desc_short = desc_short.replace('Uniform', 'Unif.')
        desc_short = desc_short.replace('Skewed', 'Skew.')
        desc_short = desc_short.replace('Beta', 'β')
        desc_short = desc_short.replace('Sigmoid', 'Sig.')
        desc_short = desc_short.replace('Noise', 'ε')
        return f"{exp_num}:\n{desc_short}"
    return exp

# ============================================================
# Load and Prepare Data
# ============================================================

print(f"Loading data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
df = df[['J', 'R', 'E', 'M', 'algo', 'kendalls_tau', 'mae']]

tau_stats = df.groupby('algo')['kendalls_tau'].apply(compute_ci_95).unstack()
mae_stats = df.groupby('algo')['mae'].apply(compute_ci_95).unstack()

# Sort: Ascending (Smaller is Better)
algo_order = tau_stats.sort_values('mean', ascending=True).index.tolist()
mae_order = mae_stats.sort_values('mean', ascending=True).index.tolist()

print("\n" + "="*60)
print("NORMALIZED TAU DISTANCE STATISTICS (Lower is Better)")
print("="*60)
print(tau_stats[['mean', 'std', 'ci_95', 'n']].round(4))

# ============================================================
# Figure 1: Bar Plot Comparison
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Normalized Tau Distance
ax1 = axes[0]
x_pos = np.arange(len(algo_order))
bar_colors = [COLORS.get(algo, '#888888') for algo in algo_order]
means = [tau_stats.loc[algo, 'mean'] for algo in algo_order]
cis = [tau_stats.loc[algo, 'ci_95'] for algo in algo_order]

ax1.bar(x_pos, means, yerr=cis, color=bar_colors, 
        edgecolor='black', linewidth=0.5, capsize=4, 
        error_kw={'linewidth': 1.2, 'ecolor': 'black'})
ax1.set_ylabel("Norm. Tau Dist.", fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(algo_order, rotation=45, ha='right')
ax1.set_ylim(0, 1.0)
ax1.set_title('(A) Normalized Tau Distance (↓ better)', fontweight='bold', loc='left')

# Random Guessing Line (Gray)
ax1.axhline(y=0.5, color=RANDOM_COLOR, linestyle=':', linewidth=1.5, alpha=0.8)
ax1.text(len(algo_order)-1, 0.52, 'Random Guessing (0.5)', color=RANDOM_COLOR, 
         fontsize=8, ha='right', va='bottom', fontweight='bold')

for i, (mean, ci) in enumerate(zip(means, cis)):
    ax1.text(i, mean + ci + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

# Panel B: MAE
ax2 = axes[1]
bar_colors_mae = [COLORS.get(algo, '#888888') for algo in mae_order]
means_mae = [mae_stats.loc[algo, 'mean'] for algo in mae_order]
cis_mae = [mae_stats.loc[algo, 'ci_95'] for algo in mae_order]

ax2.bar(x_pos, means_mae, yerr=cis_mae, color=bar_colors_mae,
        edgecolor='black', linewidth=0.5, capsize=4,
        error_kw={'linewidth': 1.2, 'ecolor': 'black'})
ax2.set_ylabel('MAE', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(mae_order, rotation=45, ha='right')
ax2.set_title('(B) Mean Absolute Error (↓ better)', fontweight='bold', loc='left')
for i, (mean, ci) in enumerate(zip(means_mae, cis_mae)):
    ax2.text(i, mean + ci + 0.03, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig1_barplot_comparison.png', dpi=300, facecolor='white')
plt.savefig(f'{OUTPUT_DIR}fig1_barplot_comparison.pdf', facecolor='white')
plt.close()
print(f"Saved: {OUTPUT_DIR}fig1_barplot_comparison.png/pdf")

# ============================================================
# Figure 2: Forest Plot with CI
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Panel A: Normalized Tau Distance
ax1 = axes[0]
y_pos = np.arange(len(algo_order))[::-1]
for i, algo in enumerate(algo_order):
    mean = tau_stats.loc[algo, 'mean']
    ci = tau_stats.loc[algo, 'ci_95']
    ci_lower = tau_stats.loc[algo, 'ci_lower']
    ci_upper = tau_stats.loc[algo, 'ci_upper']
    
    # Plot Error Bars (xerr=ci is the half-width for 95% CI)
    ax1.errorbar(mean, y_pos[i], xerr=ci, fmt='o', color=COLORS.get(algo, '#888888'), 
                 markersize=8, capsize=5, capthick=1.5, elinewidth=1.5)
    
    # Text with CI
    ax1.text(ci_upper + 0.02, y_pos[i], 
             f'{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]',
             va='center', fontsize=9)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(algo_order)
ax1.set_xlabel("Norm. Tau Dist. (95% CI)", fontweight='bold')
ax1.set_title('(A) Normalized Tau Distance', fontweight='bold', loc='left')
ax1.set_xlim(0, 1.15)

# Random Guessing Line (Gray) + Text Top Right
ax1.axvline(x=0.5, color=RANDOM_COLOR, linestyle=':', linewidth=1.5, alpha=0.8)
ax1.text(0.51,0.2, 'Random Guessing (0.5)', color=RANDOM_COLOR, 
         fontsize=9, ha='left', va='bottom', fontweight='bold')

# Panel B: MAE
ax2 = axes[1]
for i, algo in enumerate(mae_order):
    mean = mae_stats.loc[algo, 'mean']
    ci = mae_stats.loc[algo, 'ci_95']
    ci_lower = mae_stats.loc[algo, 'ci_lower']
    ci_upper = mae_stats.loc[algo, 'ci_upper']
    
    ax2.errorbar(mean, y_pos[i], xerr=ci, fmt='o', color=COLORS.get(algo, '#888888'),
                 markersize=8, capsize=5, capthick=1.5, elinewidth=1.5)
    
    # Text with CI
    ax2.text(ci_upper + 0.05, y_pos[i],
             f'{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]',
             va='center', fontsize=9)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(mae_order)
ax2.set_xlabel('MAE (95% CI)', fontweight='bold')
ax2.set_title('(B) Mean Absolute Error', fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig2_forest_plot.png', dpi=300, facecolor='white')
plt.savefig(f'{OUTPUT_DIR}fig2_forest_plot.pdf', facecolor='white')
plt.close()
print(f"Saved: {OUTPUT_DIR}fig2_forest_plot.png/pdf")

# ============================================================
# Figure 3: Box Plot with Ordered Strips (The "Sweet Spot" Style)
# ============================================================

# ── Only change fonts for this figure ────────────────────────
with plt.rc_context({
    'font.size': 14,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
}):

    # 1. Setup Mapping
    unique_exps = sorted(df['E'].unique()) 
    exp_cmap = plt.get_cmap('tab10')
    MARKERS = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'X']
    offsets = np.linspace(-0.24, 0.24, len(unique_exps))

    EXP_MAP = {
        exp: {
            'color': exp_cmap(i), 
            'marker': MARKERS[i], 
            'offset': offsets[i]
        } for i, exp in enumerate(unique_exps)
    }

    # --- UNIFIED STYLE CONSTANTS (Ensures Panel A and B are identical) ---
    BOX_ALPHA = 0.30     # Slightly increased for a "pronounced" feel
    DOT_ALPHA = 0.45     # Slightly decreased from "High Impact" to be less busy
    BOX_LINE_WIDTH = 1.2
    MEDIAN_WIDTH = 2.0
    DOT_SIZE = 16
    # ---------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))

    for panel_idx, (metric, ax, order, title, ylabel) in enumerate([
        ('kendalls_tau', axes[0], algo_order, '(A) Norm. Tau Dist. Distribution', 'Norm. Tau Dist.'),
        ('mae', axes[1], mae_order, '(B) MAE Distribution', 'MAE')
    ]):
        
        plot_data = [df[df['algo'] == algo][metric].values for algo in order]
        
        # 1. Plot the Boxes
        bp = ax.boxplot(plot_data, positions=np.arange(len(order)), widths=0.6,
                        patch_artist=True, showfliers=False, zorder=2)
        
        for patch, algo in zip(bp['boxes'], order):
            patch.set_facecolor(COLORS.get(algo, '#888888'))
            patch.set_alpha(BOX_ALPHA)     
            patch.set_edgecolor('black')
            patch.set_linewidth(BOX_LINE_WIDTH)
        
        # Style the Median line
        plt.setp(bp['medians'], color='orange', linewidth=MEDIAN_WIDTH, zorder=5)

        # 2. Plot the Vertical Ordered Strips
        for i, algo in enumerate(order):
            algo_df = df[df['algo'] == algo]
            
            for exp in unique_exps:
                exp_subset = algo_df[algo_df['E'] == exp]
                if exp_subset.empty:
                    continue
                
                vals = exp_subset[metric].values
                # Position = Algorithm_Index + Experiment_Offset
                micro_jitter = np.random.normal(0, 0.005, len(vals))
                x_pos = i + EXP_MAP[exp]['offset'] + micro_jitter
                
                ax.scatter(x_pos, vals, 
                        alpha=DOT_ALPHA, 
                        s=DOT_SIZE, 
                        color=EXP_MAP[exp]['color'], 
                        marker=EXP_MAP[exp]['marker'],
                        edgecolor='none', 
                        zorder=3)
            
            # Mean marker (White Diamond) - Top Layer
            ax.scatter(i, np.mean(algo_df[metric]), marker='D', s=80, color='white', 
                    edgecolor='black', linewidth=1.5, zorder=10)

        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=10, ha='center', fontweight='bold')
        # ax.set_xticklabels(order, rotation=0, ha='center', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', loc='left')

        if panel_idx == 0:
            ax.axhline(y=0.5, color=RANDOM_COLOR, linestyle=':', linewidth=1.5, alpha=0.8)
            ax.text(-0.4, 0.51, 'Random Guessing', color=RANDOM_COLOR, 
                    fontsize=12, ha='left', va='bottom', fontweight='bold')

    # # 3. COMPACT 2-ROW LEGEND
    # legend_elements = [
    #     plt.Line2D([0], [0], marker=EXP_MAP[exp]['marker'], color='w', 
    #             markerfacecolor=EXP_MAP[exp]['color'], 
    #             markersize=10, markeredgecolor='gray', 
    #             label=shorten_exp_title(exp).replace('\n', ' ')) 
    #     for exp in unique_exps
    # ]

    # leg = fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
    #         title="Experiment Columns (Ordered Left-to-Right: 1 to 9)",
    #          bbox_to_anchor=(0.5, 0.02), frameon=True)
    
    # # Bold legend title
    # leg.get_title().set_fontweight('bold')

    # for t in leg.get_texts():
    #     t.set_fontweight('bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22) # Tightening space

    # 4. EXPORT
    plt.savefig(f'{OUTPUT_DIR}fig3_boxplot_final.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}fig3_boxplot_final.pdf', bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 4: Tau vs MAE Scatter
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))
for algo in algo_order:
    tau_mean = tau_stats.loc[algo, 'mean']
    tau_ci = tau_stats.loc[algo, 'ci_95']
    mae_mean = mae_stats.loc[algo, 'mean']
    mae_ci = mae_stats.loc[algo, 'ci_95']
    
    # Plot Error Bars
    ax.errorbar(tau_mean, mae_mean, xerr=tau_ci, yerr=mae_ci,
                fmt='o', markersize=12, color=COLORS.get(algo, '#888888'),
                capsize=5, capthick=1.5, elinewidth=1.5, label=algo,
                markeredgecolor='black', markeredgewidth=0.5)

    # Add Text Annotation with 95% CI info
    # annotation = (f"{algo}\n"
    #               f"τ: {tau_mean:.2f}±{tau_ci:.2f}\n"
    #               f"MAE: {mae_mean:.2f}±{mae_ci:.2f}")
    annotation = (f"{algo}\n")
    
    ax.annotate(annotation, (tau_mean, mae_mean),
                xytext=(12, -5), textcoords='offset points',
                fontsize=8, va='center', ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

ax.set_xlabel("Norm. Tau Dist. (↓ better)", fontweight='bold', fontsize=12)
ax.set_ylabel('MAE (↓ better)', fontweight='bold', fontsize=12)
ax.set_title('Algorithm Performance Comparison\n(with 95% CI Annotations)', 
             fontweight='bold', fontsize=13)

# Random Guessing Line (Gray)
ax.axvline(x=0.5, color=RANDOM_COLOR, linestyle=':', linewidth=1.5, alpha=0.8)
ax.text(0.51, ax.get_ylim()[1]*0.9, 'Random Guessing', color=RANDOM_COLOR, 
        fontsize=9, ha='left', va='top', rotation=90, fontweight='bold')

# Reference lines
ax.axvline(x=tau_stats['mean'].median(), color='gray', linestyle=':', alpha=0.3)
ax.axhline(y=mae_stats['mean'].median(), color='gray', linestyle=':', alpha=0.3)

ax.legend(loc='upper left', framealpha=0.95, title="Algorithms")

# UPDATE: Set x-axis max to 0.6
ax.set_xlim(0, 0.6)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig4_tau_vs_mae.png', dpi=300, facecolor='white')
plt.savefig(f'{OUTPUT_DIR}fig4_tau_vs_mae.pdf', facecolor='white')
plt.close()
print(f"Saved: {OUTPUT_DIR}fig4_tau_vs_mae.png/pdf")

# ============================================================
# Figure 5: 3x3 Grid - Normalized Tau Distance by Experiment
# ============================================================

# Compute per-experiment statistics
exp_stats = df.groupby(['E', 'algo']).agg({
    'kendalls_tau': ['mean', 'std', 'count'],
    'mae': ['mean', 'std', 'count']
}).reset_index()
exp_stats.columns = ['Experiment', 'Algorithm', 'tau_mean', 'tau_std', 'tau_n',
                     'mae_mean', 'mae_std', 'mae_n']
exp_stats['tau_ci'] = exp_stats['tau_std'] / np.sqrt(exp_stats['tau_n']) * \
                      stats.t.ppf(0.975, exp_stats['tau_n'] - 1)
exp_stats['mae_ci'] = exp_stats['mae_std'] / np.sqrt(exp_stats['mae_n']) * \
                      stats.t.ppf(0.975, exp_stats['mae_n'] - 1)

experiments = sorted(df['E'].unique())
exp_short_titles = {exp: shorten_exp_title(exp) for exp in experiments}

n_exp = len(experiments)
n_cols = 3
n_rows = (n_exp + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12))
axes = axes.flatten()

for idx in range(n_exp, n_rows * n_cols):
    axes[idx].set_visible(False)

for idx, exp in enumerate(experiments):
    ax = axes[idx]
    exp_data = exp_stats[exp_stats['Experiment'] == exp].copy()

    # Sort Ascending (Smaller is Better)
    exp_data = exp_data.sort_values('tau_mean', ascending=True)
    
    x_pos = np.arange(len(exp_data))
    bar_colors = [COLORS.get(algo, '#888888') for algo in exp_data['Algorithm']]
    
    ax.bar(x_pos, exp_data['tau_mean'], yerr=exp_data['tau_ci'],
           color=bar_colors, edgecolor='black', linewidth=0.5,
           capsize=3, error_kw={'linewidth': 1.0, 'ecolor': 'black'}, alpha=0.8)
    
    for i, (mean, ci) in enumerate(zip(exp_data['tau_mean'], exp_data['tau_ci'])):
        ax.text(i, mean + ci + 0.02, f'{mean:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_data['Algorithm'], rotation=30, ha='right', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, TAU_YLIM)
    
    ax.set_ylabel("Norm. Tau Dist.", fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.set_title(exp_short_titles[exp], fontsize=12, fontweight='bold', pad=10)

# CREATE LEGEND SORTED BY TAU (EXPLICITLY)
# Ensure sorting by mean for legend
legend_algo_order = tau_stats.sort_values('mean', ascending=True).index.tolist()
legend_elements = []
for algo in legend_algo_order: 
    row = tau_stats.loc[algo]
    label = f"{algo} ({row['mean']:.3f} ± {row['ci_95']:.3f})"
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor=COLORS.get(algo, '#888888'),
                                     markersize=12, label=label,
                                     markeredgecolor='black', markeredgewidth=0.5, alpha=0.9))

fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
           fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5, -0.02),
           title="Algorithm (Sorted by Overall Tau Dist. ↑)", title_fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12, hspace=0.45, wspace=0.25)
plt.savefig(f'{OUTPUT_DIR}fig5_grid_tau_by_experiment.png', dpi=300, facecolor='white', 
            bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}fig5_grid_tau_by_experiment.pdf', facecolor='white',
            bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}fig5_grid_tau_by_experiment.png/pdf")

# ============================================================
# Figure 6: 3x3 Grid - MAE by Experiment
# ============================================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12))
axes = axes.flatten()

for idx in range(n_exp, n_rows * n_cols):
    axes[idx].set_visible(False)

for idx, exp in enumerate(experiments):
    ax = axes[idx]
    exp_data = exp_stats[exp_stats['Experiment'] == exp].copy()
    exp_data = exp_data.sort_values('mae_mean', ascending=True)
    
    x_pos = np.arange(len(exp_data))
    bar_colors = [COLORS.get(algo, '#888888') for algo in exp_data['Algorithm']]
    
    ax.bar(x_pos, exp_data['mae_mean'], yerr=exp_data['mae_ci'],
           color=bar_colors, edgecolor='black', linewidth=0.5,
           capsize=3, error_kw={'linewidth': 1.0, 'ecolor': 'black'}, alpha=0.8)
    
    for i, (mean, ci) in enumerate(zip(exp_data['mae_mean'], exp_data['mae_ci'])):
        ax.text(i, mean + ci + 0.03, f'{mean:.2f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_data['Algorithm'], rotation=30, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel("MAE", fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.set_title(exp_short_titles[exp], fontsize=12, fontweight='bold', pad=10)

# CREATE LEGEND SORTED BY MAE (EXPLICITLY)
# Ensure sorting by mean for legend
legend_mae_order = mae_stats.sort_values('mean', ascending=True).index.tolist()
legend_elements_mae = []
for algo in legend_mae_order:
    row = mae_stats.loc[algo]
    label = f"{algo} ({row['mean']:.3f} ± {row['ci_95']:.3f})"
    legend_elements_mae.append(plt.Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor=COLORS.get(algo, '#888888'),
                                     markersize=12, label=label,
                                     markeredgecolor='black', markeredgewidth=0.5, alpha=0.9))

fig.legend(handles=legend_elements_mae, loc='lower center', ncol=3, 
           fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5, -0.02),
           title="Algorithm (Sorted by Overall MAE ↑)", title_fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12, hspace=0.45, wspace=0.25)
plt.savefig(f'{OUTPUT_DIR}fig6_grid_mae_by_experiment.png', dpi=300, facecolor='white',
            bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}fig6_grid_mae_by_experiment.pdf', facecolor='white',
            bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}fig6_grid_mae_by_experiment.png/pdf")

# ============================================================
# Save Summary Statistics
# ============================================================

summary_table = pd.DataFrame({
    'Algorithm': algo_order,
    'Tau_Mean': [tau_stats.loc[a, 'mean'] for a in algo_order],
    'Tau_SD': [tau_stats.loc[a, 'std'] for a in algo_order],
    'Tau_95CI_Lower': [tau_stats.loc[a, 'ci_lower'] for a in algo_order],
    'Tau_95CI_Upper': [tau_stats.loc[a, 'ci_upper'] for a in algo_order],
    'MAE_Mean': [mae_stats.loc[a, 'mean'] for a in algo_order],
    'MAE_SD': [mae_stats.loc[a, 'std'] for a in algo_order],
    'MAE_95CI_Lower': [mae_stats.loc[a, 'ci_lower'] for a in algo_order],
    'MAE_95CI_Upper': [mae_stats.loc[a, 'ci_upper'] for a in algo_order],
    'N_Observations': [int(tau_stats.loc[a, 'n']) for a in algo_order]
})
summary_table.to_csv(f'{OUTPUT_DIR}summary_statistics.csv', index=False)

exp_summary = exp_stats.pivot_table(
    index='Experiment', columns='Algorithm', 
    values=['tau_mean', 'tau_ci', 'mae_mean', 'mae_ci']
)
exp_summary.to_csv(f'{OUTPUT_DIR}experiment_statistics.csv')

print(f"\nSaved: {OUTPUT_DIR}summary_statistics.csv")
print(f"Saved: {OUTPUT_DIR}experiment_statistics.csv")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(summary_table.round(4).to_string(index=False))

# ============================================================
# Statistical Tests
# ============================================================

print("\n" + "="*60)
print("PAIRWISE COMPARISONS (vs best performer)")
print("Best is lowest distance/error")
print("="*60)

best_algo = algo_order[0]
best_tau = df[df['algo'] == best_algo]['kendalls_tau']
best_mae = df[df['algo'] == best_algo]['mae']

print(f"Best Performer (Tau Dist): {best_algo}")

for algo in algo_order[1:]:
    algo_tau = df[df['algo'] == algo]['kendalls_tau']
    algo_mae = df[df['algo'] == algo]['mae']
    
    stat_tau, p_tau = stats.mannwhitneyu(best_tau, algo_tau, alternative='two-sided')
    stat_mae, p_mae = stats.mannwhitneyu(best_mae, algo_mae, alternative='two-sided')
    
    sig_tau = '***' if p_tau < 0.001 else '**' if p_tau < 0.01 else '*' if p_tau < 0.05 else ''
    sig_mae = '***' if p_mae < 0.001 else '**' if p_mae < 0.01 else '*' if p_mae < 0.05 else ''
    
    print(f"\n{algo} vs {best_algo}:")
    print(f"  Tau Dist - U={stat_tau:.1f}, p={p_tau:.2e} {sig_tau}")
    print(f"  MAE      - U={stat_mae:.1f}, p={p_mae:.2e} {sig_mae}")

print("\n" + "="*60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*60)
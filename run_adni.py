"""
run_adni.py - Apply trained models to ADNI data and generate visualizations

Usage:
    python run_adni.py
    python run_adni.py --config config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import glob
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import stats as scipy_stats
import warnings

warnings.filterwarnings("ignore")

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# ==========================================
# 1. Configuration
# ==========================================
def load_config(config_path="config.yaml"):
    config = {
        "adni_csv": "./adni.csv",
        "save_model_dir": "./models",
        "adni_results_dir": "./adni_results",
        "id_dx_json": "./id_dx.json",
        "n_samples": 256,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dropout": 0.2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    if os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for k, v in yaml_config.items():
                    if v is not None:
                        config[k] = v
    return config

CONFIG = load_config()

# ==========================================
# 2. Model Definitions
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class SimpleTransformer(nn.Module):
    def __init__(self, n_biomarkers=10, max_stage=10, config=None):
        super().__init__()
        if config is None:
            config = CONFIG

        d = config.get('d_model', 128)
        nhead = config.get('nhead', 8)
        num_layers = config.get('num_layers', 4)
        dropout = config.get('dropout', 0.2)

        self.n_biomarkers = n_biomarkers
        self.max_stage = max_stage
        self.d_model = d
        self.architecture_type = "simple"

        self.patient_encoder = nn.Sequential(
            nn.Linear(2, d), nn.LayerNorm(d), nn.ReLU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU()
        )

        self.pos_encoding = PositionalEncoding(d, max_len=n_biomarkers + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.biomarker_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ranking_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d // 2, 1)
        )

        self.stage_encoder = nn.Sequential(
            nn.Linear(n_biomarkers + 1, d * 2), nn.LayerNorm(d * 2), nn.ReLU(), nn.Dropout(dropout)
        )

        stage_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d * 2, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.stage_transformer = nn.TransformerEncoder(stage_encoder_layer, num_layers=2)

        self.stage_head = nn.Sequential(
            nn.Linear(d * 2, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward_ranking(self, x):
        batch_size, n_samples, n_features = x.shape
        n_bio = self.n_biomarkers
        biomarkers = x[:, :, :n_bio]
        diseased = x[:, :, -1:]

        bio_features = []
        for i in range(n_bio):
            bio_vals = biomarkers[:, :, i:i+1]
            combined = torch.cat([bio_vals, diseased], dim=-1)
            encoded = self.patient_encoder(combined)
            pooled = torch.mean(encoded, dim=1)
            bio_features.append(pooled)

        bio_features = torch.stack(bio_features, dim=1)
        bio_features = self.pos_encoding(bio_features)
        bio_features = self.biomarker_transformer(bio_features)
        scores = self.ranking_head(bio_features).squeeze(-1)
        return scores

    def forward_stage(self, x):
        encoded = self.stage_encoder(x)
        encoded = self.stage_transformer(encoded)
        out = self.stage_head(encoded).squeeze(-1)
        return out

    def forward(self, x):
        rank_scores = self.forward_ranking(x)
        stage_pred = self.forward_stage(x)
        return rank_scores, stage_pred


class ConnectedTransformer(nn.Module):
    def __init__(self, n_biomarkers=10, max_stage=10, config=None):
        super().__init__()
        if config is None:
            config = CONFIG

        d = config.get('d_model', 128)
        nhead = config.get('nhead', 8)
        num_layers = config.get('num_layers', 4)
        dropout = config.get('dropout', 0.2)

        self.n_biomarkers = n_biomarkers
        self.max_stage = max_stage
        self.d_model = d
        self.architecture_type = "connected"

        self.patient_encoder = nn.Sequential(
            nn.Linear(2, d), nn.LayerNorm(d), nn.ReLU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU()
        )

        self.pos_encoding = PositionalEncoding(d, max_len=n_biomarkers + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.biomarker_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ranking_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d // 2, 1)
        )

        self.abnormality_detector = nn.Sequential(
            nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1), nn.Sigmoid()
        )

        self.stage_refiner = nn.Sequential(
            nn.Linear(n_biomarkers + 1, d), nn.ReLU(), nn.Linear(d, 1)
        )

    def forward_ranking(self, x):
        batch_size, n_samples, n_features = x.shape
        n_bio = self.n_biomarkers
        biomarkers = x[:, :, :n_bio]
        diseased = x[:, :, -1:]

        bio_features = []
        for i in range(n_bio):
            bio_vals = biomarkers[:, :, i:i+1]
            combined = torch.cat([bio_vals, diseased], dim=-1)
            encoded = self.patient_encoder(combined)
            pooled = torch.mean(encoded, dim=1)
            bio_features.append(pooled)

        bio_features = torch.stack(bio_features, dim=1)
        bio_features = self.pos_encoding(bio_features)
        bio_features = self.biomarker_transformer(bio_features)
        scores = self.ranking_head(bio_features).squeeze(-1)
        return scores

    def forward_stage(self, x, biomarker_scores):
        batch_size, n_samples, n_features = x.shape
        n_bio = self.n_biomarkers
        biomarkers = x[:, :, :n_bio]
        diseased = x[:, :, -1]
        scores_normalized = torch.sigmoid(biomarker_scores)

        abnormality_probs = []
        for i in range(n_bio):
            bio_val = biomarkers[:, :, i]
            bio_score = scores_normalized[:, i:i+1].expand(-1, n_samples)
            detector_input = torch.stack([bio_val, diseased, bio_score], dim=-1)
            prob = self.abnormality_detector(detector_input.view(-1, 3)).view(batch_size, n_samples)
            abnormality_probs.append(prob)

        abnormality_probs = torch.stack(abnormality_probs, dim=-1)
        base_stage = torch.sum(abnormality_probs, dim=-1)
        refine_input = torch.cat([abnormality_probs, base_stage.unsqueeze(-1)], dim=-1)
        correction = self.stage_refiner(refine_input.view(-1, n_bio + 1)).view(batch_size, n_samples)
        stage_pred = base_stage + correction
        return stage_pred

    def forward(self, x):
        rank_scores = self.forward_ranking(x)
        stage_pred = self.forward_stage(x, rank_scores)
        return rank_scores, stage_pred


class UnifiedTransformer(nn.Module):
    """Unified architecture combining abnormality detector with patient attention."""
    def __init__(self, n_biomarkers=10, max_stage=10, config=None):
        super().__init__()
        if config is None:
            config = CONFIG

        d = config.get('d_model', 128)
        nhead = config.get('nhead', 8)
        num_layers = config.get('num_layers', 4)
        dropout = config.get('dropout', 0.2)

        self.n_biomarkers = n_biomarkers
        self.max_stage = max_stage
        self.d_model = d
        self.architecture_type = "unified"

        self.patient_encoder = nn.Sequential(
            nn.Linear(2, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU()
        )

        self.pos_encoding = PositionalEncoding(d, max_len=n_biomarkers + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.biomarker_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ranking_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d // 2, 1)
        )

        self.abnormality_detector = nn.Sequential(
            nn.Linear(3, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, 1), nn.Sigmoid()
        )

        self.stage_encoder = nn.Sequential(
            nn.Linear(n_biomarkers, d * 2), nn.LayerNorm(d * 2), nn.ReLU(), nn.Dropout(dropout)
        )

        stage_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d * 2, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.stage_transformer = nn.TransformerEncoder(stage_encoder_layer, num_layers=2)

        self.stage_head = nn.Sequential(
            nn.Linear(d * 2, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward_ranking(self, x):
        batch_size, n_samples, n_features = x.shape
        n_bio = self.n_biomarkers
        biomarkers = x[:, :, :n_bio]
        diseased = x[:, :, -1:]

        bio_features = []
        for i in range(n_bio):
            bio_vals = biomarkers[:, :, i:i+1]
            combined = torch.cat([bio_vals, diseased], dim=-1)
            encoded = self.patient_encoder(combined)
            pooled = torch.mean(encoded, dim=1)
            bio_features.append(pooled)

        bio_features = torch.stack(bio_features, dim=1)
        bio_features = self.pos_encoding(bio_features)
        bio_features = self.biomarker_transformer(bio_features)
        scores = self.ranking_head(bio_features).squeeze(-1)
        return scores

    def forward_stage(self, x, biomarker_scores):
        batch_size, n_samples, n_features = x.shape
        n_bio = self.n_biomarkers
        biomarkers = x[:, :, :n_bio]
        diseased = x[:, :, -1]

        scores_normalized = torch.sigmoid(biomarker_scores)

        abnormality_probs = []
        for i in range(n_bio):
            bio_val = biomarkers[:, :, i]
            bio_score = scores_normalized[:, i:i+1].expand(-1, n_samples)
            detector_input = torch.stack([bio_val, diseased, bio_score], dim=-1)
            prob_flat = self.abnormality_detector(detector_input.view(-1, 3))
            prob = prob_flat.view(batch_size, n_samples)
            abnormality_probs.append(prob)

        abnormality_probs = torch.stack(abnormality_probs, dim=-1)
        encoded = self.stage_encoder(abnormality_probs)
        encoded = self.stage_transformer(encoded)
        out = self.stage_head(encoded).squeeze(-1)
        return out

    def forward(self, x):
        rank_scores = self.forward_ranking(x)
        stage_pred = self.forward_stage(x, rank_scores)
        return rank_scores, stage_pred


def create_model(n_biomarkers, max_stage, architecture_type, config=None):
    if architecture_type == "simple":
        return SimpleTransformer(n_biomarkers, max_stage, config)
    elif architecture_type == "unified":
        return UnifiedTransformer(n_biomarkers, max_stage, config)
    else:
        return ConnectedTransformer(n_biomarkers, max_stage, config)


# ==========================================
# 3. Data Loading
# ==========================================
class GlobalStandardizer:
    def __init__(self):
        self.stats = {}

    def load_from_dict(self, stats_dict):
        self.stats = stats_dict

    def transform(self, values, bio_name):
        if bio_name in self.stats:
            s = self.stats[bio_name]
            return (values - s['mean']) / s['std']
        return values


def load_adni_data(csv_path, standardizer):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Strip biomarker names too
    df['biomarker'] = df['biomarker'].str.strip()
    
    d_map = df.groupby('participant')['diseased'].first().astype(str)
    d_map = d_map.str.lower().isin(['true', '1', 'yes', 't']).astype(float)
    
    df['measurement'] = pd.to_numeric(df['measurement'], errors='coerce')
    pivot_df = df.pivot(index='participant', columns='biomarker', values='measurement')
    
    # Debug: print biomarker names
    adni_biomarkers = set(pivot_df.columns)
    model_biomarkers = set(standardizer.stats.keys())
    
    valid_cols = [c for c in pivot_df.columns if c in standardizer.stats]
    
    if len(valid_cols) == 0:
        print(f"    WARNING: No matching biomarkers!")
        print(f"    ADNI biomarkers: {sorted(adni_biomarkers)}")
        print(f"    Model biomarkers: {sorted(model_biomarkers)}")
        print(f"    Intersection: {adni_biomarkers & model_biomarkers}")
        raise ValueError("No matching biomarkers between ADNI and model")
    
    data_list = []
    for c in valid_cols:
        vals = pivot_df[c].values
        vals = np.nan_to_num(vals, nan=standardizer.stats[c]['mean'])
        norm_vals = standardizer.transform(vals, c)
        data_list.append(norm_vals)
    
    data_matrix = np.stack(data_list, axis=1)
    labels = d_map.loc[pivot_df.index].values
    participant_ids = pivot_df.index.tolist()
    
    # print(f"    Matched {len(valid_cols)}/{len(adni_biomarkers)} biomarkers: {valid_cols}")
    
    return (torch.tensor(data_matrix, dtype=torch.float32), 
            torch.tensor(labels, dtype=torch.float32), 
            valid_cols, 
            participant_ids)


# ==========================================
# 4. Inference
# ==========================================
def run_inference(model, data_matrix, labels, device):
    model.eval()
    full_input = torch.cat([data_matrix, labels.unsqueeze(1)], dim=1).unsqueeze(0)
    
    with torch.no_grad():
        rank_scores, stage_pred = model(full_input.to(device))
        rank_scores = rank_scores.cpu().numpy().flatten()
        stage_pred = stage_pred.cpu().numpy().flatten()
    
    return rank_scores, stage_pred


def scores_to_ordinal_order(scores, biomarker_names):
    order = np.argsort(scores)
    ordinal_order = {biomarker_names[i]: rank + 1 for rank, i in enumerate(order)}
    return ordinal_order


def scores_to_continuous_order(scores, biomarker_names):
    return {biomarker_names[i]: float(scores[i]) for i in range(len(scores))}


def compute_dx_avg_stages(stage_pred, participant_ids, id_dx):
    """Compute average predicted stage for each diagnosis group."""
    dx_stages = {}
    for pid, stage in zip(participant_ids, stage_pred):
        dx = id_dx.get(str(pid))
        if dx is not None:
            dx_stages.setdefault(dx, []).append(float(stage))
    return {dx: float(np.mean(stages)) for dx, stages in sorted(dx_stages.items())}


# ==========================================
# 5. Visualization Functions
# ==========================================

DIAGNOSIS_COLORS = {
    'CN': '#27ae60',
    'EMCI': '#3498db',
    'LMCI': '#f39c12',
    'AD': '#c0392b',
    'SMC': '#9b59b6'
}

def plot_continuous_ranks_timeline(continuous_order, exp_name, output_dir):
    """Plot biomarkers on a 1D timeline - raw scores normalized to 0-1 scale"""
    biomarkers = list(continuous_order.keys())
    scores = np.array(list(continuous_order.values()))
    
    # Normalize scores to 0-1 range (min becomes 0, max becomes 1)
    score_min, score_max = scores.min(), scores.max()
    scores_norm = (scores - score_min) / (score_max - score_min + 1e-8)
    
    # Sort by score for better visualization
    sorted_indices = np.argsort(scores)
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    
    # Draw the main timeline axis
    ax.axhline(y=0, color='#333', linewidth=2, zorder=1)
    
    # Add tick marks at 0 and 1
    ax.plot([0, 0], [-0.02, 0.02], color='#333', linewidth=2)
    ax.plot([1, 1], [-0.02, 0.02], color='#333', linewidth=2)
    
    # Add intermediate ticks
    for tick in [0.25, 0.5, 0.75]:
        ax.plot([tick, tick], [-0.015, 0.015], color='#666', linewidth=1)
        ax.text(tick, -0.08, f'{tick:.2f}', ha='center', va='top', fontsize=8, color='#666')
    
    # Axis labels
    ax.text(0, -0.08, '0', ha='center', va='top', fontsize=9, fontweight='bold')
    ax.text(1, -0.08, '1', ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Color gradient based on position
    colors = plt.cm.coolwarm(scores_norm)
    
    # Plot each biomarker
    for idx, i in enumerate(sorted_indices):
        bio = biomarkers[i]
        score = scores_norm[i]
        
        # Alternate y positions to avoid overlap
        y_offset = 0.1 if idx % 2 == 0 else -0.1
        
        # Draw point on timeline
        ax.scatter(score, 0, color=colors[i], s=120, zorder=3, 
                   edgecolor='white', linewidth=1.5)
        
        # Draw connector line
        ax.plot([score, score], [0, y_offset * 0.5], color='#888', 
                linewidth=0.7, zorder=2)
        
        # Label
        rotation = 35 if y_offset > 0 else -35
        va = 'bottom' if y_offset > 0 else 'top'
        ax.text(score, y_offset, bio, ha='center', va=va,
                fontsize=8, rotation=rotation, fontweight='bold')
    
    # Axis labels
    ax.text(0, -0.22, 'Early\n(Disease Onset)', ha='center', va='top', 
            fontsize=9, fontweight='bold', color='#333')
    ax.text(1, -0.22, 'Late\n(Advanced Disease)', ha='center', va='top', 
            fontsize=9, fontweight='bold', color='#333')
    
    # Set strict limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.35, 0.25)
    ax.set_title(f'{exp_name}: Biomarker Event Timeline (Normalized 0-1)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{exp_name}_continuous_timeline.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{exp_name}_continuous_timeline.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


# def plot_continuous_ranks_timeline(
#     continuous_order,
#     exp_name,
#     output_dir,
#     *,
#     assume_scores_are_0_1: bool = True,
#     rescale_mode: str = "auto",  # "auto" | "none" | "minmax"
#     title_suffix: str | None = None,
# ):
#     """
#     Plot biomarkers on a 1D timeline.

#     IMPORTANT SEMANTICS
#     -------------------
#     The x-axis is a *relative latent event position*, not guaranteed to be "disease time".
#     If your scores already live in [0, 1], you should usually NOT min-max rescale them,
#     because that would forcibly map the earliest biomarker to 0 and latest to 1 (a stronger claim).

#     Parameters
#     ----------
#     continuous_order : dict[str, float]
#         Mapping biomarker -> continuous event score.
#     exp_name : str
#         Used in plot title + output filename.
#     output_dir : str
#         Directory to save figure.
#     assume_scores_are_0_1 : bool
#         If True, treat scores as meaningful on [0, 1] when they fall within (approximately) that range.
#     rescale_mode : {"auto","none","minmax"}
#         - "none": never rescale; only clip to [0,1] if assume_scores_are_0_1 is True.
#         - "minmax": always min-max normalize to [0,1].
#         - "auto": if assume_scores_are_0_1 and values are within [0,1] (with slack), do "none";
#                   otherwise do "minmax".
#     title_suffix : str | None
#         Optional extra title text.
#     """
#     if not isinstance(continuous_order, dict) or len(continuous_order) == 0:
#         raise ValueError("continuous_order must be a non-empty dict of biomarker -> score.")

#     os.makedirs(output_dir, exist_ok=True)

#     biomarkers = list(continuous_order.keys())
#     scores_raw = np.array([continuous_order[b] for b in biomarkers], dtype=float)

#     # Decide how to transform scores for plotting
#     slack = 1e-6
#     in_0_1 = (scores_raw.min() >= -slack) and (scores_raw.max() <= 1.0 + slack)

#     mode = rescale_mode.lower().strip()
#     if mode not in {"auto", "none", "minmax"}:
#         raise ValueError('rescale_mode must be one of: "auto", "none", "minmax".')

#     if mode == "none":
#         scores_plot = scores_raw.copy()
#         if assume_scores_are_0_1:
#             scores_plot = np.clip(scores_plot, 0.0, 1.0)
#         transform_note = "no rescaling"
#     elif mode == "minmax":
#         smin, smax = scores_raw.min(), scores_raw.max()
#         scores_plot = (scores_raw - smin) / (smax - smin + 1e-8)
#         transform_note = "min–max rescaled"
#     else:  # auto
#         if assume_scores_are_0_1 and in_0_1:
#             scores_plot = np.clip(scores_raw, 0.0, 1.0)
#             transform_note = "no rescaling"
#         else:
#             smin, smax = scores_raw.min(), scores_raw.max()
#             scores_plot = (scores_raw - smin) / (smax - smin + 1e-8)
#             transform_note = "min–max rescaled (auto)"

#     # Sort by the plotted positions
#     sorted_indices = np.argsort(scores_plot)

#     fig, ax = plt.subplots(figsize=(12, 3.8))

#     # Main axis line
#     ax.axhline(y=0, color="#333", linewidth=2, zorder=1)

#     # End ticks
#     ax.plot([0, 0], [-0.02, 0.02], color="#333", linewidth=2)
#     ax.plot([1, 1], [-0.02, 0.02], color="#333", linewidth=2)

#     # Intermediate ticks + labels
#     for tick in [0.25, 0.5, 0.75]:
#         ax.plot([tick, tick], [-0.015, 0.015], color="#666", linewidth=1)
#         ax.text(tick, -0.085, f"{tick:.2f}", ha="center", va="top", fontsize=8, color="#666")

#     ax.text(0, -0.085, "0", ha="center", va="top", fontsize=9, fontweight="bold", color="#333")
#     ax.text(1, -0.085, "1", ha="center", va="top", fontsize=9, fontweight="bold", color="#333")

#     # Colors by position on the plotted axis (0..1)
#     colors = plt.cm.coolwarm(scores_plot)

#     # Plot points/labels
#     for idx, i in enumerate(sorted_indices):
#         bio = biomarkers[i]
#         x = float(scores_plot[i])

#         # Alternate label heights
#         y_offset = 0.13 if idx % 2 == 0 else -0.13

#         ax.scatter(
#             x, 0,
#             color=colors[i],
#             s=120,
#             zorder=3,
#             edgecolor="white",
#             linewidth=1.5,
#         )

#         ax.plot([x, x], [0, y_offset * 0.55], color="#888", linewidth=0.7, zorder=2)

#         rotation = 35 if y_offset > 0 else -35
#         va = "bottom" if y_offset > 0 else "top"
#         ax.text(x, y_offset, bio, ha="center", va=va, fontsize=8, rotation=rotation)

#     # Semantically-correct axis annotations
#     ax.text(
#         0, -0.24,
#         "Earlier relative\nbiomarker change",
#         ha="center", va="top",
#         fontsize=9, fontweight="bold", color="#333",
#     )
#     ax.text(
#         1, -0.24,
#         "Later relative\nbiomarker change",
#         ha="center", va="top",
#         fontsize=9, fontweight="bold", color="#333",
#     )

#     # Title
#     extra = f" — {title_suffix}" if title_suffix else ""
#     ax.set_title(
#         f"{exp_name}: Biomarker Event Timeline ({transform_note}){extra}",
#         fontsize=12, fontweight="bold", pad=10,
#     )

#     # Limits + cosmetics
#     ax.set_xlim(-0.05, 1.05)
#     ax.set_ylim(-0.37, 0.27)
#     ax.axis("off")

#     plt.tight_layout()

#     outpath = os.path.join(output_dir, f"{exp_name}_continuous_timeline.pdf")
#     plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     plt.close()

#     # return outpath


def plot_stage_density_professional(stages, participant_ids, id_dx, exp_name, output_dir, is_continuous=True):
    """Plot professional KDE density curves"""
    diagnoses, valid_stages = [], []
    
    for pid, stage in zip(participant_ids, stages):
        pid_str = str(pid)
        if pid_str in id_dx:
            diagnoses.append(id_dx[pid_str])
            valid_stages.append(stage)
    
    if len(valid_stages) == 0:
        return
    
    df = pd.DataFrame({'stage': valid_stages, 'diagnosis': diagnoses})
    
    dx_order = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD']
    present_dx = [dx for dx in dx_order if dx in df['diagnosis'].unique()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_min = df['stage'].min() - 0.5
    x_max = df['stage'].max() + 0.5
    x_range = np.linspace(x_min, x_max, 300)
    
    for dx in present_dx:
        subset = df[df['diagnosis'] == dx]['stage']
        color = DIAGNOSIS_COLORS.get(dx, '#333')
        
        if len(subset) > 2:
            try:
                kde = gaussian_kde(subset, bw_method='scott')
                density = kde(x_range)
                ax.fill_between(x_range, density, alpha=0.25, color=color)
                ax.plot(x_range, density, color=color, linewidth=2.5, label=f'{dx} (n={len(subset)})')
            except:
                pass
    
    ax.set_xlabel('Disease Stage' + (' (Continuous)' if is_continuous else ' (Ordinal)'), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{exp_name}: Stage Distribution by Diagnosis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    suffix = 'continuous' if is_continuous else 'ordinal'
    plt.savefig(os.path.join(output_dir, f'{exp_name}_stage_density_{suffix}.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{exp_name}_stage_density_{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ordinal_stage_stacked_bar(stages, participant_ids, id_dx, n_bio, exp_name, output_dir):
    """Plot stacked bar chart"""
    ordinal_stages = np.clip(np.round(stages), 0, n_bio).astype(int)
    
    stage_dx_counts = {}
    for pid, stage in zip(participant_ids, ordinal_stages):
        pid_str = str(pid)
        if pid_str in id_dx:
            dx = id_dx[pid_str]
            if stage not in stage_dx_counts:
                stage_dx_counts[stage] = {}
            if dx not in stage_dx_counts[stage]:
                stage_dx_counts[stage][dx] = 0
            stage_dx_counts[stage][dx] += 1
    
    if len(stage_dx_counts) == 0:
        return
    
    dx_order = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD']
    all_dx = set()
    for counts in stage_dx_counts.values():
        all_dx.update(counts.keys())
    all_dx = [dx for dx in dx_order if dx in all_dx]
    
    stages_list = list(range(int(min(stage_dx_counts.keys())), int(max(stage_dx_counts.keys())) + 1))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8
    bottom = np.zeros(len(stages_list))
    
    for dx in all_dx:
        counts = [stage_dx_counts.get(s, {}).get(dx, 0) for s in stages_list]
        color = DIAGNOSIS_COLORS.get(dx, '#333')
        ax.bar(stages_list, counts, bar_width, bottom=bottom, label=dx, 
               color=color, edgecolor='white', linewidth=0.5)
        bottom += np.array(counts)
    
    ax.set_xlabel('Disease Stage (Ordinal)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title(f'{exp_name}: Diagnosis Distribution by Stage', fontsize=14, fontweight='bold')
    ax.set_xticks(stages_list)
    ax.legend(title='Diagnosis', loc='upper left', framealpha=0.9)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{exp_name}_ordinal_stage_stacked.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{exp_name}_ordinal_stage_stacked.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 6. Summary Report
# ==========================================
def generate_summary_report(all_results, output_dir):
    """Generate comprehensive summary"""
    
    report_path = os.path.join(output_dir, 'final_results.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ADNI BIOMARKER ORDERING RESULTS - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        if len(all_results) == 0:
            f.write("No results to report.\n")
            return report_path
        
        first_exp = list(all_results.keys())[0]
        all_biomarkers = list(all_results[first_exp]['ordinal_order'].keys())
        
        # Section 1: Arrow format
        f.write("-" * 80 + "\n")
        f.write("1. BIOMARKER EVENT ORDERING (Early → Late)\n")
        f.write("-" * 80 + "\n\n")
        
        for exp_name in sorted(all_results.keys()):
            order = all_results[exp_name]['ordinal_order']
            sorted_biomarkers = sorted(order.keys(), key=lambda x: order[x])
            order_str = " → ".join(sorted_biomarkers)
            f.write(f"{exp_name}:\n  {order_str}\n\n")
        
        # Section 2: Position Matrix with 95% CI
        f.write("-" * 80 + "\n")
        f.write("2. BIOMARKER POSITION MATRIX (Rank 1 = Earliest)\n")
        f.write("-" * 80 + "\n\n")
        
        # Collect ranks
        biomarker_ranks = {bio: [] for bio in all_biomarkers}
        for exp_name in sorted(all_results.keys()):
            order = all_results[exp_name]['ordinal_order']
            for bio, rank in order.items():
                biomarker_ranks[bio].append(rank)
        
        # Calculate statistics
        n_exps = len(all_results)
        
        # Header
        header = f"{'Biomarker':<18}"
        for exp in sorted(all_results.keys()):
            header += f"{exp:>6}"
        header += f"{'Mean':>7}{'Std':>6}{'95% CI':>14}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Sort biomarkers by mean rank
        sorted_biomarkers = sorted(all_biomarkers, key=lambda x: np.mean(biomarker_ranks[x]))
        
        for bio in sorted_biomarkers:
            ranks = np.array(biomarker_ranks[bio])
            mean_rank = np.mean(ranks)
            std_rank = np.std(ranks, ddof=1) if len(ranks) > 1 else 0
            
            # 95% CI using t-distribution
            if len(ranks) > 1:
                se = std_rank / np.sqrt(len(ranks))
                t_critical = scipy_stats.t.ppf(0.975, df=len(ranks)-1)
                ci_low = mean_rank - t_critical * se
                ci_high = mean_rank + t_critical * se
                ci_str = f"[{ci_low:.1f}, {ci_high:.1f}]"
            else:
                ci_str = "N/A"
            
            row = f"{bio:<18}"
            for r in ranks:
                row += f"{r:>6}"
            row += f"{mean_rank:>7.1f}{std_rank:>6.1f}{ci_str:>14}"
            f.write(row + "\n")
        
        # Section 3: Consensus
        f.write("\n" + "-" * 80 + "\n")
        f.write("3. CONSENSUS ORDERING (by mean rank)\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"  {' → '.join(sorted_biomarkers)}\n\n")
        
        # Section 4: Continuous
        continuous_exps = ['exp8', 'exp9']
        has_continuous = any(exp in all_results and 'continuous_order' in all_results[exp]
                            for exp in continuous_exps)

        if has_continuous:
            f.write("-" * 80 + "\n")
            f.write("4. CONTINUOUS EVENT TIMES (exp8, exp9)\n")
            f.write("-" * 80 + "\n\n")
            
            for exp_name in continuous_exps:
                if exp_name in all_results and 'continuous_order' in all_results[exp_name]:
                    f.write(f"{exp_name}:\n")
                    cont_order = all_results[exp_name]['continuous_order']
                    sorted_items = sorted(cont_order.items(), key=lambda x: x[1])
                    for bio, score in sorted_items:
                        f.write(f"  {bio:<20} {score:.4f}\n")
                    f.write("\n")
        
        # Section 5: Interpretation
        f.write("-" * 80 + "\n")
        f.write("5. BIOLOGICAL INTERPRETATION\n")
        f.write("-" * 80 + "\n\n")
        
        n_bio = len(sorted_biomarkers)
        n_early = min(4, n_bio // 3)
        n_late = min(4, n_bio // 3)
        n_mid = n_bio - n_early - n_late
        
        f.write(f"EARLY EVENTS (ranks 1-{n_early}):\n")
        f.write(f"  {', '.join(sorted_biomarkers[:n_early])}\n")
        f.write("  → Temporal lobe atrophy and amyloid accumulation\n\n")
        
        f.write(f"MIDDLE EVENTS (ranks {n_early+1}-{n_early+n_mid}):\n")
        f.write(f"  {', '.join(sorted_biomarkers[n_early:n_early+n_mid])}\n")
        f.write("  → Cognitive decline and hippocampal atrophy\n\n")
        
        f.write(f"LATE EVENTS (ranks {n_early+n_mid+1}-{n_bio}):\n")
        f.write(f"  {', '.join(sorted_biomarkers[n_early+n_mid:])}\n")
        f.write("  → Tau pathology and global brain atrophy\n\n")
        
        # Section 6: Average stage by diagnosis group per experiment
        has_dx = any('dx_avg_stages' in all_results[e] for e in all_results)
        if has_dx:
            f.write("-" * 80 + "\n")
            f.write("6. AVERAGE PREDICTED STAGE BY DIAGNOSIS GROUP\n")
            f.write("-" * 80 + "\n\n")

            # Collect all dx labels in clinical severity order
            dx_order = ['CN', 'EMCI', 'LMCI', 'AD']
            seen_dx = {dx for e in all_results.values()
                       for dx in e.get('dx_avg_stages', {}).keys()}
            all_dx = [dx for dx in dx_order if dx in seen_dx]
            all_dx += sorted(seen_dx - set(dx_order))

            # Per-experiment rows
            col_w = 10
            header = f"{'Exp':<8}" + "".join(f"{dx:>{col_w}}" for dx in all_dx)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Accumulate for cross-experiment average
            dx_across = {dx: [] for dx in all_dx}
            for exp_name in sorted(all_results.keys()):
                dx_avg = all_results[exp_name].get('dx_avg_stages', {})
                row = f"{exp_name:<8}"
                for dx in all_dx:
                    if dx in dx_avg:
                        row += f"{dx_avg[dx]:>{col_w}.2f}"
                        dx_across[dx].append(dx_avg[dx])
                    else:
                        row += f"{'N/A':>{col_w}}"
                f.write(row + "\n")

            # Average-of-averages row
            f.write("-" * len(header) + "\n")
            avg_row = f"{'Mean':<8}"
            for dx in all_dx:
                if dx_across[dx]:
                    avg_row += f"{np.mean(dx_across[dx]):>{col_w}.2f}"
                else:
                    avg_row += f"{'N/A':>{col_w}}"
            f.write(avg_row + "\n\n")

        f.write("=" * 80 + "\n")

    print(f"Summary saved to: {report_path}")
    return report_path


# ==========================================
# 7. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--models_dir', type=str, default=None,
                        help='Path to models directory (default: ./models, ignores save_model_dir in config)')
    args = parser.parse_args()

    config = load_config(args.config)

    adni_csv = config.get('adni_csv', './adni.csv')
    # Default to ./models (lowdim, named biomarkers) rather than config's save_model_dir,
    # since ADNI uses named biomarkers matching the lowdim synthetic training data.
    model_dir = args.models_dir if args.models_dir is not None else './models'
    results_dir = config.get('adni_results_dir', './adni_results')
    id_dx_path = config.get('id_dx_json', './id_dx.json')
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("ADNI Inference")
    print("=" * 60)
    
    # Load diagnosis labels
    id_dx = {}
    if os.path.exists(id_dx_path):
        with open(id_dx_path, 'r') as f:
            id_dx = json.load(f)
        print(f"Loaded {len(id_dx)} diagnosis labels")
    
    os.makedirs(results_dir, exist_ok=True)
    
    model_files = sorted(glob.glob(os.path.join(model_dir, "exp*_final_model.pth")))
    if not model_files:
        print(f"No models found in {model_dir}")
        return
    
    print(f"Found {len(model_files)} models\n")
    
    continuous_stage_exps = ['exp5', 'exp6', 'exp7', 'exp8', 'exp9']
    continuous_rank_exps = ['exp8', 'exp9']
    
    all_results = {}
    
    for model_path in model_files:
        exp_name = os.path.basename(model_path).replace('_final_model.pth', '')
        print(f"Processing {exp_name}...")

        try:
            exp_output_dir = os.path.join(results_dir, exp_name)
            os.makedirs(exp_output_dir, exist_ok=True)

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            n_bio = checkpoint['n_biomarkers']
            max_stage = checkpoint['max_stage']
            saved_config = checkpoint.get('config', config)
            architecture_type = checkpoint.get('architecture_type', 'simple')
            print(f"  architecture: {architecture_type}, n_bio: {n_bio}")

            model = create_model(n_bio, max_stage, architecture_type, config=saved_config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            standardizer = GlobalStandardizer()
            standardizer.load_from_dict(checkpoint['standardizer_stats'])

            data_matrix, labels, biomarker_names, participant_ids = load_adni_data(adni_csv, standardizer)
        except Exception as e:
            import traceback
            print(f"  ERROR loading/preparing {exp_name}: {e}")
            traceback.print_exc()
            continue
        
        rank_scores, stage_pred = run_inference(model, data_matrix, labels, device)
        
        ordinal_order = scores_to_ordinal_order(rank_scores, biomarker_names)
        
        exp_results = {
            'ordinal_order': ordinal_order,
            'n_biomarkers': n_bio,
            'architecture': architecture_type
        }
        
        results = {
            "experiment": exp_name,
            "n_biomarkers": n_bio,
            "architecture": architecture_type,
            "true_order": ordinal_order
        }
        
        if exp_name in continuous_rank_exps:
            continuous_order = scores_to_continuous_order(rank_scores, biomarker_names)
            results["true_order_continuous"] = continuous_order
            exp_results['continuous_order'] = continuous_order
            plot_continuous_ranks_timeline(continuous_order, exp_name, exp_output_dir)

        if len(id_dx) > 0:
            dx_avg = compute_dx_avg_stages(stage_pred, participant_ids, id_dx)
            results["dx_avg_stages"] = dx_avg
            exp_results['dx_avg_stages'] = dx_avg

        json_path = os.path.join(exp_output_dir, f'{exp_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if len(id_dx) > 0:
            is_continuous = exp_name in continuous_stage_exps
            plot_stage_density_professional(stage_pred, participant_ids, id_dx, exp_name, 
                                           exp_output_dir, is_continuous=is_continuous)
            plot_ordinal_stage_stacked_bar(stage_pred, participant_ids, id_dx, n_bio, exp_name, exp_output_dir)
        
        stage_results = {
            "participant_ids": [str(p) for p in participant_ids],
            "predicted_stages": stage_pred.tolist(),
            "predicted_stages_ordinal": np.clip(np.round(stage_pred), 0, n_bio).astype(int).tolist()
        }
        stage_path = os.path.join(exp_output_dir, f'{exp_name}_stages.json')
        with open(stage_path, 'w') as f:
            json.dump(stage_results, f, indent=2)
        
        all_results[exp_name] = exp_results
    
    print("\nGenerating summary report...")
    generate_summary_report(all_results, results_dir)
    
    print(f"\nDone! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
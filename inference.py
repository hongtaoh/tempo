"""
inference.py - Cross-Experiment Evaluation

For each trained model (trained on exp_i), evaluate on ALL experiments.
Produces a cross-experiment results matrix.

Usage:
    python inference.py
    python inference.py --models_dir models --test_dir test --output results/cross_experiment_results.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import glob
from scipy.stats import kendalltau, sem
import yaml
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 1. Configuration
# ==========================================
def load_config(config_path="config.yaml"):
    config = {
        "train_data_dir": "./train",
        "test_data_dir": "./test",
        "save_model_dir": "./models",
        "n_samples": 256,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dropout": 0.2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for k, v in yaml_config.items():
                    if v is not None:
                        config[k] = v
    return config

CONFIG = load_config()
models_dir = CONFIG['save_model_dir']


# ==========================================
# 2. Data Loading
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


def robust_load_file(csv_path, standardizer):
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if 'diseased' in df.columns:
            d_map = df.groupby('participant')['diseased'].first().astype(str)
            d_map = d_map.str.lower().isin(['true', '1', 'yes', 't']).astype(float)
        else:
            return None, None, None

        df['measurement'] = pd.to_numeric(df['measurement'], errors='coerce')
        pivot_df = df.pivot(index='participant', columns='biomarker', values='measurement')

        valid_cols = [c for c in pivot_df.columns if c in standardizer.stats]
        if len(valid_cols) < 2:
            return None, None, None

        data_list = []
        for c in valid_cols:
            vals = pivot_df[c].values
            vals = np.nan_to_num(vals, nan=standardizer.stats[c]['mean'])
            norm_vals = standardizer.transform(vals, c)
            data_list.append(norm_vals)

        data_matrix = np.stack(data_list, axis=1)
        labels = d_map.loc[pivot_df.index].values
        return torch.tensor(data_matrix, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), valid_cols
    except:
        return None, None, None


def get_ground_truth_data(json_data, csv_filename, biomarker_names):
    key = os.path.basename(csv_filename).replace('.csv', '')
    if key not in json_data:
        return None, None, None

    entry = json_data[key]
    order_dict = entry['true_order']
    stages_list = entry.get('true_stages', [])
    order_continuous = entry.get('true_order_continuous', {})

    ranks = [order_dict.get(name, 100) for name in biomarker_names]
    
    # Get ground truth event times: use continuous if available, otherwise use ranks
    if order_continuous:
        event_times = np.array([order_continuous.get(name, float(ranks[i])) for i, name in enumerate(biomarker_names)], dtype=np.float32)
    else:
        event_times = np.array(ranks, dtype=np.float32)
    
    return ranks, stages_list, event_times


# ==========================================
# 3. Model Definitions
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


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
    return UnifiedTransformer(n_biomarkers, max_stage, config)


# ==========================================
# 4. Evaluation Functions
# ==========================================
def compute_ci_95(values):
    """Compute 95% confidence interval using t-distribution."""
    if len(values) < 2:
        return None, None
    mean = np.mean(values)
    se = sem(values)
    ci = 1.96 * se  # Approximate 95% CI
    return mean - ci, mean + ci


def transform_tau(tau):
    """Transform Kendall's tau to (1 - tau) / 2 for consistency with benchmarking papers."""
    if tau is None:
        return None
    return (1 - tau) / 2


def evaluate_single_file(model, csv_path, gt_data, standardizer, n_bio, device):
    mat, lbl, names = robust_load_file(csv_path, standardizer)
    if mat is None:
        return None, None, None, 0

    result = get_ground_truth_data(gt_data, csv_path, names)
    if result is None or result[0] is None:
        return None, None, None, 0

    ranks, stages_ordinal, gt_event_times = result
    if len(stages_ordinal) == 0:
        return None, None, None, 0

    stages_ordinal = np.array(stages_ordinal, dtype=np.float32)
    n_p = mat.shape[0]

    if len(stages_ordinal) != n_p:
        if len(stages_ordinal) > n_p:
            stages_ordinal = stages_ordinal[:n_p]
        else:
            stages_ordinal = np.pad(stages_ordinal, (0, n_p - len(stages_ordinal)), constant_values=0)

    # Use all patients during inference (no subsampling)
    sub_mat = mat
    sub_lbl = lbl
    sub_stages = stages_ordinal
    full_input = torch.cat([sub_mat, sub_lbl.unsqueeze(1)], dim=1).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        scores, stage_pred = model(full_input.to(device))
        scores = scores.cpu().numpy().flatten()
        stage_pred = stage_pred.cpu().numpy().flatten()

    predicted_order = np.argsort(scores)
    true_ranks_for_predicted = [ranks[i] for i in predicted_order]
    tau_raw, _ = kendalltau(np.arange(len(ranks)) + 1, true_ranks_for_predicted)

    stage_pred_rounded = np.clip(np.round(stage_pred), 0, n_bio)
    mae = np.mean(np.abs(stage_pred_rounded - sub_stages))

    # Compute sequence MAE: compare predicted event times vs ground truth event times
    # Scale predicted scores to the same range as ground truth event times
    gt_min, gt_max = gt_event_times.min(), gt_event_times.max()
    
    # Normalize scores to [0, 1] first, then scale to ground truth range
    scores_min, scores_max = scores.min(), scores.max()
    if scores_max - scores_min > 1e-8:
        scores_normalized = (scores - scores_min) / (scores_max - scores_min)
    else:
        scores_normalized = np.zeros_like(scores)
    
    # Project to actual event time scale
    pred_event_times = scores_normalized * (gt_max - gt_min) + gt_min
    
    # Sequence MAE in original time units
    sequence_mae = np.mean(np.abs(pred_event_times - gt_event_times))

    return tau_raw, mae, sequence_mae, n_p


def evaluate_experiment(model, exp_folder, standardizer, n_bio, device, max_files=None):
    json_path = os.path.join(exp_folder, "true_order_and_stages.json")
    if not os.path.exists(json_path):
        print(f"  Warning: {json_path} not found")
        return {}, None, None, None, None, None, None

    with open(json_path, 'r') as f:
        gt_data = json.load(f)

    files = sorted(glob.glob(os.path.join(exp_folder, "*.csv")))
    if max_files and len(files) > max_files:
        files = files[:max_files]

    results = {}
    taus_transformed = []
    maes = []
    seq_maes = []

    for f in files:
        tau_raw, mae, sequence_mae, n_patients = evaluate_single_file(model, f, gt_data, standardizer, n_bio, device)
        filename = os.path.basename(f).replace('.csv', '')

        if tau_raw is not None and not np.isnan(tau_raw):
            tau_transformed = transform_tau(tau_raw)
            results[filename] = {
                'tau': float(tau_transformed), 
                'tau_raw': float(tau_raw), 
                'mae': float(mae) if mae else None,
                'sequence_mae': float(sequence_mae) if sequence_mae is not None else None
            }
            taus_transformed.append(tau_transformed)
            if mae is not None:
                maes.append(mae)
            if sequence_mae is not None:
                seq_maes.append(sequence_mae)

    avg_tau = np.mean(taus_transformed) if taus_transformed else None
    avg_mae = np.mean(maes) if maes else None
    avg_seq_mae = np.mean(seq_maes) if seq_maes else None
    tau_ci = compute_ci_95(taus_transformed) if len(taus_transformed) >= 2 else (None, None)
    mae_ci = compute_ci_95(maes) if len(maes) >= 2 else (None, None)
    seq_mae_ci = compute_ci_95(seq_maes) if len(seq_maes) >= 2 else (None, None)

    return results, avg_tau, avg_mae, avg_seq_mae, (taus_transformed, tau_ci), (maes, mae_ci), (seq_maes, seq_mae_ci)


def format_cell_with_ci(mean, ci_low, ci_high, decimals=3):
    """Format a cell value with 95% CI."""
    if mean is None:
        return "N/A"
    if ci_low is None or ci_high is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def save_summary_tables(all_results, test_exps, output_path):
    """Save summary tables to both TXT and CSV files."""
    lines = []
    
    # ==========================================
    # Build data matrices for all metrics
    # ==========================================
    tau_matrix = {}
    tau_raw_values = {}
    seq_mae_matrix = {}
    seq_mae_raw_values = {}
    mae_matrix = {}
    mae_raw_values = {}
    
    for model_name in sorted(all_results.keys()):
        tau_matrix[model_name] = {}
        tau_raw_values[model_name] = {}
        seq_mae_matrix[model_name] = {}
        seq_mae_raw_values[model_name] = {}
        mae_matrix[model_name] = {}
        mae_raw_values[model_name] = {}
        
        for test_exp in test_exps:
            file_results = all_results[model_name]['tested_on'].get(test_exp, {})
            if file_results:
                taus = [v['tau'] for v in file_results.values() if v.get('tau') is not None]
                seq_maes = [v['sequence_mae'] for v in file_results.values() if v.get('sequence_mae') is not None]
                maes = [v['mae'] for v in file_results.values() if v.get('mae') is not None]
                
                tau_matrix[model_name][test_exp] = np.mean(taus) if taus else None
                tau_raw_values[model_name][test_exp] = taus
                seq_mae_matrix[model_name][test_exp] = np.mean(seq_maes) if seq_maes else None
                seq_mae_raw_values[model_name][test_exp] = seq_maes
                mae_matrix[model_name][test_exp] = np.mean(maes) if maes else None
                mae_raw_values[model_name][test_exp] = maes
            else:
                tau_matrix[model_name][test_exp] = None
                tau_raw_values[model_name][test_exp] = []
                seq_mae_matrix[model_name][test_exp] = None
                seq_mae_raw_values[model_name][test_exp] = []
                mae_matrix[model_name][test_exp] = None
                mae_raw_values[model_name][test_exp] = []
    
    # ==========================================
    # CSV Output: Detailed results with ± values
    # ==========================================
    csv_rows = []
    
    for model_name in sorted(all_results.keys()):
        arch = all_results[model_name].get('architecture', 'unknown')
        
        for test_exp in test_exps:
            file_results = all_results[model_name]['tested_on'].get(test_exp, {})
            n_files = len(file_results) if file_results else 0
            
            # Helper to compute mean and ± (half CI width)
            def get_mean_pm(vals):
                if not vals:
                    return None, None
                mean = np.mean(vals)
                pm = 1.96 * sem(vals) if len(vals) >= 2 else None
                return mean, pm
            
            tau_mean, tau_pm = get_mean_pm(tau_raw_values[model_name][test_exp])
            seq_mae_mean, seq_mae_pm = get_mean_pm(seq_mae_raw_values[model_name][test_exp])
            stage_mae_mean, stage_mae_pm = get_mean_pm(mae_raw_values[model_name][test_exp])
            
            csv_rows.append({
                'model': model_name,
                'architecture': arch,
                'test_exp': test_exp,
                'n_files': n_files,
                'tau_mean': tau_mean,
                'tau_pm': tau_pm,
                'seq_mae_mean': seq_mae_mean,
                'seq_mae_pm': seq_mae_pm,
                'stage_mae_mean': stage_mae_mean,
                'stage_mae_pm': stage_mae_pm,
            })
    
    # Save detailed CSV
    df = pd.DataFrame(csv_rows)
    csv_output = output_path.replace('.json', '_summary.csv')
    df.to_csv(csv_output, index=False)
    
    # Save pivot tables for each metric
    pivot_dir = os.path.dirname(csv_output) or '.'
    
    # Tau pivot
    tau_pivot = df.pivot(index='model', columns='test_exp', values='tau_mean')
    tau_pivot['row_mean'] = tau_pivot.mean(axis=1)
    tau_pivot.loc['col_mean'] = tau_pivot.mean(axis=0)
    tau_pivot.to_csv(os.path.join(pivot_dir, 'tau_pivot.csv'))
    
    tau_pm_pivot = df.pivot(index='model', columns='test_exp', values='tau_pm')
    tau_pm_pivot.to_csv(os.path.join(pivot_dir, 'tau_pm_pivot.csv'))
    
    # Sequence MAE pivot
    seq_mae_pivot = df.pivot(index='model', columns='test_exp', values='seq_mae_mean')
    seq_mae_pivot['row_mean'] = seq_mae_pivot.mean(axis=1)
    seq_mae_pivot.loc['col_mean'] = seq_mae_pivot.mean(axis=0)
    seq_mae_pivot.to_csv(os.path.join(pivot_dir, 'seq_mae_pivot.csv'))
    
    seq_mae_pm_pivot = df.pivot(index='model', columns='test_exp', values='seq_mae_pm')
    seq_mae_pm_pivot.to_csv(os.path.join(pivot_dir, 'seq_mae_pm_pivot.csv'))
    
    # Stage MAE pivot
    stage_mae_pivot = df.pivot(index='model', columns='test_exp', values='stage_mae_mean')
    stage_mae_pivot['row_mean'] = stage_mae_pivot.mean(axis=1)
    stage_mae_pivot.loc['col_mean'] = stage_mae_pivot.mean(axis=0)
    stage_mae_pivot.to_csv(os.path.join(pivot_dir, 'stage_mae_pivot.csv'))
    
    stage_mae_pm_pivot = df.pivot(index='model', columns='test_exp', values='stage_mae_pm')
    stage_mae_pm_pivot.to_csv(os.path.join(pivot_dir, 'stage_mae_pm_pivot.csv'))
    
    # ==========================================
    # TXT Output: Human-readable tables
    # ==========================================
    col_width = 12
    ci_col_width = 24
    
    # Compute column means for TXT tables
    col_means_tau = {exp: np.mean([tau_matrix[m][exp] for m in tau_matrix if tau_matrix[m][exp] is not None]) 
                     if any(tau_matrix[m][exp] is not None for m in tau_matrix) else None for exp in test_exps}
    col_means_seq_mae = {exp: np.mean([seq_mae_matrix[m][exp] for m in seq_mae_matrix if seq_mae_matrix[m][exp] is not None])
                         if any(seq_mae_matrix[m][exp] is not None for m in seq_mae_matrix) else None for exp in test_exps}
    col_means_mae = {exp: np.mean([mae_matrix[m][exp] for m in mae_matrix if mae_matrix[m][exp] is not None])
                     if any(mae_matrix[m][exp] is not None for m in mae_matrix) else None for exp in test_exps}
    
    # --- Kendall Tau Table ---
    lines.append("=" * 120)
    lines.append("SUMMARY: Average Kendall Tau Distance (1-τ)/2 (with 95% CI)")
    lines.append("Note: Lower values indicate better performance")
    lines.append("=" * 120)
    
    header = f"{'Trained|Tested':>14} |"
    for exp in test_exps:
        header += f" {exp:>{col_width}} |"
    header += f" {'Row Mean':>{col_width}} |"
    lines.append(header)
    lines.append("-" * len(header))
    
    for model_name in sorted(all_results.keys()):
        arch = all_results[model_name].get('architecture', '?')[:1].upper()
        row = f"{model_name:>12}({arch})|"
        row_vals = []
        for test_exp in test_exps:
            val = tau_matrix[model_name][test_exp]
            if val is not None:
                row += f" {val:>{col_width}.3f} |"
                row_vals.append(val)
            else:
                row += f" {'N/A':>{col_width}} |"
        row_mean = np.mean(row_vals) if row_vals else None
        row += f" {row_mean:>{col_width}.3f} |" if row_mean is not None else f" {'N/A':>{col_width}} |"
        lines.append(row)
    
    lines.append("-" * len(header))
    col_mean_row = f"{'Col Mean':>14} |"
    all_col_means = []
    for test_exp in test_exps:
        val = col_means_tau[test_exp]
        if val is not None:
            col_mean_row += f" {val:>{col_width}.3f} |"
            all_col_means.append(val)
        else:
            col_mean_row += f" {'N/A':>{col_width}} |"
    overall_mean = np.mean(all_col_means) if all_col_means else None
    col_mean_row += f" {overall_mean:>{col_width}.3f} |" if overall_mean is not None else f" {'N/A':>{col_width}} |"
    lines.append(col_mean_row)
    
    # Tau CI Table
    lines.append("")
    lines.append("=" * 120)
    lines.append("Kendall Tau Distance (1-τ)/2 - 95% Confidence Intervals")
    lines.append("=" * 120)
    
    ci_header = f"{'Trained|Tested':>14} |"
    for exp in test_exps:
        ci_header += f" {exp:^{ci_col_width}} |"
    lines.append(ci_header)
    lines.append("-" * len(ci_header))
    
    for model_name in sorted(all_results.keys()):
        arch = all_results[model_name].get('architecture', '?')[:1].upper()
        row = f"{model_name:>12}({arch})|"
        for test_exp in test_exps:
            raw_vals = tau_raw_values[model_name][test_exp]
            if raw_vals and len(raw_vals) >= 2:
                mean = np.mean(raw_vals)
                ci_low, ci_high = compute_ci_95(raw_vals)
                cell = f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
            elif raw_vals:
                cell = f"{np.mean(raw_vals):.3f}"
            else:
                cell = "N/A"
            row += f" {cell:^{ci_col_width}} |"
        lines.append(row)
    
    # --- Sequence MAE Table ---
    lines.append("")
    lines.append("=" * 120)
    lines.append("SUMMARY: Average Sequence MAE (with 95% CI)")
    lines.append("Note: MAE between predicted event times and ground truth event times (in original time units)")
    lines.append("=" * 120)
    
    header = f"{'Trained|Tested':>14} |"
    for exp in test_exps:
        header += f" {exp:>{col_width}} |"
    header += f" {'Row Mean':>{col_width}} |"
    lines.append(header)
    lines.append("-" * len(header))
    
    for model_name in sorted(all_results.keys()):
        arch = all_results[model_name].get('architecture', '?')[:1].upper()
        row = f"{model_name:>12}({arch})|"
        row_vals = []
        for test_exp in test_exps:
            val = seq_mae_matrix[model_name][test_exp]
            if val is not None:
                row += f" {val:>{col_width}.3f} |"
                row_vals.append(val)
            else:
                row += f" {'N/A':>{col_width}} |"
        row_mean = np.mean(row_vals) if row_vals else None
        row += f" {row_mean:>{col_width}.3f} |" if row_mean is not None else f" {'N/A':>{col_width}} |"
        lines.append(row)
    
    lines.append("-" * len(header))
    col_mean_row = f"{'Col Mean':>14} |"
    all_col_means = []
    for test_exp in test_exps:
        val = col_means_seq_mae[test_exp]
        if val is not None:
            col_mean_row += f" {val:>{col_width}.3f} |"
            all_col_means.append(val)
        else:
            col_mean_row += f" {'N/A':>{col_width}} |"
    overall_mean = np.mean(all_col_means) if all_col_means else None
    col_mean_row += f" {overall_mean:>{col_width}.3f} |" if overall_mean is not None else f" {'N/A':>{col_width}} |"
    lines.append(col_mean_row)
    
    # Sequence MAE CI Table
    lines.append("")
    lines.append("=" * 120)
    lines.append("Sequence MAE 95% Confidence Intervals")
    lines.append("=" * 120)
    
    ci_header = f"{'Trained|Tested':>14} |"
    for exp in test_exps:
        ci_header += f" {exp:^{ci_col_width}} |"
    lines.append(ci_header)
    lines.append("-" * len(ci_header))
    
    for model_name in sorted(all_results.keys()):
        arch = all_results[model_name].get('architecture', '?')[:1].upper()
        row = f"{model_name:>12}({arch})|"
        for test_exp in test_exps:
            raw_vals = seq_mae_raw_values[model_name][test_exp]
            if raw_vals and len(raw_vals) >= 2:
                mean = np.mean(raw_vals)
                ci_low, ci_high = compute_ci_95(raw_vals)
                cell = f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
            elif raw_vals:
                cell = f"{np.mean(raw_vals):.3f}"
            else:
                cell = "N/A"
            row += f" {cell:^{ci_col_width}} |"
        lines.append(row)
    
    # --- Stage MAE Table ---
    lines.append("")
    lines.append("=" * 120)
    lines.append("SUMMARY: Average Stage MAE (with 95% CI)")
    lines.append("=" * 120)
    
    header = f"{'Trained|Tested':>14} |"
    for exp in test_exps:
        header += f" {exp:>{col_width}} |"
    header += f" {'Row Mean':>{col_width}} |"
    lines.append(header)
    lines.append("-" * len(header))
    
    for model_name in sorted(all_results.keys()):
        arch = all_results[model_name].get('architecture', '?')[:1].upper()
        row = f"{model_name:>12}({arch})|"
        row_vals = []
        for test_exp in test_exps:
            val = mae_matrix[model_name][test_exp]
            if val is not None:
                row += f" {val:>{col_width}.3f} |"
                row_vals.append(val)
            else:
                row += f" {'N/A':>{col_width}} |"
        row_mean = np.mean(row_vals) if row_vals else None
        row += f" {row_mean:>{col_width}.3f} |" if row_mean is not None else f" {'N/A':>{col_width}} |"
        lines.append(row)

    lines.append("-" * len(header))
    col_mean_row = f"{'Col Mean':>14} |"
    all_col_means = []
    for test_exp in test_exps:
        val = col_means_mae[test_exp]
        if val is not None:
            col_mean_row += f" {val:>{col_width}.3f} |"
            all_col_means.append(val)
        else:
            col_mean_row += f" {'N/A':>{col_width}} |"
    overall_mean = np.mean(all_col_means) if all_col_means else None
    col_mean_row += f" {overall_mean:>{col_width}.3f} |" if overall_mean is not None else f" {'N/A':>{col_width}} |"
    lines.append(col_mean_row)
    
    # Stage MAE CI Table
    lines.append("")
    lines.append("=" * 120)
    lines.append("Stage MAE 95% Confidence Intervals")
    lines.append("=" * 120)
    
    ci_header = f"{'Trained|Tested':>14} |"
    for exp in test_exps:
        ci_header += f" {exp:^{ci_col_width}} |"
    lines.append(ci_header)
    lines.append("-" * len(ci_header))
    
    for model_name in sorted(all_results.keys()):
        arch = all_results[model_name].get('architecture', '?')[:1].upper()
        row = f"{model_name:>12}({arch})|"
        for test_exp in test_exps:
            raw_vals = mae_raw_values[model_name][test_exp]
            if raw_vals and len(raw_vals) >= 2:
                mean = np.mean(raw_vals)
                ci_low, ci_high = compute_ci_95(raw_vals)
                cell = f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
            elif raw_vals:
                cell = f"{np.mean(raw_vals):.3f}"
            else:
                cell = "N/A"
            row += f" {cell:^{ci_col_width}} |"
        lines.append(row)
    
    # Notes
    lines.append("")
    lines.append("=" * 120)
    lines.append("Notes:")
    lines.append("  (U) = UnifiedTransformer")
    lines.append("  Kendall Tau Distance = (1 - τ) / 2, where τ is Kendall's tau correlation")
    lines.append("  Lower tau distance indicates better ranking performance (0 = perfect, 1 = worst)")
    lines.append("  Sequence MAE = Mean absolute error between predicted event times and ground truth event times")
    lines.append("    - Predicted scores are scaled to match the ground truth event time range [min, max]")
    lines.append("    - For exp1-4: ground truth = discrete ranks (1, 2, 3, ...)")
    lines.append("    - For exp5-9: ground truth = continuous event times from true_order_continuous")
    lines.append("    - Values are in original time units (not normalized)")
    lines.append("  Stage MAE = Mean absolute error of predicted patient stages vs ground truth stages")
    lines.append("  95% CI computed using 1.96 * standard error")
    lines.append("  Row Mean = average across all test experiments for a given model")
    lines.append("  Col Mean = average across all models for a given test experiment")
    lines.append("")
    lines.append("CSV Files Generated:")
    lines.append(f"  - {csv_output} (detailed results with mean ± values)")
    lines.append(f"  - tau_pivot.csv, tau_pm_pivot.csv")
    lines.append(f"  - seq_mae_pivot.csv, seq_mae_pm_pivot.csv")
    lines.append(f"  - stage_mae_pivot.csv, stage_mae_pm_pivot.csv")
    lines.append("=" * 120)
    
    # Write TXT file
    txt_output = output_path.replace('.json', '_summary.txt')
    with open(txt_output, 'w') as f:
        f.write('\n'.join(lines))
    
    return txt_output, csv_output, '\n'.join(lines)


# ==========================================
# 5. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Cross-Experiment Evaluation')
    parser.add_argument('--models_dir', type=str, default=models_dir, help='Directory containing trained models')
    parser.add_argument('--test_dir', type=str, default=None, help='Test data directory')
    parser.add_argument('--output', type=str, default='results/cross_experiment_results.json', help='Output JSON file')
    parser.add_argument('--max_files', type=int, default=None, help='Max files per experiment')
    args = parser.parse_args()

    test_dir = args.test_dir or CONFIG.get('test_data_dir', './test')
    device = CONFIG['device']

    np.random.seed(CONFIG.get('INFER_SEED', 0))

    print("=" * 60)
    print("Cross-Experiment Evaluation")
    print("=" * 60)
    print(f"Models directory: {args.models_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Device: {device}")
    print()

    model_files = sorted(glob.glob(os.path.join(args.models_dir, "*_final_model.pth")))
    if not model_files:
        print(f"No models found in {args.models_dir}")
        return

    print(f"Found {len(model_files)} trained models:")
    for mf in model_files:
        print(f"  - {os.path.basename(mf)}")
    print()

    test_exps = sorted([d for d in os.listdir(test_dir)
                        if os.path.isdir(os.path.join(test_dir, d)) and d.startswith('exp')])
    print(f"Found {len(test_exps)} test experiments: {test_exps}")
    print()

    all_results = {}

    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('_final_model.pth', '')
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        n_bio = checkpoint['n_biomarkers']
        max_stage = checkpoint['max_stage']
        saved_config = checkpoint.get('config', CONFIG)
        architecture_type = checkpoint.get('architecture_type', 'unified')

        print(f"  n_biomarkers: {n_bio}")
        print(f"  architecture: {architecture_type}")

        model = create_model(n_bio, max_stage, architecture_type, config=saved_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        standardizer = GlobalStandardizer()
        standardizer.load_from_dict(checkpoint['standardizer_stats'])

        model_results = {'tested_on': {}, 'architecture': architecture_type}

        for test_exp in test_exps:
            exp_folder = os.path.join(test_dir, test_exp)
            is_same_exp = (test_exp == model_name)

            print(f"\n  Testing on {test_exp}" + (" (ID)" if is_same_exp else " (OOD)") + "...")

            file_results, avg_tau, avg_mae, avg_seq_mae, tau_stats, mae_stats, seq_mae_stats = evaluate_experiment(
                model, exp_folder, standardizer, n_bio, device,
                max_files=args.max_files
            )

            model_results['tested_on'][test_exp] = file_results

            tau_str = f"τ_dist={avg_tau:.4f}" if avg_tau else "τ_dist=N/A"
            mae_str = f", MAE={avg_mae:.2f}" if avg_mae else ""
            seq_mae_str = f", SeqMAE={avg_seq_mae:.3f}" if avg_seq_mae else ""
            
            # Add CI info if available
            if tau_stats and tau_stats[1][0] is not None:
                tau_str += f" [{tau_stats[1][0]:.3f}, {tau_stats[1][1]:.3f}]"
            
            print(f"    {test_exp}: {tau_str}{seq_mae_str}{mae_str} ({len(file_results)} files)")

        all_results[model_name] = model_results

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.output}")

    # Save summary tables to txt and csv
    txt_path, csv_path, summary_text = save_summary_tables(all_results, test_exps, args.output)
    print(f"Saved summary tables to {txt_path}")
    print(f"Saved CSV summary to {csv_path}")

    # Print summary to console
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
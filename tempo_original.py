import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
import glob
from scipy.stats import kendalltau
import yaml
import argparse
import copy
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Configuration
# ==========================================
def load_config(config_path="config.yaml"):
    config = {
        "train_data_dir": "./train",
        "test_data_dir": "./test",
        "n_samples": 256,
        "batch_size": 128,
        "lr": 5e-4,
        "epochs": 100,
        "patience": 20,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dropout": 0.2,
        "samples_per_file": 16,
        "lambda_ranking": 1.0,
        "lambda_stage": 0.5,
        "MAX_TRAIN_FILES": 1000,
        "VAL_FILES": 25,
        "ADAPTIVE_THRESHOLD": 25,
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
        try:
            config['batch_size'] = int(config['batch_size'])
            config['epochs'] = int(config['epochs'])
            config['n_samples'] = int(config['n_samples'])
            config['lr'] = float(config['lr'])
        except ValueError as e:
            print(f"Error parsing config types: {e}")
    else:
        print(f"Warning: {config_path} not found. Using defaults.")

    return config

CONFIG = load_config()

# ==========================================
# 2. Data Loading
# ==========================================
class GlobalStandardizer:
    def __init__(self):
        self.stats = {}

    def fit(self, exp_folder):
        print(f"Fitting standardizer on {exp_folder}...")
        all_files = glob.glob(os.path.join(exp_folder, "*.csv"))
        sums, counts, sq_sums = {}, {}, {}

        for f in all_files:
            try:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip()
                df = df.dropna(subset=['measurement'])

                for bio, val in zip(df['biomarker'], df['measurement']):
                    val = float(val)
                    if bio not in sums:
                        sums[bio] = 0.0
                        counts[bio] = 0
                        sq_sums[bio] = 0.0
                    sums[bio] += val
                    counts[bio] += 1
                    sq_sums[bio] += val ** 2
            except:
                continue

        for bio in sums:
            n = counts[bio]
            if n < 2:
                continue
            mean = sums[bio] / n
            var = (sq_sums[bio] / n) - (mean ** 2)
            self.stats[bio] = {'mean': mean, 'std': np.sqrt(max(var, 1e-6))}

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
        return None, None, None, None, None

    entry = json_data[key]
    order_dict = entry['true_order']
    stages_list = entry.get('true_stages', [])
    order_continuous = entry.get('true_order_continuous', {})
    stages_continuous = entry.get('true_stages_continuous', [])

    ranks = [order_dict.get(name, 100) for name in biomarker_names]
    max_stage = len(biomarker_names)

    return ranks, stages_list, order_continuous, stages_continuous, max_stage


# ==========================================
# 3. Dataset
# ==========================================
class AllBiomarkersDataset(Dataset):
    def __init__(self, file_list, gt_data, standardizer):
        self.data = []
        self.samples_per_file = CONFIG['samples_per_file']

        print(f"Loading {len(file_list)} files...")
        for f in file_list:
            mat, lbl, names = robust_load_file(f, standardizer)
            if mat is None:
                continue

            result = get_ground_truth_data(gt_data, f, names)
            if result[0] is None:
                continue

            ranks, stages, order_cont, stages_cont, max_stage = result
            if len(stages) == 0:
                continue

            stages = np.array(stages, dtype=np.float32)
            if len(stages) != mat.shape[0]:
                if len(stages) > mat.shape[0]:
                    stages = stages[:mat.shape[0]]
                else:
                    stages = np.pad(stages, (0, mat.shape[0] - len(stages)), constant_values=0)

            if len(stages_cont) > 0:
                stages_cont = np.array(stages_cont, dtype=np.float32)
                if len(stages_cont) != mat.shape[0]:
                    if len(stages_cont) > mat.shape[0]:
                        stages_cont = stages_cont[:mat.shape[0]]
                    else:
                        stages_cont = np.pad(stages_cont, (0, mat.shape[0] - len(stages_cont)), constant_values=0)
            else:
                stages_cont = stages.copy()

            if len(order_cont) > 0:
                event_times_cont = np.array([order_cont.get(name, float(ranks[i])) for i, name in enumerate(names)], dtype=np.float32)
                use_continuous_order = True
            else:
                event_times_cont = np.array([float(r) for r in ranks], dtype=np.float32)
                use_continuous_order = False

            self.data.append({
                'mat': mat,
                'lbl': lbl,
                'ranks': np.array(ranks, dtype=np.float32),
                'stages': stages,
                'stages_cont': stages_cont,
                'event_times_cont': event_times_cont,
                'use_continuous_order': use_continuous_order,
                'n_bio': len(names),
                'max_stage': max_stage
            })

        print(f"Loaded {len(self.data)} valid datasets.")

    def __len__(self):
        return len(self.data) * self.samples_per_file

    def __getitem__(self, idx):
        item = self.data[idx // self.samples_per_file]
        mat = item['mat']
        lbl = item['lbl']
        ranks = item['ranks']
        stages = item['stages']
        stages_cont = item['stages_cont']
        event_times_cont = item['event_times_cont']
        use_continuous_order = item['use_continuous_order']
        n_bio = item['n_bio']
        max_stage = item['max_stage']

        n_p = mat.shape[0]
        p_idx = np.random.choice(n_p, CONFIG['n_samples'], replace=(n_p < CONFIG['n_samples']))
        sub_mat = mat[p_idx]
        sub_lbl = lbl[p_idx]
        sub_stages_cont = stages_cont[p_idx]
        sub_stages_ordinal = stages[p_idx]

        full_input = torch.cat([sub_mat, sub_lbl.unsqueeze(1)], dim=1)

        if use_continuous_order:
            rank_target = torch.tensor(event_times_cont / max_stage, dtype=torch.float32)
        else:
            rank_target = torch.tensor((ranks - 1) / (max_stage - 1 + 1e-8), dtype=torch.float32)

        stage_labels_cont = torch.tensor(sub_stages_cont, dtype=torch.float32)
        stage_labels_ordinal = torch.tensor(sub_stages_ordinal, dtype=torch.float32)

        for _ in range(20):
            idx_a, idx_b = np.random.choice(n_bio, 2, replace=False)
            if ranks[idx_a] != ranks[idx_b]:
                break

        pair_indices = torch.tensor([idx_a, idx_b], dtype=torch.long)

        if use_continuous_order:
            pair_label = (event_times_cont[idx_b] - event_times_cont[idx_a]) / max_stage
        else:
            pair_label = 1.0 if ranks[idx_a] < ranks[idx_b] else 0.0
        pair_label = torch.tensor([pair_label], dtype=torch.float32)

        is_continuous_order = torch.tensor([1.0 if use_continuous_order else 0.0])

        return (full_input, rank_target, stage_labels_cont, stage_labels_ordinal,
                pair_indices, pair_label, is_continuous_order)


# ==========================================
# 4. Model Architectures
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class SimpleTransformer(nn.Module):
    """
    Original architecture for small number of biomarkers (< 25).
    
    Key features:
    - Transformer over BIOMARKERS for ranking
    - Transformer over PATIENTS for staging (key contribution!)
    """
    def __init__(self, n_biomarkers=10, max_stage=10):
        super().__init__()
        d = CONFIG['d_model']
        nhead = CONFIG.get('nhead', 8)
        num_layers = CONFIG.get('num_layers', 4)
        dropout = CONFIG.get('dropout', 0.2)

        self.n_biomarkers = n_biomarkers
        self.max_stage = max_stage
        self.d_model = d
        self.architecture_type = "simple"

        # === RANKING BRANCH: Transformer over biomarkers ===
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
            d_model=d,
            nhead=nhead,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.biomarker_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ranking_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1)
        )

        # === STAGING BRANCH: Transformer over patients ===
        self.stage_encoder = nn.Sequential(
            nn.Linear(n_biomarkers + 1, d * 2),
            nn.LayerNorm(d * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        stage_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d * 2,
            nhead=nhead,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.stage_transformer = nn.TransformerEncoder(stage_encoder_layer, num_layers=2)

        self.stage_head = nn.Sequential(
            nn.Linear(d * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
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
        batch_size, n_samples, n_features = x.shape

        encoded = self.stage_encoder(x)
        encoded = self.stage_transformer(encoded)  # Patients attend to each other!
        out = self.stage_head(encoded).squeeze(-1)

        return out

    def forward(self, x):
        rank_scores = self.forward_ranking(x)
        stage_pred = self.forward_stage(x)
        return rank_scores, stage_pred


class ConnectedTransformer(nn.Module):
    """
    Connected Ranking-Staging architecture for large number of biomarkers (>= 25).
    
    Key features:
    - Transformer over BIOMARKERS for ranking
    - Abnormality detector for staging (uses ranking info, faster than patient Transformer)
    """
    def __init__(self, n_biomarkers=10, max_stage=10):
        super().__init__()
        d = CONFIG['d_model']
        nhead = CONFIG.get('nhead', 8)
        num_layers = CONFIG.get('num_layers', 4)
        dropout = CONFIG.get('dropout', 0.2)

        self.n_biomarkers = n_biomarkers
        self.max_stage = max_stage
        self.d_model = d
        self.architecture_type = "connected"

        # === RANKING BRANCH: Transformer over biomarkers ===
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
            d_model=d,
            nhead=nhead,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.biomarker_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ranking_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1)
        )

        # === STAGING BRANCH: Abnormality detector (connected to ranking) ===
        self.abnormality_detector = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
            nn.Sigmoid()
        )

        self.stage_refiner = nn.Sequential(
            nn.Linear(n_biomarkers + 1, d),
            nn.ReLU(),
            nn.Linear(d, 1)
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
            bio_score = scores_normalized[:, i:i+1]
            bio_score_expanded = bio_score.expand(-1, n_samples)

            detector_input = torch.stack([bio_val, diseased, bio_score_expanded], dim=-1)
            detector_input_flat = detector_input.view(-1, 3)
            prob_flat = self.abnormality_detector(detector_input_flat)
            prob = prob_flat.view(batch_size, n_samples)

            abnormality_probs.append(prob)

        abnormality_probs = torch.stack(abnormality_probs, dim=-1)
        base_stage = torch.sum(abnormality_probs, dim=-1)

        refine_input = torch.cat([abnormality_probs, base_stage.unsqueeze(-1)], dim=-1)
        refine_input_flat = refine_input.view(-1, n_bio + 1)
        correction_flat = self.stage_refiner(refine_input_flat)
        correction = correction_flat.view(batch_size, n_samples)

        stage_pred = base_stage + correction

        return stage_pred

    def forward(self, x):
        rank_scores = self.forward_ranking(x)
        stage_pred = self.forward_stage(x, rank_scores)
        return rank_scores, stage_pred


def create_model(n_biomarkers, max_stage):
    """Factory function to create appropriate model based on biomarker count."""
    threshold = CONFIG.get('ADAPTIVE_THRESHOLD', 25)

    if n_biomarkers < threshold:
        print(f"Using SimpleTransformer (n_bio={n_biomarkers} < {threshold})")
        return SimpleTransformer(n_biomarkers, max_stage)
    else:
        print(f"Using ConnectedTransformer (n_bio={n_biomarkers} >= {threshold})")
        return ConnectedTransformer(n_biomarkers, max_stage)


# ==========================================
# 5. Evaluation
# ==========================================
def evaluate_on_files(model, file_list, gt_data, standardizer, n_bio):
    taus = []
    stage_maes = []

    model.eval()

    for f in file_list:
        mat, lbl, names = robust_load_file(f, standardizer)
        if mat is None:
            continue

        result = get_ground_truth_data(gt_data, f, names)
        if result[0] is None:
            continue
        ranks = result[0]
        stages_ordinal = result[1]

        if len(stages_ordinal) == 0:
            continue

        stages_ordinal = np.array(stages_ordinal, dtype=np.float32)
        n_p = mat.shape[0]

        if len(stages_ordinal) != n_p:
            if len(stages_ordinal) > n_p:
                stages_ordinal = stages_ordinal[:n_p]
            else:
                stages_ordinal = np.pad(stages_ordinal, (0, n_p - len(stages_ordinal)), constant_values=0)

        if n_p <= CONFIG['n_samples']:
            p_idx = np.arange(n_p)
        else:
            p_idx = np.random.choice(n_p, CONFIG['n_samples'], replace=False)

        sub_mat = mat[p_idx]
        sub_lbl = lbl[p_idx]
        sub_stages = stages_ordinal[p_idx]
        full_input = torch.cat([sub_mat, sub_lbl.unsqueeze(1)], dim=1).unsqueeze(0)

        with torch.no_grad():
            rank_scores, stage_pred = model(full_input.to(CONFIG['device']))
            scores = rank_scores.cpu().numpy().flatten()
            stage_pred = stage_pred.cpu().numpy().flatten()

        predicted_order = np.argsort(scores)
        true_ranks_for_predicted = [ranks[i] for i in predicted_order]
        tau, _ = kendalltau(np.arange(len(ranks)) + 1, true_ranks_for_predicted)

        if not np.isnan(tau):
            taus.append(tau)

        stage_pred_rounded = np.clip(np.round(stage_pred), 0, n_bio)
        mae = np.mean(np.abs(stage_pred_rounded - sub_stages))
        stage_maes.append(mae)

    avg_tau = np.mean(taus) if taus else None
    avg_mae = np.mean(stage_maes) if stage_maes else None

    return avg_tau, avg_mae


# ==========================================
# 6. Training Loop
# ==========================================
def train_experiment(exp_name, train_dir, n_bio, max_stage):
    print(f"\n{'='*60}")
    print(f">>> TRAINING ON {exp_name}")
    print(f"{'='*60}")

    train_folder = os.path.join(train_dir, exp_name)

    standardizer = GlobalStandardizer()
    standardizer.fit(train_folder)

    with open(os.path.join(train_folder, "true_order_and_stages.json"), 'r') as f:
        gt = json.load(f)

    all_files = sorted(glob.glob(os.path.join(train_folder, "*.csv")))

    if CONFIG.get('MAX_TRAIN_FILES') and len(all_files) > CONFIG['MAX_TRAIN_FILES']:
        all_files = all_files[:CONFIG['MAX_TRAIN_FILES']]

    val_count = CONFIG.get('VAL_FILES', 25)
    if len(all_files) <= val_count:
        val_count = max(1, len(all_files) // 5)

    train_files = all_files[:-val_count]
    val_files = all_files[-val_count:]

    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    train_ds = AllBiomarkersDataset(train_files, gt, standardizer)
    loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=CONFIG.get('num_workers', 4)
    )

    model = create_model(n_bio, max_stage).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_tau = -1.0
    best_mae = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        total_rank_loss = 0
        total_stage_loss = 0
        total_pair_loss = 0

        for (full_input, rank_target, stage_labels_cont, stage_labels_ordinal,
             pair_indices, pair_label, is_cont_order) in loader:

            full_input = full_input.to(CONFIG['device'])
            rank_target = rank_target.to(CONFIG['device'])
            # Choose stage labels: ordinal recommended for small n_bio
            # stage_labels = stage_labels_cont.to(CONFIG['device'])
            stage_labels = stage_labels_ordinal.to(CONFIG['device'])
            pair_indices = pair_indices.to(CONFIG['device'])
            pair_label = pair_label.to(CONFIG['device'])
            is_cont_order = is_cont_order.to(CONFIG['device'])

            optimizer.zero_grad()

            rank_scores, stage_pred = model(full_input)

            # === Ranking Loss ===
            l_rank_direct = F.mse_loss(rank_scores, rank_target)

            batch_size = rank_scores.shape[0]
            idx_a = pair_indices[:, 0]
            idx_b = pair_indices[:, 1]

            score_a = rank_scores[torch.arange(batch_size, device=rank_scores.device), idx_a]
            score_b = rank_scores[torch.arange(batch_size, device=rank_scores.device), idx_b]
            pair_pred = (score_b - score_a).unsqueeze(1)

            if is_cont_order[0, 0] > 0.5:
                l_rank_pair = F.mse_loss(pair_pred, pair_label)
            else:
                l_rank_pair = F.binary_cross_entropy_with_logits(pair_pred, pair_label)

            l_rank = 0.5 * l_rank_direct + 0.5 * l_rank_pair

            # === Stage Loss ===
            l_stage = F.mse_loss(stage_pred, stage_labels) / (n_bio ** 2)

            loss = CONFIG['lambda_ranking'] * l_rank + CONFIG['lambda_stage'] * l_stage
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_rank_loss += l_rank_direct.item()
            total_pair_loss += l_rank_pair.item()
            total_stage_loss += l_stage.item()

        n_batches = len(loader)
        avg_loss = total_loss / n_batches
        avg_rank = total_rank_loss / n_batches
        avg_pair = total_pair_loss / n_batches
        avg_stage = total_stage_loss / n_batches

        val_tau, val_mae = evaluate_on_files(model, val_files, gt, standardizer, n_bio)
        if val_tau is None:
            val_tau = 0
        if val_mae is None:
            val_mae = float('inf')

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} (R:{avg_rank:.4f} P:{avg_pair:.4f} S:{avg_stage:.6f}) | Val Tau: {val_tau:.4f} | Val MAE: {val_mae:.2f}")

        scheduler.step(val_tau)

        if val_tau > best_tau or (val_tau == best_tau and val_mae < best_mae):
            best_tau = val_tau
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"  ^ New best!")
        else:
            no_improve += 1
            if no_improve >= CONFIG.get('patience', 20):
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    print(f"Best Val Tau: {best_tau:.4f} | Best Val MAE: {best_mae:.2f}")
    return model, standardizer


# ==========================================
# 7. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', required=True, help="List of experiments to run")
    parser.add_argument('--train_data_dir', type=str, default=None, help="Override train_data_dir from config.yaml")
    parser.add_argument('--models_dir', type=str, default=None, help="Override save_model_dir from config.yaml")
    args = parser.parse_args()

    if args.train_data_dir:
        CONFIG['train_data_dir'] = args.train_data_dir
    if args.models_dir:
        CONFIG['save_model_dir'] = args.models_dir

    save_model_dir = CONFIG['save_model_dir']

    os.makedirs(save_model_dir, exist_ok=True)
    target_exps = args.experiments

    print(f"="*60)
    print(f"Adaptive Transformer Training")
    print(f"="*60)
    print(f"Experiments: {target_exps}")
    print(f"Device: {CONFIG['device']}")
    print(f"Adaptive threshold: {CONFIG.get('ADAPTIVE_THRESHOLD', 25)} biomarkers")

    first_exp = target_exps[0]
    train_path = os.path.join(CONFIG['train_data_dir'], first_exp)

    if not os.path.exists(train_path):
        print(f"Error: Directory {train_path} does not exist.")
        sys.exit(1)

    temp_std = GlobalStandardizer()
    temp_std.fit(train_path)

    n_bio = 10
    found_files = glob.glob(os.path.join(train_path, "*.csv"))
    if found_files:
        m, _, _ = robust_load_file(found_files[0], temp_std)
        if m is not None:
            n_bio = m.shape[1]
            print(f"Detected {n_bio} biomarkers.")

    for exp in target_exps:
        exp_path = os.path.join(CONFIG['train_data_dir'], exp)
        if not os.path.exists(exp_path):
            print(f"Skipping {exp} (Folder not found)")
            continue

        model, std = train_experiment(exp, CONFIG['train_data_dir'], n_bio, n_bio)

        save_path = os.path.join(save_model_dir, f"{exp}_final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_biomarkers': n_bio,
            'max_stage': n_bio,
            'architecture_type': model.architecture_type,
            'standardizer_stats': std.stats,
            'config': CONFIG
        }, save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
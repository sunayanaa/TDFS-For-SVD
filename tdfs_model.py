"""
Filename: tdfs_model.py
Version: 2.0.0
Description: Trains the Timbre-Disentangled Fair Singing Detector (TDFS).
             v2.0.0 loads precomputed Log-Mel spectrograms from a cached NPZ
             file instead of decoding audio at runtime, eliminating all audio
             I/O overhead. Each epoch runs in under 1 minute on T4.
             Checkpoints after every epoch with full resume capability.

Changelog:
  - v1.0.0: Initial release with static GRL alpha.
  - v1.1.0: Implemented Alpha Annealing schedule.
  - v2.0.0 (2026-06-28): Replaced audio DataLoader with NPZ spectrogram cache.
                         Eliminates Drive I/O bottleneck and DataLoader OOM kills.

Hardware: GPU (T4 recommended)
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import warnings
warnings.filterwarnings("ignore")

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from tqdm import tqdm

# ==========================================
# PATHS
# ==========================================
MANIFEST        = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
NPZ_PATH        = '/content/drive/MyDrive/paper/DecouplingTimbre/jamendo_spectrograms.npz'
RESULTS_FILE    = '/content/drive/MyDrive/paper/DecouplingTimbre/tdfs_results_v2.json'
CHECKPOINT_PATH = '/content/drive/MyDrive/paper/DecouplingTimbre/tdfs_checkpoint_v2.pth'
FINAL_MODEL     = '/content/drive/MyDrive/paper/DecouplingTimbre/tdfs_final_v2.pth'
# Also save a copy to the original Jamendo path so spl01 finds it
FINAL_MODEL_LEGACY = '/content/drive/MyDrive/datasets/Jamendo/tdfs_final_v2.pth'

# ==========================================
# 1. GRADIENT REVERSAL LAYER
# ==========================================
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# ==========================================
# 2. DATASET — loads from NPZ cache
# ==========================================
class JamendoNPZDataset(Dataset):
    def __init__(self, manifest_path, npz_path):
        print(f"Loading spectrogram cache from {npz_path}...")
        self.cache = np.load(npz_path)
        print(f"Cache loaded: {len(self.cache.files)} spectrograms.")

        df = pd.read_csv(manifest_path)
        df = df[df['pitch_class'] != 'error'].reset_index(drop=True)

        self.pitch_map = {
            'Low-Pitch (Male Proxy)':   0,
            'High-Pitch (Female Proxy)': 1,
            'Non-Vocal':                2,
            'unknown':                  3
        }

        # Keep only rows whose track_id exists in the NPZ
        df['track_id'] = df['path'].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )
        before = len(df)
        df = df[df['track_id'].isin(self.cache.files)].reset_index(drop=True)
        print(f"Manifest rows: {before} -> {len(df)} after NPZ filter.")
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_id = row['track_id']

        # Load precomputed spectrogram [80, T] and add channel dim -> [1, 80, T]
        spec = torch.tensor(self.cache[track_id], dtype=torch.float32).unsqueeze(0)

        if row['is_soft_timbre']:
            timbre_class = 1
        elif row['is_powerful_timbre']:
            timbre_class = 2
        else:
            timbre_class = 0

        labels = {
            "is_vocal":     torch.tensor(
                                1.0 if row['is_vocal'] else 0.0,
                                dtype=torch.float32),
            "pitch_class":  torch.tensor(
                                self.pitch_map.get(row['pitch_class'], 3),
                                dtype=torch.long),
            "timbre_class": torch.tensor(timbre_class, dtype=torch.long)
        }
        return spec, labels

# ==========================================
# 3. TDFS ARCHITECTURE
# ==========================================
class TDFS(nn.Module):
    def __init__(self):
        super(TDFS, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.vocal_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
        self.timbre_adversary = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3)
        )

    def forward(self, x, alpha=1.0):
        features = self.extractor(x)
        vocal_pred   = self.vocal_classifier(features)
        timbre_pred  = self.timbre_adversary(
            GradientReversalLayer.apply(features, alpha)
        )
        return vocal_pred, timbre_pred

# ==========================================
# 4. TRAINING
# ==========================================
def train_tdfs():
    EPOCHS     = 15
    BATCH_SIZE = 32
    LR         = 1e-3
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Dataset and DataLoader
    # num_workers=0 avoids worker OOM since data is already in RAM after NPZ load
    dataset    = JamendoNPZDataset(MANIFEST, NPZ_PATH)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda')
    )

    model = TDFS().to(DEVICE)

    weight           = torch.tensor([6.2]).to(DEVICE)
    criterion_vocal  = nn.BCEWithLogitsLoss(pos_weight=weight)
    criterion_timbre = nn.CrossEntropyLoss()
    optimizer        = optim.Adam(model.parameters(), lr=LR)

    start_epoch    = 0
    experiment_log = {
        "experiment_name": "TDFS_Model_Annealed_v2",
        "epoch_metrics":   []
    }

    # Resume from checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                experiment_log = json.load(f)
        print(f"Resuming from epoch {start_epoch}.")
    else:
        print("Starting TDFS training from scratch.")

    for epoch in range(start_epoch, EPOCHS):
        # Alpha annealing: 0.0 -> 1.0 over training
        p             = float(epoch) / EPOCHS
        current_alpha = 2. / (1. + np.exp(-10 * p)) - 1.

        model.train()
        running_vocal_loss  = 0.0
        running_timbre_loss = 0.0
        correct_vocal       = 0
        total               = 0

        loop = tqdm(dataloader,
                    desc=f"Epoch {epoch+1}/{EPOCHS} [Alpha: {current_alpha:.3f}]")

        for specs, labels in loop:
            specs          = specs.to(DEVICE)
            targets_vocal  = labels['is_vocal'].unsqueeze(1).to(DEVICE)
            targets_timbre = labels['timbre_class'].to(DEVICE)

            optimizer.zero_grad()
            vocal_outputs, timbre_outputs = model(specs, alpha=current_alpha)

            loss_vocal  = criterion_vocal(vocal_outputs, targets_vocal)
            loss_timbre = criterion_timbre(timbre_outputs, targets_timbre)
            total_loss  = loss_vocal + loss_timbre
            total_loss.backward()
            optimizer.step()

            running_vocal_loss  += loss_vocal.item()
            running_timbre_loss += loss_timbre.item()

            predictions    = torch.sigmoid(vocal_outputs) > 0.5
            correct_vocal += (predictions == targets_vocal).sum().item()
            total         += targets_vocal.size(0)

            loop.set_postfix(
                v_loss=loss_vocal.item(),
                t_loss=loss_timbre.item(),
                v_acc=correct_vocal / total
            )

        epoch_v_loss = running_vocal_loss  / len(dataloader)
        epoch_t_loss = running_timbre_loss / len(dataloader)
        epoch_acc    = correct_vocal / total

        print(f"Epoch {epoch+1} Summary -> "
              f"Vocal Loss: {epoch_v_loss:.4f} | "
              f"Timbre Loss: {epoch_t_loss:.4f} | "
              f"Vocal Acc: {epoch_acc:.4f} | "
              f"Alpha: {current_alpha:.3f}")

        experiment_log["epoch_metrics"].append({
            "epoch":       epoch + 1,
            "alpha":       round(current_alpha, 4),
            "vocal_loss":  round(epoch_v_loss, 4),
            "timbre_loss": round(epoch_t_loss, 4),
            "vocal_accuracy": round(epoch_acc, 4)
        })

        # Save results log
        with open(RESULTS_FILE, 'w') as f:
            json.dump(experiment_log, f, indent=4)
        os.sync()

        # Save checkpoint
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocal_loss':           epoch_v_loss,
        }, CHECKPOINT_PATH)
        os.sync()

    # Save final model to both locations so spl01 finds it
    torch.save(model.state_dict(), FINAL_MODEL)
    torch.save(model.state_dict(), FINAL_MODEL_LEGACY)
    os.sync()
    print(f"\nTraining complete.")
    print(f"Final model saved to {FINAL_MODEL}")
    print(f"Legacy copy saved to {FINAL_MODEL_LEGACY}")

if __name__ == "__main__":
    train_tdfs()
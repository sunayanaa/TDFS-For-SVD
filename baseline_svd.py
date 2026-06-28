"""
Filename: baseline_svd.py
Version: 2.0.0
Description: Trains the Baseline (Vanilla) CNN for Singing Voice Detection.
             v2.0.0 loads precomputed Log-Mel spectrograms from a cached NPZ
             file instead of decoding audio at runtime, eliminating all audio
             I/O overhead. Each epoch runs in under 1 minute on T4.
             Checkpoints after every epoch with full resume capability.

Changelog:
  - v1.0.0: Initial release.
  - v1.2.0: Self-contained script.
  - v1.3.0: Solved GPU Starvation by moving Mel-Spec math to the accelerator.
  - v1.4.0: Added BCEWithLogitsLoss pos_weight (6.2) to fix mode collapse.
  - v2.0.0 (2026-06-28): Replaced audio DataLoader with NPZ spectrogram cache.
                         Eliminates Drive I/O bottleneck and DataLoader OOM.

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
from tqdm import tqdm

# ==========================================
# PATHS
# ==========================================
MANIFEST        = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
NPZ_PATH        = '/content/drive/MyDrive/paper/DecouplingTimbre/jamendo_spectrograms.npz'
RESULTS_FILE    = '/content/drive/MyDrive/paper/DecouplingTimbre/baseline_results.json'
CHECKPOINT_PATH = '/content/drive/MyDrive/paper/DecouplingTimbre/vanilla_svd_checkpoint.pth'
FINAL_MODEL     = '/content/drive/MyDrive/paper/DecouplingTimbre/vanilla_svd_final.pth'
FINAL_MODEL_LEGACY = '/content/drive/MyDrive/datasets/Jamendo/vanilla_svd_final.pth'

# ==========================================
# 1. DATASET — loads from NPZ cache
# ==========================================
class JamendoNPZDataset(Dataset):
    def __init__(self, manifest_path, npz_path):
        print(f"Loading spectrogram cache from {npz_path}...")
        self.cache = np.load(npz_path)
        print(f"Cache loaded: {len(self.cache.files)} spectrograms.")

        df = pd.read_csv(manifest_path)
        df = df[df['pitch_class'] != 'error'].reset_index(drop=True)
        df['track_id'] = df['path'].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )
        before = len(df)
        df = df[df['track_id'].isin(self.cache.files)].reset_index(drop=True)
        print(f"Manifest rows: {before} -> {len(df)} after NPZ filter.")
        print(f"Vocal: {df['is_vocal'].sum()} | Non-vocal: {(~df['is_vocal']).sum()}")
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        track_id = row['track_id']
        spec     = torch.tensor(
            self.cache[track_id], dtype=torch.float32
        ).unsqueeze(0)  # [1, 80, 188]

        labels = {
            "is_vocal": torch.tensor(
                1.0 if row['is_vocal'] else 0.0, dtype=torch.float32
            )
        }
        return spec, labels

# ==========================================
# 2. ARCHITECTURE
# ==========================================
class VanillaSVD(nn.Module):
    def __init__(self):
        super(VanillaSVD, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 3. TRAINING
# ==========================================
def train_baseline():
    EPOCHS     = 10
    BATCH_SIZE = 32
    LR         = 1e-3
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    dataset    = JamendoNPZDataset(MANIFEST, NPZ_PATH)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda')
    )

    model = VanillaSVD().to(DEVICE)

    # pos_weight = n_non_vocal / n_vocal = 3585 / 578 = 6.2
    weight    = torch.tensor([6.2]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch    = 0
    experiment_log = {
        "experiment_name": "Vanilla_SVD_Baseline_v2",
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
        print("Starting Vanilla SVD training from scratch.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for specs, labels in loop:
            specs   = specs.to(DEVICE)
            targets = labels['is_vocal'].unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(specs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions   = torch.sigmoid(outputs) > 0.5
            correct      += (predictions == targets).sum().item()
            total        += targets.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct / total)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc  = correct / total
        print(f"Epoch {epoch+1} Summary -> "
              f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        experiment_log["epoch_metrics"].append({
            "epoch":          epoch + 1,
            "train_loss":     round(epoch_loss, 4),
            "train_accuracy": round(epoch_acc, 4)
        })

        with open(RESULTS_FILE, 'w') as f:
            json.dump(experiment_log, f, indent=4)
        os.sync()

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 epoch_loss,
        }, CHECKPOINT_PATH)
        os.sync()

    # Save final model to both locations
    torch.save(model.state_dict(), FINAL_MODEL)
    torch.save(model.state_dict(), FINAL_MODEL_LEGACY)
    os.sync()
    print(f"\nTraining complete.")
    print(f"Final model saved to {FINAL_MODEL}")
    print(f"Legacy copy saved to  {FINAL_MODEL_LEGACY}")

if __name__ == "__main__":
    train_baseline()
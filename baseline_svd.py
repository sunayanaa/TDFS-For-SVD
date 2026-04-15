"""
Filename: baseline_svd.py
Version: 1.4.0
Description: Trains a Baseline (Vanilla) CNN for Singing Voice Detection.
             
Changelog:
  - v1.2.0: Self-contained script.
  - v1.3.0: Solved GPU Starvation by moving Mel-Spec math to the accelerator.
  - v1.4.0 (2026-04-13): Added BCEWithLogitsLoss pos_weight (6.2) to fix 
                         mode collapse caused by class imbalance. Added 
                         Drive mount for strict self-containment.
"""

from google.colab import drive
drive.mount('/content/drive')

import warnings
warnings.filterwarnings("ignore", message="This DataLoader will create.*worker processes in total")

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. OPTIMIZED DATASET 
# ==========================================
class JamendoFairnessDataset(Dataset):
    def __init__(self, manifest_path, audio_dir, segment_length=48000):
        self.audio_dir = audio_dir
        self.segment_length = segment_length
        
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['pitch_class'] != 'error'].reset_index(drop=True)

        self.pitch_map = {
            'Low-Pitch (Male Proxy)': 0,
            'High-Pitch (Female Proxy)': 1,
            'Non-Vocal': 2,
            'unknown': 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_name = os.path.basename(row['path']).replace('.mp3', '.wav')
        wav_path = os.path.join(self.audio_dir, wav_name)

        waveform, sr = torchaudio.load(wav_path)
        
        if waveform.shape[1] > self.segment_length:
            max_start = waveform.shape[1] - self.segment_length
            start = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[:, start : start + self.segment_length]
        else:
            pad_amount = self.segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        labels = {
            "is_vocal": torch.tensor(1.0 if row['is_vocal'] else 0.0, dtype=torch.float32),
            "pitch_class": torch.tensor(self.pitch_map.get(row['pitch_class'], 3), dtype=torch.long)
        }
        
        return waveform, labels

# ==========================================
# 2. ARCHITECTURE 
# ==========================================
class VanillaSVD(nn.Module):
    def __init__(self):
        super(VanillaSVD, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ==========================================
# 3. TRAINING COMPONENT
# ==========================================
def train_baseline():
    MANIFEST = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
    AUDIO_DIR = '/content/drive/MyDrive/datasets/Jamendo/wav_24k'
    RESULTS_FILE = '/content/drive/MyDrive/datasets/Jamendo/exp1_baseline_results.json'
    CHECKPOINT_PATH = '/content/drive/MyDrive/datasets/Jamendo/vanilla_svd_checkpoint.pth'
    FINAL_MODEL_PATH = '/content/drive/MyDrive/datasets/Jamendo/vanilla_svd_final.pth'

    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Initialization
    dataset = JamendoFairnessDataset(MANIFEST, AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = VanillaSVD().to(DEVICE)
    
    # --- THE FIX: POSITIVE CLASS WEIGHTING ---
    # Total Negatives (3585) / Total Positives (578) = ~6.2
    # This forces the network to penalize missed vocals 6.2x more heavily.
    weight = torch.tensor([6.2]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight) 
    
    optimizer = optim.Adam(model.parameters(), lr=LR)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, f_min=0.0, f_max=12000.0, n_mels=80
    ).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    start_epoch = 0
    experiment_log = {
        "experiment_name": "Vanilla_SVD_Baseline",
        "epoch_metrics": []
    }

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        with open(RESULTS_FILE, 'r') as f:
            experiment_log = json.load(f)
    else:
        print("Starting Baseline Training from scratch...")

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for waveforms, labels in loop:
            waveforms = waveforms.to(DEVICE)
            targets = labels['is_vocal'].unsqueeze(1).to(DEVICE)

            mels = amplitude_to_db(mel_transform(waveforms))

            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct/total)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} Summary -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        experiment_log["epoch_metrics"].append({
            "epoch": epoch + 1, "train_loss": epoch_loss, "train_accuracy": epoch_acc
        })

        with open(RESULTS_FILE, 'w') as f:
            json.dump(experiment_log, f, indent=4)
            
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss,
        }, CHECKPOINT_PATH)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nTraining complete. Final model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train_baseline()
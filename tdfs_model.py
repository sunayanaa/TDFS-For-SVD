"""
Filename: tdfs_model.py
Version: 1.1.0
Description: Trains the novel Timbre-Disentangled Fair Singing Detector (TDFS).
             
Changelog:
  - v1.0.0: Initial release with static GRL alpha.
  - v1.1.0 (2026-04-13): Implemented Alpha Annealing schedule. The GRL penalty 
                         now gradually scales from 0.0 to 1.0 over the epochs, 
                         preventing the adversary from overwhelming the feature 
                         extractor during early vocal feature acquisition.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import warnings
warnings.filterwarnings("ignore", message="This DataLoader will create.*worker processes in total")

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from tqdm import tqdm

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
        output = grad_output.neg() * ctx.alpha
        return output, None

# ==========================================
# 2. DATASET
# ==========================================
class JamendoFairnessDataset(Dataset):
    def __init__(self, manifest_path, audio_dir, segment_length=48000):
        self.audio_dir = audio_dir
        self.segment_length = segment_length
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['pitch_class'] != 'error'].reset_index(drop=True)
        self.pitch_map = {'Low-Pitch (Male Proxy)': 0, 'High-Pitch (Female Proxy)': 1, 'Non-Vocal': 2, 'unknown': 3}

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

        if row['is_soft_timbre']:
            timbre_class = 1
        elif row['is_powerful_timbre']:
            timbre_class = 2
        else:
            timbre_class = 0

        labels = {
            "is_vocal": torch.tensor(1.0 if row['is_vocal'] else 0.0, dtype=torch.float32),
            "pitch_class": torch.tensor(self.pitch_map.get(row['pitch_class'], 3), dtype=torch.long),
            "timbre_class": torch.tensor(timbre_class, dtype=torch.long)
        }
        return waveform, labels

# ==========================================
# 3. TDFS ARCHITECTURE
# ==========================================
class TDFS(nn.Module):
    def __init__(self):
        super(TDFS, self).__init__()
        self.extractor = nn.Sequential(
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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.vocal_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) 
        )
        self.timbre_adversary = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3) 
        )

    def forward(self, x, alpha=1.0):
        features = self.extractor(x)
        vocal_pred = self.vocal_classifier(features)
        reversed_features = GradientReversalLayer.apply(features, alpha)
        timbre_pred = self.timbre_adversary(reversed_features)
        return vocal_pred, timbre_pred

# ==========================================
# 4. ADVERSARIAL TRAINING 
# ==========================================
def train_tdfs():
    MANIFEST = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
    AUDIO_DIR = '/content/drive/MyDrive/datasets/Jamendo/wav_24k'
    RESULTS_FILE = '/content/drive/MyDrive/datasets/Jamendo/exp2_tdfs_results_v2.json'
    CHECKPOINT_PATH = '/content/drive/MyDrive/datasets/Jamendo/tdfs_checkpoint_v2.pth'
    FINAL_MODEL_PATH = '/content/drive/MyDrive/datasets/Jamendo/tdfs_final_v2.pth'

    EPOCHS = 15
    BATCH_SIZE = 32
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    dataset = JamendoFairnessDataset(MANIFEST, AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = TDFS().to(DEVICE)
    
    weight = torch.tensor([6.2]).to(DEVICE) 
    criterion_vocal = nn.BCEWithLogitsLoss(pos_weight=weight) 
    criterion_timbre = nn.CrossEntropyLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=LR)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, f_min=0.0, f_max=12000.0, n_mels=80
    ).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    start_epoch = 0
    experiment_log = {"experiment_name": "TDFS_Model_Annealed", "epoch_metrics": []}

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        # --- THE MAGIC: ALPHA ANNEALING SCHEDULE ---
        # Progress (p) goes from 0 to 1 over the training loop
        p = float(epoch) / EPOCHS
        # Formula gradually curves alpha from 0.0 to 1.0
        current_alpha = 2. / (1. + np.exp(-10 * p)) - 1.
        
        running_vocal_loss = 0.0
        running_timbre_loss = 0.0
        correct_vocal = 0
        total = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Alpha: {current_alpha:.3f}]")
        for waveforms, labels in loop:
            waveforms = waveforms.to(DEVICE)
            targets_vocal = labels['is_vocal'].unsqueeze(1).to(DEVICE)
            targets_timbre = labels['timbre_class'].to(DEVICE)

            mels = amplitude_to_db(mel_transform(waveforms))

            optimizer.zero_grad()
            
            # Pass the dynamically changing alpha to the network
            vocal_outputs, timbre_outputs = model(mels, alpha=current_alpha)
            
            loss_vocal = criterion_vocal(vocal_outputs, targets_vocal)
            loss_timbre = criterion_timbre(timbre_outputs, targets_timbre)
            
            total_loss = loss_vocal + loss_timbre
            total_loss.backward()
            optimizer.step()

            running_vocal_loss += loss_vocal.item()
            running_timbre_loss += loss_timbre.item()
            
            predictions = torch.sigmoid(vocal_outputs) > 0.5
            correct_vocal += (predictions == targets_vocal).sum().item()
            total += targets_vocal.size(0)

            loop.set_postfix(v_loss=loss_vocal.item(), t_loss=loss_timbre.item(), v_acc=correct_vocal/total)

        epoch_v_loss = running_vocal_loss / len(dataloader)
        epoch_t_loss = running_timbre_loss / len(dataloader)
        epoch_acc = correct_vocal / total
        
        print(f"Epoch {epoch+1} Summary -> Vocal Loss: {epoch_v_loss:.4f} | Timbre Loss: {epoch_t_loss:.4f} | Vocal Acc: {epoch_acc:.4f}")

        experiment_log["epoch_metrics"].append({
            "epoch": epoch + 1, "alpha": current_alpha, "vocal_loss": epoch_v_loss, "timbre_loss": epoch_t_loss, "vocal_accuracy": epoch_acc
        })

        with open(RESULTS_FILE, 'w') as f:
            json.dump(experiment_log, f, indent=4)
            
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'vocal_loss': epoch_v_loss,
        }, CHECKPOINT_PATH)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nTraining complete. Final TDFS model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train_tdfs()
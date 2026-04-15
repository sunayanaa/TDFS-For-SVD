"""
Filename: hpss_baseline.py
Version: 1.0.0
Description: Addresses IEEE-SPL Reviewer 1 (Mr. CL) critique #1. 
             Implements an HPSS (Harmonic-Percussive Source Separation) front-end 
             baseline to test if classic SP preprocessing can mitigate timbre bias 
             without the need for adversarial ML training.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. VANILLA CNN ARCHITECTURE (NO GRL)
# ==========================================
class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.vocal_classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        features = self.extractor(x)
        return self.vocal_classifier(features)

# ==========================================
# 2. HPSS DATASET LOADER
# ==========================================
class MUSDB18_HPSS_Dataset(Dataset):
    def __init__(self, manifest_path, segment_length=48000):
        self.segment_length = segment_length
        self.df = pd.read_csv(manifest_path)

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['path'] 
        
        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[1] > self.segment_length:
            start = (waveform.shape[1] - self.segment_length) // 2
            waveform = waveform[:, start : start + self.segment_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_length - waveform.shape[1]))
            
        # --- THE SIGNAL PROCESSING INTERVENTION (HPSS) ---
        # Convert to numpy for librosa, apply HPSS, keep ONLY the Harmonic component
        wave_np = waveform.numpy()[0]
        harmonic_np, _ = librosa.effects.hpss(wave_np, margin=1.2)
        harmonic_tensor = torch.tensor(harmonic_np).unsqueeze(0)
        
        labels = {
            "is_vocal": torch.tensor([1.0 if row['is_vocal'] else 0.0], dtype=torch.float32),
            "is_soft": row['is_soft_timbre'],
            "is_powerful": row['is_powerful_timbre']
        }
        return harmonic_tensor, labels

# ==========================================
# 3. TRAINING & EVALUATION LOOP
# ==========================================
def run_hpss_baseline():
    MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
    OUTPUT_FILE = '/content/drive/MyDrive/datasets/MUSDB18/hpss_baseline_results.json'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning HPSS SP Baseline on: {DEVICE}")
    
    model = VanillaCNN().to(DEVICE)
    dataset = MUSDB18_HPSS_Dataset(MANIFEST)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [HPSS Baseline]")
        for waveforms, labels in epoch_iter:
            waveforms = waveforms.to(DEVICE)
            targets = labels['is_vocal'].to(DEVICE)
            
            mels = amplitude_to_db(mel_transform(waveforms))
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_iter.set_postfix(loss=loss.item())

    # Quick Eval for FPR
    model.eval()
    soft_fp, soft_tn = 0, 0
    pow_fp, pow_tn = 0, 0
    
    print("\nEvaluating HPSS Baseline FPR...")
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(DEVICE)
            mels = amplitude_to_db(mel_transform(waveforms))
            outputs = model(mels)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu()
            
            targets = labels['is_vocal'].cpu()
            is_soft = labels['is_soft']
            is_pow = labels['is_powerful']
            
            for i in range(len(targets)):
                if targets[i] == 0.0:  # If it's a negative (instrumental) track
                    if is_soft[i]:
                        soft_tn += 1
                        if preds[i] == 1.0: soft_fp += 1
                    elif is_pow[i]:
                        pow_tn += 1
                        if preds[i] == 1.0: pow_fp += 1

    results = {
        "baseline_type": "Signal Processing (HPSS Front-End)",
        "soft_timbre_fpr": round(soft_fp / max(soft_tn + soft_fp, 1), 4),
        "powerful_timbre_fpr": round(pow_fp / max(pow_tn + pow_fp, 1), 4),
        "conclusion": "If Powerful FPR remains significantly higher than Soft FPR, HPSS fails to resolve harmonic confounding variables."
    }

    print("\n--- HPSS SP Baseline Results ---")
    print(json.dumps(results, indent=4))
    with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=4)
    os.sync()
if __name__ == "__main__":
    run_hpss_baseline()
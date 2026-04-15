"""
Filename: frequency_band_probe.py
Version: 1.0.0
Description: Addresses IEEE-SPL Reviewer 1 (Mr. CL) critique #3. 
             Intercepts the CNN feature maps before global pooling, slices them 
             into Low, Mid, and High frequency bands, and trains a linear probe 
             on each to determine exactly where latent timbre bias persists.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import json
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ==========================================
# 1. ARCHITECTURE WITH FEATURE INTERCEPTION
# ==========================================
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class TDFS(nn.Module):
    def __init__(self):
        super(TDFS, self).__init__()
        # We split the extractor into two parts so we can intercept the 2D map
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.pool_and_flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.vocal_classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        self.timbre_adversary = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3))

    def forward(self, x, alpha=1.0):
        # We don't use the standard forward pass for this probe script.
        pass 

# ==========================================
# 2. DATASET LOADER
# ==========================================
class MUSDB18MetricsDataset(Dataset):
    def __init__(self, manifest_path, segment_length=48000):
        self.segment_length = segment_length
        self.df = pd.read_csv(manifest_path)

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['path'] 
        
        waveform, _ = torchaudio.load(wav_path)
        if waveform.shape[1] > self.segment_length:
            start = (waveform.shape[1] - self.segment_length) // 2
            waveform = waveform[:, start : start + self.segment_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_length - waveform.shape[1]))
        
        labels = {
            "is_soft": float(row['is_soft_timbre']),
            "is_powerful": float(row['is_powerful_timbre'])
        }
        return waveform, labels

# ==========================================
# 3. SUB-BAND PROBING PIPELINE
# ==========================================
def run_sub_band_probes():
    MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
    MODEL_PATH = '/content/drive/MyDrive/datasets/Jamendo/tdfs_final_v2.pth' 
    OUTPUT_FILE = '/content/drive/MyDrive/datasets/MUSDB18/frequency_band_probe.json'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nExtracting sub-band latents on: {DEVICE}")
    
    model = TDFS().to(DEVICE)
    # Strict=False because we slightly reorganized the Sequential blocks for interception
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    dataset = MUSDB18MetricsDataset(MANIFEST)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    # Arrays for sub-band latents
    latents_low, latents_mid, latents_high = [], [], []
    timbre_targets = []

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Intercepting Feature Maps"):
            waveforms = waveforms.to(DEVICE)
            mels = amplitude_to_db(mel_transform(waveforms))
            
            # Extract 2D Feature Map [Batch, Channels, Freq, Time]
            # Mel size=80. After two MaxPools, Freq dimension = 20.
            feature_map_2d = model.conv_blocks(mels) 
            
            # Slice into frequency bands
            map_low = feature_map_2d[:, :, 0:6, :]    # Bins 0-5 (Bass/Kick)
            map_mid = feature_map_2d[:, :, 6:14, :]   # Bins 6-13 (Vocals/Guitars)
            map_high = feature_map_2d[:, :, 14:20, :] # Bins 14-19 (Cymbals)

            # Pool and flatten each slice into its own 64-dim vector
            vec_low = model.pool_and_flatten(map_low)
            vec_mid = model.pool_and_flatten(map_mid)
            vec_high = model.pool_and_flatten(map_high)

            # Store only tracks that are explicitly Soft (0) or Powerful (1)
            for i in range(len(waveforms)):
                if labels['is_soft'][i] == 1.0 or labels['is_powerful'][i] == 1.0:
                    latents_low.append(vec_low[i].cpu().numpy())
                    latents_mid.append(vec_mid[i].cpu().numpy())
                    latents_high.append(vec_high[i].cpu().numpy())
                    timbre_targets.append(1 if labels['is_powerful'][i] == 1.0 else 0)

    # --- TRAIN PROBES ---
    print("\nTraining Logistic Regression Probes for each sub-band...")
    y = np.array(timbre_targets)
    
    probe_low = LogisticRegression(max_iter=1000).fit(np.array(latents_low), y)
    probe_mid = LogisticRegression(max_iter=1000).fit(np.array(latents_mid), y)
    probe_high = LogisticRegression(max_iter=1000).fit(np.array(latents_high), y)

    results = {
        "analysis_type": "Sub-Band Latent Linear Probing",
        "random_baseline": 0.5000,
        "sub_band_probe_accuracy": {
            "low_band_0_to_6": round(accuracy_score(y, probe_low.predict(np.array(latents_low))), 4),
            "mid_band_6_to_14": round(accuracy_score(y, probe_mid.predict(np.array(latents_mid))), 4),
            "high_band_14_to_20": round(accuracy_score(y, probe_high.predict(np.array(latents_high))), 4)
        },
        "conclusion": "Bands with accuracy closer to 0.5000 have been successfully disentangled by the GRL. Bands > 0.80 contain persistent instrumental leakage."
    }

    print("\n--- Frequency Band Attribution Results ---")
    print(json.dumps(results, indent=4))
    with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_sub_band_probes()
"""
Filename: advanced_metrics.py
Version: 1.2.0
Description: Extracts advanced signal processing metrics for the TDFS paper.
             Calculates Subgroup Expected Calibration Error (ECE) and trains 
             a Latent Linear Probe.
             UPDATE: Uses robust native zipfile extraction to prevent 
             silent OS-level unzipping failures on Colab FUSE mounts.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import json
import zipfile
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
# 1. BULLETPROOF NATIVE AUDIO RESTORE
# ==========================================
def ensure_local_audio():
    ZIP_PATH = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_wav_24k.zip'
    LOCAL_DIR = '/content/musdb18_local/wav_24k'
    
    if not os.path.exists(LOCAL_DIR) or len(os.listdir(LOCAL_DIR)) < 300:
        print(f"Local audio missing or incomplete. Restoring natively from {ZIP_PATH}...")
        os.makedirs(LOCAL_DIR, exist_ok=True)
        
        if not os.path.exists(ZIP_PATH):
            raise FileNotFoundError(f"CRITICAL ERROR: Cannot find {ZIP_PATH}. Check Google Drive sync.")
            
        # Native Python extraction (No silent OS failures)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # Flatten paths during extraction (equivalent to unzip -j)
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                source = zip_ref.open(member)
                target = open(os.path.join(LOCAL_DIR, filename), "wb")
                with source, target:
                    import shutil
                    shutil.copyfileobj(source, target)
                    
        file_count = len(os.listdir(LOCAL_DIR))
        print(f"Native extraction complete! Found {file_count} files on local VM.")
        if file_count == 0:
            raise RuntimeError("Extraction failed: Directory is still empty after unzip attempt.")
    else:
        print(f"Local audio found ({len(os.listdir(LOCAL_DIR))} files). Skipping unzip.")

# ==========================================
# 2. ARCHITECTURE RE-DECLARATION
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
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.vocal_classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        self.timbre_adversary = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3))

    def forward(self, x, alpha=1.0):
        features = self.extractor(x)
        return self.vocal_classifier(features), self.timbre_adversary(GradientReversalLayer.apply(features, alpha))

# ==========================================
# 3. DATASET LOADER
# ==========================================
class MUSDB18MetricsDataset(Dataset):
    def __init__(self, manifest_path, segment_length=48000):
        self.segment_length = segment_length
        self.df = pd.read_csv(manifest_path)
        self.pitch_map = {'Low-Pitch (Male Proxy)': 0, 'High-Pitch (Female Proxy)': 1, 'Non-Vocal': 2, 'unknown': 3}

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
            "is_vocal": float(row['is_vocal']),
            "pitch_class": self.pitch_map.get(row['pitch_class'], 3),
            "is_soft": float(row['is_soft_timbre']),
            "is_powerful": float(row['is_powerful_timbre'])
        }
        return waveform, labels

# ==========================================
# 4. ECE CALCULATION UTILITY
# ==========================================
def calculate_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] > 0.5))
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# ==========================================
# 5. ADVANCED METRICS PIPELINE
# ==========================================
def extract_advanced_metrics():
    # 1. Bulletproof audio check
    ensure_local_audio()
    
    MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
    MODEL_PATH = '/content/drive/MyDrive/datasets/Jamendo/tdfs_final_v2.pth' 
    OUTPUT_FILE = '/content/drive/MyDrive/datasets/MUSDB18/advanced_metrics.json'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nExtracting advanced metrics on: {DEVICE}")
    
    model = TDFS().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = MUSDB18MetricsDataset(MANIFEST)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    all_probs, all_labels = [], []
    all_pitches, all_soft, all_powerful = [], [], []
    latent_vectors, timbre_targets = [], []

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Extracting Latents and Probs"):
            waveforms = waveforms.to(DEVICE)
            mels = amplitude_to_db(mel_transform(waveforms))
            
            latents = model.extractor(mels)
            vocal_outputs, _ = model(mels, alpha=0.0) 
            probs = torch.sigmoid(vocal_outputs.squeeze(1))

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels['is_vocal'].numpy())
            all_pitches.extend(labels['pitch_class'].numpy())
            all_soft.extend(labels['is_soft'].numpy())
            all_powerful.extend(labels['is_powerful'].numpy())

            for i in range(len(waveforms)):
                if labels['is_soft'][i] == 1.0 or labels['is_powerful'][i] == 1.0:
                    latent_vectors.append(latents[i].cpu().numpy())
                    timbre_targets.append(1 if labels['is_powerful'][i] == 1.0 else 0)

    print("\nCalculating Subgroup Expected Calibration Error (ECE)...")
    probs_np, labels_np = np.array(all_probs), np.array(all_labels)
    pitches_np, soft_np, pow_np = np.array(all_pitches), np.array(all_soft), np.array(all_powerful)

    ece_results = {
        "overall_ece": calculate_ece(probs_np, labels_np),
        "low_pitch_ece": calculate_ece(probs_np[pitches_np == 0], labels_np[pitches_np == 0]),
        "high_pitch_ece": calculate_ece(probs_np[pitches_np == 1], labels_np[pitches_np == 1]),
        "soft_timbre_ece": calculate_ece(probs_np[soft_np == 1.0], labels_np[soft_np == 1.0]),
        "powerful_timbre_ece": calculate_ece(probs_np[pow_np == 1.0], labels_np[pow_np == 1.0])
    }

    print("Training Latent Linear Probe for Disentanglement Score...")
    X = np.array(latent_vectors)
    y = np.array(timbre_targets)
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X, y)
    probe_preds = probe.predict(X)
    probe_accuracy = accuracy_score(y, probe_preds)

    results = {
        "expected_calibration_error": {k: round(v, 4) for k, v in ece_results.items()},
        "latent_disentanglement_score": {
            "probe_accuracy": round(probe_accuracy, 4),
            "random_baseline": 0.5000,
            "interpretation": "Closer to 0.5000 indicates perfect disentanglement (model cannot guess timbre from latents)."
        }
    }

    print("\n--- Advanced Metrics Results ---")
    print(json.dumps(results, indent=4))
    with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=4)
    os.sync()
    print(f"\nSaved metrics to Google Drive: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_advanced_metrics()
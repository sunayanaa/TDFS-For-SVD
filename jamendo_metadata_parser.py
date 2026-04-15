"""
Filename: evaluate_baseline.py
Version: 1.1.0
Description: Audits the trained Vanilla SVD model for fairness. 
             Calculates accuracy, False Positive Rate (FPR), and False Negative 
             Rate (FNR) across the f0-derived gender proxies and timbre subgroups.
             Outputs a detailed JSON report for IEEE paper tables.

Changelog:
  - v1.0.0: Initial release.
  - v1.1.0 (2026-04-13): Made completely self-contained. Added Google Drive mount,
                         integrated Dataset and Architecture classes, and moved 
                         Mel-Spectrogram processing to GPU to match training script.
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import json
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="This DataLoader will create.*worker processes in total")

# ==========================================
# 1. DATASET COMPONENT
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
            "pitch_class": torch.tensor(self.pitch_map.get(row['pitch_class'], 3), dtype=torch.long),
            "is_soft": torch.tensor(1.0 if row['is_soft_timbre'] else 0.0, dtype=torch.float32),
            "is_powerful": torch.tensor(1.0 if row['is_powerful_timbre'] else 0.0, dtype=torch.float32)
        }
        return waveform, labels

# ==========================================
# 2. ARCHITECTURE COMPONENT
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
# 3. EVALUATION COMPONENT
# ==========================================
def evaluate_model():
    MANIFEST = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
    AUDIO_DIR = '/content/drive/MyDrive/datasets/Jamendo/wav_24k'
    MODEL_PATH = '/content/drive/MyDrive/datasets/Jamendo/vanilla_svd_final.pth'
    AUDIT_RESULTS_FILE = '/content/drive/MyDrive/datasets/Jamendo/exp1_fairness_audit.json'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Auditing on device: {DEVICE}")

    # Load Model
    model = VanillaSVD().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Data
    dataset = JamendoFairnessDataset(MANIFEST, AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # GPU Audio Transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, f_min=0.0, f_max=12000.0, n_mels=80
    ).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    # Trackers
    results = {
        "overall": {"correct": 0, "total": 0},
        "low_pitch_male_proxy": {"correct": 0, "total": 0, "false_negatives": 0},
        "high_pitch_female_proxy": {"correct": 0, "total": 0, "false_negatives": 0},
        "soft_timbre": {"correct": 0, "total": 0, "false_positives": 0, "false_negatives": 0},
        "powerful_timbre": {"correct": 0, "total": 0, "false_positives": 0, "false_negatives": 0}
    }

    pitch_map_reverse = {0: "low_pitch_male_proxy", 1: "high_pitch_female_proxy"}

    print("Running Fairness Audit...")
    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader):
            waveforms = waveforms.to(DEVICE)
            targets = labels['is_vocal'].to(DEVICE)
            pitch_classes = labels['pitch_class'].to(DEVICE)
            is_soft = labels['is_soft'].to(DEVICE)
            is_powerful = labels['is_powerful'].to(DEVICE)

            # Generate spectrograms and predictions
            mels = amplitude_to_db(mel_transform(waveforms))
            outputs = model(mels).squeeze(1)
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            # Tally metrics
            for i in range(len(targets)):
                pred = predictions[i].item()
                actual = targets[i].item()
                p_class = pitch_classes[i].item()
                soft = is_soft[i].item()
                heavy = is_powerful[i].item()

                is_correct = (pred == actual)

                # Overall
                results["overall"]["total"] += 1
                if is_correct: results["overall"]["correct"] += 1

                # Pitch / Gender Proxy
                if actual == 1.0 and p_class in pitch_map_reverse:
                    group = pitch_map_reverse[p_class]
                    results[group]["total"] += 1
                    if is_correct:
                        results[group]["correct"] += 1
                    else:
                        results[group]["false_negatives"] += 1

                # Timbre Groups
                if soft == 1.0:
                    results["soft_timbre"]["total"] += 1
                    if is_correct: results["soft_timbre"]["correct"] += 1
                    elif pred == 1.0 and actual == 0.0: results["soft_timbre"]["false_positives"] += 1
                    elif pred == 0.0 and actual == 1.0: results["soft_timbre"]["false_negatives"] += 1

                if heavy == 1.0:
                    results["powerful_timbre"]["total"] += 1
                    if is_correct: results["powerful_timbre"]["correct"] += 1
                    elif pred == 1.0 and actual == 0.0: results["powerful_timbre"]["false_positives"] += 1
                    elif pred == 0.0 and actual == 1.0: results["powerful_timbre"]["false_negatives"] += 1

    # Calculate Percentages
    final_report = {}
    for group, data in results.items():
        if data["total"] > 0:
            final_report[group] = {
                "accuracy": round(data["correct"] / data["total"], 4),
                "total_samples": data["total"]
            }
            if "false_negatives" in data:
                final_report[group]["false_negative_rate"] = round(data["false_negatives"] / data["total"], 4)
            if "false_positives" in data:
                final_report[group]["false_positive_rate"] = round(data["false_positives"] / data["total"], 4)

    # Output
    print("\n--- Fairness Audit Results ---")
    print(json.dumps(final_report, indent=4))

    with open(AUDIT_RESULTS_FILE, 'w') as f:
        json.dump(final_report, f, indent=4)
    print(f"\nDetailed report saved to {AUDIT_RESULTS_FILE}")

if __name__ == "__main__":
    evaluate_model()
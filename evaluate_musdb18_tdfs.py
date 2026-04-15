"""
Filename: evaluate_musdb18_tdfs.py
Version: 1.0.0
Description: Performs cross-dataset validation. Audits the TDFS model (trained on 
             Jamendo) against the newly synthesized, unseen MUSDB18 dataset to 
             verify zero-shot generalization and sustained fairness metrics.
Hardware: GPU
"""

from google.colab import drive
drive.mount('/content/drive')

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*argument is set as true but no accelerator is found.*")
warnings.filterwarnings("ignore", message="This DataLoader will create.*worker processes in total")


import os
import json
import subprocess
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from tqdm import tqdm

# ==========================================
# 1. AUTO-RESTORE LOCAL AUDIO
# ==========================================
def ensure_local_audio():
    ZIP_PATH = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_wav_24k.zip'
    LOCAL_DIR = '/content/musdb18_local/wav_24k'
    
    if not os.path.exists(LOCAL_DIR) or len(os.listdir(LOCAL_DIR)) < 300:
        print("Local audio missing or incomplete. Restoring from Drive...")
        os.makedirs(LOCAL_DIR, exist_ok=True)
        subprocess.run(['unzip', '-q', '-o', '-j', ZIP_PATH, '-d', LOCAL_DIR])
        print("Audio restored successfully!")
    else:
        print("Local audio found. Skipping unzip.")

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
# 3. MUSDB18 DATASET
# ==========================================
class MUSDB18FairnessDataset(Dataset):
    def __init__(self, manifest_path, segment_length=48000):
        self.segment_length = segment_length
        self.df = pd.read_csv(manifest_path)
        self.pitch_map = {'Low-Pitch (Male Proxy)': 0, 'High-Pitch (Female Proxy)': 1, 'Non-Vocal': 2, 'unknown': 3}

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['path'] # Hardcoded local path from synthesis
        
        waveform, _ = torchaudio.load(wav_path)
        if waveform.shape[1] > self.segment_length:
            start = (waveform.shape[1] - self.segment_length) // 2
            waveform = waveform[:, start : start + self.segment_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_length - waveform.shape[1]))
        
        labels = {
            "is_vocal": torch.tensor(1.0 if row['is_vocal'] else 0.0, dtype=torch.float32),
            "pitch_class": torch.tensor(self.pitch_map.get(row['pitch_class'], 3), dtype=torch.long),
            "is_soft": torch.tensor(1.0 if row['is_soft_timbre'] else 0.0, dtype=torch.float32),
            "is_powerful": torch.tensor(1.0 if row['is_powerful_timbre'] else 0.0, dtype=torch.float32)
        }
        return waveform, labels

# ==========================================
# 4. CROSS-DATASET EVALUATION
# ==========================================
def evaluate_musdb18():
    ensure_local_audio()
    
    MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
    # Loading the weights trained on Jamendo!
    MODEL_PATH = '/content/drive/MyDrive/datasets/Jamendo/tdfs_final_v2.pth' 
    AUDIT_RESULTS_FILE = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_tdfs_cross_validation.json'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning Cross-Dataset Validation on device: {DEVICE}")
    
    model = TDFS().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = MUSDB18FairnessDataset(MANIFEST)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, f_min=0.0, f_max=12000.0, n_mels=80).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    results = {
        "overall": {"correct": 0, "total": 0},
        "low_pitch_male_proxy": {"correct": 0, "total": 0, "false_negatives": 0},
        "high_pitch_female_proxy": {"correct": 0, "total": 0, "false_negatives": 0},
        "soft_timbre": {"correct": 0, "total": 0, "false_positives": 0, "false_negatives": 0},
        "powerful_timbre": {"correct": 0, "total": 0, "false_positives": 0, "false_negatives": 0}
    }
    
    pitch_map_reverse = {0: "low_pitch_male_proxy", 1: "high_pitch_female_proxy"}

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Auditing MUSDB18 Batches"):
            waveforms = waveforms.to(DEVICE)
            targets = labels['is_vocal'].to(DEVICE)
            pitch_classes = labels['pitch_class'].to(DEVICE)
            is_soft = labels['is_soft'].to(DEVICE)
            is_powerful = labels['is_powerful'].to(DEVICE)

            mels = amplitude_to_db(mel_transform(waveforms))
            vocal_outputs, _ = model(mels, alpha=0.0) 
            predictions = (torch.sigmoid(vocal_outputs.squeeze(1)) > 0.5).float()

            for i in range(len(targets)):
                pred, actual = predictions[i].item(), targets[i].item()
                p_class, soft, heavy = pitch_classes[i].item(), is_soft[i].item(), is_powerful[i].item()
                is_correct = (pred == actual)

                results["overall"]["total"] += 1
                if is_correct: results["overall"]["correct"] += 1

                if actual == 1.0 and p_class in pitch_map_reverse:
                    group = pitch_map_reverse[p_class]
                    results[group]["total"] += 1
                    if is_correct: results[group]["correct"] += 1
                    else: results[group]["false_negatives"] += 1

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

    final_report = {}
    for group, data in results.items():
        if data["total"] > 0:
            final_report[group] = {"accuracy": round(data["correct"] / data["total"], 4), "total_samples": data["total"]}
            if "false_negatives" in data: final_report[group]["false_negative_rate"] = round(data["false_negatives"] / data["total"], 4)
            if "false_positives" in data: final_report[group]["false_positive_rate"] = round(data["false_positives"] / data["total"], 4)

    print("\n--- TDFS MUSDB18 Cross-Validation Results ---")
    print(json.dumps(final_report, indent=4))
    with open(AUDIT_RESULTS_FILE, 'w') as f: json.dump(final_report, f, indent=4)
    print(f"\nDetailed report saved to {AUDIT_RESULTS_FILE}")
    os.sync()

if __name__ == "__main__":
    evaluate_musdb18()
"""
Filename: static_alpha_ablation.py
Version: 1.0.0
Description: Rapid ablation study for IEEE SPL. Trains the TDFS architecture 
             for 5 epochs with a static GRL alpha = 1.0 from the very first step. 
             Demonstrates feature extractor collapse when Adversarial Alpha 
             Annealing is omitted.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from tqdm import tqdm

# ==========================================
# 1. ARCHITECTURE
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

    def forward(self, x, alpha):
        features = self.extractor(x)
        return self.vocal_classifier(features), self.timbre_adversary(GradientReversalLayer.apply(features, alpha))

# ==========================================
# 2. DATASET LOADER
# ==========================================
class MUSDB18AblationDataset(Dataset):
    def __init__(self, manifest_path, segment_length=48000):
        self.segment_length = segment_length
        self.df = pd.read_csv(manifest_path)
        # 0: Soft, 1: Powerful, 2: Other
        def get_timbre_label(row):
            if row['is_soft_timbre']: return 0
            if row['is_powerful_timbre']: return 1
            return 2
        self.df['timbre_label'] = self.df.apply(get_timbre_label, axis=1)

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
            "is_vocal": torch.tensor([1.0 if row['is_vocal'] else 0.0], dtype=torch.float32),
            "timbre_label": torch.tensor(row['timbre_label'], dtype=torch.long)
        }
        return waveform, labels

# ==========================================
# 3. STATIC ALPHA TRAINING LOOP
# ==========================================
def run_static_alpha_ablation():
    MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
    OUTPUT_FILE = '/content/drive/MyDrive/datasets/MUSDB18/static_alpha_ablation_results.json'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning Static Alpha Ablation on: {DEVICE}")
    
    model = TDFS().to(DEVICE)
    dataset = MUSDB18AblationDataset(MANIFEST)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_vocal = nn.BCEWithLogitsLoss()
    criterion_timbre = nn.CrossEntropyLoss()

    epochs = 5
    # THE ABLATION: Hardcoding alpha to 1.0 from the start
    STATIC_ALPHA = 1.0 

    final_accuracy = 0.0
    final_vocal_loss = 0.0

    for epoch in range(epochs):
        model.train()
        correct_vocals, total_vocals = 0, 0
        running_vocal_loss = 0.0
        
        epoch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Alpha={STATIC_ALPHA}]")
        for waveforms, labels in epoch_iter:
            waveforms = waveforms.to(DEVICE)
            targets_vocal = labels['is_vocal'].to(DEVICE)
            targets_timbre = labels['timbre_label'].to(DEVICE)

            mels = amplitude_to_db(mel_transform(waveforms))
            
            optimizer.zero_grad()
            
            # Forward pass with full adversarial strength
            vocal_outputs, timbre_outputs = model(mels, alpha=STATIC_ALPHA)
            
            loss_vocal = criterion_vocal(vocal_outputs, targets_vocal)
            loss_timbre = criterion_timbre(timbre_outputs, targets_timbre)
            
            # The network tries to minimize vocal error AND maximize timbre error
            loss = loss_vocal + loss_timbre 
            loss.backward()
            optimizer.step()

            # Tracking metrics
            running_vocal_loss += loss_vocal.item()
            preds = (torch.sigmoid(vocal_outputs) > 0.5).float()
            correct_vocals += (preds == targets_vocal).sum().item()
            total_vocals += targets_vocal.size(0)

            epoch_iter.set_postfix(v_loss=loss_vocal.item(), acc=correct_vocals/total_vocals)

        final_accuracy = correct_vocals / total_vocals
        final_vocal_loss = running_vocal_loss / len(dataloader)

    results = {
        "ablation_type": "Static Alpha (No Annealing)",
        "alpha_value": STATIC_ALPHA,
        "epochs_trained": epochs,
        "final_vocal_accuracy": round(final_accuracy, 4),
        "final_vocal_loss": round(final_vocal_loss, 4),
        "conclusion": "Model expected to fail convergence (accuracy near 0.50) due to adversarial gradient overwhelming the feature extractor before fundamental acoustic features are learned."
    }

    print("\n--- Ablation Results ---")
    print(json.dumps(results, indent=4))
    with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_static_alpha_ablation()
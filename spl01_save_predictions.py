"""
Filename: spl01_save_predictions.py
Version: 2.0.0
Description: Saves per-segment sigmoid scores, binary predictions, ground truth,
             timbre group, and pitch cohort for three models on the MUSDB18
             zero-shot test set:
               - Vanilla CNN (trained on Jamendo, no preprocessing)
               - TDFS (trained on Jamendo, adversarial GRL)
               - HPSS+Vanilla (Vanilla CNN trained on Jamendo, HPSS at inference)
             All three operate in true zero-shot mode: trained on Jamendo,
             evaluated on MUSDB18. This fixes the unfair training condition
             in the original hpss_baseline.py.
             Outputs a single CSV per model to the project directory.
             Checkpoint: CSV is written after every batch with os.sync().

Changelog:
  - v1.0.0: Initial release.
  - v2.0.0 (2026-06-28): Fixed spectrogram mismatch between training (librosa)
                         and inference (torchaudio). All spectrograms now
                         computed with librosa.power_to_db(ref=np.max) to
                         match the precompute_spectrograms.py training cache.
                         Removed torchaudio MelSpectrogram from inference loop.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import json
import zipfile
import shutil
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# PROJECT CONFIG
# ==========================================
PROJECT_DIR = '/content/drive/MyDrive/paper/DecouplingTimbre'
os.makedirs(PROJECT_DIR, exist_ok=True)

MANIFEST_PATH   = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
VANILLA_MODEL   = '/content/drive/MyDrive/datasets/Jamendo/vanilla_svd_final.pth'
TDFS_MODEL      = '/content/drive/MyDrive/datasets/Jamendo/tdfs_final_v2.pth'
MUSDB18_ZIP     = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_wav_24k.zip'
LOCAL_WAV_DIR   = '/content/musdb18_local/wav_24k'

OUT_VANILLA     = os.path.join(PROJECT_DIR, 'predictions_vanilla.csv')
OUT_TDFS        = os.path.join(PROJECT_DIR, 'predictions_tdfs.csv')
OUT_HPSS        = os.path.join(PROJECT_DIR, 'predictions_hpss.csv')

SEGMENT_LENGTH  = 48000   # 2 seconds at 24 kHz
BATCH_SIZE      = 32
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Librosa mel parameters — must match precompute_spectrograms.py exactly
MEL_SR     = 24000
MEL_N_FFT  = 1024
MEL_HOP    = 256
MEL_N_MELS = 80

# ==========================================
# 1. LOCAL AUDIO RESTORE
# ==========================================
def ensure_local_audio():
    if os.path.exists(LOCAL_WAV_DIR) and len(os.listdir(LOCAL_WAV_DIR)) >= 300:
        print(f"Local audio ready ({len(os.listdir(LOCAL_WAV_DIR))} files).")
        return
    print("Restoring audio from Drive zip (native extraction)...")
    os.makedirs(LOCAL_WAV_DIR, exist_ok=True)
    with zipfile.ZipFile(MUSDB18_ZIP, 'r') as zf:
        for member in zf.namelist():
            fname = os.path.basename(member)
            if not fname:
                continue
            with zf.open(member) as src, open(os.path.join(LOCAL_WAV_DIR, fname), 'wb') as dst:
                shutil.copyfileobj(src, dst)
    print(f"Extraction complete: {len(os.listdir(LOCAL_WAV_DIR))} files.")

# ==========================================
# 2. ARCHITECTURES
# ==========================================
class VanillaSVD(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

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
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.vocal_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
        self.timbre_adversary = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3)
        )
    def forward(self, x, alpha=0.0):
        features = self.extractor(x)
        return self.vocal_classifier(features), \
               self.timbre_adversary(GradientReversalLayer.apply(features, alpha))

# ==========================================
# 3. DATASET
# ==========================================
class MUSDB18PredDataset(Dataset):
    """
    Loads MUSDB18 segments. apply_hpss flag controls whether the harmonic
    component is extracted before returning the waveform.
    """
    def __init__(self, manifest_path, segment_length=48000, apply_hpss=False):
        self.segment_length = segment_length
        self.apply_hpss = apply_hpss
        self.df = pd.read_csv(manifest_path)
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
        wav_path = row['path']

        waveform, _ = torchaudio.load(wav_path)
        # Deterministic centre crop for reproducibility
        if waveform.shape[1] > self.segment_length:
            start = (waveform.shape[1] - self.segment_length) // 2
            waveform = waveform[:, start: start + self.segment_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_length - waveform.shape[1])
            )

        if self.apply_hpss:
            wave_np = waveform.numpy()[0]
            harmonic_np, _ = librosa.effects.hpss(wave_np, margin=1.2)
            wave_np = harmonic_np
        else:
            wave_np = waveform.numpy()[0]

        # Compute spectrogram with librosa to match training representation
        # (precompute_spectrograms.py uses librosa.power_to_db with ref=np.max)
        S    = librosa.feature.melspectrogram(
            y=wave_np, sr=MEL_SR, n_fft=MEL_N_FFT,
            hop_length=MEL_HOP, n_mels=MEL_N_MELS
        )
        spec = torch.tensor(
            librosa.power_to_db(S, ref=np.max).astype(np.float32)
        ).unsqueeze(0)  # [1, 80, T]

        meta = {
            'idx': idx,
            'is_vocal': float(row['is_vocal']),
            'pitch_class': int(self.pitch_map.get(str(row['pitch_class']), 3)),
            'is_soft': float(row['is_soft_timbre']),
            'is_powerful': float(row['is_powerful_timbre']),
        }
        return spec, meta

# ==========================================
# 4. INFERENCE RUNNER
# ==========================================
def run_inference(model_type, output_csv, manifest_path):
    """
    model_type: 'vanilla' | 'tdfs' | 'hpss'
    Appends rows to output_csv batch-by-batch with os.sync() for crash safety.
    Resumes automatically if output_csv already exists.
    """
    apply_hpss = (model_type == 'hpss')

    # Load model
    if model_type in ('vanilla', 'hpss'):
        model = VanillaSVD().to(DEVICE)
        model.load_state_dict(torch.load(VANILLA_MODEL, map_location=DEVICE))
    else:
        model = TDFS().to(DEVICE)
        model.load_state_dict(torch.load(TDFS_MODEL, map_location=DEVICE))
    model.eval()

    dataset    = MUSDB18PredDataset(manifest_path, apply_hpss=apply_hpss)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Resume: find which indices already processed
    done_indices = set()
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        done_indices = set(existing['idx'].tolist())
        print(f"[{model_type}] Resuming — {len(done_indices)} segments already saved.")

    rows = []

    print(f"\n{'='*60}")
    print(f"Running inference: {model_type.upper()}")
    print(f"{'='*60}")

    with torch.no_grad():
        for specs, meta in tqdm(dataloader, desc=f"{model_type}"):
            # Skip batches fully processed
            batch_indices = meta['idx'].tolist()
            if all(i in done_indices for i in batch_indices):
                continue

            # specs are precomputed librosa spectrograms [B, 1, 80, T]
            specs = specs.to(DEVICE)

            if model_type == 'tdfs':
                vocal_out, _ = model(specs, alpha=0.0)
            else:
                vocal_out = model(specs)

            probs = torch.sigmoid(vocal_out.squeeze(1)).cpu().numpy()
            preds = (probs > 0.5).astype(float)

            for i, idx in enumerate(batch_indices):
                if idx in done_indices:
                    continue
                rows.append({
                    'idx':          idx,
                    'model':        model_type,
                    'sigmoid_score': float(probs[i]),
                    'prediction':   float(preds[i]),
                    'ground_truth': float(meta['is_vocal'][i]),
                    'pitch_class':  int(meta['pitch_class'][i]),
                    'is_soft':      float(meta['is_soft'][i]),
                    'is_powerful':  float(meta['is_powerful'][i]),
                })

            # Write checkpoint every batch
            if rows:
                batch_df = pd.DataFrame(rows)
                if os.path.exists(output_csv):
                    batch_df.to_csv(output_csv, mode='a', header=False, index=False)
                else:
                    batch_df.to_csv(output_csv, index=False)
                os.sync()
                done_indices.update([r['idx'] for r in rows])
                rows = []

    total = len(pd.read_csv(output_csv))
    print(f"[{model_type}] Done. {total} segments saved to {output_csv}")

# ==========================================
# 5. MAIN
# ==========================================
def main():
    print(f"Device: {DEVICE}")
    ensure_local_audio()

    # Patch manifest paths to local WAV dir
    df = pd.read_csv(MANIFEST_PATH)
    df['path'] = df['path'].apply(
        lambda p: os.path.join(LOCAL_WAV_DIR, os.path.basename(p))
    )
    patched_manifest = os.path.join(PROJECT_DIR, 'musdb18_manifest_patched.csv')
    df.to_csv(patched_manifest, index=False)
    os.sync()

    manifest_path_local = patched_manifest

    run_inference('vanilla', OUT_VANILLA, manifest_path_local)
    run_inference('tdfs',    OUT_TDFS,    manifest_path_local)
    run_inference('hpss',    OUT_HPSS,    manifest_path_local)

    # Write a quick summary
    summary = {}
    for label, path in [('vanilla', OUT_VANILLA), ('tdfs', OUT_TDFS), ('hpss', OUT_HPSS)]:
        df = pd.read_csv(path)
        overall_acc = (df['prediction'] == df['ground_truth']).mean()
        summary[label] = {
            'total_segments': len(df),
            'overall_accuracy': round(float(overall_acc), 4)
        }
    summary_path = os.path.join(PROJECT_DIR, 'spl01_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    os.sync()
    print(f"\nSummary saved to {summary_path}")
    print(json.dumps(summary, indent=4))

if __name__ == '__main__':
    main()
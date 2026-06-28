"""
Filename: spl03_probe_with_split.py
Version: 1.0.0
Description: Reruns the sub-band latent linear probe with a proper 70/30
             stratified train/test split, fixing the train-on-test-data issue
             in the original frequency_band_probe.py.

             Also adds:
               - Per-band bootstrap 95% CIs on test-set accuracy
               - Cross-validation (5-fold) as a secondary validation
               - Per-band confusion matrices
               - Saves results JSON and a bar chart figure

             Architecture: intercepts TDFS conv_blocks output (2D feature map,
             shape [B, 64, 20, T]) and slices along the Mel frequency axis:
               Low  band: bins 0-5   (bass/kick)
               Mid  band: bins 6-13  (vocal formants + harmonic instruments)
               High band: bins 14-19 (cymbals/air)

             Input: TDFS model from Jamendo training, MUSDB18 manifest
             
             GPU Required
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import json
import zipfile
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

if not torch.cuda.is_available():
    print("[ERROR] GPU not detected. Enable GPU.")
    sys.exit(1)
device = torch.device("cuda")
print("CUDA available: True. Proceeding...")

# Keep-alive: prevents Colab idle disconnect during long inference
import time
def keep_alive():
    while True:
        time.sleep(60)
import threading
threading.Thread(target=keep_alive, daemon=True).start()

# ==========================================
# PROJECT CONFIG
# ==========================================
PROJECT_DIR   = '/content/drive/MyDrive/paper/DecouplingTimbre'
MANIFEST_PATH = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
TDFS_MODEL    = '/content/drive/MyDrive/datasets/Jamendo/tdfs_final_v2.pth'
MUSDB18_ZIP   = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_wav_24k.zip'
LOCAL_WAV_DIR = '/content/musdb18_local/wav_24k'
RESULTS_JSON  = os.path.join(PROJECT_DIR, 'spl03_probe_results.json')
BAR_PNG       = os.path.join(PROJECT_DIR, 'spl03_subband_probe.png')
LATENTS_CACHE = os.path.join(PROJECT_DIR, 'spl03_latents_cache.npz')

SEGMENT_LENGTH = 48000
BATCH_SIZE     = 32
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_BOOTSTRAP    = 1000
RNG_SEED       = 42
rng            = np.random.default_rng(RNG_SEED)

MEL_KWARGS = dict(
    sample_rate=24000, n_fft=1024, win_length=1024,
    hop_length=256, f_min=0.0, f_max=12000.0, n_mels=80
)

# ==========================================
# 1. LOCAL AUDIO RESTORE
# ==========================================
def ensure_local_audio():
    if os.path.exists(LOCAL_WAV_DIR) and len(os.listdir(LOCAL_WAV_DIR)) >= 300:
        print(f"Local audio ready ({len(os.listdir(LOCAL_WAV_DIR))} files).")
        return
    print("Restoring audio from Drive zip...")
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
# 2. TDFS WITH INTERCEPTABLE CONV BLOCKS
# ==========================================
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class TDFSWithIntercept(nn.Module):
    """
    TDFS with conv_blocks separated from pooling so we can intercept
    the 2D spatial feature map before AdaptiveAvgPool2d.
    After two MaxPool2d(2,2), an 80-bin Mel input becomes 20 bins.
    We slice these 20 frequency bins into three sub-bands.
    """
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.pool_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.vocal_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
        self.timbre_adversary = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3)
        )

    def get_subband_vectors(self, x):
        """Returns (vec_low, vec_mid, vec_high) — each shape [B, 64]."""
        fm = self.conv_blocks(x)            # [B, 64, 20, T]
        low  = self.pool_flatten(fm[:, :, 0:6,  :])   # bins 0-5
        mid  = self.pool_flatten(fm[:, :, 6:14, :])   # bins 6-13
        high = self.pool_flatten(fm[:, :, 14:20, :])  # bins 14-19
        return low, mid, high

# ==========================================
# 3. DATASET
# ==========================================
class MUSDB18ProbeDataset(Dataset):
    def __init__(self, manifest_path, segment_length=48000):
        self.segment_length = segment_length
        df = pd.read_csv(manifest_path)
        # Patch paths to local dir
        df['path'] = df['path'].apply(
            lambda p: os.path.join(LOCAL_WAV_DIR, os.path.basename(p))
        )
        # Keep only timbre-labelled segments
        self.df = df[(df['is_soft_timbre'] == 1) | (df['is_powerful_timbre'] == 1)].reset_index(drop=True)
        print(f"Probe dataset: {len(self.df)} timbre-labelled segments "
              f"({(self.df['is_powerful_timbre'] == 1).sum()} powerful, "
              f"{(self.df['is_soft_timbre'] == 1).sum()} soft)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, _ = torchaudio.load(row['path'])
        if waveform.shape[1] > self.segment_length:
            start = (waveform.shape[1] - self.segment_length) // 2
            waveform = waveform[:, start: start + self.segment_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_length - waveform.shape[1])
            )
        label = 1 if row['is_powerful_timbre'] == 1 else 0
        return waveform, label

# ==========================================
# 4. EXTRACT LATENTS (CACHED)
# ==========================================
def extract_latents(model, dataset):
    if os.path.exists(LATENTS_CACHE):
        print("Loading cached latents...")
        cache = np.load(LATENTS_CACHE)
        return cache['low'], cache['mid'], cache['high'], cache['labels']

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    mel_transform   = torchaudio.transforms.MelSpectrogram(**MEL_KWARGS).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    lows, mids, highs, labels = [], [], [], []

    model.eval()
    with torch.no_grad():
        for waveforms, batch_labels in tqdm(dataloader, desc="Extracting sub-band latents"):
            waveforms = waveforms.to(DEVICE)
            mels = amplitude_to_db(mel_transform(waveforms))
            vec_low, vec_mid, vec_high = model.get_subband_vectors(mels)
            lows.extend(vec_low.cpu().numpy())
            mids.extend(vec_mid.cpu().numpy())
            highs.extend(vec_high.cpu().numpy())
            labels.extend(batch_labels.numpy() if isinstance(batch_labels, torch.Tensor)
                          else batch_labels)

    lows   = np.array(lows)
    mids   = np.array(mids)
    highs  = np.array(highs)
    labels = np.array(labels)

    np.savez(LATENTS_CACHE, low=lows, mid=mids, high=highs, labels=labels)
    os.sync()
    print(f"Latents cached to {LATENTS_CACHE}")
    return lows, mids, highs, labels

# ==========================================
# 5. PROBE EVALUATION
# ==========================================
def evaluate_probe(X_tr, X_te, y_tr, y_te, band_name):
    probe = LogisticRegression(max_iter=2000, random_state=RNG_SEED)
    probe.fit(X_tr, y_tr)
    preds = probe.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    cm    = confusion_matrix(y_te, preds).tolist()
    return acc, cm, probe

def bootstrap_probe_ci(X_te, y_te, probe, n=N_BOOTSTRAP):
    n_samples = len(y_te)
    boot_accs = []
    for _ in range(n):
        idx  = rng.integers(0, n_samples, size=n_samples)
        preds = probe.predict(X_te[idx])
        boot_accs.append(accuracy_score(y_te[idx], preds))
    return round(float(np.percentile(boot_accs, 2.5)), 4), \
           round(float(np.percentile(boot_accs, 97.5)), 4)

def cross_validate_probe(X, y, n_folds=5):
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    accs  = []
    for tr_idx, te_idx in skf.split(X, y):
        probe = LogisticRegression(max_iter=2000, random_state=RNG_SEED)
        probe.fit(X[tr_idx], y[tr_idx])
        accs.append(accuracy_score(y[te_idx], probe.predict(X[te_idx])))
    return round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)

# ==========================================
# 6. BAR CHART
# ==========================================
def plot_bar_chart(band_results):
    bands  = list(band_results.keys())
    accs   = [band_results[b]['test_accuracy'] for b in bands]
    lo_err = [band_results[b]['test_accuracy'] - band_results[b]['ci_95'][0] for b in bands]
    hi_err = [band_results[b]['ci_95'][1] - band_results[b]['test_accuracy'] for b in bands]

    colors = ['#3498db', '#e74c3c', '#95a5a6']
    labels = ['Low Band\n(Bins 0-5\nBass/Kick)',
              'Mid Band\n(Bins 6-13\nVocal/Guitar)',
              'High Band\n(Bins 14-19\nCymbals/Air)']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, accs, color=colors, width=0.5,
                  yerr=[lo_err, hi_err], capsize=6, error_kw={'linewidth': 1.5})
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2, label='Random chance (0.50)')
    ax.axhline(1.0, color='gray',  linestyle=':',  linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Linear Probe Test Accuracy')
    ax.set_title('Sub-Band Latent Probing: Timbre Decodability by Mel Frequency Band')
    ax.legend(fontsize=9)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(BAR_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    os.sync()
    print(f"Bar chart saved to {BAR_PNG}")

# ==========================================
# 7. MAIN
# ==========================================
def main():
    print(f"Device: {DEVICE}")
    ensure_local_audio()

    # Load model with strict=False since we reorganised Sequential blocks
    model = TDFSWithIntercept().to(DEVICE)
    state = torch.load(TDFS_MODEL, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("TDFS model loaded.")

    dataset = MUSDB18ProbeDataset(MANIFEST_PATH)
    lows, mids, highs, labels = extract_latents(model, dataset)

    print(f"\nTotal probe samples: {len(labels)} "
          f"({labels.sum()} powerful, {(1-labels).sum()} soft)")

    # 70/30 stratified split
    band_data = {'low': lows, 'mid': mids, 'high': highs}
    splits = {}
    for band_name, X in band_data.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.30, stratify=labels, random_state=RNG_SEED
        )
        splits[band_name] = (X_tr, X_te, y_tr, y_te)

    print("\n" + "="*60)
    print("Sub-band probe results (70/30 stratified split)")
    print("="*60)

    band_results = {}
    for band_name, (X_tr, X_te, y_tr, y_te) in splits.items():
        acc, cm, probe = evaluate_probe(X_tr, X_te, y_tr, y_te, band_name)
        ci_lo, ci_hi   = bootstrap_probe_ci(X_te, y_te, probe)
        cv_mean, cv_std = cross_validate_probe(
            np.concatenate([X_tr, X_te]),
            np.concatenate([y_tr, y_te])
        )
        band_results[band_name] = {
            'test_accuracy':      round(float(acc), 4),
            'ci_95':              [ci_lo, ci_hi],
            'cv_5fold_mean':      cv_mean,
            'cv_5fold_std':       cv_std,
            'confusion_matrix':   cm,
            'n_train':            int(len(y_tr)),
            'n_test':             int(len(y_te)),
        }
        print(f"\n  [{band_name.upper()} BAND]")
        print(f"    Test accuracy : {acc:.4f}  (95% CI: [{ci_lo}, {ci_hi}])")
        print(f"    5-fold CV     : {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    Confusion matrix: {cm}")

    plot_bar_chart(band_results)

    results = {
        'band_results': band_results,
        'random_baseline': 0.5,
        'split': '70/30 stratified',
        'notes': {
            'low_band':  'Mel bins 0-5 (~0-300 Hz): bass, kick drum',
            'mid_band':  'Mel bins 6-13 (~300-3000 Hz): vocal formants, harmonic instruments',
            'high_band': 'Mel bins 14-19 (~3000-12000 Hz): cymbals, air',
            'interpretation': (
                'Bands with test accuracy close to 0.50 are disentangled by the GRL. '
                'Bands > 0.80 contain persistent timbre leakage.'
            )
        }
    }

    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    os.sync()
    print(f"\nResults saved to {RESULTS_JSON}")

if __name__ == '__main__':
    main()

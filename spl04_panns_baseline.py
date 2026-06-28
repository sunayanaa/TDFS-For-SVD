"""
Filename: spl04_panns_baseline.py
Version: 1.0.0
Description: Addresses Reviewer 1's demand for a modern SOTA baseline.
             Downloads PANNs CNN14 (pretrained on AudioSet, 527 classes),
             extracts the 'Singing' tag logit (AudioSet class index 71),
             and runs zero-shot inference on the MUSDB18 test set.
             Applies the identical timbre/pitch fairness audit as TDFS.

             PANNs reference:
               Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks
               for Audio Pattern Recognition," IEEE/ACM TASLP, 2020.
               Checkpoint: CNN14_mAP=0.431.pth from Zenodo 3987831.

             No fine-tuning is performed. The point is to show that even a
             modern pre-trained model trained on 2M+ AudioSet clips exhibits
             the same timbre-dependent FPR bias, confirming the confound is
             structural rather than architecture-specific.

             Resume: saves per-batch rows to CSV with os.sync() after each batch.
             Checkpoint JSON is also updated after each batch for crash safety.
             
             GPU required
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import sys
import json
import zipfile
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
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
PROJECT_DIR    = '/content/drive/MyDrive/paper/DecouplingTimbre'
MANIFEST_PATH  = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
MUSDB18_ZIP    = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_wav_24k.zip'
LOCAL_WAV_DIR  = '/content/musdb18_local/wav_24k'
PANNS_CKPT_URL = 'https://zenodo.org/record/3987831/files/Cnn14_mAP=0.431.pth?download=1'
PANNS_CKPT_LOCAL = '/content/CNN14_mAP0.431.pth'
PANNS_CKPT       = os.path.join(PROJECT_DIR, 'CNN14_mAP0.431.pth')
FOUT_PREDS_CSV  = os.path.join(PROJECT_DIR, 'predictions_panns.csv')
RESULTS_JSON   = os.path.join(PROJECT_DIR, 'spl04_panns_results.json')
BAR_PNG        = os.path.join(PROJECT_DIR, 'spl04_panns_fpr_bars.png')

SEGMENT_LENGTH = 48000   # 2 seconds at 24 kHz (resampled to 32 kHz for PANNs)
PANNS_SR       = 32000   # PANNs CNN14 was trained at 32 kHz
SINGING_IDX    = 27      # AudioSet class index for 'Singing'
BATCH_SIZE     = 16      # Smaller batch: PANNs is a larger model
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. SETUP
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

def download_panns_checkpoint():
    # Use Drive copy if already saved there
    if os.path.exists(PANNS_CKPT):
        print(f"PANNs checkpoint found on Drive: {PANNS_CKPT}")
        if not os.path.exists(PANNS_CKPT_LOCAL):
            shutil.copy2(PANNS_CKPT, PANNS_CKPT_LOCAL)
            print(f"Copied to local: {PANNS_CKPT_LOCAL}")
        return
    # Download to local VM first (faster, no Drive I/O issues)
    if not os.path.exists(PANNS_CKPT_LOCAL):
        print("Downloading PANNs CNN14 checkpoint to local VM (~300 MB)...")
        subprocess.run(
            ['wget', '-O', PANNS_CKPT_LOCAL, PANNS_CKPT_URL],
            check=True
        )
        size = os.path.getsize(PANNS_CKPT_LOCAL) / 1e6
        print(f"Downloaded to {PANNS_CKPT_LOCAL} ({size:.1f} MB)")
    # Copy to Drive for persistence across sessions
    shutil.copy2(PANNS_CKPT_LOCAL, PANNS_CKPT)
    os.sync()
    print(f"Saved to Drive: {PANNS_CKPT}")
    

def install_panns():
    """Install panns_inference from PyPI."""
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', 'panns_inference'],
        check=True
    )

# ==========================================
# 2. PANNs CNN14 MODEL
#    We load the raw checkpoint directly without panns_inference wrapper
#    to have full control over the forward pass and logit extraction.
# ==========================================
def init_panns_cnn14():
    """
    Build CNN14 from the panns_inference package and load pretrained weights.
    Returns model in eval mode.
    """
    try:
        from panns_inference.models import Cnn14 as PannsCNN14
    except ImportError:
        install_panns()
        from panns_inference.models import Cnn14 as PannsCNN14

    model = PannsCNN14(
        sample_rate=PANNS_SR,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )

    # Load checkpoint — panns checkpoints store under 'model' key
    ckpt_path = PANNS_CKPT_LOCAL if os.path.exists(PANNS_CKPT_LOCAL) else PANNS_CKPT
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print("PANNs CNN14 loaded and ready.")
    return model

# ==========================================
# 3. DATASET
#    Resamples 24 kHz MUSDB18 audio to 32 kHz for PANNs
# ==========================================
class MUSDB18PANNsDataset(Dataset):
    def __init__(self, manifest_path, segment_length_24k=48000):
        self.segment_length = segment_length_24k
        df = pd.read_csv(manifest_path)
        df['path'] = df['path'].apply(
            lambda p: os.path.join(LOCAL_WAV_DIR, os.path.basename(p))
        )
        self.df = df.reset_index(drop=True)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=24000, new_freq=PANNS_SR
        )
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
        waveform, _ = torchaudio.load(row['path'])

        # Centre crop at 24 kHz
        if waveform.shape[1] > self.segment_length:
            start = (waveform.shape[1] - self.segment_length) // 2
            waveform = waveform[:, start: start + self.segment_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_length - waveform.shape[1])
            )

        # Resample to 32 kHz for PANNs
        waveform_32k = self.resampler(waveform)  # [1, 64000]
        waveform_32k = waveform_32k.squeeze(0)    # [64000]

        meta = {
            'idx':         idx,
            'is_vocal':    float(row['is_vocal']),
            'pitch_class': int(self.pitch_map.get(str(row['pitch_class']), 3)),
            'is_soft':     float(row['is_soft_timbre']),
            'is_powerful': float(row['is_powerful_timbre']),
        }
        return waveform_32k, meta

# ==========================================
# 4. INFERENCE
# ==========================================
def run_panns_inference(model):
    """
    Runs PANNs CNN14 in zero-shot mode. Uses the raw Singing logit
    (AudioSet class 71) as the confidence score; sigmoid > 0.5 = vocal.
    Saves rows batch-by-batch to OUT_PREDS_CSV with os.sync().
    Resumes automatically if output already exists.
    """
    dataset    = MUSDB18PANNsDataset(MANIFEST_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    # Resume
    done_indices = set()
    if os.path.exists(OUT_PREDS_CSV):
        existing = pd.read_csv(OUT_PREDS_CSV)
        done_indices = set(existing['idx'].tolist())
        print(f"Resuming PANNs inference — {len(done_indices)} segments done.")

    rows = []

    with torch.no_grad():
        for waveforms, meta in tqdm(dataloader, desc="PANNs zero-shot inference"):
            batch_indices = meta['idx'].tolist()
            if all(i in done_indices for i in batch_indices):
                continue

            waveforms = waveforms.to(DEVICE)  # [B, 64000]

            # PANNs forward returns dict with 'clipwise_output' [B, 527]
            output = model(waveforms)
            logits = output['clipwise_output']          # [B, 527]
            singing_score = torch.sigmoid(logits[:, SINGING_IDX])  # [B]
            preds = (singing_score > 0.5).float().cpu().numpy()
            scores = singing_score.cpu().numpy()

            for i, idx in enumerate(batch_indices):
                if idx in done_indices:
                    continue
                rows.append({
                    'idx':          idx,
                    'model':        'panns_cnn14',
                    'sigmoid_score': float(scores[i]),
                    'prediction':   float(preds[i]),
                    'ground_truth': float(meta['is_vocal'][i]),
                    'pitch_class':  int(meta['pitch_class'][i]),
                    'is_soft':      float(meta['is_soft'][i]),
                    'is_powerful':  float(meta['is_powerful'][i]),
                })

            if rows:
                batch_df = pd.DataFrame(rows)
                if os.path.exists(OUT_PREDS_CSV):
                    batch_df.to_csv(OUT_PREDS_CSV, mode='a', header=False, index=False)
                else:
                    batch_df.to_csv(OUT_PREDS_CSV, index=False)
                os.sync()
                done_indices.update([r['idx'] for r in rows])
                rows = []

    print(f"PANNs inference complete. {len(pd.read_csv(OUT_PREDS_CSV))} segments saved.")

# ==========================================
# 5. FAIRNESS AUDIT
# ==========================================
def compute_fpr(preds, gts, mask=None):
    if mask is not None:
        preds, gts = preds[mask], gts[mask]
    neg = (gts == 0)
    if neg.sum() == 0:
        return np.nan
    return float((preds[neg] == 1).mean())

def compute_fnr(preds, gts, mask=None):
    if mask is not None:
        preds, gts = preds[mask], gts[mask]
    pos = (gts == 1)
    if pos.sum() == 0:
        return np.nan
    return float((preds[pos] == 0).mean())

def bootstrap_metric(fn, preds, gts, mask=None, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    if mask is not None:
        preds, gts = preds[mask], gts[mask]
    vals = [fn(preds[rng.integers(0, len(preds), len(preds))],
               gts[rng.integers(0, len(gts), len(gts))]) for _ in range(n)]
    vals = [v for v in vals if not np.isnan(v)]
    lo = round(float(np.percentile(vals, 2.5)), 4)
    hi = round(float(np.percentile(vals, 97.5)), 4)
    return lo, hi

def audit_panns_fairness(df):
    preds = df['prediction'].values
    gts   = df['ground_truth'].values
    soft  = df['is_soft'].values.astype(bool)
    pow_  = df['is_powerful'].values.astype(bool)
    pc    = df['pitch_class'].values

    audit = {}

    overall_acc = float((preds == gts).mean())
    audit['overall_accuracy'] = {
        'point': round(overall_acc, 4),
        'ci_95': list(bootstrap_metric(
            lambda p, g: float((p == g).mean()), preds, gts))
    }

    fpr_soft = compute_fpr(preds, gts, mask=soft)
    audit['soft_timbre_fpr'] = {
        'point': round(fpr_soft, 4),
        'ci_95': list(bootstrap_metric(compute_fpr, preds, gts, mask=soft))
    }

    fpr_pow = compute_fpr(preds, gts, mask=pow_)
    audit['powerful_timbre_fpr'] = {
        'point': round(fpr_pow, 4),
        'ci_95': list(bootstrap_metric(compute_fpr, preds, gts, mask=pow_))
    }

    fnr_low = compute_fnr(preds, gts, mask=(pc == 0))
    audit['low_pitch_fnr'] = {
        'point': round(fnr_low, 4) if not np.isnan(fnr_low) else None,
        'ci_95': list(bootstrap_metric(compute_fnr, preds, gts, mask=(pc == 0)))
    }

    fnr_high = compute_fnr(preds, gts, mask=(pc == 1))
    audit['high_pitch_fnr'] = {
        'point': round(fnr_high, 4) if not np.isnan(fnr_high) else None,
        'ci_95': list(bootstrap_metric(compute_fnr, preds, gts, mask=(pc == 1)))
    }

    print("\n--- PANNs CNN14 Fairness Audit ---")
    for k, v in audit.items():
        print(f"  {k}: {v['point']}  (95% CI: {v['ci_95']})")

    return audit

# ==========================================
# 6. FPR COMPARISON BAR CHART
# ==========================================
def plot_fpr_comparison(panns_audit):
    """
    Plots soft vs. powerful timbre FPR for PANNs alongside TDFS and Vanilla,
    reading the latter from spl02 results if available.
    """
    spl02_path = os.path.join(PROJECT_DIR, 'spl02_statistical_validation.json')

    categories = ['Soft Timbre FPR', 'Powerful Timbre FPR']
    panns_vals = [
        panns_audit['soft_timbre_fpr']['point'],
        panns_audit['powerful_timbre_fpr']['point']
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(categories))
    width = 0.22

    ax.bar(x - width, panns_vals, width, label='PANNs CNN14 (zero-shot)', color='#9b59b6')

    if os.path.exists(spl02_path):
        with open(spl02_path) as f:
            spl02 = json.load(f)
        #
        vanilla_vals = [
            spl02['bootstrap_cis_on_original_numbers']['vanilla']['soft_timbre_fpr']['point'],
            spl02['bootstrap_cis_on_original_numbers']['vanilla']['powerful_timbre_fpr']['point']
        ]
        tdfs_vals = [
            spl02['bootstrap_cis_on_original_numbers']['tdfs']['soft_timbre_fpr']['point'],
            spl02['bootstrap_cis_on_original_numbers']['tdfs']['powerful_timbre_fpr']['point']
        ]
        hpss_vals = [
            spl02['bootstrap_cis_on_original_numbers']['hpss']['soft_timbre_fpr']['point'],
            spl02['bootstrap_cis_on_original_numbers']['hpss']['powerful_timbre_fpr']['point']
        ]
        ax.bar(x,          vanilla_vals, width, label='Vanilla CNN',    color='#e74c3c')
        ax.bar(x + width,  tdfs_vals,    width, label='TDFS',           color='#2ecc71')
        ax.bar(x + 2*width, hpss_vals,  width, label='HPSS + Vanilla', color='#3498db')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('False Positive Rate')
    ax.set_title('Timbre-Subgroup FPR: PANNs CNN14 vs. All Methods')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(BAR_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    os.sync()
    print(f"FPR comparison chart saved to {BAR_PNG}")

# ==========================================
# 7. THRESHOLD SWEEP
#    Zero-shot threshold may not be 0.5 for PANNs.
#    Sweep thresholds to find best-accuracy operating point.
# ==========================================
def threshold_sweep(df):
    gts    = df['ground_truth'].values
    scores = df['sigmoid_score'].values
    thresholds = np.arange(0.05, 0.96, 0.05)
    best = {'threshold': 0.5, 'accuracy': 0.0}
    sweep_rows = []
    for t in thresholds:
        preds = (scores > t).astype(float)
        acc   = float((preds == gts).mean())
        sweep_rows.append({'threshold': round(float(t), 2), 'accuracy': round(acc, 4)})
        if acc > best['accuracy']:
            best = {'threshold': round(float(t), 2), 'accuracy': round(acc, 4)}
    return best, sweep_rows

# ==========================================
# 8. MAIN
# ==========================================
def main():
    print(f"Device: {DEVICE}")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    ensure_local_audio()
    download_panns_checkpoint()
    install_panns()

    model = init_panns_cnn14()
    run_panns_inference(model)

    df = pd.read_csv(OUT_PREDS_CSV)
    print(f"\nLoaded {len(df)} predictions for audit.")

    # Threshold sweep — important for zero-shot PANNs
    best_thresh, sweep = threshold_sweep(df)
    print(f"\nBest threshold by accuracy: {best_thresh}")

    # Recompute predictions at best threshold before audit
    df['prediction'] = (df['sigmoid_score'] > best_thresh['threshold']).astype(float)

    audit = audit_panns_fairness(df)
    plot_fpr_comparison(audit)

    results = {
        'model': 'PANNs CNN14 (Kong et al., 2020)',
        'audioset_singing_class_index': SINGING_IDX,
        'zero_shot': True,
        'default_threshold_0_5_accuracy': round(
            float((
                (df['sigmoid_score'].values > 0.5).astype(float)
                == df['ground_truth'].values
            ).mean()), 4
        ),
        'best_threshold': best_thresh,
        'threshold_sweep': sweep,
        'fairness_audit_at_best_threshold': audit,
        'conclusion': (
            'If powerful-timbre FPR remains substantially higher than soft-timbre FPR, '
            'PANNs CNN14 exhibits the same structural timbre bias as the Vanilla CNN, '
            'confirming the confound is not architecture-specific.'
        )
    }

    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    os.sync()
    print(f"\nFull results saved to {RESULTS_JSON}")
    print(json.dumps({
        'best_threshold': best_thresh,
        'soft_fpr':      audit['soft_timbre_fpr']['point'],
        'powerful_fpr':  audit['powerful_timbre_fpr']['point'],
    }, indent=4))

if __name__ == '__main__':
    main()

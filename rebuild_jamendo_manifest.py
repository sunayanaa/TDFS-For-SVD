"""
Filename: rebuild_jamendo_manifest.py
Version: 1.0.0
Description: Reconstructs research_manifest_with_f0.csv directly from the
             existing wav_24k directory when the original manifest is lost.
             Combines the functionality of feature_extractor.py and
             rebalance_pitch.py into a single resumable script.
             Uses RMS energy for vocal detection, spectral centroid quantiles
             (25th/75th percentile) for timbre labelling, and YIN pitch
             tracking (fmin=65, fmax=300 Hz) for f0 estimation.
             Checkpoints every 50 tracks with os.sync().

Changelog:
  - v1.0.0 (2026-06-27): Initial release. Replaces feature_extractor.py +
                          rebalance_pitch.py when research_manifest.csv is
                          unavailable but wav_24k is intact.
						  
GPU not needed
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

WAV_DIR       = '/content/drive/MyDrive/datasets/Jamendo/wav_24k'
OUTPUT_CSV    = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
TARGET_SR     = 24000
SAVE_INTERVAL = 50

# Load checkpoint if exists
if os.path.exists(OUTPUT_CSV):
    df_done = pd.read_csv(OUTPUT_CSV)
    done_paths = set(df_done['path'].tolist())
    rows = df_done.to_dict('records')
    print(f"Resuming: {len(done_paths)} tracks already processed.")
else:
    done_paths = set()
    rows = []

wav_files = sorted([
    os.path.join(WAV_DIR, f)
    for f in os.listdir(WAV_DIR) if f.endswith('.wav')
])
print(f"Total WAV files: {len(wav_files)}")

count = 0
for wav_path in tqdm(wav_files):
    if wav_path in done_paths:
        continue

    try:
        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        # Vocal detection: RMS energy threshold
        rms = np.sqrt(np.mean(y**2))
        is_vocal = bool(rms > 0.005)

        # Spectral centroid for timbre labelling (computed on full track)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024)
        mean_centroid = float(np.mean(centroid))

        # Pitch tracking (YIN) — only for vocal tracks
        median_f0 = np.nan
        pitch_class = 'Non-Vocal'
        if is_vocal:
            f0 = librosa.yin(y, fmin=65, fmax=300, sr=sr)
            valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]
            if len(valid_f0) > 0:
                median_f0 = float(np.median(valid_f0))
                pitch_class = 'unknown'  # rebalance_pitch.py will fix this

        rows.append({
            'path':              wav_path,
            'is_vocal':          is_vocal,
            'median_f0':         median_f0,
            'mean_centroid':     mean_centroid,
            'pitch_class':       pitch_class,
            'is_soft_timbre':    False,   # set after quantile thresholding below
            'is_powerful_timbre': False,
        })
        done_paths.add(wav_path)

    except Exception as e:
        rows.append({
            'path': wav_path, 'is_vocal': False,
            'median_f0': np.nan, 'mean_centroid': np.nan,
            'pitch_class': 'error',
            'is_soft_timbre': False, 'is_powerful_timbre': False,
        })

    count += 1
    if count % SAVE_INTERVAL == 0:
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        os.sync()

# Final save
df = pd.DataFrame(rows)

# Timbre thresholding: bottom 25th = soft, top 75th = powerful
valid_centroids = df['mean_centroid'].dropna()
c25 = valid_centroids.quantile(0.25)
c75 = valid_centroids.quantile(0.75)
df['is_soft_timbre']     = df['mean_centroid'] <= c25
df['is_powerful_timbre'] = df['mean_centroid'] >= c75

# Pitch rebalancing: split vocal tracks at median f0
vocal_mask  = df['is_vocal'] == True
valid_f0    = df.loc[vocal_mask, 'median_f0'].dropna()
median_pitch = valid_f0.median()
print(f"Dataset median f0: {median_pitch:.2f} Hz")

df.loc[vocal_mask & (df['median_f0'] <= median_pitch), 'pitch_class'] = 'Low-Pitch (Male Proxy)'
df.loc[vocal_mask & (df['median_f0'] >  median_pitch), 'pitch_class'] = 'High-Pitch (Female Proxy)'

df.to_csv(OUTPUT_CSV, index=False)
os.sync()

print(f"\nManifest saved: {OUTPUT_CSV}")
print(df['pitch_class'].value_counts())
print(f"Soft timbre:     {df['is_soft_timbre'].sum()}")
print(f"Powerful timbre: {df['is_powerful_timbre'].sum()}")

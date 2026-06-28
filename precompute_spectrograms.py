"""
Filename: precompute_spectrograms.py
Version: 1.0.0
Description: Precomputes Log-Mel spectrograms for all 4163 Jamendo tracks
             and saves them as a single NPZ file on Drive (~500MB).
             After this runs once, tdfs_model.py and baseline_svd.py load
             from the NPZ instead of decoding audio — eliminating all audio
             I/O overhead during training.
             CPU only. Checkpoints every 100 tracks.
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

MANIFEST    = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
MP3_DIR = None  # will be set per track from manifest path
OUTPUT_NPZ  = '/content/drive/MyDrive/paper/DecouplingTimbre/jamendo_spectrograms.npz'
CACHE_CSV   = '/content/drive/MyDrive/paper/DecouplingTimbre/jamendo_spec_cache.csv'

SR          = 24000
N_FFT       = 1024
HOP_LENGTH  = 256
N_MELS      = 80
SEGMENT_LEN = 48000   # 2 seconds
SAVE_EVERY  = 100

def get_mp3_path(wav_path):
    track_id = os.path.splitext(os.path.basename(wav_path))[0]
    subdir   = track_id[-2:]
    return os.path.join(
        '/content/drive/MyDrive/datasets/Jamendo/audio_data',
        subdir, f"{track_id}.mp3"
    )

def compute_mel(mp3_path):
    y, _ = librosa.load(mp3_path, sr=SR, mono=True, duration=2.0, offset=1.0)
    if len(y) < SEGMENT_LEN:
        y = np.pad(y, (0, SEGMENT_LEN - len(y)))
    else:
        y = y[:SEGMENT_LEN]
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)

def main():
    os.makedirs('/content/drive/MyDrive/paper/DecouplingTimbre', exist_ok=True)
    df = pd.read_csv(MANIFEST)
    df = df[df['pitch_class'] != 'error'].reset_index(drop=True)
    print(f"Total tracks: {len(df)}")

    # Resume from cache if exists
    if os.path.exists(CACHE_CSV):
        cache_df = pd.read_csv(CACHE_CSV)
        done_ids = set(cache_df['track_id'].tolist())
        specs    = {row['track_id']: None for _, row in cache_df.iterrows()}
        print(f"Resuming: {len(done_ids)} tracks already cached.")
    else:
        done_ids = set()
        specs    = {}

    # Load existing NPZ if present
    if os.path.exists(OUTPUT_NPZ) and len(done_ids) > 0:
        print("Loading existing NPZ...")
        existing = np.load(OUTPUT_NPZ, allow_pickle=True)
        for k in existing.files:
            specs[k] = existing[k]
        print(f"Loaded {len(specs)} spectrograms from NPZ.")

    errors = []
    count  = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        track_id = os.path.splitext(os.path.basename(row['path']))[0]
        if track_id in done_ids:
            continue

        mp3_path = get_mp3_path(row['path'])
        if not os.path.exists(mp3_path):
            errors.append(track_id)
            continue

        try:
            mel = compute_mel(mp3_path)
            specs[track_id] = mel
            done_ids.add(track_id)
            count += 1
        except Exception as e:
            errors.append(track_id)
            continue

        # Checkpoint every SAVE_EVERY tracks
        if count % SAVE_EVERY == 0:
            np.savez_compressed(OUTPUT_NPZ, **specs)
            pd.DataFrame({'track_id': list(done_ids)}).to_csv(CACHE_CSV, index=False)
            os.sync()
            print(f"Checkpoint: {len(done_ids)} tracks saved.")

    # Final save
    np.savez_compressed(OUTPUT_NPZ, **specs)
    pd.DataFrame({'track_id': list(done_ids)}).to_csv(CACHE_CSV, index=False)
    os.sync()

    print(f"\nDone. {len(specs)} spectrograms saved to {OUTPUT_NPZ}")
    print(f"Errors: {len(errors)} tracks — {errors[:10]}")
    npz_size = os.path.getsize(OUTPUT_NPZ) / 1e6
    print(f"NPZ file size: {npz_size:.1f} MB")

if __name__ == '__main__':
    main()
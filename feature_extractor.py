"""
Filename: feature_extractor.py
Version: 1.1.0
Description: Standardizes audio to 24kHz Mono WAV for Neural Vocoder training.
             Simultaneously performs Pitch-Tracking (f0) using the YIN algorithm.
             Updated in v1.1.0 with checkpointing to allow resuming after 
             network disconnections.

Changelog:
  - v1.0.0: Initial release.
  - v1.1.0 (2026-04-12): Added checkpoint logic. The script now saves the 
                         manifest every 100 tracks and resumes from the last 
                         processed file.
"""

import pandas as pd
import librosa
import soundfile as sf
import os
import numpy as np
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')


# Define Paths
MANIFEST_PATH = '/content/drive/MyDrive/datasets/Jamendo/research_manifest.csv'
CHECKPOINT_PATH = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
INPUT_AUDIO_DIR = '/content/drive/MyDrive/datasets/Jamendo/audio_data'
OUTPUT_WAV_DIR = '/content/drive/MyDrive/datasets/Jamendo/wav_24k'
TARGET_SR = 24000
SAVE_INTERVAL = 100  # Save metadata every 100 tracks

def process_and_label_audio():
    # 1. Load the best available manifest (resume from checkpoint if it exists)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        df = pd.read_csv(CHECKPOINT_PATH)
    else:
        print("Starting fresh from master manifest...")
        df = pd.read_csv(MANIFEST_PATH)
        df['is_vocal'] = df['tags'].str.contains('---voice', na=False)
        # Filter for targets
        df = df[(df['is_vocal']) | (df['is_soft_timbre']) | (df['is_powerful_timbre'])].copy()
        df['median_f0'] = np.nan
        df['pitch_class'] = 'unknown'

    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    
    # 2. Identify remaining tracks (where pitch_class is still 'unknown')
    remaining_tracks = df[df['pitch_class'] == 'unknown']
    print(f"Total target tracks: {len(df)} | Remaining to process: {len(remaining_tracks)}")

    count = 0
    for index, row in tqdm(remaining_tracks.iterrows(), total=len(remaining_tracks)):
        mp3_path = os.path.join(INPUT_AUDIO_DIR, str(row['path']))
        wav_name = os.path.basename(mp3_path).replace('.mp3', '.wav')
        wav_path = os.path.join(OUTPUT_WAV_DIR, wav_name)

        if os.path.exists(mp3_path):
            try:
                # Load/Resample if WAV doesn't exist
                if not os.path.exists(wav_path):
                    y, sr = librosa.load(mp3_path, sr=TARGET_SR, mono=True)
                    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
                    sf.write(wav_path, y_trimmed, TARGET_SR, subtype='PCM_16')
                else:
                    # If WAV exists, just load it for pitch tracking to save time
                    y_trimmed, sr = librosa.load(wav_path, sr=TARGET_SR)

                # Signal Processing: Pitch Extraction
                if row['is_vocal']:
                    f0 = librosa.yin(y_trimmed, fmin=65, fmax=300, sr=TARGET_SR)
                    valid_f0 = f0[~np.isnan(f0)]
                    if len(valid_f0) > 0:
                        median_pitch = np.median(valid_f0)
                        df.at[index, 'median_f0'] = median_pitch
                        df.at[index, 'pitch_class'] = 'Low-Pitch (Male Proxy)' if median_pitch < 165 else 'High-Pitch (Female Proxy)'
                else:
                    # For non-vocal tracks, mark as processed so we don't visit them again
                    df.at[index, 'pitch_class'] = 'Non-Vocal'

            except Exception as e:
                df.at[index, 'pitch_class'] = 'error'
        
        # 3. Checkpoint Save
        count += 1
        if count % SAVE_INTERVAL == 0:
            df.to_csv(CHECKPOINT_PATH, index=False)

    # Final Save
    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"\nProcessing complete. Manifest saved to: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    process_and_label_audio()
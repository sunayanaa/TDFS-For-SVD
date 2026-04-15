"""
Filename: synthesize_musdb18.py
Version: 1.1.0
Description: Synthesizes a balanced training dataset entirely on local VM storage 
             to bypass Drive I/O bottlenecks. Renders mixture (Vocal) and pure 
             instrumental (Non-Vocal) paired stems. 

Changelog:
  - v1.0.0: Initial concept (flawed Drive I/O).
  - v1.1.0 (2026-04-14): Optimized to write WAVs exclusively to local Colab disk. 
                         Manifest paths point to local storage for high-speed 
                         GPU data loading during the subsequent training phase.
"""

!pip install stempeg

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import stempeg
import soundfile as sf
import librosa
import subprocess
from tqdm import tqdm

def synthesize_dataset():
    INPUT_MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_manifest_with_f0.csv'
    OUTPUT_MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_balanced_manifest.csv'
    
    # Fast Local Storage
    LOCAL_WAV_DIR = '/content/musdb18_local/wav_24k'
    
    # Safe Drive Storage for the final Zip
    DRIVE_ZIP_PATH = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_wav_24k.zip'
    
    TARGET_SR = 24000
    os.makedirs(LOCAL_WAV_DIR, exist_ok=True)

    print("Loading original manifest...")
    df = pd.read_csv(INPUT_MANIFEST)
    
    balanced_data = []
    existing_wavs = set(os.listdir(LOCAL_WAV_DIR))

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Synthesizing Paired Tracks (Local Disk)"):
        mp4_path = row['path']
        base_name = os.path.basename(mp4_path).replace('.mp4', '')
        
        vocal_wav_name = f"{base_name}_vocal.wav"
        inst_wav_name = f"{base_name}_instrumental.wav"
        
        vocal_wav_path = os.path.join(LOCAL_WAV_DIR, vocal_wav_name)
        inst_wav_path = os.path.join(LOCAL_WAV_DIR, inst_wav_name)

        if vocal_wav_name not in existing_wavs or inst_wav_name not in existing_wavs:
            try:
                S, sr = stempeg.read_stems(mp4_path)
                
                # 1. The Mixture
                mixture = S[0]
                if mixture.shape[1] > 1: mixture = np.mean(mixture, axis=1) 
                if sr != TARGET_SR: mixture = librosa.resample(mixture, orig_sr=sr, target_sr=TARGET_SR)
                sf.write(vocal_wav_path, mixture, TARGET_SR)

                # 2. The Instrumental
                instrumental = S[1] + S[2] + S[3]
                if instrumental.shape[1] > 1: instrumental = np.mean(instrumental, axis=1) 
                if sr != TARGET_SR: instrumental = librosa.resample(instrumental, orig_sr=sr, target_sr=TARGET_SR)
                sf.write(inst_wav_path, instrumental, TARGET_SR)
                
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                continue

        balanced_data.append({
            'path': vocal_wav_path, # Points to local disk!
            'is_vocal': True,
            'median_f0': row['median_f0'],
            'pitch_class': 'unknown',
            'is_soft_timbre': row['is_soft_timbre'],
            'is_powerful_timbre': row['is_powerful_timbre']
        })

        balanced_data.append({
            'path': inst_wav_path, # Points to local disk!
            'is_vocal': False,
            'median_f0': np.nan,
            'pitch_class': 'Non-Vocal',
            'is_soft_timbre': row['is_soft_timbre'],
            'is_powerful_timbre': row['is_powerful_timbre']
        })

    # Save manifest directly to drive
    balanced_df = pd.DataFrame(balanced_data)
    balanced_df.to_csv(OUTPUT_MANIFEST, index=False)
    
    print("\n--- Phase 2: Zipping Local Audio to Google Drive ---")
    print("Zipping the 300 generated WAV files into a single archive for safe storage...")
    # Zip the local folder and output the zip directly to Google Drive
    subprocess.run(['zip', '-q', '-r', '-j', DRIVE_ZIP_PATH, LOCAL_WAV_DIR])

    print("\n--- Synthesis Complete ---")
    os.sync()
    print(f"Total balanced dataset size: {len(balanced_df)} tracks.")
    print(f"Saved manifest to {OUTPUT_MANIFEST}")
    print(f"Safely archived audio to {DRIVE_ZIP_PATH}")

if __name__ == "__main__":
    synthesize_dataset()
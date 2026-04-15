"""
Filename: extract_musdb18.py
Version: 1.3.0
Description: Unified, fault-tolerant pipeline tailored specifically for the 
             MUSDB18 MP4 STEM format. Uses stempeg to read multi-track MP4 
             files directly, extracting vocals (stream 4) and calculating 
             timbre from instrumental stems (streams 1, 2, 3).

Changelog:
  - v1.2.0: Resumable pipeline for WAV folders.
  - v1.3.0 (2026-04-14): Complete overhaul to support MUSDB18 multi-track 
                         .mp4 files using stempeg. File discovery logic updated 
                         to handle flat directories with spaces in filenames.
"""

!pip install stempeg

from google.colab import drive
drive.mount('/content/drive')

import os
import subprocess
import torch
import torchaudio
import pandas as pd
import numpy as np
import stempeg
from tqdm import tqdm

def unified_pipeline():
    ZIP_PATH = '/content/drive/MyDrive/datasets/MUSDB18/musdb18.zip'
    LOCAL_EXTRACT_DIR = '/content/musdb18_local'
    OUTPUT_CSV = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_manifest_with_f0.csv'
    TARGET_SR = 24000

    # ==========================================
    # PHASE 1: IDEMPOTENT UNZIP
    # ==========================================
    print("--- Phase 1: Local Storage Verification ---")
    if not os.path.exists(ZIP_PATH):
        print(f"CRITICAL ERROR: Could not find {ZIP_PATH}. Please check your Drive.")
        return

    os.makedirs(LOCAL_EXTRACT_DIR, exist_ok=True)
    print(f"Verifying {ZIP_PATH} extraction (will instantly skip if already unzipped)...")
    subprocess.run(['unzip', '-q', '-n', ZIP_PATH, '-d', LOCAL_EXTRACT_DIR])
    print("Local storage ready.\n")

    actual_root = LOCAL_EXTRACT_DIR
    if os.path.exists(os.path.join(LOCAL_EXTRACT_DIR, 'musdb18')):
        actual_root = os.path.join(LOCAL_EXTRACT_DIR, 'musdb18')

    # ==========================================
    # PHASE 2: RESUMABLE STEM EXTRACTION
    # ==========================================
    print("--- Phase 2: Resumable Feature Extraction ---")
    
    # Locate all .mp4 files directly inside the train/test folders
    all_mp4_paths = []
    for subset in ['train', 'test']:
        subset_dir = os.path.join(actual_root, subset)
        if os.path.exists(subset_dir):
            mp4s = [os.path.join(subset_dir, f) for f in os.listdir(subset_dir) if f.endswith('.mp4')]
            all_mp4_paths.extend(mp4s)

    if not all_mp4_paths:
        print("Error: No .mp4 files found. Please check the dataset structure.")
        return

    # Load previous state to resume
    processed_paths = set()
    manifest_data = []
    
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing manifest at {OUTPUT_CSV}. Loading state...")
        existing_df = pd.read_csv(OUTPUT_CSV)
        processed_paths = set(existing_df['path'].tolist())
        manifest_data = existing_df.to_dict('records')
        print(f"Resuming: {len(processed_paths)} tracks already processed.")

    resampler = None
    last_sr = None

    for mp4_path in tqdm(all_mp4_paths, desc="Analyzing MP4 Stems"):
        if mp4_path in processed_paths:
            continue

        try:
            # stempeg returns shape: (stems, samples, channels)
            # 0: mixture, 1: drums, 2: bass, 3: other, 4: vocals
            S, sr = stempeg.read_stems(mp4_path)
            
            # Extract raw numpy arrays
            vocal_audio = S[4]
            instrumental_audio = S[1] + S[2] + S[3]
            
            # Convert to PyTorch format: (channels, samples)
            vocal_wave = torch.tensor(vocal_audio, dtype=torch.float32).transpose(0, 1)
            inst_wave = torch.tensor(instrumental_audio, dtype=torch.float32).transpose(0, 1)

            # Convert to Mono
            if vocal_wave.shape[0] > 1: vocal_wave = torch.mean(vocal_wave, dim=0, keepdim=True)
            if inst_wave.shape[0] > 1: inst_wave = torch.mean(inst_wave, dim=0, keepdim=True)

            # Resample if needed
            if sr != TARGET_SR:
                if resampler is None or last_sr != sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                    last_sr = sr
                vocal_wave = resampler(vocal_wave)
                inst_wave = resampler(inst_wave)

            # 1. Determine if Vocal
            rms_energy = torch.sqrt(torch.mean(vocal_wave ** 2)).item()
            is_vocal = rms_energy > 0.005 

            # 2. Extract Pitch (f0)
            median_f0 = np.nan
            if is_vocal:
                pitch = torchaudio.functional.detect_pitch_frequency(vocal_wave, TARGET_SR)
                voiced_pitch = pitch[pitch > 0]
                if len(voiced_pitch) > 0:
                    median_f0 = torch.median(voiced_pitch).item()

            # 3. Extract Timbre (Instrumental Only)
            centroid = torchaudio.transforms.SpectralCentroid(sample_rate=TARGET_SR, n_fft=1024)(inst_wave)
            mean_centroid = torch.mean(centroid).item()

            new_row = {
                'path': mp4_path, 
                'is_vocal': is_vocal,
                'median_f0': median_f0,
                'mean_centroid': mean_centroid,
                'pitch_class': 'unknown',
                'is_soft_timbre': False,    
                'is_powerful_timbre': False 
            }
            
            manifest_data.append(new_row)
            processed_paths.add(mp4_path)
            
            # Checkpoint
            pd.DataFrame(manifest_data).to_csv(OUTPUT_CSV, index=False)
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(mp4_path)}: {e}")

    # ==========================================
    # PHASE 3: TIMBRE THRESHOLDING
    # ==========================================
    df = pd.DataFrame(manifest_data)
    if len(df) == len(all_mp4_paths) and len(df) > 0:
        print("\n--- Phase 3: Finalizing Dataset Thresholds ---")
        centroid_25th = df['mean_centroid'].quantile(0.25)
        centroid_75th = df['mean_centroid'].quantile(0.75)
        
        df['is_soft_timbre'] = df['mean_centroid'] <= centroid_25th
        df['is_powerful_timbre'] = df['mean_centroid'] >= centroid_75th
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Extraction 100% complete. Final manifest saved to {OUTPUT_CSV}")
        print(f"Total tracks: {len(df)} | Vocal tracks: {(df['is_vocal'] == True).sum()}")
    else:
        print(f"\nExtraction paused/incomplete. Processed {len(df)} out of {len(all_mp4_paths)} tracks.")

if __name__ == "__main__":
    unified_pipeline()
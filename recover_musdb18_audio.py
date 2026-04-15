"""
Filename: recover_musdb18_audio.py
Description: Rebuilds the missing musdb18_wav_24k.zip directly from the 
             original musdb18.zip. Unpacks the MP4s, synthesizes the 300 
             WAV files, and safely packages them back to Google Drive.
"""
!pip install stempeg

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import subprocess
import pandas as pd
import numpy as np
import stempeg
import soundfile as sf
import librosa
import shutil
from tqdm import tqdm

def recover_audio():
    RAW_ZIP = '/content/drive/MyDrive/datasets/MUSDB18/musdb18.zip'
    LOCAL_EXTRACT = '/content/musdb18_local'
    LOCAL_WAV_DIR = '/content/musdb18_local/wav_24k'
    TARGET_ZIP = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_wav_24k.zip'
    MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_manifest_with_f0.csv'
    TARGET_SR = 24000

    # ==========================================
    # PHASE 1: UNPACK RAW MP4s
    # ==========================================
    if not os.path.exists(RAW_ZIP):
        print(f"CRITICAL ERROR: Could not find {RAW_ZIP}. Please check your Drive.")
        return

    print("--- Phase 1: Restoring Original MP4s ---")
    os.makedirs(LOCAL_EXTRACT, exist_ok=True)
    # -n flag ensures it skips extraction if the mp4s are already there
    subprocess.run(['unzip', '-q', '-n', RAW_ZIP, '-d', LOCAL_EXTRACT])
    print("Original MP4 stems restored.")

    # ==========================================
    # PHASE 2: SYNTHESIZE WAVs
    # ==========================================
    print("\n--- Phase 2: Synthesizing 300 WAV Files ---")
    os.makedirs(LOCAL_WAV_DIR, exist_ok=True)
    df = pd.read_csv(MANIFEST)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Rendering Audio"):
        mp4_path = row['path']
        base_name = os.path.basename(mp4_path).replace('.mp4', '')
        
        vocal_wav_path = os.path.join(LOCAL_WAV_DIR, f"{base_name}_vocal.wav")
        inst_wav_path = os.path.join(LOCAL_WAV_DIR, f"{base_name}_instrumental.wav")

        # Skip rendering if this specific track survived
        if os.path.exists(vocal_wav_path) and os.path.exists(inst_wav_path):
            continue

        try:
            # stempeg returns shape: (stems, samples, channels)
            S, sr = stempeg.read_stems(mp4_path)
            
            # 1. The Mixture
            mixture = S[0]
            if mixture.shape[1] > 1: mixture = np.mean(mixture, axis=1) 
            if sr != TARGET_SR: mixture = librosa.resample(mixture, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(vocal_wav_path, mixture, TARGET_SR)

            # 2. The Instrumental (Sum of stems 1, 2, 3)
            instrumental = S[1] + S[2] + S[3]
            if instrumental.shape[1] > 1: instrumental = np.mean(instrumental, axis=1) 
            if sr != TARGET_SR: instrumental = librosa.resample(instrumental, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(inst_wav_path, instrumental, TARGET_SR)
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    # ==========================================
    # PHASE 3: SECURE ARCHIVE TO DRIVE
    # ==========================================
    print("\n--- Phase 3: Archiving to Google Drive ---")
    file_count = len(os.listdir(LOCAL_WAV_DIR))
    print(f"Zipping {file_count} files into the final archive...")
    
    # shutil.make_archive creates the zip directly and safely
    zip_base_path = TARGET_ZIP.replace('.zip', '')
    shutil.make_archive(zip_base_path, 'zip', LOCAL_WAV_DIR)
    os.sync()
    print(f"\nRecovery Complete! Safely archived audio to {TARGET_ZIP}")
    print("You can now re-run the advanced_metrics.py script!")

if __name__ == "__main__":
    recover_audio()
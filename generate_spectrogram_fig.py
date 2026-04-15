"""
Filename:  generate_spectrogram_fig.py
Version: 1.0.0
Description: Generates spectrogram used  for Fig 1
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

def generate_spectrogram_figure():
    MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'
    LOCAL_WAV_DIR = '/content/musdb18_local/wav_24k'
    OUTPUT_PDF = '/content/drive/MyDrive/datasets/MUSDB18/spectrogram_masking.pdf'
    
    # 1. Auto-select tracks directly from your manifest
    print("Reading manifest to auto-select tracks...")
    df = pd.read_csv(MANIFEST)
    
    soft_row = df[df['is_soft_timbre'] == 1].iloc[0]
    pow_row = df[df['is_powerful_timbre'] == 1].iloc[0]
    
    # 2. Reconstruct the exact local WAV paths
    def get_local_path(original_path):
        # The manifest already contains the correct filename, just grab it
        file_name = os.path.basename(original_path)
        return os.path.join(LOCAL_WAV_DIR, file_name)

    soft_path = get_local_path(soft_row['path'])
    pow_path = get_local_path(pow_row['path'])
    
    print(f"Selected Soft Track: {os.path.basename(soft_path)}")
    print(f"Selected Powerful Track: {os.path.basename(pow_path)}")

    # Mel-spectrogram parameters
    SR = 24000
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 80
    
    def get_mel_spec(file_path):
        # Load exactly 3 seconds of audio. offset=15.0 drops the intro silence.
        y, _ = librosa.load(file_path, sr=SR, duration=3.0, offset=15.0)
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        return librosa.power_to_db(S, ref=np.max)

    # Generate Spectrograms
    try:
        S_soft = get_mel_spec(soft_path)
        S_pow = get_mel_spec(pow_path)
    except Exception as e:
        print(f"\nCRITICAL ERROR loading audio: {e}")
        return

    # Create IEEE-formatted side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    
    # Plot Soft Timbre
    librosa.display.specshow(S_soft, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=axes[0], cmap='magma')
    axes[0].set_title('Soft Timbre (Vocals Dominant)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    
    # Draw Bounding Box for Vocals (Approx 300-3000 Hz)
    rect1 = patches.Rectangle((0, 300), 3.0, 2700, linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    axes[0].add_patch(rect1)
    axes[0].text(0.1, 3500, 'Vocal Formants Clear', color='cyan', fontsize=9, fontweight='bold')

    # Plot Powerful Timbre
    librosa.display.specshow(S_pow, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=axes[1], cmap='magma')
    axes[1].set_title('Powerful Timbre (Vocals Masked)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('') 
    
    # Draw Bounding Box for Vocals
    rect2 = patches.Rectangle((0, 300), 3.0, 2700, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    axes[1].add_patch(rect2)
    axes[1].text(0.1, 3500, 'Harmonic Masking (Guitars/Synths)', color='red', fontsize=9, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\nSUCCESS! Spectrogram figure saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_spectrogram_figure()
"""
Filename: dataset_loader.py
Version: 1.0.0
Description: Custom PyTorch Dataset and DataLoader for the TDFS model. 
             Reads the f0-labeled manifest, loads 24kHz WAV files, computes 
             Mel-Spectrograms on the fly using GPU/CPU, and formats batches 
             for Contrastive Learning and Fairness Auditing.

Changelog:
  - v1.0.0 (2026-04-13): Initial release. Configured torchaudio transforms to 
                         match Neural Vocoder (BigVGAN) specifications.
"""

import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader

from google.colab import drive
drive.mount('/content/drive')


# BigVGAN / HiFi-GAN standard Mel-Spectrogram parameters
MEL_KWARGS = {
    "sample_rate": 24000,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
    "f_min": 0.0,
    "f_max": 12000.0,
    "n_mels": 80,
}

class JamendoFairnessDataset(Dataset):
    def __init__(self, manifest_path, audio_dir, segment_length=48000):
        """
        segment_length: Default 48000 samples (2 seconds at 24kHz) for uniform batching.
        """
        self.audio_dir = audio_dir
        self.segment_length = segment_length
        
        # Load manifest and drop any tracks that errored out during preprocessing
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['pitch_class'] != 'error'].reset_index(drop=True)
        
        # Initialize the Torchaudio Mel-Spectrogram transformer
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**MEL_KWARGS)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Create mapping dictionaries for the textual labels
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
        wav_name = os.path.basename(row['path']).replace('.mp3', '.wav')
        wav_path = os.path.join(self.audio_dir, wav_name)

        # 1. Load Audio
        waveform, sr = torchaudio.load(wav_path)
        
        # 2. Extract a random 2-second segment (Standard practice for SSL audio training)
        if waveform.shape[1] > self.segment_length:
            max_start = waveform.shape[1] - self.segment_length
            start = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[:, start : start + self.segment_length]
        else:
            # Pad with zeros if shorter than 2 seconds
            pad_amount = self.segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # 3. Compute Log-Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # 4. Gather Labels for the Loss Function
        labels = {
            "is_vocal": torch.tensor(1.0 if row['is_vocal'] else 0.0, dtype=torch.float32),
            "pitch_class": torch.tensor(self.pitch_map.get(row['pitch_class'], 3), dtype=torch.long),
            "is_soft": torch.tensor(1.0 if row['is_soft_timbre'] else 0.0, dtype=torch.float32),
            "is_powerful": torch.tensor(1.0 if row['is_powerful_timbre'] else 0.0, dtype=torch.float32),
            "median_f0": torch.tensor(row['median_f0'] if pd.notna(row['median_f0']) else 0.0, dtype=torch.float32)
        }

        return log_mel_spec, labels

# Quick test block to verify the pipeline
if __name__ == "__main__":
    MANIFEST = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'
    AUDIO_DIR = '/content/drive/MyDrive/datasets/Jamendo/wav_24k'
    
    print("Testing PyTorch Dataset initialization...")
    dataset = JamendoFairnessDataset(MANIFEST, AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # Fetch one batch
    for batch_mels, batch_labels in dataloader:
        print(f"Batch Mel-Spec shape: {batch_mels.shape}") # Expected: [16, 1, 80, 188]
        print(f"Batch Pitch Classes: {batch_labels['pitch_class']}")
        break
    print("Pipeline ready for training.")
"""
Filename: rebalance_pitch.py
Version: 1.0.0
Description: Utility script to calculate the true median fundamental frequency (f0) 
             across all identified vocal tracks in the dataset. Dynamically reassigns 
             'High-Pitch' and 'Low-Pitch' proxy labels based on this dataset-specific 
             median to ensure a perfectly balanced 50/50 split for the fairness audit.

Changelog:
  - v1.0.0 (2026-04-13): Initial release. Resolves the severe class imbalance (571 vs 7) 
                         caused by the hardcoded 165Hz threshold in the initial extractor.
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

MANIFEST = '/content/drive/MyDrive/datasets/Jamendo/research_manifest_with_f0.csv'

def rebalance_proxies():
    print("Loading manifest...")
    df = pd.read_csv(MANIFEST)

    # Isolate vocal tracks with valid pitch data
    vocal_mask = df['is_vocal'] == True
    valid_pitch = df.loc[vocal_mask, 'median_f0'].dropna()

    # Find the exact 50th percentile (median) of the dataset's pitch
    dataset_median_pitch = valid_pitch.median()
    print(f"Dataset Median Pitch Threshold: {dataset_median_pitch:.2f} Hz")

    # Reassign classes perfectly down the middle
    df.loc[vocal_mask & (df['median_f0'] <= dataset_median_pitch), 'pitch_class'] = 'Low-Pitch (Male Proxy)'
    df.loc[vocal_mask & (df['median_f0'] > dataset_median_pitch), 'pitch_class'] = 'High-Pitch (Female Proxy)'

    # Verify new distribution
    low_count = (df['pitch_class'] == 'Low-Pitch (Male Proxy)').sum()
    high_count = (df['pitch_class'] == 'High-Pitch (Female Proxy)').sum()

    print(f"New Low-Pitch count: {low_count}")
    print(f"New High-Pitch count: {high_count}")

    # Save the corrected manifest
    df.to_csv(MANIFEST, index=False)
    print("Manifest rebalanced and saved.")

if __name__ == "__main__":
    rebalance_proxies()
    
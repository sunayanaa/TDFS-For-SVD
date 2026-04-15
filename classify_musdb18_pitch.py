"""
Filename: classify_musdb18_pitch.py
Version: 1.0.0
Description: Finalizes the MUSDB18 dataset for the fairness audit. 
             Calculates the global median f0 across all vocal tracks to 
             establish the High-Pitch (Female Proxy) and Low-Pitch (Male Proxy) 
             boundaries. Outputs the final research manifest.
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

def classify_pitch():
    INPUT_MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_balanced_manifest.csv'
    OUTPUT_MANIFEST = '/content/drive/MyDrive/datasets/MUSDB18/musdb18_research_manifest_with_f0.csv'

    print("Loading balanced manifest...")
    df = pd.read_csv(INPUT_MANIFEST)

    # Isolate vocal tracks with valid f0 data
    vocals = df[(df['is_vocal'] == True) & (df['median_f0'].notna())]
    
    if len(vocals) == 0:
        print("Error: No vocal tracks with valid pitch found.")
        return

    # Find the exact median pitch to split the dataset perfectly in half
    dataset_median_f0 = vocals['median_f0'].median()
    print(f"Dataset Median Pitch (Split Point): {dataset_median_f0:.2f} Hz")

    # Apply classifications
    def assign_pitch_class(row):
        if not row['is_vocal']:
            return 'Non-Vocal'
        if pd.isna(row['median_f0']):
            return 'unknown'
        if row['median_f0'] > dataset_median_f0:
            return 'High-Pitch (Female Proxy)'
        else:
            return 'Low-Pitch (Male Proxy)'

    df['pitch_class'] = df.apply(assign_pitch_class, axis=1)

    # Save final research manifest
    df.to_csv(OUTPUT_MANIFEST, index=False)
    
    print("\n--- Final Dataset Distribution ---")
    print(df['pitch_class'].value_counts())
    os.sync()
    print(f"\nFinal research manifest saved to: {OUTPUT_MANIFEST}")
    print("Ready for cross-dataset validation!")

if __name__ == "__main__":
    classify_pitch()
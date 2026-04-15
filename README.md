# Timbre-Disentangled Fair Singing (TDFS) Implementation

This repository contains the official Python implementation for the paper:  
**"Decoupling Timbre: Fair Singing Voice Detection via Latent Disentanglement"**

The TDFS architecture utilizes an adversarial Gradient Reversal Layer (GRL) and Alpha Annealing to decouple target vocal features from confounding instrumental background timbres.

## 1. Requirements & Dependencies
* **Python:** 3.10+
* **Deep Learning:** `torch`, `torchaudio`
* **Signal Processing:** `librosa`, `stempeg` (for MUSDB18 MP4 support)
* **Data & Math:** `pandas`, `numpy`, `scikit-learn`
* **Utilities:** `tqdm`, `matplotlib`

## 2. Technical Specifications
* **Audio Sampling:** 24kHz Mono WAV.
* **Feature Representation:** Log-Mel Spectrograms configured to match Neural Vocoder (BigVGAN) specifications.
* **Optimization:** GPU-accelerated Mel-spectrogram processing is integrated into the training scripts to prevent CPU-bottlenecking (GPU Starvation).

## 3. Module Descriptions

### **Data Processing & Synthesis**
* **`extract_musdb18.py`**: A unified, fault-tolerant pipeline tailored for the MUSDB18 MP4 STEM format; extracts vocals and calculates timbre from instrumental streams.
* **`recover_musdb18_audio.py`**: Rebuilds the evaluation dataset directly from the original `musdb18.zip` by synthesizing 300 standardized WAV files.
* **`synthesize_musdb18.py`**: Bypasses Drive I/O bottlenecks by synthesizing the control set directly on local VM storage for high-speed training.
* **`feature_extractor.py`**: Standardizes audio to 24kHz and performs $f_0$ pitch-tracking using the YIN algorithm with a resumable checkpoint system.
* **`rebalance_pitch.py`**: Utility to resolve initial class imbalances by dynamically calculating the dataset-specific median $f_0$.
* **`dataset_loader.py`**: Custom PyTorch loader supporting on-the-fly Mel-Spectrogram computation and contrastive batch formatting.

### **Model Architectures & Training**
* **`tdfs_model.py`**: Implements the TDFS architecture with the Gradual Alpha Annealing schedule ($0.0 \to 1.0$).
* **`baseline_svd.py`**: Trains the Vanilla CNN baseline; includes a `pos_weight` (6.2) adjustment to handle intrinsic class imbalance.
* **`static_alpha_ablation.py`**: Demonstrates feature extractor collapse when using static adversarial penalties without annealing.

### **Evaluation & Fairness Audits**
* **`evaluate_musdb18_tdfs.py`**: Verifies zero-shot generalization and sustained fairness metrics on the unseen MUSDB18 dataset.
* **`evaluate_baseline.py`**: Audits the Vanilla model across $f_0$-derived gender proxies and timbre subgroups.
* **`frequency_band_probe.py`**: Intercepts CNN feature maps to perform sub-band linear probing (Low, Mid, and High frequencies).
* **`hpss_baseline.py`**: Implements a classical Signal Processing (Harmonic-Percussive Source Separation) comparison.
* **`advanced_metrics.py`**: Calculates Subgroup Expected Calibration Error (ECE) and Vocal-to-Accompaniment Ratios (VAR).
* **`generate_spectrogram_fig.py`**: Generates the spectral masking visualization for the manuscript figures.

## 4. Recommended Execution Sequence
1. **Source Recovery**: `recover_musdb18_audio.py` $\to$ `extract_musdb18.py`
2. **Synthesis & Pitch**: `synthesize_musdb18.py` $\to$ `feature_extractor.py` $\to$ `rebalance_pitch.py`
3. **Training**: `baseline_svd.py` $\to$ `tdfs_model.py`
4. **Research Analysis**: `evaluate_musdb18_tdfs.py` $\to$ `frequency_band_probe.py` $\to$ `advanced_metrics.py`

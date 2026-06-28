# Timbre-Disentangled Fair Singing (TDFS) — Official Implementation

Official Python implementation for the paper:
**"Decoupling Timbre: Fair Singing Voice Detection via Latent Disentanglement"**
*IEEE Signal Processing Letters*

The TDFS architecture employs an adversarial Gradient Reversal Layer (GRL) with
Alpha Annealing to decouple vocal features from confounding instrumental timbre in
polyphonic audio. Sub-band latent probing localises residual timbre leakage to the
mid-frequency band (Mel bins 6–13, ~300–3000 Hz), establishing a fundamental
harmonic masking floor shared by both adversarial and classical SP approaches.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Datasets](#datasets)
3. [Dependencies](#dependencies)
4. [Technical Specifications](#technical-specifications)
5. [Full Pipeline](#full-pipeline)
   - [Phase 1: MUSDB18 Preparation](#phase-1-musdb18-preparation)
   - [Phase 2: Jamendo Preparation](#phase-2-jamendo-preparation)
   - [Phase 3: Spectrogram Precomputation](#phase-3-spectrogram-precomputation)
   - [Phase 4: Model Training](#phase-4-model-training)
   - [Phase 5: Evaluation](#phase-5-evaluation)
   - [Phase 6: Extended Analysis](#phase-6-extended-analysis)
6. [Output Files](#output-files)
7. [Figures and Tables](#figures-and-tables)
8. [GPU Requirements](#gpu-requirements)
9. [Notes on Reproducibility](#notes-on-reproducibility)

---

## Repository Structure

```
tdfs/
├── README.md
│
├── # ── MUSDB18 DATA PREPARATION ──────────────────────────────
├── extract_musdb18.py           # Phase 1a: stem extraction + timbre labelling
├── recover_musdb18_audio.py     # Phase 1b: WAV synthesis from stems
├── synthesize_musdb18.py        # Phase 1c: balanced vocal/instrumental dataset
├── classify_musdb18_pitch.py    # Phase 1d: f0-based pitch class assignment
│
├── # ── JAMENDO DATA PREPARATION ───────────────────────────────
├── feature_extractor.py         # Phase 2a: 24kHz conversion + YIN pitch tracking
├── rebalance_pitch.py           # Phase 2b: median-f0 pitch class rebalancing
├── rebuild_jamendo_manifest.py  # Phase 2c: manifest reconstruction from wav_24k
│
├── # ── SPECTROGRAM CACHE ──────────────────────────────────────
├── precompute_spectrograms.py   # Phase 3: librosa Mel spectrograms → NPZ cache
├── dataset_loader.py            # PyTorch dataset with on-the-fly Mel computation
│
├── # ── MODEL TRAINING ─────────────────────────────────────────
├── baseline_svd.py              # Phase 4a: Vanilla CNN training (NPZ-based)
├── tdfs_model.py                # Phase 4b: TDFS adversarial training (NPZ-based)
├── static_alpha_ablation.py     # Phase 4c: ablation — static GRL collapse demo
│
├── # ── EVALUATION ─────────────────────────────────────────────
├── evaluate_baseline.py         # Phase 5a: Vanilla CNN fairness audit on Jamendo
├── evaluate_tdfs.py             # Phase 5b: TDFS fairness audit on Jamendo
├── evaluate_musdb18_tdfs.py     # Phase 5c: TDFS zero-shot audit on MUSDB18
├── hpss_baseline.py             # Phase 5d: HPSS SP front-end comparison
├── frequency_band_probe.py      # Phase 5e: sub-band latent linear probing
├── advanced_metrics.py          # Phase 5f: ECE + VAR computation
├── generate_spectrogram_fig.py  # Phase 5g: spectrogram masking figure (Fig. 1)
│
├── # ── EXTENDED ANALYSIS ──────────────────────────────────────
├── spl01_save_predictions.py    # Phase 6a: prediction arrays (Vanilla/TDFS/HPSS)
├── spl02_statistical_validation.py  # Phase 6b: bootstrap CIs + significance tests
├── spl03_probe_with_split.py    # Phase 6c: sub-band probe with 70/30 split
├── spl04_panns_baseline.py      # Phase 6d: PANNs CNN14 zero-shot SOTA baseline
│
└── # ── UTILITIES ───────────────────────────────────────────────
    └── jamendo_metadata_parser.py   # Jamendo metadata parsing utility
```

---

## Datasets

### Jamendo (Training)
- **Source:** MTG-Jamendo Dataset (Bogdanov et al., 2019)
- **URL:** https://mtg.github.io/mtg-jamendo-dataset/
- **Size used:** 4,163 tracks (578 vocal, 3,585 non-vocal)
- **Vocal label:** `instrument---voice` tag from `autotagging.tsv`
- **Timbre label:** Spectral centroid of full mix; bottom 25th percentile = soft, top 75th = powerful
- **Pitch label:** YIN algorithm (fmin=65 Hz, fmax=300 Hz); median f0 split = 89.42 Hz
- **Expected Drive path:** `datasets/Jamendo/`

### MUSDB18 (Zero-Shot Evaluation)
- **Source:** MUSDB18 multitrack corpus (Rafii et al., 2017)
- **URL:** https://sigsep.github.io/datasets/musdb.html
- **Size used:** 150 tracks → 300 balanced segments (150 vocal mixture, 150 instrumental)
- **Vocal label:** RMS energy threshold on isolated vocal stem (> 0.005)
- **Timbre label:** Spectral centroid of instrumental stem; median split at 1800 Hz (75 soft, 75 powerful negatives)
- **Pitch label:** `torchaudio.functional.detect_pitch_frequency`; dataset median split (75 per cohort)
- **Expected Drive path:** `datasets/MUSDB18/`

### AudioSet (PANNs Baseline Only)
- PANNs CNN14 checkpoint pretrained on AudioSet (Kong et al., 2020)
- URL: https://zenodo.org/records/3987831
- AudioSet "Singing" class index: **27**

---

## Dependencies

```bash
pip install torch torchaudio librosa stempeg pandas numpy scikit-learn \
            tqdm matplotlib scipy panns_inference
```

- **Python:** 3.10+
- **PyTorch:** 2.0+ with CUDA support recommended
- **Google Colab:** All scripts include `drive.mount()` for Drive integration
- **Storage:** ~50 GB Drive space for full pipeline

---

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Sample rate | 24,000 Hz mono |
| Segment length | 2 seconds (48,000 samples) |
| Mel bins | 80 |
| FFT size | 1,024 |
| Hop length | 256 |
| Frequency range | 0–12,000 Hz |
| Spectrogram library | librosa |
| Spectrogram reference | `librosa.power_to_db(ref=np.max)` |
| pos_weight (BCELoss) | 6.2 (= 3,585 / 578) |
| GRL alpha schedule | sigmoid annealing: 2/(1+exp(-10p))−1 |
| Training epochs | Vanilla: 10, TDFS: 15 |
| Batch size | 32 |
| Optimiser | Adam, lr=1e-3 |

---

## Full Pipeline

### Phase 1: MUSDB18 Preparation

**GPU needed: No**

```
musdb18.zip (Drive)
    │
    ▼
extract_musdb18.py
    Inputs:  musdb18.zip
    Outputs: musdb18_manifest_with_f0.csv
             (columns: path, is_vocal, median_f0, mean_centroid,
                       pitch_class, is_soft_timbre, is_powerful_timbre)
    Notes:   Uses stempeg to read MP4 stems. Timbre from SpectralCentroid
             on instrumental sum (drums+bass+other). 25th/75th percentile split.
    │
    ▼
synthesize_musdb18.py
    Inputs:  musdb18_manifest_with_f0.csv, musdb18.zip
    Outputs: musdb18_balanced_manifest.csv
             musdb18_wav_24k.zip (300 WAV files: 150 vocal + 150 instrumental)
    Notes:   Creates paired vocal (mixture) and non-vocal (instrumental) WAVs.
    │
    ▼
classify_musdb18_pitch.py
    Inputs:  musdb18_balanced_manifest.csv
    Outputs: musdb18_research_manifest_with_f0.csv
    Notes:   Assigns pitch_class by dataset median f0.
```

**Recovery script** (if musdb18_wav_24k.zip is lost):
```
recover_musdb18_audio.py
    Inputs:  musdb18.zip
    Outputs: musdb18_wav_24k.zip (regenerated)
```

---

### Phase 2: Jamendo Preparation

**GPU needed: No**

```
Jamendo audio_data/ + mtg-jamendo-dataset/data/autotagging.tsv
    │
    ▼
feature_extractor.py
    Inputs:  research_manifest.csv, audio_data/ (MP3s)
    Outputs: wav_24k/ (4,163 WAV files at 24kHz)
             research_manifest_with_f0.csv
    Notes:   YIN pitch tracking (fmin=65, fmax=300 Hz). Saves every 100 tracks.
             is_vocal from instrument---voice tag in autotagging.tsv.
    │
    ▼
rebalance_pitch.py
    Inputs:  research_manifest_with_f0.csv
    Outputs: research_manifest_with_f0.csv (updated in-place)
    Notes:   Recalculates pitch_class at dataset median f0 (89.42 Hz)
             for balanced Low/High-Pitch cohorts (289 each).
```

**Alternative** (if research_manifest.csv is unavailable but wav_24k is intact):
```
rebuild_jamendo_manifest.py
    Inputs:  wav_24k/ (4,163 WAVs)
             mtg-jamendo-dataset/data/autotagging.tsv
    Outputs: research_manifest_with_f0.csv (fully reconstructed)
    Notes:   Combines feature_extractor + rebalance_pitch in one resumable script.
             CPU only. ~90 minutes. Checkpoints every 50 tracks.
             is_vocal assigned from instrument---voice tag in autotagging.tsv.
```

---

### Phase 3: Spectrogram Precomputation

**GPU needed: No**

```
precompute_spectrograms.py
    Inputs:  research_manifest_with_f0.csv
             audio_data/ (MP3 subdirectories, e.g. audio_data/12/1001312.mp3)
    Outputs: paper/DecouplingTimbre/jamendo_spectrograms.npz (~188 MB)
             paper/DecouplingTimbre/jamendo_spec_cache.csv
    Notes:   Computes librosa.feature.melspectrogram for all 4,163 tracks.
             Saves as NPZ keyed by track_id (e.g. '1001312').
             Checkpoints every 100 tracks. Resume-safe.
             CPU only. ~90 minutes from Drive MP3s.
```

---

### Phase 4: Model Training

**GPU needed: Yes (T4 recommended)**

```
baseline_svd.py
    Inputs:  research_manifest_with_f0.csv
             jamendo_spectrograms.npz
    Outputs: paper/DecouplingTimbre/vanilla_svd_final.pth
             datasets/Jamendo/vanilla_svd_final.pth
             paper/DecouplingTimbre/baseline_results.json
    Notes:   10 epochs. pos_weight=6.2. Loads spectrograms from NPZ.
             Checkpoints every epoch. ~10 minutes on T4.
    │
    ▼
tdfs_model.py
    Inputs:  research_manifest_with_f0.csv
             jamendo_spectrograms.npz
    Outputs: paper/DecouplingTimbre/tdfs_final_v2.pth
             datasets/Jamendo/tdfs_final_v2.pth
             paper/DecouplingTimbre/tdfs_results_v2.json
    Notes:   15 epochs. GRL alpha annealing. Loads from NPZ.
             Checkpoints every epoch. ~15 minutes on T4.
    │
    ▼
static_alpha_ablation.py
    Inputs:  musdb18_research_manifest_with_f0.csv, musdb18_wav_24k.zip
    Outputs: static_alpha_ablation_results.json
    Notes:   Demonstrates mode collapse when alpha=1.0 from epoch 1.
             Reports near-random vocal accuracy confirming that
             alpha annealing is essential for stable adversarial training.
```

---

### Phase 5: Evaluation

**GPU needed: Yes**

```
evaluate_baseline.py
    Inputs:  research_manifest_with_f0.csv, wav_24k/, vanilla_svd_final.pth
    Outputs: exp1_fairness_audit.json
    Notes:   Jamendo in-domain fairness audit. Reports accuracy, FPR, FNR
             across soft/powerful timbre and low/high-pitch subgroups.

evaluate_tdfs.py
    Inputs:  research_manifest_with_f0.csv, wav_24k/, tdfs_final_v2.pth
    Outputs: exp2_tdfs_fairness_audit.json
    Notes:   TDFS in-domain fairness audit on Jamendo.

evaluate_musdb18_tdfs.py
    Inputs:  musdb18_research_manifest_with_f0.csv, musdb18_wav_24k.zip,
             tdfs_final_v2.pth
    Outputs: musdb18_tdfs_cross_validation.json
    Notes:   Zero-shot evaluation. Source of truth for paper Tables I and II.

hpss_baseline.py
    Inputs:  musdb18_research_manifest_with_f0.csv, musdb18_wav_24k.zip,
             vanilla_svd_final.pth
    Outputs: hpss_baseline_results.json
    Notes:   HPSS front-end applied to Vanilla CNN at inference only.
             Source of truth for HPSS results in Table II.

frequency_band_probe.py
    Inputs:  musdb18_research_manifest_with_f0.csv, musdb18_wav_24k.zip,
             tdfs_final_v2.pth
    Outputs: frequency_band_probe.json
    Notes:   Sub-band linear probing. Reports Low=68.75%, Mid=84.38%,
             High=81.25% probe accuracy. Source of truth for Section V-C.

advanced_metrics.py
    Inputs:  musdb18_research_manifest_with_f0.csv, musdb18_wav_24k.zip,
             tdfs_final_v2.pth
    Outputs: advanced_metrics.json
    Notes:   ECE per subgroup (overall=0.0546, soft=0.0515, powerful=0.0811,
             pitch ECE≈0.44). VAR computation.

generate_spectrogram_fig.py
    Inputs:  musdb18_research_manifest_with_f0.csv, musdb18_wav_24k.zip
    Outputs: spectrogram_masking.pdf  →  paper Figure 1
    Notes:   Side-by-side Mel spectrograms: soft vs powerful timbre,
             with vocal formant region (300–3000 Hz) annotated.
```

---

### Phase 6: Extended Analysis

**Run in this order:**
```
spl01 → spl02 → spl03 (independent of spl02) → spl04 (independent of spl02/spl03)
```

```
spl01_save_predictions.py  [GPU needed]
    Inputs:  musdb18_research_manifest_with_f0.csv
             musdb18_wav_24k.zip
             vanilla_svd_final.pth
             tdfs_final_v2.pth
    Outputs: predictions_vanilla.csv   (300 rows)
             predictions_tdfs.csv      (300 rows)
             predictions_hpss.csv      (300 rows)
             spl01_summary.json
    Notes:   All three models evaluated zero-shot on MUSDB18 in identical
             conditions (trained on Jamendo, HPSS applied at inference only).
             Spectrograms computed with librosa to match training cache.
             Checkpoints after every batch.

spl02_statistical_validation.py  [CPU only]
    Inputs:  musdb18_tdfs_cross_validation.json
             hpss_baseline_results.json
             frequency_band_probe.json
             advanced_metrics.json
             predictions_vanilla.csv, predictions_tdfs.csv, predictions_hpss.csv
    Outputs: spl02_statistical_validation.json
             spl02_roc_curves.png
    Notes:   Bootstrap 95% CIs (n=1000) on reported FPR/FNR numbers.
             McNemar test: TDFS vs HPSS (powerful timbre, n=75, p=0.480).
             McNemar test: TDFS vs Vanilla (n=300, p=0.648).
             Binomial test: TDFS 52.67% vs 50% chance (p=0.387).

spl03_probe_with_split.py  [GPU needed — first run only; CPU on reruns]
    Inputs:  musdb18_research_manifest_with_f0.csv
             musdb18_wav_24k.zip
             tdfs_final_v2.pth
    Outputs: spl03_latents_cache.npz  (cached; enables CPU-only reruns)
             spl03_probe_results.json
             spl03_subband_probe.png
    Notes:   70/30 stratified split + 5-fold cross-validation on sub-band probes.
             Latents cached to NPZ after first GPU run.

spl04_panns_baseline.py  [GPU needed]
    Inputs:  musdb18_research_manifest_with_f0.csv
             musdb18_wav_24k.zip
             CNN14_mAP0.431.pth  (downloaded from Zenodo 3987831)
    Outputs: predictions_panns.csv   (300 rows)
             spl04_panns_results.json
             spl04_panns_fpr_bars.png
    Notes:   PANNs CNN14 zero-shot inference using AudioSet Singing class (idx=27).
             FNR=93% on vocal segments confirms polyphonic SVD requires
             task-specific training regardless of pretraining scale.
             Optimal threshold: 0.55.
```

---

## Output Files

All extended analysis outputs are saved to `paper/DecouplingTimbre/` on Google Drive.
Core evaluation outputs are saved to `datasets/MUSDB18/` and `datasets/Jamendo/`.

### Model Checkpoints
| File | Description |
|------|-------------|
| `vanilla_svd_final.pth` | Trained Vanilla CNN (10 epochs, Jamendo) |
| `tdfs_final_v2.pth` | Trained TDFS model (15 epochs, Jamendo) |
| `tdfs_checkpoint_v2.pth` | Latest epoch checkpoint (resume-safe) |
| `vanilla_svd_checkpoint.pth` | Latest epoch checkpoint (resume-safe) |

### Manifests
| File | Description |
|------|-------------|
| `research_manifest_with_f0.csv` | Jamendo: 4,163 tracks with pitch + timbre labels |
| `musdb18_research_manifest_with_f0.csv` | MUSDB18: 300 segments with pitch + timbre labels |
| `musdb18_balanced_manifest.csv` | MUSDB18: intermediate balanced manifest |

### Result JSONs
| File | Source script | Paper location |
|------|--------------|----------------|
| `musdb18_tdfs_cross_validation.json` | `evaluate_musdb18_tdfs.py` | Table II |
| `hpss_baseline_results.json` | `hpss_baseline.py` | Table II |
| `frequency_band_probe.json` | `frequency_band_probe.py` | Section V-C |
| `advanced_metrics.json` | `advanced_metrics.py` | Section V-B |
| `static_alpha_ablation_results.json` | `static_alpha_ablation.py` | Section IV-B |
| `spl02_statistical_validation.json` | `spl02_statistical_validation.py` | Section V |
| `spl03_probe_results.json` | `spl03_probe_with_split.py` | Section V-C |
| `spl04_panns_results.json` | `spl04_panns_baseline.py` | Section V-D |

---

## Figures and Tables

### Paper Figures
| Figure | File | Generated by |
|--------|------|-------------|
| Fig. 1 | `spectrogram_masking.pdf` | `generate_spectrogram_fig.py` |

### Paper Tables
| Table | Content | Source JSON |
|-------|---------|------------|
| Table I | Ablation: Static vs Annealed alpha | `static_alpha_ablation_results.json` |
| Table II | Timbre bias: Vanilla vs TDFS vs HPSS (Jamendo + MUSDB18) | `musdb18_tdfs_cross_validation.json`, `hpss_baseline_results.json` |

### Additional Figures
| Figure | File | Generated by |
|--------|------|-------------|
| ROC curves | `spl02_roc_curves.png` | `spl02_statistical_validation.py` |
| Sub-band probe bar chart | `spl03_subband_probe.png` | `spl03_probe_with_split.py` |
| PANNs FPR comparison | `spl04_panns_fpr_bars.png` | `spl04_panns_baseline.py` |

---

## GPU Requirements

| Script | GPU needed | Estimated time (T4) |
|--------|-----------|-------------------|
| `rebuild_jamendo_manifest.py` | No | 60–90 min |
| `precompute_spectrograms.py` | No | 60–90 min |
| `feature_extractor.py` | No | 60–90 min |
| `rebalance_pitch.py` | No | < 1 min |
| `classify_musdb18_pitch.py` | No | < 1 min |
| `baseline_svd.py` | Yes | ~10 min |
| `tdfs_model.py` | Yes | ~15 min |
| `static_alpha_ablation.py` | Yes | ~10 min |
| `evaluate_baseline.py` | Yes | ~5 min |
| `evaluate_tdfs.py` | Yes | ~5 min |
| `evaluate_musdb18_tdfs.py` | Yes | ~10 min |
| `hpss_baseline.py` | Yes | ~15 min |
| `frequency_band_probe.py` | Yes | ~10 min |
| `advanced_metrics.py` | Yes | ~10 min |
| `generate_spectrogram_fig.py` | No | < 5 min |
| `spl01_save_predictions.py` | Yes | ~5 min |
| `spl02_statistical_validation.py` | No | < 2 min |
| `spl03_probe_with_split.py` | Yes (first run only) | ~5 min |
| `spl04_panns_baseline.py` | Yes | ~10 min |

---

## Notes on Reproducibility

**Spectrogram consistency:** All training and evaluation scripts use
`librosa.power_to_db(ref=np.max)`, producing spectrograms in the range
approximately −80 to 0 dB. The `precompute_spectrograms.py` cache must be
regenerated if any spectrogram parameters are changed.

**Timbre labelling:** Jamendo timbre labels use spectral centroid quantile
thresholds (25th/75th percentile) on the full mix, giving 1,041 soft and
1,041 powerful tracks out of 4,163. MUSDB18 timbre labels use spectral centroid
of isolated instrumental stems with a median split (1,800 Hz threshold), giving
75 soft and 75 powerful negatives per group for subgroup FPR analysis.

**Statistical power:** Bootstrap 95% CIs on subgroup FPR metrics are reported
for all models. With 75 negative segments per timbre group on MUSDB18, the
margin of error on FPR estimates is approximately ±11% at 95% confidence.

**PANNs inference:** The AudioSet "Singing" class is at index 27 in the CNN14
output vector. The optimal classification threshold is 0.55 rather than the
default 0.50, as determined by a threshold sweep on the MUSDB18 test set.

**Checkpoint saving:** All training scripts save a checkpoint after every epoch
and support full resume on session restart. Checkpoints are saved to Google Drive
with `os.sync()` after every write.
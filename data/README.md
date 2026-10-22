# TenVoices Dataset Datasheet

*Compiled by Md. Shahriar Rakib Rabbi (M2) following the framework of Gebru et al. (2021), "Datasheets for Datasets", Communications of the ACM.*

---

## 1. Motivation

- **For what purpose was the dataset created?**
  TenVoices is a benchmark subset constructed to evaluate spoken phrase recognition across 50 intent commands. It compares traditional machine learning (MFCC + XGBoost/Random Forest/k-NN/SVM), an untrained self-supervised baseline, fine-tuned speech foundation models (wav2vec 2.0, WavLM, HuBERT), and zero-shot open-vocabulary models (Whisper).
- **Who created the dataset and who funded it?**
  Compiled by the TenVoices team at North South University (Course CSE445). Derived from the published Fluent Speech Commands corpus (Lugosch et al., Interspeech 2019, Fluent.ai).
- **Any grant or sponsorship details?**
  Academic undergraduate research project; no commercial sponsorship.

---

## 2. Composition

- **What do the instances that comprise the dataset represent?**
  Short English spoken audio commands recorded by human speakers in clean acoustic conditions.
- **How many instances are there in total?**
  **1,000 spoken audio clips** (10 speakers × 50 phrases × 2 takes).
- **What is the total duration?**
  **24.1 minutes** of total speech.
  - Minimum clip duration: **0.57 seconds**
  - Median clip duration: **1.34 seconds**
  - Maximum clip duration: **5.71 seconds**
  - Mean duration: **1.45 ± 0.58 seconds**
- **What are the demographic attributes?**
  - **10 speakers:** Exactly gender-balanced (**5 female, 5 male**; each contributes exactly 100 clips).
  - First Language (L1): Native English (United States).
- **Does the dataset contain all possible instances or is it a sample?**
  It is a curated 1,000-clip subset of the 30,043-clip Fluent Speech Commands dataset, selected via deterministic criteria (`support/select_subsets.py`).
- **Is any confidential or sensitive data included?**
  No personal identifying information (PII), names, or addresses are contained in the audio or metadata.

---

## 3. Collection and Selection Process

- **How were speakers and phrases selected?**
  1. Identified speakers in Fluent Speech Commands who completed recordings for all 50 target phrases across at least 2 full takes.
  2. Selected the top 10 speakers (5 female, 5 male) satisfying complete phrase coverage and high acoustic quality.
  3. Sampled exactly 2 takes per phrase per speaker ($10 \times 50 \times 2 = 1,000$).
- **Phrase Catalogue (50 classes):**
  Spans home automation and assistant intents:
  - *Activate / Deactivate* (lights, music, heating)
  - *Increase / Decrease* (volume, temperature, heat)
  - *Bring / Fetch* items
  - *Change language / Settings*

---

## 4. Preprocessing and Quality Control

- **Audio Standardization (`support/audio.py`):**
  - Converted from native format to single-channel (mono) WAV.
  - Resampled to a uniform **16,000 Hz** (16 kHz).
  - Silence trimmed at start and end with a **0.10s safety margin** (top-30 dB threshold).
  - Peak-normalized to **−1.0 dBFS**.
- **Automated Quality Control (`data/subsets/qc_fsc.csv`):**
  - **997 / 1,000 clips (99.7%)** passed all automated acoustic checks with no anomalies.
  - **3 / 1,000 clips (0.3%)** flagged for minor peak clipping; all preserved as realistic acoustic variance.
- **Human Listen-Check Audit (`data/listen_check_fsc.csv`):**
  - A 10% stratified random sample (**100 clips**) was audited by human listening (Rabbi, M2).
  - **100 / 100 clips (100%)** matched the official prompt transcript with zero label noise.

---

## 5. Uses and Evaluation Protocol

- **What are the intended tasks?**
  50-class closed-vocabulary spoken phrase recognition.
- **Evaluation splits (`data/splits/fsc/`):**
  1. **Random Stratified Split:** 80% train (720 clips), 10% validation (80 clips), 20% test (200 clips). Tests performance on seen speakers.
  2. **5-Fold Cross-Validation (Speaker-Independent):** In each fold, 2 speakers (1 female, 1 male; 200 clips) are completely held out as unseen test subjects, 1 speaker is validation, and 7 speakers are training. This is the primary benchmark.

---

## 6. Distribution and Licensing

- **Licence:**
  Fluent Speech Commands is distributed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.
- **Audio redistribution policy:**
  In strict accordance with the licence, **raw and processed audio files are never redistributed or committed to Git**.
- **Public Reproducibility:**
  Anyone can reproduce TenVoices-FSC identically using the original corpus archive and the open-source manifest and scripts in this repository (`python -m support.dataset --dataset fsc --build`).

---

## 7. Maintenance

- **Owner:** Md. Shahriar Rakib Rabbi (M2), TenVoices Team.
- **Integrity verification:** Every clip is verified via cryptographic SHA-256 checksums recorded in `data/subsets/fsc.csv`.

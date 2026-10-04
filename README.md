# 🎙️ TenVoices: Spoken Phrase Recognition on a Self-Collected 10-Speaker Dataset

### Speech Recognition Systems · Course Project
**University Project | North South University (NSU)**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Planning-lightgrey)

| Course | Section | Group | Instructor |
| :---: | :---: | :---: | :---: |
| *TBD* | *TBD* | *TBD* | *TBD* |

<p align="center">
  <img src="images/poster.png" alt="TenVoices project poster" width="850">
</p>

<p align="center">
  <em>Project poster (planning edition).</em>
</p>

> 📋 **Status: planning stage.** This repository contains:
> - the project plan
> - the data-collection protocol
> - the phrase list
> - the poster
>
> Code, data and results will be added week by week (see the roadmap below).

## 📖 Overview
This project builds a **speech recognition system** that recognises which of **50 diverse phrases** a person has spoken. We record our own dataset, **TenVoices**, from **10 speakers**, and use it to compare traditional machine learning with a fine-tuned self-supervised speech model.

The comparison has three stages:
* **Traditional Machine Learning:** MFCC features with k-NN, SVM, Random Forest and XGBoost.
* **Deep Learning baseline:** wav2vec 2.0 with an untrained classification head, which shows the chance level.
* **Our approach:** wav2vec 2.0 **fine-tuned** on TenVoices for 3, 8 and 15 epochs, with white-noise augmentation for robustness.

Every model is tested two ways:
* on a random 80/20 split, where the speakers have been seen in training;
* with **speaker-independent 5-fold cross-validation**, where the test speakers have never been heard. This is the headline result.

---

## 📊 Dataset
We are recording our own dataset, **TenVoices v1**.
* **Speakers:** 10 adults (team members + volunteers), target 5 female / 5 male. They read English with a natural Bangladeshi accent.
* **Phrases (50 classes, 10 per category):**
    * *Voice commands*, e.g. "Turn off the fan."
    * *Numbers, times and money*, e.g. "Please pay five hundred and fifty taka at the counter."
    * *Everyday sentences and questions*, e.g. "What time does the pharmacy close tonight?"
    * *Bangladeshi names and places*, e.g. "I took a bus from Mohakhali to Sylhet."
    * *Phonetically rich Harvard sentences*, e.g. "The birch canoe slid on the smooth planks."
* **Confusable pairs on purpose:** "turn **on** / **off** the lights", "**increase** / **decrease** the volume", "**play** / **stop** the music".
* **Recordings:** every speaker reads all 50 phrases **3 times** (shuffled order each round), plus 10 phrases once more in a real noisy place. That is 160 clips per speaker and **~1,600 clips (~1 hour)** in total.
* **Format:** WAV recorded at 48 kHz, processed to 16 kHz mono. Speakers are identified only by pseudonymous IDs (S01–S10).
* **Consent & licence:** every speaker signs a consent form. Recordings of speakers who opt in will be released under **CC BY 4.0**.

Phrase list: [data/prompts/phrases.csv](data/prompts/phrases.csv) · Recording rules: [docs/DATA_COLLECTION_PROTOCOL.md](docs/DATA_COLLECTION_PROTOCOL.md)

---

## 🧠 Methodology

### 1. Traditional Machine Learning (Baseline)
We extract **MFCCs (Mel-Frequency Cepstral Coefficients)** from each clip to build a tabular dataset.
* **Feature Extraction:** Librosa, 13 MFCCs + Δ + ΔΔ, each summarised by mean and standard deviation (78 values), then standardised.
* **Models:** k-NN, SVM, Random Forest and XGBoost, with hyper-parameters tuned on validation data.

### 2. Deep Learning Baseline (Untrained)
* `facebook/wav2vec2-base` with a new classification head for the 50 phrases, evaluated **without training**. It shows the chance level of about 2%.

### 3. Our Approach: Fine-Tuning wav2vec 2.0
We fine-tune a pre-trained **wav2vec 2.0 Base** model (`facebook/wav2vec2-base`) directly on the raw audio.
* **Architecture:** CNN feature encoder + Transformer context network + classification head (50 classes).
* **Training (planned):**
    * Optimizer: AdamW, learning rate 3e-5, 10% warm-up
    * Epochs: 3, 8 and 15 (compared)
    * Batch size: 4, accumulated to 8
    * **Augmentation:** white noise added to half of the training clips, so the model learns to ignore noise
    * **Model selection:** the best epoch is chosen on validation data, never on the test set
* **Input length:** clips are padded to 6 s, so no phrase is ever cut off.
* **Hardware:** Google Colab GPU (T4) or Apple Silicon (MPS).

---

## 📏 Evaluation Protocol
* **Random split (seen speakers):** stratified 80/20 split of the clean clips (seed 42).
* **Speaker-independent CV (unseen speakers):** 5 folds, each with 2 test speakers, 1 validation speaker and 7 training speakers. Every speaker is tested exactly once.
* **Metrics:** Accuracy, Macro F1, Weighted F1, per-class F1, confusion matrices and per-speaker accuracy.
* **Noise tests:** white noise added at 20 / 10 / 5 dB SNR, plus the real noisy recordings.
* **Statistics:** 95% bootstrap confidence intervals, McNemar's test between models, Wilcoxon test over speakers.

---

## 🏆 Results & Performance (planned)
The results will be filled in after the experiments in Weeks 3–5. **Nothing below is a measured result yet.**

| Model | Accuracy (unseen speakers) | Macro F1 | Weighted F1 | Accuracy (random split) | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Wav2Vec2 (Untrained)** | TBD | TBD | TBD | TBD | Chance level without training |
| **k-NN** | TBD | TBD | TBD | TBD | Distance-based MFCC baseline |
| **SVM** | TBD | TBD | TBD | TBD | Margin-based MFCC baseline |
| **Random Forest** | TBD | TBD | TBD | TBD | Tree ensemble on MFCC features |
| **XGBoost** | TBD | TBD | TBD | TBD | Gradient boosting on MFCC features |
| **Wav2Vec2 (Fine-Tuned) 3 epochs** | TBD | TBD | TBD | TBD | Short fine-tuning |
| **Wav2Vec2 (Fine-Tuned) 8 epochs + noise** | TBD | TBD | TBD | TBD | Medium fine-tuning with augmentation |
| **Wav2Vec2 (Fine-Tuned) 15 epochs + noise** | TBD | TBD | TBD | TBD | Full fine-tuning with augmentation |

**Hypotheses** (fixed before testing; reported whether or not they hold):

| ID | Hypothesis |
| :---: | :--- |
| H1 | Fine-tuned wav2vec 2.0 beats the best traditional model on unseen speakers by ≥ 20 accuracy points |
| H2 | Traditional models lose more accuracy than wav2vec 2.0 when moving from seen to unseen speakers |
| H3 | White-noise augmentation improves accuracy on noisy audio without hurting clean accuracy |
| H4 | Longer fine-tuning has diminishing returns (8 → 15 epochs gains less than 3 → 8) |

**Planned figures:**
* Combined results table (`images/all_results_table.png`)
* A confusion matrix per model
* Accuracy vs noise level
* Per-speaker accuracy

---

## 🗓️ Development Roadmap
- Week 1: Planning, repository setup, phrase list, recording tool and pilot recordings.
- Week 2: Dataset collection from 10 speakers, quality control and data splits.
- Week 3: MFCC features and traditional ML models (k-NN, SVM, Random Forest, XGBoost).
- Week 4: wav2vec 2.0: untrained baseline and fine-tuning for 3, 8 and 15 epochs with white noise.
- Week 5: Combined results, seen vs unseen speakers, noise tests, error analysis, inference script and demo.
- Week 6: Report, slides, demo video, poster and final polish.

Tasks, owners, dependencies and progress: [docs/STATUS.md](docs/STATUS.md).

---

## 📁 Repository Structure (planned)
```text
Voices-Speech-Recognition-445/
|-- CLAUDE.md                 instructions for Claude Code (read every session)
|-- main.py                   inference: audio / video / microphone -> phrase
|-- app.py                    Gradio live demo
|-- requirements.txt
|-- configs/                  config.yaml (paths, sample rate, training settings)
|-- support/
|   |-- config.py, audio.py, dataset.py, features.py      data pipeline
|   |-- classical.py          k-NN / SVM / Random Forest / XGBoost runner
|   |-- train_wav2vec2.py     fine-tuning (--epochs, --noise, --split)
|   |-- evaluate.py           one evaluation script for every model
|   |-- metrics.py, stats.py, results_table.py
|   `-- visualization.py      confusion matrices and charts
|-- tools/                    recorder.py, qc_report.py, make_splits.py
|-- notebooks/                Colab notebook for fine-tuning
|-- tests/
|-- data/                     phrase list, metadata, splits (audio is gitignored)
|-- results/                  per-model reports and all_results.csv
|-- docs/                     STATUS.md, PROJECT_PLAN.md, protocol, consent form
|-- poster/poster.html        poster source
|-- images/                   poster, results table, confusion matrices
|-- outputs/                  large generated files (gitignored)
`-- others/                   report, slides, demo video
```

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git
    cd Voices-Speech-Recognition-445
    ```

2.  **Create a virtual environment with Python 3.11 (Recommended):**
    ```bash
    python3.11 -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage
Run every command from the repository root. Scripts in `support/` and `tools/` are run as modules (`python -m ...`) so they can import the shared code. Entry points marked *planned* are documented here once they work.

| Command | Purpose | Planned for |
| :--- | :--- | :---: |
| `python -m unittest discover -s tests -v` | Run the unit tests | ✅ ready |
| `python -m support.config` | Print the settings and resolved paths | ✅ ready |
| `python -m support.audio clip.wav [--out clip_16k.wav]` | Load and pre-process one clip (mono, 16 kHz, trimmed, normalised, padded to 6 s) | ✅ ready |
| `python -m tools.recorder --speaker S03` | Record one speaker session | Week 1 |
| `python -m support.classical --model svm --split folds` | Train and evaluate a traditional model | Week 3 |
| `python -m support.train_wav2vec2 --epochs 15 --noise white --split folds` | Fine-tune wav2vec 2.0 | Week 4 |
| `python -m support.evaluate --run w2v2_15ep_noise` | Accuracy, F1, confusion matrix for one run | Week 4 |
| `python main.py --audio clip.wav` | Predict the phrase in an audio file (also `--video`, `--mic`) | Week 5 |
| `python app.py` | Live microphone demo (Gradio) | Week 5 |

---

## 🔮 Future Work
* **Open-vocabulary recognition:** move from 50 known phrases to free speech-to-text with Whisper or wav2vec 2.0 CTC.
* **Stronger models:** compare with WavLM and XLS-R, which are pre-trained on more varied speech.
* **Bangla phrases:** add Bangla and mixed Bangla–English phrases to TenVoices.
* **More speakers and devices:** grow beyond 10 speakers and measure how results scale.

---

## 🔒 Ethics & Privacy
* **Consent:** every speaker signs a consent form ([docs/consent_form.md](docs/consent_form.md)). Public release is a separate opt-in, and speakers can withdraw before release.
* **Pseudonymity:** no names, phone numbers or student IDs are stored.
* **Storage:** raw audio stays in a private team folder and is never committed to this repository.

---

## 🔄 Team Workflow (Claude Code)
Every member works through Claude Code, one after another:
1. **Pull** the latest `main` and open Claude Code in the repository.
2. **Ask:** *"I'm M2. What is completed?"* Claude reads [CLAUDE.md](CLAUDE.md) and the task board in [docs/STATUS.md](docs/STATUS.md).
3. **Continue:** *"Complete my next tasks."* Claude only starts tasks whose dependencies are done.
4. **Wrap up:** *"Wrap up and push."* Claude runs the tests, updates the task board and handoff log, commits and pushes.
5. The next member pulls and repeats.

---

## 👥 Contributors
* **Tanvir Ahmed** - *team lead (Member 1): data pipeline and shared code, Random Forest, wav2vec 2.0 fine-tuning (15 epochs), inference*
* **Md. Shahriar Rakib Rabbi** - *Member 2: dataset collection and curation, XGBoost, wav2vec 2.0 fine-tuning (8 epochs)*
* **Tanvir Ahmed** - *Member 3: evaluation and results tables, k-NN, wav2vec 2.0 fine-tuning (3 epochs)*
* **Tanvir Ahmed** - *Member 4: demo app, figures and poster, SVM, untrained baseline*

Member 3 and Member 4 are open slots for members who join later; Tanvir Ahmed covers them for now.

---

## 📄 License
* **Code:** [MIT](LICENSE).
* **TenVoices audio:** CC BY 4.0, for speakers who opt in, once released.
* **Phrase text:** written by the team, plus Harvard sentences (IEEE 1969).

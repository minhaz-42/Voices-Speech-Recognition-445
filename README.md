# 🎙️ TenVoices: Spoken Phrase Recognition on Two 10-Speaker Datasets

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
  <em>Project poster (planning edition). It still shows the earlier self-recorded dataset plan and is updated in task T39.</em>
</p>

> 📋 **Status: planning stage.** This repository contains:
> - the project plan and the dataset guide
> - the shared code written so far (configuration, audio pre-processing, metrics, Whisper pipeline) with unit tests
> - the task board and task cards
> - the poster
>
> Data selection and results will be added week by week (see the roadmap below).

## 📖 Overview
This project builds a **speech recognition system** that recognises which of **50 phrases or sentences** a person has spoken. We evaluate it on **TenVoices**: two **10-speaker** datasets that we select by script from published speech corpora, one of spoken commands and one of non-native English speakers. On both, we compare traditional machine learning, a fine-tuned self-supervised speech model and a general-purpose speech-to-text model.

The comparison has four stages:
* **Traditional Machine Learning:** MFCC features with k-NN, SVM, Random Forest and XGBoost.
* **Deep Learning baseline:** wav2vec 2.0 with an untrained classification head, which shows the chance level.
* **Our approach:** wav2vec 2.0 **fine-tuned** on each dataset for 3, 8 and 15 epochs, with white-noise augmentation for robustness.
* **Open-vocabulary pipeline:** **Whisper** speech-to-text used **zero-shot** (no training on our data). Its transcript is matched to the closest of the 50 phrases, and its word error rate shows how a general-purpose recogniser copes with non-native accents.

Every model is tested on both datasets, in two ways:
* on a random 80/20 split, where the speakers have been seen in training;
* with **speaker-independent 5-fold cross-validation**, where the test speakers have never been heard. This is the headline result.

---

## 📊 Datasets
We do not record our own audio. **TenVoices** is two 10-speaker subsets (5 female, 5 male each) of published datasets, selected by a script with fixed rules, so anyone with the source data gets the same clips.

| | **TenVoices-FSC** (main) | **TenVoices-L2** (accents) |
| :--- | :--- | :--- |
| **Source** | [Fluent Speech Commands](https://fluent.ai/research/fluent-speech-commands/) (Lugosch et al., Interspeech 2019) | [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) v5.0 (Zhao et al., Interspeech 2018) |
| **What is said** | Smart-home commands, e.g. "turn on the lights in the kitchen" | Read sentences from the CMU ARCTIC prompts |
| **Speakers** | 10 of the 97, chosen by script | 10 non-native speakers: the 4 Hindi-L1 speakers, plus a female and a male each with Arabic, Mandarin and Spanish as first language |
| **Classes** | 50 commands, including confusable pairs ("turn **on** / **off** the lights", "**increase** / **decrease** the volume") | 50 sentences read by all 10 speakers |
| **Clips** | 2 takes per speaker and command, up to 1,000 | 1 per speaker and sentence, 500 |
| **Licence** | CC BY-NC-ND 4.0, academic research only | CC BY-NC 4.0 |

* **Why two datasets:** short commands with confusable words, and longer sentences from speakers with different accents. Hindi is the accent in L2-ARCTIC closest to Bangladeshi English.
* **Format:** 16 kHz mono after pre-processing. Speakers keep the datasets' own anonymous IDs.
* **Not in this repository:** the audio. We publish only the subset lists, the selection script and checksums; the audio stays with its publishers.

Sources, licences, download steps and selection rules: [docs/DATASETS.md](docs/DATASETS.md)

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

### 4. Open-Vocabulary Pipeline: Whisper (Zero-Shot)
* **Model:** `openai/whisper-small`, transcribing English, with **no training** on TenVoices.
* **Phrase matching:** the transcript and the 50 phrases are normalised the same way (case, punctuation, numbers), and the phrase with the smallest word-level edit distance is the prediction. Whisper then fits the same tables and tests as every other model.
* **Extra output:** the transcript itself, with word and character error rate per clip, per category and per speaker.
* **Demo:** the app shows the transcript, and says "not one of the 50 phrases" when the speech matches none of them.

---

## 📏 Evaluation Protocol
Applied to each dataset separately.
* **Random split (seen speakers):** stratified 80/20 split of the clips (seed 42).
* **Speaker-independent CV (unseen speakers):** 5 folds, each with 2 test speakers, 1 validation speaker and 7 training speakers. Every speaker is tested exactly once. On TenVoices-L2 the two test speakers of a fold share a first language, so three folds test accents never heard in training.
* **Metrics:** Accuracy, Macro F1, Weighted F1, per-class F1, confusion matrices and per-speaker accuracy (per first language on TenVoices-L2); word and character error rate for the Whisper transcripts.
* **Noise tests:** white noise added at 20 / 10 / 5 dB SNR.
* **Statistics:** 95% bootstrap confidence intervals, McNemar's test between models, Wilcoxon test over speakers, Mann–Whitney test between the two datasets.

---

## 🏆 Results & Performance (planned)
The results will be filled in after the experiments in Weeks 3–5. **Nothing below is a measured result yet.**

| Model | FSC: unseen speakers | FSC: random split | L2: unseen speakers | L2: random split | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Wav2Vec2 (Untrained)** | TBD | TBD | TBD | TBD | Chance level without training |
| **k-NN** | TBD | TBD | TBD | TBD | Distance-based MFCC baseline |
| **SVM** | TBD | TBD | TBD | TBD | Margin-based MFCC baseline |
| **Random Forest** | TBD | TBD | TBD | TBD | Tree ensemble on MFCC features |
| **XGBoost** | TBD | TBD | TBD | TBD | Gradient boosting on MFCC features |
| **Wav2Vec2 (Fine-Tuned) 3 epochs** | TBD | TBD | TBD | TBD | Short fine-tuning |
| **Wav2Vec2 (Fine-Tuned) 8 epochs + noise** | TBD | TBD | TBD | TBD | Medium fine-tuning with augmentation |
| **Wav2Vec2 (Fine-Tuned) 15 epochs + noise** | TBD | TBD | TBD | TBD | Full fine-tuning with augmentation |
| **Whisper small (zero-shot) + phrase matching** | TBD | TBD | TBD | TBD | General-purpose speech-to-text, no training |

Accuracy per dataset and split; macro and weighted F1 will be in `results/all_results.csv`.

**Hypotheses** (fixed before testing; reported whether or not they hold):

| ID | Hypothesis |
| :---: | :--- |
| H1 | Fine-tuned wav2vec 2.0 beats the best traditional model on unseen speakers by ≥ 20 accuracy points |
| H2 | Traditional models lose more accuracy than wav2vec 2.0 when moving from seen to unseen speakers |
| H3 | White-noise augmentation improves accuracy on noisy audio without hurting clean accuracy |
| H4 | Longer fine-tuning has diminishing returns (8 → 15 epochs gains less than 3 → 8) |
| H5 | Zero-shot Whisper makes more word errors on non-native speech (TenVoices-L2) than on spoken commands (TenVoices-FSC) *(draft, fixed in the analysis plan)* |

H1–H4 are tested on each dataset separately.

**Planned figures:**
* Combined results table (`images/all_results_table.png`)
* A confusion matrix per model
* Accuracy vs noise level
* Per-speaker accuracy

---

## 🗓️ Development Roadmap
- Week 1: Planning, repository setup, shared code, dataset access.
- Week 2: Subset selection and quality control for both datasets, datasheet and data splits.
- Week 3: MFCC features and traditional ML models (k-NN, SVM, Random Forest, XGBoost) on both datasets.
- Week 4: wav2vec 2.0: untrained baseline and fine-tuning for 3, 8 and 15 epochs with white noise; Whisper zero-shot speech-to-text.
- Week 5: Combined results, seen vs unseen speakers, accents, noise tests, error analysis, inference script and demo.
- Week 6: Report, slides, demo video, poster and final polish.

Tasks, owners, dependencies and progress: [docs/STATUS.md](docs/STATUS.md).

---

## 📁 Repository Structure (planned)
The code that exists today, module by module, is listed in [support/README.md](support/README.md). Folders that only contain a README are placeholders for later tasks.

```text
Voices-Speech-Recognition-445/
|-- AGENTS.md                 project rules for every member and AI assistant
|-- CONTRIBUTING.md           step-by-step workflow, by hand or with any assistant
|-- main.py                   inference: audio / video / microphone -> phrase
|-- app.py                    Gradio live demo
|-- requirements.txt
|-- configs/                  config.yaml (paths, sample rate, training settings)
|-- support/                  shared code; README.md lists every module
|   |-- config.py, audio.py, phrases.py, dataset.py, features.py   data pipeline
|   |-- classical.py          k-NN / SVM / Random Forest / XGBoost runner
|   |-- train_wav2vec2.py     fine-tuning (--epochs, --noise, --split)
|   |-- whisper_asr.py        zero-shot Whisper speech-to-text + phrase matching
|   |-- evaluate.py           one evaluation script for every model
|   |-- metrics.py, stats.py, results_table.py, error_analysis.py
|   `-- visualization.py      spectrograms, MFCCs and result charts
|-- tools/                    select_subsets.py, qc_report.py, make_splits.py
|-- notebooks/                Colab notebook for fine-tuning
|-- tests/
|-- data/                     subset and phrase lists, sources, splits (audio is gitignored)
|-- results/                  per-model, per-dataset reports and all_results.csv
|-- docs/                     STATUS.md, PROJECT_PLAN.md, DATASETS.md
|   |-- progress/             one page per member: tasks, handoff log, faculty update
|   |-- tasks/                task cards: what to build, files, tests, done when
|   `-- literature/           literature notes for the report
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
| `python -m support.whisper_asr clip.wav` | Transcribe a clip with Whisper (zero-shot) and show the 5 nearest phrases | ✅ ready |
| `python -m tools.select_subsets --dataset all` | Build the two 10-speaker subsets from the downloaded datasets | Week 2 |
| `python -m support.classical --dataset fsc --model svm --split folds` | Train and evaluate a traditional model | Week 3 |
| `python -m support.train_wav2vec2 --dataset fsc --epochs 15 --noise white --split folds` | Fine-tune wav2vec 2.0 | Week 4 |
| `python -m support.evaluate --dataset fsc --run w2v2_15ep_noise` | Accuracy, F1, confusion matrix for one run | Week 4 |
| `python -m support.whisper_asr --dataset l2arctic --model small --split folds` | Zero-shot Whisper: transcripts, WER and the matched phrase | Week 4 |
| `python main.py --audio clip.wav` | Predict the phrase in an audio file (also `--video`, `--mic`) | Week 5 |
| `python app.py` | Live microphone demo (Gradio) | Week 5 |

---

## 🔮 Future Work
* **Fine-tuned speech-to-text:** fine-tune Whisper (or wav2vec 2.0 with CTC) on non-native speech to reduce its word errors, and move fully from 50 known phrases to free speech-to-text.
* **Stronger models:** compare with WavLM and XLS-R, which are pre-trained on more varied speech.
* **Bangladeshi speakers:** test the trained models on Bangladeshi speakers reading the same commands, the accent closest to our own users.
* **More speakers:** use the full Fluent Speech Commands corpus (97 speakers) and measure how results scale with the number of training speakers.

---

## 🔒 Ethics & Licences
* **No new recordings:** we record no one and store no personal data. Both datasets were collected and released by their authors; speakers appear only under the datasets' own anonymous IDs.
* **Licences respected:** Fluent Speech Commands (CC BY-NC-ND 4.0, academic research only) and L2-ARCTIC (CC BY-NC 4.0) are used non-commercially and cited. Their audio stays in a private team folder and is never committed or re-published.

---

## 🔄 Team Workflow
Every member works on their **own branch** (`dev_rabbi`, `dev_mahir`, `dev_shafil`) and owns a fixed set of files, so nobody waits for anyone else. Members can work by hand or with any AI assistant:
1. **Sync:** switch to your branch and merge in the latest `main`.
2. **Do a task:** each task has a card in [docs/tasks/](docs/tasks/) with the files to create, the functions to use, the tests and the *done when* condition.
3. **Wrap up:** run the tests, update your progress file and faculty update in [docs/progress/](docs/progress/), commit, push your branch and open a pull request.
4. The team lead reviews and merges each branch into `main` and updates the task board in [docs/STATUS.md](docs/STATUS.md).

Step-by-step commands and a ready-made prompt for chat assistants: [CONTRIBUTING.md](CONTRIBUTING.md). Rules for coding assistants: [AGENTS.md](AGENTS.md). File ownership: [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

---

## 👥 Contributors
* **Tanvir Ahmed** - *team lead (Member 1): data pipeline and splits, shared training and evaluation code, Whisper pipeline, noise tests, inference, integration; Random Forest, wav2vec 2.0 fine-tuning (15 epochs)*
* **Md. Shahriar Rakib Rabbi** - *Member 2: datasets (access, subset selection, quality control, datasheet, subset release); XGBoost, wav2vec 2.0 fine-tuning (8 epochs)*
* **Mahir Ahmed** - *Member 3: evaluation (analysis plan, statistics, results tables, error analysis); k-NN, wav2vec 2.0 fine-tuning (3 epochs)*
* **Shafil Ahmed** - *Member 4: visualisation, demo app, poster, slides and video; SVM, untrained baseline*

---

## 📄 License
* **Code:** [MIT](LICENSE).
* **Subset lists and selection script:** MIT, like the code.
* **Audio:** not distributed here. Fluent Speech Commands: CC BY-NC-ND 4.0 ([Lugosch et al., 2019](https://arxiv.org/abs/1904.03670)); L2-ARCTIC: CC BY-NC 4.0 ([Zhao et al., 2018](https://doi.org/10.21437/Interspeech.2018-1110)).

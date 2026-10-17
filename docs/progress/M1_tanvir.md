# M1 · Tanvir Ahmed (lead): Progress

| | |
|---|---|
| **Role** | Data pipeline and splits, shared training / evaluation code, Whisper pipeline, noise tests, inference, integration; reviews and merges every member's branch |
| **Models** | Random Forest (traditional) · wav2vec 2.0 fine-tuned for 15 epochs with and without white noise · Whisper zero-shot |
| **Branch** | `main` |
| **Owns** | `main.py`, `support/config.py`, `support/audio.py`, `support/dataset.py`, `support/features.py`, `support/classical.py`, `support/evaluate.py`, `support/train_wav2vec2.py`, `support/noise.py`, `support/noise_test.py`, `support/device.py`, `support/inference.py`, `support/whisper_asr.py`, `tools/make_splits.py`, `configs/`, `notebooks/`, `requirements.txt`, `.github/`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, `docs/STATUS.md`, `docs/PROJECT_PLAN.md`, `docs/tasks/`, `results/random_forest/`, `results/w2v2_15ep_noise/`, `results/w2v2_15ep/`, `results/whisper_small/`, `results/noise/` |
| **Report** | Abstract, sections 1, 2 (with Mahir's and Shafil's literature notes), 4 and 8; final integration |

The task board and handoff log for M1 are in [STATUS.md](../STATUS.md). This file holds the faculty update.

## My tasks

| ID | Task | Depends on | Status |
|---|---|---|---|
| T00, T01 | Plan, repository, environment, config, audio pre-processing | — | ✅ |
| T29 | Whisper pipeline code | — | ✅ 2026-10-17 |
| T41 | Fluent Speech Commands downloaded and checked against the paper | — | ✅ 2026-10-17 |
| T42 | Selection script; TenVoices-FSC selected (1,000 clips) | — | ✅ 2026-10-17 |
| T43 | QC script; TenVoices-FSC 997 / 1,000 pass | — | ✅ 2026-10-17 |
| T08 | Processed copies, splits, leakage tests (TenVoices-FSC) | — | ✅ 2026-10-17 |
| T11 | Features, shared runner, evaluation ✅; Random Forest run | — | 🟡 |
| T30 | Whisper on the test clips (FSC now, L2 in T47) | — | 🟡 |
| T15 | Training script + Colab ✅; wav2vec 2.0 15 epochs + noise run | — | 🟡 |
| T19 | wav2vec 2.0 15 epochs without noise | — | ⬜ |
| T21 | Noise tests, H3 | T15, T19 | 🟡 code ✅ |
| T22 | Inference + CI ✅; benchmark | T15 | 🟡 |
| T47 | TenVoices-L2: 👤 request access (form), select, QC, splits, run every model on L2, H5 | — | ⬜ |
| T24 | Final aggregation on all merged results, reproducibility, feature freeze | all | ⬜ |
| T25 | Final report and README | T24 | ⬜ |
| — | Review and merge every member's branch | — | ongoing |

## Faculty update

Plain-language summary for the weekly project update, newest week first. Only finished work; numbers only from `results/`.

### Week of 2026-10-17
- Prepared the main dataset by code: selected **TenVoices-FSC** (10 speakers, 5 female and 5 male, × 50 commands × 2 takes = 1,000 clips) with a deterministic selection script, checked every clip (997 of 1,000 pass; 3 flagged for slight clipping), and made the random split and the 5 speaker-independent folds with automatic leakage checks.
- Re-organised the team plan so that **no member waits for another**: every member can run their models today, and each member now also owns an extra model (WavLM, HuBERT, a linear probe on frozen wav2vec 2.0) to compare against our main approach. All analysis decisions were fixed in the plan before any model was trained.
- Downloaded **Fluent Speech Commands** and checked it against its paper: 30,043 recordings from 97 speakers, with the train / validation / test lists matching the paper exactly (23,132 / 3,118 / 3,793 clips). Its source and checksum are recorded in `data/SOURCES.md`.
- Built the complete training and evaluation pipeline, so every model can run the moment the two subsets are selected: processed audio and speaker-independent splits with automatic leakage checks (`support/dataset.py`, `tools/make_splits.py`); MFCC features and one shared runner for Random Forest, XGBoost, k-NN and SVM (`support/features.py`, `support/classical.py`); wav2vec 2.0 fine-tuning with white-noise augmentation and a Colab notebook (`support/train_wav2vec2.py`); Whisper on the same test clips with error rates per speaker and first language; one shared evaluation for every model (`support/evaluate.py`); noise tests at 20 / 10 / 5 dB (`support/noise_test.py`); and an inference tool for audio, video or the microphone (`main.py`). All of it is covered by 125 unit tests on synthetic audio, and a GitHub Actions workflow is set up to run them on every push.
- Checked that real wav2vec 2.0 trains on real Fluent Speech Commands audio on a laptop GPU, and found and fixed a macOS crash that happens when PyTorch and XGBoost share one process.
- Changed how we get our data: instead of recording volunteers, we build two 10-speaker datasets from published corpora with a fixed selection script. **TenVoices-FSC** (from Fluent Speech Commands) has 50 spoken smart-home commands such as "turn on the lights" or "turn up the heat in the kitchen", two takes each. **TenVoices-L2** (from L2-ARCTIC) has 50 sentences read by non-native English speakers, including all four Hindi speakers, the accent in that corpus closest to ours. I checked both papers and licences and wrote the dataset guide (`docs/DATASETS.md`) and the new plan (v3.0): every model is now tested on both datasets. The instructor confirmed that published datasets meet the brief.
- Added a fourth pipeline to the project: **Whisper**, a general-purpose speech-to-text model, used without training. Its transcript is matched to the nearest of our 50 phrases, so it can be compared with the trained models, and it adds a new research question (RQ5) and hypothesis (H5), now about non-native speech.
- Built the Whisper pipeline (`support/whisper_asr.py`). It transcribes a recording, normalises the text (numbers, times, phone numbers), finds the nearest of our 50 phrases and measures word and character error rate. 18 unit tests. A first end-to-end check on synthetic text-to-speech clips worked; it matched every clip to the right phrase but misspelled local place names, the kind of gap H5 now tests on non-native speech.
- Brought two new members into the team (Mahir, Shafil) and split the project into independent tasks with clear file ownership. Every member now works on their own branch, by hand or with any AI assistant, using task cards that say exactly what to build; I review and merge their work.

### Week of 2026-10-01
- Wrote the project plan, the data-collection protocol, the consent form, the 50-phrase list and the task board, and published the repository.
- Built the shared audio pre-processing (`support/audio.py`). Any recording becomes mono 16 kHz audio with the silence trimmed, normalised to −1 dBFS and padded to 6 s, and speech is never cut. 23 unit tests.
- Built the shared evaluation metrics (`support/metrics.py`): accuracy, macro and weighted F1, confusion matrices and per-speaker accuracy, with 14 tests checked against a hand-worked example.
- Pinned the software environment (Python 3.11) so every member runs the same versions.

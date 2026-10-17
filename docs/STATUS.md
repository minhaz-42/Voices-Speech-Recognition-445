# 📋 TenVoices: Status Board

> **Source of truth for what is merged into `main`.** Only M1 edits this file; members record their progress in `docs/progress/<slot>_<name>.md` and M1 copies it here when merging (rules in [AGENTS.md](../AGENTS.md)). Every task that can start has a card in [docs/tasks/](tasks/).
> **Legend:** ⬜ to do · 🟡 in progress · ✅ done and merged · ⛔ blocked · 👤 needs people (recording, consent, video)

**Current phase:** Weeks 1–2: recorder and pilot (M2), plus start-now work for every member
**Next up:** M2 · Rabbi: T02 → T04 · M3 · Mahir: T10, T33, T37 · M4 · Shafil: T35, T36, T38, T39 · M1 · Tanvir: merging; T04 pilot once T02 lands
**Last updated:** 2026-10-17 by Tanvir Ahmed (M1)

## 👥 Team

| Slot | Member | Branch | Traditional model | Deep-learning run | Main workstream |
|---|---|---|---|---|---|
| M1 | Tanvir Ahmed (lead, merges) | `main` | Random Forest | 15 epochs + white noise, 15 epochs without noise, Whisper zero-shot | Data pipeline and splits, shared training / evaluation code, Whisper pipeline, noise tests, inference, integration |
| M2 | Md. Shahriar Rakib Rabbi | `dev_rabbi` | XGBoost | 8 epochs + white noise | Recorder, QC, dataset curation and release |
| M3 | Mahir Ahmed | `dev_mahir` | k-NN | 3 epochs | Evaluation: analysis plan, statistics, results tables, error analysis |
| M4 | Shafil Ahmed | `dev_shafil` | SVM | Untrained baseline | Visualisation, demo app, poster, slides, video |

## 💬 How to work

Use whatever you have: Claude Code, another AI assistant, or no assistant at all. The steps and the result are the same.

**Members (M2–M4)**
- **With an assistant that works in the repository** (Claude Code, Codex, Cursor, Copilot, Gemini CLI, …): say *"I'm Mahir (M3). What is completed? Then complete my next tasks."* It follows [AGENTS.md](../AGENTS.md), works on your branch and, when you say *"Wrap up and push"*, updates your progress file and faculty update, commits, pushes and opens a pull request.
- **With a chat assistant (ChatGPT, Gemini, …) or by hand:** follow [CONTRIBUTING.md](../CONTRIBUTING.md). It has the git commands, a ready-made prompt, and the list of files to give the assistant for each task card.
- Either way, tell Tanvir which tasks are ready to merge.

**Lead (M1)**
1. Your own tasks work the same way, directly on `main`.
2. When a member's branch is ready: *"Merge dev_mahir."* The merge routine is in [AGENTS.md](../AGENTS.md).

A task is ✅ here once it is merged. Finished work that is not merged yet shows ✅ in the member's progress file, and counts as done for that member's own next tasks.

## 🗂️ Task Board

### Start now: independent work (no recordings needed)

Each of these has a task card: `docs/tasks/<ID>.md`.

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T10 | `docs/ANALYSIS_PLAN.md`: hypotheses H1–H5 with pass criteria, both splits, metrics, statistical tests and leakage rules, written from `docs/PROJECT_PLAN.md` → M1 approves it at merge and it is **frozen** before any main experiment | M3 | T03 | ⬜ | |
| T33 | `support/stats.py`: 95% bootstrap confidence interval for accuracy, McNemar's test for two models on the same clips, Wilcoxon signed-rank test over per-speaker accuracies + tests against textbook or hand-computed cases → tests pass | M3 | T03 | ⬜ | |
| T37 | Literature notes `docs/literature/classical_and_evaluation.md`: 6–8 papers on MFCC features, k-NN / SVM / tree models for speech, and speaker-independent evaluation; each with a checked DOI or arXiv link, what they did and what we take from it → feeds report section 2 | M3 | — | ⬜ | |
| T35 | `support/visualization.py`: waveform, spectrogram, log-mel and MFCC plots of one clip, a before / after pre-processing figure, and chart helpers (model-comparison bars, accuracy vs SNR, per-speaker bars) + tests on synthetic data; example figures of your own voice in `images/` → tests pass | M4 | T01 | ⬜ | |
| T36 | Demo skeleton `app.py` (Gradio): microphone or upload → shared pre-processing → waveform and spectrogram from T35 + the 50-phrase list, with a clearly marked "model not connected yet" area (no fake predictions) → the app runs locally | M4 | T35 | ⬜ | |
| T38 | Literature notes `docs/literature/self_supervised_speech.md`: 6–8 papers on self-supervised speech models (wav2vec 2.0, HuBERT, WavLM, XLS-R) and spoken-command benchmarks; same format as T37 → feeds report section 2 | M4 | — | ⬜ | |
| T39 | Planning poster v2 (`poster/poster.html` → `images/poster.png`): team of four and the four-stage pipeline from `docs/PROJECT_PLAN.md`, no results → the poster renders and the README shows it | M4 | — | ⬜ | |
| T29 | Whisper pipeline `support/whisper_asr.py`: zero-shot transcription (`--model small`), Whisper's English text normaliser, nearest-phrase matching (word-level edit distance), WER / CER, predictions in the shared format + tests; smoke test on a few self-recorded phrases → tests pass | M1 | T01, T03 | ✅ | Tanvir · 2026-10-17 |

### Week 1: Setup, phrase list and recording tool

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T00 | Project plan, README, poster, protocol, consent form, phrase list, `CLAUDE.md`, this board → committed | M1 | — | ✅ | Tanvir · 2026-10-01 |
| T01 | Publish the GitHub repo and add collaborators; pin `requirements.txt`; `configs/config.yaml` + `support/config.py`; `support/audio.py` (load, mono, 16 kHz, trim silence, peak-normalise, pad to 6 s) + tests → tests pass and a sample WAV loads | M1 | T00 | ✅ | Tanvir · 2026-10-04 |
| T02 | Recorder tool `tools/recorder.py` (Gradio): 3 shuffled rounds of `data/prompts/phrases.csv` + the noisy block; file naming; metadata written **one file per speaker** (`data/metadata/utterances/<SPK>.csv`, `data/metadata/speakers/<SPK>.csv`) so members recording on different branches never conflict; printable phrase sheets → a test session saves correct files | M2 | T01 | 🟡 | |
| T03 | `support/metrics.py`: accuracy, macro / weighted F1, classification report, confusion-matrix plot, per-speaker accuracy + tests → hand-computed cases pass | M3 | T01 | ✅ | Tanvir · 2026-10-04 |
| T04 | 👤 Pilot sessions S01 (Tanvir) and S02 (Rabbi): consent signed, files backed up to Drive → 2 × 160 clips | M1 + M2 | T02 | ⬜ | |
| T05 | `tools/qc_report.py` (completeness, duration, clipping, silence, misread flags) run on the pilot; fix the issues found → protocol v1 frozen | M2 | T04 | ⬜ | |

### Week 2: Dataset collection

Each recorder recruits their own volunteers, collects consent, records the noisy takes and backs the files up to Drive the same day. Recruit so the 10 speakers reach the 5F / 5M target (self-reported), and name one backup volunteer each.

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T06 | 👤 Record volunteers **S05–S06** → 2 × 160 clips | M2 | T05 | ⬜ | |
| T31 | 👤 Record **S03 (self)** and volunteers **S07–S08** → 3 × 160 clips | M3 | T05 | ⬜ | |
| T32 | 👤 Record **S04 (self)** and volunteers **S09–S10** → 3 × 160 clips | M4 | T05 | ⬜ | |
| T07 | QC all sessions with `tools/qc_report.py`; listen-checks: Rabbi → S01, S03, S07–S08 · Mahir → S04, S09–S10 · Shafil → S02, S05–S06; re-record misreads → ≥ 97% of clips pass | M2 (+ M3, M4 listen-checks) | T04, T06, T31, T32 | ⬜ | |
| T08 | Processed 16 kHz copies + manifest (label, speaker, rep, condition, SHA-256); `support/dataset.py`; `tools/make_splits.py` → random 80/20 split + 5 speaker-independent folds in `data/splits/`; leakage tests → tests pass; tag `data-v1` | M1 | T07 | ⬜ | |
| T09 | Dataset statistics figure + datasheet `data/README.md` | M2 | T08 | ⬜ | |

### Week 3: Traditional machine learning

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T11 | `support/features.py` (MFCC 13 + Δ + ΔΔ, mean & std) + shared runner `support/classical.py` (scaler + model, both splits, validation tuning, saves report / predictions / confusion matrix) + shared `support/evaluate.py`; **Random Forest** results → `results/random_forest/` | M1 | T03, T08, T10 | ⬜ | |
| T12 | **XGBoost** results with the shared runner → `results/xgboost/` | M2 | T11 | ⬜ | |
| T13 | **k-NN** results with the shared runner → `results/knn/` | M3 | T11 | ⬜ | |
| T14 | **SVM** results with the shared runner → `results/svm/` | M4 | T11 | ⬜ | |
| T20 | `support/results_table.py`: every run in `results/` → `results/all_results.csv` + `images/all_results_table.png`, seen vs unseen speakers, per-speaker table, bootstrap CIs from T33; works with whichever runs exist → table built from the merged runs | M3 | T11, T33 | ⬜ | |

### Week 4: wav2vec 2.0 and Whisper

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T15 | Shared training script `support/train_wav2vec2.py` (`--epochs`, `--noise`, `--split`; validation-based model selection; fp16 on GPU) + Colab notebook; run **15 epochs + white noise** on the random split and all 5 folds → `results/w2v2_15ep_noise/` (models to Drive) | M1 | T08, T11 | ⬜ | |
| T16 | Run **8 epochs + white noise** with the shared script → `results/w2v2_8ep_noise/` | M2 | T15 | ⬜ | |
| T17 | Run **3 epochs** (no noise) with the shared script → `results/w2v2_3ep/` | M3 | T15 | ⬜ | |
| T18 | **Untrained baseline** with the shared evaluation script → `results/w2v2_untrained/` | M4 | T15 | ⬜ | |
| T19 | Run **15 epochs without noise** (ablation for H3) → `results/w2v2_15ep/` | M1 | T15 | ⬜ | |
| T30 | Run Whisper on the test clips of the random split and all 5 folds → `results/whisper_small/`; WER / CER per category and speaker; test H5; `--asr whisper` transcript mode in `main.py` | M1 | T08, T11, T22, T29 | ⬜ | |

### Week 5: Results, analysis and demo

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T40 | Final results once every run is merged: regenerate T20's tables and test H1, H2, H4 with `support/stats.py` → `results/hypotheses.csv` | M3 | T12, T13, T14, T15, T16, T17, T18, T19, T20 | ⬜ | |
| T21 | Noise tests: white noise at 20 / 10 / 5 dB on the clean test clips + the real noisy takes, for the 15-epoch runs with and without noise and every other merged run; test H3 → `results/noise/` | M1 | T15, T19 | ⬜ | |
| T34 | `support/error_analysis.py`: most-confused phrase pairs, accuracy per category and per speaker for any run; write-up for the best traditional and the best wav2vec 2.0 model → `results/errors/` | M3 | T40 | ⬜ | |
| T22 | `main.py` inference (WAV / MP3 / MP4 audio / microphone → top-5 phrases with confidence) + inference-time benchmark + unit tests + GitHub Actions CI | M1 | T15 | ⬜ | |
| T23 | Connect the demo to `main.py` (top-5 phrases with a confidence chart, Whisper transcript) + final figures with `support/visualization.py` (confusion matrices, accuracy vs SNR, per-speaker chart) → `images/` | M4 | T21, T22, T30, T36, T40 | ⬜ | |
| T24 | Clean-environment reproducibility check → **feature freeze** | M1 | T21, T22, T23, T30, T34, T40 | ⬜ | |

### Week 6: Delivery

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T25 | Final report (paper format, ~8 pages) integration + final README with real results | M1 | T24, T37, T38 | ⬜ | |
| T26 | Dataset release (consenting speakers; FLAC, metadata, datasheet, checksums) as a GitHub Release asset | M2 | T24 | ⬜ | |
| T27 | Results poster, slides, 👤 1-minute demo video | M4 | T24 | ⬜ | |
| T28 | 👤 Final checklist (every number matches `results/`), tag `v1.0`, submit | all | T25, T26, T27 | ⬜ | |

## 📝 Handoff Log (newest first)

### 2026-10-17 · Tanvir Ahmed (M1) · T29

- **Done:** `support/whisper_asr.py`. Whisper's English normaliser on both sides, with neighbouring digit groups joined, leading zeros dropped and `6.30` read as a time, so spoken and written numbers, times and phone numbers compare equal. Also WER / CER per clip and per corpus, nearest-phrase matching (WER, ties by CER) with a demo-only rejection threshold, a batched `WhisperTranscriber` (CUDA > MPS > CPU), and `run_clip_list` writing `predictions.csv` (`utt_id, speaker_id, label, pred, transcript, wer, cer, match_wer, rejected`) + `metrics.json`. New `whisper` section in `configs/config.yaml`. 18 offline tests (no model download); all 61 tests pass.
- **Smoke test (not a result):** 16 clips of 8 phrases spoken by two macOS text-to-speech voices (Indian and British English), kept outside the repo. `openai/whisper-small` ran on MPS and matched every clip to the right phrase, while misspelling local names (Mohakhali, Sylhet, Sreemangal). That is what H5 tests on real recordings.
- **Files:** `support/whisper_asr.py`, `tests/test_whisper_asr.py`, `configs/config.yaml`, `AGENTS.md`, `README.md`, `docs/STATUS.md`, `docs/progress/M1_tanvir.md`.
- **Notes:** T11's `support/evaluate.py` should read the same `utt_id, speaker_id, label, pred` columns. T30 adds `--split`. The first run downloads the model (~1 GB) into the Hugging Face cache.
- **Next:** M1 has no other unblocked task until T02 → T04; merge member branches as they come in.

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (any assistant or by hand)

- **Done:** members no longer need Claude. The rules moved to **`AGENTS.md`**, which many coding assistants read automatically; `CLAUDE.md` now only imports it. **`CONTRIBUTING.md`** gives the whole workflow as plain git commands, with a ready-made prompt for chat assistants. **Task cards** in `docs/tasks/` (T02, T10, T29, T33, T35–T39) say exactly which files to create, which existing functions to use, what to test and when a task is done, and list the files to paste into a chat assistant.
- **Files:** `AGENTS.md` (new), `CLAUDE.md`, `CONTRIBUTING.md` (new), `docs/tasks/` (new), `docs/STATUS.md`, `docs/PROJECT_PLAN.md`, `README.md`, `docs/progress/`.
- **Next:** members start their start-now tasks from the cards; M1 does T29.
- **Blockers:** none.

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (team of four, branch per member)

- **Done:** Mahir Ahmed takes **M3** (evaluation) and Shafil Ahmed takes **M4** (visualisation, demo, presentation). Every member now works on their own branch (`dev_rabbi`, `dev_mahir`, `dev_shafil`), edits only the files they own, and records progress and a plain-language **faculty update** in `docs/progress/<slot>_<name>.md`; M1 reviews, merges and keeps this board.
- **Re-balanced tasks:** M1 keeps the complex shared work (splits, shared runners, training, Whisper, noise tests T21, inference). M3 gets statistics (T33), results tables (T20 → T40), error analysis (T34) and literature notes (T37). M4 gets the visualisation module (T35), demo skeleton (T36), literature notes (T38) and poster v2 (T39). Recording is split per member: M2 S05–S06 (T06), M3 S03 + S07–S08 (T31), M4 S04 + S09–S10 (T32); M1 records only S01.
- **Start now (no recordings needed):** M3: T10, T33, T37 · M4: T35, T36, T38, T39 · M1: T29.
- **T02 change for M2:** the recorder writes metadata one file per speaker, so recordings made on different branches merge cleanly.
- **Files:** `CLAUDE.md`, `README.md`, `docs/STATUS.md`, `docs/PROJECT_PLAN.md`, `docs/DATA_COLLECTION_PROTOCOL.md`, `docs/templates/individual_update_report.md`, `docs/progress/` (new).
- **Blockers:** Mahir needs a GitHub collaborator invite (username needed). Mahir's and Shafil's git user.name are still *(add)* in `CLAUDE.md`.

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (Whisper pipeline)

- **Done:** added a fourth stage, **Whisper used zero-shot** (speech-to-text, then the nearest of the 50 phrases), to the plan, README and team guide: research question RQ5, hypothesis **H5** (Whisper errs most on Bangladeshi names and places, where fine-tuned wav2vec 2.0 beats it), experiment E8, WER / CER metrics, methodology section 4, results-table row. New tasks **T29** (pipeline code, can start now) and **T30** (runs on the dataset, test H5, transcript mode in `main.py`), both M1.
- **Files:** `docs/PROJECT_PLAN.md`, `README.md`, `CLAUDE.md`, `docs/STATUS.md`.
- **Next:** H5 goes into the analysis plan (T10) before it is frozen.
- **Blockers:** none.

### 2026-10-07 · Md. Shahriar Rakib Rabbi (M2) · T02 in progress

- **Done:** Added `support/phrases.py` to load and validate the 50-phrase catalog, including category counts, unique IDs and noisy-take flags. Added six unit tests in `tests/test_phrases.py`.
- **Validation:** Focused phrase tests, Python compilation and editor diagnostics pass. The full suite is blocked in the current Python 3.14 environment because `numpy` is not installed; the project targets Python 3.11.
- **Next:** Build the recording-plan generator and Gradio recorder using `load_phrases()`.
- **Blockers:** Python 3.11/dependencies are not installed in the current environment; no project code blocker.

### 2026-10-04 · Tanvir Ahmed (M3) · T03

- **Done:** `support/metrics.py`: `compute_metrics` (accuracy, macro F1, weighted F1), `classification_report_table` (per-class precision / recall / F1 / support + averages), `confusion_matrix_table`, `plot_confusion_matrix` (row-normalised or counts, readable at 50 classes, saves PNG/PDF/SVG), `accuracy_by_group` and `per_speaker_accuracy` (also works per category). 14 tests against a hand-computed 6-clip example (accuracy 4/6, macro F1 59/90, weighted F1 61/90).
- **Files:** `support/metrics.py`, `tests/test_metrics.py`, `docs/STATUS.md`.
- **Notes:** F1 is averaged over the classes that occur in the true or predicted labels. Pass the full phrase list as `labels` to reports and confusion matrices, so every model's tables share the same row order. `support/evaluate.py` (T11) should call these functions rather than sklearn directly.
- **Next:** T10 (analysis plan, M3) waits for T08. M1 / M3 / M4 have nothing unblocked until M2 finishes T02.
- **Blockers:** none for this task.

### 2026-10-04 · Tanvir Ahmed (M1) · T01

- **Done:** `configs/config.yaml` (seed, paths, audio settings); `support/config.py` (`load_config`, `resolve_path`); `support/audio.py` (read, int16 → float, mono, resample to 16 kHz, trim silence with a 0.1 s margin, peak-normalise to −1 dBFS, zero-pad to 6 s; a clip still longer than 6 s after trimming raises an error instead of being cut; `save_wav`; CLI). 23 unit tests. A spoken sample (44.1 kHz) loads as 6.00 s of 16 kHz mono at −1.0 dBFS.
- **Files:** `configs/config.yaml`, `support/__init__.py`, `support/config.py`, `support/audio.py`, `tools/__init__.py`, `tests/test_config.py`, `tests/test_audio.py`, `requirements.txt`, `README.md`, `CLAUDE.md`.
- **Notes for everyone:**
    - `requirements.txt` is now pinned (Python 3.11). torchaudio was removed: nothing uses it, and pip paired it with a mismatched torch. On Colab, keep the preinstalled torch (command in `requirements.txt`).
    - Run scripts as modules from the repo root: `python -m support.audio clip.wav`, `python -m tools.recorder ...`. `tools/` is a package, so M2's recorder can `from support.audio import ...`.
    - Use `support.audio.preprocess` for microphone arrays (accepts int16) and `load_audio(path, pad=False)` when padding would distort the result (MFCC statistics).
- **Next:** M2 starts T02 (recorder). M3 can start T03 (metrics).
- **Blockers:** Rabbi's collaborator invite needs Rabbi's GitHub username (Settings → Collaborators on the repo page).

### 2026-10-01 · Tanvir Ahmed (M1) · T00 (revision)

- **Done:** switched the method to spoken phrase recognition with 50 phrases, following the pipeline traditional ML → untrained baseline → fine-tuned wav2vec 2.0 (3 / 8 / 15 epochs, white noise). Rewrote the plan, README, protocol, phrase list (`data/prompts/phrases.csv`), poster and this board (new tasks T01–T28).
- **Next:** finish T01 (M1). Once T01 is pushed, M2 can start T02 and M3 can start T03.
- **Blockers:** Rabbi's GitHub username is needed to add him as a collaborator.

### 2026-10-01 · Tanvir Ahmed (M1) · T01 (part 1)

- **Done:** published the repo at https://github.com/minhaz-42/Voices-Speech-Recognition-445 and pushed all planning files.
- **Next:** finish T01 (M1).

### 2026-10-01 · Tanvir Ahmed (M1) · T00

- **Done:** project plan, README, poster, data-collection protocol, consent form, `CLAUDE.md` and this status board.

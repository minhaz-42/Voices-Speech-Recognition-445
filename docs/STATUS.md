# 📋 TenVoices: Status Board

> **Source of truth for progress.** Claude reads this file at the start of every session and updates it at the end (rules in [CLAUDE.md](../CLAUDE.md)).
> **Legend:** ⬜ to do · 🟡 in progress · ✅ done · ⛔ blocked · 👤 needs people (recording, consent, video)

**Current phase:** Week 1: setup, phrase list and recording tool
**Next up:** T02 recorder (M2 · Rabbi) and T03 metrics (M3 · Tanvir); invite Rabbi as collaborator
**Last updated:** 2026-10-04 by Tanvir Ahmed (M1)

## 👥 Team

| Slot | Member | Traditional model | Deep-learning run | Main workstream |
|---|---|---|---|---|
| M1 | Tanvir Ahmed (lead) | Random Forest | 15 epochs + white noise | Data pipeline, shared training/evaluation code, inference, integration |
| M2 | Md. Shahriar Rakib Rabbi | XGBoost | 8 epochs + white noise | Dataset collection and curation, dataset release |
| M3 | Tanvir Ahmed *(open slot)* | k-NN | 3 epochs | Evaluation: results tables, statistics, noise tests, error analysis |
| M4 | Tanvir Ahmed *(open slot)* | SVM | Untrained baseline | Demo app, figures, poster, slides, video |

## 💬 How to work with Claude

1. Pull, open Claude Code in the repo and say who you are: *"I'm M2. What is completed?"*
2. Then say: *"Complete my next tasks."* Claude checks the dependencies below and only starts tasks that are unblocked.
3. When you're done for the session, say: *"Wrap up and push."* Claude updates this board and the handoff log, commits and pushes.
4. Tell the next member to pull.

## 🗂️ Task Board

### Week 1: Setup, phrase list and recording tool

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T00 | Project plan, README, poster, protocol, consent form, phrase list, `CLAUDE.md`, this board → committed | M1 | — | ✅ | Tanvir · 2026-10-01 |
| T01 | Publish the GitHub repo ✅ and add Rabbi as collaborator (⛔ invite pending: needs Rabbi's GitHub username); pin `requirements.txt`; `configs/config.yaml` + `support/config.py`; `support/audio.py` (load, mono, 16 kHz, trim silence, peak-normalise, pad to 6 s) + tests → tests pass and a sample WAV loads | M1 | T00 | ✅ | Tanvir · 2026-10-04 |
| T02 | Recorder tool `tools/recorder.py` (Gradio): 3 shuffled rounds of `data/prompts/phrases.csv` + the noisy block; file naming; one `utterances.csv` row per clip; printable phrase sheets → a test session saves correct files | M2 | T01 | ⬜ | |
| T03 | `support/metrics.py`: accuracy, macro / weighted F1, classification report, confusion-matrix plot, per-speaker accuracy + tests → hand-computed cases pass | M3 | T01 | ⬜ | |
| T04 | 👤 Pilot sessions S01 (Tanvir) and S02 (Rabbi): consent signed, files backed up to Drive → 2 × 160 clips | M1 + M2 | T02 | ⬜ | |
| T05 | `tools/qc_report.py` (completeness, duration, clipping, silence, misread flags) run on the pilot; fix the issues found → protocol v1 frozen | M2 | T04 | ⬜ | |

### Week 2: Dataset collection

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T06 | 👤 Record volunteers S03–S06 (recruited by M1) and S07–S10 (recruited by M2), including noisy takes; target 5F/5M overall → 10/10 speakers | M1 + M2 | T05 | ⬜ | |
| T07 | QC all sessions; listen-check (M1 checks M2's speakers, M2 checks M1's); re-record misreads → ≥ 97% of clips pass | M2 (+ M1) | T06 | ⬜ | |
| T08 | Processed 16 kHz copies + manifest (label, speaker, rep, condition, SHA-256); `support/dataset.py`; `tools/make_splits.py` → random 80/20 split + 5 speaker-independent folds in `data/splits/`; leakage tests → tests pass; tag `data-v1` | M1 | T07 | ⬜ | |
| T09 | Dataset statistics figure + datasheet `data/README.md` | M2 | T08 | ⬜ | |
| T10 | `docs/ANALYSIS_PLAN.md`: hypotheses H1–H4, splits, metrics, tests → **frozen** before any main experiment | M3 | T03, T08 | ⬜ | |

### Week 3: Traditional machine learning

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T11 | `support/features.py` (MFCC 13 + Δ + ΔΔ, mean & std) + shared runner `support/classical.py` (scaler + model, both splits, validation tuning, saves report / predictions / confusion matrix) + shared `support/evaluate.py`; **Random Forest** results → `results/random_forest/` | M1 | T03, T08 | ⬜ | |
| T12 | **XGBoost** results with the shared runner → `results/xgboost/` | M2 | T11 | ⬜ | |
| T13 | **k-NN** results with the shared runner → `results/knn/` | M3 | T11 | ⬜ | |
| T14 | **SVM** results with the shared runner → `results/svm/` | M4 | T11 | ⬜ | |

### Week 4: wav2vec 2.0

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T15 | Shared training script `support/train_wav2vec2.py` (`--epochs`, `--noise`, `--split`; validation-based model selection; fp16 on GPU) + Colab notebook; run **15 epochs + white noise** on the random split and all 5 folds → `results/w2v2_15ep_noise/` (models to Drive) | M1 | T08, T11 | ⬜ | |
| T16 | Run **8 epochs + white noise** with the shared script → `results/w2v2_8ep_noise/` | M2 | T15 | ⬜ | |
| T17 | Run **3 epochs** (no noise) with the shared script → `results/w2v2_3ep/` | M3 | T15 | ⬜ | |
| T18 | **Untrained baseline** with the shared evaluation script → `results/w2v2_untrained/` | M4 | T15 | ⬜ | |
| T19 | Run **15 epochs without noise** (ablation for H3) → `results/w2v2_15ep/` | M1 | T15 | ⬜ | |

### Week 5: Results, analysis and demo

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T20 | `support/results_table.py` + `support/stats.py`: `results/all_results.csv`, `images/all_results_table.png`, seen vs unseen speakers, per-speaker table, McNemar + bootstrap CIs; test H1, H2, H4 | M3 | T12, T13, T14, T16, T17, T18 | ⬜ | |
| T21 | Noise tests (white noise at 20 / 10 / 5 dB + real noisy takes) for the best models; test H3; error analysis of the most-confused phrases → `results/noise/`, `results/errors/` | M3 | T19, T20 | ⬜ | |
| T22 | `main.py` inference (WAV / MP3 / MP4 audio / microphone → top-5 phrases with confidence) + inference-time benchmark + unit tests + GitHub Actions CI | M1 | T15 | ⬜ | |
| T23 | Gradio demo `app.py` + final figures (confusion matrices, accuracy vs SNR, per-speaker chart) → `images/` | M4 | T20, T21, T22 | ⬜ | |
| T24 | Clean-environment reproducibility check → **feature freeze** | M1 | T20, T21, T22, T23 | ⬜ | |

### Week 6: Delivery

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T25 | Final report (paper format, ~8 pages) integration + final README with real results | M1 | T24 | ⬜ | |
| T26 | Dataset release (consenting speakers; FLAC, metadata, datasheet, checksums) as a GitHub Release asset | M2 | T24 | ⬜ | |
| T27 | Results poster, slides, 👤 1-minute demo video | M4 | T24 | ⬜ | |
| T28 | 👤 Final checklist (every number matches `results/`), tag `v1.0`, submit | all | T25, T26, T27 | ⬜ | |

## 📝 Handoff Log (newest first)

### 2026-10-04 · Tanvir Ahmed (M1) · T01
- **Done:** `configs/config.yaml` (seed, paths, audio settings); `support/config.py` (`load_config`, `resolve_path`); `support/audio.py` (read, int16 → float, mono, resample to 16 kHz, trim silence with a 0.1 s margin, peak-normalise to −1 dBFS, zero-pad to 6 s; a clip still longer than 6 s after trimming raises an error instead of being cut; `save_wav`; CLI). 23 unit tests. A spoken sample (44.1 kHz) loads as 6.00 s of 16 kHz mono at −1.0 dBFS.
- **Files:** `configs/config.yaml`, `support/__init__.py`, `support/config.py`, `support/audio.py`, `tools/__init__.py`, `tests/test_config.py`, `tests/test_audio.py`, `requirements.txt`, `README.md`, `CLAUDE.md`.
- **Notes for everyone:**
  - `requirements.txt` is now pinned (Python 3.11). torchaudio was removed: nothing uses it, and pip paired it with a mismatched torch. On Colab, keep the preinstalled torch (command in `requirements.txt`).
  - Run scripts as modules from the repo root: `python -m support.audio clip.wav`, `python -m tools.recorder ...`. `tools/` is a package, so M2's recorder can `from support.audio import ...`.
  - Use `support.audio.preprocess` for microphone arrays (accepts int16) and `load_audio(path, pad=False)` when padding would distort the result (MFCC statistics).
- **Next:** M2 starts T02 (recorder). M3 can start T03 (metrics).
- **Blockers:** Rabbi's collaborator invite needs his GitHub username (Settings → Collaborators on the repo page).

### 2026-10-01 · Tanvir Ahmed (M1) · T00 (revision)
- **Done:** switched the method to spoken phrase recognition with 50 phrases, following the pipeline traditional ML → untrained baseline → fine-tuned wav2vec 2.0 (3 / 8 / 15 epochs, white noise). Rewrote the plan, README, protocol, phrase list (`data/prompts/phrases.csv`), poster and this board (new tasks T01–T28).
- **Next:** finish T01 (M1). Once T01 is pushed, M2 can start T02 and M3 can start T03.
- **Blockers:** Rabbi's GitHub username is needed to add him as a collaborator.

### 2026-10-01 · Tanvir Ahmed (M1) · T01 (part 1)
- **Done:** published the repo at https://github.com/minhaz-42/Voices-Speech-Recognition-445 and pushed all planning files.
- **Next:** finish T01 (M1).

### 2026-10-01 · Tanvir Ahmed (M1) · T00
- **Done:** project plan, README, poster, data-collection protocol, consent form, `CLAUDE.md` and this status board.

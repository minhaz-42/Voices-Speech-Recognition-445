# 📋 TenVoices: Status Board

> **Source of truth for what is merged into `main`.** Only M1 edits this file; members record their progress on their own page in [docs/team/](team/) and M1 copies it here when merging (rules in [AGENTS.md](../AGENTS.md)). Every task's card is at the end of its owner's page.
> **Legend:** ⬜ to do · 🟡 in progress · ✅ done and merged · ✖ cancelled · 👤 needs a person (dataset access, listening, video)

**Nobody waits for anyone.** Every member task below depends only on work that is already merged (shared code ✅, TenVoices-FSC subsets and splits ✅) or on the member's own earlier tasks. Work that needs everyone's results is split: members write and test the code; the lead runs it once on all merged results at the end (T24). TenVoices-L2 is prepared and run by the lead (T47).

**Current phase:** every member can start all of their tasks now, on TenVoices-FSC.
**Next up:** M2 · Rabbi: T12, T16, T44, T09 · M3 · Mahir: T10, T33, T37, T13, T17, T45 · M4 · Shafil: T35, T38, T39, T14, T18, T46 · M1 · Tanvir: T11 (Random Forest), T30, T15 / T19 runs, 👤 L2-ARCTIC form (T47)
**Last updated:** 2026-10-17 by Tanvir Ahmed (M1)

## 👥 Team

| Slot | Member | Branch | Traditional model | Deep-learning runs | Main workstream |
|---|---|---|---|---|---|
| M1 | [Tanvir Ahmed](team/M1_tanvir.md) (lead, merges) | `main` | Random Forest | wav2vec 2.0 15 epochs with and without noise, Whisper zero-shot | Data preparation, shared code, Whisper, noise tests, inference, TenVoices-L2, final aggregation |
| M2 | [Md. Shahriar Rakib Rabbi](team/M2_rabbi.md) | `dev_rabbi` | XGBoost | wav2vec 2.0 8 epochs + noise, **WavLM Base+** 15 epochs + noise | Datasheet, listen-check, dataset statistics, subset release |
| M3 | [Mahir Ahmed](team/M3_mahir.md) | `dev_mahir` | k-NN | wav2vec 2.0 3 epochs, **HuBERT Base** 15 epochs + noise | Analysis plan, statistics, results table, hypothesis tests, error analysis |
| M4 | [Shafil Ahmed](team/M4_shafil.md) | `dev_shafil` | SVM | Untrained baseline, **linear probe** on frozen wav2vec 2.0 | Visualisation, demo app, poster, slides, video |

## 💬 How to work

Use whatever you have: Claude Code, another AI assistant, or no assistant at all. The steps and the result are the same.

**Members (M2–M4)**
- **With an assistant that works in the repository** (Claude Code, Codex, Cursor, Copilot, Gemini CLI, …): say *"I'm Mahir (M3). What is completed? Then complete my next tasks."* It follows [AGENTS.md](../AGENTS.md), works on your branch and, when you say *"Wrap up and push"*, updates your progress file and faculty update, commits, pushes and opens a pull request.
- **With a chat assistant (ChatGPT, Gemini, …) or by hand:** follow [CONTRIBUTING.md](CONTRIBUTING.md). It has the git commands, a ready-made prompt, and the list of files to give the assistant for each task card.
- **Running a model:** [docs/RUNS.md](RUNS.md) shows how to get the audio and run any model on TenVoices-FSC, locally or on Colab.
- Either way, tell Tanvir which tasks are ready to merge.

**Lead (M1)**
1. Your own tasks work the same way, directly on `main`.
2. When a member's branch is ready: *"Merge dev_mahir."* The merge routine is in [AGENTS.md](../AGENTS.md).

A task is ✅ here once it is merged. Finished work that is not merged yet shows ✅ on the member's page, and counts as done for that member's own next tasks.

## 🗂️ Task Board

Runs are done when their results for TenVoices-FSC (random split + 5 folds) are in `results/<run>/fsc/`. The same command for TenVoices-L2 is run by M1 in T47.

### M2 · Rabbi

| ID | Task (→ done when) | Depends on | Status | Done by · date |
|---|---|---|---|---|
| T12 | **XGBoost** with the shared runner → `results/xgboost/fsc/` | — | ⬜ | |
| T16 | **wav2vec 2.0, 8 epochs + white noise** → `results/w2v2_8ep_noise/fsc/` | — | ⬜ | |
| T44 | **WavLM Base+, 15 epochs + white noise** (same script, `--model microsoft/wavlm-base-plus`) → `results/wavlm_15ep_noise/fsc/` | — | ⬜ | |
| T09 | Datasheet `data/README.md` for TenVoices-FSC (sources, selection, QC, statistics, licences) + dataset statistics figure + 👤 listen-check of the 100 marked clips → `data/listen_check_fsc.csv` | — | ⬜ | |
| T26 | Subset release: subset and phrase lists, selection and QC scripts, datasheet, checksums (no audio) as a GitHub Release draft | T09 | ⬜ | |

### M3 · Mahir

| ID | Task (→ done when) | Depends on | Status | Done by · date |
|---|---|---|---|---|
| T10 | `docs/ANALYSIS_PLAN.md`: the full analysis, consistent with the *Frozen analysis decisions* in PROJECT_PLAN.md (it no longer gates any run) | — | ⬜ | |
| T33 | `support/stats.py`: bootstrap CI, McNemar, Wilcoxon, Mann–Whitney + hand-computed tests | — | ⬜ | |
| T37 | Literature notes: MFCC, classical classifiers, speaker-independent evaluation | — | ⬜ | |
| T13 | **k-NN** with the shared runner → `results/knn/fsc/` | — | ⬜ | |
| T17 | **wav2vec 2.0, 3 epochs** (no noise) → `results/w2v2_3ep/fsc/` | — | ⬜ | |
| T45 | **HuBERT Base, 15 epochs + white noise** (same script, `--model facebook/hubert-base-ls960`) → `results/hubert_15ep_noise/fsc/` | — | ⬜ | |
| T20 | `support/results_table.py`: every run in `results/` → `results/all_results.csv` + table image; seen vs unseen speakers, per speaker, bootstrap CIs; tested on synthetic and your own runs | T33 | ⬜ | |
| T34 | `support/error_analysis.py`: most-confused phrase pairs, accuracy per phrase category and speaker, for any run; write-up for your own runs → `results/errors/` | T13 | ⬜ | |
| T40 | `support/hypotheses.py`: H1–H5 exactly as frozen, from whatever runs exist → `results/hypotheses.csv`; tested on synthetic results (M1 runs it on all results in T24) | T10, T20, T33 | ⬜ | |

### M4 · Shafil

| ID | Task (→ done when) | Depends on | Status | Done by · date |
|---|---|---|---|---|
| T35 | `support/visualization.py` + tests + example figures | — | ⬜ | |
| T36 | Demo skeleton `app.py` (Gradio) | T35 | ⬜ | |
| T38 | Literature notes: self-supervised speech models | — | ⬜ | |
| T39 | Planning poster v2 | — | ⬜ | |
| T14 | **SVM** with the shared runner → `results/svm/fsc/` | — | ⬜ | |
| T18 | **Untrained wav2vec 2.0 baseline** (`--untrained`) → `results/w2v2_untrained/fsc/` | — | ⬜ | |
| T46 | **Linear probe:** `support/embeddings.py` (frozen wav2vec 2.0, mean-pooled) + logistic regression with the shared runner → `results/w2v2_probe/fsc/` | — | ⬜ | |
| T23 | Demo connected to `support.inference` with your SVM model (top-5 + confidence chart); `python -m support.visualization --results` draws the figures of every run in `results/` | T14, T35, T36 | ⬜ | |
| T27 | Results poster, slides and 👤 1-minute demo video, with numbers and figures generated from `results/` (regenerated by one command in T28) | T23 | ⬜ | |

### M1 · Tanvir

| ID | Task (→ done when) | Depends on | Status | Done by · date |
|---|---|---|---|---|
| T11 | Features, shared runner and evaluation ✅; **Random Forest** → `results/random_forest/fsc/` ✅ (FSC: random split 0.310, unseen speakers 0.209) | — | ✅ | Tanvir · 2026-10-17 |
| T30 | **Whisper zero-shot** on the test clips → `results/whisper_small/fsc/` ✅ (FSC: random split 0.990, unseen speakers 0.987, WER 0.039); L2 and H5 in T47 | — | 🟡 FSC ✅ | Tanvir · 2026-10-17 (FSC) |
| T15 | Training script + Colab notebook ✅; **wav2vec 2.0, 15 epochs + white noise** → `results/w2v2_15ep_noise/fsc/` (models to Drive) | — | 🟡 code ✅ | |
| T19 | **wav2vec 2.0, 15 epochs without noise** (H3 ablation) → `results/w2v2_15ep/fsc/` | — | ⬜ | |
| T21 | Noise tests at 20 / 10 / 5 dB for T15 and T19 (and every merged run) → `results/noise/`; H3 | T15, T19 | 🟡 code ✅ | |
| T22 | `main.py`, inference, CI ✅; inference-time benchmark with the T15 model | T15 | 🟡 code ✅ | |
| T47 | **TenVoices-L2:** 👤 request L2-ARCTIC (form), download, select, QC, splits; run every merged model's command on L2; Whisper per first language; H5 | — | ⬜ | |

### At the end (needs everyone's merged work)

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T24 | Run T20 (results table), T40 (hypotheses), T34 (errors) and T23's figure command on all merged results; clean-environment reproducibility check → **feature freeze** | M1 | all runs, T20, T23, T34, T40, T47 | ⬜ | |
| T25 | Final report (paper format, ~8 pages) and README with real results, using the T37 / T38 literature notes | M1 | T24 | ⬜ | |
| T28 | 👤 Final checklist (every number matches `results/`), regenerate poster and slides (T27 command), tag `v1.0`, submit | all | T25, T26, T27 | ⬜ | |

### ✅ Done

| ID | Task | Owner | Done by · date |
|---|---|---|---|
| T00 | Project plan, README, poster, phrase list, guides, board; revised for two published datasets; analysis decisions frozen | M1 | Tanvir · 2026-10-01 … 2026-10-17 |
| T01 | Repository, pinned environment, config, audio pre-processing + tests | M1 | Tanvir · 2026-10-04 |
| T03 | `support/metrics.py` + tests | M3 slot | Tanvir · 2026-10-04 |
| T29 | Whisper pipeline code + tests | M1 | Tanvir · 2026-10-17 |
| T41 | Fluent Speech Commands downloaded and checked against the paper (`data/SOURCES.md`); L2-ARCTIC moved to T47 | M1 | Tanvir · 2026-10-17 |
| T42 | `support/select_subsets.py`; **TenVoices-FSC** selected: 10 speakers (5 F, 5 M) × 50 phrases × 2 takes = 1,000 clips | M1 | Tanvir · 2026-10-17 |
| T43 | `support/qc_report.py`; TenVoices-FSC: 997 / 1,000 clips pass, 3 flagged for clipping | M1 | Tanvir · 2026-10-17 |
| T08 | Processed copies, manifest, random split + 5 speaker-independent folds of TenVoices-FSC, leakage tests | M1 | Tanvir · 2026-10-17 |

### ✖ Cancelled

| ID | Was | Why |
|---|---|---|
| T02, T04–T07, T31, T32 | Recorder, pilot, recording sessions, recording QC | We use published datasets (2026-10-17) |

## 📝 Handoff Log (newest first)

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (simpler layout, one page per member)

- **Done:** the repository is easier to read and every member's work is in one place.
  - **One page per member** in `docs/team/` (`M1_tanvir.md`, `M2_rabbi.md`, `M3_mahir.md`, `M4_shafil.md`): role, files, models, task status, faculty update, requests, handoff log, and at the end that member's **task cards**. The 25 separate cards in `docs/tasks/` and the 4 pages in `docs/progress/` are merged into these pages; nothing was dropped.
  - **All code in `support/`:** `tools/select_subsets.py`, `tools/qc_report.py` and `tools/make_splits.py` moved there; run them as `python -m support.select_subsets` etc.
  - `CONTRIBUTING.md` and `tasks/RUNS.md` moved to `docs/`; the poster source moved to `others/poster.html`; the report template to `docs/team/individual_report_template.md`.
  - **README:** a *Team: Who Does What* table (responsibilities, models, done so far, next, link to each page) and the first measured results on TenVoices-FSC (Random Forest, Whisper), taken from `results/`.
- **Files:** moved as above; links and module paths updated in `AGENTS.md`, `README.md`, `docs/`, `support/`, `tests/`, `configs/config.yaml`. All tests pass.
- **For members:** after `git merge origin/main`, your page is `docs/team/<slot>_<name>.md`; your task cards are at the end of it.

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (no member waits for another), T42, T43, T08

- **Why:** members' tasks waited for each other (data selection by M2, the analysis plan by M3, everyone's runs for the results table and figures). Now every member task depends only on merged work or on the member's own tasks.
- **Data preparation done by M1 and merged:** **TenVoices-FSC** selected by `support/select_subsets.py` (10 speakers, 5 F and 5 M, × 50 phrases × 2 takes = **1,000 clips**, 24 min of speech; 29,980 of 30,043 source clips eligible), QC by `support/qc_report.py` (**997 / 1,000 pass**, 3 flagged for clipping), and the random split (720 / 80 / 200) plus 5 speaker-independent folds in `data/splits/fsc/`. Members get the audio themselves ([docs/RUNS.md](RUNS.md)); `support/dataset.py --build` checks every clip's SHA-256. TenVoices-L2 moves to M1 (T47, needs the form 👤).
- **Analysis decisions frozen** in `docs/PROJECT_PLAN.md` before any model was trained (pooled folds, which runs H1–H4 compare, the H5 test, no multiple-comparison correction, bootstrap settings), so T10 no longer gates any run.
- **New independent model per member (exploratory, E10):** WavLM Base+ (M2, T44), HuBERT Base (M3, T45), linear probe on frozen wav2vec 2.0 (M4, T46). The shared code supports them already (`--model …`; `--features w2v2 --model logreg`).
- **Aggregation split:** M3 writes and tests the results table (T20), error analysis (T34) and hypothesis tests (T40); M4 writes the figures command and connects the demo to their own SVM (T23); M1 runs them on all merged results at the end (T24).
- **Board:** reorganised by member. New task cards: RUNS.md, T09, T12–T14, T16–T18, T20, T23, T26, T27, T34, T40, T44–T46; T41 and T42 cards removed (done).
- **Next:** M1 runs Random Forest (T11) and Whisper (T30) on FSC, then the wav2vec 2.0 runs (T15, T19). Members can start every task on their page now.

### 2026-10-17 · Tanvir Ahmed (M1) · T08, T11, T15, T21, T22, T30 (code)

- **Done:** all of M1's pipeline code, each with unit tests on synthetic audio, so the experiments can start as soon as the subsets exist. The tasks stay 🟡 because their runs wait for data:
  - **T08** `support/dataset.py` (SHA-256-checked processed copies + manifest, split loader) and `support/make_splits.py` (random 80/20 split and 5 speaker-independent folds, 1 F + 1 M test speakers, same first language on TenVoices-L2, leakage checks).
  - **T11** `support/features.py` (78 MFCC statistics, cached), `support/evaluate.py` (one evaluation for every model; folds pooled, mean ± std) and `support/classical.py` (Random Forest / XGBoost / k-NN / SVM, grid on validation only). The grids are in `configs/config.yaml`.
  - **T15** `support/train_wav2vec2.py` (`--epochs`, `--noise white`, `--untrained`; best epoch on validation; fp16 on CUDA), `support/noise.py`, `support/device.py` and `notebooks/train_wav2vec2.ipynb` (Colab). One real training step of `facebook/wav2vec2-base` on real FSC clips runs on Apple MPS.
  - **T22** `support/inference.py` + `main.py` (audio, video via ffmpeg, microphone; top-5; `--asr whisper`; `--benchmark`) and `.github/workflows/tests.yml`. GitHub did not start the first run: *"account is locked due to a billing issue"* (fix in the GitHub account settings).
  - **T30** `support/whisper_asr.py --dataset … --split …` (each clip transcribed once; WER per speaker and first language for H5); the tests use a fixture copy of the old phrase list.
  - **T21** `support/noise_test.py` (the same white noise for every model, per clip and SNR, for paired tests of H3).
- **Found and fixed:** on macOS, PyTorch and XGBoost in one Python process crash (XGBoost after PyTorch) or hang (the other way round). The tools keep them in separate processes, `classical.py` refuses XGBoost when PyTorch is loaded, and the PyTorch tests run in a child process (`tests/_wav2vec2_check.py`). Full suite: **125 tests**, about 10 s.
- **Files:** `support/{dataset,features,evaluate,classical,noise,device,train_wav2vec2,inference,noise_test,whisper_asr}.py`, `support/make_splits.py`, `main.py`, `notebooks/train_wav2vec2.ipynb`, `.github/workflows/tests.yml`, `configs/config.yaml`, `tests/` (+ `_synthetic.py`, `_wav2vec2_check.py`, `fixtures/`), `support/README.md`, `AGENTS.md`, `README.md`, `docs/team/M1_tanvir.md`.
- **For others:** M2's `select_subsets.py` must write `data/subsets/<dataset>.csv` with the columns in DATASETS.md §2. `support/dataset.py` checks them. M4's demo can call `support.inference.load_recognizer(path).predict(samples, rate)`. M3's `stats.py` can pair noise-test predictions by `utt_id` and `snr_db`.
- **Next (M1):** once T43 and T10 are merged: build the processed copies and splits (T08), then the Random Forest (T11), then wav2vec 2.0 (T15 / T19) and Whisper (T30).

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (instructor confirmation)

- **Decision:** the instructor confirmed that the brief needs the data, not our own recordings: published datasets are acceptable. The two-dataset plan stands as it is.
- **Files:** `docs/PROJECT_PLAN.md` (open decision closed, risk removed), `docs/STATUS.md`, `docs/team/M1_tanvir.md`.
- **Blockers:** none.

### 2026-10-17 · Tanvir Ahmed (M1) · T41 (Fluent Speech Commands)

- **Done (by M1, at the lead's decision, to unblock the data work):** downloaded **Fluent Speech Commands** into `data/raw/fsc/` and checked it against the paper: 23,132 / 3,118 / 3,793 clips, 77 / 10 / 10 speakers, 30,043 WAV files all present, 248 phrasings, 31 intents, speaker demographics included. Source, reasons and SHA-256 are in `data/SOURCES.md`. The official route is a Google Group that needs a Google account. The old direct link is dead, and the Zenodo and `superb_demo` copies in DATASETS.md §1 are not the original archive (repackaged / a tiny demo subset), so that section should be updated (M2).
- **Still open in T41:** **L2-ARCTIC** needs the publisher's form (name, email, affiliation; the link arrives by email) 👤. Both archives still need to go to the team Drive (`TenVoices/raw/`).
- **Next:** Rabbi can start **T42** on the FSC half now (`data/raw/fsc/` locally, or the archive from Drive once uploaded).

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (all code visible)

- **Done:** the empty placeholder folders (`results/`, `notebooks/`, `others/`, `outputs/`, `data/raw/`, `data/processed/`, `data/splits/`) now have a README instead of `.gitkeep`, saying what goes there and which task fills it. They match the two-dataset layout. `support/README.md` is a code map of every module on `main` (task, owner, tests) and explains how to see unmerged code. New rule in `AGENTS.md` and `CONTRIBUTING.md`: members push their branch at the end of **every** session, unfinished work included (`T33: work in progress`), and open the pull request as a **draft** from the first push, so M1 can read all code before merging. M1 merges only pull requests marked ready.
- **Files:** `AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `support/README.md`, the placeholder READMEs, `.gitignore`, `docs/STATUS.md`.
- **Blockers:** none.

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (two published datasets)

- **Decision:** we no longer record our own speakers. TenVoices is now two 10-speaker subsets of published datasets, selected by script: **TenVoices-FSC** (main; Fluent Speech Commands, 10 speakers × 50 spoken commands × 2 takes, up to 1,000 clips, CC BY-NC-ND 4.0) and **TenVoices-L2** (accents; L2-ARCTIC v5.0, the 4 Hindi speakers plus a female and a male each from Arabic, Mandarin and Spanish × 50 shared sentences, 500 clips, CC BY-NC 4.0). Both papers and licences were checked on their official pages.
- **What changes:** every model now runs on both datasets with the same splits protocol (random 80/20 + speaker-independent 5 folds). On TenVoices-L2, each fold's test pair shares a first language, so folds 3–5 test accents never heard in training. H3 uses synthetic white noise only (no real noisy takes). **H5 is redrafted:** Whisper's per-speaker WER is higher on TenVoices-L2 than on TenVoices-FSC; T10 fixes the exact wording. New RQ6 (do the findings hold on both datasets?) and E9 (accents).
- **Tasks:** recording tasks cancelled (T02, T04–T07, T31, T32; see the table above). New M2 tasks **T41** (get both datasets, start now), **T42** (selection script), **T43** (QC). T08, T09, T21, T26, T30 reworded. The demo (T36) will show the 50 TenVoices-FSC commands.
- **Files:** `docs/PROJECT_PLAN.md` (v3.0), `docs/DATASETS.md` (new, replaces the recording protocol), `docs/DATA_COLLECTION_PROTOCOL.md` and `docs/consent_form.md` (removed), `docs/STATUS.md`, `AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `configs/config.yaml` (new `datasets` section), `tools/__init__.py`, `data/README.md`, `docs/team/` (T02 removed; T41, T42 new; T10, T33, T35, T36, T39 updated), `docs/team/` (lead notes and task tables).
- **Kept on purpose:** `data/prompts/phrases.csv` and `support/phrases.py` stay until T42 replaces them, because `tests/test_whisper_asr.py` still uses the old list (moved to a fixture in T30).
- **Next:** Rabbi starts T41 today. Tanvir asks the instructor whether published-data subsets satisfy "gather a dataset".
- **Blockers:** the Fluent Speech Commands access link could not be tested; DATASETS.md lists public copies and how to check them against the paper.

### 2026-10-17 · Tanvir Ahmed (M1) · T29

- **Done:** `support/whisper_asr.py`. Whisper's English normaliser on both sides, with neighbouring digit groups joined, leading zeros dropped and `6.30` read as a time, so spoken and written numbers, times and phone numbers compare equal. Also WER / CER per clip and per corpus, nearest-phrase matching (WER, ties by CER) with a demo-only rejection threshold, a batched `WhisperTranscriber` (CUDA > MPS > CPU), and `run_clip_list` writing `predictions.csv` (`utt_id, speaker_id, label, pred, transcript, wer, cer, match_wer, rejected`) + `metrics.json`. New `whisper` section in `configs/config.yaml`. 18 offline tests (no model download); all 61 tests pass.
- **Smoke test (not a result):** 16 clips of 8 phrases spoken by two macOS text-to-speech voices (Indian and British English), kept outside the repo. `openai/whisper-small` ran on MPS and matched every clip to the right phrase, while misspelling local names (Mohakhali, Sylhet, Sreemangal). That is what H5 tests on real recordings.
- **Files:** `support/whisper_asr.py`, `tests/test_whisper_asr.py`, `configs/config.yaml`, `AGENTS.md`, `README.md`, `docs/STATUS.md`, `docs/team/M1_tanvir.md`.
- **Notes:** T11's `support/evaluate.py` should read the same `utt_id, speaker_id, label, pred` columns. T30 adds `--split`. The first run downloads the model (~1 GB) into the Hugging Face cache.
- **Next:** M1 has no other unblocked task until T02 → T04; merge member branches as they come in.

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (any assistant or by hand)

- **Done:** members no longer need Claude. The rules moved to **`AGENTS.md`**, which many coding assistants read automatically; `CLAUDE.md` now only imports it. **`CONTRIBUTING.md`** gives the whole workflow as plain git commands, with a ready-made prompt for chat assistants. **Task cards** in `docs/team/` (T02, T10, T29, T33, T35–T39) say exactly which files to create, which existing functions to use, what to test and when a task is done, and list the files to paste into a chat assistant.
- **Files:** `AGENTS.md` (new), `CLAUDE.md`, `CONTRIBUTING.md` (new), `docs/team/` (new), `docs/STATUS.md`, `docs/PROJECT_PLAN.md`, `README.md`, `docs/team/`.
- **Next:** members start their start-now tasks from the cards; M1 does T29.
- **Blockers:** none.

### 2026-10-17 · Tanvir Ahmed (M1) · T00 (team of four, branch per member)

- **Done:** Mahir Ahmed takes **M3** (evaluation) and Shafil Ahmed takes **M4** (visualisation, demo, presentation). Every member now works on their own branch (`dev_rabbi`, `dev_mahir`, `dev_shafil`), edits only the files they own, and records progress and a plain-language **faculty update** in `docs/team/<slot>_<name>.md`; M1 reviews, merges and keeps this board.
- **Re-balanced tasks:** M1 keeps the complex shared work (splits, shared runners, training, Whisper, noise tests T21, inference). M3 gets statistics (T33), results tables (T20 → T40), error analysis (T34) and literature notes (T37). M4 gets the visualisation module (T35), demo skeleton (T36), literature notes (T38) and poster v2 (T39). Recording is split per member: M2 S05–S06 (T06), M3 S03 + S07–S08 (T31), M4 S04 + S09–S10 (T32); M1 records only S01.
- **Start now (no recordings needed):** M3: T10, T33, T37 · M4: T35, T36, T38, T39 · M1: T29.
- **T02 change for M2:** the recorder writes metadata one file per speaker, so recordings made on different branches merge cleanly.
- **Files:** `CLAUDE.md`, `README.md`, `docs/STATUS.md`, `docs/PROJECT_PLAN.md`, `docs/DATA_COLLECTION_PROTOCOL.md`, `docs/team/individual_report_template.md`, `docs/team/` (new).
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

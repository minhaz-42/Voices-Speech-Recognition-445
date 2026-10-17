# M1 · Tanvir Ahmed (lead): Progress

| | |
|---|---|
| **Role** | Data pipeline and splits, shared training / evaluation code, Whisper pipeline, noise tests, inference, integration; reviews and merges every member's branch |
| **Models** | Random Forest (traditional) · wav2vec 2.0 fine-tuned for 15 epochs with and without white noise · Whisper zero-shot |
| **Branch** | `main` |
| **Owns** | `main.py`, `support/config.py`, `support/audio.py`, `support/dataset.py`, `support/features.py`, `support/classical.py`, `support/evaluate.py`, `support/train_wav2vec2.py`, `support/noise.py`, `support/whisper_asr.py`, `tools/make_splits.py`, `configs/`, `notebooks/`, `requirements.txt`, `.github/`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, `docs/STATUS.md`, `docs/PROJECT_PLAN.md`, `docs/tasks/`, `results/random_forest/`, `results/w2v2_15ep_noise/`, `results/w2v2_15ep/`, `results/whisper_small/`, `results/noise/` |
| **Report** | Abstract, sections 1, 2 (with Mahir's and Shafil's literature notes), 4 and 8; final integration |

The task board and handoff log for M1 are in [STATUS.md](../STATUS.md). This file holds the faculty update.

## My tasks

| ID | Task | Depends on | Status |
|---|---|---|---|
| T00, T01 | Plan, repository, environment, config, audio pre-processing | — | ✅ |
| T29 | Whisper pipeline code `support/whisper_asr.py` | T01, T03 | ✅ 2026-10-17 |
| T00 | Switch to two published datasets (plan v3.0, DATASETS.md, board, cards) | — | ✅ 2026-10-17 |
| — | 👤 Ask the instructor whether published-data subsets satisfy "gather a dataset" | — | ⬜ |
| T08 | Processed audio, manifest, splits per dataset, leakage tests | T43 (M2) | ⬜ |
| T11 | Features, shared classical runner and evaluation; Random Forest | T08, T10 (M3) | ⬜ |
| T15 | Shared wav2vec 2.0 training script; 15 epochs + noise | T08, T11 | ⬜ |
| T19 | 15 epochs without noise (H3 ablation) | T15 | ⬜ |
| T22 | `main.py` inference, benchmark, CI | T15 | ⬜ |
| T30 | Whisper runs on all splits, H5, transcript mode | T08, T11, T22, T29 | ⬜ |
| T21 | Noise tests, H3 | T15, T19 | ⬜ |
| T24 | Reproducibility check, feature freeze | T21–T23, T30, T34, T40 | ⬜ |
| T25 | Final report and README | T24, T37, T38 | ⬜ |
| — | Review and merge every member's branch | — | ongoing |

## Faculty update

Plain-language summary for the weekly project update, newest week first. Only finished work; numbers only from `results/`.

### Week of 2026-10-17
- Changed how we get our data: instead of recording volunteers, we build two 10-speaker datasets from published corpora with a fixed selection script. **TenVoices-FSC** (from Fluent Speech Commands) has 50 spoken smart-home commands such as "turn on the lights" or "turn up the heat in the kitchen", two takes each. **TenVoices-L2** (from L2-ARCTIC) has 50 sentences read by non-native English speakers, including all four Hindi speakers, the accent in that corpus closest to ours. I checked both papers and licences and wrote the dataset guide (`docs/DATASETS.md`) and the new plan (v3.0): every model is now tested on both datasets.
- Added a fourth pipeline to the project: **Whisper**, a general-purpose speech-to-text model, used without training. Its transcript is matched to the nearest of our 50 phrases, so it can be compared with the trained models, and it adds a new research question (RQ5) and hypothesis (H5), now about non-native speech.
- Built the Whisper pipeline (`support/whisper_asr.py`). It transcribes a recording, normalises the text (numbers, times, phone numbers), finds the nearest of our 50 phrases and measures word and character error rate. 18 unit tests. A first end-to-end check on synthetic text-to-speech clips worked; it matched every clip to the right phrase but misspelled local place names, the kind of gap H5 now tests on non-native speech.
- Brought two new members into the team (Mahir, Shafil) and split the project into independent tasks with clear file ownership. Every member now works on their own branch, by hand or with any AI assistant, using task cards that say exactly what to build; I review and merge their work.

### Week of 2026-10-01
- Wrote the project plan, the data-collection protocol, the consent form, the 50-phrase list and the task board, and published the repository.
- Built the shared audio pre-processing (`support/audio.py`). Any recording becomes mono 16 kHz audio with the silence trimmed, normalised to −1 dBFS and padded to 6 s, and speech is never cut. 23 unit tests.
- Built the shared evaluation metrics (`support/metrics.py`): accuracy, macro and weighted F1, confusion matrices and per-speaker accuracy, with 14 tests checked against a hand-worked example.
- Pinned the software environment (Python 3.11) so every member runs the same versions.

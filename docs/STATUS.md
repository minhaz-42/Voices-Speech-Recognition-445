# 📋 TenVoices: Status Board

> **Source of truth for progress.** Claude reads this file at the start of every session and updates it at the end (rules in [CLAUDE.md](../CLAUDE.md)).
> **Legend:** ⬜ to do · 🟡 in progress · ✅ done · ⛔ blocked · 👤 needs people (recording, consent, video)

**Current phase:** Week 1: setup and tools
**Next up:** finish T01 (M1 · Tanvir Ahmed): add Rabbi as collaborator, pin requirements, config + audio utilities
**Last updated:** 2026-10-01 by Tanvir Ahmed (M1)

## 👥 Team

| Slot | Member | Workstream |
|---|---|---|
| M1 | Tanvir Ahmed (lead) | Setup, data pipeline, ASR models and our approach, integration |
| M2 | Md. Shahriar Rakib Rabbi | Dataset collection and curation, traditional ML baselines, augmentation, dataset release |
| M3 | Tanvir Ahmed *(open slot)* | Evaluation: metrics, experiments, statistics, error analysis |
| M4 | Tanvir Ahmed *(open slot)* | Demo app, figures, poster, slides, video |

## 💬 How to work with Claude

1. Pull, open Claude Code in the repo and say who you are: *"I'm M2. What is completed?"*
2. Then say: *"Complete my next tasks."* Claude checks the dependencies below and only starts tasks that are unblocked.
3. When you're done for the session, say: *"Wrap up and push."* Claude updates this board and the handoff log, commits and pushes.
4. Tell the next member to pull.

## 🗂️ Task Board

### Week 1: Setup and tools

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T00 | Project plan, README, poster, protocol, consent form, draft prompts, `CLAUDE.md`, this board → committed | M1 | — | ✅ | Tanvir · 2026-10-01 |
| T01 | Publish the GitHub repo and add Rabbi as collaborator; pin `requirements.txt`; `configs/config.yaml` + `support/config.py`; `support/audio.py` (load, mono, 16 kHz, loudness, VAD trim) + tests → tests pass and a sample WAV loads and trims | M1 | T00 | 🟡 | |
| T02 | Recorder tool `tools/recorder.py` (Gradio; file naming + `utterances.csv` rows); 300-sentence pool `data/prompts/unique_pool.csv` (Common Voice, CC0); printable prompt sheets → a test session saves correct files | M2 | T01 | ⬜ | |
| T03 | `support/metrics.py` (WER, CER, SER + Whisper normaliser) + `docs/METRICS_SPEC.md` + tests → hand-computed cases pass | M3 | T01 | ⬜ | |
| T04 | 👤 Pilot sessions S01 (Tanvir) and S02 (Rabbi): consent signed, files backed up to Drive → 2 × 120 recordings | M1 + M2 | T02 | ⬜ | |
| T05 | `tools/qc_report.py` (completeness, duration, clipping, silence, SNR) run on the pilot; fix the issues found → protocol v1 frozen | M2 | T04 | ⬜ | |

### Week 2: Dataset collection

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T06 | 👤 Record volunteers S03–S06 (recruited by M1) and S07–S10 (recruited by M2), including noisy takes; target 5F/5M overall → 10/10 speakers | M1 + M2 | T05 | ⬜ | |
| T07 | QC all sessions; cross-check transcripts (M1 checks M2's speakers, M2 checks M1's); re-records; spontaneous transcripts → ≥ 97% of takes pass | M2 (+ M1) | T06 | ⬜ | |
| T08 | Processed 16 kHz copies + SHA-256 manifest; `support/dataset.py`; `tools/make_splits.py` → `data/splits/folds.json`; leakage tests → tests prove no speaker or text leakage; tag `data-v1` | M1 | T07 | ⬜ | |
| T09 | Dataset statistics figure + datasheet `data/README.md` | M2 | T08 | ⬜ | |
| T10 | `docs/ANALYSIS_PLAN.md`: hypotheses H1–H4, subsets, statistical tests → **frozen** before any main experiment | M3 | T03, T08 | ⬜ | |

### Week 3: Baselines

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T11 | `support/asr.py` (Whisper tiny/base/small, wav2vec 2.0, Vosk behind one interface); zero-shot runs on all folds → `results/zero_shot/` | M1 | T08 | ⬜ | |
| T12 | `support/features.py` (MFCC + Δ + ΔΔ) and `support/classical/` (k-NN, SVM, Random Forest, XGBoost, DTW) on the 15 commands → `results/classical/` | M2 | T08 | ⬜ | |
| T13 | `support/experiments.py` + `support/stats.py`: score every fold and subset; bootstrap CIs; Wilcoxon → `results/tables/baselines.csv` | M3 | T10, T11, T12 | ⬜ | |
| T14 | `support/visualization.py`: baseline figures → `images/` | M4 | T13 | ⬜ | |
| T15 | 👤 Individual update reports, if the course requires them (written from the handoff log) | all | T13 | ⬜ | |

### Week 4: Our approach

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T16 | 👤 Noise bank (20 × 60 s, split into train-noise / test-noise) + `support/augment.py` (noise at a target SNR, speed, gain) | M2 | T08 | ⬜ | |
| T17 | `configs/lexicon_bd.txt` + domain-aware decoding + VAD in inference + LoRA notebook; train all 5 folds → `results/ours/` (adapters go to Drive, not git) | M1 | T11, T16 | ⬜ | |
| T18 | `support/commands.py` (fuzzy matching + rejection) + main results table, H1–H3 → `results/tables/main.csv` | M3 | T12, T17 | ⬜ | |
| T19 | Gradio demo `app.py` v0 (microphone → transcript, model switch, command mode) | M4 | T17 | ⬜ | |

### Week 5: Experiments and analysis

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T20 | Ablations + speed benchmark; `main.py` (transcribe / evaluate / reproduce) | M1 | T17 | ⬜ | |
| T21 | Noise sweep (20/10/5/0 dB, held-out noise) + real-noise results + error analysis → `results/` | M3 | T16, T18 | ⬜ | |
| T22 | Final figures + demo polish | M4 | T19, T21 | ⬜ | |
| T23 | Unit tests + GitHub Actions CI + clean-environment reproducibility check → **feature freeze** | M1 | T20, T21 | ⬜ | |

### Week 6: Delivery

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T24 | Final report (`others/final_report.tex`, IEEE, 8 pages) + final README | M1 | T23 | ⬜ | |
| T25 | Dataset release (consenting speakers; FLAC, metadata, datasheet, checksums) as a GitHub Release asset | M2 | T23 | ⬜ | |
| T26 | Results section of the report; check every number against `results/` | M3 | T23 | ⬜ | |
| T27 | Results poster, slides, 👤 1-minute demo video | M4 | T22 | ⬜ | |
| T28 | 👤 Final checklist, tag `v1.0`, submit | all | T24, T25, T26, T27 | ⬜ | |

## 📝 Handoff Log (newest first)

### 2026-10-01 · Tanvir Ahmed (M1) · T01 (part 1)
- **Done:** published the repo at https://github.com/minhaz-42/Voices-Speech-Recognition-445 and pushed all planning files.
- **Next:** finish T01 (M1): add Rabbi as a collaborator, pin `requirements.txt`, then `configs/config.yaml`, `support/config.py` and `support/audio.py` with tests.
- **Blockers:** Rabbi's GitHub username is needed to add him as a collaborator.

### 2026-10-01 · Tanvir Ahmed (M1) · T00
- **Done:** project plan, README, poster, data-collection protocol, consent form, draft prompts (60 shared + 5 spontaneous), `CLAUDE.md` and this status board.
- **Next:** T01 (M1). Once T01 is pushed, M2 can start T02 and M3 can start T03.
- **Blockers:** course details and deadlines are TBD; add Rabbi's git user.name to `CLAUDE.md`.

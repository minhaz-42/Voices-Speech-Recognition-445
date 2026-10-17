# M2 · Md. Shahriar Rakib Rabbi: Progress

| | |
|---|---|
| **Role** | Recorder, QC, dataset curation and release |
| **Models** | XGBoost (traditional) · wav2vec 2.0 fine-tuned for 8 epochs + white noise |
| **Branch** | `dev_rabbi` (Tanvir merges it into `main`) |
| **Owns** | `tools/recorder.py`, `tools/qc_report.py`, `support/phrases.py`, `data/prompts/`, `data/metadata/`, `data/README.md` (datasheet), `docs/DATA_COLLECTION_PROTOCOL.md`, `docs/consent_form.md`, `results/xgboost/`, `results/w2v2_8ep_noise/` |
| **Report** | Section 3 (the TenVoices dataset) |

Only Rabbi (or Rabbi's assistant) edits this file. Full task text and dependencies: [STATUS.md](../STATUS.md).

## What changed on 2026-10-17

- You now work on the branch `dev_rabbi` instead of `main`; Tanvir merges it. Work by hand or with any AI assistant: [CONTRIBUTING.md](../../CONTRIBUTING.md) has the steps, and [docs/tasks/T02.md](../tasks/T02.md) is the card for the recorder. With an assistant that works in the repository, just say *"I'm Rabbi (M2). What is completed? Then complete my next tasks."*
- `docs/STATUS.md` is edited only by Tanvir. Your task status, handoff notes and faculty update go in this file.
- **T02:** the recorder writes metadata **one file per speaker** (`data/metadata/utterances/<SPK>.csv`, `data/metadata/speakers/<SPK>.csv`), because Mahir and Shafil now record speakers on their own branches too.
- You recruit and record volunteers S05–S06; Mahir and Shafil record S03–S04 and S07–S10.

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T02 | Recorder tool `tools/recorder.py` (phrase catalogue loader done) | T01 | 🟡 | |
| T04 | 👤 Pilot session S02 (with Tanvir's S01) | T02 | ⬜ | |
| T05 | QC report `tools/qc_report.py` on the pilot → protocol v1 frozen | T04 | ⬜ | |
| T06 | 👤 Record volunteers S05–S06 | T05 | ⬜ | |
| T07 | QC all sessions, collect listen-checks, re-records (you listen-check S01, S03, S07–S08) | T04, T06, T31, T32 | ⬜ | |
| T09 | Dataset statistics figure + datasheet `data/README.md` | T08 | ⬜ | |
| T12 | XGBoost results → `results/xgboost/` | T11 | ⬜ | |
| T16 | wav2vec 2.0, 8 epochs + white noise → `results/w2v2_8ep_noise/` | T15 | ⬜ | |
| T26 | Dataset release as a GitHub Release asset | T24 | ⬜ | |

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

### Week of 2026-10-07
- Wrote `support/phrases.py`, which loads and checks the 50-phrase catalogue (5 categories of 10, unique IDs, the 10 noisy-take phrases), with 6 unit tests.
- Started the recorder tool (T02).

## Requests for the lead

*(None.)*

## Handoff log (newest first)

Entries up to 2026-10-07 are in the Handoff Log of [STATUS.md](../STATUS.md).

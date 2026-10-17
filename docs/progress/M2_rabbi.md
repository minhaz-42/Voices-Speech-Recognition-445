# M2 · Md. Shahriar Rakib Rabbi: Progress

| | |
|---|---|
| **Role** | Datasheet, listen-check, dataset statistics, subset release |
| **Models** | XGBoost (traditional) · wav2vec 2.0 8 epochs + white noise · **WavLM Base+** 15 epochs + white noise |
| **Branch** | `dev_rabbi` (Tanvir merges it into `main`) |
| **Owns** | `support/phrases.py`, `data/README.md` (datasheet), `data/metadata/`, `data/listen_check_<dataset>.csv`, `tools/dataset_stats.py`, `tools/make_release.py`, `docs/DATASETS.md`, `results/xgboost/`, `results/w2v2_8ep_noise/`, `results/wavlm_15ep_noise/` |
| **Report** | Section 3 (the TenVoices subsets) |

Only Rabbi (or Rabbi's assistant) edits this file. Full task text and dependencies: [STATUS.md](../STATUS.md).

## Note from Tanvir (2026-10-17, evening): nothing to wait for

Every one of your tasks below depends only on work that is already merged or on your own earlier tasks. Data preparation (download, selection, QC, splits) is done and merged for TenVoices-FSC, so you can run your models today: [docs/tasks/RUNS.md](../tasks/RUNS.md) shows how. Work that needs everyone's results is split: you write and test the code, and Tanvir runs it on all results at the end.

## Note from Tanvir (2026-10-17, afternoon): no more recording

- We no longer record our own speakers. TenVoices is now two 10-speaker subsets of published datasets: **TenVoices-FSC** from Fluent Speech Commands and **TenVoices-L2** from L2-ARCTIC. Everything about them is in [docs/DATASETS.md](../DATASETS.md), which you own from now on.
- **Cancelled:** the recorder (T02), the pilot (T04), protocol QC (T05), recording volunteers (T06) and session QC (T07). Your `support/phrases.py` and its tests stay; T42 extends them for the new phrase lists.
- *(Superseded the same evening: T41–T43, the dataset download, selection and QC, were done by Tanvir so that nobody waits for data. Your tasks are in **My tasks** below.)*
- I removed the recording protocol and consent form and rewrote `data/README.md` as a placeholder for the new plan; they were planning files with no work of yours in them.

## What changed on 2026-10-17 (morning)

- You now work on the branch `dev_rabbi` instead of `main`; Tanvir merges it. Work by hand or with any AI assistant: [CONTRIBUTING.md](../../CONTRIBUTING.md) has the steps, and each task has a card in [docs/tasks/](../tasks/). With an assistant that works in the repository, just say *"I'm Rabbi (M2). What is completed? Then complete my next tasks."*
- `docs/STATUS.md` is edited only by Tanvir. Your task status, handoff notes and faculty update go in this file.
- *(The recorder and recording plans from that morning were cancelled the same afternoon; see the note above.)*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T02 | Phrase catalogue loader `support/phrases.py` (the recorder part was cancelled) | — | ✅ | Rabbi · 2026-10-07 |
| T12 | XGBoost → `results/xgboost/fsc/` ([card](../tasks/T12.md)) | — | ⬜ | |
| T16 | wav2vec 2.0, 8 epochs + white noise → `results/w2v2_8ep_noise/fsc/` ([card](../tasks/T16.md)) | — | ⬜ | |
| T44 | WavLM Base+, 15 epochs + white noise → `results/wavlm_15ep_noise/fsc/` ([card](../tasks/T44.md)) | — | ⬜ | |
| T09 | Datasheet, dataset statistics figure, 👤 listen-check of 100 clips ([card](../tasks/T09.md)) | — | ⬜ | |
| T26 | Subset release package, draft GitHub Release ([card](../tasks/T26.md)) | T09 | ⬜ | |

**Start with:** T12 (minutes on a laptop) and T09. Run T16 and T44 on Colab. Data selection (formerly T41–T43) is done and merged by Tanvir, so nothing you do waits for anyone.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

### Week of 2026-10-07
- Wrote `support/phrases.py`, which loads and checks the 50-phrase catalogue (5 categories of 10, unique IDs, the 10 noisy-take phrases), with 6 unit tests.
- Started the recorder tool (T02).

## Requests for the lead

*(None.)*

## Handoff log (newest first)

Entries up to 2026-10-07 are in the Handoff Log of [STATUS.md](../STATUS.md).

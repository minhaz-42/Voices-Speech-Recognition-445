# M2 · Md. Shahriar Rakib Rabbi: Progress

| | |
|---|---|
| **Role** | Datasets: access, subset selection, QC, datasheet and subset release |
| **Models** | XGBoost (traditional) · wav2vec 2.0 fine-tuned for 8 epochs + white noise |
| **Branch** | `dev_rabbi` (Tanvir merges it into `main`) |
| **Owns** | `tools/select_subsets.py`, `tools/qc_report.py`, `support/phrases.py`, `data/prompts/`, `data/subsets/`, `data/metadata/`, `data/SOURCES.md`, `data/README.md` (datasheet), `docs/DATASETS.md`, `results/xgboost/`, `results/w2v2_8ep_noise/` |
| **Report** | Section 3 (the TenVoices subsets) |

Only Rabbi (or Rabbi's assistant) edits this file. Full task text and dependencies: [STATUS.md](../STATUS.md).

## Note from Tanvir (2026-10-17, afternoon): no more recording

- We no longer record our own speakers. TenVoices is now two 10-speaker subsets of published datasets: **TenVoices-FSC** from Fluent Speech Commands and **TenVoices-L2** from L2-ARCTIC. Everything about them is in [docs/DATASETS.md](../DATASETS.md), which you own from now on.
- **Cancelled:** the recorder (T02), the pilot (T04), protocol QC (T05), recording volunteers (T06) and session QC (T07). Your `support/phrases.py` and its tests stay; T42 extends them for the new phrase lists.
- **Your new tasks:** **T41** get both datasets (start today: the L2-ARCTIC download link comes by email), **T42** the selection script, **T43** QC of both subsets. Cards: [T41](../tasks/T41.md), [T42](../tasks/T42.md); T43 gets its card when T42 exists.
- I removed the recording protocol and consent form and rewrote `data/README.md` as a placeholder for the new plan; they were planning files with no work of yours in them.

## What changed on 2026-10-17 (morning)

- You now work on the branch `dev_rabbi` instead of `main`; Tanvir merges it. Work by hand or with any AI assistant: [CONTRIBUTING.md](../../CONTRIBUTING.md) has the steps, and each task has a card in [docs/tasks/](../tasks/). With an assistant that works in the repository, just say *"I'm Rabbi (M2). What is completed? Then complete my next tasks."*
- `docs/STATUS.md` is edited only by Tanvir. Your task status, handoff notes and faculty update go in this file.
- *(The recorder and recording plans from that morning were cancelled the same afternoon; see the note above.)*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T02 | ~~Recorder tool~~: cancelled 2026-10-17 (the phrase catalogue loader `support/phrases.py` is done and stays) | T01 | ✖ | |
| T41 | 👤 Get both datasets into `data/raw/`, team Drive, `data/SOURCES.md` | — | ⬜ | |
| T42 | Selection script `tools/select_subsets.py` → subset and phrase lists | T41 | ⬜ | |
| T43 | QC report `tools/qc_report.py` on both subsets → ≥ 97% pass | T42 | ⬜ | |
| T09 | Dataset statistics figure + datasheet `data/README.md` for both subsets | T08 | ⬜ | |
| T12 | XGBoost results on both datasets → `results/xgboost/` | T11 | ⬜ | |
| T16 | wav2vec 2.0, 8 epochs + white noise → `results/w2v2_8ep_noise/` | T15 | ⬜ | |
| T26 | Subset release (lists, script, checksums, datasheet; no audio) as a GitHub Release | T24 | ⬜ | |

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

### Week of 2026-10-07
- Wrote `support/phrases.py`, which loads and checks the 50-phrase catalogue (5 categories of 10, unique IDs, the 10 noisy-take phrases), with 6 unit tests.
- Started the recorder tool (T02).

## Requests for the lead

*(None.)*

## Handoff log (newest first)

Entries up to 2026-10-07 are in the Handoff Log of [STATUS.md](../STATUS.md).

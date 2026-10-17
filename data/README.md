# TenVoices Data (planned)

TenVoices is two 10-speaker subsets of published datasets, selected by script: **TenVoices-FSC** from Fluent Speech Commands and **TenVoices-L2** from L2-ARCTIC. Sources, licences, download steps and the selection rules: [`docs/DATASETS.md`](../docs/DATASETS.md).

This folder holds **lists and metadata only**. Audio is never committed to git, and the datasets' audio is never re-published.

| Path | Contents | Status |
|---|---|---|
| `raw/fsc/`, `raw/l2arctic/` | The extracted original datasets (gitignored) | Task T41 |
| `SOURCES.md` | Where each dataset came from: URL, version, date, licence, SHA-256, file counts | Task T41 |
| `prompts/fsc_phrases.csv`, `prompts/l2arctic_sentences.csv` | The 50 classes of each subset | Task T42 |
| `subsets/fsc.csv`, `subsets/l2arctic.csv` | The selected clips: speaker, phrase, take, source path, duration, SHA-256 | Task T42 |
| `subsets/qc_<dataset>.csv` | Quality-control results | Task T43 |
| `processed/` | 16 kHz working copies (gitignored) | Task T08 |
| `splits/<dataset>/` | Random 80/20 split and 5 speaker-independent folds | Task T08 |
| `prompts/phrases.csv`, `metadata/*_template.csv` | From the earlier recording plan; `phrases.csv` is still used by a test and goes in T30 | Retiring |

The full datasheet (motivation, composition, selection, preprocessing, uses, licences) replaces this file in task T09.

**Licences:** Fluent Speech Commands is CC BY-NC-ND 4.0 (academic research only); L2-ARCTIC is CC BY-NC 4.0. We publish only the lists, the selection script and checksums.

# TenVoices Dataset (planned)

This folder holds the dataset's **metadata only**. Audio is never committed to git.

| Path | Contents | Status |
|---|---|---|
| `prompts/prompts_v0.csv` | 60 shared prompts (5 categories) + 5 spontaneous topics, with text groups and noisy-re-record flags | Draft v0, finalised on Day 2 |
| `prompts/unique_pool.csv` | 300 unique sentences (30 per speaker) | Built on Day 2 |
| `metadata/*_template.csv` | Column schemas for `speakers.csv` and `utterances.csv` | Ready |
| `splits/folds.json` | 5 speaker-independent folds (2 test, 1 dev, 7 train speakers each) | Generated on Day 12 |
| `raw/`, `processed/` | Local audio working copies (gitignored) | Weeks 1–2 |

The full datasheet (motivation, composition, collection process, preprocessing, uses, distribution) replaces this file on **Day 13**. Collection rules: [`docs/DATA_COLLECTION_PROTOCOL.md`](../docs/DATA_COLLECTION_PROTOCOL.md).

**Licence (planned):** audio and transcripts from speakers who opted in are released under **CC BY 4.0** as a GitHub Release asset (`tenvoices-v1.0.zip`).

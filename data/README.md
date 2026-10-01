# TenVoices Dataset (planned)

This folder holds the dataset's **metadata only**. Audio is never committed to git.

| Path | Contents | Status |
|---|---|---|
| `prompts/phrases.csv` | The 50 phrases (5 categories × 10), with the 10 noisy-take flags | Draft, finalised in task T02 |
| `metadata/*_template.csv` | Column schemas for `speakers.csv` and `utterances.csv` | Ready |
| `splits/` | Random 80/20 split and 5 speaker-independent folds | Generated in task T08 |
| `raw/`, `processed/` | Local audio working copies (gitignored) | Weeks 1–2 |

The full datasheet (motivation, composition, collection process, preprocessing, uses, distribution) replaces this file in task T09. Collection rules: [`docs/DATA_COLLECTION_PROTOCOL.md`](../docs/DATA_COLLECTION_PROTOCOL.md).

**Licence (planned):** audio from speakers who opted in is released under **CC BY 4.0** as a GitHub Release asset (`tenvoices-v1.0.zip`).

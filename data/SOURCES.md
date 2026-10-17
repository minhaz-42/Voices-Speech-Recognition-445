# Data sources

Where the source audio of TenVoices came from, so anyone can check they have exactly the same files. Rules and selection: [docs/DATASETS.md](../docs/DATASETS.md). The audio itself is never committed.

## Fluent Speech Commands → `data/raw/fsc/`

| | |
|---|---|
| Downloaded | 2026-10-17 by Tanvir Ahmed (M1), task T41 |
| URL used | https://huggingface.co/datasets/leo19941227/fluent_speech_commands/resolve/main/fluent.tar.gz |
| Why this copy | The official page ([fluent.ai](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)) gives access through a Google Group that needs a Google account, and the older direct link on fluent.ai now returns 404. The two public copies first listed in DATASETS.md were not usable: the Zenodo record is a repackaged WebDataset version without the original CSV files, and the `superb_demo` link is a tiny demo subset. This copy is the original archive (checks below). |
| Version | Original release, as described in Lugosch et al., Interspeech 2019 (arXiv:1904.03670) |
| Licence | CC BY-NC-ND 4.0, academic research only (the licence PDF is inside the archive) |
| Archive | `fluent.tar.gz`, 1.55 GB, SHA-256 `c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b` |
| Layout | `fluent_speech_commands_dataset/wavs/speakers/<speakerId>/*.wav`, `…/data/{train,valid,test}_data.csv`, `…/data/speaker_demographics.csv`, `readme.md`, licence PDF |

**Checks against the paper (Table 1), all passed:**

| Split file | Clips | Speakers | Paper |
|---|---|---|---|
| `train_data.csv` | 23,132 | 77 | 23,132 / 77 |
| `valid_data.csv` | 3,118 | 10 | 3,118 / 10 |
| `test_data.csv` | 3,793 | 10 | 3,793 / 10 |
| **Total** | **30,043** WAV files, all present | 97 | 30,043 |

248 distinct transcriptions and 31 intents (action, object, location), as in the paper. `speaker_demographics.csv` has 101 rows with `speakerId`, gender, first language, fluency and age range.

## L2-ARCTIC v5.0 → `data/raw/l2arctic/`

**Not downloaded yet.** The publisher's page ([psi.engr.tamu.edu/l2-arctic-corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/)) needs a form with a person's name, email and affiliation; the download link arrives by email. A team member fills it in (task T41, 👤), then this section gets the same table as above.

## Team Drive

Both original archives also go to the private team Drive (`TenVoices/raw/fsc/`, `TenVoices/raw/l2arctic/`), unmodified, so other members can copy them instead of downloading. Never share them publicly.

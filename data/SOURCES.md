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

## L2-ARCTIC → `data/raw/l2arctic/`

| | |
|---|---|
| Downloaded | 2026-10-17 by Tanvir Ahmed (M1), task T47 |
| Source used | A free Hugging Face copy, [`NathanRoll/l2-arctic-dataset`](https://huggingface.co/datasets/NathanRoll/l2-arctic-dataset) (not gated, no form) |
| Files | `https://huggingface.co/datasets/NathanRoll/l2-arctic-dataset/resolve/main/data/train-00000-of-00002.parquet` (349 MB, SHA-256 `0475980d781735898822a76f71a8f5e1b848a24e699aed4d530541d587af3e69`) and `…/train-00001-of-00002.parquet` (364 MB, SHA-256 `a5db0202bcecd3b285e1dd47f8551f2dfc57f63c9d4076b98198c8f7a402120b`), saved in `data/raw/l2arctic_hf/` |
| Why this copy | The publisher's download ([psi.engr.tamu.edu/l2-arctic-corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/)) needs a form with a person's name, email and affiliation and an emailed link; this copy is available at once |
| Licence | CC BY-NC 4.0 (the corpus's licence, which allows sharing unmodified copies for non-commercial use) |
| Contents | 15 of the corpus's 24 speakers (Arabic ZHAA; Hindi ASI, RRBI, TNI; Korean HJK, YKWK; Mandarin BWC, LXC, NCC; Spanish EBVS, ERMS, MBMPS, NJS; Vietnamese HQTV, PNV), each with the same 150 prompts `arctic_a0001` … `arctic_a0150`; 2,250 clips, 2.2 h; the **original 44.1 kHz WAV files** stored as bytes (every clip starts with a `RIFF` header) |
| Unpacked with | `python -m support.l2arctic_from_hf` → `data/raw/l2arctic/<SPK>/wav/arctic_<id>.wav` (bytes unchanged) and `…/transcript/arctic_<id>.txt` |

**Checks, all passed:**

- **Transcripts:** compared with the official CMU ARCTIC prompt list (`http://www.festvox.org/cmu_arctic/cmuarctic.data`) after lower-casing and removing punctuation: 2,183 of 2,250 identical. The others are spelling variants ("rifle shot" for "rifle-shot") or the speaker's actual reading ("and etc"). The selection only keeps prompts whose text is identical for all 10 speakers.
- **Speakers:** the copy's first-language and gender labels are consistent for every speaker and match the published L2-ARCTIC speaker table.
- **Audio:** 44.1 kHz WAV for every clip; the selection and QC reread every selected file (`data/subsets/qc_l2arctic.csv`: 500 / 500 clips pass).

## Team Drive

Both original archives also go to the private team Drive (`TenVoices/raw/fsc/`, `TenVoices/raw/l2arctic/`), unmodified, so other members can copy them instead of downloading. Never share them publicly.

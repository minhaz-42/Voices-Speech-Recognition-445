# TenVoices: Datasets

> **Version 1.1 · 17 Oct 2026.** Replaces the recording protocol: we no longer record our own speakers. Owner: **M2**; reviewed by **M1**.
> Tasks (all M1, so no member waits for data): T41 (download), T42 (selection), T43 (QC), T08 (splits); TenVoices-L2 in T47. M2: T09 (datasheet, listen-check, statistics). **TenVoices-FSC is ready:** subsets, QC and splits are merged, and the source is recorded in [data/SOURCES.md](../data/SOURCES.md). Design and hypotheses: [PROJECT_PLAN.md](PROJECT_PLAN.md).

TenVoices is two 10-speaker subsets of published datasets, selected by script:

| | **TenVoices-FSC** (main) | **TenVoices-L2** (accents) |
|---|---|---|
| Source | Fluent Speech Commands | L2-ARCTIC v5.0 |
| Paper | Lugosch et al., Interspeech 2019, [arXiv:1904.03670](https://arxiv.org/abs/1904.03670) | Zhao et al., Interspeech 2018, [doi:10.21437/Interspeech.2018-1110](https://doi.org/10.21437/Interspeech.2018-1110) |
| Official page | [fluent.ai/research/fluent-speech-commands](https://fluent.ai/research/fluent-speech-commands/) | [psi.engr.tamu.edu/l2-arctic-corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/) |
| Licence | CC BY-NC-ND 4.0, academic research only | CC BY-NC 4.0 |
| Source size | 30,043 clips, 97 speakers, 248 phrasings, 31 intents, 19 h; splits 77 / 10 / 10 speakers | 24 speakers, 6 first languages × (2 F + 2 M), ~1,132 prompts each; the free copy we use: 15 of the speakers × the same 150 prompts |
| Audio | 16 kHz mono WAV | 44.1 kHz WAV |
| Our subset | 10 speakers (5 F, 5 M) × 50 phrases × 2 takes, up to 1,000 clips | 10 speakers (5 F, 5 M) × 50 sentences × 1 take, 500 clips |

## 1. Access and download (task T41, 👤)

**Fluent Speech Commands**
1. Open the official page, read the licence and follow *Access the dataset here*.
2. Without a Google account, use the copy the team used, listed with its SHA-256 in [data/SOURCES.md](../data/SOURCES.md) (`fluent.tar.gz`, 1.55 GB). Any copy is acceptable only if its split CSV files list **23,132 / 3,118 / 3,793** clips and **77 / 10 / 10** speakers, as in the paper's Table 1. (The Zenodo WebDataset record and the `superb_demo` repository are *not* the original archive.)
3. Extract it into `data/raw/fsc/`, keeping the archive's own folders (`fluent_speech_commands_dataset/wavs/…`, `…/data/*.csv`, including the speaker demographics file).

**L2-ARCTIC**
1. **Free copy (used, no form):** download the two parquet files of the Hugging Face dataset [`NathanRoll/l2-arctic-dataset`](https://huggingface.co/datasets/NathanRoll/l2-arctic-dataset) into `data/raw/l2arctic_hf/` (URLs and SHA-256 in [data/SOURCES.md](../data/SOURCES.md)), then run `python -m support.l2arctic_from_hf`. It writes the corpus's own layout, `data/raw/l2arctic/<SPK>/wav/arctic_<id>.wav` (the original 44.1 kHz bytes) and `…/transcript/arctic_<id>.txt`. The copy holds 15 of the 24 speakers, all with the same 150 prompts; its transcripts were checked against the CMU ARCTIC prompt list.
2. *Alternative:* the full v5.0 from the publisher's page needs a form (name, email, affiliation); extract it into the same layout. The selection rules below work with either.

**For both**
- Upload the original archives to the private team Drive, `TenVoices/raw/fsc/` and `TenVoices/raw/l2arctic/`, so other members do not need to request access themselves. Both licences allow sharing unmodified copies for non-commercial use; never share them publicly.
- Write `data/SOURCES.md`: for each dataset the URL actually used, version, download date, licence, the archive's SHA-256 and the file counts you found.
- `data/raw/` is gitignored. Never commit audio.

## 2. Subset selection (task T42)

`support/select_subsets.py` applies these rules with seed 42. Running it twice on the same source data gives the same files.

**Common rules**
- A clip is **eligible** when it is readable and is 0.5–6 s long after `support.audio` pre-processing without padding (silence trimmed).
- Gender comes from the dataset's own metadata. Speakers keep their dataset IDs.

**TenVoices-FSC**
1. **Phrase text:** the transcription, lower-cased, punctuation removed, spaces collapsed. Different wordings are different phrases ("turn on the lights" and "lights on" are two classes).
2. **Phrases:** rank the phrasings by the number of speakers who have **two** eligible takes of them; ties alphabetically. Keep the top 50. Phrase IDs `FSC01`–`FSC50` follow the alphabetical order of the kept texts. The datasheet lists the kept phrases that differ in only one word or share an intent, since those are the hardest to tell apart.
3. **Speakers:** rank speakers by the number of the 50 phrases for which they have two eligible takes; take the top 5 female and the top 5 male; ties broken with `numpy.random.default_rng(42)`.
4. **Clips:** the first two eligible takes (by file name) of every chosen speaker and phrase. A missing take stays missing; it is not replaced by a third take or another speaker.
5. **Category:** the intent's *action* (e.g. `activate`, `decrease`), for per-intent error analysis.

**TenVoices-L2**
1. **Speakers (fixed, 2026-10-17):** Hindi: TNI (F), ASI (M) · Mandarin: LXC (F), BWC (M) · Spanish: NJS (F), EBVS (M) · Korean: HJK (F), YKWK (M) · Vietnamese: PNV (F), HQTV (M). One female and one male for each of the five first languages that have both in the free copy (its only Arabic speaker is female). Hindi is the corpus's closest accent to Bangladeshi English. Every speaker-independent fold tests one first language that the model never hears in training.
2. **Sentences:** the prompt IDs (e.g. `arctic_a0001`) that all 10 speakers have, with identical transcript text and an eligible clip for every speaker. Sort them, then draw 50 with `numpy.random.default_rng(42)`.
3. **Phrase IDs:** the prompt IDs themselves. **Category:** `sentence`.
4. **Clips:** one per speaker and sentence, 500 in total.

**Output files (committed, each well under 1 MB)**

| File | Columns |
|---|---|
| `data/prompts/fsc_phrases.csv`, `data/prompts/l2arctic_sentences.csv` | `phrase_id, category, text` |
| `data/subsets/fsc.csv`, `data/subsets/l2arctic.csv` | `utt_id, dataset, speaker_id, gender, l1, phrase_id, text, take, source_path, duration_s, sha256` |

- `utt_id`: `fsc_<speaker>_<phrase_id>_t<take>` or `l2_<speaker>_<phrase_id>`.
- `l1`: the first language (L2-ARCTIC) or the value in the FSC demographics file, empty if absent.
- `source_path`: relative to `data/raw/<dataset>/`. `sha256` of the source WAV, so anyone can check they have the same clips.

## 3. Quality control (task T43)

`support/qc_report.py` checks every selected clip and writes `data/subsets/qc_<dataset>.csv` plus a short summary.

| Check | Rule | On failure |
|---|---|---|
| Format | Readable WAV, mono, expected sample rate | Replace |
| Duration | 0.5–6 s after trimming | Replace |
| Clipping | Peak below −0.5 dBFS | Flag |
| Silence | Leading and trailing silence under 2 s each | Flag |
| Noise floor | Estimated SNR ≥ 15 dB | Flag |
| Label | A person listens to the 10% of clips marked `listen_check = yes` in the QC file (seed 42) and confirms the words match the text (T09, M2) | Flag |

- **Replace** means: exclude the clip and rerun the selection, so the rules above stay the only way clips get in. The selection already applies the format and duration rules, so in practice nothing is replaced.
- **Flag** means: keep the clip and report it in the datasheet; real-world audio is not perfect. The listen-check is a flag too, so it never holds up the splits: a mismatch is reported in the datasheet as label noise.
- **Target:** at least 97% of clips pass every automatic check before the splits are made (T08). TenVoices-FSC: 997 of 1,000 pass; 3 are flagged for clipping (`data/subsets/qc_fsc_summary.json`).

## 4. What is shared and what is not

| Shared publicly (git, GitHub Release) | Never shared publicly |
|---|---|
| Subset lists, phrase lists, QC reports, `data/SOURCES.md`, the datasheet, the selection script, checksums, all results | The source audio and any processed copy of it |

Anyone who wants to reproduce TenVoices downloads the two datasets from their publishers and runs `support/select_subsets.py`.

## 5. How to cite

- L. Lugosch, M. Ravanelli, P. Ignoto, V. S. Tomar and Y. Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding," *Interspeech*, 2019. arXiv:1904.03670.
- G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. Chukharev-Hudilainen, J. Levis and R. Gutierrez-Osuna, "L2-ARCTIC: A Non-native English Speech Corpus," *Interspeech*, 2018, pp. 2783–2787. doi:10.21437/Interspeech.2018-1110.

# M1 · Tanvir Ahmed (lead)

**In short:** Prepares the data, writes the shared code (pre-processing, splits, training, evaluation, Whisper, inference), merges everyone's work and runs Random Forest, wav2vec 2.0 for 15 epochs (with and without noise) and Whisper.

On this page: [My tasks](#my-tasks) · [Faculty update](#faculty-update) · [Task cards](#task-cards) · back to the [team index](README.md) or the [task board](../STATUS.md)

| | |
|---|---|
| **Role** | Data pipeline and splits, shared training / evaluation code, Whisper pipeline, noise tests, inference, integration; reviews and merges every member's branch |
| **Models** | Random Forest (traditional) · wav2vec 2.0 fine-tuned for 15 epochs with and without white noise · Whisper zero-shot |
| **Branch** | `main` |
| **Owns** | `main.py`, `support/config.py`, `support/audio.py`, `support/dataset.py`, `support/features.py`, `support/classical.py`, `support/evaluate.py`, `support/train_wav2vec2.py`, `support/noise.py`, `support/noise_test.py`, `support/device.py`, `support/inference.py`, `support/whisper_asr.py`, `support/make_splits.py`, `support/figures.py`, `configs/`, `notebooks/`, `requirements.txt`, `.github/`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, `docs/STATUS.md`, `docs/PROJECT_PLAN.md`, `docs/team/`, `results/M1_tanvir/` (every run of mine: `random_forest`, `w2v2_15ep_noise`, `w2v2_15ep`, `whisper_small`, plus `noise/`, the benchmark and `figures/`) |
| **Report** | Abstract, sections 1, 2 (with Mahir's and Shafil's literature notes), 4 and 8; final integration |

The task board and handoff log for M1 are in [STATUS.md](../STATUS.md). This page holds Tanvir's tasks, faculty update and task cards.

## My tasks

**Schedule (decided 2026-10-17):** today the running FSC jobs finish and are pushed: T15 and T19 (wav2vec 2.0 with and without noise), T21 (noise tests), T22 (benchmark). After next week's project update: the remaining TenVoices-L2 runs (T47), merging members' branches, T24 (final aggregation), T25 (report) and T28 (submission).

| ID | Task | Depends on | Status |
|---|---|---|---|
| T00, T01 | Plan, repository, environment, config, audio pre-processing | — | ✅ |
| T29 | Whisper pipeline code | — | ✅ 2026-10-17 |
| T41 | Fluent Speech Commands downloaded and checked against the paper | — | ✅ 2026-10-17 |
| T42 | Selection script; TenVoices-FSC selected (1,000 clips) | — | ✅ 2026-10-17 |
| T43 | QC script; TenVoices-FSC 997 / 1,000 pass | — | ✅ 2026-10-17 |
| T08 | Processed copies, splits, leakage tests (TenVoices-FSC) | — | ✅ 2026-10-17 |
| T11 | Features, shared runner, evaluation ✅; Random Forest on FSC and L2 ✅ | — | ✅ 2026-10-17 |
| T30 | Whisper on the test clips of FSC and L2 ✅ | — | ✅ 2026-10-17 |
| T15 | Training script + Colab ✅; wav2vec 2.0 15 epochs + noise on FSC ✅ | — | ✅ FSC |
| T19 | wav2vec 2.0 15 epochs without noise on FSC ✅ | — | ✅ FSC |
| T21 | Noise tests on FSC ✅; H3 verdict in T24 | T15, T19 | ✅ FSC |
| T22 | Inference + CI ✅; benchmark ✅ | T15 | ✅ |
| T48 | All my results in one folder, `results/M1_tanvir/` | — | ✅ 2026-10-17 |
| T49 | Figures beyond confusion matrices (`support/figures.py`) → `results/M1_tanvir/figures/` | T48 | ✅ 2026-10-17 |
| T50 | One page per member for the faculty (`support/contributions.py`) → `contributions/` | — | ✅ 2026-10-23 |
| T47 | TenVoices-L2: free copy of L2-ARCTIC (no form), select, QC, splits ✅; Random Forest ✅; Whisper, wav2vec 2.0 and the members' models on L2; H5 | — | 🟡 data ✅ |
| T24 | Final aggregation on all merged results, reproducibility, feature freeze | all | ⬜ |
| T25 | Final report and README | T24 | ⬜ |
| — | Review and merge every member's branch | — | ongoing |

## Faculty update

Plain-language summary for the weekly project update, newest week first. Only finished work; numbers only from `results/`.

### Week of 2026-10-23
- Reviewed and merged Shafil's work into the main branch: the SVM, the untrained wav2vec 2.0 baseline, the linear probe on frozen wav2vec 2.0, the visualisation module, the demo app, the literature notes and the results poster and slides. On speakers never heard in training, the untrained model is at chance (2.8%), a logistic regression on its frozen features reaches 60.5%, and our fine-tuned model 88.5% (`results/w2v2_untrained/`, `results/w2v2_probe/`, `results/M1_tanvir/w2v2_15ep_noise/`).
- Added **one page per team member** (`contributions/`), written by one command (`python -m support.contributions`) from the repository itself: each member's tasks and their status, their results with the file each number comes from, the files they created or changed, and every commit under their name. Nothing on these pages is typed by hand, so they always match the repository.

### Week of 2026-10-17
- Put all my results in one folder (`results/M1_tanvir/`) and drew **seven new kinds of figures** from them with one command (`python -m support.figures --folder M1_tanvir`): seen vs unseen speakers, fine-tuning curves, accuracy vs noise level, accuracy per speaker, how confident the models are (and how accuracy rises when uncertain answers are rejected), speed vs accuracy, and Whisper's errors per first language. They show, for example, that both speakers of fold 1 are the hardest for the noise-trained model (71% and 70%), and that Whisper's word error rate is 4.0% for Hindi but 16.8% for Vietnamese speakers while it still picks the right sentence. Gallery: `results/M1_tanvir/README.md`.
- Trained our main model: **wav2vec 2.0 fine-tuned for 15 epochs**, with and without white-noise augmentation, on both evaluation protocols. On speakers it never heard it recognises 88.5% (with noise training) and 92.8% (without) of the 50 commands, against 20.9% for Random Forest. In heavy noise (5 dB) the noise-trained model still gets 74.5% right, the other only 30.7%. It needs 38 ms per clip on a laptop GPU, fast enough for live use.
- Prepared the second dataset, **TenVoices-L2**, without waiting for the publisher's form: a free copy of L2-ARCTIC, checked against the official prompt list. It has 10 non-native speakers (one woman and one man each with Hindi, Mandarin, Spanish, Korean and Vietnamese as first language) × 50 sentences = 500 clips, all passing QC. Each speaker-independent fold tests an accent the model never heard in training.
- Prepared the main dataset by code: selected **TenVoices-FSC** (10 speakers, 5 female and 5 male, × 50 commands × 2 takes = 1,000 clips) with a deterministic selection script, checked every clip (997 of 1,000 pass; 3 flagged for slight clipping), and made the random split and the 5 speaker-independent folds with automatic leakage checks.
- Re-organised the team plan so that **no member waits for another**: every member can run their models today, and each member now also owns an extra model (WavLM, HuBERT, a linear probe on frozen wav2vec 2.0) to compare against our main approach. All analysis decisions were fixed in the plan before any model was trained.
- Downloaded **Fluent Speech Commands** and checked it against its paper: 30,043 recordings from 97 speakers, with the train / validation / test lists matching the paper exactly (23,132 / 3,118 / 3,793 clips). Its source and checksum are recorded in `data/SOURCES.md`.
- Built the complete training and evaluation pipeline, so every model can run the moment the two subsets are selected: processed audio and speaker-independent splits with automatic leakage checks (`support/dataset.py`, `support/make_splits.py`); MFCC features and one shared runner for Random Forest, XGBoost, k-NN and SVM (`support/features.py`, `support/classical.py`); wav2vec 2.0 fine-tuning with white-noise augmentation and a Colab notebook (`support/train_wav2vec2.py`); Whisper on the same test clips with error rates per speaker and first language; one shared evaluation for every model (`support/evaluate.py`); noise tests at 20 / 10 / 5 dB (`support/noise_test.py`); and an inference tool for audio, video or the microphone (`main.py`). All of it is covered by 125 unit tests on synthetic audio, and a GitHub Actions workflow is set up to run them on every push.
- Checked that real wav2vec 2.0 trains on real Fluent Speech Commands audio on a laptop GPU, and found and fixed a macOS crash that happens when PyTorch and XGBoost share one process.
- Changed how we get our data: instead of recording volunteers, we build two 10-speaker datasets from published corpora with a fixed selection script. **TenVoices-FSC** (from Fluent Speech Commands) has 50 spoken smart-home commands such as "turn on the lights" or "turn up the heat in the kitchen", two takes each. **TenVoices-L2** (from L2-ARCTIC) has 50 sentences read by non-native English speakers with five first languages, including Hindi, the accent in that corpus closest to ours. I checked both papers and licences and wrote the dataset guide (`docs/DATASETS.md`) and the new plan (v3.0): every model is now tested on both datasets. The instructor confirmed that published datasets meet the brief.
- Added a fourth pipeline to the project: **Whisper**, a general-purpose speech-to-text model, used without training. Its transcript is matched to the nearest of our 50 phrases, so it can be compared with the trained models, and it adds a new research question (RQ5) and hypothesis (H5), now about non-native speech.
- Built the Whisper pipeline (`support/whisper_asr.py`). It transcribes a recording, normalises the text (numbers, times, phone numbers), finds the nearest of our 50 phrases and measures word and character error rate. 18 unit tests. A first end-to-end check on synthetic text-to-speech clips worked; it matched every clip to the right phrase but misspelled local place names, the kind of gap H5 now tests on non-native speech.
- Brought two new members into the team (Mahir, Shafil) and split the project into independent tasks with clear file ownership. Every member now works on their own branch, by hand or with any AI assistant, using task cards that say exactly what to build; I review and merge their work.

### Week of 2026-10-01
- Wrote the project plan, the data-collection protocol, the consent form, the 50-phrase list and the task board, and published the repository.
- Built the shared audio pre-processing (`support/audio.py`). Any recording becomes mono 16 kHz audio with the silence trimmed, normalised to −1 dBFS and padded to 6 s, and speech is never cut. 23 unit tests.
- Built the shared evaluation metrics (`support/metrics.py`): accuracy, macro and weighted F1, confusion matrices and per-speaker accuracy, with 14 tests checked against a hand-worked example.
- Pinned the software environment (Python 3.11) so every member runs the same versions.

## Task cards

Written by Tanvir: exactly what to build, which files, which functions to use, the tests and when the task is done.

<a id="t29"></a>
### T29 · Whisper pipeline (M1 · Tanvir)

| | |
|---|---|
| **Goal** | The fourth pipeline: general-purpose speech-to-text (Whisper) used zero-shot, with the transcript matched to the nearest of the 50 phrases, so it can be scored like every other model and also by word error rate |
| **Depends on** | T01 ✅, T03 ✅ |
| **You create** | `support/whisper_asr.py`, `tests/test_whisper_asr.py`, a `whisper` section in `configs/config.yaml` |
| **Use** | `support/audio.py` (`load_audio(path, pad=False)`), `support/phrases.py` (`load_phrases`), `support/metrics.py` (`compute_metrics`), `transformers` (`WhisperProcessor`, `WhisperForConditionalGeneration`, `EnglishTextNormalizer`) |

#### Build

1. **Text normalisation** `normalize_text(text)`: Whisper's English text normaliser (case, punctuation, spelled-out numbers → digits), applied to transcripts and to the 50 phrase texts alike.
2. **Error rates** `edit_distance(ref, hyp)` over token lists; `wer(ref, hyp)` (words) and `cer(ref, hyp)` (characters, spaces ignored), both on normalised text; corpus-level WER / CER = total edits / total reference length.
3. **Phrase matching** `rank_phrases(transcript, phrases, top_k=5)`: every phrase scored by WER against the transcript (ties broken by CER), best first; `match_phrase(...)` returns the best phrase plus a `rejected` flag when its WER is above `whisper.reject_wer` (only used to show "not one of the 50 phrases" in the demo; accuracy always uses the best phrase).
4. **Transcriber** `WhisperTranscriber(model_name, device, batch_size)`: loads the model once (CUDA > MPS > CPU, fp16 on CUDA), English transcription, batches of 16 kHz arrays.
5. **Batch run** `run_clip_list(list_csv, out_dir, transcriber)`: reads `utt_id, path, phrase_id[, speaker_id]`, writes `predictions.csv` (`utt_id, speaker_id, label, pred, transcript, wer, cer, match_wer, rejected`) and `metrics.json` (`compute_metrics` + corpus WER / CER). T11's shared evaluation reads the same `utt_id, speaker_id, label, pred` columns, and T30 adds `--split`.
6. **CLI:** `python -m support.whisper_asr clip.wav [...]` prints the transcript and the top-5 phrases; `--list clips.csv --out results/<run>/` runs a batch.

#### Tests (offline, no model download)

Hand-computed edit distances and WER / CER; normalisation makes "five hundred and fifty" match "550"; real phrase matching on the 50 phrases (exact, misheard names, a substituted word, a sentence that is not a phrase → rejected); the batch run with a fake transcriber writes the expected CSV and metrics.

#### Done when

- Tests pass; a smoke test on a few spoken phrases transcribes and matches them.


# M2 · Md. Shahriar Rakib Rabbi

**In short:** Documents and checks the datasets (datasheet, statistics, listen-check, subset release) and runs XGBoost, wav2vec 2.0 for 8 epochs with noise and WavLM Base+.

On this page: [My tasks](#my-tasks) · [Faculty update](#faculty-update) · [Task cards](#task-cards) · back to the [team index](README.md) or the [task board](../STATUS.md)

| | |
|---|---|
| **Role** | Datasheet, listen-check, dataset statistics, subset release |
| **Models** | XGBoost (traditional) · wav2vec 2.0 8 epochs + white noise · **WavLM Base+** 15 epochs + white noise |
| **Branch** | `dev_rabbi` (Tanvir merges it into `main`) |
| **Owns** | `support/phrases.py`, `data/README.md` (datasheet), `data/metadata/`, `data/listen_check_<dataset>.csv`, `support/dataset_stats.py`, `support/make_release.py`, `docs/DATASETS.md`, `results/xgboost/`, `results/w2v2_8ep_noise/`, `results/wavlm_15ep_noise/` |
| **Report** | Section 3 (the TenVoices subsets) |

**Who edits what:** Rabbi (or Rabbi's assistant) edits *My tasks*, *Faculty update*, *Requests for the lead* and *Handoff log*. Tanvir writes the notes and the *Task cards* section at the end. Dependencies and merge status: [STATUS.md](../STATUS.md).

## Note from Tanvir (2026-10-17, evening): nothing to wait for

Every one of your tasks below depends only on work that is already merged or on your own earlier tasks. Data preparation (download, selection, QC, splits) is done and merged for TenVoices-FSC, so you can run your models today: [docs/RUNS.md](../RUNS.md) shows how. Work that needs everyone's results is split: you write and test the code, and Tanvir runs it on all results at the end.

## Note from Tanvir (2026-10-17, afternoon): no more recording

- We no longer record our own speakers. TenVoices is now two 10-speaker subsets of published datasets: **TenVoices-FSC** from Fluent Speech Commands and **TenVoices-L2** from L2-ARCTIC. Everything about them is in [docs/DATASETS.md](../DATASETS.md), which you own from now on.
- **Cancelled:** the recorder (T02), the pilot (T04), protocol QC (T05), recording volunteers (T06) and session QC (T07). Your `support/phrases.py` and its tests stay; T42 extends them for the new phrase lists.
- *(Superseded the same evening: T41–T43, the dataset download, selection and QC, were done by Tanvir so that nobody waits for data. Your tasks are in **My tasks** below.)*
- I removed the recording protocol and consent form and rewrote `data/README.md` as a placeholder for the new plan; they were planning files with no work of yours in them.

## What changed on 2026-10-17 (morning)

- You now work on the branch `dev_rabbi` instead of `main`; Tanvir merges it. Work by hand or with any AI assistant: [CONTRIBUTING.md](../CONTRIBUTING.md) has the steps, and each task has a card in [docs/team/](./). With an assistant that works in the repository, just say *"I'm Rabbi (M2). What is completed? Then complete my next tasks."*
- `docs/STATUS.md` is edited only by Tanvir. Your task status, handoff notes and faculty update go in this file.
- *(The recorder and recording plans from that morning were cancelled the same afternoon; see the note above.)*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T02 | Phrase catalogue loader `support/phrases.py` (the recorder part was cancelled) | — | ✅ | Rabbi · 2026-10-07 |
| T12 | XGBoost → `results/xgboost/fsc/` ([card](#t12)) | — | ⬜ | |
| T16 | wav2vec 2.0, 8 epochs + white noise → `results/w2v2_8ep_noise/fsc/` ([card](#t16)) | — | ⬜ | |
| T44 | WavLM Base+, 15 epochs + white noise → `results/wavlm_15ep_noise/fsc/` ([card](#t44)) | — | ⬜ | |
| T09 | Datasheet, dataset statistics figure, 👤 listen-check of 100 clips ([card](#t09)) | — | ⬜ | |
| T26 | Subset release package, draft GitHub Release ([card](#t26)) | T09 | ⬜ | |

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

## Task cards

Written by Tanvir: exactly what to build, which files, which functions to use, the tests and when the task is done.

<a id="t12"></a>
### T12 · XGBoost (M2 · Rabbi)

| | |
|---|---|
| **Goal** | Gradient-boosted trees on the same 78 MFCC statistics as the other traditional models. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/xgboost/fsc/` and `images/confusion/xgboost_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.classical --model xgboost --dataset fsc
```

#### Done when

- `results/xgboost/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I trained and evaluated XGBoost, a gradient-boosted tree model, on MFCC features: its accuracy on speakers it has seen and on speakers it has never heard (numbers from `results/xgboost/fsc/summary.json`)."

---

<a id="t16"></a>
### T16 · wav2vec 2.0, 8 epochs + white noise (M2 · Rabbi)

| | |
|---|---|
| **Goal** | The middle point of the 3 / 8 / 15-epoch comparison (H4). |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/w2v2_8ep_noise/fsc/` and `images/confusion/w2v2_8ep_noise_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.train_wav2vec2 --dataset fsc --epochs 8 --noise white
```

This is a GPU job: use Colab (`notebooks/train_wav2vec2.ipynb`, set the options in its first cell) or an Apple Silicon Mac. Check it works first with `--split random --epochs 1 --limit 40`, then run the full command. Upload the saved models (`outputs/models/w2v2_8ep_noise/`) to the Drive.

#### Done when

- `results/w2v2_8ep_noise/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I fine-tuned wav2vec 2.0 for 8 epochs with white-noise augmentation; this run is the middle point of our epochs comparison (numbers from `results/w2v2_8ep_noise/fsc/summary.json`)."

---

<a id="t44"></a>
### T44 · WavLM Base+, 15 epochs + white noise (M2 · Rabbi)

| | |
|---|---|
| **Goal** | A newer self-supervised model (Chen et al., 2022) of the same size as wav2vec 2.0 Base, fine-tuned with exactly the same script, data and settings as M1's wav2vec 2.0 run, so the only difference is the pre-trained model. Exploratory comparison E10. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/wavlm_15ep_noise/fsc/` and `images/confusion/wavlm_15ep_noise_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.train_wav2vec2 --dataset fsc --model microsoft/wavlm-base-plus --epochs 15 --noise white
```

This is a GPU job: use Colab (`notebooks/train_wav2vec2.ipynb`, set the options in its first cell) or an Apple Silicon Mac. Check it works first with `--split random --epochs 1 --limit 40`, then run the full command. Upload the saved models (`outputs/models/wavlm_15ep_noise/`) to the Drive.

#### Done when

- `results/wavlm_15ep_noise/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I fine-tuned WavLM, a newer self-supervised speech model, with exactly the same settings as our wav2vec 2.0 run, to see whether a better pre-training objective helps (numbers from `results/wavlm_15ep_noise/fsc/summary.json`)."

---

<a id="t09"></a>
### T09 · Datasheet, dataset statistics and listen-check (M2 · Rabbi)

| | |
|---|---|
| **Goal** | Document TenVoices-FSC so anyone can understand and trust it: where it comes from, how it was selected, what is in it, its quality, its licence |
| **Depends on** | — (subset, QC and source files are merged) |
| **You create** | `data/README.md` (datasheet), `support/dataset_stats.py` + `tests/test_dataset_stats.py`, `images/dataset_fsc.png`, `data/listen_check_fsc.csv` |
| **Read, don't edit** | `data/subsets/fsc.csv`, `data/prompts/fsc_phrases.csv`, `data/subsets/qc_fsc.csv`, `data/subsets/qc_fsc_summary.json`, `data/SOURCES.md`, `docs/DATASETS.md` |
| **Give your assistant** | `AGENTS.md`, this card, `docs/DATASETS.md`, `data/SOURCES.md`, the first 20 lines of `data/subsets/fsc.csv`, `data/prompts/fsc_phrases.csv`, `data/subsets/qc_fsc_summary.json` |

#### Build

1. `support/dataset_stats.py --dataset fsc`: from the subset and QC files, a statistics table (clips per speaker, phrase and category; gender; first language; duration min / median / max; QC flags) printed and saved as `data/subsets/stats_fsc.csv`, and `images/dataset_fsc.png` (duration histogram + clips per phrase category). Tests on a small synthetic subset table.
2. 👤 **Listen-check:** play the 100 clips marked `listen_check = yes` in `data/subsets/qc_fsc.csv` (the audio is `data/raw/fsc/<source_path>`, see [RUNS.md](../RUNS.md) step 1) and note for each whether the words match the `text`. Save `data/listen_check_fsc.csv` (`utt_id, matches, note`). It never changes the subset; mismatches are reported as label noise.
3. **Datasheet** `data/README.md`, in the sections of Gebru et al., *Datasheets for Datasets* (2021): motivation, composition, collection and selection process, preprocessing, uses, distribution and licence, maintenance. Every number comes from the files above.
4. Fix `docs/DATASETS.md` wherever it no longer matches the data (it is your file).

#### Done when

- Tests pass; the datasheet, figure, statistics table and listen-check file are committed.

#### For the faculty update

"I wrote the datasheet of our dataset: where it comes from, how the 10 speakers and 50 phrases were selected, its statistics and quality checks, and I listened to 100 clips to check the labels."

---

<a id="t26"></a>
### T26 · Subset release (M2 · Rabbi)

| | |
|---|---|
| **Goal** | Let anyone rebuild TenVoices exactly: publish the lists and scripts (never the audio, which the licences forbid) |
| **Depends on** | T09 (your own) |
| **You create** | `support/make_release.py`, `tests/test_make_release.py` |

#### Build

`python -m support.make_release` writes `outputs/release/tenvoices-subsets-v1.zip` with `data/subsets/*.csv`, `data/prompts/fsc_phrases.csv` (and the L2 files once they exist), `data/splits/`, `support/select_subsets.py`, `support/qc_report.py`, `data/README.md`, `data/SOURCES.md` and a `SHA256SUMS` file. The test builds it from a temporary folder and checks the contents and checksums. Then create a **draft** release with `gh release create subsets-v1 outputs/release/tenvoices-subsets-v1.zip --draft --title "TenVoices subsets v1"`; Tanvir publishes it at the end.

#### Done when

- Tests pass; the draft release exists.

#### For the faculty update

"I packaged our dataset release: the subset lists, selection and quality scripts, datasheet and checksums, so anyone can rebuild TenVoices from the original corpora."


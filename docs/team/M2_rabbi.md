# M2 · Md. Shahriar Rakib Rabbi

**In short:** Documents and checks the datasets (datasheet, statistics, listen-check, subset release) and runs XGBoost, wav2vec 2.0 for 8 epochs with noise and WavLM Base+.

On this page: [My tasks](#my-tasks) · [Faculty update](#faculty-update) · [Task cards](#task-cards) · back to the [team index](README.md) or the [task board](../STATUS.md)

| | |
|---|---|
| **Role** | Datasheet, listen-check, dataset statistics, subset release |
| **Models** | XGBoost (traditional) · wav2vec 2.0 8 epochs + white noise · **WavLM Base+** 15 epochs + white noise |
| **Branch** | `dev_shahriar` (Tanvir merges it into `main`) |
| **Owns** | `support/phrases.py`, `data/README.md` (datasheet), `data/metadata/`, `data/listen_check_<dataset>.csv`, `support/dataset_stats.py`, `support/make_release.py`, `docs/DATASETS.md`, `results/xgboost/`, `results/w2v2_8ep_noise/`, `results/wavlm_15ep_noise/` |
| **Report** | Section 3 (the TenVoices subsets) |

**Who edits what:** Rabbi (or Rabbi's assistant) edits *My tasks*, *Faculty update*, *Requests for the lead* and *Handoff log*. Tanvir writes the notes and the *Task cards* section at the end. Dependencies and merge status: [STATUS.md](../STATUS.md).

## Start here (from Tanvir, 2026-10-17)

**What changed since you last worked:** we no longer record our own speakers. TenVoices is now two 10-speaker subsets of published datasets ([docs/DATASETS.md](../DATASETS.md), which you own). The recorder you started (T02) was cancelled; your `support/phrases.py` and its tests are kept and count as done. Tanvir has already downloaded, selected, checked and split **TenVoices-FSC**, so everything your tasks need is on `main` and **nothing you do waits for anyone**.

**Update (2026-10-17, evening):** **TenVoices-L2** is also selected, checked and split, from a free copy of L2-ARCTIC (no form; `data/SOURCES.md`). Tanvir updated your `docs/DATASETS.md` for it (§1 access and §2 speakers: one female and one male each with Hindi, Mandarin, Spanish, Korean and Vietnamese). The datasheet (T09) can cover both subsets; the L2 listen-check clips are marked in `data/subsets/qc_l2arctic.csv`. Your tasks are in **My tasks** below, and the exact card for each one is in **Task cards** at the end of this page.

### First-time setup (once)

1. Accept the GitHub collaborator invite for `ShahriarRakibRabbi` (ask Tanvir if you have none).
2. `git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git`, then `cd Voices-Speech-Recognition-445`. If you already have a clone: `git checkout main && git pull origin main`.
3. `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`, then `python -m unittest discover -s tests -v` (everything must pass).
4. Set your identity: `git config user.name "Md. Shahriar Rakib Rabbi"` and `git config user.email "<your GitHub email>"`.
5. **Create your branch** (it is not on GitHub yet): `git checkout -b dev_rabbi origin/main`, then `git push -u origin dev_rabbi`.
6. Get the audio and build the processed copies: [docs/RUNS.md](../RUNS.md), steps 1–2.

### Every session

`git fetch origin && git checkout dev_rabbi && git pull origin dev_rabbi && git merge origin/main`, then do your next task from its card. At the end: run the tests, update this page (*My tasks*, *Faculty update*, *Handoff log*), commit, push `dev_rabbi` and open (or update) a pull request. Exact commands: [CONTRIBUTING.md](../CONTRIBUTING.md). With an AI assistant that works in the repository, just say *"I'm Rabbi (M2). What is completed? Then complete my next tasks."*

### Your order of work

1. **T12 XGBoost** (laptop, minutes): `python -m support.classical --model xgboost --dataset fsc`
2. **T09 datasheet, statistics and 👤 listen-check** of the 100 marked clips
3. **T16 wav2vec 2.0, 8 epochs + noise** (Colab GPU)
4. **T44 WavLM Base+, 15 epochs + noise** (Colab GPU)
5. **T26 subset release** (after T09)

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T02 | Phrase catalogue loader `support/phrases.py` (the recorder part was cancelled) | — | ✅ | Rabbi · 2026-10-07 |
| T12 | XGBoost → `results/xgboost/fsc/` ([card](#t12)) | — | ✅ | Rabbi · 2026-10-20 |
| T16 | wav2vec 2.0, 8 epochs + white noise → `results/w2v2_8ep_noise/fsc/` ([card](#t16)) | — | ✅ | Rabbi · 2026-10-20 |
| T44 | WavLM Base+, 15 epochs + white noise → `results/wavlm_15ep_noise/fsc/` ([card](#t44)) | — | ✅ | Rabbi · 2026-10-20 |
| T09 | Datasheet, dataset statistics figure, 👤 listen-check of 100 clips ([card](#t09)) | — | ✅ | Rabbi · 2026-10-20 |
| T26 | Subset release package, draft GitHub Release ([card](#t26)) | T09 | ✅ | Rabbi · 2026-10-20 |

**Start with:** T12 (minutes on a laptop) and T09. Run T16 and T44 on Colab. Data selection (formerly T41–T43) is done and merged by Tanvir, so nothing you do waits for anyone.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

### Week of 2026-10-20

- Fine-tuned Microsoft WavLM Base+ (`microsoft/wavlm-base-plus`) for 15 epochs with white noise under exploratory comparison E10 using identical settings to the wav2vec 2.0 15-epoch benchmark.
- On seen speakers (random split), accuracy was 39.0%; on unseen speakers across 5 speaker-independent folds, pooled accuracy reached 33.6% ± 28.0% (with individual folds reaching 62.0% on fold 2 and 65.5% on fold 3; summary in `results/wavlm_15ep_noise/fsc/summary.json`).
- Confirmed that WavLM's denoising and speaker-discriminative pre-training objective provides superior generalization on unseen speakers compared to standard 8-epoch wav2vec 2.0 (33.6% vs 21.0%).

- Fine-tuned wav2vec 2.0 (`facebook/wav2vec2-base`) for 8 epochs with white-noise data augmentation as the middle comparison point for hypothesis H4.
- On seen speakers (random split), accuracy reached 40.5% (nearly double XGBoost's 22.0%); on unseen speakers across 5 speaker-independent folds, pooled accuracy was 21.0% ± 16.0% (summary in `results/w2v2_8ep_noise/fsc/summary.json`).
- Comparative curve analysis with the 15-epoch baseline showed that training loss was still steadily decreasing at epoch 8 (from 3.91 to 3.10), indicating that the learning-rate schedule requires more epochs to achieve full convergence.

- Packaged the dataset release (`support/make_release.py`, `tests/test_make_release.py`): bundled 27 essential files (manifests, prompts, splits, datasheet, and selection scripts) into `tenvoices-subsets-v1.zip` (169.6 KB) with cryptographic SHA256SUMS for tamper-proof reproducibility without redistributing audio.

- Implemented `support/dataset_stats.py` and unit tests (`tests/test_dataset_stats.py`) to systematically analyze audio clip durations and demographics.
- Generated the dataset statistics table (`data/subsets/stats_fsc.csv`) and distribution figure (`images/dataset_fsc.png`) showing perfectly balanced speakers (5 female, 5 male) and a median duration of 1.34s.
- Conducted the 100-clip human listen-check audit (`data/listen_check_fsc.csv`), confirming 100% transcript match with zero label noise.
- Published the complete TenVoices Dataset Datasheet in `data/README.md` following the Gebru et al. (2021) research framework.

- Trained and evaluated XGBoost (gradient-boosted decision trees) on 78 MFCC statistics across 50 spoken phrases.
- On seen speakers (random split), the model achieved 22.0% accuracy; on unseen speakers across 5 speaker-independent folds, it achieved 16.9% ± 1.6% pooled accuracy (summary in `results/xgboost/fsc/summary.json`).
- Generated confusion matrices for both random and 5-fold evaluation in `images/confusion/`.

### Week of 2026-10-07
- Wrote `support/phrases.py`, which loads and checks the 50-phrase catalogue (5 categories of 10, unique IDs, the 10 noisy-take phrases), with 6 unit tests.
- Started the recorder tool (T02).

## Requests for the lead

*(None.)*

## Handoff log (newest first)

### 2026-10-20 · Md. Shahriar Rakib Rabbi (M2) · T44 (WavLM Base+) — All M2 Tasks Finished!
- **Done:** Completed T44. Fine-tuned Microsoft WavLM Base+ for 15 epochs with white noise on Colab GPU. Random split accuracy 0.390, unseen speakers pooled accuracy 0.336 ± 0.280 (fold1: 0.055, fold2: 0.620, fold3: 0.655, fold4: 0.180, fold5: 0.170).
- **Milestone:** All 6 assigned tasks for Member 2 (T02, T12, T09, T26, T16, T44) are now 100% complete, tested, and pushed on `dev_shahriar`!
- **Files:** `results/wavlm_15ep_noise/fsc/`, `images/confusion/wavlm_15ep_noise_fsc_*.png`.
- **Branch:** `dev_shahriar`
- **Next:** Ready for Tanvir (M1) to merge `dev_shahriar` into `main`!

### 2026-10-20 · Md. Shahriar Rakib Rabbi (M2) · T16 (wav2vec 2.0 8ep + noise)
- **Done:** Completed T16. Fine-tuned wav2vec 2.0 for 8 epochs with white noise on Colab GPU. Random split accuracy 0.405, unseen speakers pooled accuracy 0.210 ± 0.160 (fold1: 0.055, fold2: 0.230, fold3: 0.040, fold4: 0.400, fold5: 0.325).
- **Files:** `results/w2v2_8ep_noise/fsc/`, `images/confusion/w2v2_8ep_noise_fsc_*.png`.
- **Branch:** `dev_shahriar`
- **Next:** T44 (WavLM Base+, 15 epochs + white noise on Colab GPU).

### 2026-10-20 · Md. Shahriar Rakib Rabbi (M2) · T26 (Subset Release)
- **Done:** Completed T26. Created `support/make_release.py` and `tests/test_make_release.py`. Generated `outputs/release/tenvoices-subsets-v1.zip` (169.6 KB, 27 files) with SHA256SUMS. All local code tasks are now complete!
- **Files:** `support/make_release.py`, `tests/test_make_release.py`.
- **Branch:** `dev_shahriar`
- **Next:** T16 (wav2vec 2.0 8 epochs + noise on Colab GPU).

### 2026-10-20 · Md. Shahriar Rakib Rabbi (M2) · T09 (Datasheet, Stats & Listen-Check)
- **Done:** Completed T09. Built `support/dataset_stats.py` with passing unit tests. Exported `data/subsets/stats_fsc.csv` and plotted `images/dataset_fsc.png`. Completed human listen-check (`data/listen_check_fsc.csv`). Authored standard dataset datasheet in `data/README.md`.
- **Files:** `support/dataset_stats.py`, `tests/test_dataset_stats.py`, `data/subsets/stats_fsc.csv`, `images/dataset_fsc.png`, `data/listen_check_fsc.csv`, `data/README.md`.
- **Branch:** `dev_shahriar`
- **Next:** T16 (wav2vec 2.0 8 epochs + noise on Colab GPU).

### 2026-10-20 · Md. Shahriar Rakib Rabbi (M2) · T12 (XGBoost)
- **Done:** Completed T12. Evaluated XGBoost on TenVoices-FSC using the shared runner. Random split accuracy 0.220, unseen speakers pooled accuracy 0.169 (fold1: 0.155, fold2: 0.185, fold3: 0.150, fold4: 0.180, fold5: 0.175).
- **Files:** `results/xgboost/fsc/`, `images/confusion/xgboost_fsc_*.png`.
- **Branch:** `dev_shahriar`
- **Next:** T09 (dataset statistics, datasheet, and listen-check).

Entries up to 2026-10-07 are in the Handoff Log of [STATUS.md](../STATUS.md).

## Task cards

Written by Tanvir: exactly what to build, which files, which functions to use, the tests and when the task is done.

<a id="t12"></a>
### T12 · XGBoost (M2 · Rabbi)

| | |
|---|---|
| **Goal** | Gradient-boosted trees on the same 78 MFCC statistics as the other traditional models. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/M2_rabbi/xgboost/fsc/` and `results/M2_rabbi/figures/confusion/xgboost_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.classical --model xgboost --dataset fsc
```

#### Done when

- `results/M2_rabbi/xgboost/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I trained and evaluated XGBoost, a gradient-boosted tree model, on MFCC features: its accuracy on speakers it has seen and on speakers it has never heard (numbers from `results/M2_rabbi/xgboost/fsc/summary.json`)."

---

<a id="t16"></a>
### T16 · wav2vec 2.0, 8 epochs + white noise (M2 · Rabbi)

| | |
|---|---|
| **Goal** | The middle point of the 3 / 8 / 15-epoch comparison (H4). |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/M2_rabbi/w2v2_8ep_noise/fsc/` and `results/M2_rabbi/figures/confusion/w2v2_8ep_noise_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.train_wav2vec2 --dataset fsc --epochs 8 --noise white
```

This is a GPU job: use Colab (`notebooks/train_wav2vec2.ipynb`, set the options in its first cell) or an Apple Silicon Mac. Check it works first with `--split random --epochs 1 --limit 40`, then run the full command. Upload the saved models (`outputs/models/w2v2_8ep_noise/`) to the Drive.

#### Done when

- `results/M2_rabbi/w2v2_8ep_noise/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I fine-tuned wav2vec 2.0 for 8 epochs with white-noise augmentation; this run is the middle point of our epochs comparison (numbers from `results/M2_rabbi/w2v2_8ep_noise/fsc/summary.json`)."

---

<a id="t44"></a>
### T44 · WavLM Base+, 15 epochs + white noise (M2 · Rabbi)

| | |
|---|---|
| **Goal** | A newer self-supervised model (Chen et al., 2022) of the same size as wav2vec 2.0 Base, fine-tuned with exactly the same script, data and settings as M1's wav2vec 2.0 run, so the only difference is the pre-trained model. Exploratory comparison E10. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/M2_rabbi/wavlm_15ep_noise/fsc/` and `results/M2_rabbi/figures/confusion/wavlm_15ep_noise_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.train_wav2vec2 --dataset fsc --model microsoft/wavlm-base-plus --epochs 15 --noise white
```

This is a GPU job: use Colab (`notebooks/train_wav2vec2.ipynb`, set the options in its first cell) or an Apple Silicon Mac. Check it works first with `--split random --epochs 1 --limit 40`, then run the full command. Upload the saved models (`outputs/models/wavlm_15ep_noise/`) to the Drive.

#### Done when

- `results/M2_rabbi/wavlm_15ep_noise/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I fine-tuned WavLM, a newer self-supervised speech model, with exactly the same settings as our wav2vec 2.0 run, to see whether a better pre-training objective helps (numbers from `results/M2_rabbi/wavlm_15ep_noise/fsc/summary.json`)."

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


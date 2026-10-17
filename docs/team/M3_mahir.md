# M3 · Mahir Ahmed

**In short:** Decides how results are judged (analysis plan, statistics, results table, hypothesis tests, error analysis) and runs k-NN, wav2vec 2.0 for 3 epochs and HuBERT Base.

On this page: [My tasks](#my-tasks) · [Faculty update](#faculty-update) · [Task cards](#task-cards) · back to the [team index](README.md) or the [task board](../STATUS.md)

| | |
|---|---|
| **Role** | Evaluation code: analysis plan, statistics, results table, hypothesis tests, error analysis |
| **Models** | k-NN (traditional) · wav2vec 2.0 3 epochs · **HuBERT Base** 15 epochs + white noise |
| **Branch** | `dev_mahir` (Tanvir merges it into `main`) |
| **Owns** | `support/metrics.py`, `support/stats.py`, `support/results_table.py`, `support/error_analysis.py`, `support/hypotheses.py`, `docs/ANALYSIS_PLAN.md`, `docs/literature/classical_and_evaluation.md`, `results/knn/`, `results/w2v2_3ep/`, `results/hubert_15ep_noise/`, `results/all_results.csv`, `results/hypotheses.csv`, `results/errors/` |
| **Report** | Sections 5 (experiments and results) and 6 (discussion) |

**Who edits what:** Mahir (or Mahir's assistant) edits *My tasks*, *Faculty update*, *Requests for the lead* and *Handoff log*. Tanvir writes the notes and the *Task cards* section at the end. Dependencies and merge status: [STATUS.md](../STATUS.md).

## Note from Tanvir (2026-10-17, evening): nothing to wait for

Every one of your tasks below depends only on work that is already merged or on your own earlier tasks. Data preparation (download, selection, QC, splits) is done and merged for TenVoices-FSC, so you can run your models today: [docs/RUNS.md](../RUNS.md) shows how. Work that needs everyone's results is split: you write and test the code, and Tanvir runs it on all results at the end.

## First-time setup

1. Ask Tanvir for a GitHub collaborator invite and accept it.
2. `git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git`, then `cd Voices-Speech-Recognition-445`.
3. `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
4. Set your own identity: `git config user.name "<your name>"` and `git config user.email "<your GitHub email>"`.
5. **Create your branch** (it is not on GitHub yet): `git checkout -b dev_mahir origin/main`, then `git push -u origin dev_mahir`.
6. Work by hand or with any AI assistant; [CONTRIBUTING.md](../CONTRIBUTING.md) has the steps and a ready-made prompt, and the card for each of your tasks is in **Task cards** at the end of this page. Before running a model, get the audio and build the processed copies once: [docs/RUNS.md](../RUNS.md), steps 1–2. With an assistant that works in the repository (Claude Code, Codex, Cursor, …), just say: *"I'm Mahir (M3). What is completed? Then complete my next tasks."*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T03 | Metrics module `support/metrics.py` (done by Tanvir before you joined; you own it now) | — | ✅ | Tanvir · 2026-10-04 |
| T10 | Analysis plan `docs/ANALYSIS_PLAN.md`, consistent with the frozen decisions ([card](#t10)) | — | ⬜ | |
| T33 | Statistics module `support/stats.py` ([card](#t33)) | — | ⬜ | |
| T37 | Literature notes ([card](#t37)) | — | ⬜ | |
| T13 | k-NN → `results/knn/fsc/` ([card](#t13)) | — | ⬜ | |
| T17 | wav2vec 2.0, 3 epochs → `results/w2v2_3ep/fsc/` ([card](#t17)) | — | ⬜ | |
| T45 | HuBERT Base, 15 epochs + white noise → `results/hubert_15ep_noise/fsc/` ([card](#t45)) | — | ⬜ | |
| T20 | Results table `support/results_table.py` ([card](#t20)) | T33 | ⬜ | |
| T34 | Error analysis `support/error_analysis.py` on your runs ([card](#t34)) | T13 | ⬜ | |
| T40 | Hypothesis tests `support/hypotheses.py` ([card](#t40)) | T10, T20, T33 | ⬜ | |

**Start with:** T33, T13 (minutes on a laptop) and T10. Run T17 and T45 on Colab. Your code is tested on synthetic results and on your own runs; Tanvir runs it on everyone's results at the end, so nothing you do waits for anyone.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

*(Nothing yet.)*

## Requests for the lead

*(None.)*

## Handoff log (newest first)

*(Nothing yet.)*

## Task cards

Written by Tanvir: exactly what to build, which files, which functions to use, the tests and when the task is done.

<a id="t10"></a>
### T10 · Analysis plan (M3 · Mahir)

| | |
|---|---|
| **Goal** | Document exactly how every model is evaluated and how each hypothesis is judged. The decisions that matter were frozen by M1 on 2026-10-17 (*Frozen analysis decisions* in `docs/PROJECT_PLAN.md`), before any model was trained, so this plan explains and details them and does not gate any run |
| **Depends on** | T03 ✅ |
| **You create** | `docs/ANALYSIS_PLAN.md` |
| **Read, don't edit** | `docs/PROJECT_PLAN.md`: sections *Objectives, Research Questions and Hypotheses*, *Methodology*, *Experiments*, *Evaluation Protocol* |
| **Give your assistant** | `AGENTS.md`, this card, `docs/PROJECT_PLAN.md` |

#### Write these sections

1. **Purpose and freeze rule.** The plan is frozen when M1 merges it. Later changes need M1's approval and a dated entry in a *Deviations* section at the end.
2. **Data.** Two datasets, analysed separately (`docs/DATASETS.md`): **TenVoices-FSC**, 10 speakers × 50 spoken commands × 2 takes (up to 1,000 clips; use the real count from T42), and **TenVoices-L2**, 10 non-native speakers × 50 sentences × 1 take (500 clips). No clip is ever used for both training and testing within a split.
3. **Splits** (per dataset).
   - Random split: stratified 80/20 by phrase, seed 42; 10% of the training part for validation.
   - Speaker-independent CV: 5 folds, 2 test speakers (one female, one male), 1 validation speaker, 7 training speakers; every speaker tested exactly once. On TenVoices-L2 the two test speakers of a fold share a first language, so every fold holds out one language (fold 1 Hindi, 2 Mandarin, 3 Spanish, 4 Korean, 5 Vietnamese).
   - Whisper is zero-shot: scored on the same test clips, uses no training data.
4. **Runs.** A table of every run with its folders in `results/`, one per dataset (`results/<run>/fsc/`, `results/<run>/l2arctic/`; Tanvir's runs are in `results/M1_tanvir/<run>/<dataset>/`): `random_forest`, `xgboost`, `knn`, `svm`, `w2v2_untrained`, `w2v2_3ep`, `w2v2_8ep_noise`, `w2v2_15ep_noise`, `w2v2_15ep`, `whisper_small`.
5. **Metrics.** Accuracy (primary), macro F1, weighted F1, per-class F1, confusion matrix, per-speaker accuracy; on TenVoices-L2 also per first language; WER and CER for Whisper. For the 5 folds, say how results are combined: pool the predictions of all folds (each clip is tested once, so every clip of the dataset gets one test prediction) for the headline numbers, and also report mean ± standard deviation over folds.
6. **Hypotheses H1–H5.** For each one: the exact comparison (which runs, which split, which dataset), the test (from `support/stats.py`, task T33) and the pass criterion, all as in the *Frozen analysis decisions* table (H5: one-sided Mann–Whitney U over per-speaker WER, L2 > FSC). Explain how the text difference between commands and sentences is handled in the discussion. Write every hypothesis so that anyone can decide *pass* or *fail* from the numbers without judgement.
7. **Statistical tests.** Bootstrap 95% CI (10,000 resamples, seed 42), exact McNemar, Wilcoxon signed-rank over the 10 speakers (exploratory), one-sided Mann–Whitney U for H5, α = 0.05, one primary test per hypothesis and dataset without correction, as frozen; say why.
8. **Model selection and leakage rules.** Hyper-parameters and best epochs chosen on validation data only; no speaker in more than one of train / validation / test within a fold; noise-augmented copies made from training clips only.
9. **Reporting.** Every hypothesis is reported whether it holds or not; the figures and tables that will show each one.
10. **Deviations.** Empty table (date, change, reason, approved by).

#### Done when

- Every hypothesis H1–H5 has a comparison, a test and a pass criterion.
- Nothing contradicts `docs/PROJECT_PLAN.md`. If you think the plan should change, write it under **Requests for the lead** instead of changing it here.
- Every decision in the *Frozen analysis decisions* table appears in the plan unchanged, with its reason.
- M1 approves it when merging. It is needed before your own T40, and it never holds up anyone's runs.

#### For the faculty update

"I wrote the analysis plan: how every model is evaluated on seen and unseen speakers in both datasets, which statistical test decides each of our five hypotheses, and the rules that prevent data leakage. It is frozen before the experiments start."

---

<a id="t33"></a>
### T33 · Statistics module (M3 · Mahir)

| | |
|---|---|
| **Goal** | The statistical tests that decide whether one model is really better than another, so the hypotheses (H1–H5) are judged by tests and not by eye |
| **Depends on** | T03 ✅ |
| **You create** | `support/stats.py`, `tests/test_stats.py` |
| **Use, don't edit** | `support/metrics.py` (`compute_metrics`, `per_speaker_accuracy`); `scipy.stats` (`binomtest`, `wilcoxon`, `mannwhitneyu`); `numpy` |
| **Give your assistant** | `AGENTS.md`, this card, `support/metrics.py`, `tests/test_metrics.py` (style example), `requirements.txt` |

#### Build

All functions take plain lists or arrays of phrase IDs (e.g. `"CMD02"`), check that the lengths match, and raise `ValueError` on empty or mismatched input.

1. `bootstrap_accuracy_ci(y_true, y_pred, n_boot=10_000, confidence=0.95, seed=42) -> dict`
   - Percentile bootstrap: resample the clips with replacement `n_boot` times, compute accuracy each time, take the `(1 − confidence)/2` and `1 − (1 − confidence)/2` quantiles.
   - Returns `{"accuracy", "ci_low", "ci_high", "n", "n_boot", "confidence"}`.
   - Uses `numpy.random.default_rng(seed)`, so the same seed gives the same interval.
2. `mcnemar_test(y_true, pred_a, pred_b) -> dict`
   - `b` = clips model A gets right and model B gets wrong; `c` = clips A gets wrong and B gets right.
   - Exact two-sided test: `scipy.stats.binomtest(b, b + c, 0.5).pvalue`; `p_value = 1.0` when `b + c == 0`.
   - Returns `{"n", "b", "c", "p_value", "better"}` where `better` is `"A"`, `"B"` or `"tie"` (by `b` vs `c`).
3. `wilcoxon_speakers(acc_a, acc_b) -> dict`
   - Paired per-speaker accuracies of two models, either two equal-length sequences or two `{speaker_id: accuracy}` dicts (pair them by speaker; raise if the speakers differ).
   - `scipy.stats.wilcoxon(a, b)`, two-sided.
   - Returns `{"n", "statistic", "p_value", "median_difference"}`.
4. `mannwhitney_speakers(values_a, values_b, alternative="two-sided") -> dict`
   - Unpaired per-speaker values from two **different** groups of speakers, e.g. Whisper's per-speaker WER on TenVoices-L2 vs TenVoices-FSC (H5). Lists or `{speaker_id: value}` dicts; the groups may differ in size.
   - `scipy.stats.mannwhitneyu(a, b, alternative=alternative, method="exact")`; `alternative` is `"two-sided"`, `"greater"` or `"less"` (T10 decides which H5 uses).
   - Returns `{"n_a", "n_b", "statistic", "p_value", "alternative", "median_a", "median_b"}`.
5. A CLI is not needed for this task (T20 calls these functions).

#### Tests (`tests/test_stats.py`)

Hand-computed cases, with the working written in the test's docstring:
- McNemar with `b = 1`, `c = 9`: exact two-sided p = 2 × (1 + 10) / 2¹⁰ = 22/1024 = **0.021484375**.
- McNemar with identical predictions: `b = c = 0`, p = 1.0, `better = "tie"`.
- Wilcoxon on 10 speakers where model A is better on every speaker by 0.01, 0.02, …, 0.10: statistic 0, exact two-sided p = 2 / 2¹⁰ = **0.001953125**.
- Wilcoxon with dicts in a different speaker order gives the same result as aligned lists; mismatched speakers raise.
- Mann–Whitney with `a = [0.1, 0.2, 0.3]`, `b = [0.4, 0.5, 0.6]` (complete separation): U = 0, exact two-sided p = 2 / C(6, 3) = 2/20 = **0.1**; with the groups swapped and `alternative="greater"`: U = 9, p = 1/20 = **0.05**.
- Bootstrap: all predictions correct → `(1.0, 1.0, 1.0)`; the interval contains the point accuracy; the same seed gives identical results; length mismatch raises.

#### Done when

- `python -m unittest discover -s tests -v` passes.
- Every function has a docstring that says what it computes and when to use it.

#### For the faculty update

"I implemented the statistical tests we use to compare models: bootstrap confidence intervals for accuracy, McNemar's test for two models on the same clips, the Wilcoxon signed-rank test over speakers and the Mann–Whitney test for comparing our two datasets, each checked against hand-computed values."

---

<a id="t37"></a>
### T37 · Literature notes: classical speech classifiers and evaluation (M3 · Mahir)

| | |
|---|---|
| **Goal** | Background for report section 2 (related work) and for explaining our traditional baselines and evaluation choices |
| **Depends on** | — |
| **You create** | `docs/literature/classical_and_evaluation.md` |
| **Give your assistant** | `AGENTS.md`, this card, the *References* section of `docs/PROJECT_PLAN.md` |

#### What to cover (6–8 papers)

1. **MFCC features for speech recognition.** Start from Davis and Mermelstein (1980), already in the project references.
2. **Classical classifiers on speech features**: k-NN, SVM, Random Forest or gradient boosting used for spoken word, command or phrase recognition (2–3 papers).
3. **Speaker-independent evaluation**: why testing on unseen speakers matters, leave-speaker-out / grouped cross-validation (1–2 papers).
4. **Comparing classifiers statistically**: McNemar's test and related tests for comparing two models on the same test set (for example Dietterich, 1998, *Neural Computation*), and the bootstrap.

#### Format, per paper

```markdown
##### <Short title> (<first author>, <year>)
- **Reference:** authors, title, venue, year. <DOI or arXiv link>
- **What they did:** 2–3 sentences.
- **What we take from it:** 1–2 sentences, linked to TenVoices (a design choice, a baseline, a metric).
```

End with a short **Summary** paragraph (5–6 sentences) that could become the first draft of the related-work paragraph on classical methods.

#### Rules

- **Every reference must be real.** Open each DOI or arXiv link and check the title, authors and year before writing it down. AI assistants often invent plausible-looking references: never copy one without opening it.
- Prefer peer-reviewed papers; textbooks are fine for the bootstrap.
- Write the notes in your own words.

#### Done when

- 6–8 checked papers in the format above, plus the summary.

#### For the faculty update

"I reviewed the literature on MFCC features, classical classifiers for spoken commands and speaker-independent evaluation. It explains why our traditional baselines and our unseen-speaker test are set up the way they are."

---

<a id="t13"></a>
### T13 · k-NN (M3 · Mahir)

| | |
|---|---|
| **Goal** | k-nearest neighbours on the 78 MFCC statistics: the simplest traditional baseline. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/knn/fsc/` and `images/confusion/knn_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.classical --model knn --dataset fsc
```

#### Done when

- `results/knn/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I trained and evaluated k-nearest neighbours on MFCC features, on seen and on unseen speakers (numbers from `results/knn/fsc/summary.json`)."

---

<a id="t17"></a>
### T17 · wav2vec 2.0, 3 epochs (M3 · Mahir)

| | |
|---|---|
| **Goal** | The shortest fine-tuning run of the 3 / 8 / 15-epoch comparison (H4), without noise. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/w2v2_3ep/fsc/` and `images/confusion/w2v2_3ep_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.train_wav2vec2 --dataset fsc --epochs 3
```

This is a GPU job: use Colab (`notebooks/train_wav2vec2.ipynb`, set the options in its first cell) or an Apple Silicon Mac. Check it works first with `--split random --epochs 1 --limit 40`, then run the full command. Upload the saved models (`outputs/models/w2v2_3ep/`) to the Drive.

#### Done when

- `results/w2v2_3ep/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I fine-tuned wav2vec 2.0 for 3 epochs, the short end of our epochs comparison (numbers from `results/w2v2_3ep/fsc/summary.json`)."

---

<a id="t45"></a>
### T45 · HuBERT Base, 15 epochs + white noise (M3 · Mahir)

| | |
|---|---|
| **Goal** | HuBERT (Hsu et al., 2021), the same size as wav2vec 2.0 Base but pre-trained to predict cluster labels; fine-tuned with exactly the same script, data and settings as M1's wav2vec 2.0 run. Exploratory comparison E10. |
| **Depends on** | — (shared code and the TenVoices-FSC splits are merged) |
| **You create** | `results/hubert_15ep_noise/fsc/` and `images/confusion/hubert_15ep_noise_fsc_*.png` (by running the command) |
| **How** | [RUNS.md](../RUNS.md): get the audio, build the processed copies, run, commit |

#### Run

```bash
python -m support.train_wav2vec2 --dataset fsc --model facebook/hubert-base-ls960 --epochs 15 --noise white
```

This is a GPU job: use Colab (`notebooks/train_wav2vec2.ipynb`, set the options in its first cell) or an Apple Silicon Mac. Check it works first with `--split random --epochs 1 --limit 40`, then run the full command. Upload the saved models (`outputs/models/hubert_15ep_noise/`) to the Drive.

#### Done when

- `results/hubert_15ep_noise/fsc/summary.json` has both `random` and `folds` (5 folds pooled).
- The results folder and the two confusion-matrix images are committed and pushed on your branch.
- The faculty update on your page quotes the accuracies from `summary.json`.

#### For the faculty update

"I fine-tuned HuBERT, another self-supervised speech model, with exactly the same settings as our wav2vec 2.0 run, to compare pre-training methods (numbers from `results/hubert_15ep_noise/fsc/summary.json`)."

---

<a id="t20"></a>
### T20 · Results table (M3 · Mahir)

| | |
|---|---|
| **Goal** | One table with every model on both protocols, built from `results/` by code, so no number is ever copied by hand |
| **Depends on** | T33 (your own) |
| **You create** | `support/results_table.py`, `tests/test_results_table.py`, `results/all_results.csv`, `images/all_results_table.png` |
| **Use, don't edit** | `support/evaluate.py` (`list_runs`, `run_dir`, `load_run_predictions`; the `summary.json` layout), your `support/stats.py` |
| **Give your assistant** | `AGENTS.md`, this card, `support/evaluate.py`, your `support/stats.py`, one real `summary.json` (e.g. `results/M1_tanvir/random_forest/fsc/summary.json`) |

#### Build

`python -m support.results_table [--dataset fsc]` reads every run's `summary.json` and the predictions next to it and writes (find the runs with `support.evaluate.list_runs()`: some are in `results/<run>/<dataset>/`, Tanvir's in `results/M1_tanvir/<run>/<dataset>/`):
- `results/all_results.csv`: one row per run and dataset: random-split accuracy / macro F1 / weighted F1, pooled-fold accuracy / macro F1 / weighted F1, fold mean ± std, the drop from random split to unseen speakers, and 95% bootstrap intervals (your `stats.py`) for both accuracies;
- `results/per_speaker_<dataset>.csv`: pooled-fold accuracy per speaker and run;
- `images/all_results_table.png`: the table as an image.

It works with whatever runs exist (one or twenty). Tests: build fake runs in a temporary folder with `support.evaluate.save_run` and check the table, the drop and the intervals.

#### Done when

- Tests pass; the table is built from the runs merged so far. (M1 rebuilds it on all runs in T24.)

#### For the faculty update

"I built the results table generator: every model on seen and unseen speakers, with confidence intervals, computed straight from the result files."

---

<a id="t34"></a>
### T34 · Error analysis (M3 · Mahir)

| | |
|---|---|
| **Goal** | Understand *what* the models get wrong: which phrases they confuse, which phrase categories and speakers are hard |
| **Depends on** | T13 (your own) |
| **You create** | `support/error_analysis.py`, `tests/test_error_analysis.py`, `results/errors/` |
| **Use, don't edit** | `support/evaluate.py` (`load_run_predictions`), `support/dataset.py` (`load_labels`: phrase text and category), `support/metrics.py` |

#### Build

`python -m support.error_analysis --run knn --dataset fsc` writes `results/errors/<run>_<dataset>.csv` (the 10 most-confused phrase pairs on the pooled folds with their texts and counts, accuracy per phrase category and per speaker) and a short `results/errors/<run>_<dataset>.md` summary. Tests with hand-made predictions.

#### Done when

- Tests pass; run on your own runs (k-NN, then the 3-epoch and HuBERT runs when they are done) with a short written interpretation in `results/errors/README.md`. (M1 runs it on every run in T24.)

#### For the faculty update

"I analysed the errors: which commands the models confuse (for example two wordings of the same request) and which speakers are hardest."

---

<a id="t40"></a>
### T40 · Hypothesis tests (M3 · Mahir)

| | |
|---|---|
| **Goal** | Decide H1–H5 by code, exactly as frozen before the experiments, so the verdicts cannot be bent after seeing the results |
| **Depends on** | T10, T20, T33 (your own) |
| **You create** | `support/hypotheses.py`, `tests/test_hypotheses.py`, `results/hypotheses.csv` |
| **Read, don't edit** | `docs/PROJECT_PLAN.md` (*Hypotheses* and *Frozen analysis decisions*), `support/noise_test.py` output layout (`noise/` inside the run's results root, `support.evaluate.results_root(run)`: `results/M1_tanvir/noise/` for the noise tests so far), `support/whisper_asr.py` output (`wer_per_speaker.csv`) |

#### Build

`python -m support.hypotheses` writes `results/hypotheses.csv`: one row per hypothesis and dataset with the runs compared, the test statistic, the p-value, the effect (e.g. accuracy difference), the criterion and the verdict (`pass`, `fail` or `not yet computable` when a needed run is missing). Implement each decision of the frozen table literally; your own `ANALYSIS_PLAN.md` (T10) explains them. Tests: synthetic result folders where you know the right verdict (e.g. a clear H1 pass and a clear fail).

#### Done when

- Tests pass; running it on the merged runs gives verdicts or "not yet computable". (M1 runs it on all results in T24.)

#### For the faculty update

"I implemented our five hypothesis tests exactly as we fixed them before the experiments, so the verdicts come straight from the results."


# Tanvir Ahmed (M1)

Lead · data preparation, shared code, Whisper pipeline, noise tests, inference, TenVoices-L2, final aggregation and merging

| | |
|---|---|
| **Models** | Random Forest · wav2vec 2.0 15 epochs with and without noise · Whisper zero-shot |
| **GitHub** | [@minhaz-42](https://github.com/minhaz-42) |
| **Branch** | `main` |
| **Working page** | [docs/team/M1_tanvir.md](../docs/team/M1_tanvir.md) (task cards, weekly faculty update, handoff log) |
| **In numbers** | 17 tasks done, 1 in progress · 49 commits · 4 runs with results · 223 files created |

## Tasks

From the task board [docs/STATUS.md](../docs/STATUS.md): ✅ done and merged · 🟡 in progress · ⬜ to do.

| ID | Task | Status | Done by · date |
|---|---|---|---|
| T11 | Features, shared runner and evaluation ✅; **Random Forest** → `results/M1_tanvir/random_forest/fsc/` ✅ (FSC: random split 0.310, unseen speakers 0.209) | ✅ | Tanvir · 2026-10-17 |
| T30 | **Whisper zero-shot** → `results/M1_tanvir/whisper_small/{fsc,l2arctic}/` ✅: FSC unseen speakers 0.987, WER 0.039; L2 unseen speakers 0.996, WER 0.100 (per first language 0.040–0.168) | ✅ | Tanvir · 2026-10-17 |
| T15 | Training script + Colab notebook ✅; **wav2vec 2.0, 15 epochs + white noise** → `results/M1_tanvir/w2v2_15ep_noise/fsc/` ✅ (unseen speakers 0.885, random split 1.000); models in `outputs/models/` (to Drive) | ✅ FSC | Tanvir · 2026-10-17 |
| T19 | **wav2vec 2.0, 15 epochs without noise** → `results/M1_tanvir/w2v2_15ep/fsc/` ✅ (unseen speakers 0.928, random split 0.855) | ✅ FSC | Tanvir · 2026-10-17 |
| T21 | Noise tests → `results/M1_tanvir/noise/` ✅ (unseen speakers at 10 / 5 dB: with noise training 0.821 / 0.745, without 0.655 / 0.307; clean 0.885 vs 0.928); H3 decided in T24 | ✅ FSC | Tanvir · 2026-10-17 |
| T22 | `main.py`, inference, CI ✅; benchmark → `results/M1_tanvir/benchmark_fsc.csv` ✅ (wav2vec 2.0: 38.2 ms per clip on MPS, 74.9 ms on CPU) | ✅ | Tanvir · 2026-10-17 |
| T48 | **One results folder for M1:** every run of Tanvir's (`random_forest`, `w2v2_15ep_noise`, `w2v2_15ep`, `whisper_small`), the noise tests, the benchmark and the confusion matrices → `results/M1_tanvir/` (`result_folders` in the config; `support.evaluate.run_dir` / `list_runs` find it) | ✅ | Tanvir · 2026-10-17 |
| T49 | **Figures beyond confusion matrices:** `support/figures.py` draws, from the result files only, seen vs unseen speakers, fine-tuning curves, noise robustness, accuracy per speaker, confidence (right vs wrong, rejecting uncertain answers), speed vs accuracy and Whisper's errors → `results/M1_tanvir/figures/` (gallery: `results/M1_tanvir/README.md`) | ✅ | Tanvir · 2026-10-17 |
| T50 | **One page per member for the faculty:** `support/contributions.py` writes `contributions/README.md` and `contributions/<slot>_<name>.md` from the repository: tasks and status (this board), results (`summary.json` of each run), files created or changed and every commit (git history of `main`); team in `team` of the config; rerun after every merge | ✅ | Tanvir · 2026-10-23 |
| T47 | **TenVoices-L2:** data from a free copy of L2-ARCTIC (no form) ✅: 10 speakers × 50 sentences = 500 clips, QC 500 / 500, splits (each fold holds out one first language); Random Forest ✅; Whisper ✅ (WER 0.100 vs 0.039 on FSC); next: the wav2vec 2.0 runs and every merged model on L2; H5 | 🟡 data ✅ | Tanvir · 2026-10-17 (data) |
| T24 | Run T20 (results table), T40 (hypotheses), T34 (errors) and T23's figure command on all merged results; clean-environment reproducibility check → **feature freeze** | ⬜ |  |
| T25 | Final report (paper format, ~8 pages) and README with real results, using the T37 / T38 literature notes | ⬜ |  |
| T00 | Project plan, README, poster, phrase list, guides, board; revised for two published datasets; analysis decisions frozen | ✅ | Tanvir · 2026-10-01 … 2026-10-17 |
| T01 | Repository, pinned environment, config, audio pre-processing + tests | ✅ | Tanvir · 2026-10-04 |
| T03 | `support/metrics.py` + tests | ✅ | Tanvir · 2026-10-04 |
| T29 | Whisper pipeline code + tests | ✅ | Tanvir · 2026-10-17 |
| T41 | Fluent Speech Commands downloaded and checked against the paper (`data/SOURCES.md`); L2-ARCTIC moved to T47 | ✅ | Tanvir · 2026-10-17 |
| T42 | `support/select_subsets.py`; **TenVoices-FSC** selected: 10 speakers (5 F, 5 M) × 50 phrases × 2 takes = 1,000 clips | ✅ | Tanvir · 2026-10-17 |
| T43 | `support/qc_report.py`; TenVoices-FSC: 997 / 1,000 clips pass, 3 flagged for clipping | ✅ | Tanvir · 2026-10-17 |
| T08 | Processed copies, manifest, random split + 5 speaker-independent folds of TenVoices-FSC, leakage tests | ✅ | Tanvir · 2026-10-17 |

## Results

Unseen speakers: speaker-independent 5-fold cross-validation (the headline). Random split: 80 / 20, the test speakers were also heard in training.

| Run | Dataset | Accuracy, unseen speakers | Macro F1, unseen | Accuracy, random split | Macro F1, random | Source |
|---|---|---:|---:|---:|---:|---|
| Random Forest | TenVoices-FSC | 20.9% | 0.199 | 31.0% | 0.299 | [summary.json](../results/M1_tanvir/random_forest/fsc/summary.json) |
| Random Forest | TenVoices-L2 | 17.0% | 0.149 | 9.0% | 0.070 | [summary.json](../results/M1_tanvir/random_forest/l2arctic/summary.json) |
| wav2vec 2.0, 15 epochs + noise | TenVoices-FSC | 88.5% | 0.881 | 100.0% | 1.000 | [summary.json](../results/M1_tanvir/w2v2_15ep_noise/fsc/summary.json) |
| wav2vec 2.0, 15 epochs | TenVoices-FSC | 92.8% | 0.927 | 85.5% | 0.824 | [summary.json](../results/M1_tanvir/w2v2_15ep/fsc/summary.json) |
| Whisper small, zero-shot | TenVoices-FSC | 98.7% | 0.987 | 99.0% | 0.990 | [summary.json](../results/M1_tanvir/whisper_small/fsc/summary.json) |
| Whisper small, zero-shot | TenVoices-L2 | 99.6% | 0.996 | 100.0% | 1.000 | [summary.json](../results/M1_tanvir/whisper_small/l2arctic/summary.json) |

## Files

Created by Tanvir (files that still exist):

- **Code:** [main.py](../main.py) · [notebooks/train_wav2vec2.ipynb](../notebooks/train_wav2vec2.ipynb) · [support/__init__.py](../support/__init__.py) · [support/audio.py](../support/audio.py) · [support/classical.py](../support/classical.py) · [support/config.py](../support/config.py) · [support/contributions.py](../support/contributions.py) · [support/dataset.py](../support/dataset.py) · [support/device.py](../support/device.py) · [support/evaluate.py](../support/evaluate.py) · [support/features.py](../support/features.py) · [support/figures.py](../support/figures.py) · [support/inference.py](../support/inference.py) · [support/l2arctic_from_hf.py](../support/l2arctic_from_hf.py) · [support/make_splits.py](../support/make_splits.py) · [support/metrics.py](../support/metrics.py) · [support/noise.py](../support/noise.py) · [support/noise_test.py](../support/noise_test.py) · [support/qc_report.py](../support/qc_report.py) · [support/select_subsets.py](../support/select_subsets.py) · [support/train_wav2vec2.py](../support/train_wav2vec2.py) · [support/whisper_asr.py](../support/whisper_asr.py)
- **Tests:** [tests/__init__.py](../tests/__init__.py) · [tests/_synthetic.py](../tests/_synthetic.py) · [tests/_wav2vec2_check.py](../tests/_wav2vec2_check.py) · [tests/fixtures/phrases_v0.csv](../tests/fixtures/phrases_v0.csv) · [tests/test_audio.py](../tests/test_audio.py) · [tests/test_classical.py](../tests/test_classical.py) · [tests/test_config.py](../tests/test_config.py) · [tests/test_contributions.py](../tests/test_contributions.py) · [tests/test_dataset.py](../tests/test_dataset.py) · [tests/test_evaluate.py](../tests/test_evaluate.py) · [tests/test_features.py](../tests/test_features.py) · [tests/test_figures.py](../tests/test_figures.py) · [tests/test_inference.py](../tests/test_inference.py) · [tests/test_l2arctic_from_hf.py](../tests/test_l2arctic_from_hf.py) · [tests/test_make_splits.py](../tests/test_make_splits.py) · [tests/test_metrics.py](../tests/test_metrics.py) · [tests/test_noise.py](../tests/test_noise.py) · [tests/test_noise_test.py](../tests/test_noise_test.py) · [tests/test_qc_report.py](../tests/test_qc_report.py) · [tests/test_select_subsets.py](../tests/test_select_subsets.py) · [tests/test_train_wav2vec2.py](../tests/test_train_wav2vec2.py) · [tests/test_whisper_asr.py](../tests/test_whisper_asr.py)
- **Documents:** [AGENTS.md](../AGENTS.md) · [CLAUDE.md](../CLAUDE.md) · [README.md](../README.md) · [data/README.md](../data/README.md) · [data/SOURCES.md](../data/SOURCES.md) · [data/processed/README.md](../data/processed/README.md) · [data/raw/README.md](../data/raw/README.md) · [data/splits/README.md](../data/splits/README.md) · [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md) · [docs/DATASETS.md](../docs/DATASETS.md) · [docs/PROJECT_PLAN.md](../docs/PROJECT_PLAN.md) · [docs/RUNS.md](../docs/RUNS.md) · [docs/STATUS.md](../docs/STATUS.md) · [docs/team/M1_tanvir.md](../docs/team/M1_tanvir.md) · [docs/team/M2_rabbi.md](../docs/team/M2_rabbi.md) · [docs/team/M3_mahir.md](../docs/team/M3_mahir.md) · [docs/team/M4_shafil.md](../docs/team/M4_shafil.md) · [docs/team/README.md](../docs/team/README.md) · [docs/team/individual_report_template.md](../docs/team/individual_report_template.md) · [notebooks/README.md](../notebooks/README.md) · [others/README.md](../others/README.md) · [others/poster.html](../others/poster.html) · [outputs/README.md](../outputs/README.md) · [requirements.txt](../requirements.txt) · [support/README.md](../support/README.md)
- **Figures:** [images/poster.png](../images/poster.png)
- **Data and settings:** [.github/workflows/tests.yml](../.github/workflows/tests.yml) · [configs/config.yaml](../configs/config.yaml) · [data/metadata/](../data/metadata/) (2 files) · [data/prompts/](../data/prompts/) (3 files) · [data/splits/](../data/splits/) (14 files) · [data/subsets/](../data/subsets/) (6 files)
- **Results:** [results/README.md](../results/README.md) · [results/M1_tanvir/](../results/M1_tanvir/) (2 files) · [results/M1_tanvir/figures/](../results/M1_tanvir/figures/) (9 files) · [results/M1_tanvir/figures/confusion/](../results/M1_tanvir/figures/confusion/) (12 files) · [results/M1_tanvir/noise/](../results/M1_tanvir/noise/) (6 files) · [results/M1_tanvir/random_forest/](../results/M1_tanvir/random_forest/) (30 files) · [results/M1_tanvir/w2v2_15ep/](../results/M1_tanvir/w2v2_15ep/) (15 files) · [results/M1_tanvir/w2v2_15ep_noise/](../results/M1_tanvir/w2v2_15ep_noise/) (15 files) · [results/M1_tanvir/whisper_small/](../results/M1_tanvir/whisper_small/) (34 files)
- **Other:** [.gitignore](../.gitignore) · [LICENSE](../LICENSE)

## Commits

49 commits on `main` under Tanvir's name (minhaz, Tanvir Ahmed), newest first; merge commits are not counted.

<details><summary>Show all 49 commits</summary>

| Date | Commit | Message |
|---|---|---|
| 2026-10-23 | [`4ff7797`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/4ff77978c51cfd4335ae0a4858075ba3f78cce8b) | T50: one page per member for the faculty, written from the repository by support.contributions |
| 2026-10-17 | [`5a340c5`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/5a340c56fb5ccae173efd2062cb977f11c61ff6f) | T49: seven new kinds of figures for M1's runs, drawn from the result files |
| 2026-10-17 | [`4b6728e`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/4b6728e7fcdc235decb347bd214a30cb154f309b) | T48: keep all of M1's results, noise tests, benchmark and figures in results/M1_tanvir/ |
| 2026-10-17 | [`3bf15b7`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/3bf15b7b75a08432976c747b41f69c7e3369699d) | T21: results, board and progress for T19, T21 and T22 |
| 2026-10-17 | [`902ab4e`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/902ab4ec83e08f4678fed081001d022961c2ca16) | T22: inference-time benchmark on a MacBook (GPU and CPU) |
| 2026-10-17 | [`3091014`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/3091014bc4e491f358e5925a505c35c5b8066e96) | T21: noise tests at 20 / 10 / 5 dB on TenVoices-FSC |
| 2026-10-17 | [`72440d6`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/72440d64e8bb404fbf4a8cbf382576cba5b1e72e) | T19: wav2vec 2.0 fine-tuned 15 epochs without noise on TenVoices-FSC |
| 2026-10-17 | [`2e7b541`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/2e7b541303eaee7c2050d6dbc5d7b3b26ff4f7fe) | T15: wav2vec 2.0 fine-tuned 15 epochs with white noise on TenVoices-FSC |
| 2026-10-17 | [`4fb9450`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/4fb9450106a6e1b5996cd105c6da11385630af9a) | T00: schedule: merges, final aggregation and report after next week's project update |
| 2026-10-17 | [`4928f07`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/4928f07f5430074f429d01afb46c495ba79614dc) | T30: Whisper zero-shot on TenVoices-L2, with WER per speaker and first language |
| 2026-10-17 | [`a6aa1ae`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/a6aa1ae7b31a494568e6422188e6e2519984d40c) | T47: Random Forest on TenVoices-L2 |
| 2026-10-17 | [`1964334`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/19643346c3267e4c51d6f58bcc24c94c62ccd061) | T47: TenVoices-L2 from a free copy of L2-ARCTIC: 10 speakers of 5 first languages, 500 clips, QC and splits |
| 2026-10-17 | [`a846062`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/a8460628a0cd7ac4bc4251ba9ffc2f81976e55ff) | T00: start-here steps on member pages, including creating or updating each member's branch |
| 2026-10-17 | [`7062ef0`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/7062ef08c5e8f25fffdc87ca00627fa5a0b2083d) | T00: one page per member, all code in support/, clearer README |
| 2026-10-17 | [`fbfa6bf`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/fbfa6bfbb4d10b65fc1b2566690e1c2e8cbae53e) | T30: Whisper zero-shot on TenVoices-FSC, with WER per speaker |
| 2026-10-17 | [`04faea6`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/04faea66c05cbfd03fe6c36c7f62ae9eb820be89) | T11: Random Forest on TenVoices-FSC (random split and 5 speaker-independent folds) |
| 2026-10-17 | [`135febf`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/135febf80d32ba7e2fb9700454adab65bf1bfec5) | T00: every member task depends only on merged work; freeze analysis decisions; one extra model per member |
| 2026-10-17 | [`df2e593`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/df2e593a3698962234a8d37992f1321da840a9e2) | T11: cached features from any extractor and logistic regression, for the linear probe |
| 2026-10-17 | [`8090e67`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/8090e67e2fecea91556c97b025e92d28bd26c5d6) | T15: run names per model family, so WavLM and HuBERT runs use the same script |
| 2026-10-17 | [`bbe9cfd`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/bbe9cfd9c9b1916923546f13f58455d163161365) | T08: random split and 5 speaker-independent folds of TenVoices-FSC |
| 2026-10-17 | [`ac26f1b`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/ac26f1b4e854a2e1b2553649e823f9d4e936738c) | T43: quality report of TenVoices-FSC (997 of 1,000 clips pass, 3 flagged for clipping) |
| 2026-10-17 | [`70e420b`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/70e420bd611048517073ca44deebc0d79dd0745a) | T42: select TenVoices-FSC by script (10 speakers, 50 phrases, 1,000 clips) and the L2 selection code |
| 2026-10-17 | [`2025564`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/202556496b0ab83d627549f19a5306d0674fd20d) | T00: code map, commands, board and progress for the pipeline code |
| 2026-10-17 | [`eb863a8`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/eb863a8f1e847ed34494b34e2eb2ca9df8f2c488) | T22: pin the CI runner image to Ubuntu 24.04 |
| 2026-10-17 | [`cdf55ea`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/cdf55eaea4d02f1f301ef0e42828c70d2c2a69b4) | T21: noise tests with identical white noise for every model, with tests |
| 2026-10-17 | [`e9c9540`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/e9c9540533d07c917bf53c1c5b4c72c1e2606c65) | T30: Whisper on the dataset splits with WER per speaker and first language, with tests |
| 2026-10-17 | [`1d82d05`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/1d82d05414e25223a22ec08fa2257392aed68fd3) | T22: inference for audio, video and microphone with top-5 phrases, benchmark option and CI |
| 2026-10-17 | [`cf7e223`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/cf7e2236c1be2c6f1a571712afb9df0fcd2a0d62) | T15: wav2vec 2.0 training script, white-noise augmentation and Colab notebook, with tests |
| 2026-10-17 | [`ebb4811`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/ebb4811104c7f91641e717e22760411c4af1a605) | T00: instructor confirms published datasets meet the brief |
| 2026-10-17 | [`f542772`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/f5427722ce72466a3381dc93d16f3ee6298f9341) | T11: MFCC features, shared evaluation and classical runner for all four models, with tests |
| 2026-10-17 | [`6052594`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/6052594d3b885fa227fb06042bb78931744fefca) | T08: dataset loader, processed copies and speaker-independent splits, with leakage tests |
| 2026-10-17 | [`333af45`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/333af4568eed7495456f5384ffd5a91149426aae) | T41: download Fluent Speech Commands and record its source |
| 2026-10-17 | [`4495ecd`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/4495ecd16677c790964173c351c40bd373ac8ed8) | T00: describe Fluent Speech Commands with phrasings quoted from its paper |
| 2026-10-17 | [`fdbd127`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/fdbd12793844f6f8f7d589eea2a7edb115a3ef41) | T00: push unfinished work as draft pull requests; placeholder READMEs match the two datasets |
| 2026-10-17 | [`c2e6dfe`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/c2e6dfeada3b8d6f7d094882f86275466d968586) | T00: switch to two published datasets instead of recording our own |
| 2026-10-17 | [`7e71331`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/7e713312441854e6d1b833a8c307f172cfabb756) | T00: replace .gitkeep placeholders with READMEs and add a code map |
| 2026-10-17 | [`8f47ed8`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/8f47ed8db9dc2806356478e59b180286eba47afc) | T29: zero-shot Whisper pipeline with phrase matching, WER/CER and tests |
| 2026-10-17 | [`f855e05`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/f855e057c4e0c4e48d3cc74f0ca349cfec9a5fd7) | T00: workflow for any AI assistant or by hand, with task cards |
| 2026-10-17 | [`970ddf0`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/970ddf0b3ba4ca42ebd9545247ed0e0d3efa6611) | T00: assign Mahir (M3) and Shafil (M4), one branch per member, rebalanced tasks and progress pages |
| 2026-10-17 | [`254c1ba`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/254c1ba35b275b89388c8fe9c4ea269b3b9c8f53) | T00: add the zero-shot Whisper pipeline (M1) with hypothesis H5 |
| 2026-10-04 | [`ba1db61`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/ba1db6140f73ed21538b74bf616c5a69762440a4) | T03: shared metrics (accuracy, F1, reports, confusion matrix, per-speaker) with hand-computed tests |
| 2026-10-04 | [`677d16a`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/677d16a8985d3270cfa67efb24a0417097b817b0) | T01: pin requirements, add shared config and audio pre-processing with tests |
| 2026-10-01 | [`77001a2`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/77001a2d8fe4745bfcbdb00e71f746cb3a253403) | T00: switch to the phrase-recognition pipeline |
| 2026-10-01 | [`b74bd59`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/b74bd59da366a6487b984d35f725b09e3ae0b566) | T01: publish the repo and point clone instructions at it |
| 2026-10-01 | [`a51dec0`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/a51dec04b1cef121a84a4523ba1a66a8766ccaa9) | Two-member team with open slots and a Claude Code relay workflow |
| 2026-10-01 | [`b4c9cd0`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/b4c9cd0fa34c8231b954cc8c1e80984dcc7a6e5a) | Restyle docs, drop references to earlier projects, add classical ML baselines |
| 2026-10-01 | [`fd2c239`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/fd2c239544162a5be02c331e4e11fa9c868095db) | Replace the day-by-day schedule with a weekly roadmap |
| 2026-10-01 | [`08557e9`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/08557e9a9e1a0f7f9017e1295ef1fdd2aad3e12f) | Add full project plan, data collection protocol, poster and repo scaffold |
| 2026-10-01 | [`3d163ff`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/3d163ff99001945d6fa3e90fda79bba5940dae40) | Initial commit |

</details>

---

Written by `python -m support.contributions` from the repository; do not edit by hand. Overview of the team: [README.md](README.md)

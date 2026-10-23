# Md. Shahriar Rakib Rabbi (M2)

Datasheet, listen-check, dataset statistics, subset release

| | |
|---|---|
| **Models** | XGBoost · wav2vec 2.0 8 epochs + noise · WavLM Base+ 15 epochs + noise |
| **GitHub** | [@Shahrcasm](https://github.com/Shahrcasm) |
| **Branch** | `dev_shahriar` |
| **Working page** | [docs/team/M2_rabbi.md](../docs/team/M2_rabbi.md) (task cards, weekly faculty update, handoff log) |
| **In numbers** | 5 tasks done, 0 in progress · 16 commits · 3 runs with results · 60 files created |

## Tasks

From the task board [docs/STATUS.md](../docs/STATUS.md): ✅ done and merged · 🟡 in progress · ⬜ to do.

| ID | Task | Status | Done by · date |
|---|---|---|---|
| T12 | **XGBoost** with the shared runner → `results/M2_rabbi/xgboost/fsc/` | ✅ | Rabbi · 2026-10-20 |
| T16 | **wav2vec 2.0, 8 epochs + white noise** → `results/M2_rabbi/w2v2_8ep_noise/fsc/` | ✅ | Rabbi · 2026-10-20 |
| T44 | **WavLM Base+, 15 epochs + white noise** (same script, `--model microsoft/wavlm-base-plus`) → `results/M2_rabbi/wavlm_15ep_noise/fsc/` | ✅ | Rabbi · 2026-10-20 |
| T09 | Datasheet `data/README.md` for TenVoices-FSC (sources, selection, QC, statistics, licences) + dataset statistics figure + 👤 listen-check of the 100 marked clips → `data/listen_check_fsc.csv` | ✅ | Rabbi · 2026-10-20 |
| T26 | Subset release: subset and phrase lists, selection and QC scripts, datasheet, checksums (no audio) as a GitHub Release draft | ✅ | Rabbi · 2026-10-20 |

## Results

Each run in detail (model, how it was trained, every split, accuracy per speaker, confusion matrices): [results/M2_rabbi/RUNS.md](../results/M2_rabbi/RUNS.md)

Unseen speakers: speaker-independent 5-fold cross-validation (the headline). Random split: 80 / 20, the test speakers were also heard in training.

| Run | Dataset | Accuracy, unseen speakers | Macro F1, unseen | Accuracy, random split | Macro F1, random | Source |
|---|---|---:|---:|---:|---:|---|
| XGBoost | TenVoices-FSC | 16.9% | 0.158 | 22.0% | 0.208 | [summary.json](../results/M2_rabbi/xgboost/fsc/summary.json) |
| wav2vec 2.0, 8 epochs + noise | TenVoices-FSC | 21.0% | 0.198 | 40.5% | 0.287 | [summary.json](../results/M2_rabbi/w2v2_8ep_noise/fsc/summary.json) |
| WavLM Base+, 15 epochs + noise | TenVoices-FSC | 33.6% | 0.354 | 39.0% | 0.332 | [summary.json](../results/M2_rabbi/wavlm_15ep_noise/fsc/summary.json) |

## Files

Created by Rabbi (files that still exist):

- **Code:** [support/dataset_stats.py](../support/dataset_stats.py) · [support/make_release.py](../support/make_release.py) · [support/phrases.py](../support/phrases.py)
- **Tests:** [tests/test_dataset_stats.py](../tests/test_dataset_stats.py) · [tests/test_make_release.py](../tests/test_make_release.py) · [tests/test_phrases.py](../tests/test_phrases.py)
- **Figures:** [images/dataset_fsc.png](../images/dataset_fsc.png)
- **Data and settings:** [data/listen_check_fsc.csv](../data/listen_check_fsc.csv) · [data/subsets/stats_fsc.csv](../data/subsets/stats_fsc.csv)
- **Results:** [results/M2_rabbi/figures/confusion/](../results/M2_rabbi/figures/confusion/) (6 files) · [results/M2_rabbi/w2v2_8ep_noise/](../results/M2_rabbi/w2v2_8ep_noise/) (15 files) · [results/M2_rabbi/wavlm_15ep_noise/](../results/M2_rabbi/wavlm_15ep_noise/) (15 files) · [results/M2_rabbi/xgboost/](../results/M2_rabbi/xgboost/) (15 files)

Also changed by Rabbi (created by someone else):

- **Documents:** [README.md](../README.md) · [data/README.md](../data/README.md) · [docs/STATUS.md](../docs/STATUS.md) · [docs/team/M2_rabbi.md](../docs/team/M2_rabbi.md)

## Commits

16 commits on `main` under Rabbi's name (Md. Shahriar Rakib Rabbi), newest first; merge commits are not counted.

<details><summary>Show all 16 commits</summary>

| Date | Commit | Message |
|---|---|---|
| 2026-10-21 | [`e4ae37e`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/e4ae37ec9c3ac3e6cbc3295c6b6e25df0a0baab0) | docs: showcase M2 Rabbi results analysis, figures, and release tools in README |
| 2026-10-20 | [`bba77d7`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/bba77d7d060fd92dba5998118cd3fb17ddbe37c3) | docs: update README and status board for merged M2 results |
| 2026-10-20 | [`105415b`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/105415b197a3f31be3396a98d18b6300903e50db) | T44: update member page and complete all M2 tasks |
| 2026-10-20 | [`aded990`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/aded990959090a61b1c16b3acc5fff01b50884b8) | T44: WavLM Base+ 15 epochs + white noise results on TenVoices-FSC |
| 2026-10-20 | [`13fd0e9`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/13fd0e9e784ad0e19c26a754a8147c317078af1a) | T16: update member page and faculty update for wav2vec 2.0 8ep run |
| 2026-10-20 | [`a56c8ab`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/a56c8ab16bd882b85f3796a35b85127bbfa9d673) | T16: wav2vec 2.0 8 epochs + white noise results on TenVoices-FSC |
| 2026-10-20 | [`150121f`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/150121f55bde2e8d8d19dbcbe63e3993627b37b1) | T26: update member page and handoff log for subset release |
| 2026-10-20 | [`596dade`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/596dade434db40de7fc33b2682e22bb6bc7c4d10) | T26: implement support/make_release.py packaging tool |
| 2026-10-20 | [`84e88f3`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/84e88f312cfdc5aa5570cb7d6371495cce0bcfbc) | T26: add unit tests for subset release packager in tests/test_make_release.py |
| 2026-10-20 | [`79994f7`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/79994f7e592d26fffbe61c5fe4ef116b6e9d87f4) | T09: author dataset datasheet in data/README.md and update status |
| 2026-10-20 | [`c62a289`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/c62a2894292d11a36c5b9c3b091b3c062c363b83) | T09: complete 100-clip audio listen-check (data/listen_check_fsc.csv) |
| 2026-10-20 | [`5b05e70`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/5b05e707580779be6f6e688ba548282512b37e6c) | T09: generate dataset statistics table and images/dataset_fsc.png |
| 2026-10-20 | [`09c052b`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/09c052b36f63c8463f101f595e534a54ab03ea35) | T09: implement compute_dataset_stats in support/dataset_stats.py |
| 2026-10-20 | [`72962f3`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/72962f3ae0f7534459a19b726016b3b5ce1935ba) | T09: add unit test suite for dataset statistics |
| 2026-10-20 | [`a0115ae`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/a0115aeb37d8011124ae3135cb426e782689a676) | T12: evaluate XGBoost on TenVoices-FSC |
| 2026-10-07 | [`da0c2ad`](https://github.com/minhaz-42/Voices-Speech-Recognition-445/commit/da0c2ade0b5ddc028064a7f874675c856b5a0396) | T02: validate phrase catalog |

</details>

---

Written by `python -m support.contributions` from the repository; do not edit by hand. Overview of the team: [README.md](README.md)

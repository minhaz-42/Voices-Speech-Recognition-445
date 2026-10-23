# Mahir Ahmed (M3)

Analysis plan, statistics, results table, hypothesis tests, error analysis

| | |
|---|---|
| **Models** | k-NN · wav2vec 2.0 3 epochs · HuBERT Base 15 epochs + noise |
| **GitHub** | – |
| **Branch** | `dev_mahir` |
| **Working page** | [docs/team/M3_mahir.md](../docs/team/M3_mahir.md) (task cards, weekly faculty update, handoff log) |
| **In numbers** | 0 tasks done, 0 in progress · 0 commits · 0 runs with results · 0 files created |

## Tasks

From the task board [docs/STATUS.md](../docs/STATUS.md): ✅ done and merged · 🟡 in progress · ⬜ to do.

| ID | Task | Status | Done by · date |
|---|---|---|---|
| T10 | `docs/ANALYSIS_PLAN.md`: the full analysis, consistent with the *Frozen analysis decisions* in PROJECT_PLAN.md (it no longer gates any run) | ⬜ |  |
| T33 | `support/stats.py`: bootstrap CI, McNemar, Wilcoxon, Mann–Whitney + hand-computed tests | ⬜ |  |
| T37 | Literature notes: MFCC, classical classifiers, speaker-independent evaluation | ⬜ |  |
| T13 | **k-NN** with the shared runner → `results/knn/fsc/` | ⬜ |  |
| T17 | **wav2vec 2.0, 3 epochs** (no noise) → `results/w2v2_3ep/fsc/` | ⬜ |  |
| T45 | **HuBERT Base, 15 epochs + white noise** (same script, `--model facebook/hubert-base-ls960`) → `results/hubert_15ep_noise/fsc/` | ⬜ |  |
| T20 | `support/results_table.py`: every run in `results/` → `results/all_results.csv` + table image; seen vs unseen speakers, per speaker, bootstrap CIs; tested on synthetic and your own runs | ⬜ |  |
| T34 | `support/error_analysis.py`: most-confused phrase pairs, accuracy per phrase category and speaker, for any run; write-up for your own runs → `results/errors/` | ⬜ |  |
| T40 | `support/hypotheses.py`: H1–H5 exactly as frozen, from whatever runs exist → `results/hypotheses.csv`; tested on synthetic results (M1 runs it on all results in T24) | ⬜ |  |

## Results

*No results merged yet.*

## Files

*No files created by Mahir yet.*

## Commits

*Mahir's git user name is not known yet, so no commits are listed.*

---

Written by `python -m support.contributions` from the repository; do not edit by hand. Overview of the team: [README.md](README.md)

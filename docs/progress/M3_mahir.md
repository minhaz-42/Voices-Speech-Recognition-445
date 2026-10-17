# M3 · Mahir Ahmed: Progress

| | |
|---|---|
| **Role** | Evaluation code: analysis plan, statistics, results table, hypothesis tests, error analysis |
| **Models** | k-NN (traditional) · wav2vec 2.0 3 epochs · **HuBERT Base** 15 epochs + white noise |
| **Branch** | `dev_mahir` (Tanvir merges it into `main`) |
| **Owns** | `support/metrics.py`, `support/stats.py`, `support/results_table.py`, `support/error_analysis.py`, `support/hypotheses.py`, `docs/ANALYSIS_PLAN.md`, `docs/literature/classical_and_evaluation.md`, `results/knn/`, `results/w2v2_3ep/`, `results/hubert_15ep_noise/`, `results/all_results.csv`, `results/hypotheses.csv`, `results/errors/` |
| **Report** | Sections 5 (experiments and results) and 6 (discussion) |

Only Mahir (or Mahir's assistant) edits this file. Full task text and dependencies: [STATUS.md](../STATUS.md).

## Note from Tanvir (2026-10-17, evening): nothing to wait for

Every one of your tasks below depends only on work that is already merged or on your own earlier tasks. Data preparation (download, selection, QC, splits) is done and merged for TenVoices-FSC, so you can run your models today: [docs/tasks/RUNS.md](../tasks/RUNS.md) shows how. Work that needs everyone's results is split: you write and test the code, and Tanvir runs it on all results at the end.

## First-time setup

1. Ask Tanvir for a GitHub collaborator invite and accept it.
2. `git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git`, then `cd Voices-Speech-Recognition-445`.
3. `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
4. Set your own identity: `git config user.name "<your name>"` and `git config user.email "<your GitHub email>"`.
5. Work by hand or with any AI assistant; [CONTRIBUTING.md](../../CONTRIBUTING.md) has the steps and a ready-made prompt, and each task has a card in [docs/tasks/](../tasks/). With an assistant that works in the repository (Claude Code, Codex, Cursor, …), just say: *"I'm Mahir (M3). What is completed? Then complete my next tasks."*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T03 | Metrics module `support/metrics.py` (done by Tanvir before you joined; you own it now) | — | ✅ | Tanvir · 2026-10-04 |
| T10 | Analysis plan `docs/ANALYSIS_PLAN.md`, consistent with the frozen decisions ([card](../tasks/T10.md)) | — | ⬜ | |
| T33 | Statistics module `support/stats.py` ([card](../tasks/T33.md)) | — | ⬜ | |
| T37 | Literature notes ([card](../tasks/T37.md)) | — | ⬜ | |
| T13 | k-NN → `results/knn/fsc/` ([card](../tasks/T13.md)) | — | ⬜ | |
| T17 | wav2vec 2.0, 3 epochs → `results/w2v2_3ep/fsc/` ([card](../tasks/T17.md)) | — | ⬜ | |
| T45 | HuBERT Base, 15 epochs + white noise → `results/hubert_15ep_noise/fsc/` ([card](../tasks/T45.md)) | — | ⬜ | |
| T20 | Results table `support/results_table.py` ([card](../tasks/T20.md)) | T33 | ⬜ | |
| T34 | Error analysis `support/error_analysis.py` on your runs ([card](../tasks/T34.md)) | T13 | ⬜ | |
| T40 | Hypothesis tests `support/hypotheses.py` ([card](../tasks/T40.md)) | T10, T20, T33 | ⬜ | |

**Start with:** T33, T13 (minutes on a laptop) and T10. Run T17 and T45 on Colab. Your code is tested on synthetic results and on your own runs; Tanvir runs it on everyone's results at the end, so nothing you do waits for anyone.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

*(Nothing yet.)*

## Requests for the lead

*(None.)*

## Handoff log (newest first)

*(Nothing yet.)*

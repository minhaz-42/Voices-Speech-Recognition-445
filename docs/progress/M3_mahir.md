# M3 · Mahir Ahmed: Progress

| | |
|---|---|
| **Role** | Evaluation: analysis plan, statistics, results tables, error analysis |
| **Models** | k-NN (traditional) · wav2vec 2.0 fine-tuned for 3 epochs |
| **Branch** | `dev_mahir` (Tanvir merges it into `main`) |
| **Owns** | `support/metrics.py`, `support/stats.py`, `support/results_table.py`, `support/error_analysis.py`, `docs/ANALYSIS_PLAN.md`, `docs/literature/classical_and_evaluation.md`, `results/knn/`, `results/w2v2_3ep/`, `results/all_results.csv`, `results/hypotheses.csv`, `results/errors/` |
| **Report** | Sections 5 (experiments and results) and 6 (discussion) |

Only Mahir (or Mahir's assistant) edits this file. Full task text and dependencies: [STATUS.md](../STATUS.md).

## First-time setup

1. Ask Tanvir for a GitHub collaborator invite and accept it.
2. `git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git`, then `cd Voices-Speech-Recognition-445`.
3. `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
4. Set your own identity: `git config user.name "<your name>"` and `git config user.email "<your GitHub email>"`.
5. Work by hand or with any AI assistant; [CONTRIBUTING.md](../../CONTRIBUTING.md) has the steps and a ready-made prompt, and each task has a card in [docs/tasks/](../tasks/). With an assistant that works in the repository (Claude Code, Codex, Cursor, …), just say: *"I'm Mahir (M3). What is completed? Then complete my next tasks."*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T03 | Metrics module `support/metrics.py` (done by Tanvir before you joined; you own it now) | T01 | ✅ | Tanvir · 2026-10-04 |
| T10 | Analysis plan `docs/ANALYSIS_PLAN.md` (H1–H5, splits, metrics, tests) | T03 | ⬜ | |
| T33 | Statistics module `support/stats.py` (bootstrap CI, McNemar, Wilcoxon) + tests | T03 | ⬜ | |
| T37 | Literature notes: MFCC, classical classifiers, speaker-independent evaluation | — | ⬜ | |
| T31 | 👤 Record S03 (self) and volunteers S07–S08 | T05 | ⬜ | |
| T07 | 👤 Listen-check S04, S09–S10 (part of M2's QC task) | T32 | ⬜ | |
| T13 | k-NN results → `results/knn/` | T11 | ⬜ | |
| T20 | Results table `support/results_table.py` | T11, T33 | ⬜ | |
| T17 | wav2vec 2.0, 3 epochs → `results/w2v2_3ep/` | T15 | ⬜ | |
| T40 | Final results, hypotheses H1, H2, H4 | all runs, T20 | ⬜ | |
| T34 | Error analysis `support/error_analysis.py` | T40 | ⬜ | |

**This week (from 2026-10-17):** T10, T33 and T37 need no recordings and no one else's work. For the faculty update you can then show how every model will be judged (the analysis plan and hypotheses), the statistics module with its tests, and what the literature says about classical speech classifiers.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

*(Nothing yet.)*

## Requests for the lead

*(None.)*

## Handoff log (newest first)

*(Nothing yet.)*

# M4 · Shafil Ahmed: Progress

| | |
|---|---|
| **Role** | Visualisation, demo app, poster, slides, video |
| **Models** | SVM (traditional) · untrained wav2vec 2.0 baseline · **linear probe** on frozen wav2vec 2.0 |
| **Branch** | `dev_shafil` (Tanvir merges it into `main`) |
| **Owns** | `support/visualization.py`, `support/embeddings.py`, `app.py`, `poster/`, `images/poster.png`, slides and video in `others/`, `docs/literature/self_supervised_speech.md`, `results/svm/`, `results/w2v2_untrained/`, `results/w2v2_probe/` |
| **Report** | Section 7 (demo system) |

Only Shafil (or Shafil's assistant) edits this file. Full task text and dependencies: [STATUS.md](../STATUS.md).

## Note from Tanvir (2026-10-17, evening): nothing to wait for

Every one of your tasks below depends only on work that is already merged or on your own earlier tasks. Data preparation (download, selection, QC, splits) is done and merged for TenVoices-FSC, so you can run your models today: [docs/tasks/RUNS.md](../tasks/RUNS.md) shows how. Work that needs everyone's results is split: you write and test the code, and Tanvir runs it on all results at the end.

## First-time setup

1. You already have write access on GitHub (`Shafil10`), and the branch `dev_shafil` exists.
2. `git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git`, then `cd Voices-Speech-Recognition-445`.
3. `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
4. Set your own identity: `git config user.name "<your name>"` and `git config user.email "<your GitHub email>"`.
5. Work by hand or with any AI assistant; [CONTRIBUTING.md](../../CONTRIBUTING.md) has the steps and a ready-made prompt, and each task has a card in [docs/tasks/](../tasks/). With an assistant that works in the repository (Claude Code, Codex, Cursor, …), just say: *"I'm Shafil (M4). What is completed? Then complete my next tasks."*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T35 | Visualisation module + tests + example figures ([card](../tasks/T35.md)) | — | ⬜ | |
| T36 | Demo skeleton `app.py` ([card](../tasks/T36.md)) | T35 | ⬜ | |
| T38 | Literature notes ([card](../tasks/T38.md)) | — | ⬜ | |
| T39 | Planning poster v2 ([card](../tasks/T39.md)) | — | ⬜ | |
| T14 | SVM → `results/svm/fsc/` ([card](../tasks/T14.md)) | — | ⬜ | |
| T18 | Untrained wav2vec 2.0 baseline → `results/w2v2_untrained/fsc/` ([card](../tasks/T18.md)) | — | ⬜ | |
| T46 | Linear probe: `support/embeddings.py` + logistic regression → `results/w2v2_probe/fsc/` ([card](../tasks/T46.md)) | — | ⬜ | |
| T23 | Demo recognising with your SVM + figures command ([card](../tasks/T23.md)) | T14, T35, T36 | ⬜ | |
| T27 | Results poster, slides, 👤 demo video ([card](../tasks/T27.md)) | T23 | ⬜ | |

**Start with:** T14 (minutes on a laptop), T35 and T18. Your demo uses your own SVM model, so nothing you do waits for anyone.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

*(Nothing yet.)*

## Requests for the lead

*(None.)*

## Handoff log (newest first)

*(Nothing yet.)*

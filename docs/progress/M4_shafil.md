# M4 · Shafil Ahmed: Progress

| | |
|---|---|
| **Role** | Visualisation, demo app, poster, slides, video |
| **Models** | SVM (traditional) · untrained wav2vec 2.0 baseline |
| **Branch** | `dev_shafil` (Tanvir merges it into `main`) |
| **Owns** | `support/visualization.py`, `app.py`, `poster/`, `images/poster.png`, slides and video in `others/`, `docs/literature/self_supervised_speech.md`, `results/svm/`, `results/w2v2_untrained/` |
| **Report** | Section 7 (demo system) |

Only Shafil (or Shafil's assistant) edits this file. Full task text and dependencies: [STATUS.md](../STATUS.md).

## First-time setup

1. You already have write access on GitHub (`Shafil10`), and the branch `dev_shafil` exists.
2. `git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git`, then `cd Voices-Speech-Recognition-445`.
3. `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
4. Set your own identity: `git config user.name "<your name>"` and `git config user.email "<your GitHub email>"`.
5. Work by hand or with any AI assistant; [CONTRIBUTING.md](../../CONTRIBUTING.md) has the steps and a ready-made prompt, and each task has a card in [docs/tasks/](../tasks/). With an assistant that works in the repository (Claude Code, Codex, Cursor, …), just say: *"I'm Shafil (M4). What is completed? Then complete my next tasks."*

## My tasks

| ID | Task | Depends on | Status | Done · date |
|---|---|---|---|---|
| T35 | Visualisation module `support/visualization.py` + tests + example figures of your voice | T01 | ⬜ | |
| T36 | Demo skeleton `app.py` (Gradio, no model yet) | T35 | ⬜ | |
| T38 | Literature notes: self-supervised speech models, spoken-command benchmarks | — | ⬜ | |
| T39 | Planning poster v2: team of four, the two datasets, four-stage pipeline | — | ⬜ | |
| T14 | SVM results on both datasets → `results/svm/` | T11 | ⬜ | |
| T18 | Untrained wav2vec 2.0 baseline → `results/w2v2_untrained/` | T15 | ⬜ | |
| T23 | Demo connected to `main.py` + final figures | T21, T22, T30, T36, T40 | ⬜ | |
| T27 | Results poster, slides, 👤 1-minute demo video | T24 | ⬜ | |

**Note from Tanvir (2026-10-17):** we no longer record our own speakers, so your recording and listen-check tasks (T32, T07) are cancelled. TenVoices is now two 10-speaker subsets of published datasets, TenVoices-FSC (spoken commands) and TenVoices-L2 (non-native read speech) ([DATASETS.md](../DATASETS.md)). The demo (T36) will recognise the 50 TenVoices-FSC commands, and the poster (T39) must show the two datasets; both cards are updated.

**This week (from 2026-10-17):** T35, T36, T38 and T39 need no audio and no one else's work. For the faculty update you can then show spectrograms and MFCCs of real speech (what the models actually "see"), the first version of the demo app, a literature summary and the updated poster.

## Faculty update

Plain-language summary for the weekly project update, newest week first. Add to it at the end of every session: only finished work, and numbers only from `results/`.

*(Nothing yet.)*

## Requests for the lead

*(None.)*

## Handoff log (newest first)

*(Nothing yet.)*

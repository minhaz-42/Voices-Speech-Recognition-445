# CLAUDE.md

Guide for Claude Code in the **TenVoices** repository. Every team member works through Claude, one after another. Follow this file in every session.

## Project

TenVoices is a university project (North South University). We build a **speech recognition system** and evaluate it on **TenVoices**, a dataset we record ourselves: 10 speakers, ~1,200 English utterances.

We compare three families of methods:
1. Traditional ML on a 15-command task: MFCC + k-NN / SVM / Random Forest / XGBoost, and DTW.
2. Zero-shot pre-trained ASR: Vosk, wav2vec 2.0, Whisper.
3. Our approach: Whisper-small + noise-augmented LoRA + Bangladeshi-lexicon decoding.

All results use leave-2-speakers-out 5-fold cross-validation.

| File | Use |
|---|---|
| `docs/STATUS.md` | **Task board + handoff log: the source of truth for progress** |
| `docs/PROJECT_PLAN.md` | Design, evaluation protocol, roles, risks |
| `docs/DATA_COLLECTION_PROTOCOL.md` | Recording rules, file naming, metadata, QC |
| `README.md` | Public overview |

## Team

| Slot | Member | git user.name | Workstream |
|---|---|---|---|
| M1 | Tanvir Ahmed (lead) | minhaz | Setup, data pipeline, ASR models and our approach, integration |
| M2 | Md. Shahriar Rakib Rabbi | *(add)* | Dataset collection and curation, traditional ML baselines, augmentation, dataset release |
| M3 | Tanvir Ahmed *(open slot)* | minhaz | Evaluation: metrics, experiments, statistics, error analysis |
| M4 | Tanvir Ahmed *(open slot)* | minhaz | Demo app, figures, poster, slides, video |

- One person can hold several slots. "My tasks" means the tasks of every slot that person holds.
- When someone joins, update the team tables in this file, `README.md` and `docs/STATUS.md`. The open tasks of that slot then belong to the new member.

## Start of every session

1. Run `git status`. If there are uncommitted changes, show them and ask what to do.
2. Run `git pull --rebase origin main`.
3. Read `docs/STATUS.md`.
4. Identify the member: use what they said, otherwise match `git config user.name` to the team table and confirm.
5. Report briefly:
   - what is done (who, when),
   - what is in progress or blocked,
   - this member's next tasks and whether their dependencies are ✅.
6. When asked to continue or complete their part:
   - Take their next task whose dependencies are all ✅, mark it 🟡, and do it.
   - Continue task by task.
   - Never start a task whose dependencies are not ✅. Say which task and which member it is waiting for, and offer any unblocked task instead.

## While working

- Stay inside the task's scope and meet its *done when* condition. Follow `docs/PROJECT_PLAN.md`.
- 👤 tasks need people (recording speakers, consent, filming the demo video). Prepare everything (scripts, sheets, checklists), ask the member to do it, and wait for them to confirm.
- **Never invent data or numbers.** Results stay TBD until measured; every reported number must come from a file in `results/`.
- Audio is not in git. Raw recordings live in the team's private Google Drive folder `TenVoices/raw` (link in the team chat). If a task needs audio that is missing locally, tell the member which Drive folder to download into which local path (e.g. `data/raw/S03/`), then wait.
- Code rules:
  - Python 3.11, settings in `configs/config.yaml`, relative paths, CLI arguments.
  - Docstrings; `unittest` tests in `tests/`.
  - One shared module per function; no personal copies of scripts.
- After the evaluation protocol is frozen (task T10), do not change folds, metrics or hypotheses without M1's approval and a note in `docs/STATUS.md`.
- Don't mention other or earlier projects in any document.

## End of every session (whenever files changed)

1. Run the tests: `python -m unittest discover -s tests -v`. Fix failures before committing.
2. Update `docs/STATUS.md`:
   - task status, plus *Done by · date*,
   - the **Current phase / Next up** lines,
   - a new **Handoff Log** entry at the top: tasks done, files changed, key results, what the next member should do, blockers.
3. Commit only the files you changed:
   - one task per commit, message `T07: <summary>`,
   - with the member's own git identity, and no Co-Authored-By trailer.
4. Run `git pull --rebase origin main`, then `git push origin main`. We work directly on `main`. Resolve conflicts in `docs/STATUS.md` by keeping both sides' entries.
5. Tell the member, in one line, what to tell the next person.

## Never commit

- Audio (`*.wav`, `*.flac`, `*.mp3`, …), model weights or LoRA adapters
- Signed consent forms, volunteer names, phone numbers or Drive links
- Large generated files in `outputs/`

Small result tables (CSV/JSON under 1 MB) go in `results/` and **are** committed, so the next member can continue from them. Figures go in `images/`.

## Commands

- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Tests: `python -m unittest discover -s tests -v`
- *(Add new commands here when a task creates them.)*

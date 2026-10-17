# AGENTS.md

Rules for everyone working on the **TenVoices** repository: team members working by hand, and any AI coding assistant they use (Claude Code, Codex, Cursor, GitHub Copilot, Gemini, ChatGPT, …). Every member works on their **own branch**, and the lead (M1) merges the branches into `main`.

- Step-by-step commands for people, and for chat assistants that cannot run commands: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
- **Each member's work is on one page**, `docs/team/<slot>_<name>.md`: role, files, models, task status, faculty update, handoff log, and at the end the task cards (files, functions, tests, *done when*). Index: [docs/team/](docs/team/).

## Project

TenVoices is a university project (North South University). We build a **speech recognition system** that recognises which of **50 phrases** was spoken, and evaluate it on **TenVoices**: two 10-speaker subsets of published datasets, selected by script. **TenVoices-FSC** (main, from Fluent Speech Commands): 10 speakers × 50 spoken commands × 2 takes. **TenVoices-L2** (accents, from L2-ARCTIC): 10 non-native speakers × 50 read sentences. We do **not** record our own speakers; sources, licences and selection rules are in `docs/DATASETS.md`.

We compare four stages:
1. Traditional ML: MFCC features + k-NN / SVM / Random Forest / XGBoost.
2. Baseline: wav2vec 2.0 with an untrained classification head.
3. Our approach: wav2vec 2.0 fine-tuned for 3 / 8 / 15 epochs with white-noise augmentation.
4. Open-vocabulary pipeline: Whisper speech-to-text, zero-shot; the transcript is matched to the nearest phrase.

Every model is reported on both datasets, on a random 80/20 split and with speaker-independent 5-fold cross-validation (the headline). Metrics: accuracy, macro F1, weighted F1 and confusion matrices, plus word error rate for Whisper.

Repository: https://github.com/minhaz-42/Voices-Speech-Recognition-445

| File | Use |
|---|---|
| `docs/STATUS.md` | **Task board + merge log: the source of truth for what is merged** (only M1 edits it) |
| `docs/team/<slot>_<name>.md` | **One page per member.** The member edits its status table, faculty update, requests and handoff log; M1 writes the *Task cards* section at the end (what to build, which files, which functions, tests, *done when*) |
| `docs/PROJECT_PLAN.md` | Design, evaluation protocol, roles, **file ownership**, risks |
| `docs/DATASETS.md` | Dataset sources, licences, download, subset selection rules, QC |
| `docs/RUNS.md` | How to get the audio and run any model, locally or on Colab |
| `docs/CONTRIBUTING.md` | The same workflow as plain git commands, for working by hand or with a chat assistant |
| `README.md` | Public overview: results and who did what |

## Team

| Slot | Member | git user.name | GitHub | Branch | Page | Workstream |
|---|---|---|---|---|---|---|
| M1 | Tanvir Ahmed (lead, merges) | minhaz | minhaz-42 | `main` | `docs/team/M1_tanvir.md` | Data preparation, shared code, **Whisper pipeline**, noise tests, inference, TenVoices-L2, final aggregation · Random Forest · wav2vec 2.0 15 epochs (+ noise, and without) |
| M2 | Md. Shahriar Rakib Rabbi | Md. Shahriar Rakib Rabbi | ShahriarRakibRabbi | `dev_rabbi` | `docs/team/M2_rabbi.md` | Datasheet, listen-check, dataset statistics, subset release · XGBoost · wav2vec 2.0 8 epochs + noise · WavLM Base+ |
| M3 | Mahir Ahmed | *(add)* | *(add)* | `dev_mahir` | `docs/team/M3_mahir.md` | Analysis plan, statistics, results table, hypothesis tests, error analysis · k-NN · wav2vec 2.0 3 epochs · HuBERT Base |
| M4 | Shafil Ahmed | *(add)* | Shafil10 | `dev_shafil` | `docs/team/M4_shafil.md` | Visualisation, demo app, poster, slides, video · SVM · untrained baseline · linear probe |

- With an assistant that can run commands, a member only has to say who they are (*"I'm Mahir, do my part"*); the assistant follows the steps below. Without one, the member follows [CONTRIBUTING.md](docs/CONTRIBUTING.md). The result is the same either way.
- When a git user.name or GitHub login marked *(add)* becomes known, M1 fills it in here.

## Start of every session

1. Run `git status`. If there are uncommitted changes, show them and ask what to do.
2. Identify the member: use what they said, otherwise match `git config user.name` to the team table and confirm.
3. Go to the member's branch and bring in everything already merged:
   - **M1:** `git checkout main && git pull --rebase origin main`.
   - **M2–M4:** `git fetch origin`, then `git checkout <branch>`. If the branch does not exist locally, use `git checkout -b <branch> origin/<branch>` when it is on GitHub, otherwise `git checkout -b <branch> origin/main`. Then `git pull origin <branch>` (when it is on GitHub) and `git merge origin/main`. Never rebase or force-push a branch that is on GitHub.
4. Read `docs/STATUS.md` and the member's page `docs/team/<slot>_<name>.md` (their task cards are at the end of it).
5. Report briefly:
   - what is done and merged (who, when),
   - what is in progress or blocked,
   - this member's next tasks and whether their dependencies are ✅.
6. When asked to continue or complete their part:
   - Take their next task whose dependencies are all ✅, mark it 🟡 (members: on their page; M1: on the board), and do it as its task card says.
   - A dependency is ✅ when the board on `main` says so, i.e. it is merged. A dependency owned by the **same member** also counts once their own page marks it ✅.
   - Continue task by task.
   - Never start a task whose dependencies are not ✅. Say which task and which member it is waiting for, and offer any unblocked task instead.

## While working

- Stay inside the task's scope and meet its *done when* condition. Follow the task card and `docs/PROJECT_PLAN.md`.
- **File ownership** (table in `docs/PROJECT_PLAN.md`, section *Team, Roles and File Ownership*). This is what keeps every branch free of merge conflicts.
  - A member edits only the files their slot owns, their own tests (`tests/test_<their module>.py`), their own run folders in `results/`, their own figures in `images/`, and the status, faculty update, requests and handoff sections of their own page in `docs/team/`.
  - Only M1 edits `docs/STATUS.md`, `AGENTS.md`, `CLAUDE.md`, `docs/CONTRIBUTING.md`, `docs/RUNS.md`, `README.md`, `docs/PROJECT_PLAN.md`, the *Task cards* sections of the pages in `docs/team/`, `requirements.txt`, `configs/config.yaml` and the M1-owned modules.
  - If a member needs a change in a file they don't own (a new package, a config key, a fix in shared code), write it under **Requests for the lead** on their page and keep working with CLI options or function arguments meanwhile.
- 👤 tasks need people (requesting dataset access, filming the demo video). Prepare everything (scripts, sheets, checklists), ask the member to do it, and wait for them to confirm.
- **Never invent data or numbers.** Results stay TBD until measured; every reported number must come from a file in `results/`.
- **Never invent references.** Every paper in a literature note has a DOI or arXiv link that was opened and checked.
- Audio is not in git. The original dataset archives live in the team's private Google Drive folder `TenVoices/raw` (link in the team chat). If a task needs audio that is missing locally, tell the member which Drive folder to download and extract into which local path (`data/raw/fsc/` or `data/raw/l2arctic/`), then wait.
- The datasets' licences are non-commercial (Fluent Speech Commands also no-derivatives): never publish their audio or processed copies anywhere, including Hugging Face or a GitHub Release.
- Code rules:
  - Python 3.11, settings in `configs/config.yaml`, relative paths, CLI arguments.
  - Docstrings; `unittest` tests in `tests/`.
  - One shared module per function; no personal copies of scripts. A new run is a new set of CLI options (e.g. `--epochs 8 --noise white`), not a new script.
- After the evaluation protocol is frozen (task T10), do not change folds, metrics or hypotheses without M1's approval and a note in `docs/STATUS.md`.
- Don't mention other or earlier projects in any document.

## End of every session (whenever files changed)

### Members (M2–M4), on their own branch

1. Run the tests: `python -m unittest discover -s tests -v`. Fix failures before committing.
2. Update the member's page (`docs/team/<slot>_<name>.md`):
   - task status (✅ + date for finished tasks),
   - a new **Handoff log** entry at the top: tasks done and ready to merge, files changed, key results, next steps, blockers,
   - the **Faculty update** for the current week: 3–5 plain-language bullets on what the member built or did, which files, and what it shows. Only finished work; numbers only from `results/`.
3. Commit only the files you changed: one task per commit, message `T33: <summary>`, with the member's own git identity. **Unfinished work is committed and pushed too**, as `T33: work in progress`, so Tanvir can always see the latest code; nothing stays only on a laptop.
4. `git push origin <branch>`. If no pull request from this branch is open, open one into `main` with `gh pr create --base main --head <branch>`, titled like `M3 · Mahir: T10, T33`, with the handoff entry as the body; add `--draft` while a task in it is unfinished, and run `gh pr ready` once all its tasks are done. Otherwise the push updates the open one. If `gh` is not available, give the member this link instead: `https://github.com/minhaz-42/Voices-Speech-Recognition-445/compare/main...<branch>`.
5. Tell the member, in one line, what to tell Tanvir (e.g. *"dev_mahir: T10 and T33 are ready to merge"*).

### M1 (lead), on `main`

1. Run the tests.
2. Update `docs/STATUS.md` (task status + *Done by · date*, **Current phase / Next up**, a new **Handoff Log** entry at the top) and the **Faculty update** in `docs/team/M1_tanvir.md`.
3. Commit only the files you changed: one task per commit, message `T29: <summary>`.
4. `git pull --rebase origin main`, then `git push origin main`. Push after every finished task.
5. Tell the member, in one line, what to tell the next person.

### Merging a member's branch (M1, when told it is ready)

Only merge a branch whose pull request is ready (not a draft) and whose tasks are ✅ on the member's page; draft pull requests are there so M1 can read work in progress.

1. `git fetch origin`, then review `git diff main...origin/<branch>`:
   - only files the member owns (plus their tests, results, figures and page) changed,
   - no audio, model weights, personal data or files over 1 MB,
   - the member's page marks the tasks ✅.
2. `git merge --no-ff --no-commit origin/<branch>`. If a file the member does not own changed, ask M1 before keeping it.
3. Run the tests on the merged result. If they fail, `git merge --abort` and tell the member what to fix.
4. Copy the member's finished tasks onto the board in `docs/STATUS.md` (✅, *Done by <name> · date*), update **Next up**, and add a Handoff Log entry *"Merged <branch>: T10, T33"*.
5. `git add docs/STATUS.md`, `git commit -m "Merge <branch>: T10, T33"`, `git push origin main`. If a pull request is open for the branch, GitHub closes it as merged.

## Authorship

- Every commit uses the member's own git identity.
- Never name an AI tool as an author, co-author or contributor: no `Co-Authored-By` trailer, no "Generated with …" line in commits or pull requests, and no AI tool's name in commit messages or pull-request titles. Remove such lines if a tool adds them.

## Never commit

- Audio (`*.wav`, `*.flac`, `*.mp3`, …), model weights or checkpoints
- Personal data or Drive links
- Large generated files in `outputs/`

Small result tables (CSV/JSON under 1 MB) go in `results/` and **are** committed, so others can continue from them. Figures go in `images/`.

## Commands

- Setup: `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Tests: `python -m unittest discover -s tests -v`
- Run scripts in `support/` as modules from the repo root: `python -m support.audio clip.wav`, not `python support/audio.py`.
- Print settings: `python -m support.config`
- Pre-process one clip: `python -m support.audio clip.wav [--out clip_16k.wav] [--no-pad]`
- Whisper on clips: `python -m support.whisper_asr clip.wav [...]`; on a clip list: `python -m support.whisper_asr --list clips.csv --out <folder>` (columns `utt_id, path, phrase_id[, speaker_id]`)
- Run any model on TenVoices-FSC (get the audio, build, run, commit): [docs/RUNS.md](docs/RUNS.md)
- Data preparation (M1): `python -m support.select_subsets --dataset fsc` → `python -m support.qc_report --dataset fsc` → `python -m support.dataset --dataset fsc --build` → `python -m support.make_splits --dataset fsc`
- Other self-supervised models: `python -m support.train_wav2vec2 --dataset fsc --model microsoft/wavlm-base-plus --epochs 15 --noise white` (run name from the model) · linear probe: `python -m support.classical --model logreg --features w2v2 --run w2v2_probe --dataset fsc`
- Traditional model: `python -m support.classical --dataset fsc --model random_forest [--split all|folds|random|foldN]`
- wav2vec 2.0: `python -m support.train_wav2vec2 --dataset fsc --epochs 15 --noise white [--split …] [--limit 40]`; baseline: `--untrained`; Colab: `notebooks/train_wav2vec2.ipynb`
- Whisper on the splits: `python -m support.whisper_asr --dataset l2arctic --split all`
- Rebuild a run's metrics: `python -m support.evaluate --dataset fsc --run random_forest` · noise test: `python -m support.noise_test --run w2v2_15ep_noise --dataset fsc`
- Inference: `python main.py --audio clip.wav --model outputs/models/<run>/<dataset>/<split>[.joblib]` (also `--video`, `--mic 4`, `--asr whisper`, `--benchmark 50`)
- On macOS, never use PyTorch (wav2vec 2.0, Whisper) and XGBoost in the same Python process: their OpenMP libraries clash (crash or hang). The scripts above keep them apart; tests that need PyTorch run in a child process.
- Review a member's branch (M1): `git fetch origin && git diff --stat main...origin/dev_mahir`
- *(Add new commands here when a task creates them.)*

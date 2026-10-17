# Contributing to TenVoices

How to do your tasks **by hand or with any AI assistant**. The rules are in [AGENTS.md](../AGENTS.md); this page gives the exact steps. Examples use Mahir (M3, branch `dev_mahir`); replace them with your own slot, name and branch:

| Slot | Member | Branch | Your page |
|---|---|---|---|
| M2 | Rabbi | `dev_rabbi` | `docs/team/M2_rabbi.md` |
| M3 | Mahir | `dev_mahir` | `docs/team/M3_mahir.md` |
| M4 | Shafil | `dev_shafil` | `docs/team/M4_shafil.md` |

## 1. One-time setup

You need write access to the GitHub repository; ask Tanvir for an invite.

```bash
git clone https://github.com/minhaz-42/Voices-Speech-Recognition-445.git
cd Voices-Speech-Recognition-445
python3.11 -m venv venv
source venv/bin/activate              # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
git config user.name "Your Name"
git config user.email "you@example.com"   # the email of your GitHub account
python -m unittest discover -s tests -v   # everything should pass
```

## 2. Start of every work session

```bash
git fetch origin
git checkout dev_mahir      # first time: git checkout -b dev_mahir origin/dev_mahir
                            # (or origin/main if the branch is not on GitHub yet)
git pull origin dev_mahir   # skip if the branch is not on GitHub yet
git merge origin/main       # brings in everything Tanvir has merged
```

Then read:
- [docs/STATUS.md](STATUS.md): what is merged and what each task depends on,
- your page `docs/team/<slot>_<name>.md`: your tasks and their status at the top, and the task card of your next task in the *Task cards* section at the end.

To run a model, [docs/RUNS.md](RUNS.md) shows how to get the audio and run it, locally or on Colab.

Take your next task whose dependencies are ✅ (merged into `main`, or your own task that you already finished). Mark it 🟡 on your page.

## 3. Do the task

Whatever tool you use, the task card tells you which files to create, which existing functions to use, what to test and when you are done. **Only edit the files the card lists.**

**A. An assistant that works inside the repository** (Claude Code, Codex, Cursor, GitHub Copilot, Gemini CLI, …). Most of them read `AGENTS.md` automatically; if yours does not, tell it to read it first. Then say:

> I'm Mahir (M3). Do task T33 following AGENTS.md and the T33 card on my page, docs/team/M3_mahir.md.

**B. A chat assistant** (ChatGPT, Gemini, Claude.ai, Copilot chat, …). It cannot see your files, so give it the context. Start a new chat with this prompt, and paste (or attach) the files listed under **Give your assistant** in the task card:

```text
I'm working on TenVoices, a university speech-recognition project.
Follow the project rules (AGENTS.md) and the task card below exactly.
- Write complete files that I can save as they are, at the paths the card gives.
- Python 3.11, docstrings, unittest tests; use only packages in requirements.txt.
- Use the existing functions the card lists instead of rewriting them.
- Never invent data, numbers or references. If you are unsure about a
  reference, say so instead of guessing.

=== AGENTS.md ===
(paste AGENTS.md)

=== Task card ===
(paste the T33 card from the Task cards section of docs/team/M3_mahir.md)

=== Existing files ===
(paste each file listed under "Give your assistant")
```

Save what it writes into the right files, run the tests (step 4), and paste any error back into the chat until everything passes.

**C. By hand.** Follow the task card; the existing modules in `support/` show the code style.

Whichever way you choose: read and run what an assistant gives you, check every literature reference by opening its link, and remove any `Co-Authored-By` or "Generated with …" line a tool adds to your commits.

## 4. Check

```bash
python -m unittest discover -s tests -v
```

All tests must pass, not only yours.

## 5. Update your page

In `docs/team/<slot>_<name>.md`:
- **My tasks:** set the task to ✅ with the date.
- **Faculty update:** under `### Week of <date>`, 3–5 plain-language bullets: what you built or did, which files, and what it shows. Only finished work; numbers only from `results/`.
- **Handoff log:** a new entry at the top:

  ```markdown
  ### 2026-10-20 · T33
  - **Done (ready to merge):** what you finished
  - **Files:** the files you created or changed
  - **Results:** test count, figures, numbers from results/ (if any)
  - **Next:** your next task
  - **Blockers:** none, or what you are waiting for
  ```
- **Requests for the lead:** anything you need changed in a file you don't own.

## 6. Commit and push

```bash
git status                                   # only your own files should be listed
git add support/stats.py tests/test_stats.py docs/team/M3_mahir.md
git commit -m "T33: add bootstrap CI, McNemar and Wilcoxon tests"
git push origin dev_mahir                    # first push: git push -u origin dev_mahir
```

- One task per commit, message `T<ID>: <summary>`.
- **Push at the end of every session, even if the task is not finished**: commit it as `T33: work in progress`. Tanvir can then see your code at any time, and nothing is lost if your laptop is.
- Never commit audio (including the datasets' clips), model files or personal data.

## 7. Ask for a merge

Open the pull request after your **first** push, so Tanvir can follow your work: go to `https://github.com/minhaz-42/Voices-Speech-Recognition-445/compare/main...dev_mahir`, give it a title like `M3 · Mahir: T10, T33`, paste your handoff entry as the description, and choose **Create draft pull request** while the task is unfinished. Later pushes update it automatically. With the GitHub CLI: `gh pr create --draft --base main --head dev_mahir`.

When the task is done, click **Ready for review** (or `gh pr ready`) and tell Tanvir which tasks are ready. Once they are merged, step 2 of your next session brings them in.

## Troubleshooting

| Problem | What to do |
|---|---|
| `git merge origin/main` reports a conflict | You probably changed a file you don't own. Run `git merge --abort` and ask Tanvir. |
| A test fails in a module you don't own | Don't fix it yourself; write it under **Requests for the lead**. |
| `git push` is rejected | `git pull origin dev_mahir`, then push again. Never use `--force`. |
| A package is missing | Run `pip install -r requirements.txt`. If the task needs a new package, ask under **Requests for the lead**. |
| `ModuleNotFoundError: support` | Run commands from the repository root, scripts as modules: `python -m support.audio clip.wav`. |

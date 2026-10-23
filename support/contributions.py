"""One page per team member for the faculty: who did what, written from the repository (task T50).

Nothing on the pages is typed by hand, so they always match the repository:

| Section | Read from |
|---|---|
| Role, models, GitHub, branch | ``team`` in ``configs/config.yaml`` |
| Tasks and their status | the member's section of the task board ``docs/STATUS.md`` (plus rows owned or done by them elsewhere on the board) |
| Results | ``summary.json`` of each of the member's runs, found with ``support.evaluate.run_dir`` |
| Files created / edited | the git history: who added each file that still exists, and who changed it later |
| Commits | every commit (merges not counted) whose author is one of the member's ``git_names`` |

Output: ``contributions/README.md`` (overview), ``contributions/<slot>_<short>.md`` per member and, in
each member's own results folder, ``results/<slot>_<short>/RUNS.md`` with each of their runs in detail:
model, how it was trained (settings from ``summary.json``), data sizes, every split, accuracy per test
speaker and the confusion matrices. Run it again after every merge.

Usage::

    python -m support.contributions                      # history of main
    python -m support.contributions --ref HEAD --out contributions
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from support.config import PROJECT_ROOT, load_config, resolve_path
from support.evaluate import FOLD_NAMES, figures_root, run_dir
from support.figures import run_label

BOARD = "docs/STATUS.md"
MODEL_NAMES = {"svm": "SVM", "random_forest": "Random Forest", "xgboost": "XGBoost", "knn": "k-NN",
               "logreg": "logistic regression"}
COLLAPSED_GROUPS = ("Results", "Data and settings")
GROUP_ORDER = ["Code", "Tests", "Documents", "Figures", "Data and settings", "Results", "Other"]


def split_row(line: str) -> list[str]:
    """Cells of a Markdown table row (escaped ``\\|`` stays inside a cell)."""
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def read_tables(text: str) -> list[tuple[str, list[dict[str, str]]]]:
    """Every Markdown table of a document with the heading above it.

    Returns:
        ``(heading, rows)`` per table; each row maps the header cells to its cells.
    """
    tables: list[tuple[str, list[dict[str, str]]]] = []
    heading, header, rows = "", None, []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cells = split_row(line)
            if header is None:
                header = cells
            elif not all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                rows.append(dict(zip(header, cells)))
            continue
        if header is not None:
            tables.append((heading, rows))
            header, rows = None, []
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
    return tables


def _cell(row: dict[str, str], prefix: str) -> str:
    """The cell whose header starts with ``prefix`` (``""`` if there is none)."""
    return next((v for k, v in row.items() if k.startswith(prefix)), "")


def member_tasks(member: dict[str, Any], board: str) -> list[dict[str, str]]:
    """The member's tasks on the board: their own section, plus rows they own or did elsewhere.

    Returns:
        ``id, task, status, done`` per task, in board order. Rows of tables without a status column
        (the *Done* table) count as done.
    """
    own = f"{member['slot']} · {member['short']}"
    tasks = []
    for heading, rows in read_tables(board):
        for row in rows:
            task_id, done = _cell(row, "ID"), _cell(row, "Done")
            if not task_id.startswith("T"):
                continue
            mine = (heading == own or _cell(row, "Owner") == member["slot"]
                    or done.startswith(member["short"] + " "))
            if mine:
                status = _cell(row, "Status") or ("✅" if done else "⬜")
                tasks.append({"id": task_id, "task": _cell(row, "Task"), "status": status, "done": done})
    return tasks


def git(root: Path, *args: str) -> str:
    """Output of a git command run in ``root``."""
    return subprocess.run(["git", "-C", str(root), "-c", "core.quotePath=off", *args], check=True,
                          capture_output=True, text=True).stdout


def read_history(root: Path, ref: str = "main") -> list[dict[str, Any]]:
    """Every commit reachable from ``ref`` except merges, oldest first, with its file changes.

    Returns:
        ``hash, author, date, subject, changes`` per commit; ``changes`` holds
        ``(status letter, path, old path or None)``.
    """
    out = git(root, "log", ref, "--no-merges", "-M", "--reverse", "--date=short", "--name-status",
              "--format=%x1e%H%x1f%an%x1f%ad%x1f%s")
    commits = []
    for record in out.split("\x1e")[1:]:
        lines = record.strip("\n").split("\n")
        sha, author, date, subject = lines[0].split("\x1f")
        changes = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            letter = parts[0][0]
            changes.append((letter, parts[2], parts[1]) if letter in "RC" else (letter, parts[1], None))
        commits.append({"hash": sha, "author": author, "date": date, "subject": subject, "changes": changes})
    return commits


def attribute_files(history: list[dict[str, Any]], existing: set[str]) -> dict[str, dict[str, set[str]]]:
    """Who created each file that still exists, and who else changed it.

    A renamed file keeps its creator. Returns ``{author: {"created": paths, "edited": paths}}``.
    """
    creator: dict[str, str] = {}
    editors: dict[str, set[str]] = defaultdict(set)
    for commit in history:
        author = commit["author"]
        for letter, path, old in commit["changes"]:
            if letter == "A":
                creator[path] = author
            elif letter in "RC":
                creator[path] = creator.pop(old, author) if letter == "R" else creator.get(old, author)
                if letter == "R" and old in editors:
                    editors[path] |= editors.pop(old)
                editors[path].add(author)
            elif letter == "D":
                creator.pop(path, None)
                editors.pop(path, None)
            else:
                editors[path].add(author)
    files: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"created": set(), "edited": set()})
    for path in existing:
        if path in creator:
            files[creator[path]]["created"].add(path)
        for author in editors.get(path, set()) - {creator.get(path)}:
            files[author]["edited"].add(path)
    return files


def file_group(path: str) -> str:
    """Which list a file is shown in: code, tests, documents, figures, data, results or other."""
    suffix = Path(path).suffix.lower()
    if path.startswith("results/"):
        return "Results"
    if path.startswith("tests/"):
        return "Tests"
    if suffix in (".py", ".ipynb"):
        return "Code"
    if path.startswith("images/") or suffix in (".png", ".jpg", ".svg"):
        return "Figures"
    if path.startswith("docs/") or suffix in (".md", ".html", ".pdf", ".txt"):
        return "Documents"
    if path.startswith(("data/", "configs/")) or suffix in (".csv", ".yaml", ".yml", ".json"):
        return "Data and settings"
    return "Other"


def folder_of(path: str, datasets: list[str]) -> str:
    """The folder a result or data file is listed under: the run folder (cut before the dataset) or its parent."""
    parts = path.split("/")
    cut = next((i for i, p in enumerate(parts[:-1]) if p in datasets), len(parts) - 1)
    return "/".join(parts[:max(cut, 1)]) + "/"


def run_results(member: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Accuracy and macro F1 of each of the member's runs on each dataset that has a ``summary.json``."""
    rows = []
    for run in member.get("runs") or []:
        for dataset, dcfg in cfg["datasets"].items():
            path = run_dir(run, dataset, cfg) / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            pooled = (summary.get("folds") or {}).get("pooled") or {}
            random = summary.get("random") or {}
            rows.append({"run": run, "label": run_label(run), "dataset": dcfg.get("name", dataset),
                         "unseen_acc": pooled.get("accuracy"), "unseen_f1": pooled.get("macro_f1"),
                         "random_acc": random.get("accuracy"), "random_f1": random.get("macro_f1"),
                         "source": path})
    return rows


def describe_run(summary: dict[str, Any]) -> tuple[str, str]:
    """What a run is and how it was trained, in words, from its ``summary.json``.

    Returns:
        ``(model, training)``.
    """
    if summary.get("zero_shot"):
        return (f"`{summary['model']}`, zero-shot: it transcribes each clip and the transcript is matched to the "
                "nearest of the phrases", "No training on our data; only the test clips are transcribed.")
    if "feature_kind" in summary:
        name = MODEL_NAMES.get(summary["model"], summary["model"])
        feats = summary.get("features")
        if summary["feature_kind"] == "mfcc" and isinstance(feats, dict):
            features = (f"78 MFCC statistics per clip ({feats.get('n_mfcc')} MFCCs + deltas + delta-deltas, mean and "
                        f"standard deviation; FFT {feats.get('n_fft')}, hop {feats.get('hop_length')}, "
                        f"{feats.get('n_mels')} mel bands)")
        else:
            features = "the frozen wav2vec 2.0 Base features of each clip (mean-pooled, `support/embeddings.py`)"
        return (f"{name} on {features}", "Hyper-parameters chosen per split on the validation speakers "
                f"({summary.get('selection', 'best validation accuracy')}); the choice is in the table below.")
    if summary.get("untrained"):
        return (f"`{summary['model']}` with a new, untrained classification head (baseline)",
                "Not trained: the pre-trained model is evaluated as it is.")
    s = summary.get("settings") or {}
    aug = summary.get("augmentation")
    noise = (f"white noise added to {100 * float(aug['p']):.0f}% of the training clips at "
             f"{aug['snr_db'][0]}–{aug['snr_db'][1]} dB SNR" if summary.get("noise") == "white" and aug
             else "no noise augmentation")
    return (f"`{summary['model']}`, fine-tuned for {summary.get('epochs')} epochs",
            f"{noise}; learning rate {s.get('learning_rate')}, batch {s.get('batch_size')} × "
            f"{s.get('grad_accum')} accumulation steps, {100 * float(s.get('warmup_ratio', 0)):.0f}% warm-up, "
            f"feature encoder {'frozen' if s.get('freeze_feature_encoder') else 'trained'}; the epoch with the best "
            f"validation accuracy is kept; trained on `{summary.get('device')}`.")


def _chosen(split_info: dict[str, Any]) -> str:
    """The setting kept for a split: hyper-parameters, or the best epoch."""
    if "params" in split_info:
        return ", ".join(f"{k}={v}" for k, v in split_info["params"].items())
    if split_info.get("trained") is False:
        return "–"
    return f"epoch {split_info['best_epoch']}" if "best_epoch" in split_info else "–"


def run_details(member: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Everything the runs page shows for each of the member's runs and datasets that have a ``summary.json``."""
    details = []
    for run in member.get("runs") or []:
        for dataset, dcfg in cfg["datasets"].items():
            folder = run_dir(run, dataset, cfg)
            if not (folder / "summary.json").exists():
                continue
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            speakers = []
            if (folder / "folds" / "per_speaker.csv").exists():
                with (folder / "folds" / "per_speaker.csv").open(encoding="utf-8") as f:
                    speakers = list(csv.DictReader(f))
            confusion = [p for p in (figures_root(run, cfg) / "confusion" / f"{run}_{dataset}_{s}.png"
                                     for s in ("folds", "random")) if p.exists()]
            details.append({"run": run, "label": run_label(run), "dataset": dataset,
                            "dataset_name": dcfg.get("name", dataset), "summary": summary, "folder": folder,
                            "speakers": speakers, "confusion": confusion})
    return details


def _repo_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    """A path relative to the repository root (as given when it is outside)."""
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def render_runs(member: dict[str, Any], details: list[dict[str, Any]], base: str = "../../",
                root: Path = PROJECT_ROOT) -> str:
    """The Markdown page with each of the member's runs in detail; ``base`` leads from the page to the repository root."""
    lines = [f"# {member['name']} ({member['slot']}): runs and results", "",
             f"Every number on this page is read from the run's files in [results/]({base}results/). **Unseen "
             "speakers:** speaker-independent 5-fold cross-validation, each fold tests speakers never heard in "
             "training (the headline). **Random split:** 80 / 20, the test speakers were also heard in training.", ""]
    if not details:
        planned = ", ".join(f"{run_label(r)} (`{r}`)" for r in member.get("runs") or [])
        lines += [f"*No runs merged yet.* Planned: {planned}.", ""]
    else:
        lines += ["Runs: " + " · ".join(f"[{d['label']}, {d['dataset_name']}](#{_anchor(d)})" for d in details), ""]
    for d in details:
        s = d["summary"]
        model, training = describe_run(s)
        splits = s.get("splits") or {}
        sizes = []
        for name in ("random", "fold1"):
            info = splits.get(name) or {}
            if "n_train" in info:
                label = "random split" if name == "random" else "each fold"
                sizes.append(f"{label} {info['n_train']} / {info['n_val']} / {info['n_test']}")
        data = ("; ".join(sizes) + " (train / validation / test clips)") if sizes else \
            f"test clips only: {s['random'].get('n', '–')} (random split), {s['folds']['pooled'].get('n', '–')} (5 folds)"
        lines += [f"## {d['label']} · {d['dataset_name']}", "",
                  "| | |", "|---|---|",
                  f"| **Model** | {model} |", f"| **How** | {training} |", f"| **Data** | {data} |",
                  f"| **Run folder** | {_link(_repo_path(d['folder'], root) + '/', base=base)} |", ""]
        wer = s.get("error_rates") or {}
        head = "| Split | Test clips | Accuracy | Macro F1 | Kept | Validation accuracy |"
        rule = "|---|---:|---:|---:|---|---:|"
        if wer:
            head, rule = head + " Word error rate |", rule + "---:|"
        lines += [head, rule]
        for name in ["random"] + FOLD_NAMES:
            metrics = s["random"] if name == "random" else (s["folds"].get("per_fold") or {}).get(name)
            if not metrics:
                continue
            info = splits.get(name) or {}
            val = info.get("val_accuracy")
            row = (f"| {'Random split (seen speakers)' if name == 'random' else 'Fold ' + name[-1]} | {metrics.get('n', '–')} | "
                   f"{_pct(metrics['accuracy'])} | {_f1(metrics['macro_f1'])} | {_chosen(info) if info else '–'} | "
                   f"{_pct(val) if val is not None else '–'} |")
            if wer:
                row += f" {_pct((wer.get(name) or {}).get('wer'))} |"
            lines.append(row)
        pooled, mean, std = s["folds"]["pooled"], s["folds"].get("mean") or {}, s["folds"].get("std") or {}
        row = (f"| **All folds (unseen speakers)** | {pooled.get('n', '–')} | **{_pct(pooled['accuracy'])}** | "
               f"**{_f1(pooled['macro_f1'])}** | fold mean {_pct(mean.get('accuracy'))} ± "
               f"{100 * std.get('accuracy', 0):.1f} | |")
        if wer:
            row += f" {_pct((s.get('error_rates_pooled_folds') or {}).get('wer'))} |"
        lines += [row, ""]
        if d["speakers"]:
            lines += ["Accuracy per test speaker (each speaker scored by the fold that held it out):", "",
                      "| Speaker | Test clips | Correct | Accuracy |", "|---|---:|---:|---:|"]
            lines += [f"| `{r['speaker_id']}` | {r['n']} | {r['correct']} | {_pct(float(r['accuracy']))} |"
                      for r in d["speakers"]]
            lines.append("")
        if d["confusion"]:
            lines.append("Confusion matrices (" + " · ".join(
                "unseen-speaker folds" if p.stem.endswith("folds") else "random split" for p in d["confusion"]) + "):")
            lines += ["", " ".join(f'<img src="{base}{_repo_path(p, root)}" alt="{p.stem}" width="420">' for p in d["confusion"]),
                      ""]
    lines += ["---", "", f"Written by `python -m support.contributions` from the result files; do not edit by hand. "
              f"Back to [{member['name']}]({base}contributions/{member_folder(member)}.md) · "
              f"[who did what]({base}contributions/README.md)", ""]
    return "\n".join(lines)


def _anchor(detail: dict[str, Any]) -> str:
    """GitHub's anchor for a run's heading on the runs page."""
    text = f"{detail['label']} · {detail['dataset_name']}".lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _n(count: int, word: str) -> str:
    """``1 commit``, ``2 commits``."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _f1(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def _text(s: str) -> str:
    """Text safe inside a Markdown table cell."""
    return s.replace("|", "\\|").replace("<", "&lt;")


def _link(path: str, label: str | None = None, base: str = "../") -> str:
    """Link to a repository path from a page ``base`` below the root (``contributions/`` by default)."""
    return f"[{label or path}]({base}{path})"


def member_folder(member: dict[str, Any]) -> str:
    """The member's folder name, e.g. ``M2_rabbi`` (``results/M2_rabbi/``, ``contributions/M2_rabbi.md``)."""
    return f"{member['slot']}_{member['short'].lower()}"


def file_lines(paths: set[str], datasets: list[str]) -> list[str]:
    """Bullet lines per group; result and data files are listed by folder with a file count."""
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(paths):
        groups[file_group(path)].append(path)
    lines = []
    for group in GROUP_ORDER:
        if not groups[group]:
            continue
        if group in COLLAPSED_GROUPS:
            folders: dict[str, list[str]] = defaultdict(list)
            for path in groups[group]:
                folders[folder_of(path, datasets)].append(path)
            items = [_link(p[0]) if len(p) == 1 else f"{_link(d)} ({len(p)} files)" for d, p in sorted(folders.items())]
        else:
            items = [_link(p) for p in groups[group]]
        lines.append(f"- **{group}:** " + " · ".join(items))
    return lines


def render_member(member: dict[str, Any], tasks: list[dict[str, str]], results: list[dict[str, Any]],
                  files: dict[str, set[str]], commits: list[dict[str, Any]], datasets: list[str],
                  ref: str = "main", repo_url: str | None = None) -> str:
    """The Markdown page of one member."""
    done = sum(t["status"].startswith("✅") for t in tasks)
    active = sum(t["status"].startswith("🟡") for t in tasks)
    github = f"[@{member['github']}](https://github.com/{member['github']})" if member.get("github") else "–"
    lines = [
        f"# {member['name']} ({member['slot']})", "", member["role"], "",
        "| | |", "|---|---|",
        f"| **Models** | {member.get('models', '–')} |",
        f"| **GitHub** | {github} |",
        f"| **Branch** | `{member.get('branch', '–')}` |",
        f"| **Working page** | {_link(member['page'])} (task cards, weekly faculty update, handoff log) |",
        f"| **In numbers** | {_n(done, 'task')} done, {active} in progress · {_n(len(commits), 'commit')} · "
        f"{_n(len({r['run'] for r in results}), 'run')} with results · {_n(len(files['created']), 'file')} created |",
        "", "## Tasks", "",
        f"From the task board {_link(BOARD)}: ✅ done and merged · 🟡 in progress · ⬜ to do.", "",
    ]
    if tasks:
        lines += ["| ID | Task | Status | Done by · date |", "|---|---|---|---|"]
        lines += [f"| {t['id']} | {t['task']} | {t['status']} | {t['done']} |" for t in tasks]
    else:
        lines.append("*No tasks on the board.*")
    runs_page = f"results/{member_folder(member)}/RUNS.md"
    lines += ["", "## Results", "", f"Each run in detail (model, how it was trained, every split, accuracy per "
              f"speaker, confusion matrices): {_link(runs_page)}", ""]
    if results:
        lines += ["Unseen speakers: speaker-independent 5-fold cross-validation (the headline). "
                  "Random split: 80 / 20, the test speakers were also heard in training.", "",
                  "| Run | Dataset | Accuracy, unseen speakers | Macro F1, unseen | Accuracy, random split "
                  "| Macro F1, random | Source |", "|---|---|---:|---:|---:|---:|---|"]
        for r in results:
            source = r["source"].relative_to(PROJECT_ROOT).as_posix() if r["source"].is_relative_to(PROJECT_ROOT) \
                else r["source"].as_posix()
            lines.append(f"| {r['label']} | {r['dataset']} | {_pct(r['unseen_acc'])} | {_f1(r['unseen_f1'])} | "
                         f"{_pct(r['random_acc'])} | {_f1(r['random_f1'])} | {_link(source, 'summary.json')} |")
    else:
        lines.append("*No results merged yet.*")
    lines += ["", "## Files", ""]
    if files["created"]:
        lines += [f"Created by {member['short']} (files that still exist):", ""] + file_lines(files["created"], datasets)
    else:
        lines.append(f"*No files created by {member['short']} yet.*")
    if files["edited"]:
        lines += ["", f"Also changed by {member['short']} (created by someone else):", ""]
        lines += file_lines(files["edited"], datasets)
    lines += ["", "## Commits", ""]
    if not member.get("git_names"):
        lines.append(f"*{member['short']}'s git user name is not known yet, so no commits are listed.*")
    elif not commits:
        lines.append(f"*No commits on `{ref}` under {member['short']}'s name yet.*")
    else:
        lines += [f"{_n(len(commits), 'commit')} on `{ref}` under {member['short']}'s name "
                  f"({', '.join(member['git_names'])}), newest first; merge commits are not counted.", "",
                  f"<details><summary>Show all {_n(len(commits), 'commit')}</summary>", "",
                  "| Date | Commit | Message |", "|---|---|---|"]
        for c in reversed(commits):
            short = c["hash"][:7]
            sha = f"[`{short}`]({repo_url}/commit/{c['hash']})" if repo_url else f"`{short}`"
            lines.append(f"| {c['date']} | {sha} | {_text(c['subject'])} |")
        lines += ["", "</details>"]
    lines += ["", "---", "", "Written by `python -m support.contributions` from the repository; do not edit by hand. "
              "Overview of the team: [README.md](README.md)", ""]
    return "\n".join(lines)


def render_index(summaries: list[dict[str, Any]], ref: str = "main") -> str:
    """The overview page: one row per member."""
    lines = [
        "# Who did what", "",
        "One page per team member, written by `python -m support.contributions` from the repository: the git "
        f"history of `{ref}`, the task board ([docs/STATUS.md](../{BOARD})) and the result files in "
        "[results/](../results/). Nothing is typed by hand, so the pages match the repository; they are "
        "regenerated after every merge.", "",
        "| Member | Role | Tasks done | In progress | Commits | Runs with results | Work | Runs and results |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for s in summaries:
        runs = f"results/{s['file'][:-3]}/RUNS.md"
        lines.append(f"| **{s['name']}** ({s['slot']}) | {s['role']} | {s['done']} | {s['active']} | "
                     f"{s['commits']} | {s['runs']} | [{s['file']}]({s['file']}) | [{runs}](../{runs}) |")
    return "\n".join(lines + [""])


def repo_web_url(root: Path) -> str | None:
    """``https://github.com/<owner>/<repo>`` from the ``origin`` remote, or ``None``."""
    try:
        url = git(root, "remote", "get-url", "origin").strip()
    except subprocess.CalledProcessError:
        return None
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return f"https://github.com/{match[1]}" if match else None


def write_contributions(cfg: dict[str, Any] | None = None, ref: str = "main", out: str | Path | None = None,
                        root: Path = PROJECT_ROOT) -> list[Path]:
    """Write the overview and one page per member of ``team`` in the config.

    Returns:
        The files written.
    """
    cfg = cfg if cfg is not None else load_config()
    out = Path(out) if out is not None else root / "contributions"
    out.mkdir(parents=True, exist_ok=True)
    board = (root / BOARD).read_text(encoding="utf-8")
    history = read_history(root, ref)
    existing = set(git(root, "ls-tree", "-r", "--name-only", ref).splitlines())
    by_author = attribute_files(history, existing)
    datasets = list(cfg["datasets"])
    repo_url = repo_web_url(root)
    written, summaries = [], []
    for member in cfg["team"]:
        names = set(member.get("git_names") or [])
        commits = [c for c in history if c["author"] in names]
        files = {"created": set(), "edited": set()}
        for name in names:
            files["created"] |= by_author.get(name, {}).get("created", set())
            files["edited"] |= by_author.get(name, {}).get("edited", set())
        files["edited"] -= files["created"]
        tasks = member_tasks(member, board)
        results = run_results(member, cfg)
        name = f"{member_folder(member)}.md"
        path = out / name
        path.write_text(render_member(member, tasks, results, files, commits, datasets, ref, repo_url),
                        encoding="utf-8")
        runs_path = resolve_path(cfg, "results") / member_folder(member) / "RUNS.md"
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        runs_path.write_text(render_runs(member, run_details(member, cfg), root=root), encoding="utf-8")
        written += [path, runs_path]
        summaries.append({"name": member["name"], "slot": member["slot"], "role": member["role"], "file": name,
                          "done": sum(t["status"].startswith("✅") for t in tasks),
                          "active": sum(t["status"].startswith("🟡") for t in tasks),
                          "commits": len(commits), "runs": len({r["run"] for r in results})})
    index = out / "README.md"
    index.write_text(render_index(summaries, ref), encoding="utf-8")
    return [index] + written


def main() -> None:
    """Write the contribution pages."""
    parser = argparse.ArgumentParser(description="One page per team member: tasks, results, files and commits.")
    parser.add_argument("--ref", default="main", help="git branch or commit whose history is read (default: main)")
    parser.add_argument("--out", default=None, help="output folder (default: contributions/)")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    for path in write_contributions(load_config(args.config), args.ref, args.out):
        print(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)


if __name__ == "__main__":
    main()

"""One page per team member for the faculty: who did what, written from the repository (task T50).

Nothing on the pages is typed by hand, so they always match the repository:

| Section | Read from |
|---|---|
| Role, models, GitHub, branch | ``team`` in ``configs/config.yaml`` |
| Tasks and their status | the member's section of the task board ``docs/STATUS.md`` (plus rows owned or done by them elsewhere on the board) |
| Results | ``summary.json`` of each of the member's runs, found with ``support.evaluate.run_dir`` |
| Files created / edited | the git history: who added each file that still exists, and who changed it later |
| Commits | every commit (merges not counted) whose author is one of the member's ``git_names`` |

Output: ``contributions/README.md`` (overview) and ``contributions/<slot>_<short>.md`` per member.
Run it again after every merge.

Usage::

    python -m support.contributions                      # history of main
    python -m support.contributions --ref HEAD --out contributions
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from support.config import PROJECT_ROOT, load_config
from support.evaluate import run_dir
from support.figures import run_label

BOARD = "docs/STATUS.md"
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


def _link(path: str, label: str | None = None) -> str:
    """Link from ``contributions/`` to a repository path."""
    return f"[{label or path}](../{path})"


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
    lines += ["", "## Results", ""]
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
        "| Member | Role | Tasks done | In progress | Commits | Runs with results | Page |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        lines.append(f"| **{s['name']}** ({s['slot']}) | {s['role']} | {s['done']} | {s['active']} | "
                     f"{s['commits']} | {s['runs']} | [{s['file']}]({s['file']}) |")
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
        name = f"{member['slot']}_{member['short'].lower()}.md"
        path = out / name
        path.write_text(render_member(member, tasks, results, files, commits, datasets, ref, repo_url),
                        encoding="utf-8")
        written.append(path)
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

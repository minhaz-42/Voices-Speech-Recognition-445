"""Tests for support/contributions.py on a small git repository built in a temporary folder."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from support import contributions as ct
from support.config import load_config

BOARD = """# Board

### M4 · Shafil

| ID | Task (→ done when) | Depends on | Status | Done by · date |
|---|---|---|---|---|
| T14 | **SVM** → `results/svm/fsc/` | — | ✅ | Shafil · 2026-10-18 |
| T27 | Poster and slides | T23 | 🟡 poster ✅ | |

### M1 · Tanvir

| ID | Task (→ done when) | Depends on | Status | Done by · date |
|---|---|---|---|---|
| T11 | Random Forest | — | ✅ | Tanvir · 2026-10-17 |

### At the end

| ID | Task (→ done when) | Owner | Depends on | Status | Done by · date |
|---|---|---|---|---|---|
| T24 | Final aggregation | M1 | all | ⬜ | |
| T28 | Submit | all | T25 | ⬜ | |

### ✅ Done

| ID | Task | Owner | Done by · date |
|---|---|---|---|
| T03 | `support/metrics.py` a \\| b | M3 slot | Tanvir · 2026-10-04 |
| T35 | Figures | M4 | Shafil · 2026-10-23 |
"""

SHAFIL = {"slot": "M4", "name": "Shafil Ahmed", "short": "Shafil", "role": "Figures", "models": "SVM",
          "git_names": ["Shafil Ahmed"], "github": "Shafil10", "branch": "dev_shafil", "page": "docs/team/M4_shafil.md",
          "runs": ["svm"]}
TANVIR = {"slot": "M1", "name": "Tanvir Ahmed", "short": "Tanvir", "role": "Lead", "models": "RF",
          "git_names": ["minhaz"], "github": "minhaz-42", "branch": "main", "page": "docs/team/M1_tanvir.md",
          "runs": ["random_forest"]}
MAHIR = {"slot": "M3", "name": "Mahir Ahmed", "short": "Mahir", "role": "Analysis", "models": "k-NN",
         "git_names": [], "github": None, "branch": "dev_mahir", "page": "docs/team/M3_mahir.md", "runs": ["knn"]}


def run_git(root: Path, *args: str, author: str | None = None) -> None:
    """Run git in ``root``, committing as ``author`` when given."""
    who = ["-c", f"user.name={author}", "-c", f"user.email={author.split()[0].lower()}@example.org"] if author else []
    subprocess.run(["git", "-C", str(root), "-c", "commit.gpgsign=false", *who, *args], check=True,
                   capture_output=True)


def write(root: Path, path: str, text: str) -> None:
    """Write a file inside the repository."""
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text, encoding="utf-8")


def commit(root: Path, author: str, message: str) -> None:
    """Stage everything and commit it as ``author``."""
    run_git(root, "add", "-A")
    run_git(root, "commit", "-m", message, author=author)


def make_repo(root: Path) -> None:
    """minhaz creates a module and a note and later moves the note; on a branch, Shafil edits the module and
    adds a run; the branch comes back with a merge commit."""
    run_git(root, "init", "-b", "main")
    write(root, "support/core.py", "x = 1\n")
    write(root, "notes.md", "notes\n")
    write(root, "docs/STATUS.md", BOARD)
    commit(root, "minhaz", "T01: core module")
    run_git(root, "checkout", "-b", "dev_shafil")
    write(root, "support/core.py", "x = 2\n")
    write(root, "support/visualization.py", "y = 1\n")
    summary = {"folds": {"pooled": {"accuracy": 0.223, "macro_f1": 0.219}},
               "random": {"accuracy": 0.305, "macro_f1": 0.304}}
    for name in ("summary.json", "random/metrics.json", "folds/metrics.json"):
        write(root, f"results/svm/fsc/{name}", json.dumps(summary))
    commit(root, "Shafil Ahmed", "T14: SVM | first run")
    run_git(root, "checkout", "main")
    run_git(root, "mv", "notes.md", "docs/notes.md")
    commit(root, "minhaz", "T00: move the notes")
    run_git(root, "merge", "--no-ff", "-m", "Merge dev_shafil", "dev_shafil", author="minhaz")


class TestBoard(unittest.TestCase):
    """Reading the task board."""

    def test_read_tables_keeps_headings_and_escaped_pipes(self):
        tables = ct.read_tables(BOARD)
        self.assertEqual([h for h, _ in tables], ["M4 · Shafil", "M1 · Tanvir", "At the end", "✅ Done"])
        self.assertEqual(tables[0][1][1]["Status"], "🟡 poster ✅")
        self.assertIn("a \\| b", tables[3][1][0]["Task"])

    def test_member_tasks_own_section_owner_and_done_by(self):
        shafil = [t["id"] for t in ct.member_tasks(SHAFIL, BOARD)]
        tanvir = ct.member_tasks(TANVIR, BOARD)
        self.assertEqual(shafil, ["T14", "T27", "T35"])
        self.assertEqual([t["id"] for t in tanvir], ["T11", "T24", "T03"])
        self.assertEqual(tanvir[2]["status"], "✅")  # the Done table has no status column
        self.assertEqual(ct.member_tasks(MAHIR, BOARD), [])


class TestFiles(unittest.TestCase):
    """Grouping files for the page."""

    def test_groups(self):
        self.assertEqual(ct.file_group("support/audio.py"), "Code")
        self.assertEqual(ct.file_group("tests/test_audio.py"), "Tests")
        self.assertEqual(ct.file_group("data/README.md"), "Documents")
        self.assertEqual(ct.file_group("data/splits/fsc/fold1.csv"), "Data and settings")
        self.assertEqual(ct.file_group("images/confusion/svm_fsc_folds.png"), "Figures")
        self.assertEqual(ct.file_group("results/svm/fsc/summary.json"), "Results")

    def test_result_files_are_listed_by_run_folder(self):
        self.assertEqual(ct.folder_of("results/svm/fsc/folds/metrics.json", ["fsc"]), "results/svm/")
        self.assertEqual(ct.folder_of("results/M1_tanvir/noise/rf_fsc.csv", ["fsc"]), "results/M1_tanvir/noise/")
        lines = ct.file_lines({"results/svm/fsc/a.json", "results/svm/fsc/b.json", "data/listen_check_fsc.csv"},
                              ["fsc"])
        self.assertIn("[results/svm/](../results/svm/) (2 files)", "\n".join(lines))
        self.assertIn("[data/listen_check_fsc.csv](../data/listen_check_fsc.csv)", "\n".join(lines))


class TestRepository(unittest.TestCase):
    """History, attribution and the written pages, on a temporary repository."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_history_skips_merges_and_attributes_files(self):
        history = ct.read_history(self.root, "main")
        self.assertEqual([c["author"] for c in history], ["minhaz", "Shafil Ahmed", "minhaz"])
        existing = set(ct.git(self.root, "ls-tree", "-r", "--name-only", "main").splitlines())
        files = ct.attribute_files(history, existing)
        self.assertIn("docs/notes.md", files["minhaz"]["created"])  # a rename keeps the creator
        self.assertNotIn("notes.md", files["minhaz"]["created"])
        self.assertIn("support/visualization.py", files["Shafil Ahmed"]["created"])
        self.assertEqual(files["Shafil Ahmed"]["edited"], {"support/core.py"})
        self.assertEqual(files["minhaz"]["edited"], set())

    def test_write_contributions(self):
        cfg = copy.deepcopy(load_config())
        cfg["paths"]["results"] = str(self.root / "results")
        cfg["result_folders"] = {}
        cfg["team"] = [TANVIR, MAHIR, SHAFIL]
        written = ct.write_contributions(cfg, "main", self.root / "contributions", root=self.root)
        self.assertEqual([p.name for p in written], ["README.md", "M1_tanvir.md", "M3_mahir.md", "M4_shafil.md"])
        shafil = (self.root / "contributions" / "M4_shafil.md").read_text(encoding="utf-8")
        self.assertIn("1 commit on `main`", shafil)
        self.assertIn("T14: SVM \\| first run", shafil)
        self.assertIn("| SVM | TenVoices-FSC | 22.3% | 0.219 | 30.5% | 0.304 |", shafil)
        self.assertIn("[support/visualization.py](../support/visualization.py)", shafil)
        self.assertIn("2 tasks done, 1 in progress", shafil)
        mahir = (self.root / "contributions" / "M3_mahir.md").read_text(encoding="utf-8")
        self.assertIn("git user name is not known yet", mahir)
        self.assertIn("*No results merged yet.*", mahir)
        index = (self.root / "contributions" / "README.md").read_text(encoding="utf-8")
        self.assertIn("| **Shafil Ahmed** (M4) | Figures | 2 | 1 | 1 | 1 | [M4_shafil.md](M4_shafil.md) |", index)


if __name__ == "__main__":
    unittest.main()

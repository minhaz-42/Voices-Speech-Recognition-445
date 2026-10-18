"""Tests for support/embeddings.py.

The embedding itself runs in a child process (``tests/_embeddings_check.py``): on macOS,
PyTorch and XGBoost must not share a process, and the classical-model tests use XGBoost.
This module only reads the child's JSON report, so it never imports PyTorch.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from support.config import PROJECT_ROOT


class TestEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        proc = subprocess.run([sys.executable, "-m", "tests._embeddings_check", str(folder)], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"the embedding check failed:\n{proc.stderr[-3000:]}")
        cls.report = json.loads((folder / "report.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_vector_per_clip(self):
        embed = self.report["embed"]
        self.assertEqual(embed["shape"], [3, 32])  # 3 clips of different lengths, hidden size 32
        self.assertEqual(embed["dtype"], "float32")

    def test_layer_option_and_repeatability(self):
        self.assertTrue(self.report["embed"]["layer_changes_output"])
        self.assertTrue(self.report["embed"]["repeatable"])

    def test_cache_layout(self):
        cache = self.report["cache"]
        self.assertEqual(cache["name"], "fsc_w2v2.npz")
        self.assertEqual(cache["keys"], ["features", "settings", "utt_ids"])
        self.assertEqual(cache["n_clips"], cache["n_manifest"])
        self.assertEqual(cache["width"], 32)
        self.assertEqual(cache["features_dtype"], "float32")
        self.assertEqual(cache["utt_ids_kind"], "U")  # a plain string array, not objects
        self.assertEqual(len(cache["settings"]), 2)
        self.assertTrue(cache["settings"][0].startswith("model="))
        self.assertEqual(cache["settings"][1], "layer=last")

    def test_same_output_on_rerun(self):
        self.assertTrue(self.report["cache"]["same_on_rerun"])

    def test_classical_reads_the_cache(self):
        cache = self.report["cache"]
        self.assertEqual(cache["cached_shape"], [5, 32])
        self.assertTrue(cache["cached_matches"])


if __name__ == "__main__":
    unittest.main()

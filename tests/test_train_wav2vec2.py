"""Tests for support/train_wav2vec2.py.

The training itself runs in a child process (``tests/_wav2vec2_check.py``): on macOS,
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


class TestTrainWav2vec2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        proc = subprocess.run([sys.executable, "-m", "tests._wav2vec2_check", str(folder)], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"the training check failed:\n{proc.stderr[-3000:]}")
        cls.report = json.loads((folder / "report.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_names(self):
        self.assertEqual(self.report["run_names"], ["w2v2_15ep_noise", "w2v2_3ep", "w2v2_untrained"])

    def test_training_selects_best_validation_epoch(self):
        summary = self.report["trained"]
        self.assertEqual(summary["run"], "w2v2_3ep_noise")
        for split in ("random", "fold1"):
            info = summary["splits"][split]
            accs = [c["val_accuracy"] for c in info["curve"]]
            self.assertEqual(len(accs), 3)
            self.assertEqual(info["best_epoch"], accs.index(max(accs)) + 1)  # first best epoch
            self.assertEqual(info["val_accuracy"], max(accs))
            self.assertTrue(info["trained"])
        self.assertEqual(summary["random"]["n"], summary["splits"]["random"]["n_test"])
        self.assertEqual(summary["augmentation"], self.report["augmentation_cfg"])

    def test_outputs_are_saved(self):
        files = self.report["files"]
        self.assertTrue(files["predictions_fold1"])
        self.assertTrue(files["model_config"])
        self.assertTrue(files["extractor_config"])
        self.assertFalse(files["untrained_model_saved"])  # run with save_models=False

    def test_untrained_baseline_does_not_train(self):
        info = self.report["untrained"]["splits"]["random"]
        self.assertFalse(info["trained"])
        self.assertEqual(info["best_epoch"], 0)
        self.assertEqual(len(info["curve"]), 1)
        self.assertIsNone(self.report["untrained"]["augmentation"])

    def test_noise_is_only_added_to_training_clips(self):
        d = self.report["dataset"]
        self.assertEqual(d["shape"], [16000])
        self.assertTrue(d["label_ok"])
        self.assertTrue(d["clean_repeatable"])
        self.assertTrue(d["noise_changes_clip"])
        self.assertTrue(d["noise_differs_each_pass"])

    def test_saved_model_recognises_raw_audio(self):
        inf = self.report["inference"]
        self.assertTrue(inf["is_wav2vec2"])
        self.assertEqual(inf["labels"], ["P01", "P02", "P03"])
        self.assertEqual(inf["top_ids"], ["P01", "P02", "P03"])
        self.assertAlmostEqual(inf["score_sum"], 1.0, places=4)
        self.assertTrue(inf["sorted"])
        self.assertEqual(inf["text_of_p01"], "phrase number 1")
        self.assertTrue(inf["too_long_raises"])

    def test_unknown_noise_option_raises(self):
        self.assertTrue(self.report["bad_noise_raises"])


if __name__ == "__main__":
    unittest.main()

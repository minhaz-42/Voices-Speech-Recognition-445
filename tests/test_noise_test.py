"""Tests for support/noise_test.py with a k-NN model trained on synthetic audio (no PyTorch).

The wav2vec 2.0 path is covered in tests/_wav2vec2_check.py.
"""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from support import classical
from support import noise_test as nt
from support.dataset import build_processed
from support.noise import measured_snr_db
from tests._synthetic import make_config, make_subset, write_audio
from support.make_splits import make_splits


class TestNoisyClip(unittest.TestCase):
    def test_deterministic_per_clip_and_snr(self):
        clean = np.sin(np.linspace(0, 200, 8000)).astype(np.float32)
        a = nt.noisy_clip(clean, 10, "utt1")
        np.testing.assert_array_equal(a, nt.noisy_clip(clean, 10, "utt1"))      # same for every model
        self.assertFalse(np.array_equal(a, nt.noisy_clip(clean, 10, "utt2")))   # differs between clips
        self.assertFalse(np.array_equal(a, nt.noisy_clip(clean, 5, "utt1")))    # and between SNRs
        self.assertAlmostEqual(measured_snr_db(clean, a), 10, delta=0.3)


class TestRunNoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        subset = write_audio(make_subset("fsc", n_phrases=4, takes=2), tmp / "raw")  # 80 clips
        cfg = make_config(tmp, "fsc", subset)
        cfg["paths"].update(results=str(tmp / "results"), images=str(tmp / "images"), outputs=str(tmp / "outputs"))
        cfg["classical"] = {"knn": {"n_neighbors": [1]}}
        cls.cfg, cls.root = cfg, tmp
        build_processed("fsc", cfg)
        make_splits("fsc", cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            classical.run("knn", "fsc", ["random", "fold1", "fold2"], cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_summary_and_predictions(self):
        summary, preds = nt.run_noise_test("knn", "fsc", ["random", "fold1", "fold2"], self.cfg, snrs=[20, 0])
        self.assertEqual(sorted(set(summary["snr_db"])), ["0", "20", "clean"])
        self.assertEqual(sorted(set(summary["protocol"])), ["folds", "random"])
        # each SNR scores every test clip once: random 16 clips, two folds 2 x 16
        self.assertEqual(len(preds), 3 * (16 + 32))
        clean_folds = summary[(summary["snr_db"] == "clean") & (summary["protocol"] == "folds")].iloc[0]
        self.assertEqual(clean_folds["n"], 32)
        saved = pd.read_csv(self.root / "results" / "noise" / "knn_fsc.csv")
        self.assertEqual(len(saved), len(summary))

    def test_clean_matches_the_normal_evaluation(self):
        summary, _ = nt.run_noise_test("knn", "fsc", ["random"], self.cfg, snrs=[10])
        clean = summary[summary["snr_db"] == "clean"].iloc[0]["accuracy"]
        normal = pd.read_csv(self.root / "results" / "knn" / "fsc" / "predictions_random.csv")
        self.assertAlmostEqual(clean, float((normal["label"] == normal["pred"]).mean()))

    def test_same_predictions_on_a_rerun(self):
        _, a = nt.run_noise_test("knn", "fsc", ["fold1"], self.cfg, snrs=[5])
        _, b = nt.run_noise_test("knn", "fsc", ["fold1"], self.cfg, snrs=[5])
        pd.testing.assert_frame_equal(a, b)

    def test_member_folder(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["result_folders"] = {"M9_test": ["knn"]}
        nt.run_noise_test("knn", "fsc", ["fold1"], cfg, snrs=[10])
        self.assertTrue((self.root / "results" / "M9_test" / "noise" / "knn_fsc_predictions.csv").exists())

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            nt.model_path("svm", "fsc", "fold1", self.cfg)


if __name__ == "__main__":
    unittest.main()

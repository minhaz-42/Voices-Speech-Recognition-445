"""End-to-end test of support/classical.py on synthetic audio (each phrase has its own pitch)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from support import classical
from support.dataset import build_processed
from tests._synthetic import make_config, make_subset, write_audio
from tools.make_splits import make_splits

SMALL_GRIDS = {
    "random_forest": {"n_estimators": [30], "max_depth": [None, 5]},
    "xgboost": {"n_estimators": [30], "max_depth": [2]},
    "knn": {"n_neighbors": [1, 3]},
    "svm": {"C": [1, 10], "gamma": ["scale"]},
}


class TestClassicalRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        subset = write_audio(make_subset("fsc", n_phrases=5, takes=2), tmp / "raw")  # 100 clips
        cls.cfg = make_config(tmp, "fsc", subset)
        cls.cfg["paths"].update(results=str(tmp / "results"), images=str(tmp / "images"), outputs=str(tmp / "outputs"))
        cls.cfg["classical"] = SMALL_GRIDS
        cls.root = tmp
        build_processed("fsc", cls.cfg)
        make_splits("fsc", cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_param_grid_expands_lists(self):
        self.assertEqual(len(classical.param_grid("random_forest", self.cfg)), 2)
        with self.assertRaises(KeyError):
            classical.param_grid("random_forest", {"classical": {}})
        with self.assertRaises(ValueError):
            classical.make_model("lstm", {})

    def test_every_model_learns_the_pitches(self):
        for name in classical.MODELS:
            with self.subTest(model=name):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = classical.run(name, "fsc", ["random", "fold1", "fold2"], self.cfg)
                self.assertGreater(summary["random"]["accuracy"], 0.9)
                self.assertGreater(summary["folds"]["pooled"]["accuracy"], 0.9)
                self.assertEqual(summary["folds"]["folds_run"], ["fold1", "fold2"])
                saved = json.loads((self.root / "results" / name / "fsc" / "summary.json").read_text())
                self.assertIn("params", saved["splits"]["random"])

    def test_selection_never_uses_test_clips(self):
        pred, info = classical.run_split("knn", "fsc", "fold3", self.cfg, [f"P{i:02d}" for i in range(1, 6)])
        self.assertEqual((info["n_train"], info["n_val"], info["n_test"]), (70, 10, 20))
        self.assertEqual(pred["speaker_id"].nunique(), 2)
        self.assertTrue(set(pred["confidence"]) <= set(np.round(np.linspace(0, 1, 10001), 4)))

    def test_saved_model_predicts_phrase_ids(self):
        with contextlib.redirect_stdout(io.StringIO()):
            classical.run("svm", "fsc", ["random"], self.cfg)
        bundle = joblib.load(self.root / "outputs" / "models" / "svm" / "fsc" / "random.joblib")
        self.assertEqual(sorted(bundle["labels"]), [f"P{i:02d}" for i in range(1, 6)])
        self.assertEqual(len(bundle["labels"]), len(bundle["model"].classes_))

    def test_split_argument(self):
        self.assertEqual(classical.parse_splits("all")[0], "random")
        self.assertEqual(len(classical.parse_splits("folds")), 5)
        self.assertEqual(classical.parse_splits("fold4"), ["fold4"])


if __name__ == "__main__":
    unittest.main()

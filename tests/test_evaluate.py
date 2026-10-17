"""Tests for support/evaluate.py with hand-made predictions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from support import evaluate as ev
from support.config import load_config


def preds(rows):
    return pd.DataFrame(rows, columns=["utt_id", "speaker_id", "label", "pred"])


FOLD1 = preds([("u1", "S1", "A", "A"), ("u2", "S1", "B", "B"), ("u3", "S2", "A", "B"), ("u4", "S2", "B", "B")])
FOLD2 = preds([("u5", "S3", "A", "A"), ("u6", "S3", "B", "A"), ("u7", "S4", "A", "A"), ("u8", "S4", "B", "B")])


class TestChecks(unittest.TestCase):
    def test_bad_predictions_raise(self):
        with self.assertRaises(ValueError):
            ev.check_predictions(FOLD1.drop(columns="pred"))
        with self.assertRaises(ValueError):
            ev.check_predictions(FOLD1.iloc[:0])
        with self.assertRaises(ValueError):
            ev.check_predictions(pd.concat([FOLD1, FOLD1.iloc[:1]]))


class TestFolds(unittest.TestCase):
    def test_pooled_mean_and_std(self):
        s = ev.summarise_folds({"fold1": FOLD1, "fold2": FOLD2})
        self.assertAlmostEqual(s["pooled"]["accuracy"], 6 / 8)
        self.assertEqual(s["pooled"]["n"], 8)
        self.assertAlmostEqual(s["per_fold"]["fold1"]["accuracy"], 0.75)
        self.assertAlmostEqual(s["mean"]["accuracy"], 0.75)
        self.assertAlmostEqual(s["std"]["accuracy"], 0.0)

    def test_pooling_rejects_a_clip_in_two_folds(self):
        with self.assertRaises(ValueError):
            ev.summarise_folds({"fold1": FOLD1, "fold2": FOLD1})


class TestSaveRun(unittest.TestCase):
    def test_writes_results_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(load_config())
            cfg["paths"]["results"] = str(Path(tmp) / "results")
            cfg["paths"]["images"] = str(Path(tmp) / "images")
            summary = ev.save_run("demo", "fsc", {"random": FOLD1, "fold1": FOLD1, "fold2": FOLD2},
                                  labels=["A", "B", "C"], extra={"model": "demo"}, cfg=cfg)
            root = Path(tmp) / "results" / "demo" / "fsc"
            self.assertAlmostEqual(summary["random"]["accuracy"], 0.75)
            self.assertAlmostEqual(summary["folds"]["pooled"]["accuracy"], 0.75)
            self.assertEqual(summary["model"], "demo")
            for name in ("summary.json", "predictions_random.csv", "predictions_fold2.csv",
                         "random/metrics.json", "folds/classification_report.csv", "folds/per_speaker.csv",
                         "folds/confusion_matrix.csv"):
                self.assertTrue((root / name).exists(), name)
            self.assertTrue((Path(tmp) / "images" / "confusion" / "demo_fsc_folds.png").exists())
            report = pd.read_csv(root / "random" / "classification_report.csv", index_col=0)
            self.assertIn("C", report.index)  # the full label list keeps absent phrases as rows
            self.assertEqual(json.loads((root / "summary.json").read_text())["folds"]["folds_run"], ["fold1", "fold2"])
            reloaded = ev.load_run_predictions("demo", "fsc", cfg)
            self.assertEqual(sorted(reloaded), ["fold1", "fold2", "random"])

    def test_member_folder_keeps_runs_and_figures_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(load_config())
            cfg["paths"]["results"] = str(Path(tmp) / "results")
            cfg["paths"]["images"] = str(Path(tmp) / "images")
            cfg["result_folders"] = {"M9_test": ["demo"]}
            ev.save_run("demo", "fsc", {"random": FOLD1, "fold1": FOLD1}, labels=["A", "B"], cfg=cfg)
            ev.save_run("other", "l2arctic", {"fold2": FOLD2}, labels=["A", "B"], cfg=cfg)
            member = Path(tmp) / "results" / "M9_test"
            self.assertTrue((member / "demo" / "fsc" / "summary.json").exists())
            self.assertTrue((member / "figures" / "confusion" / "demo_fsc_random.png").exists())
            self.assertFalse((Path(tmp) / "results" / "demo").exists())
            self.assertTrue((Path(tmp) / "images" / "confusion" / "other_l2arctic_folds.png").exists())
            self.assertEqual(sorted(ev.load_run_predictions("demo", "fsc", cfg)), ["fold1", "random"])
            runs = ev.list_runs(cfg=cfg)
            self.assertEqual(list(zip(runs["run"], runs["dataset"], runs["folder"])),
                             [("demo", "fsc", "M9_test"), ("other", "l2arctic", "")])
            self.assertEqual(list(ev.list_runs("fsc", cfg)["run"]), ["demo"])

    def test_run_in_two_folders_raises(self):
        cfg = copy.deepcopy(load_config())
        cfg["result_folders"] = {"M8_a": ["demo"], "M9_b": ["demo"]}
        with self.assertRaises(ValueError):
            ev.run_dir("demo", "fsc", cfg)
        cfg["result_folders"] = None
        self.assertEqual(ev.results_root("demo", cfg), ev.resolve_path(cfg, "results"))


if __name__ == "__main__":
    unittest.main()

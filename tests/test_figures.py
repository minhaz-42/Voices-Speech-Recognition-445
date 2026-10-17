"""Tests for support/figures.py on fake results written with support.evaluate.save_run."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from support import figures as fg
from support.evaluate import FOLD_NAMES, results_root, run_dir, save_run
from support.make_splits import make_splits
from tests._synthetic import make_config, make_subset

RUNS = ["random_forest", "w2v2_15ep", "whisper_small"]


def fake_predictions(split_file: Path, seed: int, whisper: bool = False) -> pd.DataFrame:
    """Test clips of a split; every fourth answer is wrong."""
    rng = np.random.default_rng(seed)
    test = pd.read_csv(split_file)
    test = test[test["part"] == "test"].reset_index(drop=True)
    wrong = test.index % 4 == 0
    pred = np.where(wrong, "P01", test["phrase_id"])
    pred = np.where(wrong & (test["phrase_id"] == "P01"), "P02", pred)
    table = pd.DataFrame({"utt_id": test["utt_id"], "speaker_id": test["speaker_id"], "label": test["phrase_id"],
                          "pred": pred})
    if whisper:
        return table.assign(l1="English", wer=np.where(wrong, 0.5, 0.0), cer=np.where(wrong, 0.2, 0.0))
    return table.assign(confidence=np.round(rng.uniform(0.05, 0.4, len(table)), 4))


class TestHelpers(unittest.TestCase):
    def test_run_labels(self):
        self.assertEqual(fg.run_label("w2v2_15ep_noise"), "wav2vec 2.0, 15 epochs + noise")
        self.assertEqual(fg.run_label("hubert_15ep_noise"), "HuBERT Base, 15 epochs + noise")
        self.assertEqual(fg.run_label("w2v2_3ep"), "wav2vec 2.0, 3 epochs")
        self.assertEqual(fg.run_label("random_forest"), "Random Forest")
        self.assertEqual(fg.run_label("something_new"), "something_new")

    def test_coverage_curve(self):
        curve = fg.coverage_curve([0.9, 0.8, 0.8, 0.1], [True, False, True, False])
        np.testing.assert_allclose(curve["coverage"], [0.25, 0.75, 1.0])
        np.testing.assert_allclose(curve["accuracy"], [1.0, 2 / 3, 0.5])

    def test_noise_by_fold(self):
        preds = pd.DataFrame({"split": ["fold1", "fold1", "fold2", "random"], "snr_db": ["5", "5", "5", "5"],
                              "label": ["A", "B", "A", "A"], "pred": ["A", "A", "A", "B"]})
        table = fg.noise_by_fold(preds)
        self.assertEqual(list(table["split"]), ["fold1", "fold2"])  # the random split is not a fold
        self.assertEqual(list(table["accuracy"]), [0.5, 1.0])

    def test_training_curves_mark_the_best_epoch(self):
        summary = {"splits": {"fold1": {"best_epoch": 2, "curve": [
            {"epoch": 1, "train_loss": 3.0, "val_loss": 3.1, "val_accuracy": 0.1},
            {"epoch": 2, "train_loss": 2.0, "val_loss": 2.2, "val_accuracy": 0.5}]}}}
        table = fg.training_curves(summary)
        self.assertEqual(list(table["best"]), [False, True])
        self.assertTrue(fg.training_curves({"splits": {"random": {"params": {}}}}).empty)  # traditional models

    def test_unknown_folder_raises(self):
        with self.assertRaises(KeyError):
            fg.folder_runs("M9_nobody", {"result_folders": {"M1_tanvir": ["random_forest"]}})


class TestDrawAll(unittest.TestCase):
    """Three fake runs in a member folder: a traditional model, a fine-tuned model and Whisper."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cfg = make_config(tmp, "fsc", make_subset("fsc", n_phrases=4, takes=2))
        cfg["paths"].update(results=str(tmp / "results"), images=str(tmp / "images"))
        cfg["result_folders"] = {"M9_test": RUNS}
        make_splits("fsc", cfg)
        splits = tmp / "splits" / "fsc"
        labels = ["P01", "P02", "P03", "P04"]
        names = ["random"] + FOLD_NAMES
        curve = [{"epoch": e, "train_loss": 4 - e, "val_loss": 4.1 - e, "val_accuracy": 0.3 * e} for e in (1, 2, 3)]
        save_run("random_forest", "fsc", {n: fake_predictions(splits / f"{n}.csv", i) for i, n in enumerate(names)},
                 labels, cfg=cfg)
        save_run("w2v2_15ep", "fsc", {n: fake_predictions(splits / f"{n}.csv", i) for i, n in enumerate(names)},
                 labels, extra={"splits": {n: {"best_epoch": 3, "curve": curve} for n in names}}, cfg=cfg)
        whisper = {n: fake_predictions(splits / f"{n}.csv", i, whisper=True) for i, n in enumerate(names)}
        save_run("whisper_small", "fsc", whisper, labels, cfg=cfg)
        folder = run_dir("whisper_small", "fsc", cfg)
        pd.DataFrame({"l1": ["English"], "n": [80], "wer": [0.125], "cer": [0.05], "accuracy": [0.75]}).to_csv(
            folder / "wer_per_l1.csv", index=False)
        pooled = pd.concat([whisper[n] for n in FOLD_NAMES])
        pooled.groupby("speaker_id", as_index=False)["wer"].mean().to_csv(folder / "wer_per_speaker.csv", index=False)

        root = results_root("w2v2_15ep", cfg)
        (root / "noise").mkdir(parents=True)
        rows, preds = [], []
        for snr, acc in zip(["clean", "20", "10", "5"], [0.9, 0.8, 0.6, 0.3]):
            rows += [{"run": "w2v2_15ep", "dataset": "fsc", "protocol": p, "snr_db": snr, "n": 80, "accuracy": acc,
                      "macro_f1": acc} for p in ("random", "folds")]
            preds += [{"utt_id": f"u{k}", "speaker_id": "spk01", "split": f"fold{k % 5 + 1}", "snr_db": snr,
                       "label": "P01", "pred": "P01" if k < 10 * acc else "P02"} for k in range(10)]
        pd.DataFrame(rows).to_csv(root / "noise" / "w2v2_15ep_fsc.csv", index=False)
        pd.DataFrame(preds).to_csv(root / "noise" / "w2v2_15ep_fsc_predictions.csv", index=False)
        pd.DataFrame([{"run": "w2v2_15ep", "dataset": "fsc", "split_model": "random", "device": "cpu",
                       "mean_ms_per_clip": 70.0, "median_ms_per_clip": 69.0, "repeats": 5, "model_size_mb": 378.0},
                      {"run": "random_forest", "dataset": "fsc", "split_model": "random", "device": "cpu",
                       "mean_ms_per_clip": 12.0, "median_ms_per_clip": 12.0, "repeats": 5, "model_size_mb": 70.0}]
                     ).to_csv(root / "benchmark_fsc.csv", index=False)
        cls.cfg, cls.root = cfg, tmp

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_figure_is_drawn_into_the_member_folder(self):
        written = fg.draw_all("M9_test", self.cfg)
        self.assertEqual(sorted(p.name for p in written),
                         sorted(["seen_vs_unseen.png", "training_curves_fsc.png", "noise_robustness_fsc.png",
                                 "per_speaker_fsc.png", "confidence_fsc.png", "speed_vs_accuracy_fsc.png",
                                 "whisper_errors.png"]))
        for path in written:
            self.assertEqual(path.parent, self.root / "results" / "M9_test" / "figures")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_figures_without_their_inputs_are_skipped(self):
        cfg = dict(self.cfg, result_folders={"M9_test": ["random_forest"]})
        out = self.root / "only_rf"
        written = fg.draw_all("M9_test", cfg, out=out)
        # no curves, noise test or Whisper files for this run; its benchmark row still gives the speed figure
        self.assertEqual(sorted(p.name for p in written),
                         ["confidence_fsc.png", "per_speaker_fsc.png", "seen_vs_unseen.png", "speed_vs_accuracy_fsc.png"])
        self.assertEqual(fg.draw_all("M9_test", cfg, datasets=["l2arctic"], out=out), [])

    def test_speakers_know_their_test_fold(self):
        info = fg.speaker_info("fsc", self.cfg)
        self.assertEqual(len(info), 10)
        self.assertEqual(sorted(info["fold"].value_counts()), [2] * 5)
        self.assertEqual(list(info["fold"]), sorted(info["fold"]))


if __name__ == "__main__":
    unittest.main()

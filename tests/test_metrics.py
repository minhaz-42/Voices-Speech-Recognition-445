"""Tests for support/metrics.py against hand-computed values.

Worked example used throughout (6 clips, 3 phrases, 3 speakers):

    clip     0    1    2    3    4    5
    true     A    A    A    B    B    C
    pred     A    A    B    B    C    C
    speaker  S1   S1   S2   S2   S3   S3

    accuracy = 4 / 6
    A: TP 2, FP 0, FN 1 -> P 1,   R 2/3, F1 0.8
    B: TP 1, FP 1, FN 1 -> P 1/2, R 1/2, F1 0.5
    C: TP 1, FP 1, FN 0 -> P 1/2, R 1,   F1 2/3
    macro F1    = (0.8 + 0.5 + 2/3) / 3       = 59/90
    weighted F1 = (3*0.8 + 2*0.5 + 1*2/3) / 6 = 61/90
    per speaker: S1 2/2, S2 1/2, S3 1/2
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from support import metrics

Y_TRUE = ["A", "A", "A", "B", "B", "C"]
Y_PRED = ["A", "A", "B", "B", "C", "C"]
SPEAKERS = ["S1", "S1", "S2", "S2", "S3", "S3"]


class TestComputeMetrics(unittest.TestCase):
    def test_hand_computed_example(self):
        m = metrics.compute_metrics(Y_TRUE, Y_PRED)
        self.assertEqual(m["n"], 6)
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["macro_f1"], 59 / 90)
        self.assertAlmostEqual(m["weighted_f1"], 61 / 90)

    def test_perfect_predictions(self):
        m = metrics.compute_metrics(Y_TRUE, Y_TRUE)
        self.assertEqual((m["accuracy"], m["macro_f1"], m["weighted_f1"]), (1.0, 1.0, 1.0))

    def test_all_wrong_with_unseen_prediction(self):
        # "D" is never a true label but is predicted, so it counts with F1 = 0.
        m = metrics.compute_metrics(["A", "B"], ["D", "D"])
        self.assertEqual(m["accuracy"], 0.0)
        self.assertEqual(m["macro_f1"], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            metrics.compute_metrics(["A", "B"], ["A"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            metrics.compute_metrics([], [])


class TestClassificationReport(unittest.TestCase):
    def test_per_class_values(self):
        table = metrics.classification_report_table(Y_TRUE, Y_PRED)
        np.testing.assert_allclose(table.loc["A", ["precision", "recall", "f1"]].astype(float), [1.0, 2 / 3, 0.8])
        np.testing.assert_allclose(table.loc["B", ["precision", "recall", "f1"]].astype(float), [0.5, 0.5, 0.5])
        np.testing.assert_allclose(table.loc["C", ["precision", "recall", "f1"]].astype(float), [0.5, 1.0, 2 / 3])
        self.assertEqual(list(table.loc[["A", "B", "C"], "support"]), [3, 2, 1])

    def test_averages_match_compute_metrics(self):
        table = metrics.classification_report_table(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(table.loc["macro avg", "f1"], 59 / 90)
        self.assertAlmostEqual(table.loc["weighted avg", "f1"], 61 / 90)
        self.assertEqual(table.loc["macro avg", "support"], 6)

    def test_label_order_and_absent_label(self):
        # "Z" has no clips: it is listed with zeros but does not change the averages.
        table = metrics.classification_report_table(Y_TRUE, Y_PRED, labels=["C", "B", "A", "Z"])
        self.assertEqual(list(table.index[:4]), ["C", "B", "A", "Z"])
        self.assertEqual(table.loc["Z", "support"], 0)
        self.assertAlmostEqual(table.loc["macro avg", "f1"], 59 / 90)


class TestConfusionMatrix(unittest.TestCase):
    def test_hand_computed_matrix(self):
        table = metrics.confusion_matrix_table(Y_TRUE, Y_PRED, labels=["A", "B", "C"])
        np.testing.assert_array_equal(table.to_numpy(), [[2, 1, 0], [0, 1, 1], [0, 0, 1]])
        self.assertEqual(table.index.name, "true")
        self.assertEqual(table.columns.name, "predicted")

    def test_plot_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            for normalize in (True, False):
                path = Path(tmp) / "figs" / f"cm_{normalize}.png"
                metrics.plot_confusion_matrix(Y_TRUE, Y_PRED, path=path, normalize=normalize, title="test")
                self.assertGreater(path.stat().st_size, 0)

    def test_plot_fifty_classes(self):
        labels = [f"P{i:02d}" for i in range(50)]
        fig = metrics.plot_confusion_matrix(labels, labels[1:] + labels[:1], labels=labels)
        self.assertEqual(len(fig.axes[0].get_xticklabels()), 50)


class TestGroupAccuracy(unittest.TestCase):
    def test_per_speaker(self):
        table = metrics.per_speaker_accuracy(Y_TRUE, Y_PRED, SPEAKERS)
        self.assertEqual(list(table["speaker_id"]), ["S1", "S2", "S3"])
        self.assertEqual(list(table["n"]), [2, 2, 2])
        self.assertEqual(list(table["correct"]), [2, 1, 1])
        np.testing.assert_allclose(table["accuracy"], [1.0, 0.5, 0.5])

    def test_by_category(self):
        categories = ["cmd", "cmd", "cmd", "num", "num", "num"]
        table = metrics.accuracy_by_group(Y_TRUE, Y_PRED, categories, name="category")
        self.assertEqual(list(table.columns), ["category", "n", "correct", "accuracy"])
        np.testing.assert_allclose(table["accuracy"], [2 / 3, 2 / 3])

    def test_group_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            metrics.per_speaker_accuracy(Y_TRUE, Y_PRED, SPEAKERS[:-1])


if __name__ == "__main__":
    unittest.main()

"""Tests for support/presentation.py: placeholders, values from fake results, the checklist."""

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from support import presentation as pres
from support.config import load_config


class TestFill(unittest.TestCase):
    def test_known_and_unknown_placeholders(self):
        values = {"fsc:svm:unseen:accuracy": ("22.3%", "results/svm/fsc/summary.json")}
        text, used = pres.fill("SVM {{fsc:svm:unseen:accuracy}}, k-NN {{fsc:knn:unseen:accuracy}}, "
                               "again {{fsc:svm:unseen:accuracy}}", values)
        self.assertEqual(text, "SVM 22.3%, k-NN TBD, again 22.3%")
        self.assertEqual([u["key"] for u in used], ["fsc:svm:unseen:accuracy", "fsc:knn:unseen:accuracy"])
        self.assertEqual(used[1]["source"], "")

    def test_text_without_placeholders_is_unchanged(self):
        self.assertEqual(pres.fill("no numbers {here}", {}), ("no numbers {here}", []))


class TestCollectAndChecklist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from support import evaluate

        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cfg = copy.deepcopy(load_config())
        cfg["paths"]["results"] = str(cls.dir / "results")
        cfg["paths"]["images"] = str(cls.dir / "images")
        cfg["result_folders"] = {}
        pred = pd.DataFrame({"utt_id": ["a", "b", "c", "d"], "speaker_id": ["s1", "s1", "s2", "s2"],
                             "label": ["A", "B", "A", "B"], "pred": ["A", "B", "A", "A"]})
        evaluate.save_run("svm", "fsc", {"random": pred, "fold1": pred}, labels=["A", "B"], cfg=cfg)
        demo = cls.dir / "results" / "demo_check"
        demo.mkdir(parents=True)
        pd.DataFrame({"file": ["x.wav", "y.wav"], "recognizer": "whisper_small", "expected": ["A", "B"],
                      "phrase_id": ["A", "A"], "text": "", "score": 1.0,
                      "correct": [True, False]}).to_csv(demo / "me.csv", index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_values_from_results(self):
        values = pres.collect_values(self.dir / "results")
        self.assertEqual(values["fsc:svm:random:accuracy"][0], "75.0%")
        self.assertEqual(values["fsc:svm:unseen:accuracy"][0], "75.0%")
        self.assertTrue(values["fsc:svm:unseen:accuracy"][1].endswith("svm/fsc/summary.json"))
        self.assertEqual(values["demo:whisper_small"][0], "1/2")

    def test_build_without_rendering_writes_html_and_checklist(self):
        others = self.dir / "others"
        others.mkdir()
        (others / "slides.template.html").write_text("<p>{{fsc:svm:unseen:accuracy}} {{fsc:knn:unseen:accuracy}}</p>",
                                                     encoding="utf-8")
        (others / "README.md").write_text("# others/\n", encoding="utf-8")
        written = pres.build(self.dir / "results", others, do_render=False)
        self.assertIn(others / "slides.html", written)
        self.assertEqual((others / "slides.html").read_text(encoding="utf-8"), "<p>75.0% TBD</p>")
        readme = (others / "README.md").read_text(encoding="utf-8")
        self.assertTrue(readme.startswith("# others/"))
        self.assertIn("| `fsc:svm:unseen:accuracy` | 75.0% |", readme)
        self.assertIn("not merged yet", readme)
        pres.build(self.dir / "results", others, do_render=False)  # a rerun replaces the block
        self.assertEqual((others / "README.md").read_text(encoding="utf-8").count(pres.CHECKLIST_START), 1)


class TestShrinkPng(unittest.TestCase):
    def test_large_png_goes_under_the_limit_and_small_one_is_untouched(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.png"
            noise = np.random.default_rng(0).integers(0, 256, (600, 800, 3), dtype=np.uint8)
            Image.fromarray(noise).save(big)
            self.assertGreater(big.stat().st_size, 1_000_000)
            size = pres.shrink_png(big)
            self.assertLessEqual(size, 1_000_000)
            with Image.open(big) as image:
                self.assertEqual(image.size, (800, 600))

            small = Path(tmp) / "small.png"
            Image.new("RGB", (40, 30), "white").save(small)
            before = small.read_bytes()
            pres.shrink_png(small)
            self.assertEqual(small.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

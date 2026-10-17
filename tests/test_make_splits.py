"""Tests for tools/make_splits.py: split sizes, fold design and leakage guards."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._synthetic import L1_ORDER, L2_SPEAKERS, make_config, make_subset
from tools import make_splits as ms


def speakers_of(subset):
    return subset.groupby("speaker_id", as_index=False)[["gender", "l1"]].first()


class TestRandomSplit(unittest.TestCase):
    def test_fsc_sizes_and_stratification(self):
        subset = make_subset("fsc")  # 10 speakers x 50 phrases x 2 takes = 1,000 clips
        split = ms.random_split(subset, seed=42)
        counts = split["part"].value_counts()
        self.assertEqual((counts["test"], counts["val"], counts["train"]), (200, 80, 720))
        # 20 clips per phrase -> exactly 4 test clips per phrase
        self.assertEqual(set(split[split["part"] == "test"].groupby("phrase_id").size()), {4})
        ms.check_split(split, subset, speaker_independent=False)

    def test_l2_falls_back_to_unstratified_validation(self):
        subset = make_subset("l2arctic", takes=1)  # 500 clips; 40 validation clips < 50 phrases
        split = ms.random_split(subset, seed=42)
        counts = split["part"].value_counts()
        self.assertEqual((counts["test"], counts["val"], counts["train"]), (100, 40, 360))
        self.assertEqual(set(split[split["part"] == "test"].groupby("phrase_id").size()), {2})

    def test_same_seed_same_split(self):
        subset = make_subset("fsc", n_phrases=10)
        pd.testing.assert_frame_equal(ms.random_split(subset, 42), ms.random_split(subset, 42))
        self.assertFalse(ms.random_split(subset, 42).equals(ms.random_split(subset, 7)))


class TestSpeakerFolds(unittest.TestCase):
    def check_design(self, folds, speakers):
        ms.check_folds(folds, speakers, fold_by_l1=False)
        self.assertEqual(len(folds), 5)
        vals = [v for f in folds["val"] for v in f]
        self.assertEqual(len(vals), len(set(vals)))           # validation speaker at most once
        for fold in folds.itertuples():
            self.assertEqual((len(fold.test), len(fold.val), len(fold.train)), (2, 1, 7))

    def test_fsc_folds(self):
        speakers = speakers_of(make_subset("fsc", n_phrases=2))
        self.check_design(ms.speaker_folds(speakers, seed=42), speakers)

    def test_l2_folds_follow_first_languages(self):
        speakers = speakers_of(make_subset("l2arctic", n_phrases=2, takes=1))
        folds = ms.speaker_folds(speakers, seed=42, fold_by_l1=True, l1_order=L1_ORDER)
        self.check_design(folds, speakers)
        ms.check_folds(folds, speakers, fold_by_l1=True)
        l1 = {s: lang for s, _, lang in L2_SPEAKERS}
        self.assertEqual([l1[f.test[0]] for f in folds.itertuples()],
                         ["Hindi", "Hindi", "Arabic", "Mandarin", "Spanish"])

    def test_unbalanced_genders_raise(self):
        speakers = speakers_of(make_subset("fsc", n_phrases=1))
        speakers.loc[0, "gender"] = "M" if speakers.loc[0, "gender"] == "F" else "F"
        with self.assertRaises(ValueError):
            ms.speaker_folds(speakers)

    def test_language_group_that_cannot_be_paired_raises(self):
        speakers = speakers_of(make_subset("l2arctic", n_phrases=1, takes=1))
        speakers.loc[speakers["speaker_id"] == "ABA", "l1"] = "Mandarin"  # Arabic: 1 F, Mandarin: 1 F + 2 M
        with self.assertRaises(ValueError):
            ms.speaker_folds(speakers, fold_by_l1=True, l1_order=L1_ORDER)


class TestLeakageGuards(unittest.TestCase):
    def test_speaker_in_two_parts_is_caught(self):
        subset = make_subset("fsc", n_phrases=2)
        folds = ms.speaker_folds(speakers_of(subset))
        split = ms.fold_split(subset, folds.iloc[0])
        ms.check_split(split, subset, speaker_independent=True)
        leaked = split.copy()
        test_spk = folds.iloc[0]["test"][0]
        leaked.loc[leaked[leaked["speaker_id"] == test_spk].index[0], "part"] = "train"
        with self.assertRaises(ValueError):
            ms.check_split(leaked, subset, speaker_independent=True)

    def test_missing_or_repeated_clip_is_caught(self):
        subset = make_subset("fsc", n_phrases=2)
        split = ms.random_split(subset)
        with self.assertRaises(ValueError):
            ms.check_split(split.iloc[1:], subset, speaker_independent=False)
        with self.assertRaises(ValueError):
            ms.check_split(pd.concat([split, split.iloc[:1]]), subset, speaker_independent=False)


class TestMakeSplits(unittest.TestCase):
    def test_writes_files_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = make_config(tmp, "l2arctic", make_subset("l2arctic", takes=1))
            splits = ms.make_splits("l2arctic", cfg)
            out = tmp / "splits" / "l2arctic"
            names = ["random"] + [f"fold{k}" for k in range(1, 6)] + ["folds"]
            self.assertEqual(sorted(p.stem for p in out.glob("*.csv")), sorted(names))
            first = {n: (out / f"{n}.csv").read_bytes() for n in names}
            ms.make_splits("l2arctic", cfg)
            self.assertEqual(first, {n: (out / f"{n}.csv").read_bytes() for n in names})
            fold1 = pd.read_csv(out / "fold1.csv")
            self.assertEqual(list(fold1.columns), ["utt_id", "speaker_id", "phrase_id", "part"])
            self.assertEqual(len(splits["folds"]), 5)


if __name__ == "__main__":
    unittest.main()

"""Tests for support/dataset.py with a tiny synthetic subset written to a temporary folder."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from support import dataset as ds
from tests._synthetic import make_config, make_subset, write_audio
from tools.make_splits import make_splits


class TestSubsetLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_subset_loads(self):
        cfg = make_config(self.dir, "fsc", make_subset("fsc", n_phrases=3))
        subset = ds.load_subset("fsc", cfg)
        self.assertEqual(len(subset), 10 * 3 * 2)
        self.assertEqual(list(ds.load_labels("fsc", cfg)["phrase_id"]), ["P01", "P02", "P03"])

    def test_missing_subset_points_to_t42(self):
        cfg = make_config(self.dir, "fsc", make_subset("fsc", n_phrases=1))
        (self.dir / "subset.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "T42"):
            ds.load_subset("fsc", cfg)

    def test_bad_subsets_raise(self):
        subset = make_subset("fsc", n_phrases=1)
        for broken in (subset.drop(columns="sha256"),
                       pd.concat([subset, subset.iloc[:1]]),
                       subset.assign(dataset="l2arctic")):
            cfg = make_config(self.dir, "fsc", broken)
            with self.assertRaises(ValueError):
                ds.load_subset("fsc", cfg)

    def test_unknown_dataset_and_split(self):
        with self.assertRaises(KeyError):
            ds.dataset_config("nope")
        with self.assertRaises(ValueError):
            ds.split_path("fsc", "fold9")


class TestProcessedCopies(unittest.TestCase):
    """10 speakers x 2 phrases x 1 take of synthetic audio at 22.05 kHz."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        subset = write_audio(make_subset("fsc", n_phrases=2, takes=1), self.dir / "raw")
        self.cfg = make_config(self.dir, "fsc", subset)

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_writes_16k_mono_and_manifest(self):
        manifest = ds.build_processed("fsc", self.cfg)
        self.assertEqual(list(manifest.columns), ds.MANIFEST_COLUMNS)
        self.assertEqual(len(manifest), 20)
        info = sf.info(manifest.loc[0, "path"])
        self.assertEqual((info.samplerate, info.channels), (16000, 1))
        # 0.5 s tone + 2 x 0.1 s margin, silence trimmed (within the 32 ms trim frame)
        self.assertTrue(np.allclose(manifest["duration_s"], 0.7, atol=0.05))
        reloaded = ds.load_manifest("fsc", self.cfg)
        self.assertEqual(list(reloaded["utt_id"]), list(manifest["utt_id"]))

    def test_checksum_mismatch_raises(self):
        first = self.dir / "raw" / ds.load_subset("fsc", self.cfg).loc[0, "source_path"]
        sf.write(first, np.zeros(1000), 22050)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ds.build_processed("fsc", self.cfg)
        ds.build_processed("fsc", self.cfg, verify=False)  # explicitly skipped

    def test_missing_source_raises(self):
        (self.dir / "raw" / ds.load_subset("fsc", self.cfg).loc[0, "source_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            ds.build_processed("fsc", self.cfg)

    def test_split_needs_manifest_then_joins(self):
        make_splits("fsc", self.cfg)
        with self.assertRaisesRegex(FileNotFoundError, "--build"):
            ds.load_split("fsc", "fold1", self.cfg)
        ds.build_processed("fsc", self.cfg)
        parts = ds.load_split("fsc", "fold1", self.cfg)
        self.assertEqual(sum(len(p) for p in parts.values()), 20)
        self.assertEqual(parts["test"]["speaker_id"].nunique(), 2)
        self.assertEqual(set(parts["train"]["speaker_id"]) & set(parts["test"]["speaker_id"]), set())
        clip = ds.read_processed(parts["test"].loc[0, "path"])
        self.assertEqual((clip.dtype, clip.ndim), (np.float32, 1))


if __name__ == "__main__":
    unittest.main()

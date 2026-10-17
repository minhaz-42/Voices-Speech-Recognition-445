"""Tests for support/features.py."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from support import features
from support.config import load_config


def tone(freq, seconds=0.8, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestMfccFeatures(unittest.TestCase):
    def test_shape_names_and_dtype(self):
        f = features.mfcc_features(tone(440))
        self.assertEqual(f.shape, (78,))
        self.assertEqual(f.dtype, np.float32)
        self.assertEqual(len(features.feature_names()), 78)
        self.assertEqual(features.feature_names()[:2], ["mfcc1_mean", "mfcc2_mean"])
        self.assertTrue(np.all(np.isfinite(f)))

    def test_steady_tone_has_near_zero_deltas(self):
        f = features.mfcc_features(tone(440, seconds=1.5))
        self.assertLess(np.abs(f[26:39]).max(), np.abs(f[:13]).max() / 10)  # mean of delta vs mean of MFCC

    def test_different_pitches_give_different_features(self):
        self.assertGreater(np.linalg.norm(features.mfcc_features(tone(300)) - features.mfcc_features(tone(1500))), 1.0)

    def test_very_short_clip_still_works(self):
        self.assertEqual(features.mfcc_features(tone(440, seconds=0.05)).shape, (78,))


class TestExtractCache(unittest.TestCase):
    def test_cache_is_reused_and_extended(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for i, f in enumerate((300, 600, 900)):
                paths.append(str(tmp / f"c{i}.wav"))
                sf.write(paths[-1], tone(f), 16000)
            cache = tmp / "feats.npz"
            first = features.extract(["a", "b"], paths[:2], cache, cfg)
            self.assertEqual(first.shape, (2, 78))
            with mock.patch.object(features, "mfcc_features", wraps=features.mfcc_features) as spy:
                again = features.extract(["b", "a", "c"], [paths[1], paths[0], paths[2]], cache, cfg)
                self.assertEqual(spy.call_count, 1)  # only the new clip is computed
            np.testing.assert_allclose(again[:2], first[::-1])

    def test_cache_with_other_settings_is_ignored(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "c.wav")
            sf.write(path, tone(500), 16000)
            cache = Path(tmp) / "feats.npz"
            features.extract(["a"], [path], cache, cfg)
            other = {**cfg, "features": {**cfg["features"], "n_mfcc": 20}}
            with mock.patch.object(features, "mfcc_features", return_value=np.zeros(120, np.float32)) as spy:
                out = features.extract(["a"], [path], cache, other)
                self.assertEqual(spy.call_count, 1)
            self.assertEqual(out.shape, (1, 120))


if __name__ == "__main__":
    unittest.main()

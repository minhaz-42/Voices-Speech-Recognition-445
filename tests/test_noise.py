"""Tests for support/noise.py."""

import unittest

import numpy as np

from support import noise


def tone(seconds=1.0, sr=16000, amplitude=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestWhiteNoise(unittest.TestCase):
    def test_requested_snr_is_met(self):
        rng = np.random.default_rng(0)
        clean = tone(seconds=2.0)
        for snr in (20, 10, 5, 0):
            noisy = noise.add_white_noise(clean, snr, rng)
            self.assertAlmostEqual(noise.measured_snr_db(clean, noisy), snr, delta=0.2)
            self.assertEqual((noisy.dtype, noisy.shape), (np.float32, clean.shape))

    def test_signal_power_of_a_sine(self):
        self.assertAlmostEqual(noise.signal_power(tone(amplitude=0.5)), 0.125, places=3)  # A^2 / 2

    def test_silence_and_empty_are_unchanged(self):
        rng = np.random.default_rng(0)
        self.assertFalse(np.any(noise.add_white_noise(np.zeros(100), 10, rng)))
        self.assertEqual(noise.add_white_noise(np.zeros(0), 10, rng).size, 0)
        self.assertEqual(noise.measured_snr_db(tone(), tone()), float("inf"))

    def test_same_seed_same_noise(self):
        a = noise.add_white_noise(tone(), 10, np.random.default_rng(42))
        b = noise.add_white_noise(tone(), 10, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


class TestAugment(unittest.TestCase):
    def test_probability_zero_and_one(self):
        clean = tone()
        out, snr = noise.augment(clean, np.random.default_rng(0), p=0.0)
        self.assertIsNone(snr)
        np.testing.assert_array_equal(out, clean)
        out, snr = noise.augment(clean, np.random.default_rng(0), p=1.0, snr_range_db=(5, 30))
        self.assertTrue(5 <= snr <= 30)
        self.assertAlmostEqual(noise.measured_snr_db(clean, out), snr, delta=0.3)

    def test_about_half_the_clips_get_noise(self):
        rng = np.random.default_rng(1)
        hits = sum(noise.augment(tone(0.05), rng, p=0.5)[1] is not None for _ in range(2000))
        self.assertTrue(900 < hits < 1100)


if __name__ == "__main__":
    unittest.main()

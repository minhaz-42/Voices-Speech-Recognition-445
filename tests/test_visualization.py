"""Tests for support/visualization.py, using a synthetic tone and tiny tables."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib.figure import Figure

from support import visualization as viz

SR = 16000


def speech_like(sample_rate=SR, lead_s=0.3, speech_s=1.0, tail_s=0.3):
    """Silence, a 1 s sine tone standing in for speech, then silence."""
    t = np.arange(int(round(speech_s * sample_rate))) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    lead = np.zeros(int(round(lead_s * sample_rate)), dtype=np.float32)
    tail = np.zeros(int(round(tail_s * sample_rate)), dtype=np.float32)
    return np.concatenate([lead, tone, tail])


class TestClipFigures(unittest.TestCase):
    def setUp(self):
        self.samples = speech_like()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertSaved(self, path):
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)

    def test_waveform(self):
        path = self.dir / "sub" / "wave.png"
        fig = viz.plot_waveform(self.samples, SR, path=path)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].get_xlabel(), "Time (s)")
        self.assertSaved(path)

    def test_spectrogram_linear_and_mel(self):
        for kind in ("linear", "mel"):
            path = self.dir / f"spec_{kind}.png"
            fig = viz.plot_spectrogram(self.samples, SR, kind=kind, path=path)
            self.assertEqual(len(fig.axes), 2)  # the spectrogram and its colour bar
            self.assertIn("dB", fig.axes[0].get_title())
            self.assertSaved(path)

    def test_spectrogram_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            viz.plot_spectrogram(self.samples, SR, kind="cqt")

    def test_mfcc(self):
        path = self.dir / "mfcc.png"
        fig = viz.plot_mfcc(self.samples, SR, n_mfcc=13, path=path)
        self.assertEqual(len(fig.axes), 2)  # the heatmap and its colour bar
        self.assertIn("13", fig.axes[0].get_title())
        self.assertSaved(path)

    def test_overview_has_four_panels(self):
        path = self.dir / "overview.png"
        fig = viz.clip_overview(self.samples, SR, title="tone", path=path)
        self.assertEqual(len(fig.axes), 4)
        self.assertSaved(path)

    def test_preprocessing_before_after(self):
        wav = self.dir / "clip.wav"
        sf.write(wav, speech_like(48000), 48000)
        out = self.dir / "pre.png"
        fig = viz.plot_preprocessing(wav, out_path=out)
        self.assertEqual(len(fig.axes), 2)
        before, after = (ax.get_title() for ax in fig.axes)
        self.assertIn("48 kHz", before)
        self.assertIn("16 kHz", after)
        self.assertIn("6.00 s", after)  # padded to the configured 6 s
        self.assertSaved(out)

    def test_silent_clip_does_not_crash(self):
        fig = viz.clip_overview(np.zeros(SR, dtype=np.float32), SR)
        self.assertEqual(len(fig.axes), 4)


if __name__ == "__main__":
    unittest.main()

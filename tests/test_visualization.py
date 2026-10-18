"""Tests for support/visualization.py, using a synthetic tone and tiny tables."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
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


class TestResultCharts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.models = pd.DataFrame({
            "model": ["svm", "random_forest"],
            "accuracy": [0.4, 0.3],
            "low": [0.35, 0.25],
            "high": [0.45, 0.36],
        })
        self.snr = pd.DataFrame({
            "model": ["a", "a", "a", "b", "b", "b"],
            "snr_db": ["clean", "20", "5", "clean", "20", "5"],
            "accuracy": [0.9, 0.8, 0.5, 0.7, 0.4, 0.1],
        })
        self.speakers = pd.DataFrame({
            "speaker_id": ["s1", "s2", "s1", "s2"],
            "model": ["a", "a", "b", "b"],
            "accuracy": [0.5, 0.6, 0.7, 0.8],
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_model_comparison(self):
        path = self.dir / "models.png"
        fig = viz.plot_model_comparison(self.models, path=path)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].patches), 2)
        self.assertGreater(path.stat().st_size, 0)

    def test_model_comparison_with_error_bars(self):
        fig = viz.plot_model_comparison(self.models, error_cols=("low", "high"))
        self.assertEqual(len(fig.axes), 1)

    def test_accuracy_vs_snr_one_line_per_model(self):
        path = self.dir / "snr.png"
        fig = viz.plot_accuracy_vs_snr(self.snr, path=path)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_lines()), 2)
        self.assertIn("clean", [t.get_text() for t in ax.get_xticklabels()])
        self.assertGreater(path.stat().st_size, 0)

    def test_per_speaker_grouped_and_single(self):
        path = self.dir / "speakers.png"
        fig = viz.plot_per_speaker(self.speakers, path=path)
        self.assertEqual(len(fig.axes[0].patches), 4)  # 2 speakers x 2 models
        self.assertGreater(path.stat().st_size, 0)
        fig = viz.plot_per_speaker(self.speakers[self.speakers["model"] == "a"].drop(columns="model"))
        self.assertEqual(len(fig.axes[0].patches), 2)

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            viz.plot_model_comparison(self.models.drop(columns="accuracy"))
        with self.assertRaises(ValueError):
            viz.plot_model_comparison(self.models, error_cols=("low", "upper"))
        with self.assertRaises(ValueError):
            viz.plot_accuracy_vs_snr(self.snr.drop(columns="snr_db"))
        with self.assertRaises(ValueError):
            viz.plot_per_speaker(self.speakers.drop(columns="speaker_id"))


if __name__ == "__main__":
    unittest.main()

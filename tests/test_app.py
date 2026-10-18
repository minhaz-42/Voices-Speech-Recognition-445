"""Tests for app.py (the Gradio demo), with synthetic int16 microphone audio."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import app
from support.inference import load_recognizer
from tests.test_inference import PITCHES, save_classical_model, tone


def mic_clip(sample_rate=48000, speech_s=1.0, pad_s=0.3):
    """int16 audio as the microphone widget returns it: silence, a tone, silence."""
    t = np.arange(int(speech_s * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    pad = np.zeros(int(pad_s * sample_rate))
    return (np.concatenate([pad, tone, pad]) * 32767).astype(np.int16)


class TestProcessAudio(unittest.TestCase):
    def test_no_audio_gives_message_and_no_figure(self):
        processed, figure, info = app.process_audio(None)
        self.assertIsNone(processed)
        self.assertIsNone(figure)
        self.assertEqual(info, app.EMPTY_MESSAGE)

    def test_empty_audio_gives_message(self):
        processed, figure, info = app.process_audio((16000, np.zeros(0, dtype=np.int16)))
        self.assertIsNone(figure)
        self.assertEqual(info, app.EMPTY_MESSAGE)

    def test_microphone_clip_is_processed(self):
        processed, figure, info = app.process_audio((48000, mic_clip()))
        rate, samples = processed
        self.assertEqual(rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(samples.size / rate, 1.2, delta=0.1)  # 1 s tone + 0.1 s margins, not padded
        self.assertEqual(len(figure.axes), 4)
        self.assertIn("48 kHz", info)
        self.assertIn("dBFS", info)

    def test_stereo_clip_is_processed(self):
        clip = mic_clip(44100)
        processed, figure, _ = app.process_audio((44100, np.stack([clip, clip], axis=1)))
        self.assertEqual(processed[0], 16000)
        self.assertEqual(len(figure.axes), 4)

    def test_too_long_clip_gives_message(self):
        processed, figure, info = app.process_audio((16000, mic_clip(16000, speech_s=7.0)))
        self.assertIsNone(processed)
        self.assertIsNone(figure)
        self.assertIn("6 s", info)

    def test_silent_clip_gives_message(self):
        processed, figure, info = app.process_audio((16000, np.zeros(16000, dtype=np.int16)))
        self.assertIsNone(figure)
        self.assertIn("silent", info)


class TestApp(unittest.TestCase):
    def test_phrase_table_has_the_50_commands(self):
        rows = app.phrase_table()
        self.assertEqual(len(rows), 50)
        self.assertEqual(rows[0][0], "FSC01")

    def test_build_app_without_launching(self):
        import gradio as gr

        demo = app.build_app(model_path=None)
        self.assertIsInstance(demo, gr.Blocks)


class TestRecognise(unittest.TestCase):
    """The demo with a small SVM saved like support.classical does (see tests/test_inference.py)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.tmp.name) / "svm.joblib"
        save_classical_model(cls.model_path)
        cls.recognizer = load_recognizer(cls.model_path)
        cls.texts = {"P01": "low tone", "P02": "middle tone", "P03": "high tone"}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_top_phrases_and_chart(self):
        clip = (tone(PITCHES["P02"], sr=48000) * 32767).astype(np.int16)
        rows, chart = app.recognise((48000, clip), self.recognizer, self.texts, k=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][:3], [1, "P02", "middle tone"])
        self.assertTrue(rows[0][3].endswith("%"))
        self.assertEqual(len(chart.axes), 1)
        self.assertEqual(len(chart.axes[0].patches), 3)

    def test_no_audio_gives_no_prediction(self):
        self.assertEqual(app.recognise(None, self.recognizer), ([], None))

    def test_load_model(self):
        recognizer, status = app.load_model(self.model_path)
        self.assertIsNotNone(recognizer)
        self.assertIn("svm.joblib", status)
        missing, status = app.load_model(Path(self.tmp.name) / "missing.joblib")
        self.assertIsNone(missing)
        self.assertIn("No model", status)

    def test_build_app_with_a_model(self):
        import gradio as gr

        self.assertIsInstance(app.build_app(recognizer=self.recognizer), gr.Blocks)


if __name__ == "__main__":
    unittest.main()

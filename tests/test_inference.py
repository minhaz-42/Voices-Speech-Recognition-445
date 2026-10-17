"""Tests for support/inference.py and main.py (traditional model, Whisper with a fake transcriber, video input).

The wav2vec 2.0 recogniser is tested in the child process of tests/_wav2vec2_check.py
(PyTorch must not share a process with XGBoost on macOS).
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

import main
from support import inference
from support.audio import preprocess
from support.classical import make_model
from support.config import PROJECT_ROOT, load_config
from support.features import mfcc_features

PITCHES = {"P01": 300, "P02": 900, "P03": 1800}


def tone(freq, sr=16000, seconds=0.6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    pad = np.zeros(int(0.2 * sr))
    return np.concatenate([pad, 0.4 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.size), pad])


def save_classical_model(path: Path) -> None:
    """An SVM trained on MFCC features of three pitches, saved like support.classical does.

    Like the real pipeline, features come from pre-processed (trimmed, normalised) clips.
    """
    cfg = load_config()
    feat_cfg = cfg["features"]
    x, y = [], []
    for i, freq in enumerate(PITCHES.values()):
        for seed in range(6):
            clip = preprocess(tone(freq * (1 + 0.01 * seed), seed=seed), 16000, cfg["audio"], pad=False)
            x.append(mfcc_features(clip, 16000, feat_cfg))
            y.append(i)
    model = make_model("svm", {"C": 10}).fit(np.stack(x), np.array(y))
    joblib.dump({"model": model, "labels": list(PITCHES), "params": {"C": 10}, "features": feat_cfg}, path)


class TestTopK(unittest.TestCase):
    def test_order_texts_and_cut(self):
        out = inference.top_k(np.array([0.1, 0.6, 0.3]), ["A", "B", "C"], k=2, texts={"B": "bee"})
        self.assertEqual([r["phrase_id"] for r in out], ["B", "C"])
        self.assertEqual(out[0]["text"], "bee")
        self.assertEqual(out[1]["text"], "")


class TestClassicalRecognizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.model_path = cls.dir / "svm.joblib"
        save_classical_model(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_recognises_raw_audio_at_any_rate(self):
        rec = inference.load_recognizer(self.model_path)
        self.assertIsInstance(rec, inference.ClassicalRecognizer)
        for pid, freq in PITCHES.items():
            stereo_48k = np.stack([tone(freq, sr=48000, seed=9)] * 2, axis=1)
            top = rec.predict(stereo_48k, 48000, k=3)
            self.assertEqual(top[0]["phrase_id"], pid)
            self.assertEqual(len(top), 3)
            self.assertAlmostEqual(sum(r["score"] for r in top), 1.0, places=5)

    def test_int16_microphone_audio(self):
        pcm = (tone(900) * 32767).astype(np.int16)
        self.assertEqual(inference.load_recognizer(self.model_path).predict(pcm, 16000)[0]["phrase_id"], "P02")

    def test_unknown_model_path_raises(self):
        with self.assertRaises(ValueError):
            inference.load_recognizer(self.dir)

    def test_main_cli_with_benchmark(self):
        wav = self.dir / "clip.wav"
        sf.write(wav, tone(1800), 16000)
        proc = subprocess.run([sys.executable, "main.py", "--audio", str(wav), "--model", str(self.model_path),
                               "--benchmark", "3"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("1. P03", proc.stdout)
        self.assertIn("ms per clip", proc.stdout)

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is not installed")
    def test_video_input(self):
        wav, mp4 = self.dir / "v.wav", self.dir / "v.mp4"
        sf.write(wav, tone(300, sr=44100), 44100)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=1",
                        "-i", str(wav), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(mp4)], check=True)
        samples, sr = inference.read_input(mp4)
        self.assertEqual(sr, 16000)
        self.assertEqual(inference.load_recognizer(self.model_path).predict(samples, sr)[0]["phrase_id"], "P01")


class TestWhisperRecognizer(unittest.TestCase):
    def test_transcript_and_nearest_phrases(self):
        phrases = [{"phrase_id": "A", "text": "turn on the lights"}, {"phrase_id": "B", "text": "turn off the lights"},
                   {"phrase_id": "C", "text": "bring me the newspaper"}]
        seen = []

        def fake(clips):
            seen.extend(clips)
            return ["Turn off the light."]

        rec = inference.WhisperRecognizer(phrases, transcriber=fake)
        out = rec.predict(tone(500, sr=44100), 44100, k=2)
        self.assertEqual(out["transcript"], "Turn off the light.")
        self.assertEqual([r["phrase_id"] for r in out["top"]], ["B", "A"])
        self.assertAlmostEqual(out["top"][0]["score"], 0.75)  # 1 - WER (1 of 4 words wrong)
        self.assertFalse(out["rejected"])
        self.assertEqual(seen[0].ndim, 1)  # pre-processed mono 16 kHz clip


class TestMainHelpers(unittest.TestCase):
    def test_benchmark_and_size(self):
        class Stub:
            def predict(self, samples, sr):
                return []

        b = main.benchmark(Stub(), np.zeros(10), 16000, repeats=4)
        self.assertEqual(b["repeats"], 4)
        self.assertGreaterEqual(b["mean_ms"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.bin").write_bytes(b"x" * 2_000_000)
            self.assertAlmostEqual(main.model_size_mb(Path(tmp)), 2.0)

    def test_phrase_texts_missing_list_is_empty(self):
        cfg = load_config()
        cfg["datasets"]["fsc"]["phrases"] = "data/prompts/does_not_exist.csv"
        self.assertEqual(main.phrase_texts("fsc", cfg), {})


if __name__ == "__main__":
    unittest.main()

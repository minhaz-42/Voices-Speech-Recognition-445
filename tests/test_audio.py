"""Tests for support/audio.py, using synthetic tones written to a temporary folder."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from support import audio
from support.config import load_config


def tone(sample_rate, seconds, freq=440.0, amplitude=0.3):
    """A sine tone of the given length."""
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(sample_rate, seconds):
    """Digital silence of the given length."""
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


def speech_like(sample_rate, lead_s=0.5, speech_s=1.0, tail_s=0.7):
    """Silence, a tone standing in for speech, then silence."""
    return np.concatenate([silence(sample_rate, lead_s), tone(sample_rate, speech_s), silence(sample_rate, tail_s)])


class TestConversions(unittest.TestCase):
    def test_int16_to_float32(self):
        out = audio.to_float32(np.array([0, 16384, -32768], dtype=np.int16))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, [0.0, 0.5, -1.0])

    def test_unsigned_pcm_raises(self):
        with self.assertRaises(ValueError):
            audio.to_float32(np.array([128, 255], dtype=np.uint8))

    def test_to_mono_averages_channels(self):
        stereo = np.array([[1.0, 0.0], [0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(audio.to_mono(stereo), [0.5, 0.5, 0.0])

    def test_to_mono_keeps_mono(self):
        mono = tone(16000, 0.1)
        self.assertIs(audio.to_mono(mono), mono)

    def test_resample_changes_length(self):
        out = audio.resample(tone(48000, 1.0), 48000, 16000)
        self.assertEqual(out.size, 16000)


class TestTrimNormalisePad(unittest.TestCase):
    def test_trim_keeps_speech_and_margin(self):
        sr = 16000
        trimmed = audio.trim_silence(speech_like(sr, 1.0, 1.0, 1.0), sr, top_db=30, margin_s=0.1)
        # 1 s of speech + 2 x 0.1 s margin, to within the 32 ms trim frame.
        self.assertAlmostEqual(trimmed.size / sr, 1.2, delta=0.05)

    def test_trim_silent_clip_is_unchanged(self):
        quiet = silence(16000, 1.0)
        self.assertEqual(audio.trim_silence(quiet, 16000, top_db=30).size, quiet.size)

    def test_peak_dbfs(self):
        self.assertAlmostEqual(audio.peak_dbfs(np.array([0.0, -0.5, 0.25])), -6.0206, places=3)
        self.assertEqual(audio.peak_dbfs(silence(16000, 0.1)), float("-inf"))

    def test_peak_normalise(self):
        out = audio.peak_normalise(tone(16000, 0.5, amplitude=0.1), target_dbfs=-1.0)
        self.assertAlmostEqual(audio.peak_dbfs(out), -1.0, places=4)

    def test_peak_normalise_silence_is_unchanged(self):
        out = audio.peak_normalise(silence(16000, 0.1), target_dbfs=-1.0)
        self.assertFalse(np.any(out))

    def test_pad_to_exact_length(self):
        out = audio.pad_to_duration(tone(16000, 2.0), 16000, 6.0)
        self.assertEqual(out.size, 96000)
        self.assertFalse(np.any(out[32000:]))

    def test_pad_never_truncates(self):
        with self.assertRaises(ValueError):
            audio.pad_to_duration(tone(16000, 6.5), 16000, 6.0)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()["audio"]
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sample_wav_loads(self):
        """A 48 kHz stereo WAV comes out as 6 s of mono 16 kHz audio peaking at -1 dBFS."""
        sr = 48000
        clip = speech_like(sr)
        path = self.dir / "S99_CMD02_r1_clean.wav"
        sf.write(path, np.stack([clip, clip], axis=1), sr, subtype="PCM_16")

        out = audio.load_audio(path, self.cfg)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.size, 6 * 16000)
        self.assertAlmostEqual(audio.peak_dbfs(out), self.cfg["peak_dbfs"], places=2)
        # The leading silence is gone: the speech starts within the kept margin.
        first_sound = np.argmax(np.abs(out) > 0.01) / 16000
        self.assertLess(first_sound, self.cfg["trim_margin_s"] + 0.05)

    def test_flac_and_no_pad(self):
        sr = 44100
        path = self.dir / "clip.flac"
        sf.write(path, speech_like(sr), sr)
        out = audio.load_audio(path, self.cfg, pad=False)
        self.assertAlmostEqual(out.size / 16000, 1.0 + 2 * self.cfg["trim_margin_s"], delta=0.05)

    def test_int16_microphone_input(self):
        clip = (speech_like(16000) * 32767).astype(np.int16)
        out = audio.preprocess(clip, 16000, self.cfg)
        self.assertEqual(out.size, 96000)

    def test_too_long_clip_raises(self):
        clip = tone(16000, 7.0)
        with self.assertRaises(ValueError):
            audio.preprocess(clip, 16000, self.cfg)

    def test_save_and_read_round_trip(self):
        path = self.dir / "nested" / "out.wav"
        audio.save_wav(path, tone(16000, 0.5), 16000)
        samples, sr = audio.read_audio(path)
        self.assertEqual(sr, 16000)
        self.assertEqual(samples.size, 8000)


if __name__ == "__main__":
    unittest.main()

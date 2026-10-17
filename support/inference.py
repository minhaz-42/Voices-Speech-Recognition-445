"""Recognise a phrase in new audio with a trained model: shared by main.py, the demo app and the noise tests.

Three recognisers, all returning the top-k phrases with a score:

- :class:`ClassicalRecognizer`: a saved traditional model (``outputs/models/<model>/<dataset>/<split>.joblib``),
- :class:`Wav2Vec2Recognizer`: a saved wav2vec 2.0 model folder (``outputs/models/<run>/<dataset>/<split>/``),
- :class:`WhisperRecognizer`: zero-shot Whisper, which also returns the transcript.

Audio goes through the same pre-processing as the training data (mono, 16 kHz, trimmed,
peak-normalised). Use :func:`load_recognizer` to open a saved model by its path.

On macOS, load either a wav2vec 2.0 / Whisper model or an XGBoost model in one process,
not both: PyTorch and XGBoost bring clashing OpenMP libraries.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from support.audio import pad_to_duration, preprocess, read_audio
from support.config import load_config


def top_k(scores: np.ndarray, labels: Sequence[str], k: int = 5,
          texts: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """The ``k`` best labels of one score vector, best first, as ``{"phrase_id", "text", "score"}``."""
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    texts = texts or {}
    return [{"phrase_id": labels[i], "text": texts.get(labels[i], ""), "score": float(scores[i])} for i in order]


class Recognizer:
    """Base class: subclasses set ``labels`` and implement :meth:`scores`."""

    labels: list[str]

    def __init__(self, audio_cfg: dict[str, Any] | None = None):
        self.audio_cfg = audio_cfg if audio_cfg is not None else load_config()["audio"]
        self.sample_rate = int(self.audio_cfg["sample_rate"])

    def scores(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """Scores of shape ``(len(clips), len(labels))`` for pre-processed, unpadded 16 kHz clips."""
        raise NotImplementedError

    def prepare(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """The shared pre-processing, without padding."""
        return preprocess(samples, sample_rate, self.audio_cfg, pad=False)

    def predict(self, samples: np.ndarray, sample_rate: int, k: int = 5,
                texts: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Top-k phrases for raw audio (any rate, mono or stereo, float or int16)."""
        return top_k(self.scores([self.prepare(samples, sample_rate)])[0], self.labels, k, texts)


class ClassicalRecognizer(Recognizer):
    """A traditional model saved by ``support.classical`` (scaler + classifier + label order)."""

    def __init__(self, path: str | Path, audio_cfg: dict[str, Any] | None = None):
        import joblib

        super().__init__(audio_cfg)
        bundle = joblib.load(path)
        if bundle.get("feature_kind", "mfcc") != "mfcc":
            raise ValueError(f"{path} uses '{bundle['feature_kind']}' features; inference supports MFCC models")
        self.model, self.labels, self.feature_cfg = bundle["model"], list(bundle["labels"]), bundle["features"]
        self.path = Path(path)

    def scores(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        from support.classical import predict_scores
        from support.features import mfcc_features

        x = np.stack([mfcc_features(c, self.sample_rate, self.feature_cfg) for c in clips])
        return predict_scores(self.model, x)


class Wav2Vec2Recognizer(Recognizer):
    """A wav2vec 2.0 model folder saved by ``support.train_wav2vec2``; clips are padded to 6 s."""

    def __init__(self, model_dir: str | Path, device: str | None = None, audio_cfg: dict[str, Any] | None = None,
                 batch_size: int = 8):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        from support.device import pick_device

        super().__init__(audio_cfg)
        self.device = pick_device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = AutoModelForAudioClassification.from_pretrained(model_dir).to(self.device).eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]
        self.batch_size, self.path = batch_size, Path(model_dir)

    def scores(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        max_s = float(self.audio_cfg["max_duration_s"])
        out = []
        for start in range(0, len(clips), self.batch_size):
            batch = [pad_to_duration(c, self.sample_rate, max_s) for c in clips[start:start + self.batch_size]]
            x = self.extractor(batch, sampling_rate=self.sample_rate, return_tensors="pt")["input_values"]
            with torch.no_grad():
                out.append(torch.softmax(self.model(input_values=x.to(self.device)).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)


class WhisperRecognizer:
    """Zero-shot Whisper: transcript plus the nearest phrases (score = 1 − WER, floored at 0)."""

    def __init__(self, phrases: Sequence[dict[str, str]], transcriber=None, cfg: dict[str, Any] | None = None):
        from support.whisper_asr import WhisperTranscriber

        self.cfg = cfg if cfg is not None else load_config()
        w = self.cfg["whisper"]
        self.transcriber = transcriber or WhisperTranscriber(w["model"], None, w["batch_size"], w["language"])
        self.phrases, self.reject_wer = list(phrases), float(w["reject_wer"])

    def predict(self, samples: np.ndarray, sample_rate: int, k: int = 5) -> dict[str, Any]:
        """``{"transcript", "top", "rejected"}`` for raw audio."""
        from support.whisper_asr import rank_phrases

        clip = preprocess(samples, sample_rate, self.cfg["audio"], pad=False)
        transcript = self.transcriber([clip])[0]
        ranking = rank_phrases(transcript, self.phrases, top_k=k)
        top = [{"phrase_id": r["phrase_id"], "text": r["text"], "score": r["similarity"]} for r in ranking]
        return {"transcript": transcript, "top": top, "rejected": ranking[0]["wer"] > self.reject_wer}


def load_recognizer(path: str | Path, device: str | None = None) -> Recognizer:
    """Open a saved model: a ``.joblib`` file (traditional) or a model folder (wav2vec 2.0)."""
    path = Path(path)
    if path.suffix == ".joblib":
        return ClassicalRecognizer(path)
    if (path / "config.json").exists():
        return Wav2Vec2Recognizer(path, device)
    raise ValueError(f"{path} is neither a .joblib model nor a saved wav2vec 2.0 folder")


def read_input(path: str | Path) -> tuple[np.ndarray, int]:
    """Read audio from WAV / FLAC / OGG / MP3, or the audio track of a video (MP4, MOV …) with ffmpeg."""
    try:
        return read_audio(path)
    except Exception:  # noqa: BLE001 - soundfile cannot read it: let ffmpeg decode it
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"cannot read {path}; install ffmpeg to read video or other formats") from None
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
        return read_audio(wav)


def record_microphone(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Record from the default microphone with ffmpeg (macOS, Linux or Windows)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("recording needs ffmpeg; or use the demo app (python app.py)")
    source = {"Darwin": ["-f", "avfoundation", "-i", ":0"], "Linux": ["-f", "alsa", "-i", "default"],
              "Windows": ["-f", "dshow", "-i", "audio=default"]}.get(platform.system())
    if source is None:
        raise RuntimeError(f"microphone recording is not set up for {platform.system()}")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "mic.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *source, "-t", str(seconds), "-ac", "1",
                        "-ar", str(sample_rate), str(wav)], check=True)
        return read_audio(wav)[0]

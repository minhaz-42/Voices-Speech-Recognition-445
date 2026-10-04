"""Shared audio loading and pre-processing.

Every model sees audio prepared the same way (settings in the ``audio`` section
of ``configs/config.yaml``):

1. convert to float32 and mix down to mono,
2. resample to 16 kHz,
3. trim leading and trailing silence (keeping a short margin),
4. peak-normalise,
5. zero-pad at the end to 6 s. Clips are never truncated: a clip that is still
   longer than 6 s after trimming raises an error, so the problem shows up in QC
   instead of silently cutting the phrase.

Usage::

    python -m support.audio data/raw/S01/S01_CMD02_r1_clean.wav
    python -m support.audio clip.wav --out outputs/clip_16k.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from support.config import load_config


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as it is stored.

    Args:
        path: WAV, FLAC, OGG or MP3 file.

    Returns:
        ``(samples, sample_rate)``. ``samples`` is float32 in [-1, 1] with shape
        ``(frames,)`` for mono files or ``(frames, channels)`` otherwise.
    """
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return samples, int(sample_rate)


def to_float32(samples: np.ndarray) -> np.ndarray:
    """Convert signed integer PCM (e.g. int16 from a microphone widget) to float32 in [-1, 1].

    Float input is only cast to float32.

    Raises:
        ValueError: For unsigned integer input, whose zero level is not 0.
    """
    samples = np.asarray(samples)
    if np.issubdtype(samples.dtype, np.signedinteger):
        return (samples / float(np.iinfo(samples.dtype).max + 1)).astype(np.float32)
    if np.issubdtype(samples.dtype, np.unsignedinteger):
        raise ValueError(f"unsigned PCM ({samples.dtype}) is not supported")
    return samples.astype(np.float32, copy=False)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average the channels of a ``(frames, channels)`` array; mono input is returned unchanged."""
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return samples.mean(axis=1)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {samples.shape}")


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio from ``orig_sr`` to ``target_sr`` Hz."""
    if orig_sr == target_sr:
        return samples
    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    top_db: float,
    margin_s: float = 0.0,
    frame_s: float = 0.032,
) -> np.ndarray:
    """Remove leading and trailing silence, keeping ``margin_s`` seconds around the speech.

    Args:
        samples: Mono audio.
        sample_rate: Sample rate in Hz.
        top_db: Frames more than this many dB below the clip's peak count as silence.
        margin_s: Seconds of the original audio kept before and after the detected speech.
        frame_s: Analysis frame length (hop is a quarter of it). Speech edges are found
            to within about half a frame.

    Returns:
        The trimmed audio. Empty or fully silent input is returned unchanged.
    """
    if samples.size == 0:
        return samples
    frame_length = max(4, int(round(frame_s * sample_rate)))
    _, (start, end) = librosa.effects.trim(
        samples, top_db=top_db, frame_length=frame_length, hop_length=frame_length // 4
    )
    if end <= start:
        return samples
    margin = int(round(margin_s * sample_rate))
    return samples[max(0, start - margin):min(samples.size, end + margin)]


def peak_dbfs(samples: np.ndarray) -> float:
    """Peak level in dBFS (0 dBFS = full scale). Returns ``-inf`` for silence."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return 20.0 * np.log10(peak) if peak > 0 else float("-inf")


def peak_normalise(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale the audio so that its peak sits at ``target_dbfs``. Silence is returned unchanged."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return samples * (10.0 ** (target_dbfs / 20.0) / peak)


def pad_to_duration(samples: np.ndarray, sample_rate: int, duration_s: float) -> np.ndarray:
    """Zero-pad the end of the audio to exactly ``duration_s`` seconds.

    Raises:
        ValueError: If the audio is already longer, because speech must never be cut.
    """
    target = int(round(duration_s * sample_rate))
    if samples.size > target:
        raise ValueError(
            f"clip is {samples.size / sample_rate:.2f} s after trimming, longer than "
            f"{duration_s:.2f} s; it is not truncated (check the recording in QC)"
        )
    return np.pad(samples, (0, target - samples.size))


def preprocess(
    samples: np.ndarray,
    sample_rate: int,
    audio_cfg: dict[str, Any] | None = None,
    pad: bool = True,
) -> np.ndarray:
    """Apply the shared pre-processing chain to audio already in memory.

    Args:
        samples: Audio as ``(frames,)`` or ``(frames, channels)``, float or integer PCM.
        sample_rate: Sample rate of ``samples`` in Hz.
        audio_cfg: The ``audio`` section of the config. Defaults to ``configs/config.yaml``.
        pad: Zero-pad to ``max_duration_s``. Turn it off when the padding would distort
            the result, e.g. for MFCC means and standard deviations.

    Returns:
        Mono float32 audio at ``audio_cfg["sample_rate"]``.
    """
    cfg = audio_cfg if audio_cfg is not None else load_config()["audio"]
    target_sr = int(cfg["sample_rate"])

    out = to_float32(samples)
    if cfg.get("mono", True):
        out = to_mono(out)
    out = resample(out, sample_rate, target_sr)
    out = trim_silence(out, target_sr, cfg["trim_top_db"], cfg.get("trim_margin_s", 0.0))
    out = peak_normalise(out, cfg["peak_dbfs"])
    if pad:
        out = pad_to_duration(out, target_sr, cfg["max_duration_s"])
    return out.astype(np.float32, copy=False)


def load_audio(path: str | Path, audio_cfg: dict[str, Any] | None = None, pad: bool = True) -> np.ndarray:
    """Read an audio file and apply :func:`preprocess`.

    Returns:
        Mono float32 audio at the configured sample rate (16 kHz).
    """
    samples, sample_rate = read_audio(path)
    return preprocess(samples, sample_rate, audio_cfg, pad=pad)


def save_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono audio as a 16-bit PCM WAV file, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def main() -> None:
    """Load one file, pre-process it and print what changed."""
    parser = argparse.ArgumentParser(description="Load and pre-process one audio file.")
    parser.add_argument("path", help="audio file (WAV, FLAC, OGG or MP3)")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    parser.add_argument("--no-pad", action="store_true", help="skip padding to max_duration_s")
    parser.add_argument("--out", default=None, help="save the processed audio to this WAV file")
    args = parser.parse_args()

    audio_cfg = load_config(args.config)["audio"]
    raw, raw_sr = read_audio(args.path)
    channels = 1 if raw.ndim == 1 else raw.shape[1]
    processed = preprocess(raw, raw_sr, audio_cfg, pad=not args.no_pad)
    target_sr = int(audio_cfg["sample_rate"])

    print(f"file:      {args.path}")
    print(f"original:  {raw_sr} Hz, {channels} channel(s), {raw.shape[0] / raw_sr:.2f} s, "
          f"peak {peak_dbfs(raw):.1f} dBFS")
    print(f"processed: {target_sr} Hz, mono, {processed.size / target_sr:.2f} s "
          f"({processed.size} samples), peak {peak_dbfs(processed):.1f} dBFS")
    if args.out:
        save_wav(args.out, processed, target_sr)
        print(f"saved:     {args.out}")


if __name__ == "__main__":
    main()

"""Figures of what the models "see" and chart helpers for the results.

Every function draws on a :class:`matplotlib.figure.Figure` (not ``pyplot``), like
``plot_confusion_matrix`` in ``support/metrics.py``, so the code works in scripts,
tests and the Gradio demo. Each function returns the figure; an optional ``path``
saves it as a 150 dpi PNG and creates the folder.

Usage::

    python -m support.visualization clip.wav --out images/example_overview.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import librosa.display
import numpy as np
from matplotlib.figure import Figure

from support.audio import peak_dbfs, preprocess, read_audio, to_mono
from support.config import load_config

DPI = 150
N_FFT = 512        # 32 ms window at 16 kHz, as in the MFCC features
HOP_LENGTH = 160   # 10 ms hop
N_MELS = 64        # bands of the log-mel spectrogram


def _save(fig: Figure, path: str | Path | None) -> Figure:
    """Save ``fig`` to ``path`` (if given) at :data:`DPI`, creating the folder, and return it."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI)
    return fig


def _draw_waveform(ax, samples: np.ndarray, sample_rate: int, title: str | None) -> None:
    """Draw amplitude over time in seconds on ``ax``."""
    samples = np.asarray(samples, dtype=np.float32)
    times = np.arange(samples.size) / float(sample_rate)
    ax.plot(times, samples, linewidth=0.6, color="tab:blue")
    ax.set_xlim(0.0, max(times[-1] if samples.size else 0.0, 1.0 / sample_rate))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"Waveform ({sample_rate / 1000:g} kHz, {samples.size / sample_rate:.2f} s)")


def plot_waveform(
    samples: np.ndarray,
    sample_rate: int,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Draw the amplitude of a mono clip over time.

    Args:
        samples: Mono float audio in [-1, 1].
        sample_rate: Sample rate in Hz.
        title: Figure title. Defaults to the sample rate and duration.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.
    """
    fig = Figure(figsize=(8, 2.8))
    _draw_waveform(fig.add_subplot(), samples, sample_rate, title)
    fig.tight_layout()
    return _save(fig, path)


def _draw_spectrogram(
    fig: Figure, ax, samples: np.ndarray, sample_rate: int, kind: str, title: str | None, colorbar: bool = True
) -> None:
    """Draw a log-power (``"linear"``) or 64-band log-mel (``"mel"``) spectrogram in dB on ``ax``."""
    samples = np.asarray(samples, dtype=np.float32)
    hop = HOP_LENGTH
    if kind == "linear":
        power = np.abs(librosa.stft(samples, n_fft=N_FFT, hop_length=hop)) ** 2
        y_axis, label = "linear", "Log-power spectrogram"
    elif kind == "mel":
        power = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate, n_fft=N_FFT, hop_length=hop, n_mels=N_MELS
        )
        y_axis, label = "mel", f"Log-mel spectrogram ({N_MELS} bands)"
    else:
        raise ValueError(f"kind must be 'linear' or 'mel', got {kind!r}")
    db = librosa.power_to_db(power, ref=np.max(power) if power.size and power.max() > 0 else 1.0)
    image = librosa.display.specshow(
        db, sr=sample_rate, hop_length=hop, x_axis="time", y_axis=y_axis, ax=ax, cmap="magma"
    )
    if colorbar:
        fig.colorbar(image, ax=ax, format="%+.0f dB", label="dB")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or f"{label} (dB)")


def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    kind: str = "linear",
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Draw the spectrogram of a mono clip in dB.

    Args:
        samples: Mono float audio.
        sample_rate: Sample rate in Hz.
        kind: ``"linear"`` for a log-power spectrogram (STFT), ``"mel"`` for a
            64-band log-mel spectrogram.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: For an unknown ``kind``.
    """
    fig = Figure(figsize=(8, 3.2))
    _draw_spectrogram(fig, fig.add_subplot(), samples, sample_rate, kind, title)
    fig.tight_layout()
    return _save(fig, path)


def _draw_mfcc(
    fig: Figure, ax, samples: np.ndarray, sample_rate: int, n_mfcc: int, title: str | None, colorbar: bool = True
) -> None:
    """Draw an MFCC heatmap (coefficient index over time) on ``ax``."""
    samples = np.asarray(samples, dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sample_rate, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40
    )
    image = librosa.display.specshow(
        mfcc, sr=sample_rate, hop_length=HOP_LENGTH, x_axis="time", ax=ax, cmap="coolwarm"
    )
    if colorbar:
        fig.colorbar(image, ax=ax, label="value")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC coefficient")
    ax.set_title(title or f"{n_mfcc} MFCCs")


def plot_mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Draw the MFCCs of a mono clip as a heatmap.

    This figure is for display only; the models' features come from ``support/features.py``.

    Args:
        samples: Mono float audio.
        sample_rate: Sample rate in Hz.
        n_mfcc: Number of coefficients.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.
    """
    fig = Figure(figsize=(8, 3.2))
    _draw_mfcc(fig, fig.add_subplot(), samples, sample_rate, n_mfcc, title)
    fig.tight_layout()
    return _save(fig, path)


def clip_overview(
    samples: np.ndarray,
    sample_rate: int,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """One figure with four panels: waveform, log-power spectrogram, log-mel spectrogram and MFCCs.

    Args:
        samples: Mono float audio.
        sample_rate: Sample rate in Hz.
        title: Overall title above the panels.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure with exactly four axes.
    """
    fig = Figure(figsize=(9, 11))
    axes = fig.subplots(4, 1)
    _draw_waveform(axes[0], samples, sample_rate, None)
    # No colour bars, so the figure has exactly four axes and the panels share one time axis width.
    _draw_spectrogram(fig, axes[1], samples, sample_rate, "linear", "Log-power spectrogram (dB)", colorbar=False)
    _draw_spectrogram(fig, axes[2], samples, sample_rate, "mel", f"Log-mel spectrogram, {N_MELS} bands (dB)",
                      colorbar=False)
    _draw_mfcc(fig, axes[3], samples, sample_rate, 13, "13 MFCCs", colorbar=False)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, path)


def plot_preprocessing(
    path_to_audio: str | Path,
    audio_cfg: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
) -> Figure:
    """Show a clip before and after the shared pre-processing.

    Top: the raw waveform at its own sample rate (mixed to mono for display).
    Bottom: the processed waveform (16 kHz, trimmed, peak-normalised, padded to 6 s).
    Durations and peak levels are in the titles.

    Args:
        path_to_audio: Audio file.
        audio_cfg: The ``audio`` section of the config. Defaults to ``configs/config.yaml``.
        out_path: If given, the figure is saved there.

    Returns:
        The matplotlib figure with two axes.
    """
    cfg = audio_cfg if audio_cfg is not None else load_config()["audio"]
    raw, raw_sr = read_audio(path_to_audio)
    processed = preprocess(raw, raw_sr, cfg, pad=True)
    target_sr = int(cfg["sample_rate"])
    raw_mono = to_mono(raw)
    channels = 1 if raw.ndim == 1 else raw.shape[1]

    fig = Figure(figsize=(9, 5.5))
    top, bottom = fig.subplots(2, 1)
    _draw_waveform(top, raw_mono, raw_sr,
                   f"Before: {raw_sr / 1000:g} kHz, {channels} channel(s), {raw_mono.size / raw_sr:.2f} s, "
                   f"peak {peak_dbfs(raw_mono):.1f} dBFS")
    _draw_waveform(bottom, processed, target_sr,
                   f"After: {target_sr / 1000:g} kHz mono, trimmed, normalised, padded to "
                   f"{processed.size / target_sr:.2f} s, peak {peak_dbfs(processed):.1f} dBFS")
    fig.tight_layout()
    return _save(fig, out_path)

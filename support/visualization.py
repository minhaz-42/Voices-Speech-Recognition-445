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

import numpy as np
from matplotlib.figure import Figure

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


def _draw_spectrogram(fig: Figure, ax, samples: np.ndarray, sample_rate: int, kind: str, title: str | None) -> None:
    """Draw a log-power (``"linear"``) or 64-band log-mel (``"mel"``) spectrogram in dB on ``ax``."""
    import librosa
    import librosa.display

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

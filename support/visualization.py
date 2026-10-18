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

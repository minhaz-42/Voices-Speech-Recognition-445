"""White noise at a chosen signal-to-noise ratio (SNR).

Used in two places, with the settings in ``configs/config.yaml``:

- **training augmentation** (``augmentation``): white noise at a random SNR on a share of
  the *training* clips only (wav2vec 2.0 runs with ``--noise white``);
- **noise tests** (``noise_test``): the test clips at fixed SNRs (20 / 10 / 5 dB), task T21.

SNR is measured against the clip's mean power. The processed clips are trimmed, so
that power is the power of the speech, not of leading or trailing silence.
"""

from __future__ import annotations

import numpy as np


def signal_power(samples: np.ndarray) -> float:
    """Mean power of a signal."""
    samples = np.asarray(samples, dtype=np.float64)
    return float(np.mean(samples ** 2)) if samples.size else 0.0


def add_white_noise(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian white noise so that signal power / noise power = ``snr_db``.

    Silent or empty input is returned unchanged (its SNR is undefined).
    """
    power = signal_power(samples)
    if power == 0.0:
        return np.asarray(samples, dtype=np.float32)
    noise_std = np.sqrt(power / 10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, noise_std, size=np.shape(samples))
    return (np.asarray(samples, dtype=np.float64) + noise).astype(np.float32)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR of ``noisy`` relative to ``clean``, in dB."""
    noise_power = signal_power(np.asarray(noisy, np.float64) - np.asarray(clean, np.float64))
    return float("inf") if noise_power == 0.0 else 10.0 * np.log10(signal_power(clean) / noise_power)


def augment(samples: np.ndarray, rng: np.random.Generator, p: float = 0.5,
            snr_range_db: tuple[float, float] = (5.0, 30.0)) -> tuple[np.ndarray, float | None]:
    """With probability ``p``, add white noise at an SNR drawn uniformly from ``snr_range_db``.

    Returns:
        The (possibly noisy) clip and the SNR used, or ``None`` when no noise was added.
    """
    if rng.random() >= p:
        return np.asarray(samples, dtype=np.float32), None
    snr = float(rng.uniform(*snr_range_db))
    return add_white_noise(samples, snr, rng), snr

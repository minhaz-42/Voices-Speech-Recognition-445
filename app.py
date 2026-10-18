"""TenVoices live demo (Gradio): speak or upload a phrase and see what the system hears.

Task T36 builds the skeleton: the clip is pre-processed exactly like the training data
and shown as audio and as a four-panel figure (waveform, spectrogram, log-mel, MFCC).
The trained model is connected in task T23; until then the app shows no predictions.

Usage::

    python app.py [--port 7860] [--share]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from support.audio import peak_dbfs, preprocess
from support.config import load_config
from support.visualization import clip_overview

EMPTY_MESSAGE = "No audio yet: record a phrase with the microphone or upload a clip, then press Analyse."


def process_audio(audio: tuple[int, np.ndarray] | None, audio_cfg: dict[str, Any] | None = None):
    """Pre-process one clip from the audio widget and describe it.

    Args:
        audio: What ``gr.Audio(type="numpy")`` returns: ``(sample_rate, samples)`` with
            int16 or float samples, mono or ``(frames, channels)``; or ``None``.
        audio_cfg: The ``audio`` section of the config. Defaults to ``configs/config.yaml``.

    Returns:
        ``(processed, figure, info)``: the processed clip as ``(16000, samples)`` for
        playback, the :func:`clip_overview` figure and a short text. For missing, silent
        or too long audio, ``processed`` and ``figure`` are ``None`` and ``info`` says why.
    """
    if audio is None:
        return None, None, EMPTY_MESSAGE
    sample_rate, samples = audio
    samples = np.asarray(samples)
    if samples.size == 0:
        return None, None, EMPTY_MESSAGE

    cfg = audio_cfg if audio_cfg is not None else load_config()["audio"]
    target_sr = int(cfg["sample_rate"])
    try:
        processed = preprocess(samples, int(sample_rate), cfg, pad=False)
    except ValueError as err:
        return None, None, f"Cannot use this clip: {err}"
    if processed.size == 0 or not np.any(processed):
        return None, None, "The clip is silent: please record or upload a spoken phrase."

    max_s = float(cfg["max_duration_s"])
    if processed.size / target_sr > max_s:
        return None, None, (f"The clip is {processed.size / target_sr:.2f} s long after trimming the silence; "
                            f"phrases must fit in {max_s:g} s, so it is not cut. Please record a shorter clip.")

    raw_seconds = samples.shape[0] / float(sample_rate)
    figure = clip_overview(processed, target_sr, title="What the model receives")
    info = (f"Input: {int(sample_rate) / 1000:g} kHz, {raw_seconds:.2f} s\n"
            f"After pre-processing: {target_sr / 1000:g} kHz mono, {processed.size / target_sr:.2f} s of speech, "
            f"peak {peak_dbfs(processed):.1f} dBFS")
    return (target_sr, processed), figure, info

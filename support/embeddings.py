"""Frozen wav2vec 2.0 embeddings for the linear probe (task T46).

Every processed clip goes through the pre-trained wav2vec 2.0 model **without any
training**; the hidden states of one Transformer layer are averaged over time into
one vector per clip. ``support/classical.py --model logreg --features w2v2`` then
trains logistic regression on these vectors, which shows how much the pre-trained
features already know about the 50 phrases.

PyTorch is imported inside the functions, so importing this module stays cheap
(and never loads PyTorch next to XGBoost; see AGENTS.md).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def embed_clips(
    samples_list: Sequence[np.ndarray],
    model: Any,
    extractor: Any,
    device: str = "cpu",
    layer: int | None = None,
    sample_rate: int = 16000,
) -> np.ndarray:
    """One vector per clip: the time-averaged hidden states of ``layer`` of a frozen model.

    Clips are embedded one at a time, so no padding is needed and the average is over
    the real frames only.

    Args:
        samples_list: Processed mono float32 clips at ``sample_rate``.
        model: A ``Wav2Vec2Model`` (or any model returning ``hidden_states``).
        extractor: Its feature extractor (normalises the waveform like in pre-training).
        device: ``cpu``, ``cuda`` or ``mps``.
        layer: Hidden-state index: 0 is the output of the CNN encoder, ``n`` the
            ``n``-th Transformer layer. ``None`` (default) takes the last layer.
        sample_rate: Sample rate of the clips in Hz.

    Returns:
        float32 array of shape ``(len(samples_list), hidden_size)``.
    """
    import torch

    model.eval()
    model.to(device)
    vectors = []
    with torch.no_grad():
        for samples in samples_list:
            inputs = extractor(np.asarray(samples, dtype=np.float32), sampling_rate=sample_rate,
                               return_tensors="pt")
            out = model(inputs["input_values"].to(device), output_hidden_states=True)
            states = out.hidden_states[-1 if layer is None else layer]  # (1, frames, hidden)
            vectors.append(states.mean(dim=1).squeeze(0).float().cpu().numpy())
    hidden = int(model.config.hidden_size)
    return np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, hidden), dtype=np.float32)

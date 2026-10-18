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

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from support.config import load_config, resolve_path
from support.dataset import load_manifest, read_processed

DEFAULT_MODEL = "facebook/wav2vec2-base"


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


def features_path(dataset: str, cfg: dict[str, Any]) -> Path:
    """``outputs/features/<dataset>_w2v2.npz``, the file ``support.classical --features w2v2`` reads."""
    return resolve_path(cfg, "outputs") / "features" / f"{dataset}_w2v2.npz"


def extract_dataset(
    dataset: str,
    model_name: str = DEFAULT_MODEL,
    layer: int | None = None,
    cfg: dict[str, Any] | None = None,
    device: str | None = None,
) -> Path:
    """Embed every clip of the dataset's manifest and save the cache for the linear probe.

    The ``.npz`` file holds exactly:

    - ``utt_ids``: ``np.array(ids, dtype=str)``, one per row;
    - ``features``: float32, shape ``(n_clips, hidden_size)`` (768 for wav2vec 2.0 Base);
    - ``settings``: ``np.array(["model=...", "layer=..."], dtype=str)``.

    Args:
        dataset: ``fsc`` or ``l2arctic``.
        model_name: Pre-trained checkpoint (a Hugging Face name or a local folder).
        layer: Hidden-state index (see :func:`embed_clips`); ``None`` = last layer.
        cfg: Project config. Defaults to ``configs/config.yaml``.
        device: ``cuda``, ``mps`` or ``cpu`` (default: best available).

    Returns:
        Path of the saved ``.npz`` file.
    """
    from tqdm import tqdm
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    from support.device import pick_device

    cfg = cfg if cfg is not None else load_config()
    device = pick_device(device)
    manifest = load_manifest(dataset, cfg)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    sample_rate = int(cfg["audio"]["sample_rate"])

    vectors = []
    for path in tqdm(manifest["path"], desc=f"embedding {dataset}", unit="clip"):
        vectors.append(embed_clips([read_processed(path)], model, extractor, device, layer, sample_rate)[0])
    features = np.stack(vectors).astype(np.float32)

    out = features_path(dataset, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    layer_name = "last" if layer is None else str(layer)
    np.savez(out, utt_ids=np.array(manifest["utt_id"].tolist(), dtype=str), features=features,
             settings=np.array([f"model={model_name}", f"layer={layer_name}"], dtype=str))
    return out

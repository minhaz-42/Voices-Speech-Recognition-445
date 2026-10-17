"""MFCC summary features for the traditional models.

Each clip becomes 78 numbers: 13 MFCCs, their deltas (Δ) and delta-deltas (ΔΔ), each
summarised over time by its mean and standard deviation. The clips are the processed
copies (16 kHz, trimmed, normalised, **not padded**), so padding never dilutes the
statistics. Settings: the ``features`` section of ``configs/config.yaml``.

Features are cached per dataset in ``outputs/features/<dataset>_mfcc.npz`` (keyed by
``utt_id``), so every model run reuses them.

Usage::

    python -m support.features --dataset fsc      # compute and cache all clips
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np

from support.config import load_config, resolve_path


def feature_names(n_mfcc: int = 13) -> list[str]:
    """Names of the feature columns, in order."""
    return [f"{kind}{i}_{stat}" for kind in ("mfcc", "d1_", "d2_") for stat in ("mean", "std")
            for i in range(1, n_mfcc + 1)]


def mfcc_features(samples: np.ndarray, sample_rate: int = 16000, feat_cfg: dict[str, Any] | None = None) -> np.ndarray:
    """The 78-value summary of one clip.

    Args:
        samples: Mono float audio.
        sample_rate: Its sample rate.
        feat_cfg: The ``features`` section of the config (default: config file).

    Returns:
        ``[mean(MFCC), std(MFCC), mean(Δ), std(Δ), mean(ΔΔ), std(ΔΔ)]`` as float32.
    """
    cfg = feat_cfg if feat_cfg is not None else load_config()["features"]
    mfcc = librosa.feature.mfcc(y=np.asarray(samples, dtype=np.float32), sr=sample_rate,
                                n_mfcc=int(cfg["n_mfcc"]), n_fft=int(cfg["n_fft"]),
                                hop_length=int(cfg["hop_length"]), n_mels=int(cfg["n_mels"]))
    # The default 9-frame delta window needs at least 9 frames; shorter clips use edge padding.
    mode = "interp" if mfcc.shape[1] >= 9 else "nearest"
    d1 = librosa.feature.delta(mfcc, order=1, mode=mode)
    d2 = librosa.feature.delta(mfcc, order=2, mode=mode)
    return np.concatenate([np.concatenate([m.mean(axis=1), m.std(axis=1)]) for m in (mfcc, d1, d2)]).astype(np.float32)


def extract(utt_ids: Sequence[str], paths: Sequence[str], cache: str | Path | None = None,
            cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Features of many clips, reusing and extending a cache file.

    Args:
        utt_ids: Clip IDs (cache keys).
        paths: Processed WAV paths, in the same order.
        cache: ``.npz`` cache file; ``None`` disables caching.
        cfg: Settings (default: config file).

    Returns:
        An array of shape ``(len(utt_ids), 78)`` in the given order.
    """
    from support.dataset import read_processed  # local import: dataset imports nothing from here

    cfg = cfg if cfg is not None else load_config()
    feat_cfg, sr = cfg["features"], int(cfg["audio"]["sample_rate"])
    known: dict[str, np.ndarray] = {}
    cache = Path(cache) if cache is not None else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if data["settings"].tolist() == _settings_key(feat_cfg):
                known = dict(zip(data["utt_ids"].tolist(), data["features"]))
    missing = [(u, p) for u, p in zip(utt_ids, paths) if u not in known]
    for utt, path in missing:
        known[utt] = mfcc_features(read_processed(path), sr, feat_cfg)
    if cache is not None and missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(known)
        np.savez_compressed(cache, utt_ids=np.array(ids), features=np.stack([known[u] for u in ids]),
                            settings=np.array(_settings_key(feat_cfg)))
    return np.stack([known[u] for u in utt_ids])


def _settings_key(feat_cfg: dict[str, Any]) -> list[str]:
    """The feature settings as strings, so a cache built with other settings is ignored."""
    return [f"{k}={feat_cfg[k]}" for k in sorted(feat_cfg)]


def cache_path(dataset: str, cfg: dict[str, Any] | None = None) -> Path:
    """``outputs/features/<dataset>_mfcc.npz``."""
    cfg = cfg if cfg is not None else load_config()
    return resolve_path(cfg, "outputs") / "features" / f"{dataset}_mfcc.npz"


def main() -> None:
    """Compute and cache the features of every processed clip of a dataset."""
    from support.dataset import load_manifest

    parser = argparse.ArgumentParser(description="Compute MFCC summary features for a dataset.")
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = load_manifest(args.dataset, cfg)
    feats = extract(manifest["utt_id"], manifest["path"], cache_path(args.dataset, cfg), cfg)
    print(f"{args.dataset}: {feats.shape[0]} clips x {feats.shape[1]} features -> {cache_path(args.dataset, cfg)}")


if __name__ == "__main__":
    main()

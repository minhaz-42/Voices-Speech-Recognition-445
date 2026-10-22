"""Runs the embedding scenarios in a separate process and writes a JSON report.

``tests/test_embeddings.py`` starts this with ``python -m tests._embeddings_check <folder>``.
PyTorch must not share a process with XGBoost on macOS (their OpenMP libraries clash),
so everything that needs PyTorch happens here, and the test only reads the report.
Uses a tiny randomly initialised wav2vec 2.0 (offline, CPU, a few seconds).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HIDDEN = 32


def save_tiny_model(path: Path) -> None:
    """A tiny ``Wav2Vec2Model`` and its feature extractor, saved like a real checkpoint."""
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    torch.manual_seed(0)
    cfg = Wav2Vec2Config(hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16, 16), conv_kernel=(10, 8), conv_stride=(10, 8), num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2, mask_time_prob=0.0)
    Wav2Vec2Model(cfg).save_pretrained(path)
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                             return_attention_mask=False).save_pretrained(path)


def main(out: Path) -> dict:
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    from support import embeddings
    from support.classical import load_cached_features
    from support.dataset import build_processed, load_manifest
    from tests._synthetic import make_config, make_subset, write_audio

    subset = write_audio(make_subset("fsc", n_phrases=2, takes=1), out / "raw")  # 20 clips
    cfg = make_config(out, "fsc", subset)
    cfg["paths"].update(outputs=str(out / "outputs"))
    cfg["audio"]["max_duration_s"] = 1.0  # synthetic clips are 0.9 s
    report: dict = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        save_tiny_model(out / "tiny")
        build_processed("fsc", cfg)

        # embed_clips on clips of different lengths, last and first layer
        model = Wav2Vec2Model.from_pretrained(out / "tiny")
        extractor = AutoFeatureExtractor.from_pretrained(out / "tiny")
        rng = np.random.default_rng(0)
        clips = [0.1 * rng.standard_normal(n).astype(np.float32) for n in (8000, 12000, 16000)]
        last = embeddings.embed_clips(clips, model, extractor, "cpu")
        first = embeddings.embed_clips(clips, model, extractor, "cpu", layer=0)
        report["embed"] = {"shape": list(last.shape), "dtype": str(last.dtype),
                           "layer_changes_output": not np.allclose(last, first),
                           "repeatable": bool(np.allclose(last, embeddings.embed_clips(clips, model, extractor)))}

        # extract_dataset writes the cache that classical.py reads
        path = embeddings.extract_dataset("fsc", str(out / "tiny"), cfg=cfg, device="cpu")
        with np.load(path, allow_pickle=False) as data:
            keys = sorted(data.files)
            utt_ids, features, settings = data["utt_ids"], data["features"], data["settings"]
        again = embeddings.extract_dataset("fsc", str(out / "tiny"), cfg=cfg, device="cpu")
        with np.load(again, allow_pickle=False) as data:
            same = bool(np.allclose(features, data["features"]))
        manifest = load_manifest("fsc", cfg)
        ids = manifest["utt_id"].tolist()[:5]
        cached = load_cached_features("fsc", "w2v2", ids, cfg)

        # ProbeRecognizer: a probe saved like support.classical does, used on raw 48 kHz stereo audio
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        labels = sorted(manifest["label"].unique())
        y = manifest["label"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(features, y)
        joblib.dump({"model": clf, "labels": labels, "params": {"C": 1.0}, "features": cfg["features"],
                     "feature_kind": "w2v2"}, out / "probe.joblib")
        rec = embeddings.ProbeRecognizer(out / "probe.joblib", model_name=str(out / "tiny"), device="cpu",
                                         audio_cfg=cfg["audio"])
        t = np.arange(int(0.5 * 48000)) / 48000
        stereo = np.stack([0.3 * np.sin(2 * np.pi * 300 * t)] * 2, axis=1)
        top = rec.predict(stereo, 48000, k=2, texts={labels[0]: "first"})
        joblib.dump({"model": clf, "labels": labels, "features": cfg["features"], "feature_kind": "mfcc"},
                    out / "mfcc.joblib")
        try:
            embeddings.ProbeRecognizer(out / "mfcc.joblib", model_name=str(out / "tiny"), device="cpu")
            wrong_kind_raises = False
        except ValueError:
            wrong_kind_raises = True
    report["probe"] = {"labels": labels, "n_top": len(top), "ids": [r["phrase_id"] for r in top],
                       "scores": [r["score"] for r in top], "texts": {r["phrase_id"]: r["text"] for r in top},
                       "wrong_kind_raises": wrong_kind_raises}

    report["cache"] = {
        "name": path.name, "keys": keys, "n_clips": int(features.shape[0]), "n_manifest": len(manifest),
        "width": int(features.shape[1]), "features_dtype": str(features.dtype),
        "utt_ids_kind": utt_ids.dtype.kind, "settings": settings.tolist(),
        "same_on_rerun": same, "cached_shape": list(cached.shape),
        "cached_matches": bool(np.allclose(cached, features[:5])),
    }
    return report


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    (folder / "report.json").write_text(json.dumps(main(folder), default=str), encoding="utf-8")

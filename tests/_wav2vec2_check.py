"""Runs the wav2vec 2.0 training scenarios in a separate process and writes a JSON report.

``tests/test_train_wav2vec2.py`` starts this with ``python -m tests._wav2vec2_check <folder>``.
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


def save_tiny_model(path: Path, n_labels: int) -> None:
    """A 29k-parameter wav2vec 2.0 classifier and its feature extractor, saved like a real checkpoint."""
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16, 16), conv_kernel=(10, 8), conv_stride=(10, 8), num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2, classifier_proj_size=16, num_labels=n_labels,
                         mask_time_prob=0.0)
    Wav2Vec2ForSequenceClassification(cfg).save_pretrained(path)
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                             return_attention_mask=False).save_pretrained(path)


def main(out: Path) -> dict:
    from support import train_wav2vec2 as tw
    from support.dataset import build_processed, load_split
    from tests._synthetic import make_config, make_subset, write_audio
    from tools.make_splits import make_splits

    subset = write_audio(make_subset("fsc", n_phrases=3, takes=2), out / "raw")  # 60 clips
    cfg = make_config(out, "fsc", subset)
    cfg["paths"].update(results=str(out / "results"), images=str(out / "images"), outputs=str(out / "outputs"))
    cfg["audio"]["max_duration_s"] = 1.0  # synthetic clips are 0.7 s
    cfg["wav2vec2"].update(model=str(out / "tiny"), learning_rate=1e-3)
    report: dict = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        save_tiny_model(out / "tiny", 3)
        build_processed("fsc", cfg)
        make_splits("fsc", cfg)
        report["run_names"] = [tw.default_run_name(15, "white", False), tw.default_run_name(3, "none", False),
                               tw.default_run_name(15, "white", True)]
        report["trained"] = tw.run("fsc", ["random", "fold1"], epochs=3, noise="white", cfg=cfg, device="cpu")
        report["untrained"] = tw.run("fsc", ["random"], untrained=True, cfg=cfg, device="cpu", save_models=False)
        try:
            tw.run("fsc", ["random"], noise="pink", cfg=cfg, device="cpu")
            report["bad_noise_raises"] = False
        except ValueError:
            report["bad_noise_raises"] = True

    # Augmentation is a training-only switch: without it a clip is identical on every pass.
    parts = load_split("fsc", "random", cfg)
    index = {"P01": 0, "P02": 1, "P03": 2}
    clean = tw.ClipDataset(parts["val"], index, 16000, 1.0)
    noisy = tw.ClipDataset(parts["val"], index, 16000, 1.0, augment_cfg={"p": 1.0, "snr_db": [5, 5]}, seed=0)
    a, label = clean[0]
    report["dataset"] = {
        "shape": list(a.shape), "label_ok": label == index[parts["val"].loc[0, "label"]],
        "clean_repeatable": bool(np.array_equal(a, clean[0][0])),
        "noise_changes_clip": not np.array_equal(a, noisy[0][0]),
        "noise_differs_each_pass": not np.array_equal(noisy[0][0], noisy[0][0]),
    }
    report["files"] = {
        "predictions_fold1": (out / "results" / "w2v2_3ep_noise" / "fsc" / "predictions_fold1.csv").exists(),
        "model_config": (out / "outputs" / "models" / "w2v2_3ep_noise" / "fsc" / "fold1" / "config.json").exists(),
        "extractor_config": (out / "outputs" / "models" / "w2v2_3ep_noise" / "fsc" / "fold1"
                             / "preprocessor_config.json").exists(),
        "untrained_model_saved": (out / "outputs" / "models" / "w2v2_untrained").exists(),
    }
    report["augmentation_cfg"] = cfg["augmentation"]
    return report


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    (folder / "report.json").write_text(json.dumps(main(folder), default=str), encoding="utf-8")

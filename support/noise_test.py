"""Noise tests (task T21): how saved models hold up when white noise is added to the test clips.

For every split of a run, the saved model of that split (``outputs/models/<run>/<dataset>/<split>``)
predicts its test clips clean and with white noise at each SNR of ``noise_test.snr_db``
(20 / 10 / 5 dB). The noise of a clip depends only on the clip, the SNR and the seed, so
**every model hears exactly the same noisy audio** and the results can be compared clip
by clip (McNemar's test for H3). Noise is added to the processed clip (speech plus its
short margins), before a wav2vec 2.0 model pads it to 6 s.

Output (small, committed):

- ``results/noise/<run>_<dataset>.csv``: accuracy and F1 per SNR, for the random split
  and the pooled folds;
- ``results/noise/<run>_<dataset>_predictions.csv``: every prediction (utt_id, split, SNR).

Usage::

    python -m support.noise_test --run w2v2_15ep_noise --dataset fsc
    python -m support.noise_test --run random_forest --dataset l2arctic --split folds
"""

from __future__ import annotations

import argparse
import zlib
from typing import Any, Sequence

import numpy as np
import pandas as pd

from support.config import load_config, resolve_path
from support.dataset import load_split, parse_splits, read_processed
from support.evaluate import FOLD_NAMES
from support.metrics import compute_metrics
from support.noise import add_white_noise

CLEAN = "clean"


def noisy_clip(samples: np.ndarray, snr_db: float, utt_id: str, seed: int = 42) -> np.ndarray:
    """The clip with white noise at ``snr_db``; the same clip, SNR and seed always give the same noise."""
    rng = np.random.default_rng([seed, int(round(snr_db * 100)) + 100_000, zlib.crc32(utt_id.encode())])
    return add_white_noise(samples, snr_db, rng)


def model_path(run: str, dataset: str, split: str, cfg: dict[str, Any]):
    """The saved model of one split: a ``.joblib`` file (traditional) or a folder (wav2vec 2.0)."""
    root = resolve_path(cfg, "outputs") / "models" / run / dataset
    for candidate in (root / f"{split}.joblib", root / split):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no saved model for {run} / {dataset} / {split} in {root}")


def run_noise_test(run: str, dataset: str, splits: Sequence[str], cfg: dict[str, Any] | None = None,
                   device: str | None = None, snrs: Sequence[float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each split's test clips clean and at each SNR with that split's saved model.

    Returns:
        ``(summary, predictions)``, also written to ``results/noise/``.
    """
    from support.inference import load_recognizer

    cfg = cfg if cfg is not None else load_config()
    seed = int(cfg.get("seed", 42))
    snrs = list(snrs if snrs is not None else cfg["noise_test"]["snr_db"])
    rows = []
    for split in splits:
        recognizer = load_recognizer(model_path(run, dataset, split, cfg), device)
        test = load_split(dataset, split, cfg)["test"]
        clean = [read_processed(p) for p in test["path"]]
        for snr in [CLEAN, *snrs]:
            clips = clean if snr == CLEAN else [noisy_clip(c, float(snr), u, seed) for c, u in zip(clean, test["utt_id"])]
            scores = recognizer.scores(clips)
            preds = [recognizer.labels[i] for i in scores.argmax(axis=1)]
            rows.extend({"utt_id": u, "speaker_id": s, "split": split, "snr_db": str(snr), "label": l, "pred": p}
                        for u, s, l, p in zip(test["utt_id"], test["speaker_id"], test["label"], preds))
    predictions = pd.DataFrame(rows)

    summary_rows = []
    for snr, by_snr in predictions.groupby("snr_db", sort=False):
        groups = {"random": by_snr[by_snr["split"] == "random"], "folds": by_snr[by_snr["split"].isin(FOLD_NAMES)]}
        for protocol, frame in groups.items():
            if len(frame):
                m = compute_metrics(frame["label"].tolist(), frame["pred"].tolist())
                summary_rows.append({"run": run, "dataset": dataset, "protocol": protocol, "snr_db": snr,
                                     "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    summary = pd.DataFrame(summary_rows)

    out = resolve_path(cfg, "results") / "noise"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"{run}_{dataset}.csv", index=False)
    predictions.to_csv(out / f"{run}_{dataset}_predictions.csv", index=False)
    return summary, predictions


def main() -> None:
    """Run the noise test of one saved run."""
    parser = argparse.ArgumentParser(description="Accuracy of a saved run with white noise added to its test clips.")
    parser.add_argument("--run", required=True, help="run name, e.g. w2v2_15ep_noise or random_forest")
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--split", default="all", type=parse_splits, help="all (default), folds, random or foldN")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (wav2vec 2.0 only)")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    summary, _ = run_noise_test(args.run, args.dataset, args.split, load_config(args.config), args.device)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

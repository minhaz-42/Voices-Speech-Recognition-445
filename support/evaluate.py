"""One evaluation for every model: predictions in, metrics, reports and confusion matrices out.

Every runner (traditional models, wav2vec 2.0, Whisper) hands its test predictions to
:func:`save_run`, so all models are scored by the same code and stored the same way:

``results/<run>/<dataset>/``
    ``summary.json``          random split + speaker-independent folds (pooled, per fold, mean ± std)
    ``predictions_<split>.csv``  ``utt_id, speaker_id, label, pred`` (+ extra columns of the run)
    ``random/``, ``folds/``   ``metrics.json``, ``classification_report.csv``, ``per_speaker.csv``,
                              ``confusion_matrix.csv``
``images/confusion/<run>_<dataset>_<random|folds>.png``

The folds' headline numbers pool the test predictions of all 5 folds: every clip is
tested exactly once, so the pooled set covers the whole dataset.

Usage::

    python -m support.evaluate --run random_forest --dataset fsc   # rebuild from saved predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from support.config import load_config, resolve_path
from support.metrics import (classification_report_table, compute_metrics, confusion_matrix_table,
                             per_speaker_accuracy, plot_confusion_matrix)

REQUIRED_COLUMNS = ["utt_id", "speaker_id", "label", "pred"]
FOLD_NAMES = [f"fold{k}" for k in range(1, 6)]
HEADLINE = ("accuracy", "macro_f1", "weighted_f1")


def check_predictions(pred: pd.DataFrame) -> None:
    """Raise ``ValueError`` if columns are missing, the table is empty or a clip repeats."""
    missing = [c for c in REQUIRED_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"predictions are missing columns: {missing}")
    if pred.empty:
        raise ValueError("predictions are empty")
    if pred["utt_id"].duplicated().any():
        raise ValueError("a clip appears more than once in the predictions")


def evaluate_predictions(pred: pd.DataFrame, labels: Sequence[str] | None = None,
                         out_dir: str | Path | None = None, figure: str | Path | None = None,
                         title: str | None = None) -> dict[str, Any]:
    """Metrics of one set of predictions, optionally written to ``out_dir``.

    Args:
        pred: ``utt_id, speaker_id, label, pred``.
        labels: Full phrase list, so tables keep the same order for every model.
        out_dir: Folder for ``metrics.json`` and the CSV tables.
        figure: Path of the confusion-matrix image.
        title: Figure title.

    Returns:
        ``compute_metrics`` output plus ``n_speakers``.
    """
    check_predictions(pred)
    y, p = pred["label"].tolist(), pred["pred"].tolist()
    metrics = compute_metrics(y, p) | {"n_speakers": int(pred["speaker_id"].nunique())}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        classification_report_table(y, p, labels).to_csv(out_dir / "classification_report.csv")
        per_speaker_accuracy(y, p, pred["speaker_id"].tolist()).to_csv(out_dir / "per_speaker.csv", index=False)
        confusion_matrix_table(y, p, labels).to_csv(out_dir / "confusion_matrix.csv")
    if figure is not None:
        plot_confusion_matrix(y, p, labels, path=figure, title=title)
    return metrics


def summarise_folds(fold_preds: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Pooled metrics over all folds, each fold's metrics, and their mean and standard deviation."""
    per_fold = {name: compute_metrics(df["label"].tolist(), df["pred"].tolist()) for name, df in fold_preds.items()}
    pooled = pd.concat(fold_preds.values(), ignore_index=True)
    check_predictions(pooled)
    return {
        "pooled": compute_metrics(pooled["label"].tolist(), pooled["pred"].tolist()),
        "per_fold": per_fold,
        "mean": {k: float(np.mean([m[k] for m in per_fold.values()])) for k in HEADLINE},
        "std": {k: float(np.std([m[k] for m in per_fold.values()], ddof=1)) if len(per_fold) > 1 else 0.0
                for k in HEADLINE},
    }


def run_dir(run: str, dataset: str, cfg: dict[str, Any] | None = None) -> Path:
    """``results/<run>/<dataset>/``."""
    cfg = cfg if cfg is not None else load_config()
    return resolve_path(cfg, "results") / run / dataset


def save_run(run: str, dataset: str, split_preds: dict[str, pd.DataFrame], labels: Sequence[str] | None = None,
             extra: dict[str, Any] | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Store and score a run's test predictions.

    Args:
        run: Run name, e.g. ``random_forest`` or ``w2v2_15ep_noise``.
        dataset: ``fsc`` or ``l2arctic``.
        split_preds: ``{"random": df, "fold1": df, …}``; any subset of the splits.
        labels: Full phrase list (row order of reports and matrices).
        extra: Run details stored in ``summary.json`` (settings, chosen hyper-parameters …).
        cfg: Settings (default: config file).

    Returns:
        The summary that is written to ``summary.json``.
    """
    cfg = cfg if cfg is not None else load_config()
    root = run_dir(run, dataset, cfg)
    root.mkdir(parents=True, exist_ok=True)
    figures = resolve_path(cfg, "images") / "confusion"
    summary: dict[str, Any] = {"run": run, "dataset": dataset}

    for name, df in split_preds.items():
        check_predictions(df)
        df.to_csv(root / f"predictions_{name}.csv", index=False)
    if "random" in split_preds:
        summary["random"] = evaluate_predictions(split_preds["random"], labels, root / "random",
                                                 figures / f"{run}_{dataset}_random.png",
                                                 f"{run} · {dataset} · random split")
    folds = {k: v for k, v in split_preds.items() if k in FOLD_NAMES}
    if folds:
        summary["folds"] = summarise_folds(folds) | {"folds_run": sorted(folds)}
        pooled = pd.concat(folds.values(), ignore_index=True)
        evaluate_predictions(pooled, labels, root / "folds", figures / f"{run}_{dataset}_folds.png",
                             f"{run} · {dataset} · unseen speakers ({len(folds)} folds pooled)")
    summary.update(extra or {})
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def load_run_predictions(run: str, dataset: str, cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """All ``predictions_<split>.csv`` files of a run."""
    root = run_dir(run, dataset, cfg)
    files = sorted(root.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"no predictions in {root}")
    return {f.stem.removeprefix("predictions_"): pd.read_csv(f, dtype={"utt_id": str, "speaker_id": str,
                                                                       "label": str, "pred": str})
            for f in files}


def main() -> None:
    """Rebuild a run's metrics, tables and figures from its saved predictions."""
    parser = argparse.ArgumentParser(description="Score a run from its saved predictions.")
    parser.add_argument("--run", required=True, help="run folder in results/, e.g. random_forest")
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()

    from support.dataset import load_labels

    cfg = load_config(args.config)
    previous = run_dir(args.run, args.dataset, cfg) / "summary.json"
    extra = {}
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        extra = {k: v for k, v in old.items() if k not in {"run", "dataset", "random", "folds"}}
    summary = save_run(args.run, args.dataset, load_run_predictions(args.run, args.dataset, cfg),
                       list(load_labels(args.dataset, cfg)["phrase_id"]), extra, cfg)
    print(json.dumps({k: summary[k] for k in ("random", "folds") if k in summary}, indent=2)[:2000])


if __name__ == "__main__":
    main()

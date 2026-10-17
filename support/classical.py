"""Shared runner for the traditional models: MFCC features + scaler + classifier.

One script for all four models (Random Forest, XGBoost, k-NN, SVM). For every split it:

1. loads the split's train / validation / test clips (``support.dataset.load_split``),
2. computes or reuses the MFCC summary features (``support.features``),
3. tries every hyper-parameter combination of the model's grid (``classical`` section of
   ``configs/config.yaml``), fitting on train and scoring on **validation only**,
4. scores the best combination once on test,

then hands all test predictions to ``support.evaluate.save_run``. Fitted models go to
``outputs/models/<run>/<dataset>/<split>.joblib`` (gitignored) for inference.

Usage::

    python -m support.classical --model random_forest --dataset fsc            # random split + 5 folds
    python -m support.classical --model svm --dataset l2arctic --split fold3
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from support.config import load_config, resolve_path
from support.dataset import load_labels, load_split, parse_splits
from support.evaluate import save_run
from support.features import cache_path, extract

MODELS = ("random_forest", "xgboost", "knn", "svm", "logreg")


def make_model(name: str, params: dict[str, Any], seed: int = 42) -> Pipeline:
    """``StandardScaler`` + the classifier ``name`` with ``params``."""
    if name == "random_forest":
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    elif name == "xgboost":
        if sys.platform == "darwin" and "torch" in sys.modules:
            # PyTorch and XGBoost bring separate OpenMP libraries on macOS; in one process
            # XGBoost crashes (segmentation fault). Run traditional models in their own process.
            raise RuntimeError("XGBoost cannot run in a process that has already loaded PyTorch on macOS; "
                               "run python -m support.classical on its own")
        from xgboost import XGBClassifier

        clf = XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="mlogloss", **params)
    elif name == "knn":
        clf = KNeighborsClassifier(**params)
    elif name == "svm":
        clf = SVC(kernel="rbf", random_state=seed, **params)
    elif name == "logreg":
        clf = LogisticRegression(max_iter=5000, random_state=seed, **params)
    else:
        raise ValueError(f"unknown model '{name}' (use one of {MODELS})")
    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def param_grid(name: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every hyper-parameter combination of a model, in a fixed order."""
    try:
        grid = cfg["classical"][name]
    except KeyError:
        raise KeyError(f"no grid for '{name}' in the classical section of the config") from None
    return list(ParameterGrid({k: v if isinstance(v, list) else [v] for k, v in grid.items()}))


def predict_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """Class scores that sum to 1 per clip, one column per ``model.classes_``.

    Probabilities for Random Forest, XGBoost and k-NN; for the SVM, a softmax of its
    one-vs-rest decision values (a ranking score for the top-5 list, not a calibrated
    probability).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    d = model.decision_function(x)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_index(model: Pipeline, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Highest-scoring class (as the model's own label values) and its score."""
    scores = predict_scores(model, x)
    return model.classes_[scores.argmax(axis=1)], scores.max(axis=1)


def fit_select(name: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
               grid: Sequence[dict[str, Any]], seed: int = 42) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Fit every combination on train, keep the one with the best validation accuracy.

    Ties go to the earlier combination in the grid. The test set is never seen here.

    Returns:
        The fitted best model, its parameters and every combination's validation accuracy.
    """
    best, best_params, best_acc, tried = None, None, -1.0, []
    for params in grid:
        model = make_model(name, params, seed).fit(x_train, y_train)
        acc = float(np.mean(predict_index(model, x_val)[0] == y_val))
        tried.append({"params": params, "val_accuracy": acc})
        if acc > best_acc:
            best, best_params, best_acc = model, params, acc
    return best, best_params, tried


def load_cached_features(dataset: str, kind: str, utt_ids: Sequence[str], cfg: dict[str, Any]) -> np.ndarray:
    """Features of another extractor, read from ``outputs/features/<dataset>_<kind>.npz``.

    The file holds ``utt_ids`` (a string array, ``np.array(ids, dtype=str)``, one per row)
    and ``features`` (a 2-D float array), the same layout as the MFCC cache, e.g. frozen
    wav2vec 2.0 embeddings for the linear probe.
    """
    path = resolve_path(cfg, "outputs") / "features" / f"{dataset}_{kind}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run the '{kind}' feature extractor first")
    with np.load(path, allow_pickle=False) as data:
        row = {u: i for i, u in enumerate(data["utt_ids"].tolist())}
        missing = [u for u in utt_ids if u not in row]
        if missing:
            raise ValueError(f"{len(missing)} clips have no '{kind}' features, e.g. {missing[:3]}")
        return data["features"][[row[u] for u in utt_ids]].astype(np.float32)


def run_split(name: str, dataset: str, split: str, cfg: dict[str, Any], classes: list[str],
              features: str = "mfcc", run_name: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train, select and test one model on one split.

    Args:
        features: ``mfcc`` (computed and cached here) or the name of a cached feature file.
        run_name: Folder name for the saved model (default: the model name).

    Returns:
        Test predictions (``utt_id, speaker_id, label, pred, confidence``) and selection details.
    """
    parts = load_split(dataset, split, cfg)
    if features == "mfcc":
        cache = cache_path(dataset, cfg)
        x = {p: extract(df["utt_id"], df["path"], cache, cfg) for p, df in parts.items()}
    else:
        x = {p: load_cached_features(dataset, features, list(df["utt_id"]), cfg) for p, df in parts.items()}
    unknown = sorted(set().union(*(set(df["label"]) for df in parts.values())) - set(classes))
    if unknown:
        raise ValueError(f"{split}: labels not in the phrase list: {unknown[:5]}")
    # Number the phrases seen in training 0..k-1 (XGBoost needs consecutive labels);
    # a validation or test phrase never seen in training can only be predicted wrong.
    present = sorted(set(parts["train"]["label"]))
    index = {c: i for i, c in enumerate(present)}
    y = {p: df["label"].map(lambda c: index.get(c, -1)).to_numpy() for p, df in parts.items()}

    seed = int(cfg.get("seed", 42))
    model, params, tried = fit_select(name, x["train"], y["train"], x["val"], y["val"], param_grid(name, cfg), seed)
    pred_idx, confidence = predict_index(model, x["test"])
    test = parts["test"]
    pred = pd.DataFrame({"utt_id": test["utt_id"], "speaker_id": test["speaker_id"], "label": test["label"],
                         "pred": [present[i] for i in pred_idx], "confidence": confidence.round(4)})

    models_dir = resolve_path(cfg, "outputs") / "models" / (run_name or name) / dataset
    models_dir.mkdir(parents=True, exist_ok=True)
    # "labels" gives the phrase ID of each predict_scores column, for inference (main.py).
    joblib.dump({"model": model, "labels": [present[i] for i in model.classes_], "params": params,
                 "features": cfg["features"], "feature_kind": features}, models_dir / f"{split}.joblib")
    info = {"params": params, "val_accuracy": max(t["val_accuracy"] for t in tried), "grid": tried,
            "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(test)}
    return pred, info


def run(name: str, dataset: str, splits: Sequence[str], cfg: dict[str, Any] | None = None,
        features: str = "mfcc", run_name: str | None = None) -> dict[str, Any]:
    """Run a model on several splits and save the results with ``support.evaluate``.

    The results folder is ``results/<run_name>/`` (default: the model name for MFCC
    features, ``<features>_<model>`` otherwise).
    """
    cfg = cfg if cfg is not None else load_config()
    run_name = run_name or (name if features == "mfcc" else f"{features}_{name}")
    classes = list(load_labels(dataset, cfg)["phrase_id"])
    preds, details = {}, {}
    for split in splits:
        preds[split], details[split] = run_split(name, dataset, split, cfg, classes, features, run_name)
        print(f"{run_name} · {dataset} · {split}: best {details[split]['params']} "
              f"(validation accuracy {details[split]['val_accuracy']:.3f})")
    extra = {"model": name, "feature_kind": features,
             "features": cfg["features"] if features == "mfcc" else features, "selection": "best validation accuracy",
             "splits": {s: {k: v for k, v in d.items() if k != "grid"} for s, d in details.items()},
             "grids": {s: d["grid"] for s, d in details.items()}}
    return save_run(run_name, dataset, preds, classes, extra, cfg)


def main() -> None:
    """Train and evaluate one traditional model."""
    parser = argparse.ArgumentParser(description="MFCC features + traditional classifier, both evaluation protocols.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--split", default="all", type=parse_splits, help="all (default), folds, random or foldN")
    parser.add_argument("--features", default="mfcc",
                        help="mfcc (default) or the name of a cached feature file outputs/features/<dataset>_<name>.npz")
    parser.add_argument("--run", default=None, help="results folder name (default: model name, or <features>_<model>)")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    summary = run(args.model, args.dataset, args.split, load_config(args.config), args.features, args.run)
    print(json.dumps({k: summary[k] for k in ("random", "folds") if k in summary}, indent=2)[:1500])


if __name__ == "__main__":
    main()

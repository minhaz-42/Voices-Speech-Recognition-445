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
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from support.config import load_config, resolve_path
from support.dataset import SPLIT_NAMES, load_labels, load_split
from support.evaluate import save_run
from support.features import cache_path, extract

MODELS = ("random_forest", "xgboost", "knn", "svm")


def make_model(name: str, params: dict[str, Any], seed: int = 42) -> Pipeline:
    """``StandardScaler`` + the classifier ``name`` with ``params``."""
    if name == "random_forest":
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    elif name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="mlogloss", **params)
    elif name == "knn":
        clf = KNeighborsClassifier(**params)
    elif name == "svm":
        clf = SVC(kernel="rbf", random_state=seed, **params)
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


def run_split(name: str, dataset: str, split: str, cfg: dict[str, Any],
              classes: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train, select and test one model on one split.

    Returns:
        Test predictions (``utt_id, speaker_id, label, pred, confidence``) and selection details.
    """
    parts = load_split(dataset, split, cfg)
    cache = cache_path(dataset, cfg)
    x = {p: extract(df["utt_id"], df["path"], cache, cfg) for p, df in parts.items()}
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

    models_dir = resolve_path(cfg, "outputs") / "models" / name / dataset
    models_dir.mkdir(parents=True, exist_ok=True)
    # "labels" gives the phrase ID of each predict_scores column, for inference (main.py).
    joblib.dump({"model": model, "labels": [present[i] for i in model.classes_], "params": params,
                 "features": cfg["features"]}, models_dir / f"{split}.joblib")
    info = {"params": params, "val_accuracy": max(t["val_accuracy"] for t in tried), "grid": tried,
            "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(test)}
    return pred, info


def run(name: str, dataset: str, splits: Sequence[str], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a model on several splits and save the results with ``support.evaluate``."""
    cfg = cfg if cfg is not None else load_config()
    classes = list(load_labels(dataset, cfg)["phrase_id"])
    preds, details = {}, {}
    for split in splits:
        preds[split], details[split] = run_split(name, dataset, split, cfg, classes)
        print(f"{name} · {dataset} · {split}: best {details[split]['params']} "
              f"(validation accuracy {details[split]['val_accuracy']:.3f})")
    extra = {"model": name, "features": cfg["features"], "selection": "best validation accuracy",
             "splits": {s: {k: v for k, v in d.items() if k != "grid"} for s, d in details.items()},
             "grids": {s: d["grid"] for s, d in details.items()}}
    return save_run(name, dataset, preds, classes, extra, cfg)


def parse_splits(value: str) -> list[str]:
    """``all`` → every split, ``folds`` → the 5 folds, otherwise one split name."""
    if value == "all":
        return list(SPLIT_NAMES)
    if value == "folds":
        return SPLIT_NAMES[1:]
    if value not in SPLIT_NAMES:
        raise argparse.ArgumentTypeError(f"--split must be all, folds or one of {SPLIT_NAMES}")
    return [value]


def main() -> None:
    """Train and evaluate one traditional model."""
    parser = argparse.ArgumentParser(description="MFCC features + traditional classifier, both evaluation protocols.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--split", default="all", type=parse_splits, help="all (default), folds, random or foldN")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    summary = run(args.model, args.dataset, args.split, load_config(args.config))
    print(json.dumps({k: summary[k] for k in ("random", "folds") if k in summary}, indent=2)[:1500])


if __name__ == "__main__":
    main()

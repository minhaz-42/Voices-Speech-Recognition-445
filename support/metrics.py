"""Shared evaluation metrics, used for every model so the numbers are comparable.

- :func:`compute_metrics`: accuracy (primary), macro F1 and weighted F1
- :func:`classification_report_table`: per-class precision, recall, F1 and support
- :func:`confusion_matrix_table` and :func:`plot_confusion_matrix`
- :func:`accuracy_by_group` and :func:`per_speaker_accuracy`

Labels are phrase IDs such as ``"CMD02"``. Pass the full phrase list as ``labels``
so tables and confusion matrices keep the same row order for every model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def _check_lengths(*arrays: Sequence) -> None:
    """Raise if the arrays are empty or differ in length."""
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"inputs differ in length: {sorted(lengths)}")
    if lengths == {0}:
        raise ValueError("inputs are empty")


def compute_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Accuracy, macro F1 and weighted F1.

    F1 is averaged over the classes that occur in ``y_true`` or ``y_pred``; a class
    that is predicted but never correct scores 0.

    Returns:
        ``{"n", "accuracy", "macro_f1", "weighted_f1"}``.
    """
    _check_lengths(y_true, y_pred)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def classification_report_table(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support, plus macro and weighted averages.

    Args:
        y_true: True phrase IDs.
        y_pred: Predicted phrase IDs.
        labels: Row order. Defaults to the sorted classes in ``y_true`` and ``y_pred``.
            Averages are computed over the classes that occur, as in :func:`compute_metrics`.

    Returns:
        A table indexed by label, with ``macro avg`` and ``weighted avg`` as the last rows.
    """
    _check_lengths(y_true, y_pred)
    present = sorted(set(y_true) | set(y_pred))
    labels = list(labels) if labels is not None else present
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    table = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=pd.Index(labels, name="label"))
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average=average, zero_division=0)
        table.loc[f"{average} avg"] = [p, r, f, len(y_true)]
    table["support"] = table["support"].astype(int)
    return table


def confusion_matrix_table(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> pd.DataFrame:
    """Confusion matrix as a table: rows are true labels, columns are predictions."""
    _check_lengths(y_true, y_pred)
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="predicted"))


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence | None = None,
    path: str | Path | None = None,
    normalize: bool = True,
    title: str | None = None,
) -> Figure:
    """Draw the confusion matrix as a heatmap and optionally save it.

    Args:
        y_true: True phrase IDs.
        y_pred: Predicted phrase IDs.
        labels: Axis order. Defaults to the sorted classes in ``y_true`` and ``y_pred``.
        path: If given, the figure is saved there (PNG, PDF or SVG by extension).
        normalize: Show each row as a fraction of its true class (recall) instead of counts.
        title: Figure title.

    Returns:
        The matplotlib figure.
    """
    table = confusion_matrix_table(y_true, y_pred, labels)
    values = table.to_numpy(dtype=float)
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)

    n = len(table)
    size = max(5.0, 0.22 * n + 2.5)
    fig = Figure(figsize=(size + 1.0, size))
    ax = fig.add_subplot()
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class" if normalize else "clips")

    font = 10 if n <= 15 else 6
    ax.set_xticks(range(n), labels=table.columns, rotation=90, fontsize=font)
    ax.set_yticks(range(n), labels=table.index, fontsize=font)
    ax.set_xlabel("Predicted phrase")
    ax.set_ylabel("True phrase")
    if title:
        ax.set_title(title)
    if n <= 15:
        threshold = values.max() / 2 if values.size else 0
        for i in range(n):
            for j in range(n):
                text = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > threshold else "black")
    fig.tight_layout()

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def accuracy_by_group(y_true: Sequence, y_pred: Sequence, groups: Sequence, name: str = "group") -> pd.DataFrame:
    """Accuracy within each group, e.g. per speaker or per phrase category.

    Returns:
        A table with columns ``name, n, correct, accuracy``, sorted by group.
    """
    _check_lengths(y_true, y_pred, groups)
    frame = pd.DataFrame({name: list(groups), "correct": np.asarray(y_true) == np.asarray(y_pred)})
    table = frame.groupby(name, sort=True)["correct"].agg(n="size", correct="sum").reset_index()
    table["correct"] = table["correct"].astype(int)
    table["accuracy"] = table["correct"] / table["n"]
    return table


def per_speaker_accuracy(y_true: Sequence, y_pred: Sequence, speakers: Sequence) -> pd.DataFrame:
    """Accuracy for each speaker: columns ``speaker_id, n, correct, accuracy``."""
    return accuracy_by_group(y_true, y_pred, speakers, name="speaker_id")

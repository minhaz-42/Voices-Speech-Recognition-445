"""Figures for the runs kept in one member folder (task T49): more than confusion matrices.

Every figure is drawn only from the files in ``results/<folder>/`` (plus the subset and fold
lists in ``data/``), so it always matches the numbers in the tables. A figure whose input
files do not exist yet is skipped. Output: ``results/<folder>/figures/``.

| File | Chart | Read from |
|---|---|---|
| ``seen_vs_unseen.png`` | dumbbell: random-split vs unseen-speaker accuracy per run and dataset, each fold as a dot | ``summary.json`` |
| ``training_curves_<dataset>.png`` | training / validation loss and validation accuracy per epoch, mean and range over the folds, best epochs marked | ``summary.json`` (``splits.*.curve``) |
| ``noise_robustness_<dataset>.png`` | accuracy vs SNR on unseen and on seen speakers, with the range of the folds | ``noise/<run>_<dataset>.csv`` and ``_predictions.csv`` |
| ``per_speaker_<dataset>.png`` | heatmap: unseen-speaker accuracy per speaker and run, with gender, first language and test fold | ``folds/per_speaker.csv`` |
| ``confidence_<dataset>.png`` | score of right vs wrong answers, and accuracy when only the most confident answers are kept | ``predictions_fold*.csv`` (``confidence``) |
| ``speed_vs_accuracy_<dataset>.png`` | time per clip vs unseen-speaker accuracy, bubble area = model size | ``benchmark_<dataset>.csv``, ``summary.json`` |
| ``whisper_errors.png`` | Whisper's word / character error rate per first language and per speaker, and per clip | ``wer_per_l1.csv``, ``wer_per_speaker.csv``, predictions |

Usage::

    python -m support.figures --folder M1_tanvir                  # every figure, both datasets
    python -m support.figures --folder M1_tanvir --dataset fsc
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from support.config import load_config, resolve_path
from support.dataset import load_subset
from support.evaluate import FOLD_NAMES, list_runs, results_root, run_dir

DATASET_NAMES = {"fsc": "TenVoices-FSC", "l2arctic": "TenVoices-L2"}
SNR_ORDER = ["clean", "20", "10", "5"]
_BASE_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "knn": "k-NN", "svm": "SVM",
               "logreg": "Logistic regression", "w2v2_untrained": "wav2vec 2.0, untrained",
               "w2v2_probe": "wav2vec 2.0 linear probe", "whisper_small": "Whisper small, zero-shot"}
_SSL_NAMES = {"w2v2": "wav2vec 2.0", "wavlm": "WavLM Base+", "hubert": "HuBERT Base"}
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
_DEVICE_NAMES = {"mps": "Apple GPU", "cuda": "NVIDIA GPU", "cpu": "CPU"}


# ---------------------------------------------------------------- labels and inputs

def run_label(run: str) -> str:
    """Readable name of a run, e.g. ``w2v2_15ep_noise`` → ``wav2vec 2.0, 15 epochs + noise``."""
    if run in _BASE_NAMES:
        return _BASE_NAMES[run]
    match = re.fullmatch(r"(w2v2|wavlm|hubert)_(\d+)ep(_noise)?", run)
    if match:
        return f"{_SSL_NAMES[match[1]]}, {match[2]} epochs" + (" + noise" if match[3] else "")
    return run


def run_colors(runs: Sequence[str]) -> dict[str, str]:
    """One fixed colour per run, in the order of the folder's run list."""
    return {run: _COLORS[i % len(_COLORS)] for i, run in enumerate(runs)}


def folder_runs(folder: str, cfg: dict[str, Any]) -> list[str]:
    """The runs listed for a member folder under ``result_folders`` in the config."""
    runs = (cfg.get("result_folders") or {}).get(folder)
    if not runs:
        raise KeyError(f"no runs listed for {folder!r} under result_folders in the config")
    return list(runs)


def load_summaries(folder: str, cfg: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """``{(run, dataset): summary}`` for every saved run of the folder, in the folder's run order."""
    runs = folder_runs(folder, cfg)
    found = list_runs(cfg=cfg)
    found = found[found["folder"] == folder]
    order = {run: i for i, run in enumerate(runs)}
    found = found.assign(order=found["run"].map(order)).sort_values(["order", "dataset"])
    return {(r.run, r.dataset): json.loads((Path(r.path) / "summary.json").read_text(encoding="utf-8"))
            for r in found.itertuples()}


def fold_predictions(run: str, dataset: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """The test predictions of every fold of a run, in one table with a ``split`` column."""
    root = run_dir(run, dataset, cfg)
    frames = [pd.read_csv(root / f"predictions_{name}.csv", dtype={"speaker_id": str}).assign(split=name)
              for name in FOLD_NAMES if (root / f"predictions_{name}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def speaker_info(dataset: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """``speaker_id, gender, l1, fold`` per speaker (``fold`` = the fold that tests the speaker)."""
    speakers = load_subset(dataset, cfg).groupby("speaker_id", as_index=False)[["gender", "l1"]].first()
    folds_file = resolve_path(cfg, "splits") / dataset / "folds.csv"
    test_fold = {}
    if folds_file.exists():
        for row in pd.read_csv(folds_file).itertuples():
            for speaker in str(row.test).split(";"):
                test_fold[speaker] = int(str(row.fold).removeprefix("fold"))
    return speakers.assign(fold=speakers["speaker_id"].map(test_fold)).sort_values(["fold", "gender", "speaker_id"],
                                                                                   ignore_index=True)


def training_curves(summary: dict[str, Any]) -> pd.DataFrame:
    """``split, epoch, train_loss, val_loss, val_accuracy, best`` from a fine-tuned run's summary."""
    rows = []
    for split, info in (summary.get("splits") or {}).items():
        for point in info.get("curve") or []:
            rows.append({"split": split, **point, "best": point["epoch"] == info.get("best_epoch")})
    return pd.DataFrame(rows)


def noise_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per SNR and fold (``snr_db, split, accuracy``) from a noise test's predictions."""
    folds = predictions[predictions["split"].isin(FOLD_NAMES)]
    correct = (folds["label"] == folds["pred"]).rename("accuracy")
    return correct.groupby([folds["snr_db"].astype(str), folds["split"]]).mean().reset_index()


def coverage_curve(confidence: Sequence[float], correct: Sequence[bool]) -> pd.DataFrame:
    """Accuracy on the clips kept when only the most confident answers are accepted.

    Clips are ranked by confidence (highest first); equal confidences enter together.

    Returns:
        ``coverage`` (fraction of clips kept) and ``accuracy`` (on the kept clips), one row
        per distinct confidence value, ending at coverage 1.
    """
    table = pd.DataFrame({"confidence": np.asarray(confidence, float), "correct": np.asarray(correct, bool)})
    grouped = table.groupby("confidence")["correct"].agg(["sum", "count"]).sort_index(ascending=False)
    kept, right = grouped["count"].cumsum(), grouped["sum"].cumsum()
    return pd.DataFrame({"coverage": (kept / len(table)).to_numpy(), "accuracy": (right / kept).to_numpy()})


# ---------------------------------------------------------------- plots

def _percent_axis(ax, axis: str = "x") -> None:
    ticks = np.linspace(0, 1, 6)
    if axis == "x":
        ax.set_xlim(-0.02, 1.04)
        ax.set_xticks(ticks, labels=[f"{t:.0%}" for t in ticks])
    else:
        ax.set_ylim(-0.02, 1.04)
        ax.set_yticks(ticks, labels=[f"{t:.0%}" for t in ticks])
    ax.grid(axis=axis, alpha=0.3)


def plot_seen_vs_unseen(summaries: dict[tuple[str, str], dict[str, Any]], chance: float | None = None,
                        colors: dict[str, str] | None = None) -> Figure:
    """Dumbbell chart: accuracy on seen speakers (random split) vs unseen speakers (pooled folds)."""
    datasets = list(dict.fromkeys(ds for _, ds in summaries))
    runs = list(dict.fromkeys(run for run, _ in summaries))
    colors = colors or run_colors(runs)
    fig = Figure(figsize=(6.2 * len(datasets), 0.65 * len(runs) + 2.2))
    axes = fig.subplots(1, len(datasets), squeeze=False)[0]
    for ax, dataset in zip(axes, datasets):
        shown = [run for run in runs if (run, dataset) in summaries]
        for y, run in enumerate(reversed(shown)):
            s = summaries[(run, dataset)]
            seen = s.get("random", {}).get("accuracy")
            folds = s.get("folds", {})
            unseen = folds.get("pooled", {}).get("accuracy")
            if seen is not None and unseen is not None:
                ax.plot([seen, unseen], [y, y], color="0.75", lw=3, zorder=1)
            per_fold = [m["accuracy"] for m in folds.get("per_fold", {}).values()]
            ax.scatter(per_fold, y + np.linspace(-0.07, 0.07, len(per_fold)), s=14, color=colors[run], alpha=0.45,
                       zorder=2)
            if seen is not None:
                ax.scatter([seen], [y], s=90, facecolor="white", edgecolor=colors[run], lw=2, zorder=3)
                ax.annotate(f"{seen:.1%}", (seen, y), xytext=(0, 11), textcoords="offset points", ha="center",
                            fontsize=8, color="0.35")
            if unseen is not None:
                ax.scatter([unseen], [y], s=90, marker="D", color=colors[run], zorder=3)
                ax.annotate(f"{unseen:.1%}", (unseen, y), xytext=(0, -19), textcoords="offset points",
                            ha="center", fontsize=8, fontweight="bold", color=colors[run])
        ax.set_yticks(range(len(shown)), labels=[run_label(r) for r in reversed(shown)])
        ax.set_ylim(-0.7, len(shown) - 0.3)
        _percent_axis(ax, "x")
        if chance:
            ax.axvline(chance, color="0.5", ls=":", lw=1)
            ax.text(chance, len(shown) - 0.45, " chance", fontsize=8, color="0.4", va="top")
        ax.set_xlabel("accuracy")
        ax.set_title(DATASET_NAMES.get(dataset, dataset))
    handles = [Line2D([], [], ls="", marker="o", mfc="white", mec="0.2", ms=8, label="seen speakers (random split)"),
               Line2D([], [], ls="", marker="D", color="0.2", ms=7, label="unseen speakers (5 folds pooled)"),
               Line2D([], [], ls="", marker="o", color="0.2", alpha=0.45, ms=4, label="one fold")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Seen vs unseen speakers", fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_training_curves(curves: dict[str, pd.DataFrame], dataset: str) -> Figure:
    """Per run: loss and validation accuracy per epoch, mean over the folds with their range."""
    fig = Figure(figsize=(11, 3.4 * len(curves) + 0.6))
    axes = fig.subplots(len(curves), 2, squeeze=False)
    for (loss_ax, acc_ax), (run, table) in zip(axes, curves.items()):
        folds = table[table["split"].isin(FOLD_NAMES)]
        for column, style, name in (("train_loss", "-", "training loss"), ("val_loss", "--", "validation loss")):
            stats = folds.groupby("epoch")[column].agg(["mean", "min", "max"])
            loss_ax.plot(stats.index, stats["mean"], style, color="#1f77b4" if column == "train_loss" else "#d62728",
                         lw=2, label=f"{name} (mean of folds)")
            loss_ax.fill_between(stats.index, stats["min"], stats["max"], alpha=0.15,
                                 color="#1f77b4" if column == "train_loss" else "#d62728")
        loss_ax.set_ylabel("cross-entropy loss")
        loss_ax.set_title(f"{run_label(run)}: loss", fontsize=10)
        loss_ax.legend(fontsize=8, frameon=False)
        for split, part in folds.groupby("split"):
            acc_ax.plot(part["epoch"], part["val_accuracy"], color="0.6", lw=0.8)
            best = part[part["best"]]
            acc_ax.scatter(best["epoch"], best["val_accuracy"], marker="*", s=70, color="#ff7f0e", zorder=3)
        stats = folds.groupby("epoch")["val_accuracy"].agg(["mean", "min", "max"])
        acc_ax.plot(stats.index, stats["mean"], color="#2ca02c", lw=2.2, label="mean of the 5 folds")
        acc_ax.fill_between(stats.index, stats["min"], stats["max"], color="#2ca02c", alpha=0.12)
        random = table[table["split"] == "random"]
        if len(random):
            acc_ax.plot(random["epoch"], random["val_accuracy"], ":", color="0.25", lw=1.4, label="random split")
        acc_ax.plot([], [], color="0.6", lw=0.8, label="one fold")
        acc_ax.scatter([], [], marker="*", s=70, color="#ff7f0e", label="epoch kept (best validation)")
        _percent_axis(acc_ax, "y")
        acc_ax.set_ylabel("validation accuracy")
        acc_ax.set_title(f"{run_label(run)}: validation (an unseen speaker per fold)", fontsize=10)
        acc_ax.legend(fontsize=8, frameon=False, loc="lower right")
        for ax in (loss_ax, acc_ax):
            ax.set_xlabel("epoch")
            ax.set_xticks(sorted(table["epoch"].unique()))
    fig.suptitle(f"Fine-tuning curves · {DATASET_NAMES.get(dataset, dataset)}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_noise(noise: dict[str, tuple[pd.DataFrame, pd.DataFrame]], dataset: str, chance: float | None = None,
               colors: dict[str, str] | None = None) -> Figure:
    """Accuracy vs SNR, on unseen speakers (pooled folds, with the range of the folds) and on seen speakers."""
    colors = colors or run_colors(list(noise))
    fig = Figure(figsize=(11, 4.8))
    unseen_ax, seen_ax = fig.subplots(1, 2, sharey=True)
    points: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for run, (summary, by_fold) in noise.items():
        summary = summary.assign(snr_db=summary["snr_db"].astype(str))
        for ax, protocol in ((unseen_ax, "folds"), (seen_ax, "random")):
            part = summary[summary["protocol"] == protocol].set_index("snr_db").reindex(SNR_ORDER)
            if part["accuracy"].isna().all():
                continue
            ax.plot(range(len(SNR_ORDER)), part["accuracy"], marker="o", lw=2, color=colors[run], label=run_label(run))
            for x, value in enumerate(part["accuracy"]):
                if not np.isnan(value):
                    points.setdefault((id(ax), x), []).append((run, value))
            if protocol == "folds" and len(by_fold):
                spread = by_fold.groupby("snr_db")["accuracy"].agg(["min", "max"]).reindex(SNR_ORDER)
                unseen_ax.fill_between(range(len(SNR_ORDER)), spread["min"], spread["max"], color=colors[run], alpha=0.12)
    for ax in (unseen_ax, seen_ax):
        for x in range(len(SNR_ORDER)):
            here = points.get((id(ax), x), [])
            for run, value in here:
                # a label goes below its point when a higher point at the same SNR is close above it
                below = any(0 < other - value < 0.08 for _, other in here)
                ax.annotate(f"{value:.0%}", (x, value), xytext=(7, -11 if below else 4), textcoords="offset points",
                            fontsize=7.5, color=colors[run])
    labels = ["clean", "20 dB", "10 dB", "5 dB"]
    for ax, title in ((unseen_ax, "Unseen speakers (5 folds pooled; band = range of the folds)"),
                      (seen_ax, "Seen speakers (random split)")):
        ax.set_xticks(range(len(SNR_ORDER)), labels=labels)
        ax.set_xlim(-0.3, len(SNR_ORDER) - 0.6)
        ax.set_xlabel("noise added to the test clips (signal-to-noise ratio)")
        ax.set_title(title, fontsize=10)
        if chance:
            ax.axhline(chance, color="0.5", ls=":", lw=1)
            ax.text(-0.27, chance, "chance", fontsize=7.5, color="0.4", ha="left", va="bottom")
    _percent_axis(unseen_ax, "y")
    seen_ax.grid(axis="y", alpha=0.3)
    unseen_ax.set_ylabel("accuracy")
    handles, names = unseen_ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.suptitle(f"Noise robustness · {DATASET_NAMES.get(dataset, dataset)}", fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_per_speaker(table: pd.DataFrame, speakers: pd.DataFrame, dataset: str) -> Figure:
    """Heatmap of unseen-speaker accuracy: one row per run, one column per speaker (grouped by test fold)."""
    order = [s for s in speakers["speaker_id"] if s in table.columns]
    values = table[order].to_numpy(dtype=float)
    info = speakers.set_index("speaker_id")
    labels = []
    for speaker in order:
        row = info.loc[speaker]
        name = speaker if len(speaker) <= 5 else speaker[:5] + "…"
        detail = str(row["gender"])[:1].upper() if dataset == "fsc" else f"{row['l1']}, {str(row['gender'])[:1].upper()}"
        fold = f"fold {int(row['fold'])}" if pd.notna(row["fold"]) else ""
        labels.append(f"{name}\n{detail}\n{fold}")
    fig = Figure(figsize=(1.0 * len(order) + 3.5, 0.62 * len(table) + 2.0))
    ax = fig.add_subplot()
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if values[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(order)), labels=labels, fontsize=8)
    ax.set_yticks(range(len(table)), labels=[run_label(r) for r in table.index], fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="accuracy")
    ax.set_title(f"Accuracy per speaker, each tested when never heard in training · "
                 f"{DATASET_NAMES.get(dataset, dataset)}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confidence(predictions: dict[str, pd.DataFrame], dataset: str, colors: dict[str, str] | None = None) -> Figure:
    """Scores of right vs wrong answers per run, and accuracy when only confident answers are kept."""
    colors = colors or run_colors(list(predictions))
    n = len(predictions)
    fig = Figure(figsize=(11, max(4.2, 2.1 * n + 0.8)))
    grid = fig.add_gridspec(n, 2, width_ratios=[1.15, 1])
    curve_ax = fig.add_subplot(grid[:, 1])
    for i, (run, table) in enumerate(predictions.items()):
        ax = fig.add_subplot(grid[i, 0])
        correct = (table["label"] == table["pred"]).to_numpy()
        bins = np.linspace(0, table["confidence"].max() * 1.02, 30)
        ax.hist(table.loc[correct, "confidence"], bins=bins, color=colors[run], alpha=0.75,
                label=f"right ({correct.sum()})")
        ax.hist(table.loc[~correct, "confidence"], bins=bins, histtype="step", color="0.15", lw=1.4,
                label=f"wrong ({(~correct).sum()})")
        ax.set_title(run_label(run), fontsize=9.5)
        ax.set_ylabel("clips")
        ax.legend(fontsize=8, frameon=False)
        if i == n - 1:
            ax.set_xlabel("score of the predicted phrase (top-1)")
        curve = coverage_curve(table["confidence"], correct)
        curve_ax.plot(curve["coverage"], curve["accuracy"], color=colors[run], lw=2, label=run_label(run))
        curve_ax.scatter([1], [correct.mean()], color=colors[run], zorder=3)
    _percent_axis(curve_ax, "x")
    _percent_axis(curve_ax, "y")
    curve_ax.invert_xaxis()
    curve_ax.set_xlabel("clips answered (the rest are rejected as uncertain)")
    curve_ax.set_ylabel("accuracy on the answered clips")
    curve_ax.set_title("Rejecting uncertain answers", fontsize=10)
    curve_ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle(f"How confident are the models? · {DATASET_NAMES.get(dataset, dataset)}, unseen speakers",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_speed_vs_accuracy(table: pd.DataFrame, dataset: str, colors: dict[str, str] | None = None) -> Figure:
    """Time per clip (log scale) vs unseen-speaker accuracy; bubble area shows the model size."""
    colors = colors or run_colors(list(dict.fromkeys(table["run"])))
    fig = Figure(figsize=(7.5, 5.0))
    ax = fig.add_subplot()
    for i, row in enumerate(table.itertuples()):
        ax.scatter(row.mean_ms_per_clip, row.accuracy, s=40 + 2.2 * row.model_size_mb, color=colors[row.run],
                   alpha=0.45 if row.device == "cpu" else 0.9, edgecolor="0.2", lw=0.8, zorder=3)
        up = i % 2 == 0
        ax.annotate(f"{_DEVICE_NAMES.get(row.device, row.device)}\n{row.mean_ms_per_clip:.0f} ms", (row.mean_ms_per_clip,
                    row.accuracy), xytext=(0, 22 if up else -22), textcoords="offset points", fontsize=8,
                    ha="center", va="bottom" if up else "top")
    sizes = table.groupby("run", sort=False)["model_size_mb"].first()
    handles = [Line2D([], [], ls="", marker="o", ms=9, color=colors[run], label=f"{run_label(run)} ({mb:.0f} MB)")
               for run, mb in sizes.items()]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="center right")
    ax.set_xscale("log")
    low, high = table["mean_ms_per_clip"].min(), table["mean_ms_per_clip"].max()
    ax.set_xlim(low / 2, high * 2.5)
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000) if low / 2 <= t <= high * 2.5]
    ax.set_xticks(ticks, labels=[str(t) for t in ticks])
    _percent_axis(ax, "y")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("time per clip incl. pre-processing (ms, log scale)")
    ax.set_ylabel("accuracy on unseen speakers")
    ax.set_title(f"Speed vs accuracy · {DATASET_NAMES.get(dataset, dataset)} (bubble area = model size)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_whisper_errors(per_l1: pd.DataFrame, per_speaker: pd.DataFrame, clips: pd.DataFrame) -> Figure:
    """Whisper error rates per first language, per speaker and per clip, on both datasets.

    Args:
        per_l1: ``dataset, l1, wer, cer`` (from ``wer_per_l1.csv``).
        per_speaker: ``dataset, speaker_id, wer`` (+ ``l1``).
        clips: ``dataset, wer, label, pred`` for the pooled folds.
    """
    fig = Figure(figsize=(14, 4.6))
    bar_ax, spk_ax, cdf_ax = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.4, 0.8, 1.1]})
    names = [("FSC: " if ds == "fsc" else "L2: ") + str(l1).replace("English (United States)", "US English")
             for ds, l1 in zip(per_l1["dataset"], per_l1["l1"])]
    x = np.arange(len(per_l1))
    bar_ax.bar(x - 0.2, per_l1["wer"], 0.4, color="#1f77b4", label="word error rate")
    bar_ax.bar(x + 0.2, per_l1["cer"], 0.4, color="#9ecae1", label="character error rate")
    for xi, value in zip(x, per_l1["wer"]):
        bar_ax.annotate(f"{value:.1%}", (xi - 0.2, value), xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=7.5)
    bar_ax.set_xticks(x, labels=names, rotation=25, ha="right", fontsize=8.5)
    bar_ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    bar_ax.grid(axis="y", alpha=0.3)
    bar_ax.set_title("Per first language", fontsize=10)
    bar_ax.legend(fontsize=8, frameon=False)

    datasets = list(dict.fromkeys(per_speaker["dataset"]))
    for i, dataset in enumerate(datasets):
        part = per_speaker[per_speaker["dataset"] == dataset]
        jitter = np.linspace(-0.15, 0.15, len(part))
        spk_ax.scatter(i + jitter, part["wer"], s=36, color="#1f77b4" if dataset == "fsc" else "#d62728", alpha=0.8)
        spk_ax.hlines(part["wer"].median(), i - 0.3, i + 0.3, color="0.2", lw=1.5)
    spk_ax.set_xticks(range(len(datasets)), labels=[DATASET_NAMES.get(d, d) for d in datasets], fontsize=9)
    spk_ax.set_xlim(-0.6, len(datasets) - 0.4)
    spk_ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    spk_ax.grid(axis="y", alpha=0.3)
    spk_ax.set_ylabel("word error rate")
    spk_ax.set_title("Per speaker (line = median)", fontsize=10)

    for dataset in datasets:
        part = clips[clips["dataset"] == dataset]
        values = np.sort(part["wer"].to_numpy())
        wrong = int((part["label"] != part["pred"]).sum())
        cdf_ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", lw=2,
                    color="#1f77b4" if dataset == "fsc" else "#d62728",
                    label=f"{DATASET_NAMES.get(dataset, dataset)} ({wrong} of {len(part)} clips matched wrongly)")
    cdf_ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _percent_axis(cdf_ax, "y")
    cdf_ax.set_xlabel("word error rate of the clip")
    cdf_ax.set_ylabel("share of clips with at most this error")
    cdf_ax.set_title("Per clip (cumulative)", fontsize=10)
    cdf_ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Whisper small, zero-shot: transcription errors on unseen speakers", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- drawing a whole folder

def _save(fig: Figure, path: Path, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    written.append(path)


def draw_all(folder: str, cfg: dict[str, Any] | None = None, datasets: Sequence[str] | None = None,
             out: str | Path | None = None) -> list[Path]:
    """Draw every figure the folder's results allow; returns the files written.

    Args:
        folder: Member folder under ``result_folders`` in the config, e.g. ``M1_tanvir``.
        cfg: Settings (default: config file).
        datasets: Datasets to draw (default: every dataset with results).
        out: Output folder (default: ``results/<folder>/figures/``).
    """
    cfg = cfg if cfg is not None else load_config()
    runs = folder_runs(folder, cfg)
    root = resolve_path(cfg, "results") / folder
    out = Path(out) if out is not None else root / "figures"
    summaries = load_summaries(folder, cfg)
    if datasets is not None:
        summaries = {key: s for key, s in summaries.items() if key[1] in datasets}
    shown = list(dict.fromkeys(ds for _, ds in summaries))
    written: list[Path] = []
    if not summaries:
        return written

    colors = run_colors(runs)
    chance = {ds: 1 / cfg["datasets"][ds]["n_phrases"] for ds in shown}
    _save(plot_seen_vs_unseen(summaries, min(chance.values()), colors), out / "seen_vs_unseen.png", written)

    for dataset in shown:
        here = [run for run in runs if (run, dataset) in summaries]
        curves = {run: training_curves(summaries[(run, dataset)]) for run in here}
        curves = {run: c for run, c in curves.items() if len(c)}
        if curves:
            _save(plot_training_curves(curves, dataset), out / f"training_curves_{dataset}.png", written)

        noise = {}
        for run in here:
            file = results_root(run, cfg) / "noise" / f"{run}_{dataset}.csv"
            if file.exists():
                predictions = file.with_name(f"{run}_{dataset}_predictions.csv")
                by_fold = noise_by_fold(pd.read_csv(predictions, dtype={"snr_db": str})) if predictions.exists() \
                    else pd.DataFrame(columns=["snr_db", "split", "accuracy"])
                noise[run] = (pd.read_csv(file, dtype={"snr_db": str}), by_fold)
        if noise:
            _save(plot_noise(noise, dataset, chance[dataset], colors), out / f"noise_robustness_{dataset}.png", written)

        per_speaker = {}
        for run in here:
            file = run_dir(run, dataset, cfg) / "folds" / "per_speaker.csv"
            if file.exists():
                per_speaker[run] = pd.read_csv(file, dtype={"speaker_id": str}).set_index("speaker_id")["accuracy"]
        if per_speaker:
            table = pd.DataFrame(per_speaker).T
            _save(plot_per_speaker(table, speaker_info(dataset, cfg), dataset), out / f"per_speaker_{dataset}.png",
                  written)

        confident = {run: p for run in here if "confidence" in (p := fold_predictions(run, dataset, cfg)).columns}
        if confident:
            _save(plot_confidence(confident, dataset, colors), out / f"confidence_{dataset}.png", written)

        bench_file = root / f"benchmark_{dataset}.csv"
        if bench_file.exists():
            bench = pd.read_csv(bench_file)
            bench = bench[bench["run"].isin(here)]
            bench = bench.assign(accuracy=[summaries[(r, dataset)]["folds"]["pooled"]["accuracy"] for r in bench["run"]])
            if len(bench):
                _save(plot_speed_vs_accuracy(bench, dataset, colors), out / f"speed_vs_accuracy_{dataset}.png", written)

    whisper = [run for run in runs if any((run_dir(run, ds, cfg) / "wer_per_l1.csv").exists() for ds in shown)]
    if whisper:
        run = whisper[0]
        per_l1, per_spk, clips = [], [], []
        for dataset in shown:
            root_ds = run_dir(run, dataset, cfg)
            if (root_ds / "wer_per_l1.csv").exists():
                per_l1.append(pd.read_csv(root_ds / "wer_per_l1.csv").assign(dataset=dataset))
                per_spk.append(pd.read_csv(root_ds / "wer_per_speaker.csv", dtype={"speaker_id": str}).assign(dataset=dataset))
                clips.append(fold_predictions(run, dataset, cfg).assign(dataset=dataset))
        _save(plot_whisper_errors(pd.concat(per_l1, ignore_index=True), pd.concat(per_spk, ignore_index=True),
                                  pd.concat(clips, ignore_index=True)), out / "whisper_errors.png", written)
    return written


def main() -> None:
    """Draw every figure of a member folder."""
    parser = argparse.ArgumentParser(description="Figures for the runs of one member folder.")
    parser.add_argument("--folder", required=True, help="member folder under result_folders, e.g. M1_tanvir")
    parser.add_argument("--dataset", default="all", help="fsc, l2arctic or all (default)")
    parser.add_argument("--out", default=None, help="output folder (default: results/<folder>/figures/)")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    datasets = None if args.dataset == "all" else [args.dataset]
    for path in draw_all(args.folder, load_config(args.config), datasets, args.out):
        print(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)


if __name__ == "__main__":
    main()

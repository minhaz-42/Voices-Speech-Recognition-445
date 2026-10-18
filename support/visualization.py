"""Figures of what the models "see" and chart helpers for the results.

Every function draws on a :class:`matplotlib.figure.Figure` (not ``pyplot``), like
``plot_confusion_matrix`` in ``support/metrics.py``, so the code works in scripts,
tests and the Gradio demo. Each function returns the figure; an optional ``path``
saves it as a 150 dpi PNG and creates the folder.

Usage::

    python -m support.visualization clip.wav --out images/example_overview.png
    python -m support.visualization clip.wav --out images/example_overview.png         --preprocessing images/example_preprocessing.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from support.audio import peak_dbfs, preprocess, read_audio, to_mono
from support.config import load_config, resolve_path

DPI = 150
N_FFT = 512        # 32 ms window at 16 kHz, as in the MFCC features
HOP_LENGTH = 160   # 10 ms hop
N_MELS = 64        # bands of the log-mel spectrogram


def _save(fig: Figure, path: str | Path | None) -> Figure:
    """Save ``fig`` to ``path`` (if given) at :data:`DPI`, creating the folder, and return it."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI)
    return fig


def _draw_waveform(ax, samples: np.ndarray, sample_rate: int, title: str | None) -> None:
    """Draw amplitude over time in seconds on ``ax``."""
    samples = np.asarray(samples, dtype=np.float32)
    times = np.arange(samples.size) / float(sample_rate)
    ax.plot(times, samples, linewidth=0.6, color="tab:blue")
    ax.set_xlim(0.0, max(times[-1] if samples.size else 0.0, 1.0 / sample_rate))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"Waveform ({sample_rate / 1000:g} kHz, {samples.size / sample_rate:.2f} s)")


def plot_waveform(
    samples: np.ndarray,
    sample_rate: int,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Draw the amplitude of a mono clip over time.

    Args:
        samples: Mono float audio in [-1, 1].
        sample_rate: Sample rate in Hz.
        title: Figure title. Defaults to the sample rate and duration.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.
    """
    fig = Figure(figsize=(8, 2.8))
    _draw_waveform(fig.add_subplot(), samples, sample_rate, title)
    fig.tight_layout()
    return _save(fig, path)


def _draw_spectrogram(
    fig: Figure, ax, samples: np.ndarray, sample_rate: int, kind: str, title: str | None, colorbar: bool = True
) -> None:
    """Draw a log-power (``"linear"``) or 64-band log-mel (``"mel"``) spectrogram in dB on ``ax``."""
    samples = np.asarray(samples, dtype=np.float32)
    hop = HOP_LENGTH
    if kind == "linear":
        power = np.abs(librosa.stft(samples, n_fft=N_FFT, hop_length=hop)) ** 2
        y_axis, label = "linear", "Log-power spectrogram"
    elif kind == "mel":
        power = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate, n_fft=N_FFT, hop_length=hop, n_mels=N_MELS
        )
        y_axis, label = "mel", f"Log-mel spectrogram ({N_MELS} bands)"
    else:
        raise ValueError(f"kind must be 'linear' or 'mel', got {kind!r}")
    db = librosa.power_to_db(power, ref=np.max(power) if power.size and power.max() > 0 else 1.0)
    image = librosa.display.specshow(
        db, sr=sample_rate, hop_length=hop, x_axis="time", y_axis=y_axis, ax=ax, cmap="magma"
    )
    if colorbar:
        fig.colorbar(image, ax=ax, format="%+.0f dB", label="dB")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or f"{label} (dB)")


def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    kind: str = "linear",
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Draw the spectrogram of a mono clip in dB.

    Args:
        samples: Mono float audio.
        sample_rate: Sample rate in Hz.
        kind: ``"linear"`` for a log-power spectrogram (STFT), ``"mel"`` for a
            64-band log-mel spectrogram.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: For an unknown ``kind``.
    """
    fig = Figure(figsize=(8, 3.2))
    _draw_spectrogram(fig, fig.add_subplot(), samples, sample_rate, kind, title)
    fig.tight_layout()
    return _save(fig, path)


def _draw_mfcc(
    fig: Figure, ax, samples: np.ndarray, sample_rate: int, n_mfcc: int, title: str | None, colorbar: bool = True
) -> None:
    """Draw an MFCC heatmap (coefficient index over time) on ``ax``."""
    samples = np.asarray(samples, dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sample_rate, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40
    )
    image = librosa.display.specshow(
        mfcc, sr=sample_rate, hop_length=HOP_LENGTH, x_axis="time", ax=ax, cmap="coolwarm"
    )
    if colorbar:
        fig.colorbar(image, ax=ax, label="value")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC coefficient")
    ax.set_title(title or f"{n_mfcc} MFCCs")


def plot_mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Draw the MFCCs of a mono clip as a heatmap.

    This figure is for display only; the models' features come from ``support/features.py``.

    Args:
        samples: Mono float audio.
        sample_rate: Sample rate in Hz.
        n_mfcc: Number of coefficients.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.
    """
    fig = Figure(figsize=(8, 3.2))
    _draw_mfcc(fig, fig.add_subplot(), samples, sample_rate, n_mfcc, title)
    fig.tight_layout()
    return _save(fig, path)


def clip_overview(
    samples: np.ndarray,
    sample_rate: int,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """One figure with four panels: waveform, log-power spectrogram, log-mel spectrogram and MFCCs.

    Args:
        samples: Mono float audio.
        sample_rate: Sample rate in Hz.
        title: Overall title above the panels.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure with exactly four axes.
    """
    fig = Figure(figsize=(9, 11))
    axes = fig.subplots(4, 1)
    _draw_waveform(axes[0], samples, sample_rate, None)
    # No colour bars, so the figure has exactly four axes and the panels share one time axis width.
    _draw_spectrogram(fig, axes[1], samples, sample_rate, "linear", "Log-power spectrogram (dB)", colorbar=False)
    _draw_spectrogram(fig, axes[2], samples, sample_rate, "mel", f"Log-mel spectrogram, {N_MELS} bands (dB)",
                      colorbar=False)
    _draw_mfcc(fig, axes[3], samples, sample_rate, 13, "13 MFCCs", colorbar=False)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, path)


def plot_preprocessing(
    path_to_audio: str | Path,
    audio_cfg: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
) -> Figure:
    """Show a clip before and after the shared pre-processing.

    Top: the raw waveform at its own sample rate (mixed to mono for display).
    Bottom: the processed waveform (16 kHz, trimmed, peak-normalised, padded to 6 s).
    Durations and peak levels are in the titles.

    Args:
        path_to_audio: Audio file.
        audio_cfg: The ``audio`` section of the config. Defaults to ``configs/config.yaml``.
        out_path: If given, the figure is saved there.

    Returns:
        The matplotlib figure with two axes.
    """
    cfg = audio_cfg if audio_cfg is not None else load_config()["audio"]
    raw, raw_sr = read_audio(path_to_audio)
    processed = preprocess(raw, raw_sr, cfg, pad=True)
    target_sr = int(cfg["sample_rate"])
    raw_mono = to_mono(raw)
    channels = 1 if raw.ndim == 1 else raw.shape[1]

    fig = Figure(figsize=(9, 5.5))
    top, bottom = fig.subplots(2, 1)
    _draw_waveform(top, raw_mono, raw_sr,
                   f"Before: {raw_sr / 1000:g} kHz, {channels} channel(s), {raw_mono.size / raw_sr:.2f} s, "
                   f"peak {peak_dbfs(raw_mono):.1f} dBFS")
    _draw_waveform(bottom, processed, target_sr,
                   f"After: {target_sr / 1000:g} kHz mono, trimmed, normalised, padded to "
                   f"{processed.size / target_sr:.2f} s, peak {peak_dbfs(processed):.1f} dBFS")
    fig.tight_layout()
    return _save(fig, out_path)


def _require_columns(table: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``ValueError`` naming every column of ``columns`` that ``table`` lacks."""
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"table is missing columns: {missing}")


def plot_model_comparison(
    table: pd.DataFrame,
    metric: str = "accuracy",
    error_cols: tuple[str, str] | None = None,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Bar chart with one bar per model.

    Args:
        table: One row per model, with a ``model`` column and the ``metric`` column.
        metric: Column to plot (values in [0, 1] are shown as fractions).
        error_cols: Optional ``(low, high)`` columns, e.g. a bootstrap confidence
            interval, drawn as error bars.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If a required column is missing.
    """
    _require_columns(table, ["model", metric] + list(error_cols or []))
    values = table[metric].to_numpy(dtype=float)
    yerr = None
    if error_cols:
        low = table[error_cols[0]].to_numpy(dtype=float)
        high = table[error_cols[1]].to_numpy(dtype=float)
        yerr = np.vstack([values - low, high - values])

    fig = Figure(figsize=(max(6.0, 0.8 * len(table) + 2.0), 4.5))
    ax = fig.add_subplot()
    positions = np.arange(len(table))
    bars = ax.bar(positions, values, yerr=yerr, capsize=4 if yerr is not None else 0, color="tab:blue")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=8)
    ax.set_xticks(positions, labels=table["model"].astype(str), rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    if np.nanmax(values, initial=0.0) <= 1.0:
        ax.set_ylim(0.0, 1.08)
    ax.set_title(title or f"{metric.replace('_', ' ').capitalize()} per model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_accuracy_vs_snr(
    table: pd.DataFrame,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Line chart of accuracy against signal-to-noise ratio, one line per model.

    Args:
        table: Columns ``model``, ``snr_db`` and ``accuracy``. Clean audio can be
            given as ``snr_db = "clean"`` (as in ``noise/*.csv``) or ``inf``; it is
            drawn at the right edge, labelled "clean".
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If a required column is missing.
    """
    _require_columns(table, ["model", "snr_db", "accuracy"])
    snr = pd.to_numeric(table["snr_db"], errors="coerce").replace([np.inf], np.nan)
    finite = snr.dropna()
    clean_x = (finite.max() + 5.0) if not finite.empty else 0.0
    data = table.assign(_x=snr.fillna(clean_x))

    fig = Figure(figsize=(7, 4.5))
    ax = fig.add_subplot()
    for model, rows in data.groupby("model", sort=False):
        rows = rows.sort_values("_x")
        ax.plot(rows["_x"], rows["accuracy"].astype(float), marker="o", label=str(model))
    ticks = sorted(data["_x"].unique())
    labels = ["clean" if (x == clean_x and snr.isna().any()) else f"{x:g}" for x in ticks]
    ax.set_xticks(ticks, labels=labels)
    ax.set_xlabel("Signal-to-noise ratio (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title or "Accuracy vs noise level")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_per_speaker(
    table: pd.DataFrame,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Bar chart of accuracy per speaker, grouped by model when a ``model`` column exists.

    Args:
        table: Columns ``speaker_id`` and ``accuracy`` (e.g. ``folds/per_speaker.csv``),
            optionally ``model``.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If a required column is missing.
    """
    _require_columns(table, ["speaker_id", "accuracy"])
    data = table if "model" in table.columns else table.assign(model="")
    pivot = data.pivot_table(index="speaker_id", columns="model", values="accuracy", aggfunc="mean", sort=False)
    speakers = [str(s) for s in pivot.index]
    models = list(pivot.columns)
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(speakers))

    fig = Figure(figsize=(max(7.0, 0.6 * len(speakers) * max(len(models), 1) + 2.0), 4.5))
    ax = fig.add_subplot()
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(positions + offset, pivot[model].to_numpy(dtype=float), width, label=str(model) or None)
    ax.set_xticks(positions, labels=speakers, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Speaker")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title or "Accuracy per speaker")
    ax.grid(axis="y", alpha=0.3)
    if any(str(m) for m in models):
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_top_k(
    top: list[dict],
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Horizontal bar chart of a recogniser's best phrases and their confidence, best on top.

    Args:
        top: ``[{"phrase_id", "text", "score"}, ...]`` as returned by
            ``support.inference`` recognisers, best first.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If ``top`` is empty.
    """
    if not top:
        raise ValueError("no predictions to plot")
    labels = [f"{r['phrase_id']}  {r.get('text', '')}".strip() for r in top][::-1]
    scores = [float(r["score"]) for r in top][::-1]
    fig = Figure(figsize=(7, 0.55 * len(top) + 1.2))
    ax = fig.add_subplot()
    colors = ["tab:blue"] * (len(top) - 1) + ["tab:orange"]  # the best phrase stands out
    bars = ax.barh(range(len(top)), scores, color=colors)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in scores], padding=3, fontsize=9)
    ax.set_yticks(range(len(top)), labels=labels, fontsize=9)
    ax.set_xlim(0.0, 1.15)
    ax.set_xlabel("Confidence")
    ax.set_title(title or f"Top {len(top)} phrases")
    fig.tight_layout()
    return _save(fig, path)


def _results_cfg(results: str | Path | None) -> dict[str, Any]:
    """The project config, with ``paths.results`` pointing at ``results`` when given."""
    cfg = load_config()
    if results is not None:
        cfg["paths"]["results"] = str(Path(results).resolve())
    return cfg


def results_table(results: str | Path | None = None, dataset: str | None = None) -> pd.DataFrame:
    """Accuracy and macro F1 of every saved run, from the runs' ``summary.json`` files.

    Runs are found with ``support.evaluate.list_runs`` (so Tanvir's runs in
    ``results/M1_tanvir/`` are included). A run without folds or without a random split
    simply has no row for that protocol.

    Args:
        results: Results folder (default: ``paths.results`` of the config).
        dataset: Keep only this dataset (default: all).

    Returns:
        ``run, dataset, protocol, accuracy, macro_f1`` with protocol ``random`` or ``unseen``.
    """
    import json

    from support.evaluate import list_runs

    rows = []
    for run in list_runs(dataset, _results_cfg(results)).itertuples():
        summary = json.loads((Path(run.path) / "summary.json").read_text(encoding="utf-8"))
        if "random" in summary:
            rows.append({"run": run.run, "dataset": run.dataset, "protocol": "random",
                         "accuracy": summary["random"]["accuracy"], "macro_f1": summary["random"]["macro_f1"]})
        if "folds" in summary:
            pooled = summary["folds"]["pooled"]
            rows.append({"run": run.run, "dataset": run.dataset, "protocol": "unseen",
                         "accuracy": pooled["accuracy"], "macro_f1": pooled["macro_f1"]})
    return pd.DataFrame(rows, columns=["run", "dataset", "protocol", "accuracy", "macro_f1"])


PROTOCOL_LABELS = {"random": "random split (speakers seen)", "unseen": "unseen speakers (5 folds)"}


def plot_protocol_comparison(
    table: pd.DataFrame,
    metric: str = "accuracy",
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Grouped bars: per run, the random split next to unseen speakers.

    Args:
        table: Rows of :func:`results_table` (columns ``run``, ``protocol`` and ``metric``).
        metric: ``accuracy`` or ``macro_f1``.
        title: Figure title.
        path: If given, the figure is saved there.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If a required column is missing.
    """
    _require_columns(table, ["run", "protocol", metric])
    pivot = table.pivot_table(index="run", columns="protocol", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=[p for p in PROTOCOL_LABELS if p in pivot.columns])
    if "unseen" in pivot.columns:  # weakest to strongest on unseen speakers, the headline
        pivot = pivot.sort_values("unseen", na_position="first")
    runs, protocols = list(pivot.index), list(pivot.columns)
    width = 0.8 / max(len(protocols), 1)
    positions = np.arange(len(runs))

    fig = Figure(figsize=(max(7.0, 1.1 * len(runs) + 2.0), 4.8))
    ax = fig.add_subplot()
    for i, protocol in enumerate(protocols):
        values = pivot[protocol].to_numpy(dtype=float)
        bars = ax.bar(positions + (i - (len(protocols) - 1) / 2) * width, values, width,
                      label=PROTOCOL_LABELS[protocol])
        ax.bar_label(bars, labels=["" if np.isnan(v) else f"{v:.2f}" for v in values], padding=2, fontsize=7)
    ax.set_xticks(positions, labels=runs, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_ylim(0.0, 1.1)
    ax.set_title(title or f"{metric.replace('_', ' ').capitalize()}: seen vs unseen speakers")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def per_speaker_table(results: str | Path | None = None, dataset: str = "fsc") -> pd.DataFrame:
    """Unseen-speaker accuracy per speaker for every run (``folds/per_speaker.csv``).

    Returns:
        ``model, speaker_id, accuracy``; empty when no run has folds.
    """
    from support.evaluate import list_runs

    frames = []
    for run in list_runs(dataset, _results_cfg(results)).itertuples():
        path = Path(run.path) / "folds" / "per_speaker.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"speaker_id": str}).assign(model=run.run))
    if not frames:
        return pd.DataFrame(columns=["model", "speaker_id", "accuracy"])
    return pd.concat(frames, ignore_index=True)[["model", "speaker_id", "accuracy"]]


def noise_table(results: str | Path | None = None, dataset: str = "fsc", protocol: str = "folds") -> pd.DataFrame:
    """Accuracy per noise level from every ``noise/<run>_<dataset>.csv`` under the results folder.

    Returns:
        ``model, snr_db, accuracy`` for one protocol (``folds`` = unseen speakers); empty
        when no noise test exists yet.
    """
    root = resolve_path(_results_cfg(results), "results")
    frames = [pd.read_csv(p, dtype={"snr_db": str}) for p in sorted(root.rglob(f"noise/*_{dataset}.csv"))]
    if not frames:
        return pd.DataFrame(columns=["model", "snr_db", "accuracy"])
    table = pd.concat(frames, ignore_index=True)
    table = table[(table["dataset"] == dataset) & (table["protocol"] == protocol)]
    return table.rename(columns={"run": "model"})[["model", "snr_db", "accuracy"]].reset_index(drop=True)


def main() -> None:
    """Draw the overview (and optionally the pre-processing figure) of one audio file."""
    parser = argparse.ArgumentParser(description="Draw waveform, spectrograms and MFCCs of one audio file.")
    parser.add_argument("path", help="audio file (WAV, FLAC, OGG or MP3)")
    parser.add_argument("--out", required=True, help="PNG for the four-panel overview")
    parser.add_argument("--preprocessing", default=None, help="also save the before/after pre-processing PNG here")
    parser.add_argument("--title", default=None, help="title of the overview (default: the file name)")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()

    audio_cfg = load_config(args.config)["audio"]
    raw, raw_sr = read_audio(args.path)
    processed = preprocess(raw, raw_sr, audio_cfg, pad=False)
    clip_overview(processed, int(audio_cfg["sample_rate"]), title=args.title or Path(args.path).name, path=args.out)
    print(f"saved: {args.out}")
    if args.preprocessing:
        plot_preprocessing(args.path, audio_cfg, out_path=args.preprocessing)
        print(f"saved: {args.preprocessing}")


if __name__ == "__main__":
    main()

"""Quality report of a selected TenVoices subset (docs/DATASETS.md §3).

Every clip of ``data/subsets/<dataset>.csv`` is checked on its source file:

| Check | Rule | On failure |
|---|---|---|
| Format | readable, mono, the dataset's sample rate | replace (the selection already excludes these) |
| Duration | 0.5–6 s after trimming | replace (the selection already excludes these) |
| Clipping | peak below −0.5 dBFS | flag |
| Silence | leading and trailing silence under 2 s each | flag |
| Noise floor | estimated SNR ≥ 15 dB | flag |

Flagged clips stay in the subset (real-world audio is not perfect) and are reported in
the datasheet. A random 10 % of the clips (seed 42) is marked ``listen_check = yes`` for a
person to confirm the words match the text (task T09); that check never blocks the splits.

Output (committed): ``data/subsets/qc_<dataset>.csv`` and ``qc_<dataset>_summary.json``.

Usage::

    python -m support.qc_report --dataset fsc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from support.audio import peak_dbfs, to_mono
from support.config import PROJECT_ROOT, load_config
from support.dataset import dataset_config, load_subset

EXPECTED_RATE = {"fsc": 16000, "l2arctic": 44100}
LIMITS = {"min_s": 0.5, "max_s": 6.0, "clip_dbfs": -0.5, "silence_s": 2.0, "snr_db": 15.0, "listen_share": 0.10}


def estimate_snr_db(samples: np.ndarray, sample_rate: int) -> float:
    """Rough SNR: loud frames (95th percentile of 25 ms frame RMS) over quiet frames (10th percentile), in dB."""
    frame, hop = int(0.025 * sample_rate), int(0.010 * sample_rate)
    if samples.size < frame:
        return float("nan")
    rms = librosa.feature.rms(y=samples, frame_length=frame, hop_length=hop, center=False)[0]
    rms = np.maximum(rms, 1e-10)
    return float(20 * np.log10(np.percentile(rms, 95) / np.percentile(rms, 10)))


def check_clip(path: Path, expected_rate: int, top_db: float, duration_s: float) -> dict[str, Any]:
    """Measurements and flags of one source clip."""
    try:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as err:  # noqa: BLE001
        return {"readable": False, "status": "replace", "flags": f"unreadable ({type(err).__name__})"}
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    mono = to_mono(samples)
    _, (start, end) = librosa.effects.trim(mono, top_db=top_db)
    row = {"readable": True, "channels": channels, "sample_rate": rate, "duration_s": duration_s,
           "peak_dbfs": round(peak_dbfs(mono), 2), "lead_silence_s": round(start / rate, 3),
           "trail_silence_s": round((mono.size - end) / rate, 3), "snr_db": round(estimate_snr_db(mono, rate), 1)}
    replace, flags = [], []
    if channels != 1 or rate != expected_rate:
        replace.append("format")
    if not LIMITS["min_s"] <= duration_s <= LIMITS["max_s"]:
        replace.append("duration")
    if row["peak_dbfs"] >= LIMITS["clip_dbfs"]:
        flags.append("clipping")
    if row["lead_silence_s"] >= LIMITS["silence_s"] or row["trail_silence_s"] >= LIMITS["silence_s"]:
        flags.append("silence")
    if not row["snr_db"] >= LIMITS["snr_db"]:
        flags.append("low_snr")
    row["status"] = "replace" if replace else ("flag" if flags else "pass")
    row["flags"] = ";".join(replace + flags)
    return row


def qc_report(dataset: str, cfg: dict[str, Any] | None = None, write: bool = True) -> tuple[pd.DataFrame, dict]:
    """Check every clip of a subset. Returns ``(table, summary)``."""
    cfg = cfg if cfg is not None else load_config()
    subset = load_subset(dataset, cfg)
    raw = Path(dataset_config(dataset, cfg)["raw"])
    raw = raw if raw.is_absolute() else PROJECT_ROOT / raw
    top_db = float(cfg["audio"]["trim_top_db"])
    rows = [{"utt_id": r.utt_id, **check_clip(raw / r.source_path, EXPECTED_RATE[dataset], top_db, float(r.duration_s))}
            for r in subset.itertuples()]
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    n_listen = int(round(LIMITS["listen_share"] * len(table)))
    listen = set(rng.choice(table["utt_id"].to_numpy(), size=n_listen, replace=False))
    table["listen_check"] = np.where(table["utt_id"].isin(listen), "yes", "")
    flag_counts = table["flags"].str.split(";").explode().value_counts()
    summary = {
        "dataset": dataset, "n": len(table),
        "pass": int((table["status"] == "pass").sum()), "flag": int((table["status"] == "flag").sum()),
        "replace": int((table["status"] == "replace").sum()),
        "pass_rate": round(float((table["status"] == "pass").mean()), 4),
        "flags": {k: int(v) for k, v in flag_counts.items() if k},
        "listen_check_clips": n_listen, "limits": LIMITS,
        "snr_db_median": float(table["snr_db"].median()), "peak_dbfs_max": float(table["peak_dbfs"].max()),
    }
    if write:
        out = Path(dataset_config(dataset, cfg)["subset"])
        out = (out if out.is_absolute() else PROJECT_ROOT / out).parent
        table.to_csv(out / f"qc_{dataset}.csv", index=False)
        (out / f"qc_{dataset}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return table, summary


def main() -> None:
    """Write the QC report of one subset and print its summary."""
    parser = argparse.ArgumentParser(description="Quality report of a selected TenVoices subset.")
    parser.add_argument("--dataset", required=True, choices=list(EXPECTED_RATE))
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    _, summary = qc_report(args.dataset, load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

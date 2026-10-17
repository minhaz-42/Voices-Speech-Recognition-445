"""Synthetic TenVoices subsets for tests: same columns as data/subsets/<dataset>.csv, no real audio."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from support.config import load_config
from support.dataset import SUBSET_COLUMNS, sha256_file

L2_SPEAKERS = [  # speaker, gender, first language (the fixed TenVoices-L2 design)
    ("ASI", "M", "Hindi"), ("RRBI", "M", "Hindi"), ("SVBI", "F", "Hindi"), ("TNI", "F", "Hindi"),
    ("ABA", "M", "Arabic"), ("SKA", "F", "Arabic"), ("BWC", "M", "Mandarin"), ("LXC", "F", "Mandarin"),
    ("EBVS", "M", "Spanish"), ("NJS", "F", "Spanish"),
]
L1_ORDER = ["Hindi", "Arabic", "Mandarin", "Spanish"]


def fsc_speakers() -> list[tuple[str, str, str]]:
    """Ten speakers, five female and five male, with FSC-style IDs."""
    return [(f"spk{i:02d}", "F" if i % 2 else "M", "English") for i in range(1, 11)]


def make_subset(dataset: str = "fsc", speakers=None, n_phrases: int = 50, takes: int = 2) -> pd.DataFrame:
    """A subset table with every speaker saying every phrase ``takes`` times."""
    speakers = speakers or (L2_SPEAKERS if dataset == "l2arctic" else fsc_speakers())
    rows = []
    for spk, gender, l1 in speakers:
        for p in range(1, n_phrases + 1):
            pid = f"P{p:02d}"
            for take in range(1, takes + 1):
                utt = f"{dataset}_{spk}_{pid}_t{take}"
                rows.append({"utt_id": utt, "dataset": dataset, "speaker_id": spk, "gender": gender,
                             "l1": l1, "phrase_id": pid, "text": f"phrase number {p}", "take": str(take),
                             "source_path": f"{spk}/{utt}.wav", "duration_s": "1.0", "sha256": ""})
    return pd.DataFrame(rows, columns=SUBSET_COLUMNS)


def write_audio(subset: pd.DataFrame, raw_dir: Path, sample_rate: int = 22050, seed: int = 0) -> pd.DataFrame:
    """Write a short tone with silence around it for every clip; fill in ``sha256``."""
    rng = np.random.default_rng(seed)
    subset = subset.copy()
    for i, row in subset.iterrows():
        freq = 150 + 80 * int(row["phrase_id"][1:])  # a clearly different pitch per phrase
        t = np.arange(int(0.5 * sample_rate)) / sample_rate
        tone = 0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.size)
        pad = np.zeros(int(0.2 * sample_rate))
        path = raw_dir / row["source_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.concatenate([pad, tone, pad]), sample_rate)
        subset.at[i, "sha256"] = sha256_file(path)
    return subset


def make_config(tmp: Path, dataset: str, subset: pd.DataFrame) -> dict:
    """The project config with every data path redirected into ``tmp``; writes the subset CSV."""
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["processed"] = str(tmp / "processed")
    cfg["paths"]["splits"] = str(tmp / "splits")
    dcfg = cfg["datasets"][dataset]
    dcfg["raw"] = str(tmp / "raw")
    dcfg["subset"] = str(tmp / "subset.csv")
    dcfg["phrases"] = str(tmp / "phrases.csv")
    subset.to_csv(tmp / "subset.csv", index=False)
    phrases = subset.drop_duplicates("phrase_id")[["phrase_id", "text"]].assign(category="test")
    phrases[["phrase_id", "category", "text"]].to_csv(tmp / "phrases.csv", index=False)
    return cfg

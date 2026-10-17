"""Make the random split and the 5 speaker-independent folds of one TenVoices subset.

Rules (docs/PROJECT_PLAN.md, *Evaluation Protocol*), applied to each dataset separately:

- **Random split (seen speakers):** stratified 80/20 by phrase (seed 42); 10% of the
  training part becomes validation (stratified when every phrase can be represented,
  otherwise a plain seeded draw).
- **Speaker-independent folds (unseen speakers):** 5 folds, each with 2 test speakers
  (one female, one male), 1 validation speaker and 7 training speakers; every speaker
  is a test speaker exactly once and a validation speaker at most once. With
  ``fold_by_l1`` (TenVoices-L2) the two test speakers of a fold share a first language,
  and the folds follow the language order of the configured speaker list.

Output (committed, small): ``data/splits/<dataset>/random.csv``, ``fold1.csv`` …
``fold5.csv`` with ``utt_id, speaker_id, phrase_id, part`` and ``folds.csv`` with the
speakers of every fold. Every split is checked for leakage before it is written.

Usage::

    python -m tools.make_splits --dataset fsc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from support.config import load_config, resolve_path
from support.dataset import PARTS, SPLIT_COLUMNS, dataset_config, load_subset


def _gender(value: str) -> str:
    """``f`` or ``m`` from values such as ``F``, ``female`` or ``Male``."""
    g = str(value).strip().lower()[:1]
    if g not in {"f", "m"}:
        raise ValueError(f"gender must be female or male for the fold design, got {value!r}")
    return g


def random_split(subset: pd.DataFrame, seed: int = 42, test_size: float = 0.2,
                 val_size: float = 0.1) -> pd.DataFrame:
    """Stratified train / val / test split of the clips (seen speakers).

    Returns:
        ``utt_id, speaker_id, phrase_id, part`` in the subset's order.
    """
    idx = subset.index.to_numpy()
    labels = subset["phrase_id"].to_numpy()
    trainval, test = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    n_val = int(round(val_size * len(trainval)))
    tv_labels = subset.loc[trainval, "phrase_id"].to_numpy()
    stratify = tv_labels if n_val >= len(set(tv_labels)) else None
    train, val = train_test_split(trainval, test_size=n_val, stratify=stratify, random_state=seed)

    parts = pd.Series("", index=subset.index)
    parts[train], parts[val], parts[test] = "train", "val", "test"
    out = subset[["utt_id", "speaker_id", "phrase_id"]].copy()
    out["part"] = parts
    return out.reset_index(drop=True)


def speaker_folds(speakers: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                  fold_by_l1: bool = False, l1_order: list[str] | None = None) -> pd.DataFrame:
    """Assign test, validation and training speakers to each fold.

    Args:
        speakers: One row per speaker with ``speaker_id, gender, l1``.
        n_folds: Number of folds; there must be exactly ``n_folds`` female and male speakers.
        seed: Seed for pairing female and male speakers.
        fold_by_l1: Pair test speakers within the same first language.
        l1_order: Order of the language groups (default: alphabetical).

    Returns:
        ``fold, test, val, train`` where ``fold`` is ``fold1`` … and the speaker
        columns hold sorted lists of speaker IDs.
    """
    speakers = speakers.assign(g=speakers["gender"].map(_gender)).sort_values("speaker_id")
    counts = speakers["g"].value_counts()
    if counts.get("f", 0) != n_folds or counts.get("m", 0) != n_folds:
        raise ValueError(f"need {n_folds} female and {n_folds} male speakers, got {dict(counts)}")

    rng = np.random.default_rng(seed)
    if fold_by_l1:
        groups = l1_order or sorted(speakers["l1"].unique())
        unknown = set(speakers["l1"]) - set(groups)
        if unknown:
            raise ValueError(f"first languages missing from l1_order: {sorted(unknown)}")
        group_list = [speakers[speakers["l1"] == l1] for l1 in groups if (speakers["l1"] == l1).any()]
    else:
        group_list = [speakers]

    pairs = []
    for group in group_list:
        f = list(group.loc[group["g"] == "f", "speaker_id"])
        m = list(group.loc[group["g"] == "m", "speaker_id"])
        if len(f) != len(m):
            raise ValueError(f"cannot pair speakers of {group['l1'].iloc[0]!r}: {len(f)} female, {len(m)} male")
        rng.shuffle(f)
        rng.shuffle(m)
        pairs.extend(zip(f, m))

    if len(pairs) != n_folds:
        raise ValueError(f"expected {n_folds} test pairs, got {len(pairs)}")
    everyone = set(speakers["speaker_id"])
    rows = []
    for k, (f, m) in enumerate(pairs):
        nxt_f, nxt_m = pairs[(k + 1) % n_folds]
        val = nxt_f if k % 2 == 0 else nxt_m      # alternate the validation speaker's gender
        rows.append({"fold": f"fold{k + 1}", "test": sorted([f, m]), "val": [val],
                     "train": sorted(everyone - {f, m, val})})
    return pd.DataFrame(rows)


def fold_split(subset: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """The ``part`` of every clip in one fold."""
    part = {s: "test" for s in fold["test"]} | {s: "val" for s in fold["val"]} | {s: "train" for s in fold["train"]}
    out = subset[["utt_id", "speaker_id", "phrase_id"]].copy()
    out["part"] = out["speaker_id"].map(part)
    return out.reset_index(drop=True)


def check_split(split: pd.DataFrame, subset: pd.DataFrame, speaker_independent: bool) -> None:
    """Raise ``ValueError`` on any leakage or bookkeeping error.

    Every clip of the subset appears exactly once with a known part, and in a
    speaker-independent split no speaker appears in more than one part.
    """
    if sorted(split["utt_id"]) != sorted(subset["utt_id"]):
        raise ValueError("the split does not cover every clip exactly once")
    if not set(split["part"]) <= set(PARTS) or split["part"].isna().any():
        raise ValueError(f"unknown parts: {sorted(set(split['part'].dropna()) - set(PARTS))}")
    if speaker_independent:
        parts_per_speaker = split.groupby("speaker_id")["part"].nunique()
        leaking = sorted(parts_per_speaker[parts_per_speaker > 1].index)
        if leaking:
            raise ValueError(f"speakers in more than one part: {leaking}")


def check_folds(folds: pd.DataFrame, speakers: pd.DataFrame, fold_by_l1: bool) -> None:
    """Every speaker is tested exactly once; each test pair is one female and one male
    (and shares a first language with ``fold_by_l1``)."""
    tested = [s for t in folds["test"] for s in t]
    if sorted(tested) != sorted(speakers["speaker_id"]):
        raise ValueError("every speaker must be a test speaker exactly once")
    info = speakers.set_index("speaker_id")
    for fold in folds.itertuples():
        if sorted(_gender(info.loc[s, "gender"]) for s in fold.test) != ["f", "m"]:
            raise ValueError(f"{fold.fold}: test speakers must be one female and one male")
        if fold_by_l1 and info.loc[fold.test, "l1"].nunique() != 1:
            raise ValueError(f"{fold.fold}: test speakers must share a first language")
        if set(fold.val) & set(fold.test) or set(fold.train) & (set(fold.test) | set(fold.val)):
            raise ValueError(f"{fold.fold}: a speaker is in more than one part")


def make_splits(dataset: str, cfg: dict[str, Any] | None = None,
                out_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Make, check and save every split of one dataset.

    Returns:
        ``{"random": df, "fold1": df, …, "folds": df}``.
    """
    cfg = cfg if cfg is not None else load_config()
    dcfg = dataset_config(dataset, cfg)
    scfg = cfg.get("splits", {})
    seed = int(cfg.get("seed", 42))
    n_folds = int(scfg.get("n_folds", 5))
    fold_by_l1 = bool(dcfg.get("fold_by_l1", False))

    subset = load_subset(dataset, cfg)
    speakers = subset.groupby("speaker_id", as_index=False)[["gender", "l1"]].first()
    l1_order = None
    if fold_by_l1 and dcfg.get("speakers"):
        l1_of = speakers.set_index("speaker_id")["l1"]
        l1_order = list(dict.fromkeys(l1_of[s] for s in dcfg["speakers"] if s in l1_of.index))

    splits: dict[str, pd.DataFrame] = {
        "random": random_split(subset, seed, float(scfg.get("test_size", 0.2)), float(scfg.get("val_size", 0.1)))}
    check_split(splits["random"], subset, speaker_independent=False)
    folds = speaker_folds(speakers, n_folds, seed, fold_by_l1, l1_order)
    check_folds(folds, speakers, fold_by_l1)
    for fold in folds.itertuples(index=False):
        split = fold_split(subset, pd.Series(fold._asdict()))
        check_split(split, subset, speaker_independent=True)
        splits[fold.fold] = split

    out_dir = Path(out_dir) if out_dir is not None else resolve_path(cfg, "splits") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split[SPLIT_COLUMNS].to_csv(out_dir / f"{name}.csv", index=False)
    table = folds.copy()
    for col in ("test", "val", "train"):
        table[col] = table[col].map(";".join)
    table.to_csv(out_dir / "folds.csv", index=False)
    splits["folds"] = folds
    return splits


def main() -> None:
    """Make the splits of one dataset and print the fold design."""
    parser = argparse.ArgumentParser(description="Make the random split and speaker-independent folds.")
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()

    splits = make_splits(args.dataset, load_config(args.config))
    counts = splits["random"]["part"].value_counts()
    print(f"random: train {counts.get('train', 0)}, val {counts.get('val', 0)}, test {counts.get('test', 0)}")
    for fold in splits["folds"].itertuples():
        print(f"{fold.fold}: test {', '.join(fold.test)} · val {fold.val[0]} · train {len(fold.train)} speakers")


if __name__ == "__main__":
    main()

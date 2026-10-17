"""Load the TenVoices subsets, build their processed 16 kHz copies and serve the splits.

Data flow (docs/DATASETS.md):

1. ``data/subsets/<dataset>.csv`` lists the selected clips (T42, committed).
2. ``python -m support.dataset --dataset fsc --build`` checks every source clip's
   SHA-256, applies the shared pre-processing (mono, 16 kHz, trimmed, peak-normalised,
   not padded) and writes ``data/processed/<dataset>/<utt_id>.wav`` plus
   ``manifest.csv`` (gitignored: the licences forbid re-publishing audio).
3. ``tools/make_splits.py`` writes ``data/splits/<dataset>/{random,fold1..fold5}.csv``
   (committed).
4. :func:`load_split` joins a split with the manifest, so every model trains and
   tests on exactly the same clips.

Usage::

    python -m support.dataset --dataset fsc --build
    python -m support.dataset --dataset l2arctic --summary
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from support.audio import load_audio, save_wav
from support.config import PROJECT_ROOT, load_config, resolve_path

SUBSET_COLUMNS = ["utt_id", "dataset", "speaker_id", "gender", "l1", "phrase_id", "text",
                  "take", "source_path", "duration_s", "sha256"]
MANIFEST_COLUMNS = ["utt_id", "dataset", "speaker_id", "gender", "l1", "label", "text", "take",
                    "path", "duration_s", "source_sha256"]
SPLIT_COLUMNS = ["utt_id", "speaker_id", "phrase_id", "part"]
PARTS = ("train", "val", "test")
SPLIT_NAMES = ["random"] + [f"fold{k}" for k in range(1, 6)]


def _resolve(path: str | Path) -> Path:
    """Absolute paths stay; relative ones are taken from the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    """A path relative to the project root when possible (for portable manifests)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def dataset_config(dataset: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``datasets.<dataset>`` section of the config.

    Raises:
        KeyError: If the dataset is not configured.
    """
    cfg = cfg if cfg is not None else load_config()
    try:
        return cfg["datasets"][dataset]
    except KeyError:
        known = ", ".join(cfg.get("datasets", {}))
        raise KeyError(f"unknown dataset '{dataset}' (configured: {known})") from None


def load_subset(dataset: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read and check ``data/subsets/<dataset>.csv``.

    Raises:
        FileNotFoundError: If the subset has not been selected yet (task T42).
        ValueError: If columns are missing, IDs repeat or rows belong to another dataset.
    """
    path = _resolve(dataset_config(dataset, cfg)["subset"])
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet: run tools/select_subsets.py (task T42)")
    subset = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in SUBSET_COLUMNS if c not in subset.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if subset["utt_id"].duplicated().any():
        raise ValueError(f"{path} repeats utt_id values: {sorted(subset.loc[subset['utt_id'].duplicated(), 'utt_id'])[:5]}")
    others = set(subset["dataset"]) - {dataset}
    if others:
        raise ValueError(f"{path} has rows of other datasets: {sorted(others)}")
    return subset


def load_labels(dataset: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """The dataset's phrase list (``phrase_id, category, text``), in file order."""
    path = _resolve(dataset_config(dataset, cfg)["phrases"])
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet: run tools/select_subsets.py (task T42)")
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"phrase_id", "text"} - set(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return labels


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def processed_dir(dataset: str, cfg: dict[str, Any] | None = None) -> Path:
    """``data/processed/<dataset>/``."""
    cfg = cfg if cfg is not None else load_config()
    return resolve_path(cfg, "processed") / dataset


def build_processed(dataset: str, cfg: dict[str, Any] | None = None, verify: bool = True,
                    overwrite: bool = False) -> pd.DataFrame:
    """Write the processed 16 kHz copy of every selected clip and the manifest.

    Args:
        dataset: ``fsc`` or ``l2arctic``.
        cfg: Settings (default: ``configs/config.yaml``).
        verify: Check each source file's SHA-256 against the subset list first.
        overwrite: Rewrite processed files that already exist.

    Returns:
        The manifest, also saved as ``data/processed/<dataset>/manifest.csv``.

    Raises:
        FileNotFoundError: If source clips are missing.
        ValueError: If source checksums differ from the subset list.
    """
    cfg = cfg if cfg is not None else load_config()
    subset = load_subset(dataset, cfg)
    raw_root = _resolve(dataset_config(dataset, cfg)["raw"])
    sources = [raw_root / p for p in subset["source_path"]]

    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source clips are missing, e.g. {missing[:3]}")
    if verify:
        bad = [row.utt_id for row, src in zip(subset.itertuples(), sources) if sha256_file(src) != row.sha256]
        if bad:
            raise ValueError(f"{len(bad)} source clips differ from the subset list (SHA-256), e.g. {bad[:3]}")

    out_dir = processed_dir(dataset, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_cfg = cfg["audio"]
    sr = int(audio_cfg["sample_rate"])
    rows = []
    for row, src in zip(subset.itertuples(), sources):
        dst = out_dir / f"{row.utt_id}.wav"
        if overwrite or not dst.exists():
            save_wav(dst, load_audio(src, audio_cfg, pad=False), sr)
        rows.append({
            "utt_id": row.utt_id, "dataset": dataset, "speaker_id": row.speaker_id,
            "gender": row.gender, "l1": row.l1, "label": row.phrase_id, "text": row.text,
            "take": row.take, "path": _relative(dst), "duration_s": round(sf.info(dst).duration, 3),
            "source_sha256": row.sha256,
        })
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def load_manifest(dataset: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read ``data/processed/<dataset>/manifest.csv``.

    Raises:
        FileNotFoundError: If the processed copies have not been built.
    """
    path = processed_dir(dataset, cfg) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run python -m support.dataset --dataset {dataset} --build")
    return pd.read_csv(path, dtype={"utt_id": str, "speaker_id": str, "label": str, "l1": str, "take": str},
                       keep_default_na=False)


def split_path(dataset: str, split: str, cfg: dict[str, Any] | None = None) -> Path:
    """``data/splits/<dataset>/<split>.csv``."""
    cfg = cfg if cfg is not None else load_config()
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split '{split}' (use one of {SPLIT_NAMES})")
    return resolve_path(cfg, "splits") / dataset / f"{split}.csv"


def load_split(dataset: str, split: str, cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """The train / val / test clips of one split, joined with the manifest.

    Returns:
        ``{"train": df, "val": df, "test": df}``; each row has the manifest columns
        (``label`` is the phrase ID, ``path`` the processed WAV).
    """
    path = split_path(dataset, split, cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run python -m tools.make_splits --dataset {dataset}")
    parts = pd.read_csv(path, dtype=str, keep_default_na=False)
    manifest = load_manifest(dataset, cfg)
    joined = parts[["utt_id", "part"]].merge(manifest, on="utt_id", how="left", validate="one_to_one")
    unknown = joined.loc[joined["path"].isna(), "utt_id"]
    if len(unknown):
        raise ValueError(f"{len(unknown)} clips of {path.name} are not in the manifest, e.g. {list(unknown[:3])}")
    return {part: joined[joined["part"] == part].drop(columns="part").reset_index(drop=True) for part in PARTS}


def read_processed(path: str | Path) -> np.ndarray:
    """Read a processed clip (already mono, 16 kHz, trimmed and normalised) as float32."""
    samples, _ = sf.read(str(_resolve(path)), dtype="float32", always_2d=False)
    return samples


def summary(dataset: str, cfg: dict[str, Any] | None = None) -> str:
    """Counts of the subset (and the manifest, when built) as text."""
    subset = load_subset(dataset, cfg)
    lines = [f"{dataset}: {len(subset)} clips, {subset['speaker_id'].nunique()} speakers, "
             f"{subset['phrase_id'].nunique()} phrases",
             "speakers per gender: " + ", ".join(f"{g}={n}" for g, n in
                                                 subset.groupby("gender")["speaker_id"].nunique().items())]
    try:
        manifest = load_manifest(dataset, cfg)
        lines.append(f"processed: {len(manifest)} clips, {manifest['duration_s'].sum() / 60:.1f} min")
    except FileNotFoundError:
        lines.append("processed: not built yet")
    return "\n".join(lines)


def main() -> None:
    """Build the processed copies of a subset or print its summary."""
    parser = argparse.ArgumentParser(description="Build and inspect the processed TenVoices subsets.")
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    parser.add_argument("--build", action="store_true", help="write processed copies and the manifest")
    parser.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check of the source clips")
    parser.add_argument("--overwrite", action="store_true", help="rewrite processed files that exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.build:
        manifest = build_processed(args.dataset, cfg, verify=not args.no_verify, overwrite=args.overwrite)
        print(f"wrote {len(manifest)} clips to {processed_dir(args.dataset, cfg)}")
    print(summary(args.dataset, cfg))


if __name__ == "__main__":
    main()

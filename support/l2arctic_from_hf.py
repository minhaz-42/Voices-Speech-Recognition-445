"""Unpack the free Hugging Face copy of L2-ARCTIC into the corpus's own folder layout.

The publisher's download needs a form; this public copy
(`NathanRoll/l2-arctic-dataset <https://huggingface.co/datasets/NathanRoll/l2-arctic-dataset>`_,
CC BY-NC 4.0 like the corpus) holds 15 of the 24 speakers with the same 150 prompts
(``arctic_a0001`` … ``arctic_a0150``), as the original 44.1 kHz WAV files inside two
parquet files. This tool writes them as

    data/raw/l2arctic/<SPK>/wav/arctic_<id>.wav          (the original bytes, unchanged)
    data/raw/l2arctic/<SPK>/transcript/arctic_<id>.txt

so ``python -m support.select_subsets --dataset l2arctic`` works exactly as with the
official archive. Sources, checksums and checks: ``data/SOURCES.md``.

Usage::

    python -m support.l2arctic_from_hf                      # reads data/raw/l2arctic_hf/*.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from support.config import PROJECT_ROOT

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "l2arctic_hf"
DEFAULT_TARGET = PROJECT_ROOT / "data" / "raw" / "l2arctic"


def unpack(source: str | Path = DEFAULT_SOURCE, target: str | Path = DEFAULT_TARGET) -> dict[str, int]:
    """Write every clip of the parquet files as WAV + transcript; returns clips per speaker.

    Raises:
        FileNotFoundError: If there are no parquet files in ``source``.
        ValueError: If a clip is not a WAV file or a speaker repeats a prompt.
    """
    files = sorted(Path(source).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no .parquet files in {source}")
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for file in files:
        table = pq.read_table(file, columns=["speaker", "file_id", "transcript", "audio"])
        for row in table.to_pylist():
            speaker, prompt = row["speaker"], f"arctic_{row['file_id']}"
            audio = row["audio"]["bytes"]
            if audio[:4] != b"RIFF":
                raise ValueError(f"{speaker}/{prompt} is not a WAV file")
            if (speaker, prompt) in seen:
                raise ValueError(f"{speaker}/{prompt} appears twice")
            seen.add((speaker, prompt))
            folder = Path(target) / speaker
            (folder / "wav").mkdir(parents=True, exist_ok=True)
            (folder / "transcript").mkdir(parents=True, exist_ok=True)
            (folder / "wav" / f"{prompt}.wav").write_bytes(audio)
            (folder / "transcript" / f"{prompt}.txt").write_text(row["transcript"].strip() + "\n", encoding="utf-8")
            counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def main() -> None:
    """Unpack the parquet files and print the clips per speaker."""
    parser = argparse.ArgumentParser(description="Unpack the Hugging Face copy of L2-ARCTIC.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="folder with the .parquet files")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="output folder (default data/raw/l2arctic)")
    args = parser.parse_args()
    counts = unpack(args.source, args.target)
    print(f"{sum(counts.values())} clips from {len(counts)} speakers: {counts}")


if __name__ == "__main__":
    main()

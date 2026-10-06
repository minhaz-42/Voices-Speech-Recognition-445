"""Load and validate the TenVoices phrase catalog."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

from support.config import load_config, resolve_path


class Phrase(TypedDict):
    """One validated phrase from the TenVoices phrase catalog."""

    phrase_id: str
    category: str
    noisy_take: bool
    text: str


REQUIRED_COLUMNS = {"phrase_id", "category", "noisy_take", "text"}
EXPECTED_CATEGORIES = {
    "command",
    "number",
    "everyday",
    "named_entity",
    "phonetic",
}


def load_phrases(path: str | Path | None = None) -> list[Phrase]:
    """Load and validate the TenVoices phrase catalog.

    Args:
        path: CSV path. If omitted, use the configured phrases path.

    Returns:
        A list of validated phrase dictionaries.

    Raises:
        ValueError: If the file has invalid columns or phrase data.
    """
    if path is None:
        cfg = load_config()
        path = resolve_path(cfg, "phrases")

    path = Path(path)

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("phrase CSV has no header")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"phrase CSV is missing columns: {sorted(missing)}")

        phrases: list[Phrase] = []

        for row_number, row in enumerate(reader, start=2):
            phrase_id = (row["phrase_id"] or "").strip()
            category = (row["category"] or "").strip()
            noisy_value = (row["noisy_take"] or "").strip().lower()
            text = (row["text"] or "").strip()

            if not phrase_id:
                raise ValueError(f"row {row_number}: phrase_id is empty")

            if not category:
                raise ValueError(f"row {row_number}: category is empty")

            if category not in EXPECTED_CATEGORIES:
                raise ValueError(
                    f"row {row_number}: unknown category {category!r}"
                )

            if noisy_value not in {"yes", "no"}:
                raise ValueError(
                    f"row {row_number}: noisy_take must be yes or no"
                )

            if not text:
                raise ValueError(f"row {row_number}: text is empty")

            phrases.append(
                {
                    "phrase_id": phrase_id,
                    "category": category,
                    "noisy_take": noisy_value == "yes",
                    "text": text,
                }
            )

    _validate_catalog(phrases)
    return phrases


def _validate_catalog(phrases: list[Phrase]) -> None:
    """Validate the expected TenVoices catalog structure."""
    if len(phrases) != 50:
        raise ValueError(f"expected 50 phrases, found {len(phrases)}")

    phrase_ids = [phrase["phrase_id"] for phrase in phrases]
    if len(set(phrase_ids)) != len(phrase_ids):
        raise ValueError("phrase_id values must be unique")

    categories = {phrase["category"] for phrase in phrases}
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            f"expected categories {sorted(EXPECTED_CATEGORIES)}, "
            f"found {sorted(categories)}"
        )

    for category in EXPECTED_CATEGORIES:
        count = sum(phrase["category"] == category for phrase in phrases)
        if count != 10:
            raise ValueError(
                f"category {category!r} should contain 10 phrases, found {count}"
            )

    noisy_count = sum(phrase["noisy_take"] for phrase in phrases)
    if noisy_count != 10:
        raise ValueError(f"expected 10 noisy phrases, found {noisy_count}")

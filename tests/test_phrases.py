"""Tests for support/phrases.py."""

import tempfile
import unittest
from pathlib import Path

from support.phrases import load_phrases


class TestLoadPhrases(unittest.TestCase):
    def test_project_phrase_catalog(self):
        phrases = load_phrases()

        self.assertEqual(len(phrases), 50)
        self.assertEqual(len({phrase["phrase_id"] for phrase in phrases}), 50)

    def test_each_category_has_ten_phrases(self):
        phrases = load_phrases()

        counts = {}
        for phrase in phrases:
            category = phrase["category"]
            counts[category] = counts.get(category, 0) + 1

        self.assertEqual(
            counts,
            {
                "command": 10,
                "number": 10,
                "everyday": 10,
                "named_entity": 10,
                "phonetic": 10,
            },
        )

    def test_ten_phrases_are_marked_for_noisy_recording(self):
        phrases = load_phrases()

        noisy = [phrase for phrase in phrases if phrase["noisy_take"]]

        self.assertEqual(len(noisy), 10)

    def test_known_phrase_is_loaded_correctly(self):
        phrases = load_phrases()

        cmd02 = next(
            phrase for phrase in phrases if phrase["phrase_id"] == "CMD02"
        )

        self.assertEqual(cmd02["category"], "command")
        self.assertTrue(cmd02["noisy_take"])
        self.assertEqual(cmd02["text"], "Turn off the fan.")

    def test_invalid_noisy_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "phrases.csv"
            path.write_text(
                "phrase_id,category,noisy_take,text\n"
                "CMD01,command,maybe,Turn on the lights.\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_phrases(path)

    def test_duplicate_phrase_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "phrases.csv"
            rows = [
                "phrase_id,category,noisy_take,text",
                "CMD01,command,no,Turn on the lights.",
                "CMD01,command,no,Open the door.",
            ]
            path.write_text("\n".join(rows), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_phrases(path)


if __name__ == "__main__":
    unittest.main()

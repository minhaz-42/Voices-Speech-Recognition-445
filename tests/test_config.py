"""Tests for support/config.py."""

import tempfile
import unittest
from pathlib import Path

from support.config import PROJECT_ROOT, load_config, resolve_path


class TestLoadConfig(unittest.TestCase):
    def test_default_config_has_expected_audio_settings(self):
        cfg = load_config()
        self.assertEqual(cfg["audio"]["sample_rate"], 16000)
        self.assertEqual(cfg["audio"]["max_duration_s"], 6.0)
        self.assertEqual(cfg["seed"], 42)

    def test_missing_section_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("paths:\n  raw: data/raw\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_non_mapping_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.yaml"
            path.write_text("- a\n- b\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


class TestResolvePath(unittest.TestCase):
    def test_relative_path_is_joined_to_project_root(self):
        path = resolve_path(load_config(), "phrases")
        self.assertTrue(path.is_absolute())
        self.assertEqual(path, PROJECT_ROOT / "data" / "prompts" / "phrases.csv")
        self.assertTrue(path.exists())

    def test_absolute_path_is_kept(self):
        absolute = str(Path(tempfile.gettempdir()).resolve())
        cfg = {"paths": {"x": absolute}, "audio": {}}
        self.assertEqual(resolve_path(cfg, "x"), Path(absolute))

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            resolve_path(load_config(), "does_not_exist")


if __name__ == "__main__":
    unittest.main()

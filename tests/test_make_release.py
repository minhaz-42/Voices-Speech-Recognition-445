"""Tests for support.make_release."""

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from support.make_release import build_release_package


class TestMakeRelease(unittest.TestCase):
    """Test building release package and verifying its contents."""

    def test_build_release_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "outputs" / "release"
            out_dir.mkdir(parents=True, exist_ok=True)
            zip_path = out_dir / "test-release.zip"

            # Build the release package to the temporary zip path
            built_path = build_release_package(output_path=zip_path)
            self.assertTrue(built_path.exists())
            self.assertTrue(zipfile.is_zipfile(built_path))

            # Inspect zip contents
            with zipfile.ZipFile(built_path, "r") as zf:
                namelist = zf.namelist()

                # Core files that must be in the release
                self.assertIn("SHA256SUMS", namelist)
                self.assertIn("data/README.md", namelist)
                self.assertIn("data/SOURCES.md", namelist)
                self.assertIn("data/subsets/fsc.csv", namelist)
                self.assertIn("support/select_subsets.py", namelist)
                self.assertIn("support/qc_report.py", namelist)

                # Read and verify SHA256SUMS
                checksums_text = zf.read("SHA256SUMS").decode("utf-8")
                self.assertGreater(len(checksums_text.strip().splitlines()), 5)

                # Verify at least one file's hash against the SHA256SUMS entry
                for line in checksums_text.strip().splitlines():
                    expected_hash, rel_path = line.split("  ", 1)
                    file_bytes = zf.read(rel_path)
                    computed_hash = hashlib.sha256(file_bytes).hexdigest()
                    self.assertEqual(expected_hash, computed_hash, f"Hash mismatch for {rel_path}")


if __name__ == "__main__":
    unittest.main()

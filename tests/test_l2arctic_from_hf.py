"""Tests for support/l2arctic_from_hf.py with a small synthetic parquet file."""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from support import l2arctic_from_hf as l2hf


def wav_bytes(seconds=0.3, sr=44100):
    buf = io.BytesIO()
    sf.write(buf, 0.1 * np.sin(np.arange(int(seconds * sr)) / sr * 2 * np.pi * 300), sr, format="WAV")
    return buf.getvalue()


def write_parquet(path, rows):
    table = pa.table({"speaker": [r[0] for r in rows], "file_id": [r[1] for r in rows],
                      "transcript": [r[2] for r in rows],
                      "audio": [{"bytes": r[3], "path": f"arctic_{r[1]}.wav"} for r in rows]})
    pq.write_table(table, path)


class TestUnpack(unittest.TestCase):
    def test_writes_the_corpus_layout_with_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "src").mkdir()
            audio = wav_bytes()
            write_parquet(tmp / "src" / "a.parquet", [("TNI", "a0001", "Author of the danger trail ", audio),
                                                      ("ASI", "a0001", "Author of the danger trail", audio)])
            write_parquet(tmp / "src" / "b.parquet", [("TNI", "a0002", "Not at this particular case", audio)])
            counts = l2hf.unpack(tmp / "src", tmp / "out")
            self.assertEqual(counts, {"TNI": 2, "ASI": 1})
            wav = tmp / "out" / "TNI" / "wav" / "arctic_a0001.wav"
            self.assertEqual(wav.read_bytes(), audio)
            self.assertEqual((tmp / "out" / "TNI" / "transcript" / "arctic_a0001.txt").read_text(),
                             "Author of the danger trail\n")
            self.assertEqual(sf.info(wav).samplerate, 44100)

    def test_rejects_non_wav_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "src").mkdir()
            write_parquet(tmp / "src" / "a.parquet", [("TNI", "a0001", "x", b"ID3 not a wav")])
            with self.assertRaises(ValueError):
                l2hf.unpack(tmp / "src", tmp / "out")
            write_parquet(tmp / "src" / "a.parquet", [("TNI", "a0001", "x", wav_bytes())] * 2)
            with self.assertRaises(ValueError):
                l2hf.unpack(tmp / "src", tmp / "out")

    def test_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                l2hf.unpack(Path(tmp), Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()

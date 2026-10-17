"""Tests for support/qc_report.py on synthetic clips with known problems."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.test_select_subsets import base_config, write_tone
from support import qc_report as qc


class TestQcReport(unittest.TestCase):
    def test_flags_and_listen_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw" / "fsc"
            specs = {"ok": {}, "clipped": {"peak": 1.0}, "long_silence": {"lead": 2.5}, "noisy": {"noise": 0.05}}
            rows = []
            for i in range(20):
                kind = list(specs)[i % 4] if i < 4 else "ok"
                path = raw / f"c{i:02d}.wav"
                write_tone(path, 0.6, seed=i, **specs[kind])
                rows.append({"utt_id": f"fsc_s_{kind}_{i}", "dataset": "fsc", "speaker_id": "s", "gender": "female",
                             "l1": "", "phrase_id": "FSC01", "text": "x", "take": 1, "source_path": path.name,
                             "duration_s": 0.8, "sha256": ""})
            cfg = base_config(tmp)
            (tmp / "subsets").mkdir()
            pd.DataFrame(rows).to_csv(tmp / "subsets" / "fsc.csv", index=False)
            table, summary = qc.qc_report("fsc", cfg)
            flags = dict(zip(table["utt_id"].str.split("_").str[2], table["flags"]))
            self.assertEqual(flags["clipped"], "clipping")
            self.assertEqual(flags["long"], "silence")
            self.assertEqual(flags["noisy"], "low_snr")
            self.assertEqual(summary["pass"], 17)
            self.assertEqual(summary["replace"], 0)
            self.assertEqual(int((table["listen_check"] == "yes").sum()), 2)  # 10 % of 20
            self.assertTrue((tmp / "subsets" / "qc_fsc_summary.json").exists())

    def test_snr_estimate(self):
        sr = 16000
        rng = np.random.default_rng(0)
        clip = np.concatenate([np.zeros(sr // 2), 0.5 * np.sin(np.arange(sr) / sr * 2 * np.pi * 300), np.zeros(sr // 2)])
        quiet = clip + 0.001 * rng.standard_normal(clip.size)
        loud = clip + 0.05 * rng.standard_normal(clip.size)
        self.assertGreater(qc.estimate_snr_db(quiet, sr), qc.estimate_snr_db(loud, sr) + 20)


if __name__ == "__main__":
    unittest.main()

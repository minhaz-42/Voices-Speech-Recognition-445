"""Tests for support.dataset_stats."""

import unittest
import pandas as pd
from support.dataset_stats import compute_dataset_stats


class TestDatasetStats(unittest.TestCase):
    """Test statistics computation on synthetic subset data."""

    def setUp(self):
        # A tiny synthetic subset table of 4 clips
        self.synthetic_df = pd.DataFrame([
            {
                "utt_id": "c1",
                "speaker_id": "S01",
                "gender": "F",
                "phrase_id": "p01",
                "category": "lights",
                "duration_s": 1.5,
                "qc_flag": "ok",
            },
            {
                "utt_id": "c2",
                "speaker_id": "S01",
                "gender": "F",
                "phrase_id": "p02",
                "category": "lights",
                "duration_s": 2.5,
                "qc_flag": "ok",
            },
            {
                "utt_id": "c3",
                "speaker_id": "S02",
                "gender": "M",
                "phrase_id": "p01",
                "category": "lights",
                "duration_s": 2.0,
                "qc_flag": "ok",
            },
            {
                "utt_id": "c4",
                "speaker_id": "S02",
                "gender": "M",
                "phrase_id": "p03",
                "category": "volume",
                "duration_s": 4.0,
                "qc_flag": "clipping",
            },
        ])

    def test_compute_stats_counts_and_durations(self):
        stats = compute_dataset_stats(self.synthetic_df)

        # Check total clips and speakers
        self.assertEqual(stats["total_clips"], 4)
        self.assertEqual(stats["unique_speakers"], 2)
        self.assertEqual(stats["gender_counts"]["F"], 2)
        self.assertEqual(stats["gender_counts"]["M"], 2)

        # Check durations: [1.5, 2.0, 2.5, 4.0]
        self.assertAlmostEqual(stats["duration_min_s"], 1.5)
        self.assertAlmostEqual(stats["duration_max_s"], 4.0)
        self.assertAlmostEqual(stats["duration_median_s"], 2.25)

        # Check QC flags
        self.assertEqual(stats["qc_flags"]["clipping"], 1)
        self.assertEqual(stats["qc_flags"]["ok"], 3)


if __name__ == "__main__":
    unittest.main()

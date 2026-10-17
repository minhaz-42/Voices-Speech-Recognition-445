"""Tests for support/select_subsets.py on synthetic archives laid out like the real ones."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from support.config import load_config
from support import select_subsets as sel


def write_tone(path: Path, seconds: float, sr: int = 16000, channels: int = 1, lead: float = 0.2, peak: float = 0.3,
               noise: float = 0.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    sig = peak * np.sin(2 * np.pi * 440 * t)
    clip = np.concatenate([np.zeros(int(lead * sr)), sig, np.zeros(int(0.2 * sr))])
    clip = clip + noise * rng.standard_normal(clip.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.stack([clip] * channels, axis=1) if channels > 1 else clip, sr)


def base_config(tmp: Path) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["outputs"] = str(tmp / "outputs")
    for name in ("fsc", "l2arctic"):
        d = cfg["datasets"][name]
        d.update(raw=str(tmp / "raw" / name), subset=str(tmp / "subsets" / f"{name}.csv"),
                 phrases=str(tmp / "prompts" / f"{name}.csv"))
    return cfg


def make_fsc(tmp: Path) -> dict:
    """6 speakers (3 F, 3 M), 4 phrasings; takes vary so the ranking rules matter."""
    root = tmp / "raw" / "fsc" / sel.FSC_ROOT
    phrasings = {"Turn on the lights.": "activate", "Turn off the lights": "deactivate",
                 "Bring me my shoes": "bring", "Lights on": "activate"}
    takes = {  # speaker -> {phrasing index: number of takes}
        "spkF1": {0: 2, 1: 2, 2: 2, 3: 1}, "spkF2": {0: 2, 1: 2, 2: 1, 3: 1}, "spkF3": {0: 1, 1: 1, 2: 1, 3: 2},
        "spkM1": {0: 2, 1: 2, 2: 2, 3: 2}, "spkM2": {0: 2, 1: 3, 2: 1, 3: 1}, "spkM3": {0: 2, 1: 1, 2: 1, 3: 1},
    }
    rows, i = [], 0
    for spk, plan in takes.items():
        for k, n in plan.items():
            text = list(phrasings)[k]
            for _ in range(n):
                i += 1
                rel = f"wavs/speakers/{spk}/{i:04d}.wav"
                write_tone(root / rel, 0.6, seed=i)
                rows.append({"path": rel, "speakerId": spk, "transcription": text, "action": phrasings[text],
                             "object": "x", "location": "none"})
    # an ineligible extra take: 0.2 s of speech, shorter than 0.5 s after trimming
    i += 1
    write_tone(root / f"wavs/speakers/spkF3/{i:04d}.wav", 0.2, seed=i)
    rows.append({"path": f"wavs/speakers/spkF3/{i:04d}.wav", "speakerId": "spkF3",
                 "transcription": "Bring me my shoes", "action": "bring", "object": "x", "location": "none"})
    (root / "data").mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    third = len(frame) // 3
    for part, chunk in zip(("train", "valid", "test"), (frame.iloc[:third], frame.iloc[third:2 * third],
                                                        frame.iloc[2 * third:])):
        chunk.to_csv(root / "data" / f"{part}_data.csv")
    pd.DataFrame({"speakerId": list(takes), "First Language spoken": "English (United States)",
                  "gender": ["female"] * 3 + ["male"] * 3}).to_csv(root / "data" / "speaker_demographics.csv", index=False)
    return takes


class TestSelectFsc(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        make_fsc(self.dir)
        self.cfg = base_config(self.dir)
        self.cfg["datasets"]["fsc"].update(n_phrases=3, speakers_per_gender=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rules(self):
        subset, phrases = sel.select_fsc(self.cfg)
        # speakers with two takes: turn on 5, turn off 4, bring 2, lights on 2 -> keep the top 3, ties alphabetically
        self.assertEqual(list(phrases["text"]), ["bring me my shoes", "turn off the lights", "turn on the lights"])
        self.assertEqual(list(phrases["phrase_id"]), ["FSC01", "FSC02", "FSC03"])
        self.assertEqual(list(phrases["category"]), ["bring", "deactivate", "activate"])
        # two-take counts on kept phrases: F1 3, F2 2, F3 0 / M1 3, M2 1 (three takes count), M3 1
        self.assertEqual(sorted(subset["speaker_id"].unique()), ["spkF1", "spkF2", "spkM1", "spkM2"])
        takes = subset.groupby(["speaker_id", "phrase_id"]).size()
        self.assertEqual(takes[("spkM2", "FSC02")], 2)           # three takes -> the first two
        self.assertEqual(takes[("spkF2", "FSC01")], 1)           # missing take stays missing
        first_two = subset[(subset["speaker_id"] == "spkM2") & (subset["phrase_id"] == "FSC02")]["source_path"]
        self.assertEqual(list(first_two), sorted(first_two)[:2])
        self.assertTrue(subset["utt_id"].str.match(r"fsc_spk\w+_FSC0\d_t[12]$").all())
        self.assertTrue((self.dir / "subsets" / "fsc.csv").exists())

    def test_ineligible_clip_is_never_selected(self):
        subset, _ = sel.select_fsc(self.cfg)
        self.assertTrue((subset["duration_s"].astype(float) >= 0.5).all())
        cache = pd.read_csv(self.dir / "outputs" / "select" / "fsc_eligibility.csv")
        self.assertEqual(int((~cache["eligible"]).sum()), 1)

    def test_same_files_on_a_rerun(self):
        first = sel.select_fsc(self.cfg)[0]
        pd.testing.assert_frame_equal(first, sel.select_fsc(self.cfg)[0])


class TestSelectL2(unittest.TestCase):
    def test_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw" / "l2arctic" / "l2arctic_release_v5.0"   # archives may nest one folder
            prompts = {f"arctic_a{i:04d}": f"sentence {i}" for i in range(1, 9)}
            for n, spk in enumerate(sel.L2_SPEAKERS):
                for pid, text in prompts.items():
                    if spk == "TNI" and pid == "arctic_a0008":
                        continue                                    # one speaker lacks a prompt
                    write_tone(raw / spk / "wav" / f"{pid}.wav", 0.6, sr=44100, seed=n)
                    (raw / spk / "transcript").mkdir(parents=True, exist_ok=True)
                    changed = "a different sentence" if (spk == "NJS" and pid == "arctic_a0007") else text
                    (raw / spk / "transcript" / f"{pid}.txt").write_text(changed + "\n")
            cfg = base_config(tmp)
            cfg["datasets"]["l2arctic"]["n_phrases"] = 3
            subset, phrases = sel.select_l2arctic(cfg)
            self.assertEqual(len(phrases), 3)
            self.assertTrue(set(phrases["phrase_id"]) <= {f"arctic_a{i:04d}" for i in range(1, 7)})
            self.assertEqual(list(phrases["phrase_id"]), sorted(phrases["phrase_id"]))
            self.assertEqual(len(subset), 30)
            self.assertEqual(set(subset.loc[subset["speaker_id"] == "TNI", "l1"]), {"Hindi"})
            self.assertEqual(subset.groupby("gender")["speaker_id"].nunique().to_dict(), {"female": 5, "male": 5})
            self.assertTrue(subset["source_path"].str.startswith("l2arctic_release_v5.0/").all())


if __name__ == "__main__":
    unittest.main()

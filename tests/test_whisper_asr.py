"""Tests for support/whisper_asr.py. Offline: no Whisper model is downloaded."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from support import whisper_asr as asr
from support.config import load_config
from support.phrases import load_phrases

# The original 50-phrase list, kept as a fixture so these tests do not depend on data/prompts/.
PHRASES = Path(__file__).parent / "fixtures" / "phrases_v0.csv"


class TestErrorRates(unittest.TestCase):
    def test_edit_distance_hand_cases(self):
        self.assertEqual(asr.edit_distance("kitten", "sitting"), 3)
        self.assertEqual(asr.edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]), 2)  # 1 sub + 1 ins
        self.assertEqual(asr.edit_distance([], ["a", "b"]), 2)
        self.assertEqual(asr.edit_distance(["a"], []), 1)

    def test_wer_substitution_deletion_insertion(self):
        self.assertAlmostEqual(asr.wer("Turn on the lights.", "turn off the lights"), 1 / 4)
        self.assertAlmostEqual(asr.wer("Turn on the lights.", "turn the lights"), 1 / 4)
        self.assertAlmostEqual(asr.wer("Turn on the lights.", "please turn on the lights now"), 2 / 4)
        self.assertEqual(asr.wer("Call my mother.", "call my mother"), 0.0)

    def test_cer_ignores_spaces(self):
        # "callmymother" (12 chars) vs "callmymothers": one insertion.
        self.assertAlmostEqual(asr.cer("Call my mother.", "call my mothers"), 1 / 12)
        self.assertEqual(asr.cer("Call my mother.", "call mymother"), 0.0)

    def test_empty_reference_raises(self):
        with self.assertRaises(ValueError):
            asr.wer("...", "anything")
        with self.assertRaises(ValueError):
            asr.wer("[music]", "anything")

    def test_corpus_rates_pool_the_edits(self):
        # 1 edit over 4 words + 0 edits over 3 words = 1/7, not the mean of 1/4 and 0.
        rates = asr.corpus_error_rates(["Turn on the lights.", "Call my mother."],
                                       ["turn off the lights", "call my mother"])
        self.assertAlmostEqual(rates["wer"], 1 / 7)


class TestNormalisation(unittest.TestCase):
    def test_case_and_punctuation(self):
        self.assertEqual(asr.normalize_text("Turn off the fan."), "turn off the fan")

    def test_spelled_numbers_match_digits(self):
        self.assertEqual(asr.normalize_text("Please pay five hundred and fifty taka at the counter."),
                         asr.normalize_text("please pay 550 taka at the counter"))
        self.assertEqual(asr.normalize_text("The Padma Bridge opened in two thousand and twenty two."),
                         asr.normalize_text("The Padma Bridge opened in 2022."))

    def test_phone_numbers_compare_equal(self):
        spoken = "My phone number is zero one seven one two, three four five, six seven eight."
        self.assertEqual(asr.normalize_text(spoken), asr.normalize_text("My phone number is 01712345678."))
        self.assertEqual(asr.normalize_text(spoken), asr.normalize_text("My phone number is 01712-345678."))
        self.assertEqual(asr.wer(spoken, "My phone number is 01712 345 678."), 0.0)

    def test_times_compare_equal(self):
        self.assertEqual(asr.normalize_text("Set an alarm for six thirty."),
                         asr.normalize_text("Set an alarm for 6:30."))
        self.assertEqual(asr.normalize_text("Set an alarm for six thirty."),
                         asr.normalize_text("Set an alarm for 6.30."))
        self.assertEqual(asr.normalize_text("The temperature today is thirty two degrees."),
                         "the temperature today is 32 degrees")


class TestPhraseMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phrases = load_phrases(PHRASES)
        cls.reject = load_config()["whisper"]["reject_wer"]

    def match(self, transcript):
        return asr.match_phrase(transcript, self.phrases, self.reject)

    def test_every_phrase_matches_itself(self):
        for phrase in self.phrases:
            m = self.match(phrase["text"])
            self.assertEqual(m["phrase_id"], phrase["phrase_id"])
            self.assertEqual(m["wer"], 0.0)

    def test_confusable_pairs(self):
        self.assertEqual(self.match("turn on the lights")["phrase_id"], "CMD01")
        self.assertEqual(self.match("turn off the fan")["phrase_id"], "CMD02")
        self.assertEqual(self.match("decrease the volume")["phrase_id"], "CMD10")
        self.assertEqual(self.match("increase the volume")["phrase_id"], "CMD09")

    def test_misheard_names_still_match(self):
        m = self.match("I took a bus from Mohakali to Silet")
        self.assertEqual(m["phrase_id"], "NER01")
        self.assertAlmostEqual(m["wer"], 2 / 8)
        self.assertFalse(m["rejected"])

    def test_digits_in_transcript(self):
        self.assertEqual(self.match("Please pay 550 taka at the counter.")["phrase_id"], "NUM03")

    def test_unrelated_sentence_is_rejected(self):
        self.assertTrue(self.match("What is the weather like in London today?")["rejected"])
        self.assertTrue(self.match("hello")["rejected"])

    def test_ranking_is_sorted_and_cut(self):
        ranking = asr.rank_phrases("turn off the lights", self.phrases, top_k=5)
        self.assertEqual(len(ranking), 5)
        keys = [(r["wer"], r["cer"]) for r in ranking]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual({r["phrase_id"] for r in ranking[:2]}, {"CMD01", "CMD02"})
        self.assertEqual(len(asr.rank_phrases("turn off the lights", self.phrases, top_k=None)), 50)


class FakeTranscriber:
    """Returns fixed transcripts in order and records the audio it was given."""

    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.seen = []

    def __call__(self, audios):
        self.seen.extend(audios)
        out, self.transcripts = self.transcripts[:len(audios)], self.transcripts[len(audios):]
        return out


class TestRunClipList(unittest.TestCase):
    def test_writes_predictions_and_metrics(self):
        sr = 48000
        t = np.arange(sr) / sr
        tone = np.concatenate([np.zeros(sr // 2), 0.3 * np.sin(2 * np.pi * 300 * t), np.zeros(sr // 2)])
        transcripts = ["Turn off the fan.", "turn on the lights", "what is the weather like in London today"]
        labels = ["CMD02", "CMD02", "CMD01"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = []
            for i, label in enumerate(labels):
                path = tmp / f"S99_{label}_r{i + 1}_clean.wav"
                sf.write(path, tone, sr)
                rows.append({"utt_id": path.stem, "path": str(path), "phrase_id": label, "speaker_id": "S99"})
            pd.DataFrame(rows).to_csv(tmp / "clips.csv", index=False)

            fake = FakeTranscriber(transcripts)
            predictions, metrics = asr.run_clip_list(tmp / "clips.csv", tmp / "out", fake, load_phrases(PHRASES),
                                                     reject_wer=0.5, batch_size=2, model_name="fake")

            self.assertEqual(len(fake.seen), 3)
            self.assertTrue(all(a.ndim == 1 for a in fake.seen))  # mono 16 kHz arrays
            self.assertEqual(list(predictions.columns), asr.PREDICTION_COLUMNS)
            self.assertEqual(list(predictions["pred"]), ["CMD02", "CMD01", "EVQ09"])
            self.assertEqual(list(predictions["rejected"]), [False, False, True])
            self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
            self.assertEqual(metrics["n_rejected"], 1)
            saved = json.loads((tmp / "out" / "metrics.json").read_text())
            self.assertEqual(saved["model"], "fake")
            self.assertEqual(len(pd.read_csv(tmp / "out" / "predictions.csv")), 3)

    def test_unknown_phrase_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.csv"
            pd.DataFrame([{"utt_id": "x", "path": "x.wav", "phrase_id": "XYZ99"}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                asr.run_clip_list(path, Path(tmp) / "out", FakeTranscriber([]), load_phrases(PHRASES))


class TestRunSplits(unittest.TestCase):
    """Whisper on the dataset splits (T30) with a fake transcriber on a synthetic subset."""

    def test_scores_every_split_and_transcribes_each_clip_once(self):
        import contextlib
        import io

        from support.dataset import build_processed
        from tests._synthetic import make_config, make_subset, write_audio
        from support.make_splits import make_splits

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            subset = write_audio(make_subset("l2arctic", n_phrases=3, takes=1), tmp / "raw")  # 30 clips
            cfg = make_config(tmp, "l2arctic", subset)
            cfg["paths"].update(results=str(tmp / "results"), images=str(tmp / "images"))
            build_processed("l2arctic", cfg)
            make_splits("l2arctic", cfg)

            class Echo:
                calls = 0

                def __call__(self, clips):
                    Echo.calls += len(clips)
                    return ["Phrase number 1."] * len(clips)  # always the text of P01

            splits = ["random"] + [f"fold{k}" for k in range(1, 6)]
            with contextlib.redirect_stdout(io.StringIO()):
                summary = asr.run_splits("l2arctic", splits, Echo(), cfg, run="whisper_fake", model_name="fake")
            self.assertEqual(Echo.calls, 30)                 # folds cover every clip; random reuses them
            self.assertEqual(summary["n_transcribed"], 30)
            self.assertAlmostEqual(summary["folds"]["pooled"]["accuracy"], 1 / 3)
            self.assertTrue(summary["zero_shot"])
            root = tmp / "results" / "whisper_fake" / "l2arctic"
            per_l1 = pd.read_csv(root / "wer_per_l1.csv")
            self.assertEqual(sorted(per_l1["l1"]), ["Arabic", "Hindi", "Mandarin", "Spanish"])
            self.assertEqual(int(per_l1.loc[per_l1["l1"] == "Hindi", "n"].iloc[0]), 12)
            per_spk = pd.read_csv(root / "wer_per_speaker.csv")
            self.assertEqual(len(per_spk), 10)
            preds = pd.read_csv(root / "predictions_fold1.csv")
            self.assertIn("transcript", preds.columns)
            self.assertEqual(set(preds.loc[preds["label"] == "P01", "wer"]), {0.0})

    def test_helpers(self):
        self.assertEqual(asr.run_name_for("openai/whisper-small"), "whisper_small")
        frame = pd.DataFrame({"speaker_id": ["a", "a", "b"], "label": ["X", "Y", "X"], "pred": ["X", "X", "X"],
                              "transcript": ["turn on", "turn off", "turn"]})
        table = asr.error_rates_by(frame, {"X": "turn on", "Y": "turn off"}, "speaker_id")
        self.assertEqual(list(table["n"]), [2, 1])
        self.assertAlmostEqual(table.loc[0, "wer"], 0.0)       # speaker a: both exact
        self.assertAlmostEqual(table.loc[1, "wer"], 0.5)       # speaker b: one of two words missing
        self.assertAlmostEqual(table.loc[0, "accuracy"], 0.5)


class TestConfig(unittest.TestCase):
    def test_whisper_settings(self):
        wcfg = load_config()["whisper"]
        self.assertEqual(wcfg["model"], "openai/whisper-small")
        self.assertEqual(wcfg["language"], "en")
        self.assertGreater(wcfg["reject_wer"], 0)


if __name__ == "__main__":
    unittest.main()

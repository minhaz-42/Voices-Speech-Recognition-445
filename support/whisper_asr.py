"""Zero-shot Whisper speech-to-text, matched to the nearest of the 50 phrases.

The fourth pipeline of the project. Whisper transcribes a clip without any
training on TenVoices; the transcript is then compared with every phrase, and
the phrase with the lowest word error rate (WER) is the prediction. That gives
Whisper the same accuracy / F1 tables as the trained models, and the transcript
itself gives WER and character error rate (CER).

Text is normalised the same way on both sides with Whisper's English text
normaliser (case, punctuation, spelled-out numbers to digits). Neighbouring digit
groups are then joined and leading zeros dropped, so "six thirty", "6:30" and "6.30"
all become 630, and a phone number compares equal whether spoken or written as digits.

Usage::

    python -m support.whisper_asr clip.wav [more.wav ...]          # transcript + top-5 phrases
    python -m support.whisper_asr --list clips.csv --out results/whisper_small/

``clips.csv`` has the columns ``utt_id, path, phrase_id`` and optionally
``speaker_id``; paths are relative to the repository root.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from support.audio import load_audio
from support.config import PROJECT_ROOT, load_config
from support.metrics import compute_metrics
from support.phrases import Phrase, load_phrases

Transcriber = Callable[[list[np.ndarray]], list[str]]

PREDICTION_COLUMNS = ["utt_id", "speaker_id", "label", "pred", "transcript", "wer", "cer", "match_wer", "rejected"]


# ---------------------------------------------------------------- text and error rates

@lru_cache(maxsize=1)
def _normalizer():
    """Whisper's English text normaliser (no British-to-American spelling map; our phrases need none)."""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer({})


_DIGIT_RUN = re.compile(r"\b\d+(?: \d+)+\b")
_CLOCK_DOT = re.compile(r"\b(\d{1,2})\.(\d{2})\b")


def normalize_text(text: str) -> str:
    """Normalise a transcript or phrase for comparison: lower case, no punctuation, numbers as digits."""
    text = _CLOCK_DOT.sub(r"\1:\2", text)  # British "6.30" is a time like "6:30"; no phrase has decimals
    text = " ".join(t for t in _normalizer()(text).split() if any(ch.isalnum() for ch in t))
    text = _DIGIT_RUN.sub(lambda m: m.group(0).replace(" ", ""), text)
    # The normaliser keeps a leading zero for "zero one seven ..." but drops it from "01712...".
    return " ".join(t.lstrip("0") or "0" if t.isdigit() else t for t in text.split())


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance (substitutions, deletions, insertions) between two token sequences."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def _words(text: str) -> list[str]:
    return normalize_text(text).split()


def _chars(text: str) -> list[str]:
    return list(normalize_text(text).replace(" ", ""))


def _rate(ref_tokens: list, hyp_tokens: list) -> float:
    if not ref_tokens:
        raise ValueError("the reference is empty after normalisation")
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate of ``hypothesis`` against ``reference`` (both normalised). Can exceed 1."""
    return _rate(_words(reference), _words(hypothesis))


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate (spaces ignored) of ``hypothesis`` against ``reference``."""
    return _rate(_chars(reference), _chars(hypothesis))


def corpus_error_rates(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    """Corpus-level WER and CER: total edits divided by total reference length."""
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty and equally long")
    edits_w = edits_c = len_w = len_c = 0
    for ref, hyp in zip(references, hypotheses):
        rw, hw, rc, hc = _words(ref), _words(hyp), _chars(ref), _chars(hyp)
        edits_w += edit_distance(rw, hw)
        edits_c += edit_distance(rc, hc)
        len_w += len(rw)
        len_c += len(rc)
    return {"wer": edits_w / len_w, "cer": edits_c / len_c}


# ---------------------------------------------------------------- phrase matching

def rank_phrases(transcript: str, phrases: Sequence[Phrase], top_k: int | None = 5) -> list[dict[str, Any]]:
    """Score every phrase against a transcript, best first.

    Each phrase is the reference for its own WER (ties broken by CER), so the
    score says how far the transcript is from that phrase.

    Returns:
        Up to ``top_k`` dicts ``{"phrase_id", "text", "wer", "cer", "similarity"}``,
        where ``similarity = max(0, 1 - wer)`` is a simple confidence for display.
    """
    hyp_w, hyp_c = _words(transcript), _chars(transcript)
    scored = []
    for phrase in phrases:
        ref_w, ref_c = _words(phrase["text"]), _chars(phrase["text"])
        w = edit_distance(ref_w, hyp_w) / len(ref_w)
        c = edit_distance(ref_c, hyp_c) / len(ref_c)
        scored.append({"phrase_id": phrase["phrase_id"], "text": phrase["text"],
                       "wer": w, "cer": c, "similarity": max(0.0, 1.0 - w)})
    scored.sort(key=lambda s: (s["wer"], s["cer"]))
    return scored if top_k is None else scored[:top_k]


def match_phrase(transcript: str, phrases: Sequence[Phrase], reject_wer: float) -> dict[str, Any]:
    """The nearest phrase to a transcript.

    Returns:
        The best entry of :func:`rank_phrases` plus ``rejected``: True when even the
        best phrase is more than ``reject_wer`` away. Rejection is only shown in the
        demo ("not one of the 50 phrases"); accuracy always uses the best phrase.
    """
    best = rank_phrases(transcript, phrases, top_k=1)[0]
    return {**best, "rejected": best["wer"] > reject_wer}


# ---------------------------------------------------------------- Whisper model

def pick_device(device: str | None = None) -> str:
    """``device`` if given, otherwise CUDA, then Apple MPS, then CPU."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WhisperTranscriber:
    """Batch English transcription with a Hugging Face Whisper checkpoint.

    Call it with a list of mono 16 kHz float arrays; it returns one transcript per array.
    """

    sample_rate = 16000

    def __init__(self, model_name: str = "openai/whisper-small", device: str | None = None,
                 batch_size: int = 8, language: str = "en"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.language = language
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=self.dtype)
        self.model.to(self.device).eval()

    def __call__(self, audios: list[np.ndarray]) -> list[str]:
        import torch

        transcripts: list[str] = []
        for start in range(0, len(audios), self.batch_size):
            batch = [np.asarray(a, dtype=np.float32) for a in audios[start:start + self.batch_size]]
            inputs = self.processor(batch, sampling_rate=self.sample_rate, return_tensors="pt",
                                    return_attention_mask=True)
            with torch.no_grad():
                ids = self.model.generate(inputs.input_features.to(self.device, self.dtype),
                                          attention_mask=inputs.attention_mask.to(self.device),
                                          language=self.language, task="transcribe")
            transcripts.extend(t.strip() for t in self.processor.batch_decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        return transcripts


# ---------------------------------------------------------------- batch runs

def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def run_clip_list(
    list_csv: str | Path,
    out_dir: str | Path,
    transcriber: Transcriber,
    phrases: Sequence[Phrase] | None = None,
    reject_wer: float = 0.5,
    batch_size: int = 8,
    model_name: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transcribe every clip of a list, match it to a phrase and save the results.

    Args:
        list_csv: CSV with ``utt_id, path, phrase_id`` and optionally ``speaker_id``.
        out_dir: Folder for ``predictions.csv`` and ``metrics.json``.
        transcriber: Callable mapping a list of 16 kHz arrays to transcripts.
        phrases: The phrase catalogue (default: ``load_phrases()``).
        reject_wer: Threshold for the ``rejected`` flag (see :func:`match_phrase`).
        batch_size: Clips loaded and transcribed at a time.
        model_name: Stored in ``metrics.json``.

    Returns:
        The predictions table and the metrics dictionary.
    """
    phrases = list(phrases) if phrases is not None else load_phrases()
    texts = {p["phrase_id"]: p["text"] for p in phrases}
    clips = pd.read_csv(list_csv, dtype=str).fillna("")
    missing = {"utt_id", "path", "phrase_id"} - set(clips.columns)
    if missing:
        raise ValueError(f"{list_csv} is missing columns: {sorted(missing)}")
    unknown = sorted(set(clips["phrase_id"]) - set(texts))
    if unknown:
        raise ValueError(f"unknown phrase IDs in {list_csv}: {unknown}")
    if "speaker_id" not in clips.columns:
        clips["speaker_id"] = ""

    audio_cfg = load_config()["audio"]
    rows = []
    for start in range(0, len(clips), batch_size):
        chunk = clips.iloc[start:start + batch_size]
        audios = [load_audio(_resolve(p), audio_cfg, pad=False) for p in chunk["path"]]
        for clip, transcript in zip(chunk.itertuples(index=False), transcriber(audios)):
            label_text = texts[clip.phrase_id]
            match = match_phrase(transcript, phrases, reject_wer)
            rows.append({
                "utt_id": clip.utt_id, "speaker_id": clip.speaker_id, "label": clip.phrase_id,
                "pred": match["phrase_id"], "transcript": transcript,
                "wer": wer(label_text, transcript), "cer": cer(label_text, transcript),
                "match_wer": match["wer"], "rejected": match["rejected"],
            })

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    metrics = compute_metrics(predictions["label"], predictions["pred"])
    metrics.update(corpus_error_rates([texts[l] for l in predictions["label"]], predictions["transcript"]))
    metrics.update({"n_rejected": int(predictions["rejected"].sum()), "model": model_name,
                    "reject_wer": reject_wer})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return predictions, metrics


# ---------------------------------------------------------------- CLI

def main() -> None:
    """Transcribe files and show the top phrases, or run a clip list and save the results."""
    parser = argparse.ArgumentParser(description="Zero-shot Whisper speech-to-text matched to the 50 phrases.")
    parser.add_argument("audio", nargs="*", help="audio files to transcribe")
    parser.add_argument("--list", dest="list_csv", help="CSV of clips (utt_id, path, phrase_id[, speaker_id])")
    parser.add_argument("--out", help="output folder for --list, e.g. results/whisper_small/")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    parser.add_argument("--model", default=None, help="Hugging Face checkpoint (default: config whisper.model)")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    if not args.audio and not args.list_csv:
        parser.error("give audio files or --list")
    if args.list_csv and not args.out:
        parser.error("--list needs --out")

    cfg = load_config(args.config)
    wcfg = cfg["whisper"]
    model_name = args.model or wcfg["model"]
    batch_size = args.batch_size or wcfg["batch_size"]
    transcriber = WhisperTranscriber(model_name, args.device, batch_size, wcfg["language"])
    phrases = load_phrases()

    if args.list_csv:
        _, metrics = run_clip_list(args.list_csv, args.out, transcriber, phrases,
                                   wcfg["reject_wer"], batch_size, model_name)
        print(json.dumps(metrics, indent=2))
        print(f"saved: {args.out}")
        return

    transcripts = transcriber([load_audio(p, cfg["audio"], pad=False) for p in args.audio])
    for path, transcript in zip(args.audio, transcripts):
        print(f"\n{path}\n  transcript: {transcript!r}")
        ranking = rank_phrases(transcript, phrases, args.top_k or wcfg["top_k"])
        if ranking[0]["wer"] > wcfg["reject_wer"]:
            print("  -> not one of the 50 phrases")
        for r in ranking:
            print(f"  {r['phrase_id']}  WER {r['wer']:.2f}  {r['text']}")


if __name__ == "__main__":
    main()

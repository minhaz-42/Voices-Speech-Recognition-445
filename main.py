"""TenVoices inference: which phrase was spoken? Prints the top-5 phrases with their scores.

Usage::

    python main.py --audio clip.wav --model outputs/models/w2v2_15ep_noise/fsc/random
    python main.py --video clip.mp4 --model outputs/models/svm/fsc/random.joblib
    python main.py --mic 4 --model outputs/models/w2v2_15ep_noise/fsc/random     # record 4 s
    python main.py --audio clip.wav --asr whisper                                # transcript + nearest phrases
    python main.py --audio clip.wav --model <model> --benchmark 50               # inference time per clip

``--dataset`` (default ``fsc``) selects the phrase list used to show the phrase texts
(and, with ``--asr whisper``, the phrases to match). Video and microphone input need ffmpeg.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from support.config import load_config


def phrase_texts(dataset: str, cfg: dict) -> dict[str, str]:
    """Phrase ID → text for display; empty when the phrase list does not exist yet."""
    from support.dataset import load_labels

    try:
        labels = load_labels(dataset, cfg)
    except FileNotFoundError:
        return {}
    return dict(zip(labels["phrase_id"], labels["text"]))


def model_size_mb(path: Path) -> float:
    """Size of a model file or folder on disk."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def show(result: list[dict], header: str) -> None:
    """Print a top-k list with a small bar per score."""
    print(header)
    for rank, r in enumerate(result, start=1):
        bar = "█" * int(round(20 * r["score"]))
        print(f"  {rank}. {r['phrase_id']:<14} {r['score']:6.3f} {bar:<20} {r['text']}")


def benchmark(recognizer, samples: np.ndarray, sample_rate: int, repeats: int) -> dict[str, float]:
    """Milliseconds per clip for pre-processing + prediction (after one warm-up run)."""
    recognizer.predict(samples, sample_rate)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        recognizer.predict(samples, sample_rate)
        times.append(1000 * (time.perf_counter() - start))
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times), "repeats": repeats}


def main() -> None:
    """Parse the options, read the input and print the recognised phrases."""
    # The score bars use "█", which a Windows console (cp1252) cannot encode: print UTF-8 there.
    if hasattr(sys.stdout, "reconfigure") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Recognise which TenVoices phrase was spoken.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="audio file (WAV, FLAC, OGG, MP3)")
    source.add_argument("--video", help="video file; its audio track is used (needs ffmpeg)")
    source.add_argument("--mic", type=float, metavar="SECONDS", help="record from the microphone (needs ffmpeg)")
    parser.add_argument("--model", help="saved model: a .joblib file or a wav2vec 2.0 folder in outputs/models/")
    parser.add_argument("--asr", choices=["whisper"], help="zero-shot Whisper: transcript + nearest phrases")
    parser.add_argument("--dataset", default="fsc", help="phrase list to show / match (default fsc)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--benchmark", type=int, metavar="N", help="time N predictions and report ms per clip")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    if not args.model and not args.asr:
        parser.error("give --model or --asr whisper")

    from support.inference import WhisperRecognizer, load_recognizer, read_input, record_microphone

    cfg = load_config(args.config)
    if args.mic:
        samples, sr = record_microphone(args.mic), 16000
    else:
        samples, sr = read_input(args.audio or args.video)
    texts = phrase_texts(args.dataset, cfg)

    if args.asr == "whisper":
        phrases = [{"phrase_id": k, "text": v} for k, v in texts.items()]
        if not phrases:
            parser.error(f"the phrase list of '{args.dataset}' does not exist yet (task T42)")
        recognizer = WhisperRecognizer(phrases, cfg=cfg)
        result = recognizer.predict(samples, sr, args.top_k)
        print(f"transcript: {result['transcript']!r}")
        if result["rejected"]:
            print("-> not one of the phrases")
        show(result["top"], "nearest phrases (score = 1 - word error rate):")
    else:
        recognizer = load_recognizer(args.model, args.device)
        try:
            show(recognizer.predict(samples, sr, args.top_k, texts), f"top {args.top_k} phrases ({args.model}):")
        except ValueError as err:  # e.g. longer than 6 s after trimming
            raise SystemExit(f"cannot recognise this clip: {err}")

    if args.benchmark:
        b = benchmark(recognizer, samples, sr, args.benchmark)
        device = getattr(recognizer, "device", "cpu")
        size = model_size_mb(Path(args.model)) if args.model else float("nan")
        print(f"inference: {b['mean_ms']:.1f} ms per clip (median {b['median_ms']:.1f}, {b['repeats']} runs) "
              f"on {device}; model size {size:.1f} MB")


if __name__ == "__main__":
    main()

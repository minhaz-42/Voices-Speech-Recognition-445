"""TenVoices live demo (Gradio): speak or upload a phrase and see what the system hears.

The clip is pre-processed exactly like the training data and shown as audio and as a
four-panel figure (waveform, spectrogram, log-mel, MFCC) (T36). A saved model, by default
the SVM of task T14, recognises it and shows the top-5 phrases with their confidence (T23).

Usage::

    python app.py [--model outputs/models/svm/fsc/random.joblib] [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from support.audio import peak_dbfs, preprocess
from support.config import load_config
from support.dataset import load_labels
from support.visualization import clip_overview, plot_top_k

DATASET = "fsc"  # the demo recognises the 50 spoken commands of TenVoices-FSC
NOT_CONNECTED = "No model loaded: start the app with --model <saved model>."
DEFAULT_MODEL = "outputs/models/svm/fsc/random.joblib"  # your SVM from T14 (python -m support.classical --model svm)
TOP_K = 5
EMPTY_MESSAGE = "No audio yet: record a phrase with the microphone or upload a clip, then press Analyse."


def process_audio(audio: tuple[int, np.ndarray] | None, audio_cfg: dict[str, Any] | None = None):
    """Pre-process one clip from the audio widget and describe it.

    Args:
        audio: What ``gr.Audio(type="numpy")`` returns: ``(sample_rate, samples)`` with
            int16 or float samples, mono or ``(frames, channels)``; or ``None``.
        audio_cfg: The ``audio`` section of the config. Defaults to ``configs/config.yaml``.

    Returns:
        ``(processed, figure, info)``: the processed clip as ``(16000, samples)`` for
        playback, the :func:`clip_overview` figure and a short text. For missing, silent
        or too long audio, ``processed`` and ``figure`` are ``None`` and ``info`` says why.
    """
    if audio is None:
        return None, None, EMPTY_MESSAGE
    sample_rate, samples = audio
    samples = np.asarray(samples)
    if samples.size == 0:
        return None, None, EMPTY_MESSAGE

    cfg = audio_cfg if audio_cfg is not None else load_config()["audio"]
    target_sr = int(cfg["sample_rate"])
    try:
        processed = preprocess(samples, int(sample_rate), cfg, pad=False)
    except ValueError as err:
        return None, None, f"Cannot use this clip: {err}"
    if processed.size == 0 or not np.any(processed):
        return None, None, "The clip is silent: please record or upload a spoken phrase."

    max_s = float(cfg["max_duration_s"])
    if processed.size / target_sr > max_s:
        return None, None, (f"The clip is {processed.size / target_sr:.2f} s long after trimming the silence; "
                            f"phrases must fit in {max_s:g} s, so it is not cut. Please record a shorter clip.")

    raw_seconds = samples.shape[0] / float(sample_rate)
    figure = clip_overview(processed, target_sr, title="What the model receives")
    info = (f"Input: {int(sample_rate) / 1000:g} kHz, {raw_seconds:.2f} s\n"
            f"After pre-processing: {target_sr / 1000:g} kHz mono, {processed.size / target_sr:.2f} s of speech, "
            f"peak {peak_dbfs(processed):.1f} dBFS")
    return (target_sr, processed), figure, info


def phrase_texts(dataset: str = DATASET) -> dict[str, str]:
    """``{phrase_id: text}`` of the dataset's phrases, to label the predictions."""
    labels = load_labels(dataset)
    return dict(zip(labels["phrase_id"].astype(str), labels["text"].astype(str)))


def recognise(audio: tuple[int, np.ndarray] | None, recognizer, texts: dict[str, str] | None = None,
              k: int = TOP_K):
    """The top-k phrases for one clip from the audio widget.

    Args:
        audio: ``(sample_rate, samples)`` from ``gr.Audio(type="numpy")``, or ``None``.
        recognizer: A ``support.inference`` recogniser (``predict(samples, rate, k, texts)``).
        texts: ``{phrase_id: text}`` to show with each phrase ID.
        k: Number of phrases.

    Returns:
        ``(rows, figure)``: ``[[rank, phrase_id, text, "confidence %"], ...]`` and the
        :func:`plot_top_k` chart, or ``([], None)`` for missing audio or a clip the
        recogniser cannot use.
    """
    if audio is None or np.asarray(audio[1]).size == 0:
        return [], None
    sample_rate, samples = audio
    try:
        top = recognizer.predict(np.asarray(samples), int(sample_rate), k=k, texts=texts)
    except ValueError:
        return [], None
    rows = [[i + 1, r["phrase_id"], r["text"], f"{r['score']:.1%}"] for i, r in enumerate(top)]
    return rows, plot_top_k(top, title="What the model heard")


def phrase_table(dataset: str = DATASET) -> list[list[str]]:
    """The dataset's phrases as ``[phrase_id, text]`` rows for the phrase table."""
    labels = load_labels(dataset)
    return labels[["phrase_id", "text"]].astype(str).values.tolist()


def whisper_text(audio: tuple[int, np.ndarray] | None, whisper) -> str:
    """Whisper's transcript of one clip and the nearest of the 50 phrases, as one line.

    Args:
        audio: ``(sample_rate, samples)`` from the audio widget, or ``None``.
        whisper: A ``support.inference.WhisperRecognizer``.
    """
    if audio is None or np.asarray(audio[1]).size == 0:
        return ""
    try:
        out = whisper.predict(np.asarray(audio[1]), int(audio[0]), k=1)
    except ValueError as err:
        return f"Cannot use this clip: {err}"
    best = out["top"][0]
    verdict = "not one of the 50 phrases" if out["rejected"] else f"nearest phrase {best['phrase_id']}: {best['text']}"
    return f'"{out["transcript"].strip()}" -> {verdict}'


def load_model(path: str | Path | None):
    """Open a saved model with ``support.inference.load_recognizer``.

    Returns:
        ``(recognizer, status)``: the recogniser (``None`` when ``path`` is missing or
        cannot be opened) and a one-line status for the app.
    """
    if not path:
        return None, NOT_CONNECTED
    if not Path(path).exists():
        return None, (f"No model at {path}: train one with python -m support.classical --model svm "
                      f"--dataset {DATASET}, or start the app with --model <saved model>.")
    from support.inference import load_recognizer

    try:
        return load_recognizer(path), f"Model: {path}"
    except (ValueError, OSError) as err:
        return None, f"Cannot open {path}: {err}"


def build_app(model_path: str | Path | None = DEFAULT_MODEL, recognizer=None):
    """The Gradio interface (built, not launched).

    Args:
        model_path: Saved model to recognise with (a ``.joblib`` file or a wav2vec 2.0 folder).
        recognizer: An already loaded recogniser; overrides ``model_path`` (used by the tests).

    Returns:
        A ``gr.Blocks`` app: audio input and an Analyse button; the processed audio,
        the overview figure and an info text; the top-5 phrases with a confidence chart;
        the table of the 50 phrases.
    """
    import gradio as gr

    if recognizer is not None:
        status = f"Model: {getattr(recognizer, 'path', 'loaded')}"
    else:
        recognizer, status = load_model(model_path)
    texts = phrase_texts()
    whisper_cache: dict = {}

    def get_whisper():
        """Load Whisper the first time the checkbox is used (it is a large download)."""
        if "model" not in whisper_cache:
            from support.inference import WhisperRecognizer

            phrases = [{"phrase_id": pid, "text": text} for pid, text in texts.items()]
            whisper_cache["model"] = WhisperRecognizer(phrases)
        return whisper_cache["model"]

    def analyse(audio, use_whisper):
        processed, overview, info = process_audio(audio)
        rows, chart = recognise(audio, recognizer, texts) if recognizer is not None and processed else ([], None)
        transcript = whisper_text(audio, get_whisper()) if use_whisper and processed else ""
        return processed, overview, info, rows, chart, transcript

    with gr.Blocks(title="TenVoices demo") as app:
        gr.Markdown("# TenVoices: spoken phrase recognition\n"
                    "Say one of the 50 smart-home commands below (or upload a clip). The app shows what the "
                    "system receives after pre-processing (16 kHz mono, silence trimmed, peak-normalised) "
                    "and the five phrases the model finds most likely.")
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your phrase")
                use_whisper = gr.Checkbox(label="Also transcribe with Whisper (zero-shot; downloads the model "
                                                "the first time)", value=False)
                analyse_btn = gr.Button("Analyse", variant="primary")
                info = gr.Textbox(label="Clip", lines=2, interactive=False)
                processed = gr.Audio(label="After pre-processing", type="numpy", interactive=False)
            with gr.Column():
                gr.Markdown(f"**Prediction** · {status}")
                top_table = gr.Dataframe(headers=["Rank", "ID", "Phrase", "Confidence"], interactive=False)
                top_plot = gr.Plot(label="Confidence of the top 5 phrases")
                transcript = gr.Textbox(label="Whisper transcript", interactive=False)
        plot = gr.Plot(label="Waveform, spectrogram, log-mel spectrogram and MFCCs")
        with gr.Accordion("The 50 phrases (TenVoices-FSC)", open=False):
            gr.Dataframe(value=phrase_table(), headers=["ID", "Phrase"], interactive=False)
        analyse_btn.click(analyse, inputs=[audio_in, use_whisper],
                          outputs=[processed, plot, info, top_table, top_plot, transcript])
    return app


def main() -> None:
    """Launch the demo in the browser."""
    parser = argparse.ArgumentParser(description="TenVoices live demo (Gradio).")
    parser.add_argument("--port", type=int, default=7860, help="local port (default: 7860)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"saved model: .joblib or wav2vec 2.0 folder (default: {DEFAULT_MODEL})")
    parser.add_argument("--share", action="store_true", help="also create a temporary public Gradio link")
    args = parser.parse_args()
    build_app(args.model).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

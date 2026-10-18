"""TenVoices live demo (Gradio): speak or upload a phrase and see what the system hears.

Task T36 builds the skeleton: the clip is pre-processed exactly like the training data
and shown as audio and as a four-panel figure (waveform, spectrogram, log-mel, MFCC).
The trained model is connected in task T23; until then the app shows no predictions.

Usage::

    python app.py [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from support.audio import peak_dbfs, preprocess
from support.config import load_config
from support.dataset import load_labels
from support.visualization import clip_overview, plot_top_k

DATASET = "fsc"  # the demo recognises the 50 spoken commands of TenVoices-FSC
NOT_CONNECTED = "Model not connected yet: predictions arrive with the trained model (task T23)."
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


def build_app():
    """The Gradio interface (built, not launched).

    Returns:
        A ``gr.Blocks`` app: audio input and an Analyse button; the processed audio,
        the overview figure and an info text; a prediction box (not connected yet) and
        the table of the 50 phrases.
    """
    import gradio as gr

    with gr.Blocks(title="TenVoices demo") as app:
        gr.Markdown("# TenVoices: spoken phrase recognition\n"
                    "Say one of the 50 smart-home commands below (or upload a clip) and see exactly what the "
                    "system receives after pre-processing: 16 kHz mono, silence trimmed, peak-normalised.")
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your phrase")
                analyse = gr.Button("Analyse", variant="primary")
                info = gr.Textbox(label="Clip", lines=2, interactive=False)
                processed = gr.Audio(label="After pre-processing", type="numpy", interactive=False)
                gr.Textbox(value=NOT_CONNECTED, label="Prediction", interactive=False)
            with gr.Column():
                plot = gr.Plot(label="Waveform, spectrogram, log-mel spectrogram and MFCCs")
        with gr.Accordion("The 50 phrases (TenVoices-FSC)", open=False):
            gr.Dataframe(value=phrase_table(), headers=["ID", "Phrase"], interactive=False)
        analyse.click(process_audio, inputs=audio_in, outputs=[processed, plot, info])
    return app


def main() -> None:
    """Launch the demo in the browser."""
    parser = argparse.ArgumentParser(description="TenVoices live demo (Gradio).")
    parser.add_argument("--port", type=int, default=7860, help="local port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="also create a temporary public Gradio link")
    args = parser.parse_args()
    build_app().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

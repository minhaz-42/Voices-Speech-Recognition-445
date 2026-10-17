"""Pick the PyTorch device: CUDA (Colab GPU), then Apple Silicon (MPS), then CPU."""

from __future__ import annotations


def pick_device(device: str | None = None) -> str:
    """``device`` if given, otherwise ``cuda``, ``mps`` or ``cpu``, whichever is available first."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

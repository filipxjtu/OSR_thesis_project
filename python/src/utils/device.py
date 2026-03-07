from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    """
    Resolves the requested device string into an available torch.device.

    Priority logic:
        - If 'cuda' requested but unavailable → fallback to cpu
        - If 'mps' requested but unavailable → fallback to cpu
        - If 'auto' requested → choose best available
    """

    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")
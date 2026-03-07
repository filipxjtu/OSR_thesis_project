from __future__ import annotations

import numpy as np


def compute_stft(
    x: np.ndarray,
    fs: int,
    win_length: int = 256,
    hop_length: int = 128,
    n_fft: int = 256,
) -> np.ndarray:
    """
    Deterministic STFT aligned with dataset_spec_v1.md
    No padding
    No centering
    Hann window
    One-sided spectrum
    log(1 + |STFT|)
    """

    if x.ndim != 1:
        raise ValueError("STFT expects 1D signal.")

    window = np.hanning(win_length)

    frames = []
    for start in range(0, len(x) - win_length + 1, hop_length):
        segment = x[start:start + win_length]
        segment = segment * window
        spectrum = np.fft.rfft(segment, n=n_fft)
        magnitude = np.abs(spectrum)
        frames.append(magnitude)

    S = np.stack(frames, axis=1)  # (freq_bins, time_frames)

    S = np.log1p(S)

    mean_val = np.mean(S)
    if mean_val > 0:
        S = S / mean_val
        #S = (S - np.mean(S)) / (np.std(S) + 1e-8)

    return S
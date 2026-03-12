from __future__ import annotations

import numpy as np


def compute_stft(x: np.ndarray, win_length: int = 128, hop_length: int = 64, n_fft: int = 128) -> np.ndarray:

    """ Deterministic STFT with No padding and No centering """

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

    s = np.stack(frames, axis=1)  # (freq_bins, time_frames)
    s = np.log1p(s)

    mean_val = np.mean(s)
    if mean_val > 0:
        s = s / mean_val
        #S = (S - np.mean(S)) / (np.std(S) + 1e-8)
    return s
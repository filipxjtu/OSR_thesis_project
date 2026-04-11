from __future__ import annotations

import numpy as np

def compute_stft(x: np.ndarray, win_length: int = 128, hop_length: int = 32, n_fft: int = 256) -> np.ndarray:

    if x.ndim != 1:
        raise ValueError("STFT expects 1D signal.")

    window = np.hanning(win_length)

    frames = []
    for start in range(0, len(x) - win_length + 1, hop_length):
        segment = x[start:start + win_length]
        segment = segment * window

        # full FFT
        spectrum = np.fft.fft(segment, n=n_fft)
        spectrum = np.fft.fftshift(spectrum)

        mag = np.abs(spectrum)
        log_mag = np.log1p(mag)

        frames.append(np.expand_dims(log_mag, axis=0))   #(1,F) per frame

    if len(frames) == 0:
        raise ValueError("BadSTFT: zero frames. Check signal length vs win_length.")

    s = np.stack(frames, axis=2)  # (c=1, F, T)

    return s
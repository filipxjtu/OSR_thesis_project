from __future__ import annotations

import numpy as np

def compute_stft(x: np.ndarray,
                 win_length: int = 128,
                 hop_length: int = 32,
                 n_fft: int = 1024
                 ) -> np.ndarray:

    if x.ndim != 1:
        raise ValueError("STFT expects 1D signal.")
    if n_fft < win_length:
        raise ValueError(f"n_fft ({n_fft}) must be >= win_length ({win_length}).")

    window = np.hanning(win_length)

    frames = []
    for start in range(0, len(x) - win_length + 1, hop_length):
        segment = x[start:start + win_length]
        segment = segment * window

        spectrum = np.fft.fft(segment, n=n_fft)
        spectrum = np.fft.fftshift(spectrum)

        frames.append(spectrum)  # (1, F) per frame

    if len(frames) == 0:
        raise ValueError("BadSTFT: zero frames. Check signal length vs win_length.")

    s = np.stack(frames, axis=1)  # (F, T) complex

    # Channel 0: log-magnitude
    log_mag = np.log1p(np.abs(s))  # (F, T)

    # Channel 1: phase difference, wrapped to [-pi, pi]
    phase = np.angle(s)  # (F, T)
    d_phi = np.diff(phase, axis=1, prepend=phase[:, :1])
    d_phi = np.mod(d_phi + np.pi, 2 * np.pi) - np.pi  # wrap

    s = np.stack([log_mag, d_phi], axis=0)  # (c=2, F, T)
    return s


def compute_if(x: np.ndarray) -> np.ndarray:
    """Computes the instantaneous frequency via the derivative of unwrapped phase."""
    if x.ndim != 1:
        raise ValueError("IF expects 1D signal.")

    phase = np.unwrap(np.angle(x))
    inst_freq = np.gradient(phase)

    return inst_freq
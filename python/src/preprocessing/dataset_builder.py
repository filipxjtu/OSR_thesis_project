from __future__ import annotations

import numpy as np
import torch

from .stft import compute_stft
from ..dataio.dataset_artifact import DatasetArtifact


def build_feature_tensor(
        artifact: DatasetArtifact,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert dataset to a model-ready dual tensors.
    STFT Output:  X: (Ns, 1, F, T),
    IQ output: x_iq = (Ns, 2, N) ,
    Labels:  y: (Ns,).
    """

    # stft parameters
    win_length = 128
    hop_length = 32
    N_fft = 256

    x_raw = artifact.X  #(N, Ns)
    y = artifact.y.reshape(-1)

    N = x_raw.shape[0]
    Ns = x_raw.shape[1]

    # raw IQ tensor (Ns, 2, N)
    x_iq = np.empty((Ns, 2, N), dtype=np.float32)
    x_iq[:, 0, :] = np.real(x_raw).T
    x_iq[:, 1, :] = np.imag(x_raw).T

    # STFT tensor (Ns, 1, F, T)
    features = []
    for i in range(Ns):
        signal = x_raw[:, i]
        s = compute_stft(signal, win_length, hop_length, N_fft)

        if not np.isfinite(s).all():
            raise ValueError("DatasetBuilder: STFT contains NaN or Inf")

        features.append(s)

    x_stft = np.stack(features, axis=0)  # (Ns, 1, F, T)
    assert x_stft.shape[1] == 1, "DatasetBuilder: Expected 1-channel STFT (log_mag)"

    # convert to PyTorch tensors
    x_stft_tensor = torch.from_numpy(np.ascontiguousarray(x_stft.astype(np.float32)))
    x_iq_tensor = torch.from_numpy(np.ascontiguousarray(x_iq.astype(np.float32)))
    y_tensor = torch.from_numpy(y.astype(np.int64))

    return x_stft_tensor, x_iq_tensor, y_tensor
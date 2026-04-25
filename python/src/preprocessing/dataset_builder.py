from __future__ import annotations

import numpy as np
import torch

from .stft import compute_stft, compute_if
from ..dataio.dataset_artifact import DatasetArtifact


def build_feature_tensor(
        artifact: DatasetArtifact,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert dataset to model-ready tri-tensors.
    STFT Output:  x_stft: (Ns, 1, F, T)
    IQ Output:    x_iq: (Ns, 2, N)
    IF Output:    x_if: (Ns, 1, N)
    Labels:       y: (Ns,)
    """

    # stft parameters
    win_length = 128
    hop_length = 32
    N_fft = 1024

    x_raw = artifact.X  #(N, Ns)
    y = artifact.y.reshape(-1)

    N = x_raw.shape[0]
    Ns = x_raw.shape[1]

    x_iq = np.empty((Ns, 3, N), dtype=np.float32)
    x_iq[:, 0, :] = np.real(x_raw).T
    x_iq[:, 1, :] = np.imag(x_raw).T
    x_iq[:, 2, :] = np.abs(x_raw).T

    x_if = np.empty((Ns, 1, N), dtype=np.float32)
    features = []

    for i in range(Ns):
        signal = x_raw[:, i]

        # STFT computation
        s = compute_stft(signal, win_length, hop_length, N_fft)
        if not np.isfinite(s).all():
            raise ValueError("DatasetBuilder: STFT contains NaN or Inf")
        features.append(s)

        # IF computation
        x_if[i, 0, :] = compute_if(signal)

    x_stft = np.stack(features, axis=0)
    assert x_stft.shape[1] == 1, "DatasetBuilder: Expected 1-channel STFT (log_mag)"

    x_stft_tensor = torch.from_numpy(np.ascontiguousarray(x_stft.astype(np.float32)))
    x_iq_tensor = torch.from_numpy(np.ascontiguousarray(x_iq.astype(np.float32)))
    x_if_tensor = torch.from_numpy(np.ascontiguousarray(x_if.astype(np.float32)))
    y_tensor = torch.from_numpy(y.astype(np.int64))

    return x_stft_tensor, x_iq_tensor, x_if_tensor, y_tensor
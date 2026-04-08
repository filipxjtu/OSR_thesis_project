from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset

class FeatureTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
        X_stft: (N, 1, F, T),    X_iq: (N, 2, N_samples),    y: (N,)
        Returns (x_stft_i, x_iq_i, y_i)
    """

    def __init__(self, x_stft: torch.Tensor, x_iq: torch.Tensor, y: torch.Tensor) -> None:
        if not isinstance(x_stft, torch.Tensor) or not isinstance(x_iq, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("DatasetBuilder: inputs must be torch.Tensors")

        if x_stft.ndim != 4:
            raise ValueError(f"DatasetBuilder: expected x_stft to be 4D (N,1,F,T); got {tuple(x_stft.shape)}")
        if x_stft.shape[1] != 1:
            raise ValueError(f"DatasetBuilder: expected STFT channel dimension = 1; got X_stft.shape[1]={x_stft.shape[1]}")
        if x_iq.ndim != 3:
            raise ValueError(f"DatasetBuilder: expected X_iq to be 3D (N,2,1024); got {tuple(x_iq.shape)}")
        if x_iq.shape[1] != 2:
            raise ValueError(f"DatasetBuilder: expected IQ channel dimension = 2; got X_iq.shape[1]={x_iq.shape[1]}")
        if y.ndim != 1:
            raise ValueError(f"DatasetBuilder: expected y to be 1D (N,); got {tuple(y.shape)}")
        if not (x_stft.shape[0] == x_iq.shape[0] == y.shape[0]):
            raise ValueError(f"DatasetBuilder: batch sizes must match. Got X_stft={x_stft.shape[0]}, X_iq={x_iq.shape[0]}, y={y.shape[0]}")

        # Keep canonical dtypes for training
        self.X_stft = x_stft.float()
        self.X_iq = x_iq.float()
        self.self_y = y.long()

    def __len__(self) -> int:
        return int(self.self_y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_stft[idx], self.X_iq[idx], self.self_y[idx]

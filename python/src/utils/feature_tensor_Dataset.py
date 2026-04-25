from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class FeatureTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Tri-Branch Dataset.
    x_stft: (N, 2, F, T)
    x_iq: (N, 3, N_samples)
    x_if: (N, 1, N_samples)
    y: (N,)
    """

    def __init__(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            y: torch.Tensor
    ) -> None:

        if not all(isinstance(t, torch.Tensor) for t in [x_stft, x_iq, x_if, y]):
            raise TypeError("DatasetBuilder: inputs must be torch.Tensors")

        if x_stft.ndim != 4:
            raise ValueError(f"DatasetBuilder: expected x_stft to be 4D (N,1,F,T); got {tuple(x_stft.shape)}")
        if x_stft.shape[1] != 2:
            raise ValueError(
                f"DatasetBuilder: expected STFT channel dimension = 2; got x_stft.shape[1]={x_stft.shape[1]}")

        if x_iq.ndim != 3:
            raise ValueError(f"DatasetBuilder: expected x_iq to be 3D (N,2,1024); got {tuple(x_iq.shape)}")
        if x_iq.shape[1] != 3:
            raise ValueError(f"DatasetBuilder: expected IQ channel dimension = 2; got x_iq.shape[1]={x_iq.shape[1]}")

        if x_if.ndim != 3:
            raise ValueError(f"DatasetBuilder: expected x_if to be 3D (N,1,1024); got {tuple(x_if.shape)}")
        if x_if.shape[1] != 1:
            raise ValueError(f"DatasetBuilder: expected IF channel dimension = 1; got x_if.shape[1]={x_if.shape[1]}")

        if y.ndim != 1:
            raise ValueError(f"DatasetBuilder: expected y to be 1D (N,); got {tuple(y.shape)}")

        if not (x_stft.shape[0] == x_iq.shape[0] == x_if.shape[0] == y.shape[0]):
            raise ValueError(
                f"DatasetBuilder: batch sizes must match. Got STFT={x_stft.shape[0]}, "
                f"IQ={x_iq.shape[0]}, IF={x_if.shape[0]}, y={y.shape[0]}"
            )

        self.X_stft = x_stft.float()
        self.X_iq = x_iq.float()
        self.X_if = x_if.float()
        self.self_y = y.long()

    def __len__(self) -> int:
        return int(self.self_y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_stft[idx], self.X_iq[idx], self.X_if[idx], self.self_y[idx]
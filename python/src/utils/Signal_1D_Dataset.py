from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset

class FeatureTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    X: (N, 1, F, T),     y:(N,),
    Returns (x_i, y_i)
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("X and y must be torch.Tensors")

        if x.ndim != 4:
            raise ValueError(f"Expected X to be 4D (N,1,F,T); got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected channel dimension = 1; got X.shape[1]={x.shape[1]}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be 1D (N,); got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must match in N. Got X={x.shape[0]}, y={y.shape[0]}")

        # Keep canonical dtypes for training
        self.X = x.float()
        self.y = y.long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
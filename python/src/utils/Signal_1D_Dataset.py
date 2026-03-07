from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class FeatureTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Wraps prebuilt feature tensors for training/eval.

    X: torch.Tensor of shape (N, 1, F, T), dtype float32/float64
    y: torch.Tensor of shape (N,), dtype int64 recommended

    Returns:
        (x_i, y_i)
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("X and y must be torch.Tensors")

        if X.ndim != 4:
            raise ValueError(f"Expected X to be 4D (N,1,F,T); got {tuple(X.shape)}")
        if X.shape[1] != 1:
            raise ValueError(f"Expected channel dimension = 1; got X.shape[1]={X.shape[1]}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be 1D (N,); got {tuple(y.shape)}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must match in N. Got X={X.shape[0]}, y={y.shape[0]}")

        # Keep canonical dtypes for training
        self.X = X.float()
        self.y = y.long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
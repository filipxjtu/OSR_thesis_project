from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, random_split


def split_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Deterministic train/val split.
    """

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")

    dataset = TensorDataset(X, y)

    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_set, val_set
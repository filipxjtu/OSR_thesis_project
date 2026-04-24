from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, Subset


def split_dataset(
        x_stft: torch.Tensor,
        x_iq: torch.Tensor,
        x_if: torch.Tensor,
        y: torch.Tensor,
        train_ratio: float = 0.8,
        seed: int = 42
):
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)

    train_indices = []
    val_indices = []

    classes = torch.unique(y)

    for c in classes:
        class_indices = torch.where(y == c)[0]
        perm = class_indices[torch.randperm(len(class_indices), generator=generator)]
        train_size = int(train_ratio * len(class_indices))
        train_indices.append(perm[:train_size])
        val_indices.append(perm[train_size:])

    train_indices = torch.cat(train_indices)
    val_indices = torch.cat(val_indices)

    dataset = TensorDataset(x_stft, x_iq, x_if, y)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


def split_unknown(
        x_stft: torch.Tensor,
        x_iq: torch.Tensor,
        x_if: torch.Tensor,
        y: torch.Tensor,
        train_ratio: float = 0.50,
        val_ratio: float = 0.125,
        seed: int = 42
):
    total_len = len(y)
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(total_len, generator=generator)

    train_end = int(train_ratio * total_len)
    val_end = train_end + int(val_ratio * total_len)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    dataset = TensorDataset(x_stft, x_iq, x_if, y)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, Subset


def split_dataset(x_stft: torch.Tensor, x_iq: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8, seed: int = 42):

    """ Deterministic stratified train/val split. Ensures class proportions are preserved. """

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

    dataset = TensorDataset(x_stft, x_iq, y)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


def split_unknown(
        x_stft: torch.Tensor,
        x_iq: torch.Tensor,
        y: torch.Tensor,
        train_ratio: float = 0.50,
        val_ratio: float = 0.125,
        seed: int = 42
):
    """
    Randomly shuffles and splits unknown anomaly data into train, val, and test sets.
    Default ratios: 50% Train, 12.5% Val, 37.5% Test.
    """
    total_len = len(y)
    generator = torch.Generator().manual_seed(seed)

    # Generate a perfectly shuffled list of indices
    indices = torch.randperm(total_len, generator=generator)

    # Calculate dynamic split points
    train_end = int(train_ratio * total_len)
    val_end = train_end + int(val_ratio * total_len)

    # Slice the shuffled indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    dataset = TensorDataset(x_stft, x_iq, y)

    # Wrap in PyTorch Subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set
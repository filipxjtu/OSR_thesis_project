from __future__ import annotations

import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def create_train_loader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
) -> DataLoader:
    """
    Standard training DataLoader.
    Shuffling enabled.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def create_eval_loader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
) -> DataLoader:
    """
    Evaluation DataLoader.
    No shuffling.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
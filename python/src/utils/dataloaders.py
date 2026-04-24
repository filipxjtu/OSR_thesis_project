from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def create_train_loader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
) -> DataLoader:

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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
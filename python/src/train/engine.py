from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:

    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_stft, x_iq, y in loader:
        x_stft = x_stft.to(device)
        x_iq = x_iq.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x_stft, x_iq)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Empty DataLoader encountered.")

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_stft, x_iq, y in loader:
        x_stft = x_stft.to(device)
        x_iq = x_iq.to(device)
        y = y.to(device)

        logits = model(x_stft, x_iq)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        predicts = torch.argmax(logits, dim=1)
        total_correct += (predicts == y).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Empty DataLoader encountered.")

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return mean_loss, accuracy
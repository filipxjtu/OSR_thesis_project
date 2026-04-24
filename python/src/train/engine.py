from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion_ce: nn.Module,
        criterion_supcon: nn.Module,
        device: torch.device,
        lambda_supcon: float = 0.1,
) -> tuple[float, float, float]:
    """
    Joint CE + SupCon training.
    Single forward pass via return_fingerprint=True so gradients flow through
    the backbone for both losses.

    Returns (mean_total_loss, mean_ce_loss, mean_supcon_loss) per sample.
    """
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_samples = 0

    for x_stft, x_iq, x_if, y in loader:
        x_stft = x_stft.to(device)
        x_iq = x_iq.to(device)
        x_if = x_if.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # single forward pass; z is already L2-normalized by the model
        logits, z = model(x_stft, x_iq, x_if, return_fingerprint=True)

        ce_loss = criterion_ce(logits, y)
        supcon_loss = criterion_supcon(z, y)
        loss = ce_loss + lambda_supcon * supcon_loss

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_ce += ce_loss.item() * bs
        total_supcon += supcon_loss.item() * bs
        total_samples += bs

    if total_samples == 0:
        raise RuntimeError("Empty DataLoader encountered.")

    return (total_loss / total_samples,
            total_ce / total_samples,
            total_supcon / total_samples)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion_ce: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """
    Standard CE-only evaluation. SupCon is not evaluated; val accuracy is the signal.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_stft, x_iq, x_if, y in loader:
        x_stft = x_stft.to(device)
        x_iq = x_iq.to(device)
        x_if = x_if.to(device)
        y = y.to(device)

        logits = model(x_stft, x_iq, x_if)
        loss = criterion_ce(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        predicts = torch.argmax(logits, dim=1)
        total_correct += (predicts == y).sum().item()
        total_samples += bs

    if total_samples == 0:
        raise RuntimeError("Empty DataLoader encountered.")

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return mean_loss, accuracy
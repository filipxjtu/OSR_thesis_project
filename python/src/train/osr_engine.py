from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ..utils import combined_loss


def train_phase1_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int = 1,
) -> float:
    """Trains the backbone, updates EMA, and applies ArcFace logic via known classes."""
    model.train()
    total_loss, total_samples = 0.0, 0

    for x_stft, x_iq, y in loader:
        x_stft, x_iq, y = x_stft.to(device), x_iq.to(device), y.to(device)

        known = y != -1
        if not known.any():
            continue

        optimizer.zero_grad()

        logits = model.collect_and_update(x_stft[known], x_iq[known], y[known], epoch=epoch)
        loss = criterion(logits, y[known])

        loss.backward()
        optimizer.step()

        batch_size = known.sum().item()
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def train_phase2_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lambda_osr: float,
        lambda_entropy: float,
        device: torch.device,
) -> float:
    """Trains the calibrator using the combined OSR loss."""
    model.train()
    total_loss, total_samples = 0.0, 0

    for x_stft, x_iq, y in loader:
        x_stft, x_iq, y = x_stft.to(device), x_iq.to(device), y.to(device)

        optimizer.zero_grad()

        logits, unknown_score, _ = model.forward_with_osr(x_stft, x_iq)

        loss = combined_loss(
            logits, unknown_score, y,
            lambda_osr=lambda_osr,
            lambda_entropy=lambda_entropy,
        )

        loss.backward()
        optimizer.step()

        batch_size = x_stft.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_osr(
        model: nn.Module,
        loader_known: DataLoader | None,
        loader_osr: DataLoader | None,
        device: torch.device
) -> tuple[float, float, float, float]:
    """Returns: (known_accuracy, auroc, unknown_recall, false_alarm_rate)"""
    model.eval()
    all_labels, all_scores, all_preds = [], [], []

    if loader_known is not None:
        for x_stft, x_iq, y in loader_known:
            x_stft, x_iq = x_stft.to(device), x_iq.to(device)
            logits, score, _ = model.forward_with_osr(x_stft, x_iq)

            all_labels.append(y.cpu().numpy())
            all_scores.append(score.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    if loader_osr is not None:
        for x_stft, x_iq, y in loader_osr:
            x_stft, x_iq = x_stft.to(device), x_iq.to(device)
            logits, score, _ = model.forward_with_osr(x_stft, x_iq)

            all_labels.append(y.cpu().numpy())
            all_scores.append(score.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    if not all_labels:
        return 0.0, 0.5, 0.0, 0.0

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_preds = np.concatenate(all_preds)

    known_mask = all_labels != -1
    unknown_mask = all_labels == -1

    binary_labels = np.zeros_like(all_labels)
    binary_labels[unknown_mask] = 1

    try:
        auroc = float(roc_auc_score(binary_labels, all_scores))
    except ValueError:
        auroc = 0.5

    known_count = np.sum(known_mask)
    unknown_count = np.sum(unknown_mask)

    # Standardized evaluation threshold to preserve cross-run apples-to-apples metrics
    rejected = all_scores > 0.5

    known_acc = float(np.mean(all_preds[known_mask] == all_labels[known_mask])) if known_count > 0 else 0.0
    unknown_recall = float(np.mean(rejected[unknown_mask])) if unknown_count > 0 else 0.0
    false_alarm_rate = float(np.mean(rejected[known_mask])) if known_count > 0 else 0.0

    return known_acc, auroc, unknown_recall, false_alarm_rate
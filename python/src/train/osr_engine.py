from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ..utils import combined_loss


@torch.no_grad()
def populate_codebook_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        epoch: int = 1,
) -> None:
    model.eval()
    for x_stft, x_iq, x_if, y in loader:
        known = y != -1
        if not known.any():
            continue
        x_stft = x_stft[known].to(device)
        x_iq   = x_iq[known].to(device)
        x_if   = x_if[known].to(device)
        y_k    = y[known].to(device)
        model.collect_and_update(x_stft, x_iq, x_if, y_k, epoch=epoch)


def train_phase2_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lambda_osr: float,
        device: torch.device,
) -> float:
    """
    Train the calibrator for one epoch on mixed knowns + proxy unknowns.

    Uses forward_with_osr_logits so the loss can apply BCEWithLogitsLoss to
    the raw calibrator logit, which avoids the vanishing-gradient near
    sigmoid saturation that the old MSE-on-sigmoid path suffered from.
    """
    model.train()
    for p in model.base.parameters():
        p.requires_grad = False
    model.base.eval()

    total_loss, total_samples = 0.0, 0

    for x_stft, x_iq, x_if, y in loader:
        x_stft = x_stft.to(device)
        x_iq   = x_iq.to(device)
        x_if   = x_if.to(device)
        y      = y.to(device)

        optimizer.zero_grad()

        logits, unknown_score, unknown_logit = model.forward_with_osr_logits(
            x_stft, x_iq, x_if
        )

        loss = combined_loss(
            logits, unknown_score, y,
            lambda_osr=lambda_osr,
            unknown_logit=unknown_logit,
        )

        loss.backward()
        optimizer.step()

        bs = x_stft.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(1, total_samples)


@torch.no_grad()
def collect_validation_scores(
        model: nn.Module,
        loader_known: DataLoader,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the calibrator over the validation KNOWNS and collect
    (unknown_scores, predicted_classes). Used to set per-class thresholds
    via the (1 - target_fpr) percentile.
    """
    model.eval()
    scores: list[torch.Tensor] = []
    preds:  list[torch.Tensor] = []

    for x_stft, x_iq, x_if, _ in loader_known:
        x_stft = x_stft.to(device)
        x_iq   = x_iq.to(device)
        x_if   = x_if.to(device)

        logits, score, _ = model.forward_with_osr(x_stft, x_iq, x_if)
        scores.append(score.detach().cpu())
        preds.append(logits.argmax(dim=1).detach().cpu())

    if not scores:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.cat(scores, dim=0), torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate_osr(
        model: nn.Module,
        loader_known: DataLoader | None,
        loader_osr: DataLoader | None,
        device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_preds:  list[np.ndarray] = []

    for loader in (loader_known, loader_osr):
        if loader is None:
            continue
        for x_stft, x_iq, x_if, y in loader:
            x_stft = x_stft.to(device)
            x_iq   = x_iq.to(device)
            x_if   = x_if.to(device)

            logits, score, _ = model.forward_with_osr(x_stft, x_iq, x_if)

            all_labels.append(y.cpu().numpy())
            all_scores.append(score.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    if not all_labels:
        return 0.0, 0.5, 0.0, 0.0

    labels_arr = np.concatenate(all_labels)
    scores_arr = np.concatenate(all_scores)
    preds_arr  = np.concatenate(all_preds)

    known_mask   = labels_arr != -1
    unknown_mask = labels_arr == -1

    binary_labels = np.zeros_like(labels_arr)
    binary_labels[unknown_mask] = 1

    try:
        auroc = float(roc_auc_score(binary_labels, scores_arr))
    except ValueError:
        auroc = 0.5

    known_count   = int(np.sum(known_mask))
    unknown_count = int(np.sum(unknown_mask))

    thresholds_np     = model.class_thresholds.cpu().numpy()
    per_sample_thresh = thresholds_np[preds_arr]
    rejected          = scores_arr > per_sample_thresh

    known_acc        = float(np.mean(preds_arr[known_mask] == labels_arr[known_mask])) if known_count > 0 else 0.0
    unknown_recall   = float(np.mean(rejected[unknown_mask])) if unknown_count > 0 else 0.0
    false_alarm_rate = float(np.mean(rejected[known_mask]))   if known_count > 0 else 0.0

    return known_acc, auroc, unknown_recall, false_alarm_rate
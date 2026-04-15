from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def combined_loss(
    logits: torch.Tensor,
    unknown_score: torch.Tensor,
    labels: torch.Tensor,
    lambda_osr: float = 0.40,
) -> torch.Tensor:
    """  Pure loss for Phase 2 calibrator: OSR MSE exclusively. """

    device = logits.device

    known   = labels != -1
    unknown = ~known

    # Known samples should score near 0 (not unknown)
    osr_known = (
        unknown_score[known].pow(2).mean()
        if known.any()
        else torch.tensor(0.0, device=device)
    )

    # Unknown samples should score near 1 (unknown)
    osr_unknown = (
        (1.0 - unknown_score[unknown]).pow(2).mean()
        if unknown.any()
        else torch.tensor(0.0, device=device)
    )

    return lambda_osr * (osr_known + osr_unknown)


def compute_osr_metrics(
    preds: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Diagnostic OSR metrics using a fixed global threshold."""
    known   = labels != -1
    unknown = ~known
    eps = 1e-8

    rejected    = scores > threshold
    final_preds = preds.clone()
    final_preds[rejected] = -1

    total_known   = max(1, int(known.sum()))
    total_unknown = max(1, int(unknown.sum()))

    correct_known   = ((final_preds == labels) & known).sum().item()
    correct_unknown = (rejected & unknown).sum().item()
    false_rejected  = (rejected & known).sum().item()

    known_acc       = correct_known / total_known
    unknown_recall  = correct_unknown / total_unknown
    fpr             = false_rejected / total_known

    total_rejected      = max(1, int(rejected.sum()))
    detection_precision = correct_unknown / total_rejected
    f1 = (
        2 * unknown_recall * detection_precision
        / (unknown_recall + detection_precision + eps)
    )

    if known.any() and unknown.any():
        y_bin = np.zeros(labels.shape[0])
        y_bin[unknown.cpu().numpy().astype(bool)] = 1
        s = scores.detach().cpu().numpy()
        try:
            auroc = float(roc_auc_score(y_bin, s))
        except ValueError:
            auroc = 0.5
    else:
        auroc = 0.5

    return {
        "known_acc":      known_acc,
        "unknown_recall": unknown_recall,
        "fpr":            fpr,
        "auroc":          auroc,
        "f1_unknown":     f1,
    }
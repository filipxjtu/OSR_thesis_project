from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def combined_loss(
    logits: torch.Tensor,
    unknown_score: torch.Tensor,
    labels: torch.Tensor,
    lambda_osr: float = 0.20,
    lambda_entropy: float = 0.15,
) -> torch.Tensor:
    """
    Combined loss for Phase 2 calibrator: Cross-Entropy + OSR MSE + Entropy.
    CE is included for numerical grounding on known batches.
    """
    device = logits.device
    eps = 1e-8

    known = labels != -1
    unknown = ~known

    ce = (
        F.cross_entropy(logits[known], labels[known])
        if known.any() else torch.tensor(0.0, device=device)
    )

    osr_known = (
        unknown_score[known].pow(2).mean()
        if known.any() else torch.tensor(0.0, device=device)
    )

    osr_unknown = (
        (1.0 - unknown_score[unknown]).pow(2).mean()
        if unknown.any() else torch.tensor(0.0, device=device)
    )

    # Penalize low entropy to prevent score collapse
    entropy = -(
        unknown_score * torch.log(unknown_score + eps)
        + (1.0 - unknown_score) * torch.log(1.0 - unknown_score + eps)
    ).mean()

    return ce + lambda_osr * (osr_known + osr_unknown) + lambda_entropy * entropy


def compute_osr_metrics(
    preds: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculates diagnostic OSR metrics."""
    known = labels != -1
    unknown = ~known
    eps = 1e-8

    rejected = scores > threshold
    final_preds = preds.clone()
    final_preds[rejected] = -1

    total_known = max(1, int(known.sum()))
    total_unknown = max(1, int(unknown.sum()))

    correct_known = ((final_preds == labels) & known).sum().item()
    correct_unknown = (rejected & unknown).sum().item()
    false_rejected = (rejected & known).sum().item()

    known_acc = correct_known / total_known
    unknown_recall = correct_unknown / total_unknown
    fpr = false_rejected / total_known

    denom = correct_known + false_rejected
    precision = correct_known / denom if denom > 0 else 0.0
    f1 = 2 * unknown_recall * precision / (unknown_recall + precision + eps)

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
        "known_acc": known_acc,
        "unknown_recall": unknown_recall,
        "fpr": fpr,
        "auroc": auroc,
        "f1_unknown": f1,
    }
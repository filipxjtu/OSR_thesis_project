from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def combined_loss(
    logits: torch.Tensor,
    unknown_score: torch.Tensor,
    labels: torch.Tensor,
    lambda_osr: float = 0.3,
    lambda_entropy: float = 0.1,
) -> torch.Tensor:

    device = logits.device

    known = labels != -1
    unknown = labels == -1

    ce = (
        F.cross_entropy(logits[known], labels[known])
        if known.any()
        else torch.tensor(0.0, device=device)
    )

    loss_known = (
        unknown_score[known].pow(2).mean()
        if known.any()
        else torch.tensor(0.0, device=device)
    )

    loss_unknown = (
        (1 - unknown_score[unknown]).pow(2).mean()
        if unknown.any()
        else torch.tensor(0.0, device=device)
    )

    osr = loss_known + loss_unknown

    entropy = -(
        unknown_score * torch.log(unknown_score + 1e-8)
        + (1 - unknown_score) * torch.log(1 - unknown_score + 1e-8)
    ).mean()

    return ce + lambda_osr * osr + lambda_entropy * entropy


def compute_osr_metrics(
    preds: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
):

    device = scores.device

    known = labels != -1
    unknown = labels == -1

    pred_final = preds.clone()
    pred_final[scores > threshold] = -1

    known_total = max(1, int(known.sum().item()))
    unknown_total = max(1, int(unknown.sum().item()))

    known_acc = ((pred_final == labels) & known).sum().item() / known_total
    unknown_rec = ((scores > threshold) & unknown).sum().item() / unknown_total

    if known.any() and unknown.any():
        y = torch.cat([
            torch.zeros(int(known.sum()), device=device),
            torch.ones(int(unknown.sum()), device=device),
        ]).cpu().numpy()

        s = torch.cat([
            scores[known],
            scores[unknown],
        ]).detach().cpu().numpy()

        auroc = roc_auc_score(y, s)
    else:
        auroc = 0.5

    return {
        "known_acc": known_acc,
        "unknown_recall": unknown_rec,
        "auroc": auroc,
    }
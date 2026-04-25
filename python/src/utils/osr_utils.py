from __future__ import annotations

import torch


def combined_loss(
    logits: torch.Tensor,
    unknown_score: torch.Tensor,
    labels: torch.Tensor,
    lambda_osr: float = 0.40,
) -> torch.Tensor:
    device = logits.device

    known   = labels != -1
    unknown = ~known

    osr_known = (
        unknown_score[known].pow(2).mean()
        if known.any()
        else torch.tensor(0.0, device=device)
    )

    osr_unknown = (
        (1.0 - unknown_score[unknown]).pow(2).mean()
        if unknown.any()
        else torch.tensor(0.0, device=device)
    )

    return lambda_osr * (osr_known + osr_unknown)
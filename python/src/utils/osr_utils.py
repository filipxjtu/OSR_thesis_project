from __future__ import annotations

import torch
import torch.nn.functional as F


def combined_loss(
    logits: torch.Tensor,
    unknown_score: torch.Tensor,
    labels: torch.Tensor,
    lambda_osr: float = 0.40,
    unknown_logit: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    OSR calibrator loss.

    Targets:
      - knowns   (label != -1) → unknown_score should be 0
      - unknowns (label == -1) → unknown_score should be 1

    The loss is balanced across the two groups regardless of their batch
    proportion (each group's contribution is its own mean), then summed and
    weighted by lambda_osr.

    If `unknown_logit` (the pre-sigmoid output of the calibrator) is provided,
    we use binary_cross_entropy_with_logits — numerically stable and avoids
    the vanishing-gradient pathology of MSE-on-sigmoid near saturation.

    If only `unknown_score` (post-sigmoid) is provided, we fall back to the
    original MSE formulation. Kept for backward compatibility with any
    caller that still passes the post-sigmoid score.

    `logits` is unused for the loss itself (the closed-set classifier is
    frozen during Phase 2) but kept in the signature for API stability.
    """
    device = unknown_score.device

    known   = labels != -1
    unknown = ~known

    if unknown_logit is not None:
        # BCE-with-logits path (preferred)
        if known.any():
            target_k = torch.zeros_like(unknown_logit[known])
            osr_known = F.binary_cross_entropy_with_logits(
                unknown_logit[known], target_k, reduction="mean"
            )
        else:
            osr_known = torch.tensor(0.0, device=device)

        if unknown.any():
            target_u = torch.ones_like(unknown_logit[unknown])
            osr_unknown = F.binary_cross_entropy_with_logits(
                unknown_logit[unknown], target_u, reduction="mean"
            )
        else:
            osr_unknown = torch.tensor(0.0, device=device)
    else:
        # Legacy MSE-on-sigmoid path
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
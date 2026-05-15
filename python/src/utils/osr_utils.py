from __future__ import annotations

import torch
import torch.nn.functional as F


def combined_loss(
    logits: torch.Tensor,
    unknown_score: torch.Tensor,
    labels: torch.Tensor,
    lambda_osr: float = 0.40,
    unknown_logit: torch.Tensor | None = None,
    class_weights: torch.Tensor = None
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
    we use a focal-weighted binary_cross_entropy_with_logits — numerically
    stable and focuses learning on hard examples.

    If only `unknown_score` (post-sigmoid) is provided, we fall back to the
    original MSE formulation. Kept for backward compatibility with any
    caller that still passes the post-sigmoid score.

    `logits` is unused for the loss itself (the closed-set classifier is
    frozen during Phase 2) but kept in the signature for API stability.
    """
    device = unknown_score.device

    known   = labels != -1
    unknown = ~known

    gamma = 2.0  # focusing parameter; increase if you still see overlap

    if unknown_logit is not None:
        # ---------- knowns (target = 0) ----------
        if known.any():
            target_k = torch.zeros_like(unknown_logit[known])
            bce_k = F.binary_cross_entropy_with_logits(
                unknown_logit[known], target_k, reduction="none"
            )
            p_k = torch.sigmoid(unknown_logit[known])   # p(unknown) for knowns
            # hard knowns have p_k → 1 (calibrator wrongly thinks they are unknown)
            osr_known = ((p_k + 1e-6) ** gamma * bce_k).mean()
        else:
            osr_known = torch.tensor(0.0, device=device)

        # ---------- unknowns (target = 1) ----------
        if unknown.any():
            unk_logit = unknown_logit[unknown]
            target_u = torch.ones_like(unk_logit)
            bce_u = F.binary_cross_entropy_with_logits(
                unk_logit, target_u, reduction="none"
            )
            p_u = torch.sigmoid(unk_logit)   # p(unknown) for unknowns
            # hard unknowns have p_u → 0 (calibrator wrongly thinks they are known)
            if class_weights is not None:
                pred_u = logits[unknown].argmax(dim=1)
                w = class_weights.to(unk_logit.device)[pred_u]
                focal_u = (1.0 - p_u + 1e-6) ** gamma * bce_u
                osr_unknown = (focal_u * w).sum() / w.sum().clamp(min=1e-6)
            else:
                osr_unknown = ((1.0 - p_u + 1e-6) ** gamma * bce_u).mean()
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
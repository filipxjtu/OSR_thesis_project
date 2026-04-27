from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OSRHParams:
    # Codebook
    k_centroids: int = 4
    ema_momentum: float = 0.95
    codebook_beta: float = 1.0

    # Curriculum
    warmup_epochs: int = 30
    threshold_recal_interval: int = 5

    # Optimization
    lr_backbone: float = 1e-3
    lr_calibrator: float = 1e-3
    batch_size: int = 32

    # Loss Weights
    lambda_osr: float = 0.40
    lambda_supcon: float = 0.1

    # Threshold calibration
    # Per-class threshold is set at the (1 - target_fpr) percentile of the
    # unknown_score distribution on validation knowns. So a target_fpr of 0.10
    # means: at this threshold, ~10% of validation knowns would be (wrongly)
    # rejected as unknown. Lower values are stricter (fewer rejections, higher
    # FPR risk on out-of-distribution knowns); higher values are more eager
    # to flag unknown.
    target_fpr: float = 0.10
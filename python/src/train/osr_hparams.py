from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OSRHParams:
    # Codebook
    k_centroids: int = 4
    ema_momentum: float = 0.95
    codebook_beta: float = 0.9

    # Curriculum
    warmup_epochs: int = 30
    threshold_recal_interval: int = 5

    # Optimization
    lr_backbone: float = 1e-3
    calibrator_weight_decay = 1e-4
    lr_calibrator: float = 1e-3
    batch_size: int = 32

    # Loss weights
    lambda_osr: float = 0.40
    lambda_supcon: float = 0.1

    # Threshold calibration
    target_fpr: float = 0.1
    fpr_cap: float = 0.4

    early_stopping_patience = 10
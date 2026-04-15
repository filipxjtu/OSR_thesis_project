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
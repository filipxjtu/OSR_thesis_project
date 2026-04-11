from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OSRHParams:
    # Codebook
    k_centroids: int = 4
    ema_momentum: float = 0.95

    # Curriculum
    warmup_epochs: int = 30

    # Optimization
    lr_backbone: float = 1e-3
    lr_calibrator: float = 1e-3
    batch_size: int = 32

    # Loss Weights
    lambda_osr: float = 0.20
    lambda_entropy: float = 0.15
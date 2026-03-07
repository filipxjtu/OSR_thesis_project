from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class HParams:
    """
    Centralized training hyperparameter definition.

    This object must be passed explicitly.
    No hidden globals allowed.
    """

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Training control
    epochs: int = 20
    batch_size: int = 32

    # Device
    device: str = "cpu"  # "cpu" or "cuda"

    # Logging
    log_interval: int = 10

    # Optional reproducibility seed
    seed: Optional[int] = None
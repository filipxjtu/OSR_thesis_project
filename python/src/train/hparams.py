from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(slots=True)
class HParams:

    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0

    # training control
    epochs: int = 20
    batch_size: int = 32

    # device
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logging
    log_interval: int = 10

    # optional reproducibility seed
    seed: Optional[int] = None
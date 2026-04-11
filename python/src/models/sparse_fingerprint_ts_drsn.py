from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts_ms_va_drsn import TS_MS_VA_DRSN, ResidualShrinkageBlockCW


# Sparse-code collector (hook-based)

class _SparseCodeCollector:
    __slots__ = ("_codes", "_hooks")

    def __init__(self):
        self._codes: List[torch.Tensor] = []
        self._hooks: List = []

    def register(self, model: nn.Module) -> "_SparseCodeCollector":
        for mod in model.modules():
            if isinstance(mod, ResidualShrinkageBlockCW):
                self._hooks.append(
                    mod.register_forward_hook(self._make_hook(mod))
                )
        return self

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, block: ResidualShrinkageBlockCW):
        def _hook(_m, _i, _o):
            thresh = getattr(block, "last_threshold", None)
            gap = getattr(block, "last_gap", None)
            if thresh is None or gap is None:
                return

            # thresh: [B, C, 1, 1] -> squeeze spatial dims
            # gap: [B, C]
            survived = (gap >= thresh.squeeze(-1).squeeze(-1)).float()
            self._codes.append(survived)

        return _hook

    def get_code(self) -> Optional[torch.Tensor]:
        """Concatenate codes from all blocks -> [B, total_C]."""
        if not self._codes:
            return None
        return torch.cat(self._codes, dim=1)


# Per-class codebook

class _SparseCodebook(nn.Module):
    """
    Stores k soft centroids per class, updated online via EMA.
    Not gradient-trained; updated via model.collect_and_update() during Phase 1.
    """

    def __init__(
            self,
            num_classes: int,
            code_dim: int,
            k: int = 4,
            ema_momentum: float = 0.95,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.k = k
        self.ema_momentum = ema_momentum

        # Centroids [num_classes, k, code_dim] -> uninformed prior = 0.5
        self.register_buffer("centroids", torch.full((num_classes, k, code_dim), 0.5))
        self.register_buffer("initialised", torch.zeros(num_classes, k, dtype=torch.bool))

    @torch.no_grad()
    def update(self, codes: torch.Tensor, labels: torch.Tensor):
        """EMA update for known classes present in the batch."""
        for c in labels.unique():
            if c.item() == -1:
                continue

            mask = labels == c
            class_codes = codes[mask]
            c_idx = int(c.item())

            for kid in range(self.k):
                centroid = self.centroids[c_idx, kid]

                # Initialize centroid from the first sample
                if not self.initialised[c_idx, kid]:
                    self.centroids[c_idx, kid] = class_codes[0]
                    self.initialised[c_idx, kid] = True
                    class_codes = class_codes[1:]
                    if class_codes.shape[0] == 0:
                        break
                    continue

                # Soft assignment: use samples closer than (mean + 0.1)
                dists = (class_codes - centroid).abs().mean(dim=1)
                assigned = class_codes[dists < dists.mean() + 0.1]

                if assigned.shape[0] == 0:
                    continue

                mean_assigned = assigned.mean(dim=0)
                m = self.ema_momentum
                self.centroids[c_idx, kid] = m * centroid + (1 - m) * mean_assigned

    def hamming_distance(self, codes: torch.Tensor, pred_class: torch.Tensor) -> torch.Tensor:
        """
        Vectorized minimum normalized Hamming distance to nearest centroid of predicted class.
        Returns [B] in [0, 1]. Low = known match, High = likely unknown.
        """
        cents = self.centroids[pred_class]
        d = (codes.unsqueeze(1) - cents).abs().mean(dim=2)
        return d.min(dim=1).values

    def convergence_stats(self) -> Dict[str, torch.Tensor]:
        """Returns per-class centroid spread to monitor codebook health."""
        return {
            "mean_activation_per_class": self.centroids.mean(dim=[1, 2]),
            "spread_per_class": self.centroids.std(dim=1).mean(dim=1),
            "pct_initialised": self.initialised.float().mean(),
        }


# Main model

class SparseFingerprint_TS_DRSN(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            k_centroids: int = 4,
            ema_momentum: float = 0.95,
            warmup_epochs: int = 30,
            use_pretrained: bool = False,
            pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs

        # Two-Stream Backbone
        self.base = TS_MS_VA_DRSN(num_classes=num_classes)

        if use_pretrained and pretrained_path:
            self.base.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
            print(f"[SparseFingerprint] Loaded backbone from {pretrained_path}")

        # Codebook (built lazily on first forward pass)
        self._code_dim: Optional[int] = None
        self._codebook: Optional[_SparseCodebook] = None
        self._k = k_centroids
        self._ema_mom = ema_momentum

        # Score Calibrator (Frozen Phase 1, Trained Phase 2)
        self.score_calibrator = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Learnable temperature for classification logits
        self._log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self._log_temperature) + 1e-3

    def current_phase(self, epoch: int) -> int:
        return 1 if epoch <= self.warmup_epochs else 2

    def set_phase(self, epoch: int):
        """
        Call at the start of each epoch to set the correct freeze state.

        Phase 1: backbone trainable, calibrator frozen.
        Phase 2: backbone frozen, calibrator trainable.
        """
        phase = self.current_phase(epoch)
        if phase == 1:
            self._unfreeze_backbone()
            self._freeze_calibrator()
        else:
            self._freeze_backbone()
            self._unfreeze_calibrator()

    def _freeze_backbone(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self):
        for p in self.base.parameters():
            p.requires_grad = True

    def _freeze_calibrator(self):
        for p in self.score_calibrator.parameters():
            p.requires_grad = False

    def _unfreeze_calibrator(self):
        for p in self.score_calibrator.parameters():
            p.requires_grad = True

    def freeze_base(self):
        self._freeze_backbone()

    def unfreeze_base(self):
        self._unfreeze_backbone()

    def _ensure_codebook(self, code_dim: int, device: torch.device):
        """Lazily create the codebook on first forward pass."""
        if self._codebook is None:
            self._codebook = _SparseCodebook(
                num_classes=self.num_classes,
                code_dim=code_dim,
                k=self._k,
                ema_momentum=self._ema_mom,
            ).to(device)
            self._code_dim = code_dim

    def codebook_ready(self) -> bool:
        """True once the codebook has been initialized from real data."""
        if self._codebook is None:
            return False
        return bool(self._codebook.initialised.all().item())

    def get_codebook_stats(self) -> Optional[Dict]:
        if self._codebook is None:
            return None
        return self._codebook.convergence_stats()

    def _forward_with_code(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embedding [B,256], logits [B,C], sparse_code [B,D])."""
        collector = _SparseCodeCollector().register(self.base)
        try:
            embedding = self.base.extract_embedding(x_stft, x_iq)
        finally:
            collector.remove()

        logits = self.base.classifier(embedding) / self.temperature
        sparse_code = collector.get_code()

        if sparse_code is None:
            sparse_code = torch.zeros(x_stft.size(0), 1, device=x_stft.device)

        return embedding, logits, sparse_code

    def collect_and_update(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase 1 helper: one forward pass + EMA codebook update.
        Unknown labels (-1) are silently ignored by the codebook.
        """
        _, logits, code = self._forward_with_code(x_stft, x_iq)
        self._ensure_codebook(code.size(1), x_stft.device)
        self._codebook.update(code, labels)
        return logits

    def forward_with_osr(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Returns logits [B, num_classes] and unknown_score [B]."""
        if x_stft.ndim != 4 or x_stft.shape[1] != 1:
            raise ValueError(f"Expected x_stft (N,1,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq (N,2,L), got {tuple(x_iq.shape)}")

        _, logits, code = self._forward_with_code(x_stft, x_iq)
        self._ensure_codebook(code.size(1), x_stft.device)

        pred_class = logits.argmax(dim=1)
        ham_dist = self._codebook.hamming_distance(code, pred_class)
        max_prob = logits.softmax(dim=1).max(dim=1).values

        calib_input = torch.stack([ham_dist, 1.0 - max_prob], dim=1)
        unknown_score = self.score_calibrator(calib_input).squeeze(1)

        return logits, unknown_score, None

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_osr(x_stft, x_iq)
        return logits

    def predict_with_rejection(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            unknown_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns predictions and confidence. Predictions == -1 if rejected."""
        logits, unknown_score, _ = self.forward_with_osr(x_stft, x_iq)
        probs = logits.softmax(dim=1)
        confidence, predictions = probs.max(dim=1)

        predictions = predictions.clone()
        predictions[unknown_score > unknown_threshold] = -1
        return predictions, confidence

    @torch.no_grad()
    def extract_embedding(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
    ) -> torch.Tensor:
        embedding, _, _ = self._forward_with_code(x_stft, x_iq)
        return embedding
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts_ms_va_drsn import TS_MS_VA_DRSN, ResidualShrinkageBlockCW



# Sparse-code collector (hook-based, single forward pass)

class _SparseCodeCollector:
    """
    Collects the binary activation mask from every ResidualShrinkageBlockCW.
    In the Two-Stream architecture, this automatically isolates the STFT
    macro-frequency envelope sparsity patterns, ignoring the raw IQ branch.
    """

    __slots__ = ("_codes", "_hooks")

    def __init__(self):
        self._codes: List[torch.Tensor] = []  # each [B, C] float (binary 0/1)
        self._hooks: list = []

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
            # True where channel survived (gap >= threshold)
            # thresh is [B, C, 1, 1], gap is [B, C]
            survived = (gap >= thresh.squeeze(-1).squeeze(-1)).float()  # [B, C]
            self._codes.append(survived)

        return _hook

    def get_code(self) -> Optional[torch.Tensor]:
        """Concatenate codes from all blocks -> [B, total_C]"""
        if not self._codes:
            return None
        return torch.cat(self._codes, dim=1)



# Per-class codebook (EMA-updated, k centroids per class)

class _SparseCodebook(nn.Module):
    """
    Stores k centroids per class as soft (float) prototypes in [0, 1].
    Updated online via Exponential Moving Average during training.
    """

    def __init__(
            self,
            num_classes: int,
            code_dim: int,
            k: int = 4,
            ema_momentum: float = 0.99,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.k = k
        self.ema_momentum = ema_momentum

        # Centroids: [num_classes, k, code_dim]
        self.register_buffer(
            "centroids",
            torch.full((num_classes, k, code_dim), 0.5),
        )
        self.register_buffer(
            "initialised",
            torch.zeros(num_classes, k, dtype=torch.bool),
        )

    @torch.no_grad()
    def update(self, codes: torch.Tensor, labels: torch.Tensor):
        for c in labels.unique():
            # Skip unknown labels (-1) if they accidentally slip through
            if c.item() == -1: continue

            mask = labels == c
            class_codes = codes[mask]  # [n, D]
            c_idx = c.item()

            for kid in range(self.k):
                centroid = self.centroids[c_idx, kid]  # [D]

                if not self.initialised[c_idx, kid]:
                    self.centroids[c_idx, kid] = class_codes[0]
                    self.initialised[c_idx, kid] = True
                    class_codes = class_codes[1:]
                    if class_codes.shape[0] == 0:
                        break
                    continue

                dists = (class_codes - centroid).abs().mean(dim=1)  # [n]
                assigned = class_codes[dists < dists.mean() + 0.1]  # soft threshold

                if assigned.shape[0] == 0:
                    continue

                mean_assigned = assigned.mean(dim=0)
                m = self.ema_momentum
                self.centroids[c_idx, kid] = m * centroid + (1 - m) * mean_assigned

    def hamming_distance(self, codes: torch.Tensor, pred_class: torch.Tensor) -> torch.Tensor:
        B = codes.size(0)
        distances = torch.zeros(B, device=codes.device)

        for i in range(B):
            c_idx = pred_class[i].item()
            cents = self.centroids[c_idx]  # [k, D]
            d = (codes[i].unsqueeze(0) - cents).abs().mean(dim=1)  # [k]
            distances[i] = d.min()

        return distances  # [B]  in [0, 1]


# Main model — SparseFingerprint_TS_DRSN

class SparseFingerprint_TS_DRSN(nn.Module):
    """
    Two-Stream Sparse Activation Code Fingerprinting for Open-Set Recognition.
    """

    def __init__(
            self,
            num_classes: int = 10,
            k_centroids: int = 4,
            ema_momentum: float = 0.99,
            use_pretrained: bool = False,
            pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Two-Stream Backbone
        self.base = TS_MS_VA_DRSN(num_classes=num_classes)

        if use_pretrained and pretrained_path:
            self.base.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
            print(f"[SparseFingerprint] Loaded pretrained weights from {pretrained_path}")

        #  Codebook
        self._code_dim: Optional[int] = None
        self._codebook: Optional[_SparseCodebook] = None
        self._k_centroids = k_centroids
        self._ema_momentum = ema_momentum

        #  Calibration Head (Hamming_Dist, 1-MaxProb) -> Unknown_Score
        self.score_calibrator = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self._log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self._log_temperature) + 1e-3

    def _ensure_codebook(self, code_dim: int, device: torch.device):
        if self._codebook is None:
            self._codebook = _SparseCodebook(
                num_classes=self.num_classes,
                code_dim=code_dim,
                k=self._k_centroids,
                ema_momentum=self._ema_momentum,
            ).to(device)
            self._code_dim = code_dim

    def _forward_with_code(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @torch.no_grad()
    def collect_and_update(self, x_stft: torch.Tensor, x_iq: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Runs a forward pass and updates the EMA codebook using Known labels. """
        _, logits, code = self._forward_with_code(x_stft, x_iq)
        self._ensure_codebook(code.size(1), x_stft.device)
        self._codebook.update(code, labels)
        return logits

    def forward_with_osr(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[Dict]]:
        if x_stft.ndim != 4 or x_stft.shape[1] != 1:
            raise ValueError(f"Expected x_stft (N,1,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq (N,2,L), got {tuple(x_iq.shape)}")

        embedding, logits, code = self._forward_with_code(x_stft, x_iq)

        self._ensure_codebook(code.size(1), x_stft.device)
        assert self._codebook is not None

        pred_class = logits.argmax(dim=1)
        ham_dist = self._codebook.hamming_distance(code, pred_class)
        max_prob = logits.softmax(dim=1).max(dim=1).values

        # Calibrated score
        calib_input = torch.stack([ham_dist, 1.0 - max_prob], dim=1)
        unknown_score = self.score_calibrator(calib_input).squeeze(1)

        return logits, unknown_score, None

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_osr(x_stft, x_iq)
        return logits

    def predict_with_rejection(self, x_stft: torch.Tensor, x_iq: torch.Tensor, unknown_threshold: float = 0.5):
        logits, unknown_score, _ = self.forward_with_osr(x_stft, x_iq)
        probs = logits.softmax(dim=1)
        confidence, predictions = probs.max(dim=1)
        predictions = predictions.clone()
        predictions[unknown_score > unknown_threshold] = -1
        return predictions, confidence

    @torch.no_grad()
    def extract_embedding(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        embedding, _, _ = self._forward_with_code(x_stft, x_iq)
        return embedding

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = True
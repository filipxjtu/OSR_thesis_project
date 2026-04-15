from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts_ms_va_drsn import TS_MS_VA_DRSN, ResidualShrinkageBlockCW, IQPhysicsBranch


class ArcMarginProduct(nn.Module):
    """
    Projects features onto a hypersphere and applies a strict angular margin penalty.
    Forces maximum intra-class compactness and inter-class separation.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 16.0, m: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class _SparseCodeCollector:
    __slots__ = ("_codes", "_hooks")

    def __init__(self):
        self._codes: List[torch.Tensor] = []
        self._hooks: List = []

    def register(self, model: nn.Module) -> "_SparseCodeCollector":
        for mod in model.modules():
            if isinstance(mod, ResidualShrinkageBlockCW):
                self._hooks.append(
                    mod.register_forward_hook(self._make_stft_hook(mod))
                )
            elif isinstance(mod, IQPhysicsBranch):
                self._hooks.append(
                    mod.register_forward_hook(self._make_iq_hook())
                )
        return self

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_stft_hook(self, block: ResidualShrinkageBlockCW):
        def _hook(_m, _i, _o):
            thresh = getattr(block, "last_threshold", None)
            gap = getattr(block, "last_gap", None)
            if thresh is None or gap is None:
                return
            survived = (gap >= thresh.squeeze(-1).squeeze(-1)).float()
            self._codes.append(survived)

        return _hook

    def _make_iq_hook(self):
        def _hook(_m, _i, _o):
            # _o is already (B, out_features) after AdaptiveAvgPool1d + view in IQPhysicsBranch
            binary_code = (_o > _o.mean(dim=1, keepdim=True)).float()
            self._codes.append(binary_code)

        return _hook

    def get_code(self) -> Optional[torch.Tensor]:
        if not self._codes:
            warnings.warn(
                "[SparseCodeCollector] No codes collected — hooks may have fired before "
                "DRSN blocks populated last_threshold/last_gap.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return torch.cat(self._codes, dim=1)


class _SparseCodebook(nn.Module):
    def __init__(
            self,
            num_classes: int,
            code_dim: int,
            k: int = 4,
            ema_momentum: float = 0.95,
            beta: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.k = k
        self.ema_momentum = ema_momentum
        self.beta = beta

        self.register_buffer("centroids", torch.full((num_classes, k, code_dim), 0.5))
        self.register_buffer("initialised", torch.zeros(num_classes, k, dtype=torch.bool))
        self.register_buffer("update_counts", torch.zeros(num_classes, k, dtype=torch.long))

    @torch.no_grad()
    def update(self, codes: torch.Tensor, labels: torch.Tensor, current_momentum: float = 0.95):
        for c in labels.unique():
            if c.item() == -1:
                continue

            mask = labels == c
            class_codes = codes[mask]
            c_idx = int(c.item())

            for kid in range(self.k):
                centroid = self.centroids[c_idx, kid]

                if not self.initialised[c_idx, kid]:
                    self.centroids[c_idx, kid] = class_codes[0]
                    self.initialised[c_idx, kid] = True
                    self.update_counts[c_idx, kid] = 1
                    class_codes = class_codes[1:]
                    if class_codes.shape[0] == 0:
                        break
                    continue

                dists = (class_codes - centroid).abs().mean(dim=1)

                std_dist = dists.std(unbiased=False).item() if dists.size(0) > 1 else 0.0
                assigned = class_codes[dists <= dists.mean() + (self.beta * std_dist)]

                if assigned.shape[0] == 0:
                    continue

                mean_assigned = assigned.mean(dim=0)
                m = current_momentum
                self.centroids[c_idx, kid] = m * centroid + (1 - m) * mean_assigned
                self.update_counts[c_idx, kid] += assigned.shape[0]

    def code_distance(self, codes: torch.Tensor, pred_class: torch.Tensor) -> torch.Tensor:
        cents = self.centroids[pred_class]
        d = (codes.unsqueeze(1) - cents).abs().mean(dim=2)
        return d.min(dim=1).values

    def convergence_stats(self) -> Dict[str, torch.Tensor]:
        return {
            "mean_activation_per_class": self.centroids.mean(dim=[1, 2]),
            "spread_per_class": self.centroids.std(dim=1).mean(dim=1),
            "pct_initialised": self.initialised.float().mean(),
            "mean_updates_per_centroid": self.update_counts.float().mean(),
        }


class SparseFingerprint_TS_DRSN(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            k_centroids: int = 4,
            ema_momentum: float = 0.95,
            warmup_epochs: int = 30,
            codebook_beta: float = 1.0,
            threshold_recal_interval: int = 5,
            use_pretrained: bool = False,
            pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.threshold_recal_interval = threshold_recal_interval

        self.base = TS_MS_VA_DRSN(num_classes=num_classes)

        # Strip the dead backbone linear classification head to avoid conflicting parameters
        self.base.classifier = nn.Identity()

        if use_pretrained and pretrained_path:
            self.base.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
            print(f"[SparseFingerprint] Loaded backbone from {pretrained_path}")

        self.arcface = ArcMarginProduct(in_features=256, out_features=num_classes)

        self._code_dim: Optional[int] = None
        self._codebook: Optional[_SparseCodebook] = None
        self._k = k_centroids
        self._ema_mom = ema_momentum
        self._beta = codebook_beta

        # Input: [code_distance, 1-max_prob, emb_norm] — LayerNorm aligns scales before the MLP
        self.score_calibrator = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.phase2_active = False
        self.register_buffer("class_thresholds", torch.full((num_classes,), 0.5))

    def current_phase(self) -> int:
        return 2 if self.phase2_active else 1

    def set_phase(self):
        phase = self.current_phase()
        if phase == 1:
            self._unfreeze_backbone()
            self._freeze_calibrator()
        else:
            self._freeze_backbone()
            self._unfreeze_calibrator()

    def check_dynamic_phase_switch(self, epoch: int) -> bool:
        if self.phase2_active:
            return True

        if epoch >= self.warmup_epochs:
            self.phase2_active = True
            self.calibrate_class_thresholds()
            return True

        if epoch < 23:
            return False

        stats = self.get_codebook_stats()
        if stats is None:
            return False

        pct_init = float(stats["pct_initialised"])
        mean_updates = float(stats["mean_updates_per_centroid"])
        spread = float(stats["spread_per_class"].mean())

        # Guard against triggering on the initial state where all k centroids
        # are identical (spread ≈ 0 does not mean converged, just uninitialised)
        genuinely_converged = (
            pct_init >= 1.0
            and mean_updates > self._k
            and spread < 0.05
        )

        if genuinely_converged:
            self.phase2_active = True
            self.calibrate_class_thresholds()
            return True

        return False

    @torch.no_grad()
    def calibrate_class_thresholds(self, base_threshold: float = 0.5):
        if self._codebook is not None:
            spreads = self._codebook.convergence_stats()["spread_per_class"]
            norm_spreads = spreads / (spreads.max() + 1e-6)

            adjusted = base_threshold * (0.8 + 0.4 * (1 - norm_spreads))
            self.class_thresholds.copy_(adjusted.clamp(0.15, 0.85))

    def _freeze_backbone(self):
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.arcface.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self):
        for p in self.base.parameters():
            p.requires_grad = True
        for p in self.arcface.parameters():
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
        if self._codebook is None:
            self._codebook = _SparseCodebook(
                num_classes=self.num_classes,
                code_dim=code_dim,
                k=self._k,
                ema_momentum=self._ema_mom,
                beta=self._beta,
            ).to(device)
            self._code_dim = code_dim

    def codebook_ready(self) -> bool:
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
            labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        collector = _SparseCodeCollector().register(self.base)
        try:
            embedding = self.base.extract_embedding(x_stft, x_iq)
        finally:
            collector.remove()

        logits = self.arcface(embedding, labels)
        sparse_code = collector.get_code()

        if sparse_code is None:
            fallback_dim = self._code_dim if self._code_dim is not None else 1
            sparse_code = torch.zeros(x_stft.size(0), fallback_dim, device=x_stft.device)

        return embedding, logits, sparse_code

    def collect_and_update(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            labels: torch.Tensor,
            epoch: int = 1,
    ) -> torch.Tensor:
        _, logits, code = self._forward_with_code(x_stft, x_iq, labels=labels)
        self._ensure_codebook(code.size(1), x_stft.device)

        current_momentum = min(0.95, 0.85 + 0.1 * (epoch / max(1, self.warmup_epochs)))
        self._codebook.update(code, labels, current_momentum=current_momentum)
        return logits

    def forward_with_osr(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        if x_stft.ndim != 4 or x_stft.shape[1] != 1:
            raise ValueError(f"Expected x_stft (N,1,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq (N,2,L), got {tuple(x_iq.shape)}")

        embedding, logits, code = self._forward_with_code(x_stft, x_iq, labels=None)
        self._ensure_codebook(code.size(1), x_stft.device)

        pred_class = logits.argmax(dim=1)
        code_dist = self._codebook.code_distance(code, pred_class)

        max_prob = logits.softmax(dim=1).max(dim=1).values

        emb_norm = embedding.norm(dim=1)
        emb_norm_normalised = (emb_norm / (emb_norm.detach().mean() + 1e-6)).clamp(0, 3) / 3.0

        calib_input = torch.stack([code_dist, 1.0 - max_prob, emb_norm_normalised], dim=1)
        unknown_score = self.score_calibrator(calib_input).squeeze(1)

        return logits, unknown_score, None

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_osr(x_stft, x_iq)
        return logits

    def predict_with_rejection(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, unknown_score, _ = self.forward_with_osr(x_stft, x_iq)

        probs = logits.softmax(dim=1)
        confidence, predictions = probs.max(dim=1)

        predictions = predictions.clone()
        thresh = self.class_thresholds[predictions]
        predictions[unknown_score > thresh] = -1

        return predictions, confidence

    @torch.no_grad()
    def extract_embedding(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
    ) -> torch.Tensor:
        embedding, _, _ = self._forward_with_code(x_stft, x_iq, labels=None)
        return embedding
"""
python/src/legacy_models/densenet_121.py
=========================================
DenseNet-121 baseline for fair comparison against AsymmetricTriNet.

Design decisions:
  - Input: 2-channel STFT at native spatial size (no interpolation).
    DenseNet uses adaptive avg-pooling before its classifier, so any
    spatial size ≥ 32×32 is accepted.
  - Weights: ImageNet pretrained.
  - Input stem: features.conv0 replaced 3-ch → 2-ch. The 2-ch weights are
    initialised so the expected pre-activation magnitude matches the
    pretrained 3-ch stem on a uniform input: mean(RGB) * (3 / 2) = mean * 1.5.
  - Classifier head: backbone.classifier replaced with Linear(1024, num_classes).
  - Freezing: two-phase.
      Phase 1: freeze everything except new classifier.
      Phase 2: unfreeze denseblock4 + norm5 + classifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


class LiteratureBaseline_DenseNet121(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        backbone = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # ── Adapt input stem: 3-ch → 2-ch ────────────────────────────
        orig_conv = backbone.features.conv0            # Conv2d(3, 64, 7, stride=2, padding=3)
        new_stem  = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            mean_w = orig_conv.weight.mean(dim=1, keepdim=True)   # (64, 1, 7, 7)
            new_stem.weight.copy_(mean_w.repeat(1, 2, 1, 1) * (3.0 / 2.0))
        backbone.features.conv0 = new_stem

        # ── Replace classifier head ───────────────────────────────────
        in_ftrs  = backbone.classifier.in_features    # 1024
        backbone.classifier = nn.Linear(in_ftrs, num_classes)
        nn.init.xavier_uniform_(backbone.classifier.weight)
        nn.init.zeros_(backbone.classifier.bias)

        self.backbone    = backbone
        self.num_classes = num_classes

    # ── Freeze / unfreeze API ─────────────────────────────────────────────

    def freeze_for_phase1(self):
        """Phase 1: freeze all features, only train new classifier."""
        for p in self.backbone.features.parameters():
            p.requires_grad_(False)
        for p in self.backbone.classifier.parameters():
            p.requires_grad_(True)

    def unfreeze_for_phase2(self):
        """
        Phase 2: unfreeze denseblock4 + norm5 + classifier.
        Everything before denseblock4 stays frozen.
        """
        # Start fully frozen
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # Unfreeze final dense block, final norm, and classifier
        for name, p in self.backbone.named_parameters():
            if any(name.startswith(f"features.{k}") for k in ("denseblock4", "norm5")):
                p.requires_grad_(True)
            elif name.startswith("classifier"):
                p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    # ── Forward ──────────────────────────────────────────────────────────

    def _trunk(self, x_stft: torch.Tensor) -> torch.Tensor:
        """Extract 1024-dim penultimate features — no interpolation."""
        features = self.backbone.features(x_stft)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)   # (B, 1024)

    def forward(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
        return_fingerprint: bool = False,
    ):
        feat   = self._trunk(x_stft)
        logits = self.backbone.classifier(feat)
        if return_fingerprint:
            return logits, F.normalize(feat, p=2, dim=1)
        return logits

    def extract_embedding(self, x_stft, x_iq, x_if) -> torch.Tensor:
        return self._trunk(x_stft)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
"""
python/src/legacy_models/resnet_18.py
======================================
ResNet-18 baseline for fair comparison against AsymmetricTriNet.

Design decisions:
  - Input: 2-channel STFT at native spatial size (no interpolation).
    ResNet-18 uses adaptive avg-pooling before FC, so any spatial size
    ≥ 32×32 works. No distortion.
  - Weights: ImageNet pretrained.
  - Input stem: Conv1 replaced 3-ch → 2-ch. The 2-ch weights are initialised
    so the expected pre-activation magnitude matches the pretrained 3-ch
    stem on a uniform input: each new channel = mean(RGB) * (3 / 2) = mean * 1.5.
  - Classifier head: backbone.fc replaced with Linear(512, num_classes).
  - Freezing: two-phase.
      Phase 1: freeze all except fc.
      Phase 2: unfreeze layer4 + fc.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class LiteratureBaseline_ResNet18(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ── Adapt input stem: 3-ch → 2-ch ────────────────────────────
        orig_conv = backbone.conv1                     # Conv2d(3, 64, 7, stride=2, padding=3)
        new_stem  = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            mean_w = orig_conv.weight.mean(dim=1, keepdim=True)   # (64, 1, 7, 7)
            new_stem.weight.copy_(mean_w.repeat(1, 2, 1, 1) * (3.0 / 2.0))
        backbone.conv1 = new_stem

        # ── Replace classifier head ───────────────────────────────────
        in_ftrs  = backbone.fc.in_features             # 512
        backbone.fc = nn.Linear(in_ftrs, num_classes)
        nn.init.xavier_uniform_(backbone.fc.weight)
        nn.init.zeros_(backbone.fc.bias)

        self.backbone    = backbone
        self.num_classes = num_classes

    # ── Freeze / unfreeze API ─────────────────────────────────────────────

    def freeze_for_phase1(self):
        """Phase 1: freeze everything except the new classifier head (fc)."""
        for name, p in self.backbone.named_parameters():
            p.requires_grad_(name.startswith("fc."))

    def unfreeze_for_phase2(self):
        """Phase 2: additionally unfreeze layer4 (last residual block)."""
        for name, p in self.backbone.named_parameters():
            p.requires_grad_(
                name.startswith("fc.") or
                name.startswith("layer4.")
            )

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    # ── Forward ──────────────────────────────────────────────────────────

    def _trunk(self, x_stft: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim penultimate features — no interpolation needed."""
        x = self.backbone.conv1(x_stft)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)   # (B, 512)

    def forward(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
        return_fingerprint: bool = False,
    ):
        feat   = self._trunk(x_stft)
        logits = self.backbone.fc(feat)
        if return_fingerprint:
            return logits, F.normalize(feat, p=2, dim=1)
        return logits

    def extract_embedding(self, x_stft, x_iq, x_if) -> torch.Tensor:
        return self._trunk(x_stft)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
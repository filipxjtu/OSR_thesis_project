"""
python/src/legacy_models/vgg_16.py
===================================
VGG-16 baseline for fair comparison against AsymmetricTriNet.

Design decisions:
  - Input: 2-channel STFT (log-magnitude + wrapped phase diff), interpolated
    to 224×224. VGG-16 requires this size due to its fixed FC layers — no
    way around it. Bilinear upsampling is standard practice.
  - Weights: ImageNet pretrained (torchvision weights API, no deprecation).
  - Input stem: original 3-ch Conv replaced with 2-ch Conv. The 2-ch weights
    are initialised so that the *expected pre-activation magnitude* is
    preserved relative to the pretrained 3-ch stem on a uniform input, i.e.
    each new channel is initialised with mean(RGB) * (3 / n_new) = mean * 1.5.
    This is the standard "channel reduction" init from the cross-modal
    transfer literature (Kornia / timm follow the same rule).
  - Classifier head: layers 0-5 of the original classifier are KEPT (they
    are 4096→4096 FC layers that carry learned representations). Only
    layer 6 (4096→1000) is replaced with 4096→num_classes.
  - Freezing: two-phase API.
      Phase 1 (head-only): all features frozen, only classifier[6] trains.
      Phase 2 (last-block): unfreeze features[24:] (last conv block) +
                            full classifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


class LiteratureBaseline_VGG16(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ── Load ImageNet weights (non-deprecated API) ────────────────
        backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # ── Adapt input stem: 3-ch → 2-ch ────────────────────────────
        orig_conv = backbone.features[0]              # Conv2d(3, 64, 3, padding=1)
        new_stem  = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=True)
        with torch.no_grad():
            # Preserve expected pre-activation magnitude on a uniform input:
            # each of the 2 new channels gets mean(RGB) * (3 / 2) = mean * 1.5.
            mean_w = orig_conv.weight.mean(dim=1, keepdim=True)   # (64, 1, 3, 3)
            new_stem.weight.copy_(mean_w.repeat(1, 2, 1, 1) * (3.0 / 2.0))
            if orig_conv.bias is not None:
                new_stem.bias.copy_(orig_conv.bias)
        backbone.features[0] = new_stem

        # ── Replace classifier head (keep FC 0-5, swap FC 6) ─────────
        in_ftrs = backbone.classifier[6].in_features   # 4096
        backbone.classifier[6] = nn.Linear(in_ftrs, num_classes)
        nn.init.xavier_uniform_(backbone.classifier[6].weight)
        nn.init.zeros_(backbone.classifier[6].bias)

        self.backbone    = backbone
        self.num_classes = num_classes

    # ── Freeze / unfreeze API ─────────────────────────────────────────────

    def freeze_for_phase1(self):
        """Phase 1: freeze entire backbone, only train the new head."""
        for p in self.backbone.features.parameters():
            p.requires_grad_(False)
        # Freeze old FC layers too — only new head trains
        for i in range(6):
            for p in self.backbone.classifier[i].parameters():
                p.requires_grad_(False)
        # New head stays trainable
        for p in self.backbone.classifier[6].parameters():
            p.requires_grad_(True)

    def unfreeze_for_phase2(self):
        """Phase 2: unfreeze last conv block (features[24:]) + full classifier."""
        # features[24:] = MaxPool + Conv+ReLU+Conv+ReLU+Conv+ReLU (last block)
        for i, layer in enumerate(self.backbone.features):
            requires = (i >= 24)
            for p in layer.parameters():
                p.requires_grad_(requires)
        for p in self.backbone.classifier.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    # ── Forward ──────────────────────────────────────────────────────────

    def _trunk(self, x_stft: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features (input to classifier[6])."""
        # VGG-16 requires 224×224 — upsample from native STFT spatial size
        x = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        # Run through the original FC layers 0-5 to get the 4096-dim embedding
        for i in range(6):
            x = self.backbone.classifier[i](x)
        return x   # (B, 4096)

    def forward(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
        return_fingerprint: bool = False,
    ):
        feat   = self._trunk(x_stft)
        logits = self.backbone.classifier[6](feat)
        if return_fingerprint:
            return logits, F.normalize(feat, p=2, dim=1)
        return logits

    def extract_embedding(self, x_stft, x_iq, x_if) -> torch.Tensor:
        return self._trunk(x_stft)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
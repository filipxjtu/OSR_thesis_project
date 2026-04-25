from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LiteratureBaseline_DenseNet121(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()

        self.backbone = models.densenet121(pretrained=pretrained)

        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            mean_w = original_conv.weight.mean(dim=1, keepdim=True)
            self.backbone.features.conv0.weight[:] = mean_w.repeat(1, 2, 1, 1) * 0.5

        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_ftrs, num_classes)

    def _trunk(self, x_stft: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            return_fingerprint: bool = False,
    ):
        feat = self._trunk(x_stft)
        logits = self.backbone.classifier(feat)

        if return_fingerprint:
            z = F.normalize(feat, p=2, dim=1)
            return logits, z

        return logits

    def extract_embedding(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> torch.Tensor:
        return self._trunk(x_stft)
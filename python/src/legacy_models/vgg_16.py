from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LiteratureBaseline_VGG16(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()

        self.backbone = models.vgg16(pretrained=pretrained)

        original_conv = self.backbone.features[0]
        self.backbone.features[0] = nn.Conv2d(
            2, 64, kernel_size=3, stride=1, padding=1
        )
        with torch.no_grad():
            mean_w = original_conv.weight.mean(dim=1, keepdim=True)
            self.backbone.features[0].weight[:] = mean_w.repeat(1, 2, 1, 1) * 0.5
            if original_conv.bias is not None and self.backbone.features[0].bias is not None:
                self.backbone.features[0].bias[:] = original_conv.bias

        num_ftrs = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def _trunk(self, x_stft: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(6):
            x = self.backbone.classifier[i](x)
        return x

    def forward(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            return_fingerprint: bool = False,
    ):
        feat = self._trunk(x_stft)
        logits = self.backbone.classifier[6](feat)

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
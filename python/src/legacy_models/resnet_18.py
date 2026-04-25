from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LiteratureBaseline_ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()

        self.backbone = models.resnet18(pretrained=pretrained)

        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            mean_w = original_conv.weight.mean(dim=1, keepdim=True)
            self.backbone.conv1.weight[:] = mean_w.repeat(1, 2, 1, 1) * 0.5

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _trunk(self, x_stft: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            return_fingerprint: bool = False,
    ):
        feat = self._trunk(x_stft)
        logits = self.backbone.fc(feat)

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
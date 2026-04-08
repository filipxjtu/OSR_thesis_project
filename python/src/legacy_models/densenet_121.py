import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LiteratureBaseline_DenseNet121(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # 1. Load standard DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)

        # 2. Modify the first convolutional layer (from 3 channels to 1 channel)
        # DenseNet121's first layer is self.backbone.features.conv0
        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            self.backbone.features.conv0.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

        # 3. Modify the final classifier
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        # Ignore IQ. Stretch STFT to 224x224
        x_stft_resized = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x_stft_resized)

    def extract_embedding(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        # Ignore IQ. Stretch STFT.
        x = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass through dense blocks
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        return out
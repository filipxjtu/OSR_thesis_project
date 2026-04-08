import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LiteratureBaseline_ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        # 1. Load standard ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # 2. Modify the first convolutional layer
        # ResNet expects 3-channel RGB. Your STFT is 1-channel (log_mag).
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Preserve the pre-trained edge-detection capability
        with torch.no_grad():
            self.backbone.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

        # 3. Modify the final classifier
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        # Ignore IQ. Stretch STFT to 224x224 to prevent spatial collapse
        x_stft_resized = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x_stft_resized)

    def extract_embedding(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        # CRITICAL FIX: Stretch STFT here as well!
        x = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass through the ResNet blocks
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)  # Ignore IDE warning here, this is valid PyTorch
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
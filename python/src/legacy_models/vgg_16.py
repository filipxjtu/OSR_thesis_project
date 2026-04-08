import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LiteratureBaseline_VGG16(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # 1. Load standard VGG16
        self.backbone = models.vgg16(pretrained=pretrained)

        # 2. Modify the first convolutional layer (from 3 channels to 1 channel)
        # VGG16's first layer is accessed via self.backbone.features[0]
        original_conv = self.backbone.features[0]
        self.backbone.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1
        )

        # Preserve pre-trained weights by averaging the RGB channels into one
        with torch.no_grad():
            self.backbone.features[0].weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
            if original_conv.bias is not None:
                self.backbone.features[0].bias[:] = original_conv.bias

        # 3. Modify the final classifier
        # VGG16's final linear layer is accessed via self.backbone.classifier[6]
        num_ftrs = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        # Ignore IQ. Stretch STFT to 224x224 to prevent spatial collapse
        x_stft_resized = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x_stft_resized)

    def extract_embedding(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        # Ignore IQ. Stretch STFT.
        x = F.interpolate(x_stft, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass through conv layers
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Pass through the first two linear layers of the classifier (stop before the final 10-class output)
        for i in range(6):
            x = self.backbone.classifier[i](x)

        return x
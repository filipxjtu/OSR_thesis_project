from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Minimal closed-set CNN for STFT feature tensors.
    Expected input shape: (N, 1, F, T)
    Output: logits of shape (N, num_classes)
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (N,1,F,T), got {tuple(x.shape)}")

        z = self.features(x)
        z = torch.flatten(z, start_dim=1)

        logits = self.classifier(z)

        return logits

    def extract_embedding(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        return x
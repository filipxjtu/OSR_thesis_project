from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.stft_branch = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.iq_branch = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.if_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(192, num_classes),
        )

    def _forward_features(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> torch.Tensor:
        z_stft = torch.flatten(self.stft_branch(x_stft), start_dim=1)
        z_iq   = torch.flatten(self.iq_branch(x_iq),     start_dim=1)
        z_if   = torch.flatten(self.if_branch(x_if),     start_dim=1)
        return torch.cat([z_stft, z_iq, z_if], dim=1)

    def forward(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            return_fingerprint: bool = False,
    ):
        if x_stft.ndim != 4 or x_stft.shape[1] != 2:
            raise ValueError(f"SimpleCNN: Expected x_stft (N,2,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 3:
            raise ValueError(f"SimpleCNN: Expected x_iq (N,3,L), got {tuple(x_iq.shape)}")
        if x_if.ndim != 3 or x_if.shape[1] != 1:
            raise ValueError(f"SimpleCNN: Expected x_if (N,1,L), got {tuple(x_if.shape)}")

        fused = self._forward_features(x_stft, x_iq, x_if)
        logits = self.classifier(fused)

        if return_fingerprint:
            z = F.normalize(fused, p=2, dim=1)
            return logits, z

        return logits

    def extract_embedding(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_features(x_stft, x_iq, x_if)
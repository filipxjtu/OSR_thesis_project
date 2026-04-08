from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Minimal dual-branch CNN for Multi-Domain RF features.
    Expected input: x_stft (N, 1, F, T), x_iq (N, 2, L)
    Output: logits of shape (N, num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # STREAM 1: Semantic Branch (STFT)
        self.stft_branch = nn.Sequential(
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

        # STREAM 2: Physics Branch (IQ)
        self.iq_branch = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        # FUSION & CLASSIFIER
        # 64 (STFT) + 64 (IQ) = 128
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def _forward_features(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        # 1. Process STFT
        z_stft = self.stft_branch(x_stft)
        z_stft = torch.flatten(z_stft, start_dim=1)  # (N, 64)

        # 2. Process IQ
        z_iq = self.iq_branch(x_iq)
        z_iq = torch.flatten(z_iq, start_dim=1)  # (N, 64)

        # 3. Early Late-Fusion
        fused_embedding = torch.cat([z_stft, z_iq], dim=1)  # (N, 128)

        return fused_embedding

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:

        # Strict sanity checks for the new dataloader
        if x_stft.ndim != 4 or x_stft.shape[1] != 1:
            raise ValueError(f"SimpleCNN: Expected x_stft (N,1,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"SimpleCNN: Expected x_iq (N,2,L), got {tuple(x_iq.shape)}")

        # Extract features and classify
        fused_embedding = self._forward_features(x_stft, x_iq)
        logits = self.classifier(fused_embedding)

        return logits

    def extract_embedding(self, x_stft: torch.Tensor, x_iq: torch.Tensor) -> torch.Tensor:
        """ Returns the 128D fused feature vector for t-SNE plotting """
        return self._forward_features(x_stft, x_iq)
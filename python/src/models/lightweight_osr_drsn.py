from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Lightweight building blocks ---

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ECA2d(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)                     # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)      # [B,1,C]
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)    # [B,C,1,1]
        return x * self.sigmoid(y)


class LightweightDRSNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.abs_mean_pool = nn.AdaptiveAvgPool2d(1)
        self.thresh_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        abs_out = torch.abs(out)
        channel_means = self.abs_mean_pool(abs_out)

        scales = channel_means.squeeze(-1).transpose(-1, -2)
        scales = self.thresh_conv(scales).transpose(-1, -2).unsqueeze(-1)

        alpha = self.sigmoid(scales)
        thres = alpha * channel_means

        out = torch.sign(out) * torch.relu(abs_out - thres)
        out = out + residual
        return F.relu(out)


# --- 1D Lightweight Physics Branch ---

class LightweightIQBranch(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, out_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1)


# --- TWO-STREAM LIGHTWEIGHT OSR MODEL ---

class Lightweight_OSR_DRSN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        n_iterations: int = 3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.n_iterations = n_iterations

        # --- STREAM 1: Semantic Backbone (STFT, 1 Channel) ---
        self.stem_stft = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.layer1 = LightweightDRSNBlock(16, 32, stride=2)
        self.layer2 = LightweightDRSNBlock(32, 64, stride=2)
        self.layer3 = LightweightDRSNBlock(64, 128, stride=2)

        self.attention = ECA2d(k_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- STREAM 2: Physics Backbone (IQ, 2 Channels) ---
        self.iq_branch = LightweightIQBranch(out_features=128)

        # --- FUSION & OSR COMPONENTS ---
        # 128D (STFT) + 128D (IQ) = 256D
        self.classifier = nn.Linear(256, num_classes)

        # Iterative refinement on the fused 256D vector
        self.refiner = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        )

    def extract_features(self, x_stft, x_iq):
        # 1. Process STFT
        stft_out = self.stem_stft(x_stft)
        stft_out = self.layer1(stft_out)
        stft_out = self.layer2(stft_out)
        stft_out = self.layer3(stft_out)

        stft_out = self.attention(stft_out)
        stft_out = self.pool(stft_out)
        stft_flat = torch.flatten(stft_out, 1)

        # 2. Process IQ
        iq_flat = self.iq_branch(x_iq)

        # 3. Fuse
        fused_features = torch.cat([stft_flat, iq_flat], dim=1)
        return fused_features

    def forward_with_osr(self, x_stft, x_iq):
        if x_stft.ndim != 4 or x_stft.shape[1] != 1:
            raise ValueError(f"Expected x_stft (N,1,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq (N,2,L), got {tuple(x_iq.shape)}")

        features = self.extract_features(x_stft, x_iq)

        # Apply iterative refinement to the 256D vector
        for i in range(self.n_iterations - 1):
            features = features + self.refiner(features)

        logits = self.classifier(features)

        # Simple Unknown Scoring (1 - max probability)
        probs = F.softmax(logits, dim=-1)
        unknown_score = 1.0 - probs.max(dim=-1)[0]

        return logits, unknown_score, None

    def forward(self, x_stft, x_iq):
        logits, _, _ = self.forward_with_osr(x_stft, x_iq)
        return logits

    def extract_embedding(self, x_stft, x_iq):
        return self.extract_features(x_stft, x_iq)

    def freeze_base(self):
        for p in self.stem_stft.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = False
        for p in self.layer3.parameters():
            p.requires_grad = False
        for p in self.iq_branch.parameters():
            p.requires_grad = False

    def unfreeze_base(self):
        for p in self.parameters():
            p.requires_grad = True
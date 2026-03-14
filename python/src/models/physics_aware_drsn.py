from __future__ import annotations

import torch
import torch.nn as nn


class SoftThreshold(nn.Module):
    """ channel-wise soft thresholding. """

    def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:

        abs_x = torch.abs(x)

        sub = abs_x - threshold
        zeros = torch.zeros_like(sub)

        n_sub = torch.max(sub, zeros)

        return torch.sign(x) * n_sub

class ResidualShrinkageBlockCW(nn.Module):
    """
    residual shrinkage block with channel-wise thresholds
    |x| global pooling
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):

        super().__init__()

        # residual branch
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # shrinkage layers
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(out_channels, out_channels // 4)
        self.fc2 = nn.Linear(out_channels // 4, out_channels)

        self.sigmoid = nn.Sigmoid()

        self.soft_threshold = SoftThreshold()

        self.last_threshold = None

        # skip connection
        self.skip = nn.Identity()

        if stride != 1 or in_channels != out_channels:

            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # ----- shrinkage mechanism -----

        abs_out = torch.abs(out)

        gap = self.gap(abs_out).view(out.size(0), -1)

        scale = self.fc1(gap)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)

        threshold = gap * scale
        threshold = threshold.unsqueeze(2).unsqueeze(3)

        self.last_threshold = threshold.detach()

        out = self.soft_threshold(out, threshold)

        # ----- residual connection -----

        out = out + identity
        out = self.relu(out)

        return out


class TrajectoryBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=2, padding=(2, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PhysicsAwareDRSN(nn.Module):

    def __init__(self, num_classes: int = 7):

        super().__init__()

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # residual shrinkage stages
        self.layer1 = nn.Sequential(
            ResidualShrinkageBlockCW(32, 32),
            ResidualShrinkageBlockCW(32, 32),
        )

        self.layer2 = nn.Sequential(
            ResidualShrinkageBlockCW(32, 64, stride=2),
            ResidualShrinkageBlockCW(64, 64),
        )

        self.layer3 = nn.Sequential(
            ResidualShrinkageBlockCW(64, 128, stride=2),
            ResidualShrinkageBlockCW(128, 128),
        )

        self.trajectory_branch = TrajectoryBranch()
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(f"Expected (N,1,F,T), got {tuple(x.shape)}")

        stem_feat = self.stem(x)

        #DRSN branch
        drsn_feat = self.layer1(stem_feat)
        drsn_feat = self.layer2(drsn_feat)
        drsn_feat = self.layer3(drsn_feat)

        #trajectory branch
        traj_feat = self.trajectory_branch(stem_feat)

        #fusion
        fused = torch.cat([drsn_feat, traj_feat], dim=1)
        fused = self.fusion(fused)

        fused = self.pool(fused)

        fused = torch.flatten(fused, 1)

        logits = self.classifier(fused)

        return logits


    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        stem_feat = self.stem(x)

        drsn_feat = self.layer1(stem_feat)
        drsn_feat = self.layer2(drsn_feat)
        drsn_feat = self.layer3(drsn_feat)

        traj_feat = self.trajectory_branch(stem_feat)

        fused = torch.cat([drsn_feat, traj_feat], dim=1)
        fused = self.fusion(fused)

        fused = self.pool(fused)
        fused = torch.flatten(fused, 1)

        return fused
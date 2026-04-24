from __future__ import annotations

import torch
import torch.nn as nn


class SoftThreshold(nn.Module):
    """ channel-wise soft thresholding. """

    def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        sub = abs_x - threshold
        n_sub = torch.clamp(sub, min=0)
        return torch.sign(x) * n_sub


class MultiScaleConv(nn.Module):
    """ local branch (3x3),  wider branch (dilated 3x3), projection branch (1x1) """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        branch_channels = max(out_channels // 2, 8)

        self.branch_local = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_pointwise = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_local(x)
        b2 = self.branch_dilated(x)
        b3 = self.branch_pointwise(x)
        fused = torch.cat([b1, b2, b3], dim=1)
        return self.fuse(fused)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=(2, 3))  # GAP
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y


class ResidualShrinkageBlockCW(nn.Module):
    """ residual shrinkage block with 2D spatial variance tracking """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        hidden = max(out_channels // 4, 8)

        self.conv1 = MultiScaleConv(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, hidden)
        self.fc2 = nn.Linear(hidden, out_channels)
        self.noise_fc = nn.Linear(out_channels, out_channels)
        self.var_fc1 = nn.Linear(out_channels, hidden)
        self.var_fc2 = nn.Linear(hidden, out_channels)

        self.sigmoid = nn.Sigmoid()
        self.soft_threshold = SoftThreshold()

        self.last_threshold: torch.Tensor | None = None
        self.last_gap: torch.Tensor | None = None
        self.last_var_stat: torch.Tensor | None = None

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        abs_out = torch.abs(out)
        gap = self.gap(abs_out).view(out.size(0), -1)

        spatial_mean = out.mean(dim=(2, 3), keepdim=True)
        spatial_var = ((out - spatial_mean) ** 2).mean(dim=(2, 3))

        scale = self.relu(self.fc1(gap))
        scale = self.sigmoid(self.fc2(scale))

        noise_gate = torch.sigmoid(self.noise_fc(gap))
        var_gate = torch.relu(self.var_fc1(spatial_var))
        var_gate = torch.sigmoid(self.var_fc2(var_gate))

        threshold = scale * gap * (noise_gate + var_gate) / 2
        threshold = threshold.unsqueeze(2).unsqueeze(3)

        self.last_threshold = threshold.detach()
        self.last_gap = gap.detach()
        self.last_var_stat = spatial_var.detach()

        out = self.soft_threshold(out, threshold)
        out = out + identity
        return self.relu(out)


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


class IQPhysicsBranch(nn.Module):
    """ 1D CNN to extract micro-phase and timing information directly from complex IQ """

    def __init__(self, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            # Extract high-frequency phase features (in_channels=2 for I and Q)
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # Mid-level features
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # High-level representations
            nn.Conv1d(64, out_features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),

            # Global Average Pooling to 1D Vector
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(out.size(0), -1)


class IFPhysicsBranch(nn.Module):
    """ 1D CNN to extract features directly from Instantaneous Frequency """

    def __init__(self, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            # Extract phase-derivative features (in_channels=1 for IF)
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # Mid-level features
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # High-level representations
            nn.Conv1d(64, out_features, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),

            # Global Average Pooling to 1D Vector
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(out.size(0), -1)


class TS_MS_VA_DRSN(nn.Module):
    """
    Multi-Domain Tri-Branch Fusion Network:
      - STREAM 1 (STFT): Multi-Scale 2D DRSN + Trajectory Branch
      - STREAM 2 (IQ): 1D Physics CNN
      - STREAM 3 (IF): 1D Physics CNN for Phase Derivatives
      - Deep Late-Fusion -> Classifier
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # STREAM 1: Semantic Branch (STFT)
        self.stft_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

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

        self.stft_fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))

        # STREAM 2: Physics Branch (IQ)
        self.iq_branch = IQPhysicsBranch(out_features=128)

        # STREAM 3: Instantaneous Frequency Branch (IF)
        self.if_branch = IFPhysicsBranch(out_features=128)

        # MULTI-DOMAIN FUSION -> 128D (STFT) + 128D (IQ) + 128D (IF) = 384D Feature Vector
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Prevent overfitting on the fused vector
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def _forward_backbone(self, x_stft: torch.Tensor, x_iq: torch.Tensor, x_if: torch.Tensor) -> torch.Tensor:
        # 1. Process STFT Stream
        stft_stem_feat = self.stft_stem(x_stft)
        drsn_feat = self.layer3(self.layer2(self.layer1(stft_stem_feat)))
        traj_feat = self.trajectory_branch(stft_stem_feat)

        stft_fused = torch.cat([drsn_feat, traj_feat], dim=1)
        stft_emb = self.pool2d(self.stft_fusion(stft_fused))
        stft_emb_flat = torch.flatten(stft_emb, 1)  # (Batch, 128)

        # 2. Process Raw IQ Stream
        iq_emb_flat = self.iq_branch(x_iq)  # (Batch, 128)

        # 3. Process Instantaneous Frequency Stream
        if_emb_flat = self.if_branch(x_if)  # (Batch, 128)

        # 4. Concatenate Domains (384D)
        multi_domain_emb = torch.cat([stft_emb_flat, iq_emb_flat, if_emb_flat], dim=1)

        return multi_domain_emb

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor, x_if: torch.Tensor) -> torch.Tensor:

        if x_stft.ndim != 4 or x_stft.shape[1] != 1:
            raise ValueError(f"Expected x_stft (N,1,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq (N,2,1024), got {tuple(x_iq.shape)}")
        if x_if.ndim != 3 or x_if.shape[1] != 1:
            raise ValueError(f"Expected x_if (N,1,1024), got {tuple(x_if.shape)}")

        embedding = self._forward_backbone(x_stft, x_iq, x_if)
        return self.classifier(embedding)

    def extract_embedding(self, x_stft: torch.Tensor, x_iq: torch.Tensor, x_if: torch.Tensor) -> torch.Tensor:
        """ Returns the 384D Multi-Domain Fused Vector for t-SNE plotting """
        return self._forward_backbone(x_stft, x_iq, x_if)
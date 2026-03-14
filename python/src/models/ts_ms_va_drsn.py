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

        y = x.mean(dim=(2,3))      #GAP
        y = self.fc(y)
        y = y.view(b, c, 1, 1)

        return x * y

class ResidualShrinkageBlockCW(nn.Module):
    """ residual shrinkage block with
          - channel-wise thresholds,
          - multiscale first convolution
          - noise-aware + variance-aware threshold modulation
          - |x| global pooling
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        hidden = max(out_channels // 4, 8)

        # residual branch
        self.conv1 = MultiScaleConv(in_channels, out_channels, stride=stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.se = SEBlock(out_channels)

        # shrinkage layers
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

        # skip connection
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

        #shrinkage mechanism
        abs_out = torch.abs(out)
        gap = self.gap(abs_out).view(out.size(0), -1)

        spatial_mean = out.mean(dim=(2, 3), keepdim=True)
        spatial_var = ((out - spatial_mean) ** 2).mean(dim=(2, 3))

        scale = self.relu(self.fc1(gap))
        scale = self.sigmoid(self.fc2(scale))


        noise_gate = torch.sigmoid(self.noise_fc(gap))
        var_gate = torch.relu(self.var_fc1(spatial_var))
        var_gate = torch.sigmoid(self.var_fc2(var_gate))

        threshold = scale * gap * (noise_gate + var_gate)/2
        threshold = threshold.unsqueeze(2).unsqueeze(3)

        self.last_threshold = threshold.detach()
        self.last_gap = gap.detach()
        self.last_var_stat = spatial_var.detach()

        out = self.soft_threshold(out, threshold)

        #residual connection
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


class TS_MS_VA_DRSN(nn.Module):
    """
    physics aware drsn network with:
      - multiscale residual shrinkage backbone
      - variance aware threshold
      - trajectory branch
      - late fusion
    """

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

        #trajectory branch
        self.trajectory_branch = TrajectoryBranch()

        #fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)


    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:

        stem_feat = self.stem(x)

        # DRSN branch
        drsn_feat = self.layer3(self.layer2(self.layer1(stem_feat)))

        # trajectory branch
        traj_feat = self.trajectory_branch(stem_feat)

        # fusion
        fused = torch.cat([drsn_feat, traj_feat], dim=1)
        fused = self.pool(self.fusion(fused))
        return torch.flatten(fused, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(f"Expected (N,1,F,T), got {tuple(x.shape)}")

        embedding = self._forward_backbone(x)
        return  self.classifier(embedding)


    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_backbone(x)
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
AsymmetricTriNet — Noise-robust, cubic-thresholded, tri-modal classifier.
"""


def _cubic_threshold(x: torch.Tensor, tau: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # FIX: ref is a magnitude statistic; detach so grads don't flow back through mean(|x|)
    threshold = tau * ref.detach()
    abs_x = torch.abs(x)

    inside = (abs_x < threshold)
    y_outside = x - torch.sign(x) * (2.0 / 3.0) * threshold

    safe_thr = threshold.clamp(min=1e-4)
    y_inside = (x ** 3) / (3.0 * safe_thr ** 2)

    return torch.where(inside, y_inside, y_outside)


class CoordAtt2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = F.relu(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))), inplace=True)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


class SKBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv_fast = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        self.conv_slow = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        mip = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mip, bias=False),
            nn.LayerNorm(mip), nn.ReLU(inplace=True),
            nn.Linear(mip, channels * 2, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        u_fast, u_slow = self.conv_fast(x), self.conv_slow(x)
        s = (u_fast + u_slow).mean(dim=-1)
        z = self.fc(s).view(b, 2, c)
        a = F.softmax(z, dim=1)
        return u_fast * a[:, 0].unsqueeze(-1) + u_slow * a[:, 1].unsqueeze(-1)


class InlineDRSN1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self.gate(x).unsqueeze(-1)
        ref = torch.abs(x).mean(dim=-1, keepdim=True)
        return _cubic_threshold(x, tau, ref)


class InlineDRSN2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        ref = torch.abs(x).mean(dim=(2, 3), keepdim=True)
        return _cubic_threshold(x, tau, ref)


class FusedDRSN(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self.fc(x)
        ref = torch.abs(x).mean(dim=1, keepdim=True)
        return _cubic_threshold(x, tau, ref)


class TrajectoryBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=2, padding=(2, 0), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            InlineDRSN2D(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            InlineDRSN2D(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EnhancedSTFTBranch(nn.Module):
    def __init__(self, branch_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.coord_path = nn.Sequential(
            CoordAtt2D(32),
            InlineDRSN2D(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            CoordAtt2D(64),
            InlineDRSN2D(64)
        )

        self.trajectory_path = TrajectoryBranch()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.compress = nn.Sequential(
            nn.Linear(192, branch_dim, bias=False),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_features = self.stem(x)

        coord_feat = self.coord_path(stem_features)
        traj_feat = self.trajectory_path(stem_features)

        coord_flat = torch.flatten(self.pool(coord_feat), 1)
        traj_flat = torch.flatten(self.pool(traj_feat), 1)

        merged = torch.cat([coord_flat, traj_flat], dim=1)
        return self.compress(merged)


class ModalityReliabilityGate(nn.Module):
    def __init__(self, branch_dim: int, num_branches: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(branch_dim * num_branches, branch_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(branch_dim, num_branches, bias=False)
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        combined = torch.cat(features, dim=-1)
        weights = torch.sigmoid(self.fc(combined))
        return [f * weights[:, i].unsqueeze(-1) for i, f in enumerate(features)]


class AsymmetricTriNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            branch_dim: int = 128,
            fingerprint_dim: int = 256,
            modality_dropout: float = 0.1,
            num_transformer_layers: int = 2,
            nhead: int = 4,
            use_cls_token: bool = True,
            supcon_dim: int = 128,
    ):
        super().__init__()
        self.mod_drop_p = modality_dropout
        self.branch_dim = branch_dim
        self.fingerprint_dim = fingerprint_dim
        self.use_cls_token = use_cls_token

        self.stft_branch = EnhancedSTFTBranch(branch_dim=branch_dim)

        self.iq_branch = nn.Sequential(
            nn.Conv1d(3, 32, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            SKBlock1D(32),
            InlineDRSN1D(32),

            nn.Conv1d(32, 64, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            SKBlock1D(64),
            InlineDRSN1D(64),

            nn.Conv1d(64, branch_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(branch_dim), nn.ReLU(inplace=True),
            InlineDRSN1D(branch_dim),
            nn.AdaptiveAvgPool1d(1)
        )

        self.if_branch = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            SKBlock1D(32),
            InlineDRSN1D(32),

            nn.Conv1d(32, 64, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            SKBlock1D(64),
            InlineDRSN1D(64),

            nn.Conv1d(64, branch_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(branch_dim), nn.ReLU(inplace=True),
            InlineDRSN1D(branch_dim),
            nn.AdaptiveAvgPool1d(1)
        )

        self.reliability_gate = ModalityReliabilityGate(branch_dim, 3)
        self.modality_embed = nn.Parameter(torch.randn(1, 3, branch_dim) * 0.02)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, branch_dim) * 0.02)

        self.transformer_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=branch_dim,
                nhead=nhead,
                dim_feedforward=branch_dim * 2,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )

        fused_dim = branch_dim if use_cls_token else branch_dim * 3
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fused_dim, fingerprint_dim, bias=False),
            nn.BatchNorm1d(fingerprint_dim),
        )
        self.drsn_refiner = FusedDRSN(dim=fingerprint_dim, reduction=4)

        # NEW: SupCon projection head — MLP to a smaller L2-normalized embedding space
        # Only used during training for the contrastive loss, not for classification
        self.supcon_head = nn.Sequential(
            nn.Linear(fingerprint_dim, fingerprint_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fingerprint_dim, supcon_dim, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fingerprint_dim, num_classes),
        )

    def _modality_dropout(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.training or self.mod_drop_p <= 0:
            return feats
        n = len(feats)
        mask = torch.bernoulli(torch.full((n,), 1.0 - self.mod_drop_p, device=feats[0].device))
        if mask.sum() == 0:
            mask[torch.randint(n, (1,))] = 1.0
        scale = float(n) / mask.sum()
        return [f * mask[i] * scale for i, f in enumerate(feats)]

    def _fuse(self, f1, f2, f3):
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)
        f3 = F.normalize(f3, p=2, dim=1)

        f1, f2, f3 = self.reliability_gate([f1, f2, f3])

        tokens = torch.stack([f1, f2, f3], dim=1) + self.modality_embed

        if self.use_cls_token:
            b = tokens.size(0)
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            out = self.transformer_fusion(tokens)
            fused = out[:, 0]
        else:
            fused = torch.flatten(self.transformer_fusion(tokens), 1)

        fp = self.fingerprint_proj(fused)
        fp = self.drsn_refiner(fp)
        return fp

    def forward(self, x_stft: torch.Tensor, x_iq: torch.Tensor, x_if: torch.Tensor,
                return_fingerprint: bool = False):
        # NEW: return_fingerprint=True returns (logits, supcon_projection) for contrastive training
        f1 = self.stft_branch(x_stft)
        f2 = torch.flatten(self.iq_branch(x_iq), 1)
        f3 = torch.flatten(self.if_branch(x_if), 1)

        f1, f2, f3 = self._modality_dropout([f1, f2, f3])

        fp = self._fuse(f1, f2, f3)
        logits = self.classifier(fp)

        if return_fingerprint:
            z = F.normalize(self.supcon_head(fp), p=2, dim=1)
            return logits, z

        return logits

    @torch.no_grad()
    def extract_fingerprint(self, x_stft: torch.Tensor, x_iq: torch.Tensor, x_if: torch.Tensor) -> torch.Tensor:
        f1 = self.stft_branch(x_stft)
        f2 = torch.flatten(self.iq_branch(x_iq), 1)
        f3 = torch.flatten(self.if_branch(x_if), 1)
        return self._fuse(f1, f2, f3)

    @torch.no_grad()
    def extract_embedding(self, x_stft, x_iq, x_if) -> torch.Tensor:
        return self.extract_fingerprint(x_stft, x_iq, x_if)
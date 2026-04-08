from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from .ts_ms_va_drsn import TS_MS_VA_DRSN, ResidualShrinkageBlockCW


class OSRFeatureCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.entropies = []
        self.shrinkage_ratios = []
        self.variances = []

    def capture(self, block: ResidualShrinkageBlockCW):
        if block.last_threshold is not None:
            thresholds = block.last_threshold.squeeze()

            thresh_norm = thresholds / (thresholds.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(thresh_norm * torch.log(thresh_norm + 1e-8)).sum(dim=1)

            self.entropies.append(entropy)

            if block.last_gap is not None:
                gap = block.last_gap
                shrinkage = (gap < thresholds).float().mean(dim=1)
                self.shrinkage_ratios.append(shrinkage)

            if block.last_var_stat is not None:
                variance = block.last_var_stat.mean(dim=1)
                self.variances.append(variance)

    def aggregate(self, device):
        if not self.entropies:
            return (
                torch.zeros(1, 1, device=device),
                torch.zeros(1, 1, device=device),
                torch.zeros(1, 1, device=device),
            )

        ent = torch.stack(self.entropies).mean(dim=0, keepdim=True).T

        shr = (
            torch.stack(self.shrinkage_ratios).mean(dim=0, keepdim=True).T
            if self.shrinkage_ratios
            else torch.zeros_like(ent)
        )

        var = (
            torch.stack(self.variances).mean(dim=0, keepdim=True).T
            if self.variances
            else torch.zeros_like(ent)
        )

        return ent, shr, var


class IterativeOSR_TS_MS_VA_DRSN(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            n_iterations: int = 3,
            hidden_dim: int = 256,  # Updated to 256 (128 STFT + 128 IQ)
            dropout: float = 0.3,
            use_pretrained: bool = False,
            pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.n_iterations = n_iterations

        self.base = TS_MS_VA_DRSN(num_classes=num_classes)

        if use_pretrained and pretrained_path:
            self.base.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

        # Onsager correction network (operates on the 256D fused embedding)
        self.onsager = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_iterations - 1)
        ])

        # Input Modulation - STFT Stream (1 Channel, 2D)
        self.input_mod_stft = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, 3, padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_iterations - 1)
        ])

        # Input Modulation - IQ Stream (2 Channels, 1D)
        self.input_mod_iq = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2, 2, 5, padding=2, bias=False),
                nn.BatchNorm1d(2),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_iterations - 1)
        ])

        # OSR Meta Classifier
        self.osr_meta = nn.Sequential(
            nn.Linear(n_iterations * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def _capture_osr(self, x_stft, x_iq):
        collector = OSRFeatureCollector()
        hooks = []

        for m in self.base.modules():
            if isinstance(m, ResidualShrinkageBlockCW):
                hooks.append(m.register_forward_hook(lambda _, __, ___, b=m: collector.capture(b)))

        _ = self.base.extract_embedding(x_stft, x_iq)

        for h in hooks:
            h.remove()

        return collector.aggregate(x_stft.device)

    def _onsager(self, z, t):
        if t >= len(self.onsager):
            return z
        return z + 0.1 * self.onsager[t](z)

    def _modulate(self, x_stft, x_iq, z, t):
        if t >= len(self.input_mod_stft):
            return x_stft, x_iq

        # Calculate dynamic modulation scale based on the fused embedding
        z_scale = torch.sigmoid(z.mean(dim=1))

        scale_stft = z_scale.view(-1, 1, 1, 1) * 0.1
        scale_iq = z_scale.view(-1, 1, 1) * 0.1

        # Modulate both domains
        mod_stft = self.input_mod_stft[t](x_stft) + scale_stft
        mod_iq = self.input_mod_iq[t](x_iq) + scale_iq

        return mod_stft, mod_iq

    def forward_with_osr(self, x_stft, x_iq):
        if x_stft.ndim != 4 or x_stft.shape[1] != 1:
            raise ValueError(f"Expected x_stft (N,1,F,T), got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq (N,2,L), got {tuple(x_iq.shape)}")

        B = x_stft.size(0)

        feats = []
        prev = None

        for t in range(self.n_iterations):
            # Capture block dynamics
            ent, shr, var = self._capture_osr(x_stft, x_iq)

            # Extract current state embedding
            emb = self.base.extract_embedding(x_stft, x_iq)

            if t > 0:
                res = torch.norm(emb - prev, dim=1, keepdim=True)
            else:
                res = torch.zeros(B, 1, device=x_stft.device)

            feats.append(torch.cat([ent, shr, var, res], dim=1))

            if t < self.n_iterations - 1:
                z = self._onsager(emb, t)
                x_stft, x_iq = self._modulate(x_stft, x_iq, z, t)
                prev = z

        stacked = torch.stack(feats, dim=1)
        flat = stacked.view(B, -1)

        unknown_score = self.osr_meta(flat).squeeze(1)
        logits = self.base.classifier(emb) / self.temperature

        return logits, unknown_score, None

    def forward(self, x_stft, x_iq):
        logits, _, _ = self.forward_with_osr(x_stft, x_iq)
        return logits

    def extract_embedding(self, x_stft, x_iq):
        for t in range(self.n_iterations):
            z = self.base.extract_embedding(x_stft, x_iq)
            if t < self.n_iterations - 1:
                x_stft, x_iq = self._modulate(x_stft, x_iq, self._onsager(z, t), t)
        return z
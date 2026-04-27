from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
AsymmetricTriNet — noise-robust, cubic-thresholded, tri-modal classifier.

Revision notes vs. the prior implementation:

  1. DRSN blocks are now *actually residual*: out = x + cubic_threshold(F(x)).
     The previous InlineDRSN was a pure thresholding transform, which silently
     contradicted the formulation Zhao et al. cite (and that the thesis also
     cites). The "R" in DRSN now means what it says.

  2. The 1D branches (IQ, IF) now have explicit residual shortcuts around each
     conv stage: y = ReLU(BN(Conv(x)) + 1x1_proj(x)). This is a single-conv
     residual stage, not a two-conv ResNet basic block — chosen so we get the
     gradient-highway / regularization benefit of skip connections without
     doubling the conv count and inflating params.

  3. One transformer layer instead of two. With four tokens (CLS + three
     modalities), one MHA layer is enough cross-modal interaction. Saves
     ~130k params in a model that's overfitting.

  4. One DRSN per branch, at the deepest (widest-channel) stage. Early-stage
     thresholding suppresses still-shallow features where it doesn't make
     sense; late-stage thresholding operates on the abstract features the
     thesis story actually applies to.

  5. SK blocks moved to the lightest (32-channel) stage only. Multi-scale
     receptive field selection is most valuable on shallow features where
     the kernel size matters; on deep features it's redundant given the
     transformer-fusion already mixes scales globally.

  6. Stochastic depth (Huang et al. 2016) on the residual blocks. Drop-prob
     ramps linearly across four depth levels: 0.0 (shallow 1D residual stages)
     → sd_max (fingerprint refinement). Parallel branches (IQ/IF) and the
     two STFT DRSNs share their respective depth indices, since they sit at
     the same depth in the network. Off at inference. Conservative max=0.10
     for our model size; we can push higher if overfitting persists.

  7. Cosine classifier (NormFace-style) replaces the bare Linear head. Logits
     become s · cos(W, fp), which makes ||fp|| a useful confidence signal —
     directly relevant to the OSR calibrator that reads emb_norm.

  8. Default modality dropout bumped 0.1 → 0.25 to actually exercise all three
     branches during training (was almost never firing at 0.1).

Educated-guess hyperparameters — flagged so you can retune later:
  - stochastic_depth_max_p = 0.10  (literature recipes use 0.2–0.5; ours is small)
  - modality_dropout = 0.25         (literature uses 0.15–0.30 for tri-modal)
  - cosine_scale  = 16              (NormFace/CosFace literature range 8–64)
  - cosine_margin = 0.0             (pure NormFace; no margin, conservative)
"""


# =============================================================================
# Cubic threshold core (unchanged math)
# =============================================================================

def _cubic_threshold(x: torch.Tensor, tau: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    threshold = tau * ref.detach()
    abs_x = torch.abs(x)

    inside = (abs_x < threshold)
    y_outside = x - torch.sign(x) * (2.0 / 3.0) * threshold

    safe_thr = threshold.clamp(min=1e-4)
    y_inside = (x ** 3) / (3.0 * safe_thr ** 2)

    return torch.where(inside, y_inside, y_outside)


# =============================================================================
# Stochastic depth — drop entire residual branches at training time
# =============================================================================

def stochastic_drop_path(
        x: torch.Tensor,
        drop_p: float,
        training: bool,
) -> torch.Tensor:
    if drop_p <= 0.0 or not training:
        return x
    keep = 1.0 - drop_p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


# =============================================================================
# Coordinate Attention (unchanged)
# =============================================================================

class CoordAtt2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1  = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(mip)
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


# =============================================================================
# Selective Kernel block (unchanged)
# =============================================================================

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


# =============================================================================
# Residual DRSN — actually residual (matches Zhao et al.)
# =============================================================================

class ResidualDRSN1D(nn.Module):
    """
    Residual cubic-threshold block: out = x + cubic_threshold(F(x), τ(F(x))).
    The pre-shrinkage transform F is depthwise-separable: cheap, channel-preserving.
    """

    def __init__(self, channels: int, reduction: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.drop_path = drop_path

        self.transform = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),               # depthwise
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm1d(channels),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.transform(x)
        tau = self.gate(out).unsqueeze(-1)
        ref = torch.abs(out).mean(dim=-1, keepdim=True)
        out = _cubic_threshold(out, tau, ref)
        out = stochastic_drop_path(out, self.drop_path, self.training)
        return x + out


class ResidualDRSN2D(nn.Module):
    """2D analogue of ResidualDRSN1D."""

    def __init__(self, channels: int, reduction: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.drop_path = drop_path

        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.transform(x)
        tau = self.gate(out).view(x.size(0), x.size(1), 1, 1)
        ref = torch.abs(out).mean(dim=(2, 3), keepdim=True)
        out = _cubic_threshold(out, tau, ref)
        out = stochastic_drop_path(out, self.drop_path, self.training)
        return x + out


class FusedDRSN(nn.Module):
    """
    Vector-valued residual cubic-threshold for the fused fingerprint.
    This is where the "sparse activation fingerprint" is born.
    """

    def __init__(self, dim: int, reduction: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.drop_path = drop_path

        self.transform = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.transform(x)
        tau = self.fc(out)
        ref = torch.abs(out).mean(dim=1, keepdim=True)
        out = _cubic_threshold(out, tau, ref)
        out = stochastic_drop_path(out, self.drop_path, self.training)
        return x + out


# =============================================================================
# Single-conv residual stage for the 1D branches
# =============================================================================

class ResidualConv1DStage(nn.Module):
    """
    Wraps an existing single-conv stage with a residual shortcut.
        y = ReLU(BN(Conv(x)) + shortcut(x))

    Picked over a 2-conv ResNet basic block to get gradient-highway and
    regularization benefits without doubling the conv count (which would
    inflate params by 80%+ in this model).

    When stride > 1 or in_ch != out_ch, the shortcut is a 1×1 conv that
    matches the residual branch's spatial / channel dims.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            stride: int = 1,
            drop_path: float = 0.0,
    ):
        super().__init__()
        self.drop_path = drop_path
        pad = kernel_size // 2

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.bn(self.conv(x))
        out = stochastic_drop_path(out, self.drop_path, self.training)
        return F.relu(out + identity, inplace=True)


# =============================================================================
# Cosine classifier (NormFace) — magnitude-meaningful logits
# =============================================================================

class CosineClassifier(nn.Module):
    """
    NormFace-style classifier. logits = s · cos(fp, W_c).
    Both fp and W are L2-normalized; s is a learned scalar (stored as log-scale
    to stay positive under unconstrained optimization).

    Optional CosFace margin m subtracts m from the target-class logit at
    training time only. Defaults to m=0 = pure NormFace.

    Why this matters for OSR: the cosine logit is invariant to ||fp||, but the
    SupCon head's gradient flow still rewards confident samples by growing
    fingerprint magnitude. The OSR calibrator's emb_norm feature becomes
    informative.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0,
                 margin: float = 0.0, learnable_scale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        init_log_scale = torch.log(torch.tensor(float(scale)))
        if learnable_scale:
            self.log_scale = nn.Parameter(init_log_scale.clone())
        else:
            self.register_buffer("log_scale", init_log_scale.clone())

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(self, fp: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        fp_n = F.normalize(fp,         p=2, dim=1)
        w_n  = F.normalize(self.weight, p=2, dim=1)
        cos  = fp_n @ w_n.t()                                  # (B, C)  ∈ [-1, 1]

        if self.margin > 0.0 and labels is not None and self.training:
            # CosFace-style additive margin on the target class only. Note: this
            # can drive cos slightly below -1 in pathological early-training
            # cases. Standard CosFace does not clamp here; if you set margin
            # large (>0.3) and see early-training instability, consider clamp.
            one_hot = F.one_hot(labels, self.num_classes).to(cos.dtype)
            cos = cos - one_hot * self.margin

        return self.scale * cos


# =============================================================================
# Modality reliability gate (unchanged)
# =============================================================================

class ModalityReliabilityGate(nn.Module):
    def __init__(self, branch_dim: int, num_branches: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(branch_dim * num_branches, branch_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(branch_dim, num_branches, bias=False),
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        combined = torch.cat(features, dim=-1)
        weights  = torch.sigmoid(self.fc(combined))
        return [f * weights[:, i].unsqueeze(-1) for i, f in enumerate(features)]


# =============================================================================
# STFT branch — slim, single DRSN per path at deepest stage
# =============================================================================

class TrajectoryBranch(nn.Module):
    """
    Captures pulse-shape and modulation-trajectory cues with elongated kernels.
    DRSN moved to the deepest stage only.
    """

    def __init__(self, drop_path: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=2, padding=(2, 0), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResidualDRSN2D(128, drop_path=drop_path),  # deepest stage only
        )

    def forward(self, x):
        return self.net(x)


class EnhancedSTFTBranch(nn.Module):
    """
    Two-path STFT branch:
      - coord_path: CoordAtt-driven channel/spatial attention (frequency × time)
      - trajectory_path: elongated kernels for pulse-shape cues
    A single DRSN per path, at the deepest stage.
    """

    def __init__(self, branch_dim: int = 128, drop_path: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.coord_path = nn.Sequential(
            CoordAtt2D(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            CoordAtt2D(64),
            ResidualDRSN2D(64, drop_path=drop_path),
        )

        self.trajectory_path = TrajectoryBranch(drop_path=drop_path)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 64 (coord) + 128 (traj) = 192
        self.compress = nn.Sequential(
            nn.Linear(64 + 128, branch_dim, bias=False),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_features = self.stem(x)

        coord_feat = self.coord_path(stem_features)
        traj_feat  = self.trajectory_path(stem_features)

        coord_flat = torch.flatten(self.pool(coord_feat), 1)
        traj_flat  = torch.flatten(self.pool(traj_feat),  1)

        merged = torch.cat([coord_flat, traj_flat], dim=1)
        return self.compress(merged)


# =============================================================================
# Asymmetric TriNet — main module
# =============================================================================

class AsymmetricTriNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            branch_dim: int = 128,
            fingerprint_dim: int = 256,
            modality_dropout: float = 0.25,
            num_transformer_layers: int = 1,
            nhead: int = 4,
            use_cls_token: bool = True,
            supcon_dim: int = 128,
            stochastic_depth_max_p: float = 0.10,
            cosine_scale: float = 16.0,
            cosine_margin: float = 0.0,
    ):
        super().__init__()
        self.mod_drop_p      = modality_dropout
        self.branch_dim      = branch_dim
        self.fingerprint_dim = fingerprint_dim
        self.use_cls_token   = use_cls_token

        # ---- Stochastic-depth ramp -------------------------------
        # We enumerate every block that consumes a drop_path rate and assign
        # it an explicit depth index. Parallel branches (IQ ↔ IF, and the
        # two STFT paths inside the STFT branch) share the same depth index
        # because they sit at the same position in the network's depth.
        #
        # Depth indexing (5 distinct depths in our network):
        #   depth 0 — first 1D residual stage          (iq_stage2, if_stage2)
        #   depth 1 — second 1D residual stage         (iq_stage3, if_stage3)
        #   depth 2 — branch-output DRSN               (iq_drsn, if_drsn,
        #                                              + both STFT DRSNs,
        #                                              all four sit at the
        #                                              "deep abstract" stage
        #                                              of their respective
        #                                              branches)
        #   depth 3 — fused fingerprint refinement     (drsn_refiner)
        #
        # Linear ramp drop_p(d) = sd_max · d / 3.
        sd_max = stochastic_depth_max_p
        n_depths = 4
        def sd(d: int) -> float:
            return sd_max * d / max(1, n_depths - 1)

        sd_branch_drsn = sd(2)   # all four branch-output DRSNs share this depth
        sd_stage_1     = sd(0)
        sd_stage_2     = sd(1)
        sd_fingerprint = sd(3)   # = sd_max

        # ---- STFT branch ----
        # Both DRSNs inside (coord_path's at 64ch, trajectory_path's at 128ch)
        # are at the deepest stage of the STFT branch; both get sd_branch_drsn.
        self.stft_branch = EnhancedSTFTBranch(
            branch_dim=branch_dim,
            drop_path=sd_branch_drsn,
        )

        # ---- IQ branch (1D, 3 inputs: real, imag, |x|) ----
        # Stem (no residual), SK at the cheap 32-ch stage, single-conv residual
        # stages with shortcuts, single DRSN at the deepest 128-ch stage.
        self.iq_stem = nn.Sequential(
            nn.Conv1d(3, 32, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.iq_sk     = SKBlock1D(32)                                          # multi-scale at 32ch
        self.iq_stage2 = ResidualConv1DStage(32, 64,         kernel_size=5, stride=4, drop_path=sd_stage_1)
        self.iq_stage3 = ResidualConv1DStage(64, branch_dim, kernel_size=3, stride=2, drop_path=sd_stage_2)
        self.iq_drsn   = ResidualDRSN1D(branch_dim, drop_path=sd_branch_drsn)   # deepest, residual cubic-threshold
        self.iq_pool   = nn.AdaptiveAvgPool1d(1)

        # ---- IF branch (1D, 1 input: instantaneous freq) ----
        self.if_stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.if_sk     = SKBlock1D(32)
        self.if_stage2 = ResidualConv1DStage(32, 64,         kernel_size=5, stride=4, drop_path=sd_stage_1)
        self.if_stage3 = ResidualConv1DStage(64, branch_dim, kernel_size=3, stride=2, drop_path=sd_stage_2)
        self.if_drsn   = ResidualDRSN1D(branch_dim, drop_path=sd_branch_drsn)
        self.if_pool   = nn.AdaptiveAvgPool1d(1)

        # ---- Cross-modal fusion ----
        self.reliability_gate = ModalityReliabilityGate(branch_dim, 3)
        self.modality_embed   = nn.Parameter(torch.randn(1, 3, branch_dim) * 0.02)

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

        # ---- Fingerprint projection + sparse-fingerprint refinement ----
        fused_dim = branch_dim if use_cls_token else branch_dim * 3
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fused_dim, fingerprint_dim, bias=False),
            nn.BatchNorm1d(fingerprint_dim),
        )
        # The FusedDRSN at the fingerprint is where the sparse-activation-
        # fingerprint contract is realised. Highest drop-path rate in the ramp.
        self.drsn_refiner = FusedDRSN(dim=fingerprint_dim, reduction=4, drop_path=sd_fingerprint)

        # ---- SupCon projection (unchanged; discarded after closed-set training) ----
        self.supcon_head = nn.Sequential(
            nn.Linear(fingerprint_dim, fingerprint_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fingerprint_dim, supcon_dim, bias=False),
        )

        # ---- Cosine classifier (NormFace) ----
        # Scale is held FIXED (not learnable) so it doesn't interact with the
        # trainer's weight_decay on Adam(model.parameters(), ...). If you want
        # a learnable scale, pass it through and exclude it from weight_decay
        # in the optimizer config.
        self.classifier_dropout = nn.Dropout(0.3)
        self.classifier = CosineClassifier(
            in_features=fingerprint_dim,
            num_classes=num_classes,
            scale=cosine_scale,
            margin=cosine_margin,
            learnable_scale=False,
        )

    # -------------------------------------------------------------
    # Backward-compat shims for code that calls model.iq_branch / if_branch
    # as sequential modules (e.g. the OSR _backbone_outputs).
    # The OLD iq_branch ended in AdaptiveAvgPool1d(1) returning shape (B, C, 1);
    # callers then ran torch.flatten(..., 1) → (B, C). We reproduce that contract
    # here by returning (B, C, 1).
    # -------------------------------------------------------------
    def iq_branch(self, x: torch.Tensor) -> torch.Tensor:
        x = self.iq_stem(x)
        x = self.iq_sk(x)
        x = self.iq_stage2(x)
        x = self.iq_stage3(x)
        x = self.iq_drsn(x)
        return self.iq_pool(x)                                  # (B, C, 1)

    def if_branch(self, x: torch.Tensor) -> torch.Tensor:
        x = self.if_stem(x)
        x = self.if_sk(x)
        x = self.if_stage2(x)
        x = self.if_stage3(x)
        x = self.if_drsn(x)
        return self.if_pool(x)                                  # (B, C, 1)

    # -------------------------------------------------------------
    # Modality dropout (unchanged behaviour, higher default p)
    # -------------------------------------------------------------
    def _modality_dropout(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.training or self.mod_drop_p <= 0:
            return feats
        n = len(feats)
        mask = torch.bernoulli(torch.full((n,), 1.0 - self.mod_drop_p, device=feats[0].device))
        if mask.sum() == 0:
            mask[torch.randint(n, (1,))] = 1.0
        scale = float(n) / mask.sum()
        return [f * mask[i] * scale for i, f in enumerate(feats)]

    # -------------------------------------------------------------
    # Branch forwards (used by the model's own forward path)
    # -------------------------------------------------------------
    def _iq_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.iq_branch(x), 1)

    def _if_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.if_branch(x), 1)

    # -------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------
    def _fuse(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        # Reliability gate operates on the RAW branch outputs so it can read
        # both magnitude (which BN-then-ReLU outputs carry as a noise/quality
        # signal) and direction. If we L2-normalised first, the gate would be
        # blind to magnitude and could only judge reliability from angle.
        f1, f2, f3 = self.reliability_gate([f1, f2, f3])

        # L2-normalise the gated tokens so the modality_embed (init scale 0.02)
        # contributes a meaningful identity perturbation on top of unit-norm
        # tokens before they reach the transformer.
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)
        f3 = F.normalize(f3, p=2, dim=1)

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

    # -------------------------------------------------------------
    # Public forward API — kept compatible with engine.py / OSR model
    # -------------------------------------------------------------
    def forward(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            return_fingerprint: bool = False,
            labels: torch.Tensor | None = None,
    ):
        """
        Standard forward.

        Returns:
            logits                                       (B, num_classes)
        Or, with return_fingerprint=True:
            (logits, supcon_z)  — supcon_z is L2-normalized

        `labels` is only used by the cosine classifier when a non-zero CosFace
        margin is configured. The CE loss path in train/engine.py does not
        currently pass labels, which is fine because we default to margin=0
        (pure NormFace, no margin).
        """
        f1 = self.stft_branch(x_stft)
        f2 = self._iq_forward(x_iq)
        f3 = self._if_forward(x_if)

        f1, f2, f3 = self._modality_dropout([f1, f2, f3])

        fp = self._fuse(f1, f2, f3)
        logits = self.classifier(self.classifier_dropout(fp), labels=labels)

        if return_fingerprint:
            z = F.normalize(self.supcon_head(fp), p=2, dim=1)
            return logits, z

        return logits

    # -------------------------------------------------------------
    # Embedding extractors (used by OSR codebook + diagnostics)
    # -------------------------------------------------------------
    @torch.no_grad()
    def extract_fingerprint(self, x_stft: torch.Tensor, x_iq: torch.Tensor, x_if: torch.Tensor) -> torch.Tensor:
        f1 = self.stft_branch(x_stft)
        f2 = self._iq_forward(x_iq)
        f3 = self._if_forward(x_if)
        return self._fuse(f1, f2, f3)

    @torch.no_grad()
    def extract_embedding(self, x_stft, x_iq, x_if) -> torch.Tensor:
        return self.extract_fingerprint(x_stft, x_iq, x_if)

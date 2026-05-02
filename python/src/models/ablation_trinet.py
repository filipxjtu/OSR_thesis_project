"""
python/src/models/ablation_trinet.py
=====================================
Thin subclass of AsymmetricTriNet that adds a `disabled_branches`
parameter to forward(). Drop this file into python/src/models/ and
add the import to python/src/models/__init__.py.

Design:
  - Does NOT touch asymmetric_trinet.py at all.
  - Zeroes specified branch outputs AFTER their forward passes and AFTER
    modality dropout, but BEFORE _fuse(). This means:
      * All branch parameters still receive gradients from CE+SupCon.
        Wait — actually no: zeroing BEFORE fuse means the zero token
        flows through the transformer and produces zero gradient back
        through that branch. This is intentional: the disabled branch
        contributes nothing to the loss, so its weights are not
        updated except via weight decay. This is the honest ablation.
      * The reliability gate and transformer always receive 3 tokens,
        keeping the fusion architecture structurally identical across
        all variants.

Branch index convention (same as token order in _fuse):
    0 → STFT (f1)
    1 → IQ   (f2)
    2 → IF   (f3)

Usage:
    model = AblationTriNet(disabled_branches={1, 2})  # STFT-only
    model = AblationTriNet(disabled_branches={2})      # STFT + IQ
    model = AblationTriNet(disabled_branches=set())    # full model
"""

from __future__ import annotations

from typing import Optional, Set

import torch
import torch.nn.functional as F

from .asymmetric_trinet import AsymmetricTriNet


class AblationTriNet(AsymmetricTriNet):
    """
    AsymmetricTriNet with selectable branch masking for ablation studies.
    All constructor arguments are forwarded unchanged to the parent.
    The `disabled_branches` set is stored as an instance attribute so
    that train_model() doesn't need to know about it — the model itself
    enforces the mask on every forward call.
    """

    def __init__(self, *args, disabled_branches: Optional[Set[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Validate
        if disabled_branches is None:
            disabled_branches = set()
        invalid = disabled_branches - {0, 1, 2}
        if invalid:
            raise ValueError(f"disabled_branches must be subset of {{0,1,2}}, got extra: {invalid}")
        if disabled_branches == {0, 1, 2}:
            raise ValueError("Cannot disable all three branches.")
        self.disabled_branches: Set[int] = disabled_branches

    # ------------------------------------------------------------------
    # Override forward — identical to parent except for the zero-masking
    # block inserted between modality dropout and _fuse().
    # ------------------------------------------------------------------
    def forward(
        self,
        x_stft: torch.Tensor,
        x_iq: torch.Tensor,
        x_if: torch.Tensor,
        return_fingerprint: bool = False,
        labels: Optional[torch.Tensor] = None,
    ):
        f1 = self.stft_branch(x_stft)
        f2 = self._iq_forward(x_iq)
        f3 = self._if_forward(x_if)

        # Apply modality dropout (training only, parent implementation)
        f1, f2, f3 = self._modality_dropout([f1, f2, f3])

        # ── Ablation masking ──────────────────────────────────────────
        # Zero out disabled branch outputs so they contribute nothing to
        # the fused representation. The reliability gate will learn to
        # assign near-zero weights to these constant-zero tokens.
        if 0 in self.disabled_branches:
            f1 = torch.zeros_like(f1)
        if 1 in self.disabled_branches:
            f2 = torch.zeros_like(f2)
        if 2 in self.disabled_branches:
            f3 = torch.zeros_like(f3)
        # ─────────────────────────────────────────────────────────────

        fp = self._fuse(f1, f2, f3)
        logits = self.classifier(self.classifier_dropout(fp), labels=labels)

        if return_fingerprint:
            z = F.normalize(self.supcon_head(fp), p=2, dim=1)
            return logits, z

        return logits

    def extra_repr(self) -> str:
        names = {0: "STFT", 1: "IQ", 2: "IF"}
        disabled_str = ", ".join(names[i] for i in sorted(self.disabled_branches)) or "none"
        return f"disabled_branches={{{disabled_str}}}"
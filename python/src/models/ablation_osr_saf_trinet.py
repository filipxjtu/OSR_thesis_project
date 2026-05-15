"""
python/src/models/ablation_osr_saf_trinet.py
=============================================
Thin subclass of OsrSAF_TriNet for the **codebook ablation study**.

What it does:
  Wraps `score_calibrator` with a fixed mask buffer that zeroes the
  calibrator inputs originating from the disabled codebook(s). Phase 1
  still fills both codebooks normally (cheap, keeps state-dict shapes
  uniform); the calibrator simply never sees the masked features.

Calibrator input layout (8-dim, defined in OsrSAF_TriNet.forward_with_osr_logits):
    idx 0 : code_dist             ── Cosine codebook
    idx 1 : unc                   ── Backbone (1 - softmax max)
    idx 2 : emb_norm_normalised   ── Backbone
    idx 3 : runner_up_dist        ── Cosine codebook
    idx 4 : margin_codebook       ── Cosine codebook
    idx 5 : logit_margin_squashed ── Backbone
    idx 6 : hamming_dist_pred     ── Hamming codebook
    idx 7 : hamming_margin        ── Hamming codebook

Cosine indices : {0, 3, 4}
Hamming indices: {6, 7}

Variants:
    "full"     → mask = all ones                          (control)
    "cosine"   → zero hamming features {6, 7}             (cosine codebook only)
    "hamming"  → zero cosine features  {0, 3, 4}          (hamming codebook only)
    "neither"  → zero all 5 codebook features {0,3,4,6,7} (backbone-only floor)

Usage:
    model = AblationOsrSAF_TriNet(
        num_classes=10,
        disabled_codebooks={"hamming"},   # → cosine-only variant
        use_pretrained=True,
        pretrained_path=str(closed_set_ckpt),
    )

Design notes:
  - Always wraps the calibrator (even the "full" variant carries an
    all-ones mask) so that all four variants have an *identical*
    state-dict layout. This makes evaluation code uniform and avoids
    accidental architecture drift between variants.
  - The mask is a `register_buffer`, so it follows the model to the
    correct device and is part of the saved state_dict.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

import torch
import torch.nn as nn

from .osr_saf_trinet import OsrSAF_TriNet


_VALID_CODEBOOKS = {"cosine", "hamming"}

# Indices into the 8-dim calibrator input vector
_COSINE_IDX = (0, 3, 4)
_HAMMING_IDX = (6, 7)


class _MaskedCalibrator(nn.Module):
    """Wraps the original score_calibrator with a fixed input-mask buffer."""

    def __init__(self, original: nn.Module, mask: torch.Tensor):
        super().__init__()
        self.original = original
        # Buffer (not a parameter): tracks the device, saved in state_dict,
        # but receives no gradient updates.
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x * self.mask)


def _build_mask(disabled: Set[str]) -> torch.Tensor:
    """Return an 8-dim float mask with zeros at positions belonging to disabled codebooks."""
    mask = torch.ones(8, dtype=torch.float32)
    if "cosine" in disabled:
        for i in _COSINE_IDX:
            mask[i] = 0.0
    if "hamming" in disabled:
        for i in _HAMMING_IDX:
            mask[i] = 0.0
    return mask


class AblationOsrSAF_TriNet(OsrSAF_TriNet):
    """
    OsrSAF_TriNet with selectable codebook masking for ablation studies.

    All constructor arguments are forwarded to the parent. The
    `disabled_codebooks` set controls which codebook(s) are zeroed out
    of the calibrator input vector.
    """

    def __init__(
        self,
        *args,
        disabled_codebooks: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Normalize and validate
        if disabled_codebooks is None:
            disabled_codebooks = set()
        disabled = {c.lower() for c in disabled_codebooks}
        invalid = disabled - _VALID_CODEBOOKS
        if invalid:
            raise ValueError(
                f"disabled_codebooks must be a subset of {_VALID_CODEBOOKS}, "
                f"got extra: {invalid}"
            )

        self.disabled_codebooks: Set[str] = disabled

        # Wrap the calibrator with the fixed input mask.
        mask = _build_mask(disabled)
        self.score_calibrator = _MaskedCalibrator(self.score_calibrator, mask)

    def extra_repr(self) -> str:
        if not self.disabled_codebooks:
            return "disabled_codebooks={} (full)"
        return f"disabled_codebooks={sorted(self.disabled_codebooks)}"
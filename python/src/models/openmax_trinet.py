"""
python/src/models/openmax_trinet.py
====================================
OpenMax (Bendale & Boult, CVPR 2016) on top of AsymmetricTriNet.

Design parallels OsrSAF_TriNet:
  - Holds a frozen AsymmetricTriNet backbone.
  - Phase 1 (offline): collect activation vectors (AVs = pre-softmax logits),
    compute per-class Mean Activation Vector (MAV), fit a per-class Weibull
    distribution to the tail of in-class distances from MAV.
  - Phase 2 (offline): calibrate a single rejection threshold on validation
    knowns + proxy unknowns using Youden's J under an FPR cap.
  - Inference: revise the top-α AVs by the per-class Weibull CDF (the more
    "tail-like" the distance, the more activation mass we move to a synthetic
    "unknown" logit), softmax over (K+1) entries, reject if unknown_score
    exceeds the calibrated threshold.

Note on the cosine classifier:
  AsymmetricTriNet uses a NormFace-style cosine classifier so the AVs are
  bounded ≈ [-scale, +scale]. This does NOT break OpenMax — distances from
  the per-class MAV inside this bounded space are still informative for tail
  fitting. Using the cosine-scaled AVs keeps OpenMax's input scale comparable
  to the closed-set decision surface.

Why no asymmetric_for_openmax.py:
  AsymmetricTriNet.forward(x_stft, x_iq, x_if) already returns the AV (the
  cosine logits). No closed-set modification is required, so we keep the
  closed-set file untouched. This module wraps a frozen AsymmetricTriNet and
  exposes the OpenMax-specific interface on top.


"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import weibull_min

from .asymmetric_trinet import AsymmetricTriNet


class OpenMaxTriNet(nn.Module):
    """
    OpenMax wrapper around a (frozen) AsymmetricTriNet backbone.

    Stored state (registered buffers, survives state_dict save/load):
        mavs            (K, K)   per-class Mean Activation Vector
        weibull_shape   (K,)     per-class Weibull shape parameter (k)
        weibull_loc     (K,)     per-class Weibull location parameter (always 0 with floc=0)
        weibull_scale   (K,)     per-class Weibull scale parameter (lambda)
        threshold       ()       single global rejection threshold on unknown score
        n_fit_per_class (K,)     number of training samples used per class for the fit
        _fitted         ()       boolean flag indicating the fit has been completed
    """

    def __init__(
        self,
        num_classes: int = 10,
        alpha_rank: Optional[int] = None,
        tail_size: int = 20,
        distance: str = "euclidean",   # "euclidean" or "cosine"
        # backbone construction kwargs (must match the closed-set checkpoint)
        branch_dim: int = 128,
        fingerprint_dim: int = 256,
        modality_dropout: float = 0.3,
        num_transformer_layers: int = 1,
        nhead: int = 4,
        use_cls_token: bool = True,
        supcon_dim: int = 128,
        # backbone loading
        use_pretrained: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        if distance not in ("euclidean", "cosine"):
            raise ValueError(f"distance must be 'euclidean' or 'cosine', got '{distance}'")

        self.num_classes = num_classes
        self.alpha_rank = alpha_rank if alpha_rank is not None else min(10, num_classes)
        self.tail_size = tail_size
        self.distance = distance

        # ---- Backbone (frozen after pretrained load) ------------------------
        self.base = AsymmetricTriNet(
            num_classes=num_classes,
            branch_dim=branch_dim,
            fingerprint_dim=fingerprint_dim,
            modality_dropout=modality_dropout,
            num_transformer_layers=num_transformer_layers,
            nhead=nhead,
            use_cls_token=use_cls_token,
            supcon_dim=supcon_dim,
        )

        if use_pretrained and pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.base.load_state_dict(state, strict=False)
            print(f"[OpenMaxTriNet] Loaded backbone from {pretrained_path}")

        # ---- Fitted state (zeros until fit() is called) ---------------------
        self.register_buffer("mavs", torch.zeros(num_classes, num_classes))
        self.register_buffer("weibull_shape", torch.ones(num_classes))
        self.register_buffer("weibull_loc",   torch.zeros(num_classes))
        self.register_buffer("weibull_scale", torch.ones(num_classes))
        self.register_buffer("threshold", torch.tensor(0.5))
        self.register_buffer("n_fit_per_class", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("_fitted", torch.tensor(False))

    # =========================================================================
    # AV extraction
    # =========================================================================

    @torch.no_grad()
    def extract_av(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
    ) -> torch.Tensor:
        """Returns the activation vector AV = closed-set logits, shape (B, K)."""
        self.base.eval()
        return self.base(x_stft, x_iq, x_if)  # (B, K)

    # =========================================================================
    # Distance helpers
    # =========================================================================

    def _pairwise_distance(
        self,
        av: torch.Tensor,    # (B, K)
        mavs: torch.Tensor,  # (K, K)
    ) -> torch.Tensor:
        """Returns (B, K): distance from each AV to each class MAV."""
        if self.distance == "euclidean":
            return torch.cdist(av, mavs, p=2)
        # cosine distance = 1 - cosine_similarity
        av_n  = F.normalize(av,   p=2, dim=1)
        mav_n = F.normalize(mavs, p=2, dim=1)
        return 1.0 - av_n @ mav_n.t()

    def _self_distance(
        self,
        avs_c: torch.Tensor,  # (N_c, K)
        mav_c: torch.Tensor,  # (K,)
    ) -> torch.Tensor:
        """Returns (N_c,): distance from each AV to its own class MAV."""
        if self.distance == "euclidean":
            return torch.norm(avs_c - mav_c.unsqueeze(0), p=2, dim=1)
        av_n  = F.normalize(avs_c, p=2, dim=1)
        mav_n = F.normalize(mav_c, p=2, dim=0).unsqueeze(0)
        return 1.0 - (av_n * mav_n).sum(dim=1)

    # =========================================================================
    # Fitting (Phase 1, offline)
    # =========================================================================

    @torch.no_grad()
    def fit_from_avs(
        self,
        avs: torch.Tensor,          # (N, K) collected on training set
        labels: torch.Tensor,       # (N,)   integer class labels
        only_correctly_classified: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Compute MAVs and fit per-class Weibull tails. Pure tensor / scipy work.

        Returns a small info dict with per-class fit diagnostics.
        """
        device = self.mavs.device
        avs    = avs.to(device).float()
        labels = labels.to(device).long()

        if only_correctly_classified:
            preds = avs.argmax(dim=1)
            keep_mask = (preds == labels)
            avs    = avs[keep_mask]
            labels = labels[keep_mask]
            if verbose:
                print(f"[OpenMax fit] kept {int(keep_mask.sum())}/{int(keep_mask.numel())} "
                      f"correctly-classified samples")

        info = {"per_class": {}, "tail_size": self.tail_size, "distance": self.distance}

        new_mavs   = torch.zeros_like(self.mavs)
        new_shape  = torch.ones_like(self.weibull_shape)
        new_loc    = torch.zeros_like(self.weibull_loc)
        new_scale  = torch.ones_like(self.weibull_scale)
        new_counts = torch.zeros_like(self.n_fit_per_class)

        for c in range(self.num_classes):
            mask = (labels == c)
            n_c = int(mask.sum().item())
            new_counts[c] = n_c

            if n_c < 2:
                if verbose:
                    print(f"[OpenMax fit] class {c}: only {n_c} sample(s) — using zero MAV "
                          f"and unit Weibull (this class will not reject anything).")
                info["per_class"][c] = {"n": n_c, "fit_ok": False}
                continue

            avs_c = avs[mask]                # (n_c, K)
            mav_c = avs_c.mean(dim=0)        # (K,)
            new_mavs[c] = mav_c

            distances = self._self_distance(avs_c, mav_c)  # (n_c,)

            # Tail = top-η largest distances (most extreme correctly-classified samples)
            k_tail = min(self.tail_size, n_c)
            tail = torch.topk(distances, k_tail, largest=True).values.cpu().numpy()

            # Guard: degenerate tails (all zeros, single value, etc.)
            if tail.size < 2 or float(np.std(tail)) < 1e-9:
                if verbose:
                    print(f"[OpenMax fit] class {c}: degenerate tail "
                          f"(n={tail.size}, std={float(np.std(tail)):.2e}) — fallback to unit Weibull")
                info["per_class"][c] = {"n": n_c, "tail_n": int(tail.size), "fit_ok": False}
                continue

            try:
                shape, loc, scale = weibull_min.fit(tail, floc=0.0)
                if not (np.isfinite(shape) and np.isfinite(scale) and scale > 1e-9 and shape > 1e-9):
                    raise RuntimeError("non-finite or degenerate Weibull params")
                new_shape[c] = float(shape)
                new_loc[c]   = float(loc)
                new_scale[c] = float(scale)
                info["per_class"][c] = {
                    "n":         n_c,
                    "tail_n":    int(tail.size),
                    "tail_max":  float(tail.max()),
                    "tail_min":  float(tail.min()),
                    "shape":     float(shape),
                    "loc":       float(loc),
                    "scale":     float(scale),
                    "fit_ok":    True,
                }
            except Exception as e:
                if verbose:
                    print(f"[OpenMax fit] class {c}: Weibull fit failed ({e}) — fallback to unit Weibull")
                info["per_class"][c] = {"n": n_c, "tail_n": int(tail.size),
                                        "fit_ok": False, "error": str(e)}

        self.mavs.copy_(new_mavs)
        self.weibull_shape.copy_(new_shape)
        self.weibull_loc.copy_(new_loc)
        self.weibull_scale.copy_(new_scale)
        self.n_fit_per_class.copy_(new_counts)
        self._fitted.fill_(True)

        if verbose:
            n_ok = sum(1 for v in info["per_class"].values() if v.get("fit_ok"))
            print(f"[OpenMax fit] completed: {n_ok}/{self.num_classes} classes fit successfully")

        return info

    # =========================================================================
    # OpenMax inference (revised AV + synthetic unknown logit)
    # =========================================================================

    def compute_openmax_logits(self, av: torch.Tensor) -> torch.Tensor:
        """
        Apply the OpenMax revision to (B, K) AVs and return (B, K+1) revised logits.
        Last column is the synthetic 'unknown' logit.
        """
        if not bool(self._fitted.item()):
            raise RuntimeError("OpenMaxTriNet has not been fit yet. Call fit_from_avs(...) first.")

        K = self.num_classes
        B = av.size(0)

        # ---- Distances + Weibull CDF per class -----------------------------
        dists = self._pairwise_distance(av, self.mavs)              # (B, K)

        # F(d; k, λ, loc=0) = 1 - exp(-((d - loc)/λ)^k) for d >= loc, else 0
        d_shift = (dists - self.weibull_loc.unsqueeze(0)).clamp(min=0.0)        # (B, K)
        ratio   = d_shift / self.weibull_scale.unsqueeze(0).clamp(min=1e-9)     # (B, K)
        cdf     = 1.0 - torch.exp(-ratio.pow(self.weibull_shape.unsqueeze(0)))  # (B, K)
        cdf     = cdf.clamp(0.0, 1.0)

        # ---- Sort AVs descending and gather CDF in that order --------------
        sorted_av, sorted_idx = av.sort(dim=1, descending=True)     # (B, K)
        cdf_sorted = cdf.gather(1, sorted_idx)                      # (B, K)

        # ---- Rank-weighted revision: only top-α get revised ---------------
        ranks = torch.arange(K, device=av.device, dtype=av.dtype)
        w_rank = ((self.alpha_rank - ranks) / float(self.alpha_rank)).clamp(min=0.0).unsqueeze(0)  # (1, K)

        rev_factor = 1.0 - cdf_sorted * w_rank                      # (B, K)
        sorted_av_revised = sorted_av * rev_factor                  # (B, K)

        # ---- Synthetic 'unknown' logit = total mass removed by revision ----
        unknown_logit = (sorted_av * (1.0 - rev_factor)).sum(dim=1, keepdim=True)  # (B, 1)

        # ---- Reorder revised AVs back to original class indexing -----------
        av_revised = torch.zeros_like(av)
        av_revised.scatter_(1, sorted_idx, sorted_av_revised)

        return torch.cat([av_revised, unknown_logit], dim=1)        # (B, K+1)

    def forward_with_openmax(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits          (B, K)   — original closed-set AV
            unknown_score   (B,)     — softmax probability of the synthetic unknown class
            openmax_probs   (B, K+1) — full OpenMax probability vector
        """
        av = self.extract_av(x_stft, x_iq, x_if)                    # (B, K)
        openmax_logits = self.compute_openmax_logits(av)            # (B, K+1)
        openmax_probs  = F.softmax(openmax_logits, dim=1)           # (B, K+1)
        unknown_score  = openmax_probs[:, -1]                       # (B,)
        return av, unknown_score, openmax_probs

    def forward_with_osr(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """API shim mirroring OsrSAF_TriNet.forward_with_osr — (logits, unknown_score)."""
        logits, unknown_score, _ = self.forward_with_openmax(x_stft, x_iq, x_if)
        return logits, unknown_score

    def forward(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
    ) -> torch.Tensor:
        """Default forward returns the original closed-set logits."""
        return self.extract_av(x_stft, x_iq, x_if)

    @torch.no_grad()
    def predict_with_rejection(
        self,
        x_stft: torch.Tensor,
        x_iq:   torch.Tensor,
        x_if:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            predictions (B,) ∈ {0,...,K-1, -1}  (-1 = rejected as unknown)
            unknown_scores (B,) ∈ [0, 1]
        """
        logits, unknown_score, _ = self.forward_with_openmax(x_stft, x_iq, x_if)

        preds = logits.argmax(dim=1).clone()
        preds[unknown_score > self.threshold] = -1
        return preds, unknown_score

    # =========================================================================
    # Threshold calibration (Youden's J under an FPR cap)
    # =========================================================================

    @torch.no_grad()
    def calibrate_threshold_youden(
        self,
        scores_known:   torch.Tensor,
        scores_unknown: torch.Tensor,
        fpr_cap: float = 0.4,
        n_grid: int = 401,
        verbose: bool = False,
    ) -> Dict:
        """Pick a single global threshold maximising TPR-FPR subject to FPR <= fpr_cap."""
        device = self.threshold.device
        sk = scores_known.detach().float().flatten().to(device)
        su = scores_unknown.detach().float().flatten().to(device)
        if sk.numel() == 0 or su.numel() == 0:
            return {"global_thr": float(self.threshold.item()), "n_known": int(sk.numel()), "n_unknown": int(su.numel())}

        grid = torch.linspace(0.0, 1.0, n_grid, device=device)
        # rejection rate at threshold t
        tpr = (su.unsqueeze(1) > grid.unsqueeze(0)).float().mean(dim=0)   # unknowns rejected = TPR
        fpr = (sk.unsqueeze(1) > grid.unsqueeze(0)).float().mean(dim=0)   # knowns wrongly rejected = FPR
        j   = tpr - fpr

        # Apply FPR cap: prefer points where fpr <= cap, fall back to argmin(fpr) if none qualify
        eligible = fpr <= fpr_cap
        if eligible.any():
            j_masked = torch.where(eligible, j, torch.full_like(j, -1.0))
            thr = float(grid[j_masked.argmax()].item())
        else:
            thr = float(grid[fpr.argmin()].item())

        idx_close = (grid - thr).abs().argmin()
        info = {
            "global_thr":    thr,
            "val_tpr_at_t":  float(tpr[idx_close].item()),
            "val_fpr_at_t":  float(fpr[idx_close].item()),
            "n_known":       int(sk.numel()),
            "n_unknown":     int(su.numel()),
            "fpr_cap":       fpr_cap,
        }
        if verbose:
            print(f"[OpenMax] Youden threshold: t={thr:.4f} | val TPR={info['val_tpr_at_t']:.3f} | "
                  f"val FPR={info['val_fpr_at_t']:.3f}")

        self.threshold.fill_(max(0.005, min(0.995, thr)))
        return info
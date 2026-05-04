from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asymmetric_trinet import AsymmetricTriNet


"""
OsrSAF_TriNet — Sparse Activation Fingerprint OSR built on AsymmetricTriNet.

Two parallel codebooks per class with k EMA prototypes:
  _CosineCodebook  : prototypes in L2-normalized fingerprint space; reads direction.
  _HammingCodebook : binary prototypes over the FusedDRSN survival mask; reads
                     channel-firing identity.

Phase 1 fills both codebooks with a frozen pretrained backbone. Phase 2 trains
the score_calibrator on proxy unknowns and recalibrates per-class thresholds
from the validation score distribution.
"""


class _CosineCodebook(nn.Module):
    """Per-class, k-centroid EMA codebook over L2-normalized embeddings."""

    def __init__(
            self,
            num_classes: int,
            code_dim: int,
            k: int = 4,
            ema_momentum: float = 0.95,
            beta: float = 0.85,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.k = k
        self.ema_momentum = ema_momentum
        self.beta = beta

        # Small random init so the first cosine read is well-defined.
        init = F.normalize(torch.randn(num_classes, k, code_dim), p=2, dim=-1) * 1e-3
        self.register_buffer("centroids", init)
        self.register_buffer("initialised", torch.zeros(num_classes, k, dtype=torch.bool))
        self.register_buffer("update_counts", torch.zeros(num_classes, k, dtype=torch.long))

    @torch.no_grad()
    def _normed_centroids(self, c_idx: int) -> torch.Tensor:
        return F.normalize(self.centroids[c_idx], p=2, dim=-1)

    @torch.no_grad()
    def update(self, codes: torch.Tensor, labels: torch.Tensor, current_momentum: float = 0.95):
        """Cold-start fill, then nearest-centroid EMA update with beta-scaled outlier guard."""
        for c in labels.unique():
            cid = int(c.item())
            if cid == -1:
                continue

            class_codes = codes[labels == c]
            if class_codes.numel() == 0:
                continue

            for kid in range(self.k):
                if self.initialised[cid, kid]:
                    continue
                if class_codes.shape[0] == 0:
                    break
                self.centroids[cid, kid] = class_codes[0]
                self.initialised[cid, kid] = True
                self.update_counts[cid, kid] = 1
                class_codes = class_codes[1:]

            if class_codes.shape[0] == 0:
                continue

            cents_normed = self._normed_centroids(cid)
            sim = class_codes @ cents_normed.t()
            dists = 1.0 - sim
            nearest = dists.argmin(dim=1)

            for kid in range(self.k):
                mask = nearest == kid
                assigned = class_codes[mask]
                if assigned.shape[0] == 0:
                    continue

                # Drop the long tail of the assignment so a noisy hit can't pull the centroid.
                if assigned.shape[0] > 1:
                    a_dists = dists[mask, kid]
                    cutoff = a_dists.mean() + self.beta * a_dists.std(unbiased=False)
                    keep = a_dists <= cutoff
                    if keep.any():
                        assigned = assigned[keep]

                mean_assigned = assigned.mean(dim=0)
                m = current_momentum
                self.centroids[cid, kid] = m * self.centroids[cid, kid] + (1.0 - m) * mean_assigned
                self.update_counts[cid, kid] += assigned.shape[0]

    @torch.no_grad()
    def code_distance_all_classes(self, codes: torch.Tensor) -> torch.Tensor:
        """Returns (B, C) cosine distance to nearest centroid of each class."""
        cents_normed = F.normalize(self.centroids, p=2, dim=-1)
        sim = torch.einsum("bd,ckd->bck", codes, cents_normed)
        dists = 1.0 - sim
        return dists.min(dim=-1).values

    def convergence_stats(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            normed = F.normalize(self.centroids, p=2, dim=-1)
            sim = normed @ normed.transpose(1, 2)
            mask = 1.0 - torch.eye(self.k, device=sim.device).unsqueeze(0)
            denom = max(1, self.k * (self.k - 1))
            spread = ((1.0 - sim) * mask).sum(dim=(1, 2)) / denom

        return {
            "mean_activation_per_class": self.centroids.norm(dim=-1).mean(dim=1),
            "spread_per_class": spread,
            "pct_initialised": self.initialised.float().mean(),
            "mean_updates_per_centroid": self.update_counts.float().mean(),
        }


class _HammingCodebook(nn.Module):
    """Per-class, k-prototype codebook over binary FusedDRSN survival masks."""

    def __init__(
            self,
            num_classes: int,
            code_dim: int,
            k: int = 4,
            ema_momentum: float = 0.95,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.k = k
        self.ema_momentum = ema_momentum

        # Soft prototypes init at 0.5 so unfilled slots give Hamming ~0.5.
        init = torch.full((num_classes, k, code_dim), 0.5)
        self.register_buffer("prototypes_soft", init)
        self.register_buffer("initialised", torch.zeros(num_classes, k, dtype=torch.bool))
        self.register_buffer("update_counts", torch.zeros(num_classes, k, dtype=torch.long))

    @torch.no_grad()
    def _all_binary_prototypes(self) -> torch.Tensor:
        return (self.prototypes_soft >= 0.5).to(self.prototypes_soft.dtype)

    @torch.no_grad()
    def update(self, masks: torch.Tensor, labels: torch.Tensor, current_momentum: float = 0.95):
        """Cold-start fill, then Hamming-nearest assignment and EMA on soft prototypes."""
        for c in labels.unique():
            cid = int(c.item())
            if cid == -1:
                continue

            class_masks = masks[labels == c]
            if class_masks.numel() == 0:
                continue

            for kid in range(self.k):
                if self.initialised[cid, kid]:
                    continue
                if class_masks.shape[0] == 0:
                    break
                self.prototypes_soft[cid, kid] = class_masks[0]
                self.initialised[cid, kid] = True
                self.update_counts[cid, kid] = 1
                class_masks = class_masks[1:]

            if class_masks.shape[0] == 0:
                continue

            bin_protos = (self.prototypes_soft[cid] >= 0.5).to(class_masks.dtype)
            diff = class_masks.unsqueeze(1) - bin_protos.unsqueeze(0)
            hdist = diff.abs().mean(dim=-1)
            nearest = hdist.argmin(dim=1)

            for kid in range(self.k):
                assign_mask = nearest == kid
                assigned = class_masks[assign_mask]
                if assigned.shape[0] == 0:
                    continue

                mean_assigned = assigned.float().mean(dim=0)
                m = current_momentum
                self.prototypes_soft[cid, kid] = (
                        m * self.prototypes_soft[cid, kid] + (1.0 - m) * mean_assigned
                )
                self.update_counts[cid, kid] += assigned.shape[0]

    @torch.no_grad()
    def hamming_distance_all_classes(self, masks: torch.Tensor) -> torch.Tensor:
        """Returns (B, C) normalized Hamming distance to nearest prototype of each class."""
        bin_protos = self._all_binary_prototypes()
        mb = masks.float()
        pb = bin_protos.float()

        # |a - b| for binary = a + b - 2 a b; means computed channelwise.
        term_a = mb.mean(dim=1, keepdim=True).unsqueeze(2)
        term_b = pb.mean(dim=-1).unsqueeze(0)
        D = mb.size(1)
        cross = torch.einsum("bd,ckd->bck", mb, pb) / D

        hdist = term_a + term_b - 2.0 * cross
        hdist = hdist.clamp(min=0.0, max=1.0)
        return hdist.min(dim=-1).values

    def convergence_stats(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            bin_protos = self._all_binary_prototypes()
            diff = bin_protos.unsqueeze(2) - bin_protos.unsqueeze(1)
            pair_hdist = diff.abs().mean(dim=-1)
            mask = 1.0 - torch.eye(self.k, device=pair_hdist.device).unsqueeze(0)
            denom = max(1, self.k * (self.k - 1))
            spread = (pair_hdist * mask).sum(dim=(1, 2)) / denom

            firing_rate = bin_protos.mean(dim=-1).mean(dim=-1)

        return {
            "spread_per_class_hamming": spread,
            "firing_rate_per_class": firing_rate,
            "pct_initialised_hamming": self.initialised.float().mean(),
            "mean_updates_per_prototype_hamming": self.update_counts.float().mean(),
        }


# Calibrator inputs: code_dist, unc, runner_up_dist, margin_codebook,
# logit_margin_squashed, hamming_dist_pred, hamming_margin.
_CALIB_INPUT_DIM = 7


class OsrSAF_TriNet(nn.Module):
    """
    Sparse Activation Fingerprint OSR on top of AsymmetricTriNet.

    Phase 1 populates both codebooks with a frozen pretrained backbone.
    Phase 2 trains the score_calibrator on proxy unknowns and periodically
    recalibrates per-class thresholds from the validation score distribution.
    """

    def __init__(
            self,
            num_classes: int = 10,
            k_centroids: int = 4,
            ema_momentum: float = 0.95,
            warmup_epochs: int = 30,
            codebook_beta: float = 0.9,
            threshold_recal_interval: int = 5,
            branch_dim: int = 128,
            fingerprint_dim: int = 256,
            modality_dropout: float = 0.1,
            num_transformer_layers: int = 2,
            nhead: int = 4,
            use_cls_token: bool = True,
            supcon_dim: int = 128,
            use_pretrained: bool = False,
            pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.threshold_recal_interval = threshold_recal_interval
        self._k = k_centroids
        self._ema_mom = ema_momentum
        self._beta = codebook_beta

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
            print(f"[OsrSAF_TriNet] Loaded backbone from {pretrained_path}")

        self._fingerprint_dim = fingerprint_dim
        self._codebook = _CosineCodebook(
            num_classes=num_classes,
            code_dim=fingerprint_dim,
            k=k_centroids,
            ema_momentum=ema_momentum,
            beta=codebook_beta,
        )
        self._hamming_codebook = _HammingCodebook(
            num_classes=num_classes,
            code_dim=fingerprint_dim,
            k=k_centroids,
            ema_momentum=ema_momentum,
        )

        # Final sigmoid is applied outside so the loss can use BCEWithLogitsLoss.
        self.score_calibrator = nn.Sequential(
            nn.Linear(_CALIB_INPUT_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        self.register_buffer("class_thresholds", torch.full((num_classes,), 0.5))

    # Codebook accessors

    def codebook_ready(self) -> bool:
        return bool(
            self._codebook.initialised.all().item()
            and self._hamming_codebook.initialised.all().item()
        )

    def get_codebook_stats(self) -> Optional[Dict]:
        cos_stats = self._codebook.convergence_stats()
        ham_stats = self._hamming_codebook.convergence_stats()
        return {**cos_stats, **ham_stats}

    # Threshold calibration from validation knowns

    @torch.no_grad()
    def calibrate_class_thresholds_formula(self, base_threshold: float = 0.5):
        """Spread-based initialiser used once before any calibrator scores exist."""
        spreads = self._codebook.convergence_stats()["spread_per_class"]
        norm_spreads = spreads / (spreads.max() + 1e-6)
        adjusted = base_threshold * (0.8 + 0.4 * (1 - norm_spreads))
        self.class_thresholds.copy_(adjusted.clamp(0.05, 0.95))

    @torch.no_grad()
    def calibrate_class_thresholds_from_scores(
            self,
            scores: torch.Tensor,
            pred_classes: torch.Tensor,
            target_fpr: float = 0.25,
            min_per_class: int = 30,
            verbose: bool = False,
    ) -> Dict[str, int]:
        """Set per-class thresholds at the (1 - target_fpr) percentile of scores per predicted class."""
        scores = scores.detach().to(self.class_thresholds.device).float()
        pred_classes = pred_classes.detach().to(self.class_thresholds.device).long()

        info = {"n_total": int(scores.numel()), "n_classes_fallback": 0, "n_classes_set": 0}
        if scores.numel() == 0:
            return info

        q = max(0.0, min(1.0, 1.0 - target_fpr))
        global_thr = float(torch.quantile(scores, q).item())

        new_thresh = self.class_thresholds.clone()
        fallback_classes = []
        for c in range(self.num_classes):
            mask = pred_classes == c
            n = int(mask.sum().item())
            if n < min_per_class:
                new_thresh[c] = global_thr
                fallback_classes.append((c, n))
                info["n_classes_fallback"] += 1
                continue
            class_scores = scores[mask]
            new_thresh[c] = torch.quantile(class_scores, q)
            info["n_classes_set"] += 1

        if verbose and fallback_classes:
            print(
                f"[OsrSAF_TriNet] threshold calibration: {len(fallback_classes)}/{self.num_classes} "
                f"classes used global fallback (n < {min_per_class}). "
                f"Fallback classes: {fallback_classes}. global_thr={global_thr:.4f}"
            )

        # Clamp only for numeric safety.
        self.class_thresholds.copy_(new_thresh.clamp(0.05, 0.95))
        return info

    # Forward variants

    @torch.no_grad()
    def calibrate_class_thresholds_youden(
            self,
            scores_known: torch.Tensor,
            pred_known: torch.Tensor,
            scores_unknown: torch.Tensor,
            pred_unknown: torch.Tensor,
            fpr_cap: float = 0.30,
            min_known_per_class: int = 30,
            min_unknown_per_class: int = 5,
            verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Set ALL class thresholds to a single global Youden's-J optimum on
        validation knowns + proxy unknowns, subject to FPR <= fpr_cap.

        Why single-global instead of per-class:
          Empirically, proxy unknowns get routed to only 2-3 predicted classes
          (those whose centroids are closest to the proxy distribution). The
          remaining 7 classes have <5 proxy samples each and would fall back to
          a "global" threshold computed on noisy mixed data. Test unknowns get
          routed to a different subset of classes, so per-class tuning on val
          doesn't transfer. A single global threshold is more robust to this
          proxy-vs-test distribution shift.

        pred_known and pred_unknown are kept in the signature for API stability
        and possible future per-class re-introduction; not used here.
        """
        sk = scores_known.detach().to(self.class_thresholds.device).float()
        su = scores_unknown.detach().to(self.class_thresholds.device).float()

        info = {
            "n_total_known": float(sk.numel()),
            "n_total_unknown": float(su.numel()),
            "global_thr": 0.5,
        }
        if sk.numel() == 0 or su.numel() == 0:
            return info

        grid = torch.linspace(0.02, 0.98, 97, device=sk.device)
        fpr = (sk.unsqueeze(1) > grid).float().mean(dim=0)
        tpr = (su.unsqueeze(1) > grid).float().mean(dim=0)
        j = tpr - fpr
        j[fpr > fpr_cap] = -1.0

        if j.max() < 0:
            # No grid point under cap — pick the lowest-FPR point.
            thr = float(grid[fpr.argmin()].item())
        else:
            thr = float(grid[j.argmax()].item())

        info["global_thr"] = thr
        info["val_tpr_at_t"] = float(tpr[(grid - thr).abs().argmin()].item())
        info["val_fpr_at_t"] = float(fpr[(grid - thr).abs().argmin()].item())

        if verbose:
            print(f"[OsrSAF_TriNet] Youden global threshold: t = {thr:.4f} | "
                  f"val TPR = {info['val_tpr_at_t']:.3f} | val FPR = {info['val_fpr_at_t']:.3f}")

        self.class_thresholds.fill_(max(0.05, min(0.95, thr)))
        return info


    def _backbone_outputs(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            want_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (fingerprint_256, logits, drsn_mask_or_None)."""
        f1 = self.base.stft_branch(x_stft)
        f2 = torch.flatten(self.base.iq_branch(x_iq), 1)
        f3 = torch.flatten(self.base.if_branch(x_if), 1)

        f1, f2, f3 = self.base._modality_dropout([f1, f2, f3])

        if want_mask:
            fp, mask = self.base._fuse(f1, f2, f3, return_mask=True)
        else:
            fp = self.base._fuse(f1, f2, f3)
            mask = None

        logits = self.base.classifier(fp)
        return fp, logits, mask

    def collect_and_update(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            labels: torch.Tensor,
            epoch: int = 1,
    ) -> None:
        """Phase 1 step: update both codebooks for a known batch (frozen backbone)."""
        fp, _, drsn_mask = self._backbone_outputs(x_stft, x_iq, x_if, want_mask=True)

        code = F.normalize(fp.detach(), p=2, dim=1)

        current_momentum = min(
            self._ema_mom,
            0.85 + (self._ema_mom - 0.85) * (epoch / max(1, self.warmup_epochs)),
        )
        self._codebook.update(code, labels, current_momentum=current_momentum)
        self._hamming_codebook.update(drsn_mask.detach(), labels, current_momentum=current_momentum)

    def forward_with_osr_logits(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, unknown_score, unknown_logit) — logit is pre-sigmoid for BCE."""
        if x_stft.ndim != 4 or x_stft.shape[1] != 2:
            raise ValueError(f"Expected x_stft (N,2,F,T) [log_mag, d_phi], got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 3:
            raise ValueError(f"Expected x_iq (N,3,L) [real, imag, abs], got {tuple(x_iq.shape)}")
        if x_if.ndim != 3 or x_if.shape[1] != 1:
            raise ValueError(f"Expected x_if (N,1,L), got {tuple(x_if.shape)}")

        fp, logits, drsn_mask = self._backbone_outputs(x_stft, x_iq, x_if, want_mask=True)
        code = F.normalize(fp, p=2, dim=1)

        # Top-2 logits give pred, runner-up, and a logit margin.
        top2_vals, top2_idx = logits.topk(2, dim=1)
        pred_class = top2_idx[:, 0]
        runner_up_class = top2_idx[:, 1]
        logit_margin = top2_vals[:, 0] - top2_vals[:, 1]
        logit_margin_squashed = torch.tanh(logit_margin / 5.0).clamp(0.0, 1.0)

        # Cosine distances: predicted class and runner-up.
        all_dists = self._codebook.code_distance_all_classes(code)
        b_idx = torch.arange(all_dists.size(0), device=all_dists.device)
        code_dist = all_dists[b_idx, pred_class]
        runner_up_dist = all_dists[b_idx, runner_up_class]
        margin_codebook = (runner_up_dist - code_dist).clamp(min=0.0)

        # Hamming distances over the FusedDRSN binary survival mask.
        all_h = self._hamming_codebook.hamming_distance_all_classes(drsn_mask)
        hamming_dist_pred = all_h[b_idx, pred_class]
        hamming_dist_runner = all_h[b_idx, runner_up_class]
        hamming_margin = (hamming_dist_runner - hamming_dist_pred).clamp(min=0.0)

        # Softmax confidence.
        max_prob = logits.softmax(dim=1).max(dim=1).values
        unc = 1.0 - max_prob

        calib_input = torch.stack(
            [code_dist, unc,
             runner_up_dist, margin_codebook, logit_margin_squashed,
             hamming_dist_pred, hamming_margin],
            dim=1,
        )

        unknown_logit = self.score_calibrator(calib_input).squeeze(1)
        unknown_score = torch.sigmoid(unknown_logit)

        return logits, unknown_score, unknown_logit

    def forward_with_osr(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference forward: returns (logits, unknown_score)."""
        logits, unknown_score, _ = self.forward_with_osr_logits(x_stft, x_iq, x_if)
        return logits, unknown_score

    def forward(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_osr(x_stft, x_iq, x_if)
        return logits

    def predict_with_rejection(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, unknown_score = self.forward_with_osr(x_stft, x_iq, x_if)

        probs = logits.softmax(dim=1)
        confidence, predictions = probs.max(dim=1)

        predictions = predictions.clone()
        thresh = self.class_thresholds[predictions]
        predictions[unknown_score > thresh] = -1

        return predictions, confidence

    @torch.no_grad()
    def extract_embedding(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the 256-D pre-SupCon fingerprint for t-SNE / diagnostics."""
        fp, _, _ = self._backbone_outputs(x_stft, x_iq, x_if, want_mask=False)
        return fp
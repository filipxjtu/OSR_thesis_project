from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asymmetric_trinet import AsymmetricTriNet


"""
OsrSAF_TriNet — Sparse Activation Fingerprint OSR built on AsymmetricTriNet.

Lineage of the idea (cited in the thesis, not in code):
  - Bricken et al. 2023 — sparse activation patterns as structural fingerprints
    of what a network has learned about an input.
  - van den Oord et al. 2017 — VQ-VAE EMA codebook update.
  - Bendale & Boult 2016 — OpenMax: penultimate activations as class identity
    signatures for OSR without external unknown labels.

What this file is, in plain terms:
  - The closed-set AsymmetricTriNet already produces a 256-D fingerprint (after
    DRSN refinement) and a 128-D L2-normalized SupCon projection.
  - We treat the 256-D fingerprint, L2-normalized, as the per-sample identity
    code. Per class, we keep an EMA codebook of k centroids in this space.
  - At inference: cosine distance from the predicted class' nearest centroid is
    the OSR signal. A small MLP calibrator combines this distance with softmax
    confidence and embedding norm into a [0,1] unknown-score.
  - Training is two-phase: P1 trains the backbone (CE + SupCon) and populates
    the codebook online. P2 freezes the backbone and trains the calibrator on
    proxy unknowns. Per-class thresholds are recalibrated periodically in P2.

Why pre-SupCon fingerprint and not the SupCon projection itself:
  - SupCon is trained only on knowns and only with same-vs-different-class
    signal. It has no incentive to push unknowns away from the unit sphere.
  - In low-noise regimes, the 256-D fingerprint preserves off-manifold variance
    that an unknown is more likely to occupy. That structure survives in the
    fingerprint but gets folded back near a known cluster after projection.
  - Using the pre-projection fingerprint with cosine distance gives us the
    discriminative shape CE+SupCon trained, plus the unknown-detection room
    that the projection would have collapsed.
"""


class _CosineCodebook(nn.Module):
    """
    Per-class, k-centroid EMA codebook over L2-normalized embeddings.

    Distances are cosine distances in [0, 2]. Centroids are stored unnormalized
    after EMA accumulation so ratios stay stable; we re-normalize on read for
    cosine-distance computation.
    """

    def __init__(
            self,
            num_classes: int,
            code_dim: int,
            k: int = 4,
            ema_momentum: float = 0.95,
            beta: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.k = k
        self.ema_momentum = ema_momentum
        self.beta = beta

        # Centroids stored on a small random init so the first cosine call is well-defined.
        # The `initialised` flag controls when a centroid is "real" (filled with a sample).
        init = F.normalize(torch.randn(num_classes, k, code_dim), p=2, dim=-1) * 1e-3
        self.register_buffer("centroids", init)
        self.register_buffer("initialised", torch.zeros(num_classes, k, dtype=torch.bool))
        self.register_buffer("update_counts", torch.zeros(num_classes, k, dtype=torch.long))

    @torch.no_grad()
    def _normed_centroids(self, c_idx: int) -> torch.Tensor:
        return F.normalize(self.centroids[c_idx], p=2, dim=-1)

    @torch.no_grad()
    def update(self, codes: torch.Tensor, labels: torch.Tensor, current_momentum: float = 0.95):
        """
        codes:  (B, D) — L2-normalized.
        labels: (B,)   — known class indices (callers must filter out -1 first).

        For each class in the batch:
          - Fill empty centroids with samples one-by-one (cold-start).
          - Once full, assign each remaining sample to its nearest centroid by
            cosine distance, with an outlier guard (beta-scaled std clip) to
            avoid letting noisy outliers drag a centroid.
          - EMA-update each centroid toward the mean of its assignment.
        """
        for c in labels.unique():
            cid = int(c.item())
            if cid == -1:
                continue

            class_codes = codes[labels == c]
            if class_codes.numel() == 0:
                continue

            # Cold-start: fill empty centroids first
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

            # All k centroids are now alive — assign remaining samples by nearest centroid.
            cents_normed = self._normed_centroids(cid)                          # (k, D)
            sim = class_codes @ cents_normed.t()                                # (n, k)  cosine sim
            dists = 1.0 - sim                                                   # cosine distance ∈ [0, 2]
            nearest = dists.argmin(dim=1)                                       # (n,)

            for kid in range(self.k):
                mask = nearest == kid
                assigned = class_codes[mask]
                if assigned.shape[0] == 0:
                    continue

                # Outlier guard: drop the long tail of the assignment block so a single
                # noisy hit can't pull the centroid. Mirrors the spirit of the original
                # codebook's beta-scaled std clip, just in cosine space.
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
    def code_distance(self, codes: torch.Tensor, pred_class: torch.Tensor) -> torch.Tensor:
        """
        codes:      (B, D) — L2-normalized.
        pred_class: (B,)   — predicted class index per sample.
        Returns: (B,) cosine distance to the nearest centroid of the predicted class.
        """
        # Per-sample: gather the k centroids of the predicted class, normalize, distance.
        # Vectorized: build (B, k, D), distance against codes, take min over k.
        cents = self.centroids[pred_class]                                      # (B, k, D)
        cents_normed = F.normalize(cents, p=2, dim=-1)
        sim = (codes.unsqueeze(1) * cents_normed).sum(dim=-1)                   # (B, k)
        dists = 1.0 - sim                                                       # cosine distance
        return dists.min(dim=1).values                                          # (B,)

    def convergence_stats(self) -> Dict[str, torch.Tensor]:
        # `spread_per_class`: mean pairwise cosine distance among that class' k centroids.
        # Small spread -> centroids have collapsed to one mode (or are still unfilled).
        with torch.no_grad():
            normed = F.normalize(self.centroids, p=2, dim=-1)                   # (C, k, D)
            sim = normed @ normed.transpose(1, 2)                               # (C, k, k)
            # off-diagonal mean
            mask = 1.0 - torch.eye(self.k, device=sim.device).unsqueeze(0)
            denom = max(1, self.k * (self.k - 1))
            spread = ((1.0 - sim) * mask).sum(dim=(1, 2)) / denom               # (C,)

        return {
            "mean_activation_per_class": self.centroids.norm(dim=-1).mean(dim=1),
            "spread_per_class": spread,
            "pct_initialised": self.initialised.float().mean(),
            "mean_updates_per_centroid": self.update_counts.float().mean(),
        }


class OsrSAF_TriNet(nn.Module):
    """
    Sparse Activation Fingerprint OSR on top of AsymmetricTriNet.

    Phase 1: train backbone with CE + SupCon (delegated to the trinet's joint
             forward), and populate the per-class cosine codebook online.
    Phase 2: freeze backbone, train the score_calibrator on proxy unknowns,
             periodically recalibrate per-class thresholds.

    The public API matches the old SparseFingerprint_TS_DRSN so osr_engine.py
    and osr_diagnostics.py only need their tuple unpacking widened from
    (x_stft, x_iq) to (x_stft, x_iq, x_if).
    """

    def __init__(
            self,
            num_classes: int = 10,
            k_centroids: int = 4,
            ema_momentum: float = 0.95,
            warmup_epochs: int = 30,
            codebook_beta: float = 1.0,
            threshold_recal_interval: int = 5,
            # AsymmetricTriNet pass-through
            branch_dim: int = 128,
            fingerprint_dim: int = 256,
            modality_dropout: float = 0.1,
            num_transformer_layers: int = 2,
            nhead: int = 4,
            use_cls_token: bool = True,
            supcon_dim: int = 128,
            # Pretrained closed-set checkpoint
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

        # Backbone
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

        # Codebook lives in the L2-normalized 256-D fingerprint space
        self._fingerprint_dim = fingerprint_dim
        self._codebook = _CosineCodebook(
            num_classes=num_classes,
            code_dim=fingerprint_dim,
            k=k_centroids,
            ema_momentum=ema_momentum,
            beta=codebook_beta,
        )

        # Score calibrator: [code_dist, 1 - max_prob, emb_norm_normalised] -> [0, 1].
        # LayerNorm so the three features arrive on comparable scales.
        self.score_calibrator = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.phase2_active = False
        self.register_buffer("class_thresholds", torch.full((num_classes,), 0.5))

    # -----------------------------------------------------------------
    # Phase control
    # -----------------------------------------------------------------
    def current_phase(self) -> int:
        return 2 if self.phase2_active else 1

    def set_phase(self):
        if self.current_phase() == 1:
            self._unfreeze_backbone()
            self._freeze_calibrator()
        else:
            self._freeze_backbone()
            self._unfreeze_calibrator()

    def check_dynamic_phase_switch(self, epoch: int) -> bool:
        if self.phase2_active:
            return True

        if epoch >= self.warmup_epochs:
            self.phase2_active = True
            self.calibrate_class_thresholds()
            return True

        # Earliest a meaningful "converged" check makes sense
        if epoch < 23:
            return False

        stats = self.get_codebook_stats()
        if stats is None:
            return False

        pct_init = float(stats["pct_initialised"])
        mean_updates = float(stats["mean_updates_per_centroid"])
        spread = float(stats["spread_per_class"].mean())

        # NOTE on the spread threshold: we now use cosine spread (not L1 magnitude),
        # so the "small spread" criterion lives in [0, 2]. A class whose k centroids
        # have collapsed to one mode has spread ≈ 0; well-separated multi-mode
        # classes can sit comfortably above 0.1. The 0.05 cap below mirrors the
        # spirit of the old check: only trigger when centroids have genuinely
        # settled (not just initialised to identical values).
        genuinely_converged = (
            pct_init >= 1.0
            and mean_updates > self._k
            and spread < 0.05
        )

        if genuinely_converged:
            self.phase2_active = True
            self.calibrate_class_thresholds()
            return True

        return False

    @torch.no_grad()
    def calibrate_class_thresholds(self, base_threshold: float = 0.5):
        spreads = self._codebook.convergence_stats()["spread_per_class"]
        norm_spreads = spreads / (spreads.max() + 1e-6)
        # Tighter classes (low spread) → slightly lower threshold (easier to reject deviations).
        # Looser classes (high spread) → slightly higher threshold (more tolerant).
        adjusted = base_threshold * (0.8 + 0.4 * (1 - norm_spreads))
        self.class_thresholds.copy_(adjusted.clamp(0.15, 0.85))

    # -----------------------------------------------------------------
    # Freezing helpers
    # -----------------------------------------------------------------
    def _freeze_backbone(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self):
        for p in self.base.parameters():
            p.requires_grad = True

    def _freeze_calibrator(self):
        for p in self.score_calibrator.parameters():
            p.requires_grad = False

    def _unfreeze_calibrator(self):
        for p in self.score_calibrator.parameters():
            p.requires_grad = True

    # Convenience aliases the trainer expects
    def freeze_base(self):
        self._freeze_backbone()

    def unfreeze_base(self):
        self._unfreeze_backbone()

    # -----------------------------------------------------------------
    # Codebook accessors
    # -----------------------------------------------------------------
    def codebook_ready(self) -> bool:
        return bool(self._codebook.initialised.all().item())

    def get_codebook_stats(self) -> Optional[Dict]:
        return self._codebook.convergence_stats()

    # -----------------------------------------------------------------
    # Forward variants
    # -----------------------------------------------------------------
    def _backbone_outputs(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            want_supcon: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (fingerprint_256, logits, supcon_z_or_None).

        Re-implements the trinet's forward path so we can pull the 256-D
        fingerprint out without doing two forward passes. Mirrors the trinet's
        own `forward` exactly — modality dropout, fusion, classifier — but
        also exposes `fp` and (optionally) the SupCon projection.
        """
        f1 = self.base.stft_branch(x_stft)
        f2 = torch.flatten(self.base.iq_branch(x_iq), 1)
        f3 = torch.flatten(self.base.if_branch(x_if), 1)

        f1, f2, f3 = self.base._modality_dropout([f1, f2, f3])

        fp = self.base._fuse(f1, f2, f3)                                        # (B, 256)
        logits = self.base.classifier(fp)                                       # (B, num_classes)

        if want_supcon:
            z = F.normalize(self.base.supcon_head(fp), p=2, dim=1)              # (B, supcon_dim)
            return fp, logits, z

        return fp, logits, None

    def forward_phase1(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            labels: torch.Tensor,
            epoch: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase-1 training step. Returns (logits, supcon_z) for joint CE+SupCon
        loss. Side-effect: updates the codebook with this batch's L2-normalized
        fingerprints (only known samples; callers must have filtered already).
        """
        fp, logits, z = self._backbone_outputs(x_stft, x_iq, x_if, want_supcon=True)

        # L2-normalize the fingerprint for cosine codebook update
        code = F.normalize(fp.detach(), p=2, dim=1)

        # EMA momentum ramp: start lower so cold-start centroids move fast,
        # cap at the configured value once warmed up. Same shape as the old code.
        current_momentum = min(self._ema_mom, 0.85 + (self._ema_mom - 0.85) * (epoch / max(1, self.warmup_epochs)))
        self._codebook.update(code, labels, current_momentum=current_momentum)

        return logits, z

    # Kept for naming compatibility with the old trainer signature
    def collect_and_update(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
            labels: torch.Tensor,
            epoch: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_phase1(x_stft, x_iq, x_if, labels, epoch=epoch)

    def forward_with_osr(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Inference / Phase-2 forward.
        Returns (logits, unknown_score, None). Third slot kept for API parity.
        """
        if x_stft.ndim != 4 or x_stft.shape[1] != 2:
            raise ValueError(f"Expected x_stft (N,2,F,T) [log_mag, d_phi], got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 3:
            raise ValueError(f"Expected x_iq (N,3,L) [real, imag, abs], got {tuple(x_iq.shape)}")
        if x_if.ndim != 3 or x_if.shape[1] != 1:
            raise ValueError(f"Expected x_if (N,1,L), got {tuple(x_if.shape)}")

        fp, logits, _ = self._backbone_outputs(x_stft, x_iq, x_if, want_supcon=False)

        code = F.normalize(fp, p=2, dim=1)
        pred_class = logits.argmax(dim=1)
        code_dist = self._codebook.code_distance(code, pred_class)              # cosine dist ∈ [0, 2]

        max_prob = logits.softmax(dim=1).max(dim=1).values

        emb_norm = fp.norm(dim=1)
        emb_norm_normalised = (emb_norm / (emb_norm.detach().mean() + 1e-6)).clamp(0, 3) / 3.0

        calib_input = torch.stack([code_dist, 1.0 - max_prob, emb_norm_normalised], dim=1)
        unknown_score = self.score_calibrator(calib_input).squeeze(1)

        return logits, unknown_score, None

    def forward(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> torch.Tensor:
        logits, _, _ = self.forward_with_osr(x_stft, x_iq, x_if)
        return logits

    def predict_with_rejection(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, unknown_score, _ = self.forward_with_osr(x_stft, x_iq, x_if)

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
        """Returns the 256-D pre-SupCon fingerprint, for t-SNE / diagnostics."""
        fp, _, _ = self._backbone_outputs(x_stft, x_iq, x_if, want_supcon=False)
        return fp
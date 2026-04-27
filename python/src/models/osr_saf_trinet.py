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
    the OSR signal, AS IS the margin between the predicted class and the
    runner-up class. A small MLP calibrator combines these and a few softmax
    features into a [0,1] unknown-score.
  - Training is two-phase: P1 fills the per-class cosine codebook (frozen
    backbone). P2 trains the calibrator on proxy unknowns. Per-class
    thresholds are recalibrated from the ACTUAL score distribution on the
    validation knowns (data-driven, not formula-driven).

Why pre-SupCon fingerprint and not the SupCon projection itself:
  - SupCon is trained only on knowns and only with same-vs-different-class
    signal. It has no incentive to push unknowns away from the unit sphere.
  - In low-noise regimes, the 256-D fingerprint preserves off-manifold variance
    that an unknown is more likely to occupy. That structure survives in the
    fingerprint but gets folded back near a known cluster after projection.
  - Using the pre-projection fingerprint with cosine distance gives us the
    discriminative shape CE+SupCon trained, plus the unknown-detection room
    that the projection would have collapsed.

Diagnostic notes (for the bug fix in this revision):
  - Original calibrator received only the distance to the PREDICTED class.
    It had no way to see the discriminative margin between the predicted class
    and the next-best class — which is the actual OSR signal: a known sample
    is close to its true class AND comfortably far from all others, while an
    unknown is roughly equidistant from several classes.
  - Original calibrator used `LayerNorm(3)` at input. With only 3 features,
    LN normalizes ACROSS the 3 features per sample, destroying absolute scale
    information that IS the OSR signal (e.g. "code_dist=0.05 small" vs
    "code_dist=0.50 large" become similar relative-shape patterns after LN).
  - Original threshold calibration was a hand-crafted formula based on
    codebook spread, never looking at the actual unknown_score distribution
    on knowns. Replaced with percentile-from-validation-knowns calibration
    targeting a configurable false-alarm rate.
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
        cents = self.centroids[pred_class]                                      # (B, k, D)
        cents_normed = F.normalize(cents, p=2, dim=-1)
        sim = (codes.unsqueeze(1) * cents_normed).sum(dim=-1)                   # (B, k)
        dists = 1.0 - sim                                                       # cosine distance
        return dists.min(dim=1).values                                          # (B,)

    @torch.no_grad()
    def code_distance_all_classes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (B, D) — L2-normalized.
        Returns: (B, C) cosine distance to the nearest centroid of EACH class.

        This is the discriminative-margin signal: for an unknown, distances to
        the top-2 classes are typically close (ambiguous); for a clean known,
        the predicted-class distance is small and all others are large.
        """
        # centroids: (C, k, D) → normalize on read
        cents_normed = F.normalize(self.centroids, p=2, dim=-1)                 # (C, k, D)
        # codes: (B, D); we want sim of every code to every (c, k) centroid.
        # sim[b, c, k] = codes[b] · cents_normed[c, k]
        sim = torch.einsum("bd,ckd->bck", codes, cents_normed)                  # (B, C, k)
        dists = 1.0 - sim                                                       # cosine distance
        return dists.min(dim=-1).values                                         # (B, C) per-class min

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


# Number of input features the calibrator MLP consumes.
# 1. code_dist           — cosine dist to nearest centroid of predicted class (was original)
# 2. 1 - max_prob        — softmax uncertainty                                 (was original)
# 3. emb_norm_normalised — fingerprint magnitude relative to batch mean        (was original)
# 4. runner_up_dist      — cosine dist to nearest centroid of runner-up class  (NEW)
# 5. margin              — runner_up_dist - code_dist (≥ 0 when classifier
#                          agrees with codebook; near 0 when ambiguous)        (NEW)
# 6. logit_margin        — top-1 logit minus top-2 logit                       (NEW)
_CALIB_INPUT_DIM = 6


class OsrSAF_TriNet(nn.Module):
    """
    Sparse Activation Fingerprint OSR on top of AsymmetricTriNet.

    Phase 1: load pretrained backbone, populate the per-class cosine codebook
             (frozen backbone, codebook fill only).
    Phase 2: freeze backbone, train the score_calibrator on proxy unknowns,
             periodically recalibrate per-class thresholds from the validation
             score distribution.

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

        # Score calibrator: 6 features → unknown-score logit.
        # The final sigmoid is applied OUTSIDE the module so the loss can
        # use BCEWithLogitsLoss (numerically stable, no vanishing-gradient
        # near saturation). We do NOT prepend any LayerNorm — with only a
        # handful of features, LN normalizes across features per sample
        # and destroys the absolute-scale signal we actually care about.
        self.score_calibrator = nn.Sequential(
            nn.Linear(_CALIB_INPUT_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
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
            # Initial threshold uses the formula fallback; once Phase 2 has
            # produced any scores, the trainer will switch to the percentile
            # method via calibrate_class_thresholds_from_scores().
            self.calibrate_class_thresholds_formula()
            return True

        if epoch < 23:
            return False

        stats = self.get_codebook_stats()
        if stats is None:
            return False

        pct_init = float(stats["pct_initialised"])
        mean_updates = float(stats["mean_updates_per_centroid"])
        spread = float(stats["spread_per_class"].mean())

        genuinely_converged = (
            pct_init >= 1.0
            and mean_updates > self._k
            and spread < 0.05
        )

        if genuinely_converged:
            self.phase2_active = True
            self.calibrate_class_thresholds_formula()
            return True

        return False

    # -----------------------------------------------------------------
    # Threshold calibration — FORMULA fallback (legacy, used as P2 init)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def calibrate_class_thresholds_formula(self, base_threshold: float = 0.5):
        """
        Spread-based fallback used ONLY for the very first Phase-2 step,
        before any unknown_scores have been observed. Once we have scores,
        switch to calibrate_class_thresholds_from_scores().
        """
        spreads = self._codebook.convergence_stats()["spread_per_class"]
        norm_spreads = spreads / (spreads.max() + 1e-6)
        # Tighter classes (low spread) → slightly lower threshold.
        # Looser classes (high spread) → slightly higher threshold.
        adjusted = base_threshold * (0.8 + 0.4 * (1 - norm_spreads))
        self.class_thresholds.copy_(adjusted.clamp(0.15, 0.85))

    # Backward-compat alias — old trainer / external code may call this name
    @torch.no_grad()
    def calibrate_class_thresholds(self, base_threshold: float = 0.5):
        self.calibrate_class_thresholds_formula(base_threshold=base_threshold)

    # -----------------------------------------------------------------
    # Threshold calibration — DATA-DRIVEN (the new path)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def calibrate_class_thresholds_from_scores(
            self,
            scores: torch.Tensor,
            pred_classes: torch.Tensor,
            target_fpr: float = 0.10,
            min_per_class: int = 30,
    ) -> None:
        """
        Set per-class thresholds from the empirical score distribution on
        validation KNOWNS. For each class c, the threshold is the
        (1 - target_fpr) percentile of unknown_scores assigned to predicted
        class c. By construction, at this threshold the false-alarm rate on
        the validation knowns is ~target_fpr per class.

        scores:       (N,) calibrator outputs (sigmoid-applied, in [0,1]).
        pred_classes: (N,) predicted class for each sample.
        target_fpr:   fraction of validation knowns we accept rejecting (e.g. 0.10).
        min_per_class: if a class has fewer than this many samples, fall back
                       to the global percentile across all knowns instead.
        """
        scores = scores.detach().to(self.class_thresholds.device).float()
        pred_classes = pred_classes.detach().to(self.class_thresholds.device).long()

        if scores.numel() == 0:
            return

        # Global percentile as a sane fallback.
        q = max(0.0, min(1.0, 1.0 - target_fpr))
        global_thr = float(torch.quantile(scores, q).item())

        new_thresh = self.class_thresholds.clone()
        for c in range(self.num_classes):
            mask = pred_classes == c
            n = int(mask.sum().item())
            if n < min_per_class:
                new_thresh[c] = global_thr
                continue
            class_scores = scores[mask]
            new_thresh[c] = torch.quantile(class_scores, q)

        # Clamp to keep numerics tame; nothing else.
        self.class_thresholds.copy_(new_thresh.clamp(0.05, 0.95))

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
        own `forward` exactly — modality dropout (gated on self.training,
        which is False once base.eval() is called), fusion, classifier — but
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
        fp, logits, z = self._backbone_outputs(x_stft, x_iq, x_if, want_supcon=True)

        code = F.normalize(fp.detach(), p=2, dim=1)

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

    # -----------------------------------------------------------------
    # OSR-aware forward
    # -----------------------------------------------------------------
    def forward_with_osr(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Inference / Phase-2 forward.
        Returns (logits, unknown_score, None).
            unknown_score is sigmoid-applied, in [0, 1] — public API contract.

        Internally, the calibrator outputs LOGITS (no sigmoid in the head);
        the loss in osr_utils.combined_loss can therefore use BCEWithLogitsLoss
        for numerical stability. We apply sigmoid here so the rest of the
        pipeline (predict_with_rejection, evaluator, threshold comparisons)
        keeps working unchanged.
        """
        if x_stft.ndim != 4 or x_stft.shape[1] != 2:
            raise ValueError(f"Expected x_stft (N,2,F,T) [log_mag, d_phi], got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 3:
            raise ValueError(f"Expected x_iq (N,3,L) [real, imag, abs], got {tuple(x_iq.shape)}")
        if x_if.ndim != 3 or x_if.shape[1] != 1:
            raise ValueError(f"Expected x_if (N,1,L), got {tuple(x_if.shape)}")

        fp, logits, _ = self._backbone_outputs(x_stft, x_iq, x_if, want_supcon=False)

        code = F.normalize(fp, p=2, dim=1)

        # ----- Top-2 logits give us pred + runner-up + a "logit margin" -----
        top2_vals, top2_idx = logits.topk(2, dim=1)                             # (B, 2), (B, 2)
        pred_class      = top2_idx[:, 0]
        runner_up_class = top2_idx[:, 1]
        logit_margin    = top2_vals[:, 0] - top2_vals[:, 1]                     # (B,)
        # Squash logit margin to roughly [0, 1] via tanh; well-separated knowns
        # land near 1, ambiguous samples near 0.
        logit_margin_squashed = torch.tanh(logit_margin / 5.0).clamp(0.0, 1.0)

        # ----- Cosine distances: predicted class + runner-up class -----
        # Computing per-class min over the codebook is cheap (10 classes * 4 centroids).
        all_dists = self._codebook.code_distance_all_classes(code)              # (B, C)
        b_idx = torch.arange(all_dists.size(0), device=all_dists.device)
        code_dist        = all_dists[b_idx, pred_class]                         # (B,)
        runner_up_dist   = all_dists[b_idx, runner_up_class]                    # (B,)
        margin_codebook  = (runner_up_dist - code_dist).clamp(min=0.0)          # (B,)

        # ----- Softmax confidence -----
        max_prob = logits.softmax(dim=1).max(dim=1).values                      # (B,)
        unc      = 1.0 - max_prob                                               # (B,)

        # ----- Embedding norm (kept as-is; weak but harmless feature) -----
        emb_norm = fp.norm(dim=1)
        emb_norm_normalised = (emb_norm / (emb_norm.detach().mean() + 1e-6)).clamp(0, 3) / 3.0

        calib_input = torch.stack(
            [
                code_dist,                # 1. distance to predicted class
                unc,                      # 2. 1 - max_prob
                emb_norm_normalised,      # 3. embedding magnitude (weak)
                runner_up_dist,           # 4. distance to runner-up class  (NEW)
                margin_codebook,          # 5. codebook margin              (NEW)
                logit_margin_squashed,    # 6. logit margin (squashed)      (NEW)
            ],
            dim=1,
        )                                                                       # (B, 6)

        unknown_logit = self.score_calibrator(calib_input).squeeze(1)           # (B,)
        unknown_score = torch.sigmoid(unknown_logit)                            # (B,)  ∈ [0, 1]

        # Stash the logit on the tensor's underlying object via an attribute
        # WOULD break autograd; instead, callers that need the logit should
        # use forward_with_osr_logits() below.
        return logits, unknown_score, None

    def forward_with_osr_logits(
            self,
            x_stft: torch.Tensor,
            x_iq: torch.Tensor,
            x_if: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as forward_with_osr, but ALSO returns the pre-sigmoid logit so
        the loss can use BCEWithLogitsLoss for numerical stability.
        Returns (logits, unknown_score, unknown_logit).
        """
        if x_stft.ndim != 4 or x_stft.shape[1] != 2:
            raise ValueError(f"Expected x_stft (N,2,F,T) [log_mag, d_phi], got {tuple(x_stft.shape)}")
        if x_iq.ndim != 3 or x_iq.shape[1] != 3:
            raise ValueError(f"Expected x_iq (N,3,L) [real, imag, abs], got {tuple(x_iq.shape)}")
        if x_if.ndim != 3 or x_if.shape[1] != 1:
            raise ValueError(f"Expected x_if (N,1,L), got {tuple(x_if.shape)}")

        fp, logits, _ = self._backbone_outputs(x_stft, x_iq, x_if, want_supcon=False)
        code = F.normalize(fp, p=2, dim=1)

        top2_vals, top2_idx = logits.topk(2, dim=1)
        pred_class      = top2_idx[:, 0]
        runner_up_class = top2_idx[:, 1]
        logit_margin    = top2_vals[:, 0] - top2_vals[:, 1]
        logit_margin_squashed = torch.tanh(logit_margin / 5.0).clamp(0.0, 1.0)

        all_dists = self._codebook.code_distance_all_classes(code)
        b_idx = torch.arange(all_dists.size(0), device=all_dists.device)
        code_dist       = all_dists[b_idx, pred_class]
        runner_up_dist  = all_dists[b_idx, runner_up_class]
        margin_codebook = (runner_up_dist - code_dist).clamp(min=0.0)

        max_prob = logits.softmax(dim=1).max(dim=1).values
        unc      = 1.0 - max_prob

        emb_norm = fp.norm(dim=1)
        emb_norm_normalised = (emb_norm / (emb_norm.detach().mean() + 1e-6)).clamp(0, 3) / 3.0

        calib_input = torch.stack(
            [code_dist, unc, emb_norm_normalised,
             runner_up_dist, margin_codebook, logit_margin_squashed],
            dim=1,
        )

        unknown_logit = self.score_calibrator(calib_input).squeeze(1)
        unknown_score = torch.sigmoid(unknown_logit)

        return logits, unknown_score, unknown_logit

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
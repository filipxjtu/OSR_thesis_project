"""
python/src/train/openmax_trainer.py
====================================
Fits an OpenMaxTriNet on top of an already-trained AsymmetricTriNet checkpoint.

Pipeline:
  1. Load the closed-set asymmetric_trinet checkpoint into a frozen backbone.
  2. Stream the OSR training set through the backbone, collect AVs for knowns.
  3. Compute per-class MAV and fit per-class Weibull on the tail of distances.
  4. Calibrate a single global rejection threshold on val_known + val_unknown
     using Youden's J under an FPR cap (matches OsrSAF_TriNet's calibration).
  5. Evaluate on test_known + test_unknown (the held-out unknown file).
  6. Save the OpenMax checkpoint and a JSON training log.

Naming:
  - Checkpoint: artifacts/checkpoints/openmax_trinet_seed{seed}_n{n}.pt
  - Log:        artifacts/logs/openmax_training/openmax_trinet_seed{seed}_n{n}.json

This mirrors the OsrSAF_TriNet trainer's filename convention so the evaluator
can find both side-by-side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..utils import (
    create_eval_loader,
    resolve_device,
    load_osr_datasets,
    prepare_unique_file,
)
from ..models import OpenMaxTriNet


# =============================================================================
# Hyperparameters
# =============================================================================

@dataclass(slots=True)
class OpenMaxHParams:
    # OpenMax algorithm
    alpha_rank: int = 10            # how many top classes get revised
    tail_size: int = 20             # Weibull tail size per class
    distance: str = "euclidean"     # "euclidean" or "cosine"
    only_correct: bool = True       # fit MAVs from correctly classified samples only

    # Threshold calibration
    fpr_cap: float = 0.4            # FPR cap for Youden (matches OsrSAF default)

    # Loader
    batch_size: int = 32


# =============================================================================
# Helpers
# =============================================================================

@torch.no_grad()
def _collect_avs(model: OpenMaxTriNet, loader, device, knowns_only: bool = True):
    """Run a loader through the backbone and return concatenated (avs, labels)."""
    model.base.eval()
    all_avs, all_labels = [], []
    for x_stft, x_iq, x_if, y in loader:
        if knowns_only:
            mask = y != -1
            if not mask.any():
                continue
            x_stft = x_stft[mask]; x_iq = x_iq[mask]; x_if = x_if[mask]; y_b = y[mask]
        else:
            y_b = y
        av = model.extract_av(x_stft.to(device), x_iq.to(device), x_if.to(device))
        all_avs.append(av.cpu())
        all_labels.append(y_b.cpu())
    if not all_avs:
        return torch.empty(0, model.num_classes), torch.empty(0, dtype=torch.long)
    return torch.cat(all_avs, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def _collect_unknown_scores(model: OpenMaxTriNet, loader, device):
    """Returns (unknown_scores, predicted_classes) for any loader."""
    model.eval()
    scores, preds = [], []
    for x_stft, x_iq, x_if, _ in loader:
        logits, score = model.forward_with_osr(x_stft.to(device), x_iq.to(device), x_if.to(device))
        scores.append(score.cpu())
        preds.append(logits.argmax(dim=1).cpu())
    if not scores:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(scores, dim=0), torch.cat(preds, dim=0)


@torch.no_grad()
def _eval_known_acc(model: OpenMaxTriNet, loader, device) -> float:
    correct, total = 0, 0
    for x_stft, x_iq, x_if, y in loader:
        logits = model.extract_av(x_stft.to(device), x_iq.to(device), x_if.to(device))
        correct += (logits.argmax(1) == y.to(device)).sum().item()
        total   += y.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _eval_osr_metrics(model: OpenMaxTriNet, loader_known, loader_unknown, device):
    """Returns (known_acc, auroc, unknown_recall, false_alarm_rate)."""
    all_labels, all_scores, all_preds = [], [], []
    for loader in (loader_known, loader_unknown):
        for x_stft, x_iq, x_if, y in loader:
            logits, score = model.forward_with_osr(x_stft.to(device), x_iq.to(device), x_if.to(device))
            all_labels.append(y.cpu().numpy())
            all_scores.append(score.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    scores_arr = np.concatenate(all_scores)
    preds_arr  = np.concatenate(all_preds)

    known_mask   = labels_arr != -1
    unknown_mask = labels_arr == -1

    binary_labels = np.zeros_like(labels_arr)
    binary_labels[unknown_mask] = 1

    try:
        auroc = float(roc_auc_score(binary_labels, scores_arr))
    except ValueError:
        auroc = 0.5

    rejected = scores_arr > float(model.threshold.item())
    known_acc        = float(np.mean(preds_arr[known_mask] == labels_arr[known_mask])) if known_mask.any() else 0.0
    unknown_recall   = float(np.mean(rejected[unknown_mask]))                         if unknown_mask.any() else 0.0
    false_alarm_rate = float(np.mean(rejected[known_mask]))                           if known_mask.any() else 0.0
    return known_acc, auroc, unknown_recall, false_alarm_rate


# =============================================================================
# Main training function
# =============================================================================

def train_openmax_model(
    *,
    seed: int,
    n_per_class: int,
    spec_version: str,
    project_root: Path,
    hparams: Optional[OpenMaxHParams] = None,
):
    if hparams is None:
        hparams = OpenMaxHParams()

    pretrained_path = (
        project_root / "artifacts" / "checkpoints"
        / f"asymmetric_trinet_seed{seed}_n{n_per_class}.pt"
    )
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Closed-set checkpoint not found: {pretrained_path}\n"
            f"Train asymmetric_trinet first via train_model_runner."
        )

    torch.manual_seed(seed)
    device = resolve_device("auto")

    print(f"\n{'=' * 60}")
    print(f"OpenMaxTriNet | seed={seed} | n={n_per_class}")
    print(f"Device         : {device}")
    print(f"Closed-set ckpt: {pretrained_path.name}")
    print(f"alpha_rank     : {hparams.alpha_rank}")
    print(f"tail_size      : {hparams.tail_size}")
    print(f"distance       : {hparams.distance}")
    print(f"FPR cap        : {hparams.fpr_cap:.2f}")
    print(f"{'=' * 60}\n")

    datasets = load_osr_datasets(project_root, seed, n_per_class, spec_version)

    # OpenMax fitting only needs unshuffled passes — eval loaders for everything
    train_loader      = create_eval_loader(datasets["train"],        hparams.batch_size, device)
    val_loader_known  = create_eval_loader(datasets["val_known"],    hparams.batch_size, device)
    val_loader_osr    = create_eval_loader(datasets["val_unknown"],  hparams.batch_size, device)
    test_loader_known = create_eval_loader(datasets["test_known"],   hparams.batch_size, device)
    test_loader_osr   = create_eval_loader(datasets["test_unknown"], hparams.batch_size, device)

    model = OpenMaxTriNet(
        num_classes=10,
        alpha_rank=hparams.alpha_rank,
        tail_size=hparams.tail_size,
        distance=hparams.distance,
        use_pretrained=True,
        pretrained_path=str(pretrained_path),
    ).to(device)

    # Backbone is permanently frozen — OpenMax never trains weights
    for p in model.base.parameters():
        p.requires_grad = False
    model.base.eval()

    # ---- Stage 1: collect AVs over training knowns and fit MAVs/Weibulls -----
    print("[Stage 1] Collecting training AVs and fitting MAVs + Weibull tails")
    avs_train, labels_train = _collect_avs(model, train_loader, device, knowns_only=True)
    print(f"  collected {avs_train.size(0)} known training AVs")

    fit_info = model.fit_from_avs(
        avs_train, labels_train,
        only_correctly_classified=hparams.only_correct,
        verbose=True,
    )

    # ---- Stage 2: calibrate threshold on val knowns + proxy unknowns ---------
    print("\n[Stage 2] Calibrating rejection threshold (Youden's J under FPR cap)")
    sk, _ = _collect_unknown_scores(model, val_loader_known, device)
    su, _ = _collect_unknown_scores(model, val_loader_osr,   device)
    thr_info = model.calibrate_threshold_youden(sk, su, fpr_cap=hparams.fpr_cap, verbose=True)

    # ---- Stage 3: evaluate on the held-out test set --------------------------
    test_known_acc = _eval_known_acc(model, test_loader_known, device)
    _, test_auroc, test_recall, test_fpr = _eval_osr_metrics(
        model, test_loader_known, test_loader_osr, device,
    )

    print(f"\n{'=' * 52}")
    print(f"FINAL TEST RESULTS")
    print(f"Known accuracy  : {test_known_acc:.4f}")
    print(f"AUROC           : {test_auroc:.4f}")
    print(f"Unknown recall  : {test_recall:.4f}")
    print(f"False alarm rate: {test_fpr:.4f}")
    print(f"Youden's J      : {test_recall - test_fpr:+.4f}")
    print(f"Threshold       : {float(model.threshold):.4f}")
    print(f"{'=' * 52}\n")

    # ---- Save checkpoint and log --------------------------------------------
    ckpt_dir = project_root / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = prepare_unique_file(ckpt_dir, f"openmax_trinet_seed{seed}_n{n_per_class}.pt")
    torch.save(model.state_dict(), ckpt_path)

    log_dir = project_root / "artifacts" / "logs" / "openmax_training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = prepare_unique_file(log_dir, f"openmax_trinet_seed{seed}_n{n_per_class}.json")

    # JSON-safe per-class fit info (numpy scalars and bools)
    per_class_clean = {}
    for c, d in fit_info.get("per_class", {}).items():
        per_class_clean[str(c)] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                   for k, v in d.items()}

    log = {
        "created_utc":     datetime.now(timezone.utc).isoformat(),
        "seed":            seed,
        "n_per_class":     n_per_class,
        "pretrained_ckpt": pretrained_path.name,
        "hparams":         asdict(hparams),
        "fit_info": {
            "tail_size": fit_info.get("tail_size"),
            "distance":  fit_info.get("distance"),
            "per_class": per_class_clean,
        },
        "threshold_info":  thr_info,
        "test_metrics": {
            "test_acc":    test_known_acc,
            "test_auroc":  test_auroc,
            "test_recall": test_recall,
            "test_fpr":    test_fpr,
            "test_j":      test_recall - test_fpr,
            "threshold":   float(model.threshold),
        },
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=4)

    print(f"Checkpoint  : {ckpt_path}")
    print(f"Training log: {log_path}\n")

    return model
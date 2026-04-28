from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from python.src.dataio import load_artifact
from python.src.models import OsrSAF_TriNet
from python.src.preprocessing import build_feature_tensor
from python.src.utils import create_eval_loader, resolve_device
from python.src.analysis.osr_diagnostics import plot_osr_eval_feature_embedding


NUM_CLASSES = 10

# ============================================================================
# ADD THIS FUNCTION TO: python/src/eval/osr_evaluator.py
# Also add "evaluate_osr_model_with_tsne" to __all__ in python/src/eval/__init__.py
#
# This wraps evaluate_osr_model but keeps the model + loaders alive long
# enough to run the t-SNE embedding plot before returning.
# ============================================================================

# Required extra import at the top of osr_evaluator.py (if not already there):
#   from python.src.analysis import plot_osr_feature_embedding


def evaluate_osr_model_with_tsne(
    ckpt_seed: int,
    ckpt_n_per_class: int,
    eval_seed: int,
    eval_n_per_class: int,
    eval_spec_version: str,
    project_root: Path,
    fig_dir: Path,
    batch_size: int = 32,
    device_str: str = "auto",
    snr_label: str | None = None,      # e.g. "+4 dB"  — used in plot title
) -> dict:
    """
    Identical to evaluate_osr_model, but after computing metrics it also
    generates a t-SNE embedding plot of the eval split into `fig_dir`.

    Parameters
    ----------
    fig_dir    : Where to write  osr_feature_embedding.png  (and the
                 optional title-annotated variant).
    snr_label  : Human-readable SNR string appended to the t-SNE title,
                 e.g. "+4 dB".  Pass None to skip annotation.
    """
    from python.src.analysis.osr_diagnostics import plot_osr_feature_embedding

    device = resolve_device(device_str)

    # ------------------------------------------------------------------ #
    # 1.  Paths (mirror osr_evaluator.evaluate_osr_model exactly)
    # ------------------------------------------------------------------ #
    ckpt_path = (
        project_root
        / "artifacts"
        / "checkpoints"
        / f"osr_saf_trinet_seed{ckpt_seed}_n{ckpt_n_per_class}.pt"
    )
    eval_dataset_root = Path(f"C:/Users/user/Documents/MATLAB/eval_datasets")
    eval_known_path = (
        eval_dataset_root
        / "impaired"
        / f"impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_eval.mat"
    )
    eval_unknown_path = (
        eval_dataset_root
        / "unknown"
        / f"unknown_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}.mat"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not eval_known_path.exists():
        raise FileNotFoundError(f"Known eval dataset not found: {eval_known_path}")
    if not eval_unknown_path.exists():
        raise FileNotFoundError(f"Unknown eval dataset not found: {eval_unknown_path}")

    # ------------------------------------------------------------------ #
    # 2.  Model
    # ------------------------------------------------------------------ #
    model = OsrSAF_TriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Loaded checkpoint : {ckpt_path.name}")

    # ------------------------------------------------------------------ #
    # 3.  Datasets / loaders
    # ------------------------------------------------------------------ #
    known_artifact = load_artifact(str(eval_known_path), load_params=False)
    x_stft_k, x_iq_k, x_if_k, y_k = build_feature_tensor(known_artifact)
    known_dataset = torch.utils.data.TensorDataset(x_stft_k, x_iq_k, x_if_k, y_k)
    known_loader  = create_eval_loader(known_dataset, batch_size=batch_size, device=device)
    print(f"  Known samples     : {len(known_dataset)}")

    unknown_artifact = load_artifact(str(eval_unknown_path), load_params=False)
    x_stft_u, x_iq_u, x_if_u, _y_unk_orig = build_feature_tensor(unknown_artifact)
    y_unk_neg = torch.full((x_stft_u.size(0),), -1, dtype=torch.long)
    unknown_dataset = torch.utils.data.TensorDataset(x_stft_u, x_iq_u, x_if_u, y_unk_neg)
    unknown_loader  = create_eval_loader(unknown_dataset, batch_size=batch_size, device=device)
    print(f"  Unknown samples   : {len(unknown_dataset)}")

    # ------------------------------------------------------------------ #
    # 4.  Run the standard metric evaluation (re-use existing logic)
    # ------------------------------------------------------------------ #
    result = evaluate_osr_model(
        ckpt_seed=ckpt_seed,
        ckpt_n_per_class=ckpt_n_per_class,
        eval_seed=eval_seed,
        eval_n_per_class=eval_n_per_class,
        eval_spec_version=eval_spec_version,
        project_root=project_root,
        batch_size=batch_size,
        device_str=device_str,
    )

    # ------------------------------------------------------------------ #
    # 5.  t-SNE embedding plot
    # ------------------------------------------------------------------ #
    title_suffix = f" — SNR {snr_label}" if snr_label else ""
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating t-SNE embedding → {fig_dir}")
    plot_osr_eval_feature_embedding(
        model=model,
        loader_known=known_loader,
        loader_osr=unknown_loader,
        device=device,
        out_dir=fig_dir,
        n_classes=NUM_CLASSES,
        title_suffix=title_suffix,      # see note below
    )

    return result


def evaluate_osr_model(
    *,
    ckpt_seed: int,
    ckpt_n_per_class: int,
    eval_seed: int,
    eval_n_per_class: int,
    eval_spec_version: str,
    project_root: Path,
    batch_size: int = 64,
    device_str: str = "auto",
) -> dict:
    device = resolve_device(device_str)

    print(f"\n{'=' * 60}")
    print(f"Evaluating      : osr_saf_trinet")
    print(f"Checkpoint      : seed={ckpt_seed}, n_per_class={ckpt_n_per_class}")
    print(f"Eval datasets   : seed={eval_seed}, n_per_class={eval_n_per_class}, spec={eval_spec_version}")
    print(f"Device          : {device}")
    print(f"{'=' * 60}")

    ckpt_path = (
        project_root
        / "artifacts"
        / "checkpoints"
        / f"osr_saf_trinet_seed{ckpt_seed}_n{ckpt_n_per_class}.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"OSR checkpoint not found: {ckpt_path}\n"
            f"Run train_osr_runner first."
        )
    eval_dataset_root = Path(f"C:/Users/user/Documents/MATLAB/eval_datasets")
    eval_known_path = (
        eval_dataset_root
        / "impaired"
        / f"impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_eval.mat"
    )
    eval_unknown_path = (
        eval_dataset_root
        / "unknown"
        / f"unknown_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}.mat"
    )
    if not eval_known_path.exists():
        raise FileNotFoundError(f"Known eval dataset not found: {eval_known_path}")
    if not eval_unknown_path.exists():
        raise FileNotFoundError(f"Unknown eval dataset not found: {eval_unknown_path}")

    model = OsrSAF_TriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Loaded checkpoint : {ckpt_path.name}")

    known_artifact = load_artifact(str(eval_known_path), load_params=False)
    x_stft_k, x_iq_k, x_if_k, y_k = build_feature_tensor(known_artifact)
    known_dataset = torch.utils.data.TensorDataset(x_stft_k, x_iq_k, x_if_k, y_k)
    known_loader  = create_eval_loader(known_dataset, batch_size=batch_size, device=device)
    print(f"  Known samples     : {len(known_dataset)}")

    unknown_artifact = load_artifact(str(eval_unknown_path), load_params=False)
    x_stft_u, x_iq_u, x_if_u, _y_unk_orig = build_feature_tensor(unknown_artifact)
    y_unk_neg = torch.full((x_stft_u.size(0),), -1, dtype=torch.long)
    unknown_dataset = torch.utils.data.TensorDataset(x_stft_u, x_iq_u, x_if_u, y_unk_neg)
    unknown_loader  = create_eval_loader(unknown_dataset, batch_size=batch_size, device=device)
    print(f"  Unknown samples   : {len(unknown_dataset)}")

    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_preds:  list[np.ndarray] = []
    all_final:  list[np.ndarray] = []

    with torch.no_grad():
        for loader in (known_loader, unknown_loader):
            for x_stft, x_iq, x_if, y in loader:
                x_stft = x_stft.to(device)
                x_iq   = x_iq.to(device)
                x_if   = x_if.to(device)

                logits, score, _ = model.forward_with_osr(x_stft, x_iq, x_if)
                preds = logits.argmax(dim=1)

                final = preds.clone()
                thresh = model.class_thresholds[preds]
                final[score > thresh] = -1

                all_labels.append(y.cpu().numpy())
                all_scores.append(score.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_final.append(final.cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    scores_arr = np.concatenate(all_scores)
    preds_arr  = np.concatenate(all_preds)
    final_arr  = np.concatenate(all_final)

    known_mask   = labels_arr != -1
    unknown_mask = labels_arr == -1
    known_count   = int(np.sum(known_mask))
    unknown_count = int(np.sum(unknown_mask))

    binary_labels = np.zeros_like(labels_arr)
    binary_labels[unknown_mask] = 1

    try:
        auroc = float(roc_auc_score(binary_labels, scores_arr))
    except ValueError:
        auroc = 0.5

    rejected = (final_arr == -1)

    known_acc = (
        float(accuracy_score(labels_arr[known_mask], preds_arr[known_mask]))
        if known_count > 0 else 0.0
    )
    known_bal_acc = (
        float(balanced_accuracy_score(labels_arr[known_mask], preds_arr[known_mask]))
        if known_count > 0 else 0.0
    )
    known_f1_macro = (
        float(f1_score(labels_arr[known_mask], preds_arr[known_mask], average="macro", zero_division=0))
        if known_count > 0 else 0.0
    )

    unknown_recall   = float(np.mean(rejected[unknown_mask])) if unknown_count > 0 else 0.0
    false_alarm_rate = float(np.mean(rejected[known_mask]))   if known_count > 0 else 0.0

    total_rejected = int(rejected.sum())
    detection_precision = (
        float(np.sum(rejected & unknown_mask) / total_rejected) if total_rejected > 0 else 0.0
    )
    eps = 1e-8
    f1_unknown = (
        2 * unknown_recall * detection_precision
        / (unknown_recall + detection_precision + eps)
    )

    open_correct = (
        int(np.sum((final_arr == labels_arr) & known_mask))
        + int(np.sum((final_arr == -1) & unknown_mask))
    )
    open_set_acc = float(open_correct / max(1, len(labels_arr)))

    cm_size = NUM_CLASSES + 1
    labels_mapped = np.where(labels_arr == -1, NUM_CLASSES, labels_arr)
    final_mapped  = np.where(final_arr == -1,  NUM_CLASSES, final_arr)
    cm = confusion_matrix(labels_mapped, final_mapped, labels=list(range(cm_size))).tolist()

    per_class_acc = {}
    for c in range(cm_size):
        idx = (labels_mapped == c)
        if int(np.sum(idx)) == 0:
            per_class_acc[f"class_{c}" if c < NUM_CLASSES else "unknown"] = 0.0
        else:
            per_class_acc[f"class_{c}" if c < NUM_CLASSES else "unknown"] = float(
                np.mean(final_mapped[idx] == labels_mapped[idx])
            )

    print(f"\n  Known accuracy    : {known_acc:.4f}")
    print(f"  Known bal. acc.   : {known_bal_acc:.4f}")
    print(f"  AUROC             : {auroc:.4f}")
    print(f"  Unknown recall    : {unknown_recall:.4f}")
    print(f"  False alarm rate  : {false_alarm_rate:.4f}")
    print(f"  F1 (unknown)      : {f1_unknown:.4f}")
    print(f"  Open-set accuracy : {open_set_acc:.4f}")

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name":  "osr_saf_trinet",
        "checkpoint": {
            "seed":        ckpt_seed,
            "n_per_class": ckpt_n_per_class,
            "path":        str(ckpt_path),
        },
        "eval_dataset": {
            "seed":         eval_seed,
            "n_per_class":  eval_n_per_class,
            "spec_version": eval_spec_version,
            "known_path":   str(eval_known_path),
            "unknown_path": str(eval_unknown_path),
            "n_known":      known_count,
            "n_unknown":    unknown_count,
        },
        "device": str(device),
        "metrics": {
            "known_accuracy":          round(known_acc, 6),
            "known_balanced_accuracy": round(known_bal_acc, 6),
            "known_f1_macro":          round(known_f1_macro, 6),
            "auroc":                   round(auroc, 6),
            "unknown_recall":          round(unknown_recall, 6),
            "false_alarm_rate":        round(false_alarm_rate, 6),
            "f1_unknown":              round(f1_unknown, 6),
            "open_set_accuracy":       round(open_set_acc, 6),
            "per_class_accuracy":      per_class_acc,
            "confusion_matrix":        cm,
        },
    }

    log_dir = project_root / "artifacts" / "logs" / "osr_evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = (
        f"osr_saf_trinet"
        f"_ckpt{ckpt_seed}n{ckpt_n_per_class}"
        f"_eval{eval_seed}n{eval_n_per_class}"
        f"_{eval_spec_version}.json"
    )
    log_path = log_dir / log_name
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  Log saved         : {log_path}")
    print(f"{'=' * 60}\n")

    return result
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
from python.src.analysis import plot_osr_eval_feature_embedding


NUM_CLASSES = 10


def _resolve_unknown_eval_path(
    eval_dataset_root: Path,
    eval_spec_version: str,
    eval_seed: int,
    eval_n_per_class: int,
) -> Path:
    """
    Resolve the unknown evaluation dataset path with backward compatibility.

    Preferred (new convention): unknown_dataset_{ver}_seed{s}_n{n}_test.mat
    Fallback (legacy):          unknown_dataset_{ver}_seed{s}_n{n}.mat

    The legacy fallback exists so that already-generated eval seeds (e.g.
    seeds 118, 340, 410) keep working without regeneration. New eval seeds
    should use the _test.mat convention.
    """
    new_path = (
        eval_dataset_root
        / "unknown"
        / f"unknown_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_test.mat"
    )
    if new_path.exists():
        return new_path

    legacy_path = (
        eval_dataset_root
        / "unknown"
        / f"unknown_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}.mat"
    )
    if legacy_path.exists():
        print(f"  [warn] Falling back to legacy unknown filename: {legacy_path.name}")
        return legacy_path

    raise FileNotFoundError(
        f"Unknown eval dataset not found at either:\n"
        f"  {new_path}\n  {legacy_path}"
    )


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

    device = resolve_device(device_str)

    # Paths
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
    eval_unknown_path = _resolve_unknown_eval_path(
        eval_dataset_root, eval_spec_version, eval_seed, eval_n_per_class
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not eval_known_path.exists():
        raise FileNotFoundError(f"Known eval dataset not found: {eval_known_path}")

    # Model
    model = OsrSAF_TriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Loaded checkpoint : {ckpt_path.name}")

    # Datasets / loaders
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

    # Run the standard metric evaluation (re-use existing logic)
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

    # t-SNE embedding plot
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
        title_suffix=title_suffix,
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
    eval_unknown_path = _resolve_unknown_eval_path(
        eval_dataset_root, eval_spec_version, eval_seed, eval_n_per_class
    )
    if not eval_known_path.exists():
        raise FileNotFoundError(f"Known eval dataset not found: {eval_known_path}")

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
    known_f1 = (
        float(f1_score(labels_arr[known_mask], preds_arr[known_mask], average="macro"))
        if known_count > 0 else 0.0
    )

    unknown_recall = float(np.mean(rejected[unknown_mask])) if unknown_count > 0 else 0.0
    false_alarm    = float(np.mean(rejected[known_mask]))   if known_count   > 0 else 0.0

    # Treat correct unknown rejection as "class -1 predicted = label -1"
    final_labels = labels_arr.copy()
    pred_labels  = final_arr.copy()
    open_set_acc = float(np.mean(final_labels == pred_labels))

    if unknown_count > 0 and known_count > 0:
        # F1 for the unknown class as a binary problem (rejected vs. accepted)
        f1_unk = float(
            f1_score(
                (labels_arr == -1).astype(int),
                rejected.astype(int),
                average="binary",
            )
        )
    else:
        f1_unk = 0.0

    # Confusion matrix incl. unknown row/column
    cm_labels = list(range(NUM_CLASSES)) + [-1]
    final_for_cm = final_arr.copy()
    label_for_cm = labels_arr.copy()
    cm = confusion_matrix(label_for_cm, final_for_cm, labels=cm_labels).tolist()

    per_class_acc: dict[str, float] = {}
    for c in range(NUM_CLASSES):
        idx = labels_arr == c
        if idx.sum() == 0:
            per_class_acc[f"class_{c}"] = 0.0
        else:
            # accept only if predicted class c AND not rejected
            correct = (final_arr[idx] == c).sum()
            per_class_acc[f"class_{c}"] = float(correct / idx.sum())
    if unknown_count > 0:
        per_class_acc["unknown"] = unknown_recall

    print(f"\n{'=' * 52}")
    print(f"FINAL EVAL RESULTS")
    print(f"Known accuracy   : {known_acc:.4f}")
    print(f"Open-set accuracy: {open_set_acc:.4f}")
    print(f"AUROC            : {auroc:.4f}")
    print(f"Unknown recall   : {unknown_recall:.4f}")
    print(f"False alarm rate : {false_alarm:.4f}")
    print(f"{'=' * 52}\n")

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "osr_saf_trinet",
        "checkpoint": {
            "seed": ckpt_seed,
            "n_per_class": ckpt_n_per_class,
            "path": str(ckpt_path),
        },
        "eval_dataset": {
            "seed": eval_seed,
            "n_per_class": eval_n_per_class,
            "spec_version": eval_spec_version,
            "known_path": str(eval_known_path),
            "unknown_path": str(eval_unknown_path),
            "n_known": int(known_count),
            "n_unknown": int(unknown_count),
        },
        "device": str(device),
        "metrics": {
            "known_accuracy":         known_acc,
            "known_balanced_accuracy": known_bal_acc,
            "known_f1_macro":          known_f1,
            "auroc":                   auroc,
            "unknown_recall":          unknown_recall,
            "false_alarm_rate":        false_alarm,
            "f1_unknown":              f1_unk,
            "open_set_accuracy":       open_set_acc,
            "per_class_accuracy":      per_class_acc,
            "confusion_matrix":        cm,
        },
    }

    log_dir = project_root / "artifacts" / "logs" / "osr_evaluations"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"osr_saf_trinet_ckpt{ckpt_seed}n{ckpt_n_per_class}"
        f"_eval{eval_seed}n{eval_n_per_class}_{eval_spec_version}.json"
    )
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Eval log saved to : {log_path}")

    return result

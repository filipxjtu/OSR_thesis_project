from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from python.src.dataio import load_artifact
from python.src.models import OsrSAF_TriNet
from python.src.preprocessing import build_feature_tensor
from python.src.utils import create_eval_loader, resolve_device
# Import our repaired diagnostic functions
from python.src.analysis import (
    plot_osr_eval_feature_embedding,
    generate_osr_confusion_outputs
)

NUM_CLASSES = 10


def _resolve_unknown_eval_path(
        eval_dataset_root: Path,
        eval_spec_version: str,
        eval_seed: int,
        eval_n_per_class: int,
) -> Path:
    """Standard path resolution for unknown datasets with legacy support."""
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
    return legacy_path if legacy_path.exists() else new_path


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
        snr_label: str | None = None,
) -> dict:
    """
    Wraps standard evaluation and adds Confusion Matrix + t-SNE visualization.
    """
    # 1. Run core evaluation silently to get metrics and JSON
    result = evaluate_osr_model(
        ckpt_seed=ckpt_seed,
        ckpt_n_per_class=ckpt_n_per_class,
        eval_seed=eval_seed,
        eval_n_per_class=eval_n_per_class,
        eval_spec_version=eval_spec_version,
        project_root=project_root,
        batch_size=batch_size,
        device_str=device_str,
        verbose=False
    )

    print(f"--- Running OSR Diagnostics | SNR: {snr_label or 'Fixed'} ---")
    device = resolve_device(device_str)

    # Reload model and data for plotting
    model = OsrSAF_TriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    model.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device))
    model.eval()

    eval_dataset_root = Path(f"C:/Users/user/Documents/MATLAB/eval_datasets")
    eval_known_path = eval_dataset_root / "impaired" / f"impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_eval.mat"
    eval_unknown_path = _resolve_unknown_eval_path(eval_dataset_root, eval_spec_version, eval_seed, eval_n_per_class)

    # Load Data
    k_art = load_artifact(str(eval_known_path), load_params=False)
    xk_stft, xk_iq, xk_if, yk = build_feature_tensor(k_art)
    k_loader = create_eval_loader(torch.utils.data.TensorDataset(xk_stft, xk_iq, xk_if, yk), batch_size=batch_size,
                                  device=device)

    u_art = load_artifact(str(eval_unknown_path), load_params=False)
    xu_stft, xu_iq, xu_if, _ = build_feature_tensor(u_art)
    u_loader = create_eval_loader(
        torch.utils.data.TensorDataset(xu_stft, xu_iq, xu_if, torch.full((xu_stft.size(0),), -1)),
        batch_size=batch_size, device=device)

    # Create sub-directory for this SNR level
    snr_path = fig_dir / f"snr_{snr_label.replace(' ', '')}" if snr_label else fig_dir
    snr_path.mkdir(parents=True, exist_ok=True)

    # --- Generate Confusion Matrix ---
    print(f"  Generating Confusion Matrix...")
    generate_osr_confusion_outputs(
        model=model, loader_known=k_loader, loader_osr=u_loader,
        device=device, out_dir=snr_path, n_classes=NUM_CLASSES
    )

    # --- Generate t-SNE ---
    print(f"  Generating t-SNE Embedding...")
    plot_osr_eval_feature_embedding(
        model=model, loader_known=k_loader, loader_osr=u_loader,
        device=device, out_dir=snr_path, n_classes=NUM_CLASSES,
        title_suffix=f" — SNR {snr_label}" if snr_label else ""
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
        verbose: bool = True,
) -> dict:
    """Core metrics calculation and JSON logging."""
    device = resolve_device(device_str)

    if verbose:
        print(f"\n{'=' * 60}\nEvaluating OSR Model: seed={eval_seed}, n={eval_n_per_class}\n{'=' * 60}")

    ckpt_path = project_root / "artifacts" / "checkpoints" / f"osr_saf_trinet_seed{ckpt_seed}_n{ckpt_n_per_class}.pt"
    eval_dataset_root = Path(f"C:/Users/user/Documents/MATLAB/eval_datasets")
    eval_known_path = eval_dataset_root / "impaired" / f"impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_eval.mat"
    eval_unknown_path = _resolve_unknown_eval_path(eval_dataset_root, eval_spec_version, eval_seed, eval_n_per_class)

    model = OsrSAF_TriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Data Loaders
    k_art = load_artifact(str(eval_known_path), load_params=False)
    xk_stft, xk_iq, xk_if, yk = build_feature_tensor(k_art)
    known_loader = create_eval_loader(torch.utils.data.TensorDataset(xk_stft, xk_iq, xk_if, yk), batch_size=batch_size,
                                      device=device)

    u_art = load_artifact(str(eval_unknown_path), load_params=False)
    xu_stft, xu_iq, xu_if, _ = build_feature_tensor(u_art)
    unknown_loader = create_eval_loader(
        torch.utils.data.TensorDataset(xu_stft, xu_iq, xu_if, torch.full((xu_stft.size(0),), -1)),
        batch_size=batch_size, device=device)

    # Inference logic (identical to previous version)
    all_labels, all_scores, all_preds, all_final = [], [], [], []
    with torch.no_grad():
        for loader in (known_loader, unknown_loader):
            for x_stft, x_iq, x_if, y in loader:
                logits, score = model.forward_with_osr(x_stft.to(device), x_iq.to(device), x_if.to(device))
                preds = logits.argmax(dim=1)
                final = preds.clone()
                final[score > model.class_thresholds[preds]] = -1
                all_labels.append(y.numpy())
                all_scores.append(score.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_final.append(final.cpu().numpy())

    labels_arr, scores_arr = np.concatenate(all_labels), np.concatenate(all_scores)
    preds_arr, final_arr = np.concatenate(all_preds), np.concatenate(all_final)

    # Metrics
    known_mask = labels_arr != -1
    unk_mask = labels_arr == -1
    known_acc = accuracy_score(labels_arr[known_mask], preds_arr[known_mask]) if known_mask.any() else 0.0
    open_set_acc = np.mean(labels_arr == final_arr)
    binary_labels = (labels_arr == -1).astype(int)
    try:
        auroc = roc_auc_score(binary_labels, scores_arr)
    except:
        auroc = 0.5
    unk_recall = np.mean(final_arr[unk_mask] == -1) if unk_mask.any() else 0.0
    far = np.mean(final_arr[known_mask] == -1) if known_mask.any() else 0.0

    #print(f"\n{'=' * 40}\nResults: Acc {known_acc:.4f} | OS-Acc {open_set_acc:.4f}\n{'=' * 40}")

    # This ensures you see the result for the current SNR point right now.
    print(f"\n>>> SNR Point Evaluation Complete (Seed {eval_seed})")
    print(f"    Known Accuracy   : {100 * known_acc:.2f}%")
    print(f"    Open-set Accuracy: {100 * open_set_acc:.2f}%")
    print(f"    AUROC Score      : {auroc:.4f}")
    print(f"    Unknown Recall   : {100 * unk_recall:.2f}%")
    print(f"    False Alarm Rate : {100 * far:.2f}%")
    print("-" * 30)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"seed": ckpt_seed, "n_per_class": ckpt_n_per_class, "path": str(ckpt_path)},
        "eval_dataset": {"seed": eval_seed, "n_per_class": eval_n_per_class},
        "metrics": {
            "known_accuracy": float(known_acc),
            "auroc": float(auroc),
            "unknown_recall": float(unk_recall),
            "false_alarm_rate": float(far),
            "open_set_accuracy": float(open_set_acc),
        }
    }

    # Save JSON log
    log_dir = project_root / "artifacts" / "logs" / "osr_evaluations"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"osr_eval_seed{eval_seed}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
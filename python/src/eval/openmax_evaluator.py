from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from python.src.dataio import load_artifact
from python.src.models.openmax_trinet import OpenMaxTriNet
from python.src.preprocessing import build_feature_tensor
from python.src.utils import create_eval_loader, resolve_device

"""

Evaluates an OpenMaxTriNet checkpoint on a single SNR-specific eval dataset
(known eval split + matching unknown test split). Returns the same metrics
shape as evaluate_osr_model so the comparison script can treat both methods
identically.

"""

NUM_CLASSES = 10


def _resolve_unknown_eval_path(
    eval_dataset_root: Path,
    eval_spec_version: str,
    eval_seed: int,
    eval_n_per_class: int,
) -> Path:
    """Try the standard `_test.mat` name first, fall back to legacy unsuffixed name."""
    new_path = (
        eval_dataset_root / "unknown"
        / f"unknown_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_test.mat"
    )
    if new_path.exists():
        return new_path
    legacy = (
        eval_dataset_root / "unknown"
        / f"unknown_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}.mat"
    )
    return legacy if legacy.exists() else new_path


def evaluate_openmax_model(
    *,
    ckpt_seed: int,
    ckpt_n_per_class: int,
    eval_seed: int,
    eval_n_per_class: int,
    eval_spec_version: str,
    project_root: Path,
    eval_dataset_root: Path | None = None,
    batch_size: int = 64,
    device_str: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    Returns a dict with the same shape as evaluate_osr_model so OpenMax and
    OSR-SAF can be compared on identical metrics.

    eval_dataset_root: where the SNR-specific eval files live. Defaults to
                      project_root/artifacts/datasets which is the convention
                      used by ablation_eval.py.
    """
    device = resolve_device(device_str)

    if eval_dataset_root is None:
        eval_dataset_root = project_root / "artifacts" / "datasets"

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Evaluating OpenMax | eval_seed={eval_seed} | n={eval_n_per_class}")
        print(f"{'=' * 60}")

    ckpt_path = (
        project_root / "artifacts" / "checkpoints"
        / f"openmax_trinet_seed{ckpt_seed}_n{ckpt_n_per_class}.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"OpenMax checkpoint not found: {ckpt_path}")

    eval_known_path = (
        eval_dataset_root / "impaired"
        / f"impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_eval.mat"
    )
    eval_unknown_path = _resolve_unknown_eval_path(
        eval_dataset_root, eval_spec_version, eval_seed, eval_n_per_class,
    )

    if not eval_known_path.exists():
        raise FileNotFoundError(f"Known eval dataset not found: {eval_known_path}")
    if not eval_unknown_path.exists():
        raise FileNotFoundError(f"Unknown eval dataset not found: {eval_unknown_path}")

    # ---- Build model and load checkpoint ----
    # OpenMaxTriNet's buffers (mavs, weibull_*, threshold) come back via state_dict.
    model = OpenMaxTriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ---- Loaders ----
    k_art = load_artifact(str(eval_known_path), load_params=False)
    xk_stft, xk_iq, xk_if, yk = build_feature_tensor(k_art)
    known_loader = create_eval_loader(
        torch.utils.data.TensorDataset(xk_stft, xk_iq, xk_if, yk),
        batch_size=batch_size, device=device,
    )

    u_art = load_artifact(str(eval_unknown_path), load_params=False)
    xu_stft, xu_iq, xu_if, _ = build_feature_tensor(u_art)
    unknown_loader = create_eval_loader(
        torch.utils.data.TensorDataset(
            xu_stft, xu_iq, xu_if, torch.full((xu_stft.size(0),), -1),
        ),
        batch_size=batch_size, device=device,
    )

    # ---- Inference ----
    all_labels, all_scores, all_preds, all_final = [], [], [], []
    with torch.no_grad():
        for loader in (known_loader, unknown_loader):
            for x_stft, x_iq, x_if, y in loader:
                logits, score = model.forward_with_osr(
                    x_stft.to(device), x_iq.to(device), x_if.to(device),
                )
                preds = logits.argmax(dim=1)
                final = preds.clone()
                final[score > model.threshold] = -1

                all_labels.append(y.numpy())
                all_scores.append(score.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_final.append(final.cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    scores_arr = np.concatenate(all_scores)
    preds_arr  = np.concatenate(all_preds)
    final_arr  = np.concatenate(all_final)

    known_mask   = labels_arr != -1
    unknown_mask = labels_arr == -1

    known_acc        = accuracy_score(labels_arr[known_mask], preds_arr[known_mask]) if known_mask.any() else 0.0
    open_set_acc     = float(np.mean(labels_arr == final_arr))
    binary_labels    = (labels_arr == -1).astype(int)
    try:
        auroc = float(roc_auc_score(binary_labels, scores_arr))
    except ValueError:
        auroc = 0.5
    unknown_recall   = float(np.mean(final_arr[unknown_mask] == -1)) if unknown_mask.any() else 0.0
    false_alarm_rate = float(np.mean(final_arr[known_mask]   == -1)) if known_mask.any() else 0.0

    if verbose:
        print(f">>> SNR Point Eval (seed {eval_seed})")
        print(f"    Known Accuracy   : {100 * known_acc:.2f}%")
        print(f"    Open-set Accuracy: {100 * open_set_acc:.2f}%")
        print(f"    AUROC            : {auroc:.4f}")
        print(f"    Unknown Recall   : {100 * unknown_recall:.2f}%")
        print(f"    False Alarm Rate : {100 * false_alarm_rate:.2f}%")
        print("-" * 30)

    result = {
        "method":      "openmax",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint":  {"seed": ckpt_seed, "n_per_class": ckpt_n_per_class, "path": str(ckpt_path)},
        "eval_dataset": {"seed": eval_seed, "n_per_class": eval_n_per_class},
        "metrics": {
            "known_accuracy":   float(known_acc),
            "auroc":            auroc,
            "unknown_recall":   unknown_recall,
            "false_alarm_rate": false_alarm_rate,
            "open_set_accuracy": open_set_acc,
        },
    }

    log_dir = project_root / "artifacts" / "logs" / "openmax_evaluations"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"openmax_eval_seed{eval_seed}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
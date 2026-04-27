from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from python.src.dataio import load_artifact
from python.src.models import AsymmetricTriNet, SimpleCNN
from python.src.preprocessing import build_feature_tensor
from python.src.utils import create_eval_loader, resolve_device, FeatureTensorDataset


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "asymmetric_trinet": AsymmetricTriNet,
    "simple_cnn": SimpleCNN,
}

NUM_CLASSES = 10


def evaluate_closed_set_model(
    model_name: str,
    # --- checkpoint identity (what was trained) ---
    ckpt_seed: int,
    ckpt_n_per_class: int,
    # --- eval dataset identity (what we test on) ---
    eval_seed: int,
    eval_n_per_class: int,
    eval_spec_version: str,
    project_root: Path,
    batch_size: int = 16,
    device_str: str = "auto",
) -> dict:
    """
    Load a saved checkpoint and evaluate it on a given eval dataset.

    Checkpoint and eval dataset are fully decoupled so you can test a
    model trained on one configuration (seed, n, SNR range) against
    eval datasets from a completely different configuration (e.g. fixed
    SNR, different seed, different n_per_class).

    Checkpoint path  : artifacts/checkpoints/{model_name}_seed{ckpt_seed}_n{ckpt_n_per_class}.pt
    Eval dataset path: artifacts/datasets/impaired/impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_eval.mat
    Log output       : artifacts/logs/evaluation/{model_name}_ckpt{ckpt_seed}n{ckpt_n_per_class}_eval{eval_seed}n{eval_n_per_class}.json
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    device = resolve_device(device_str)

    print(f"\n{'=' * 60}")
    print(f"Evaluating      : {model_name}")
    print(f"Checkpoint      : seed={ckpt_seed}, n_per_class={ckpt_n_per_class}")
    print(f"Eval dataset    : seed={eval_seed}, n_per_class={eval_n_per_class}, spec={eval_spec_version}")
    print(f"Device          : {device}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------ #
    # 1. Checkpoint
    # ------------------------------------------------------------------ #
    ckpt_path = (
        project_root
        / "artifacts"
        / "checkpoints"
        / f"{model_name}_seed{ckpt_seed}_n{ckpt_n_per_class}.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first using train_model_runner.py"
        )

    # ------------------------------------------------------------------ #
    # 2. Model
    # ------------------------------------------------------------------ #
    model = MODEL_REGISTRY[model_name](num_classes=NUM_CLASSES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Loaded checkpoint : {ckpt_path.name}")

    # ------------------------------------------------------------------ #
    # 3. Eval dataset
    # ------------------------------------------------------------------ #
    eval_path = (
        project_root
        / "artifacts"
        / "datasets"
        / "impaired"
        / f"impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n_per_class}_eval.mat"
    )

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found: {eval_path}"
        )

    artifact = load_artifact(str(eval_path), load_params=False)
    x_stft, x_iq, x_if, y = build_feature_tensor(artifact)
    dataset = FeatureTensorDataset(x_stft, x_iq, x_if, y)
    loader  = create_eval_loader(dataset, batch_size=batch_size, device=device)
    print(f"  Eval samples      : {len(dataset)}")

    # ------------------------------------------------------------------ #
    # 4. Inference
    # ------------------------------------------------------------------ #
    y_true, y_pred = [], []

    with torch.no_grad():
        for x_stft_b, x_iq_b, x_if_b, y_b in loader:
            x_stft_b = x_stft_b.to(device)
            x_iq_b   = x_iq_b.to(device)
            x_if_b   = x_if_b.to(device)

            logits = model(x_stft_b, x_iq_b, x_if_b)
            preds  = logits.argmax(dim=1).cpu().tolist()

            y_true.extend(y_b.tolist())
            y_pred.extend(preds)

    # ------------------------------------------------------------------ #
    # 5. Metrics
    # ------------------------------------------------------------------ #
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wtd   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()
    report   = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
        output_dict=True,
    )

    cm_arr = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    per_class_acc = {
        f"class_{c}": float(cm_arr[c, c] / cm_arr[c].sum()) if cm_arr[c].sum() > 0 else 0.0
        for c in range(NUM_CLASSES)
    }

    print(f"\n  Accuracy          : {100 * acc:.2f}%")
    print(f"  Balanced Accuracy : {100 * bal_acc:.2f}%")
    print(f"  F1 (macro)        : {f1_macro:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Write log
    # ------------------------------------------------------------------ #
    result = {
        "created_utc":        datetime.now(timezone.utc).isoformat(),
        "model_name":         model_name,
        "checkpoint": {
            "seed":           ckpt_seed,
            "n_per_class":    ckpt_n_per_class,
            "path":           str(ckpt_path),
        },
        "eval_dataset": {
            "seed":           eval_seed,
            "n_per_class":    eval_n_per_class,
            "spec_version":   eval_spec_version,
            "path":           str(eval_path),
            "n_samples":      len(dataset),
        },
        "device":  str(device),
        "metrics": {
            "accuracy":              round(float(acc),      6),
            "balanced_accuracy":     round(float(bal_acc),  6),
            "f1_macro":              round(float(f1_macro), 6),
            "f1_weighted":           round(float(f1_wtd),   6),
            "per_class_accuracy":    per_class_acc,
            "confusion_matrix":      cm,
            "classification_report": report,
        },
    }

    log_dir  = project_root / "artifacts" / "logs" / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_name = (
        f"{model_name}"
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
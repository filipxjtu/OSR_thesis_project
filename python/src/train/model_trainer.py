from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ..analysis import generate_confusion_outputs, plot_cnn_feature_embedding, plot_threshold
from ..dataio import load_artifact
from ..models import BaselineCNN, ResidualCNN, ImprovedDRSN, PhysicsAwareDRSN, TS_MS_VA_DRSN
from ..preprocessing import build_feature_tensor, split_dataset
from ..train import train_one_epoch, evaluate, HParams
from ..utils import create_train_loader, create_eval_loader, resolve_device
from ..validation import run_validation_gate, ValidationError



MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    "first_residual_cnn": ResidualCNN,
    "improved_drsn": ImprovedDRSN,
    "physics_aware_drsn": PhysicsAwareDRSN,
    "ts_ms_va_drsn": TS_MS_VA_DRSN,
}

def collect_thresholds(model):

    thresholds = []

    for module in model.modules():

        if hasattr(module, "last_threshold") and module.last_threshold is not None:

            t = module.last_threshold
            t = t.squeeze(-1).squeeze(-1)

            thresholds.append(t.mean(dim=0).cpu())

    return thresholds

def train_model(seed: int,
                project_root: Path,
                model_name: str,
                n_per_class: int,
                spec_version: str,
                n_epochs: int,
                ):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not supported.")

    dataset_dir = project_root / "artifacts" / "datasets"

    clean_file = dataset_dir / "clean" / f"clean_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"
    train_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
    eval_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_eval.mat"


    report_name = f"validation_seed{seed}_n{n_per_class}_{spec_version}.json"
    report_dir = project_root / "reports" / "validations"
    report_path = report_dir / report_name

    if report_path.exists():
        print(f"\nExisting validation report found at: {report_path}")
        print("DATASET VALIDATION ALREADY PASSED. Starting training...\n")

    else:
        try:
            run_validation_gate(
                clean_file=str(clean_file),
                train_file=str(train_file),
                eval_file=str(eval_file),
                spec_version=spec_version,
                n_classes=7,
                report_name=report_name,
                enable_feature_checks=True,
                partial_features_check=False,
                enable_repro_check=True,
                repro_trial=2,
            )

            print("\nDATASET VALIDATION PASSED. Starting training...\n")

        except ValidationError as e:
            print("\nDATASET VALIDATION FAILED")
            raise e

    hparams = HParams(
        lr=1e-3,
        weight_decay=1e-4,
        epochs=n_epochs,
        batch_size=32,
        device="auto",
        seed=seed,
    )

    if hparams.seed:
        torch.manual_seed(hparams.seed)

    device = resolve_device(hparams.device)
    print(f"Using device: {device}\n")

    train_artifact = load_artifact(str(train_file))
    eval_artifact = load_artifact(str(eval_file))

    x, y = build_feature_tensor(train_artifact)
    x_eval, y_eval = build_feature_tensor(eval_artifact)

    train_set, val_set = split_dataset(x, y, train_ratio=0.8, seed=hparams.seed)
    test_set = TensorDataset(x_eval, y_eval)

    train_loader = create_train_loader(train_set, hparams.batch_size, device)
    val_loader = create_eval_loader(val_set, hparams.batch_size, device)

    test_loader = create_eval_loader(test_set, hparams.batch_size, device)

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

    #experiment 2 ... adding scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    metrics_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epochs": [],
        "test_result": {}
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, hparams.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        thresholds = collect_thresholds(model)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        if epoch == 1 or epoch % 5 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:02d} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Acc: {100 * val_acc:.2f}"
        )

        metrics_log["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        scheduler.step()

    model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    metrics_log["tests"] = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }

    figures_dir = project_root / "reports" / "figures" / f"{model_name}_seed{seed}_n{n_per_class}"

    generate_confusion_outputs(model, val_loader, device, figures_dir, n_classes=7)

    plot_cnn_feature_embedding(model, val_loader, device, figures_dir, n_classes=7)

    th_dir = figures_dir / "thresholds"

    plot_threshold(thresholds, th_dir)

    ckpt_dir = project_root / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{model_name}_seed{seed}_n{n_per_class}.pt"

    torch.save(best_state, ckpt_path)

    log_dir = project_root / "artifacts" / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{model_name}_training_seed{seed}_n{n_per_class}.json"

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\nModel saved to: {ckpt_path}")
    print(f"Training log saved to: {log_path}")
    print(f"figures saved to: {figures_dir}")
    print(f"saved figures: Confusion matrix, Per-class accuracy, Feature embedding and Threshold figures")
    print(f"\nBest validation accuracy: {100 * best_val_acc:.2f}")
    print(f"\nFinal TEST accuracy (eval dataset): {100 * test_acc:.2f} with test loss: {test_loss:.2f}")
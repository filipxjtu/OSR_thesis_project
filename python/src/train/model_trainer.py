from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from python.src.validation.gate import run_validation_gate
from python.src.validation.exceptions import ValidationError
from python.src.dataio.loader import load_artifact
from python.src.preprocessing.dataset_builder import build_feature_tensor
from python.src.preprocessing.splitting import split_dataset
from python.src.train.engine import train_one_epoch, evaluate
from python.src.train.hparams import HParams
from python.src.utils.dataloaders import create_train_loader, create_eval_loader
from python.src.utils.device import resolve_device
from python.src.analysis.model_diagnostics import generate_confusion_outputs

from python.src.models.baseline_cnn import BaselineCNN
#from python.src.models.baseline_cnn import [future models 1]
#from python.src.models.baseline_cnn import [future models 2]


MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    #"future model 1": FutureModel1,
    #"future model 2": FutureModel2,
}


def train_model(seed: int, project_root: Path, model_name: str = "baseline_cnn"):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not supported.")

    dataset_dir = project_root / "artifacts" / "datasets"

    clean_file = dataset_dir / "clean" / f"clean_dataset_v1_seed{seed}.mat"
    train_file = dataset_dir / "impaired" / f"impaired_dataset_v1_seed{seed}_train.mat"
    eval_file = dataset_dir / "impaired" / f"impaired_dataset_v1_seed{seed}_eval.mat"

    report_name = f"validation_seed{seed}_{model_name}.json"

    try:
        run_validation_gate(
            clean_file=str(clean_file),
            train_file=str(train_file),
            eval_file=str(eval_file),
            spec_version="v1",
            n_classes=7,
            report_name=report_name,
        )
        print("\nValidation passed. Starting training...\n")

    except ValidationError as e:
        print("\nDATASET VALIDATION FAILED")
        raise e

    hparams = HParams(
        lr=1e-3,
        weight_decay=0.0,
        epochs=50,
        batch_size=32,
        device="auto",
        seed=seed,
    )

    if hparams.seed:
        torch.manual_seed(hparams.seed)

    device = resolve_device(hparams.device)
    print(f"Using device: {device}")

    train_artifact = load_artifact(str(train_file))
    X, y = build_feature_tensor(train_artifact)

    train_set, val_set = split_dataset(
        X,
        y,
        train_ratio=0.8,
        seed=hparams.seed,
    )

    train_loader = create_train_loader(train_set, hparams.batch_size, device)
    val_loader = create_eval_loader(val_set, hparams.batch_size, device)

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)

    metrics_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epochs": [],
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        print(
            f"Epoch {epoch:02d} | "
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
            }
        )
    model.load_state_dict(best_state)

    figures_dir = project_root / "reports" / "figures" / f"model_seed{seed}"

    generate_confusion_outputs(
        model,
        val_loader,
        device,
        figures_dir,
        n_classes=7,
    )

    ckpt_dir = project_root / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{model_name}_seed{seed}.pt"
    torch.save(best_state, ckpt_path)

    log_dir = project_root / "artifacts" / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{model_name}_training_seed{seed}.json"

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\nModel saved to: {ckpt_path}")
    print(f"Training log saved to: {log_path}")
    print(f"Confusion matrix and per-class accuracy saved to: {figures_dir}")
    print(f"\nBest validation accuracy: {100 * best_val_acc:.2f}")
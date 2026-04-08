from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from ..utils import create_train_loader, create_eval_loader, resolve_device, load_osr_datasets, combined_loss
from ..train import evaluate
from ..models import IterativeOSR_TS_MS_VA_DRSN, Lightweight_OSR_DRSN, SparseFingerprint_TS_DRSN


MODEL_REGISTRY = {
    "iterative": IterativeOSR_TS_MS_VA_DRSN,
    "lightweight": Lightweight_OSR_DRSN,
    "sparse_fingerprint": SparseFingerprint_TS_DRSN,
}


def predict_with_rejection(model, x_stft, x_iq, threshold=0.5):
    logits, score, _ = model.forward_with_osr(x_stft, x_iq)
    preds = logits.argmax(dim=1)
    conf = torch.softmax(logits, dim=1).max(dim=1).values
    preds[score > threshold] = -1
    return preds, conf


def train_osr_model(
    *,
    model_name: str,
    seed: int,
    n_per_class: int,
    spec_version: str,
    project_root: Path,
    epochs: int,
):

    # check if dataset is validated
    report_dir = project_root / "reports" / "validations"

    if not (report_dir / f"validation_seed{seed}_n{n_per_class}_{spec_version}.json").exists():
        raise RuntimeError("Run validation first.")


    # handle device
    torch.manual_seed(seed)
    device = resolve_device("auto")

    # data
    datasets = load_osr_datasets(project_root, seed, n_per_class, spec_version)

    train_loader = create_train_loader(datasets["train"], 32, device)
    val_loader_known = create_eval_loader(datasets["val_known"], 32, device)
    val_loader_osr = create_eval_loader(datasets["val_osr"], 32, device)

    # model
    model_cls = MODEL_REGISTRY[model_name]

    model = model_cls(num_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0

        for x_stft, x_iq, y in train_loader:
            x_stft, x_iq, y = x_stft.to(device), x_iq.to(device), y.to(device)

            optimizer.zero_grad()

            logits, unknown_score, _ = model.forward_with_osr(x_stft, x_iq)

            loss = combined_loss(logits, unknown_score, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader_known, nn.CrossEntropyLoss(), device)

        print(f"Epoch {epoch} | Loss {total_loss:.4f} | Val Acc {val_acc:.3f}")

    return model
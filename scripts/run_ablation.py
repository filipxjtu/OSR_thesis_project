"""
scripts/run_ablation_training.py
=================================
Trains all 7 ablation variants of AsymmetricTriNet:

    Single-branch  (3):  STFT-only, IQ-only, IF-only
    Dual-branch    (3):  STFT+IQ, STFT+IF, IQ+IF
    Full model     (1):  STFT+IQ+IF  (skip if already trained)

Each variant is trained with the SAME seed, dataset, and hyperparameters
as the full model (seed=55, n_per_class=2500, 50 epochs) to ensure a
fair comparison. Checkpoints are saved as:

    asymmetric_trinet_ablation_<tag>_seed55_n2500.pt

where <tag> is one of: stft, iq, if, stft_iq, stft_if, iq_if, full

Run from project root:
    python scripts/run_ablation_training.py

IMPORTANT — add AblationTriNet to your model registry before running:
    In python/src/models/__init__.py add:
        from .ablation_trinet import AblationTriNet
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# ── Path setup (mirrors your existing runner scripts) ─────────────────────────
import sys

def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports (after path is set) ───────────────────────────────────────────────
from python.src.models.ablation_trinet import AblationTriNet
from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor, split_dataset
from python.src.train.engine import train_one_epoch, evaluate
from python.src.train.hparams import HParams
from python.src.utils import (
    create_train_loader, create_eval_loader,
    resolve_device, FeatureTensorDataset, SupConLoss, prepare_unique_file,
)
from python.src.analysis import generate_confusion_outputs


# ── Ablation variant definitions ──────────────────────────────────────────────
# Each entry: (tag, disabled_branches_set, human_label)
ABLATION_VARIANTS = [
    ("stft",    {1, 2}, "STFT only"),
    ("iq",      {0, 2}, "IQ only"),
    ("if",      {0, 1}, "IF only"),
    ("stft_iq", {2},    "STFT + IQ"),
    ("stft_if", {1},    "STFT + IF"),
    ("iq_if",   {0},    "IQ + IF"),
    ("full",    set(),  "Full model (STFT + IQ + IF)"),
]

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 55
N_PER_CLASS = 2500
SPEC_VER    = "v2"
N_EPOCHS    = 50
NUM_CLASSES = 10

HP = HParams()  # uses your existing defaults: lr=1e-3, wd=1e-4, bs=32, cosine LR


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(project_root: Path):
    """Load and split the training dataset — identical to model_trainer.py logic."""
    train_path = (
        project_root / "artifacts" / "datasets" / "impaired"
        / f"impaired_dataset_{SPEC_VER}_seed{SEED}_n{N_PER_CLASS}_train.mat"
    )
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    artifact = load_artifact(str(train_path), load_params=False)
    x_stft, x_iq, x_if, y = build_feature_tensor(artifact)
    train_set, val_set = split_dataset(x_stft, x_iq, x_if, y, train_ratio=0.8, seed=SEED)

    x_stft_tr = train_set.dataset.tensors[0][train_set.indices]
    x_iq_tr = train_set.dataset.tensors[1][train_set.indices]
    x_if_tr = train_set.dataset.tensors[2][train_set.indices]
    y_tr = train_set.dataset.tensors[3][train_set.indices]

    # Extract tensors from the Subsets for the validation split
    x_stft_val = val_set.dataset.tensors[0][val_set.indices]
    x_iq_val = val_set.dataset.tensors[1][val_set.indices]
    x_if_val = val_set.dataset.tensors[2][val_set.indices]
    y_val = val_set.dataset.tensors[3][val_set.indices]

    # Held-out test split (eval dataset)
    eval_path = (
        project_root / "artifacts" / "datasets" / "impaired"
        / f"impaired_dataset_{SPEC_VER}_seed{SEED}_n{N_PER_CLASS}_eval.mat"
    )
    artifact_eval = load_artifact(str(eval_path), load_params=False)
    x_stft_te, x_iq_te, x_if_te, y_te = build_feature_tensor(artifact_eval)

    return (
        (x_stft_tr, x_iq_tr, x_if_tr, y_tr),
        (x_stft_val, x_iq_val, x_if_val, y_val),
        (x_stft_te, x_iq_te, x_if_te, y_te),
    )


def train_variant(
    tag: str,
    disabled_branches: set,
    label: str,
    train_data: tuple,
    val_data: tuple,
    test_data: tuple,
    device: torch.device,
    project_root: Path,
) -> dict:
    """Train one ablation variant. Returns metrics dict."""

    ckpt_name = f"asymmetric_trinet_ablation_{tag}_seed{SEED}_n{N_PER_CLASS}.pt"
    ckpt_dir  = project_root / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name

    # Skip if already trained
    if ckpt_path.exists():
        print(f"\n[SKIP] {label} — checkpoint exists: {ckpt_path.name}")
        return {}

    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"  Disabled branches: {disabled_branches or 'none (full model)'}")
    print(f"{'='*70}")

    set_seed(SEED)

    # Model
    model = AblationTriNet(
        num_classes=NUM_CLASSES,
        disabled_branches=disabled_branches,
    ).to(device)

    # Data loaders
    x_stft_tr, x_iq_tr, x_if_tr, y_tr   = train_data
    x_stft_val, x_iq_val, x_if_val, y_val = val_data
    x_stft_te, x_iq_te, x_if_te, y_te   = test_data

    train_ds = FeatureTensorDataset(x_stft_tr, x_iq_tr, x_if_tr, y_tr)
    val_ds   = FeatureTensorDataset(x_stft_val, x_iq_val, x_if_val, y_val)
    test_ds  = FeatureTensorDataset(x_stft_te, x_iq_te, x_if_te, y_te)

    train_loader = create_train_loader(train_ds, batch_size=HP.batch_size)
    val_loader   = create_eval_loader(val_ds,   batch_size=HP.batch_size)
    test_loader  = create_eval_loader(test_ds,  batch_size=HP.batch_size)

    # Optimiser + scheduler (identical to model_trainer.py)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )

    # Losses
    criterion_ce     = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_supcon = SupConLoss(temperature=0.1)

    best_val_acc = 0.0
    best_state   = None
    log_epochs   = []

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, ce_loss, sc_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion_ce, criterion_supcon, device,
            lambda_supcon=0.1,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion_ce, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
            print(
                f"  Epoch {epoch:02d}/{N_EPOCHS} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Train: {tr_loss:.4f} (CE {ce_loss:.4f} / SupCon {sc_loss:.4f}) | "
                f"Val: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%"
            )

        log_epochs.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_ce_loss": ce_loss,
            "train_supcon_loss": sc_loss,
            "val_loss": val_loss, "val_accuracy": val_acc,
            "learning_rate": scheduler.get_last_lr()[0],
        })

    # Restore best, evaluate on test set
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion_ce, device)

    print(f"\n  Best Val Acc : {100*best_val_acc:.2f}%")
    print(f"  Test Acc     : {100*test_acc:.2f}%  (loss: {test_loss:.4f})")

    # Save checkpoint
    torch.save(best_state, ckpt_path)
    print(f"  Checkpoint   : {ckpt_path}")

    # Save training log
    log = {
        "created_utc":       datetime.now(timezone.utc).isoformat(),
        "ablation_tag":      tag,
        "ablation_label":    label,
        "disabled_branches": sorted(disabled_branches),
        "seed":              SEED,
        "n_per_class":       N_PER_CLASS,
        "epochs":            log_epochs,
        "best_val_acc":      best_val_acc,
        "test_acc":          test_acc,
        "test_loss":         test_loss,
    }

    log_dir  = project_root / "artifacts" / "logs" / "ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ablation_{tag}_seed{SEED}_n{N_PER_CLASS}.json"
    with log_path.open("w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log          : {log_path}")

    return log


def main():
    device = resolve_device("auto")
    print(f"Device: {device}")

    print("\nLoading dataset...")
    train_data, val_data, test_data = load_dataset(PROJECT_ROOT)
    print(f"  Train: {train_data[0].shape[0]} samples")
    print(f"  Val  : {val_data[0].shape[0]} samples")
    print(f"  Test : {test_data[0].shape[0]} samples")

    for tag, disabled, label in ABLATION_VARIANTS:
        train_variant(
            tag=tag,
            disabled_branches=disabled,
            label=label,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
            project_root=PROJECT_ROOT,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n\nAll ablation variants trained.")


if __name__ == "__main__":
    main()
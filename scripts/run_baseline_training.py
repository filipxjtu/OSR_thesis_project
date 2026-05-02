"""
scripts/run_baseline_training.py
=================================
Trains VGG-16, ResNet-18, and DenseNet-121 baselines using two-phase
fine-tuning against AsymmetricTriNet's closed-set results.

Phase 1  (epochs 1  – PHASE1_EPOCHS):  backbone frozen, head-only training.
Phase 2  (epochs P1 – N_EPOCHS):       last block + head unfrozen, lower LR.

Key design choices for fair comparison:
  - Same training dataset, same seed, same 50-epoch total budget.
  - Same Adam optimizer, same cosine LR schedule (separate per phase).
  - Label smoothing 0.1 (same as AsymmetricTriNet).
  - NO SupCon loss — baselines use CE only (they have no projection head).
  - Input to baselines: x_stft only (2-channel, native spatial size or
    224×224 for VGG-16 which requires it).
  - Checkpoints saved as: <model_name>_baseline_seed{SEED}_n{N}.pt

IMPORTANT: uses split_dataset correctly — receives two tuples,
keeps them as tuples, unpacks only when constructing DataLoader.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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

import torch
import torch.nn as nn

from python.src.legacy_models import (
    LiteratureBaseline_VGG16,
    LiteratureBaseline_ResNet18,
    LiteratureBaseline_DenseNet121,
)
from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor, split_dataset
from python.src.utils import (
    create_train_loader, create_eval_loader,
    resolve_device, FeatureTensorDataset, prepare_unique_file,
)

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 55
N_PER_CLASS   = 2500
SPEC_VER      = "v2"
N_EPOCHS      = 50
PHASE1_EPOCHS = 10          # head-only warm-up
LR_PHASE1     = 1e-3        # head-only LR (higher — head is randomly init'd)
LR_PHASE2     = 1e-4        # last-block + head LR (lower — pretrained weights)
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 32
NUM_CLASSES   = 10

MODELS = {
    "vgg_16":       LiteratureBaseline_VGG16,
    "resnet_18":    LiteratureBaseline_ResNet18,
    "densenet_121": LiteratureBaseline_DenseNet121,
}


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_datasets(project_root: Path):
    """
    Load training and test datasets.
    Returns:
        train_split  — tuple of (x_stft, x_iq, x_if, y) for training
        val_split    — tuple of (x_stft, x_iq, x_if, y) for validation
        test_data    — tuple of (x_stft, x_iq, x_if, y) for final test
    """
    train_path = (
        project_root / "artifacts" / "datasets" / "impaired"
        / f"impaired_dataset_{SPEC_VER}_seed{SEED}_n{N_PER_CLASS}_train.mat"
    )
    eval_path = (
        project_root / "artifacts" / "datasets" / "impaired"
        / f"impaired_dataset_{SPEC_VER}_seed{SEED}_n{N_PER_CLASS}_eval.mat"
    )

    for p in (train_path, eval_path):
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    # Training artifact → split into train / val
    train_artifact   = load_artifact(str(train_path), load_params=False)
    x_stft, x_iq, x_if, y = build_feature_tensor(train_artifact)

    # split_dataset returns two tuples — keep them as tuples
    train_set, val_set = split_dataset(
        x_stft, x_iq, x_if, y, train_ratio=0.8, seed=SEED
    )
    # Extract tensors from the Subsets (as you already know works)
    x_stft_tr = train_set.dataset.tensors[0][train_set.indices]
    x_iq_tr = train_set.dataset.tensors[1][train_set.indices]
    x_if_tr = train_set.dataset.tensors[2][train_set.indices]
    y_tr = train_set.dataset.tensors[3][train_set.indices]

    x_stft_val = val_set.dataset.tensors[0][val_set.indices]
    x_iq_val = val_set.dataset.tensors[1][val_set.indices]
    x_if_val = val_set.dataset.tensors[2][val_set.indices]
    y_val = val_set.dataset.tensors[3][val_set.indices]

    train_split = (x_stft_tr, x_iq_tr, x_if_tr, y_tr)
    val_split = (x_stft_val, x_iq_val, x_if_val, y_val)

    # Held-out test set
    eval_artifact = load_artifact(str(eval_path), load_params=False)
    x_stft_te, x_iq_te, x_if_te, y_te = build_feature_tensor(eval_artifact)
    test_data = (x_stft_te, x_iq_te, x_if_te, y_te)

    return train_split, val_split, test_data


def make_loaders(split_tuple, batch_size: int, shuffle: bool):
    """
    Construct a DataLoader from a (x_stft, x_iq, x_if, y) tuple.
    Unpack only here, not in the caller.
    """
    x_stft, x_iq, x_if, y = split_tuple
    ds = FeatureTensorDataset(x_stft, x_iq, x_if, y)
    if shuffle:
        return create_train_loader(ds, batch_size=batch_size)
    return create_eval_loader(ds, batch_size=batch_size)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x_stft, x_iq, x_if, y in loader:
        x_stft, x_iq, x_if, y = (
            x_stft.to(device), x_iq.to(device), x_if.to(device), y.to(device)
        )
        logits = model(x_stft, x_iq, x_if)
        loss   = criterion(logits, y)
        bs     = y.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += bs
    return total_loss / total_n, total_correct / total_n


def run_phase(
    model, loader, optimizer, scheduler,
    criterion, device, start_epoch, end_epoch,
    val_loader, phase_label, log_epochs,
):
    """Generic training loop for one phase. Returns best_val_acc, best_state."""
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for x_stft, x_iq, x_if, y in loader:
            x_stft, x_iq, x_if, y = (
                x_stft.to(device), x_iq.to(device), x_if.to(device), y.to(device)
            )
            optimizer.zero_grad()
            logits = model(x_stft, x_iq, x_if)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss    += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n       += bs

        train_loss = total_loss / total_n
        train_acc  = total_correct / total_n
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == start_epoch or epoch % 5 == 0 or epoch == end_epoch:
            print(
                f"  [{phase_label}] Ep {epoch:02d} | LR {lr_now:.6f} | "
                f"Train {train_loss:.4f} ({100*train_acc:.1f}%) | "
                f"Val {val_loss:.4f} ({100*val_acc:.2f}%)"
            )

        log_epochs.append({
            "epoch":       epoch,
            "phase":       phase_label,
            "train_loss":  train_loss,
            "train_acc":   train_acc,
            "val_loss":    val_loss,
            "val_accuracy": val_acc,
            "lr":          lr_now,
        })

    return best_val_acc, best_state


def train_one_baseline(
    name: str,
    model_cls,
    train_split, val_split, test_data,
    device, project_root: Path,
):
    ckpt_name = f"{name}_baseline_seed{SEED}_n{N_PER_CLASS}.pt"
    ckpt_dir  = project_root / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name

    if ckpt_path.exists():
        print(f"\n[SKIP] {name} — checkpoint exists: {ckpt_path.name}")
        return

    print(f"\n{'='*70}")
    print(f"  Baseline: {name}")
    print(f"  Phase 1: epochs 1–{PHASE1_EPOCHS} (head only, LR={LR_PHASE1})")
    print(f"  Phase 2: epochs {PHASE1_EPOCHS+1}–{N_EPOCHS} (last block, LR={LR_PHASE2})")
    print(f"{'='*70}")

    set_seed(SEED)

    model = model_cls(num_classes=NUM_CLASSES).to(device)

    train_loader = make_loaders(train_split, BATCH_SIZE, shuffle=True)
    val_loader   = make_loaders(val_split,   BATCH_SIZE, shuffle=False)
    test_loader  = make_loaders(test_data,   BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    log_epochs: list = []
    global_best_acc   = 0.0
    global_best_state = None

    # ── Phase 1: head only ────────────────────────────────────────────
    model.freeze_for_phase1()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Phase 1 trainable params: {trainable:,}")

    opt1 = torch.optim.Adam(
        model.trainable_parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_EPOCHS)

    best1, state1 = run_phase(
        model, train_loader, opt1, sch1, criterion, device,
        start_epoch=1, end_epoch=PHASE1_EPOCHS,
        val_loader=val_loader, phase_label="Phase1",
        log_epochs=log_epochs,
    )
    if best1 > global_best_acc:
        global_best_acc, global_best_state = best1, state1

    # ── Phase 2: last block + head ────────────────────────────────────
    model.unfreeze_for_phase2()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Phase 2 trainable params: {trainable:,}")

    phase2_epochs = N_EPOCHS - PHASE1_EPOCHS
    opt2 = torch.optim.Adam(
        model.trainable_parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY
    )
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs)

    best2, state2 = run_phase(
        model, train_loader, opt2, sch2, criterion, device,
        start_epoch=PHASE1_EPOCHS + 1, end_epoch=N_EPOCHS,
        val_loader=val_loader, phase_label="Phase2",
        log_epochs=log_epochs,
    )
    if best2 > global_best_acc:
        global_best_acc, global_best_state = best2, state2

    # ── Final test evaluation ─────────────────────────────────────────
    model.load_state_dict(global_best_state)
    test_loss, test_acc = evaluate_loader(model, test_loader, criterion, device)
    print(f"\n  Best Val Acc : {100*global_best_acc:.2f}%")
    print(f"  Test Acc     : {100*test_acc:.2f}%  (loss: {test_loss:.4f})")

    # ── Save checkpoint ───────────────────────────────────────────────
    torch.save(global_best_state, ckpt_path)
    print(f"  Checkpoint   : {ckpt_path}")

    # ── Save log ──────────────────────────────────────────────────────
    log = {
        "created_utc":    datetime.now(timezone.utc).isoformat(),
        "model":          name,
        "seed":           SEED,
        "n_per_class":    N_PER_CLASS,
        "phase1_epochs":  PHASE1_EPOCHS,
        "phase2_epochs":  N_EPOCHS - PHASE1_EPOCHS,
        "best_val_acc":   global_best_acc,
        "test_acc":       test_acc,
        "test_loss":      test_loss,
        "epochs":         log_epochs,
    }
    log_dir = project_root / "artifacts" / "logs" / "baselines"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{name}_baseline_seed{SEED}_n{N_PER_CLASS}.json").open("w") as f:
        json.dump(log, f, indent=2)


def main():
    device = resolve_device("auto")
    print(f"Device: {device}")

    print("\nLoading datasets...")
    train_split, val_split, test_data = load_datasets(PROJECT_ROOT)
    # unpack only to print shapes — not passed anywhere unpacked
    print(f"  Train : {train_split[0].shape[0]} samples")
    print(f"  Val   : {val_split[0].shape[0]} samples")
    print(f"  Test  : {test_data[0].shape[0]} samples")

    for name, cls in MODELS.items():
        train_one_baseline(
            name=name,
            model_cls=cls,
            train_split=train_split,
            val_split=val_split,
            test_data=test_data,
            device=device,
            project_root=PROJECT_ROOT,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n\nAll baselines trained.")


if __name__ == "__main__":
    main()
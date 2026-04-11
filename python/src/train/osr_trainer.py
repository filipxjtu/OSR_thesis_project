from __future__ import annotations

import json
from dataclasses import asdict

from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from ..utils import create_train_loader, create_eval_loader, resolve_device, load_osr_datasets, prepare_unique_file
from ..models import SparseFingerprint_TS_DRSN
from ..analysis import generate_osr_confusion_outputs, plot_osr_feature_embedding

from .osr_engine import train_phase1_epoch, train_phase2_epoch, evaluate_osr
from .osr_hparams import OSRHParams


def train_osr_model(
        *,
        seed: int,
        n_per_class: int,
        spec_version: str,
        project_root: Path,
        epochs: int = 50,
        hparams: OSRHParams | None = None,
        pretrained_path: Path | None = None,
):
    if hparams is None:
        hparams = OSRHParams()

    # Validation Gate
    report_dir = project_root / "reports" / "validations"
    validation_report = report_dir / f"validation_seed{seed}_n{n_per_class}_{spec_version}.json"

    if not validation_report.exists():
        raise RuntimeError("Validation reports missing. Run run_validation.py and run_val_unknowns.py first.")

    # Setup
    torch.manual_seed(seed)
    device = resolve_device("auto")

    print(f"\n{'=' * 60}")
    print(f"SparseFingerprint OSR | seed={seed} | n={n_per_class}")
    print(f"Device        : {device}")
    print(f"Phase 1 (backbone) : epochs  1 – {hparams.warmup_epochs}")
    print(f"Phase 2 (calibrator): epochs  {hparams.warmup_epochs + 1} – {epochs}")
    print(f"{'=' * 60}\n")

    # Data
    datasets = load_osr_datasets(project_root, seed, n_per_class, spec_version)

    train_loader = create_train_loader(datasets["train"], hparams.batch_size, device)
    val_loader_known = create_eval_loader(datasets["val_known"], hparams.batch_size, device)
    val_loader_osr = create_eval_loader(datasets["val_unknown"], hparams.batch_size, device)
    test_loader_known = create_eval_loader(datasets["test_known"], hparams.batch_size, device)
    test_loader_osr = create_eval_loader(datasets["test_unknown"], hparams.batch_size, device)

    # Model & Optimizers
    model = SparseFingerprint_TS_DRSN(
        num_classes=10,
        k_centroids=hparams.k_centroids,
        ema_momentum=hparams.ema_momentum,
        warmup_epochs=hparams.warmup_epochs,
        use_pretrained=pretrained_path is not None,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
    ).to(device)

    opt_backbone = torch.optim.Adam(
        model.base.parameters(),
        lr=hparams.lr_backbone, weight_decay=1e-4,
    )
    sched_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_backbone, T_max=hparams.warmup_epochs
    )

    opt_calibrator = torch.optim.Adam(
        list(model.score_calibrator.parameters()) + [model._log_temperature],
        lr=hparams.lr_calibrator, weight_decay=1e-5,
    )
    sched_calibrator = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_calibrator, T_max=max(1, epochs - hparams.warmup_epochs)
    )

    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    training_log = []
    best_auroc = 0.0
    best_state = None

    print(f"{'Ep':<5} | {'Ph':<3} | {'Loss':<8} | {'KnAcc':<7} | {'AUROC':<7} | {'Recall':<7} | {'Codebook'}")
    print("-" * 65)

    # Training Loop
    for epoch in range(1, epochs + 1):
        phase = model.current_phase(epoch)
        model.set_phase(epoch)

        if phase == 1:
            avg_loss = train_phase1_epoch(model, train_loader, opt_backbone, criterion_ce, device)
            sched_backbone.step()
        else:
            avg_loss = train_phase2_epoch(
                model, train_loader, opt_calibrator,
                hparams.lambda_osr, hparams.lambda_entropy, device
            )
            sched_calibrator.step()

        # Validation
        val_known_acc = _eval_known_acc(model, val_loader_known, device)
        val_auroc, val_recall = 0.0, 0.0

        if phase == 2:
            val_auroc, val_recall = _eval_osr(model, val_loader_known, val_loader_osr, device)

        cb_stats = model.get_codebook_stats()
        cb_str = f"{float(cb_stats['pct_initialised']) * 100:.0f}%" if cb_stats else "—"

        print(f"{epoch:02d}/{epochs} | P{phase}  | {avg_loss:<8.4f} | "
              f"{val_known_acc:<7.3f} | {val_auroc:<7.4f} | "
              f"{val_recall:<7.3f} | {cb_str}")

        training_log.append({
            "epoch": epoch,
            "phase": phase,
            "train_loss": avg_loss,
            "val_known_acc": val_known_acc,
            "val_auroc": val_auroc,
            "val_unknown_recall": val_recall,
        })

        if phase == 2 and val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load Best Checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best Phase-2 checkpoint (val AUROC = {best_auroc:.4f})")

    # Final Test Evaluation
    test_known_acc = _eval_known_acc(model, test_loader_known, device)
    test_auroc, test_recall = _eval_osr(model, test_loader_known, test_loader_osr, device)
    _, _, _, test_fpr = evaluate_osr(model, test_loader_known, test_loader_osr, device)

    print(f"\n{'=' * 52}")
    print(f"FINAL TEST RESULTS")
    print(f"Known accuracy  : {test_known_acc:.4f}")
    print(f"AUROC           : {test_auroc:.4f}")
    print(f"Unknown recall  : {test_recall:.4f}")
    print(f"False alarm rate: {test_fpr:.4f}")
    print(f"{'=' * 52}\n")

    # Save Artifacts
    ckpt_name = f"sparse_fingerprint_seed{seed}_n{n_per_class}.pt"
    ckpt_dir = prepare_unique_file(project_root / "artifacts" / "checkpoints", ckpt_name)
    torch.save(model.state_dict(), ckpt_dir)

    log_name = f"sparse_fingerprint_seed{seed}_n{n_per_class}.json"
    log_dir = prepare_unique_file(project_root / "artifacts" / "logs" / "osr_training", log_name )
    with open(log_dir, "w") as f:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "n_per_class": n_per_class,
            "total_epochs": epochs,
            "hparams": asdict(hparams),
            "test_metrics": {
                "test_acc": test_known_acc,
                "test_auroc": test_auroc,
                "test_recall": test_recall,
                "test_fpr": test_fpr,
            },
            "history": training_log,
        }, f, indent=4)

    fig_name = f"sparse_fingerprint_seed{seed}_n{n_per_class}"
    fig_base_dir = project_root / "reports" / "figures"
    fig_dir = prepare_unique_file(fig_base_dir, fig_name)
    print(f"Generating OSR diagnostics in: {fig_dir}")

    generate_osr_confusion_outputs(model, test_loader_known, test_loader_osr, device, fig_dir, unknown_threshold=0.5)
    plot_osr_feature_embedding(model, test_loader_known, test_loader_osr, device, fig_dir)

    return model

# Internal evaluation helpers
@torch.no_grad()
def _eval_known_acc(model, loader, device) -> float:
    """Closed-set accuracy on known-only val batches via backbone logits."""
    model.eval()
    correct, total = 0, 0
    for x_stft, x_iq, y in loader:
        x_stft, x_iq, y = x_stft.to(device), x_iq.to(device), y.to(device)
        emb = model.base.extract_embedding(x_stft, x_iq)
        logits = model.base.classifier(emb)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _eval_osr(model, loader_known, loader_osr, device) -> tuple[float, float]:
    """Returns (auroc, unknown_recall)."""
    _, auroc, recall, _ = evaluate_osr(model, loader_known, loader_osr, device)
    return auroc, recall
"""
scripts/run_codebook_ablation.py
=================================
Trains all 4 codebook ablation variants of OsrSAF_TriNet:

    "neither" → backbone-only calibrator (cosine + hamming features zeroed)
    "cosine"  → cosine codebook only (hamming features zeroed)
    "hamming" → hamming codebook only (cosine features zeroed)
    "full"    → both codebooks active (control)

Each variant is trained with the SAME seed, dataset, and hyperparameters
to ensure a fair comparison. Phase 1 (codebook fill) runs on a frozen
pretrained AsymmetricTriNet backbone; Phase 2 trains the score_calibrator
on proxy unknowns.

Checkpoints saved as:
    osr_saf_trinet_codebook_ablation_<tag>_seed{SEED}_n{N_PER_CLASS}.pt

Logs saved as:
    artifacts/logs/codebook_ablation/codebook_ablation_<tag>_seed{SEED}_n{N_PER_CLASS}.json

Run from project root:
    python scripts/run_codebook_ablation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


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
import numpy as np
import torch

from python.src.models.ablation_osr_saf_trinet import AblationOsrSAF_TriNet
from python.src.train.osr_engine import (
    populate_codebook_epoch,
    train_phase2_epoch,
    evaluate_osr,
    collect_validation_scores,
)
from python.src.train.osr_hparams import OSRHParams
from python.src.utils import (
    create_train_loader,
    create_eval_loader,
    resolve_device,
    load_osr_datasets,
)


# ── Codebook-ablation variant definitions ─────────────────────────────────────
# (tag, disabled_codebooks, human_label)
ABLATION_VARIANTS = [
    ("neither", {"cosine", "hamming"}, "Neither codebook (backbone-only floor)"),
    ("cosine",  {"hamming"},           "Cosine codebook only"),
    ("hamming", {"cosine"},            "Hamming codebook only"),
    ("full",    set(),                 "Full (both codebooks)"),
]

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
N_PER_CLASS  = 2500
SPEC_VER     = "v2"
NUM_CLASSES  = 10
EPOCHS       = 50          # Phase 2 epochs (matches osr_trainer default)
CODEBOOK_FILL_EPOCHS = 15   # matches osr_trainer.CODEBOOK_FILL_EPOCHS

HP = OSRHParams()


def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_variant(
    *,
    tag: str,
    disabled_codebooks: set,
    label: str,
    datasets: dict,
    device: torch.device,
    project_root: Path,
) -> dict:
    """Train one codebook-ablation variant. Returns log dict."""

    ckpt_name = f"osr_saf_trinet_codebook_ablation_{tag}_seed{SEED}_n{N_PER_CLASS}.pt"
    ckpt_dir  = project_root / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name

    if ckpt_path.exists():
        print(f"\n[SKIP] {label} — checkpoint exists: {ckpt_path.name}")
        return {"skipped": True, "ckpt": str(ckpt_path)}

    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"  Disabled codebooks: {sorted(disabled_codebooks) or 'none (full)'}")
    print(f"{'='*70}")

    set_seed(SEED)

    # ── Pretrained backbone path ─────────────────────────────────────────────
    pretrained_path = (
        project_root / "artifacts" / "checkpoints"
        / f"asymmetric_trinet_seed{SEED}_n{N_PER_CLASS}.pt"
    )
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Closed-set checkpoint not found: {pretrained_path}\n"
            f"Train the closed-set asymmetric_trinet first."
        )

    # ── Model ────────────────────────────────────────────────────────────────
    model = AblationOsrSAF_TriNet(
        num_classes=NUM_CLASSES,
        k_centroids=HP.k_centroids,
        ema_momentum=HP.ema_momentum,
        warmup_epochs=HP.warmup_epochs,
        codebook_beta=HP.codebook_beta,
        threshold_recal_interval=HP.threshold_recal_interval,
        use_pretrained=True,
        pretrained_path=str(pretrained_path),
        disabled_codebooks=disabled_codebooks,
    ).to(device)

    # Freeze backbone — Phase 2 only trains the calibrator
    for p in model.base.parameters():
        p.requires_grad = False
    model.base.eval()

    # ── Loaders ──────────────────────────────────────────────────────────────
    train_loader      = create_train_loader(datasets["train"],       HP.batch_size, device)
    val_loader_known  = create_eval_loader(datasets["val_known"],    HP.batch_size, device)
    val_loader_osr    = create_eval_loader(datasets["val_unknown"],  HP.batch_size, device)
    test_loader_known = create_eval_loader(datasets["test_known"],   HP.batch_size, device)
    test_loader_osr   = create_eval_loader(datasets["test_unknown"], HP.batch_size, device)

    # ── Stage 2.A: Codebook fill ─────────────────────────────────────────────
    print(f"\n  [Stage 2.A] Populating codebooks over {CODEBOOK_FILL_EPOCHS} epochs (frozen backbone)")
    for fill_epoch in range(1, CODEBOOK_FILL_EPOCHS + 1):
        populate_codebook_epoch(model, train_loader, device, epoch=fill_epoch)

        if fill_epoch == 1 or fill_epoch % 5 == 0 or fill_epoch == CODEBOOK_FILL_EPOCHS:
            cb_stats = model.get_codebook_stats()
            pct     = float(cb_stats["pct_initialised"]) * 100
            spread  = float(cb_stats["spread_per_class"].mean())
            updates = float(cb_stats["mean_updates_per_centroid"])
            print(f"    Fill epoch {fill_epoch:02d}/{CODEBOOK_FILL_EPOCHS} | "
                  f"init={pct:.0f}% | spread={spread:.4f} | updates/centroid={updates:.1f}")

    model.phase2_active = True

    # ── Stage 2.B: Calibrator training ───────────────────────────────────────
    opt_calibrator = torch.optim.Adam(
        model.score_calibrator.parameters(),
        lr=HP.lr_calibrator,
        weight_decay=1e-5,
    )
    sched_calibrator = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_calibrator, T_max=EPOCHS
    )

    training_log: list[dict] = []
    best_auroc = 0.0
    best_state = None

    print(f"\n  [Stage 2.B] Training calibrator on proxy unknowns")
    print(f"  {'Ep':<5} | {'Loss':<7} | {'KnAcc':<6} | {'AUROC':<6} | "
          f"{'Recall':<6} | {'FPR':<6} | {'Thr':<6}")
    print("  " + "-" * 64)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_phase2_epoch(
            model, train_loader, opt_calibrator, HP.lambda_osr, device,
        )
        sched_calibrator.step()

        # Threshold recalibration each epoch (matches osr_trainer with interval=1)
        val_scores, val_preds = collect_validation_scores(
            model, val_loader_known, device,
        )
        if val_scores.numel() > 0:
            unk_scores, _ = collect_validation_scores(model, val_loader_osr, device)
            model.calibrate_class_thresholds_youden(
                scores_known=val_scores,
                pred_known=val_preds,
                scores_unknown=unk_scores,
                pred_unknown=torch.zeros_like(unk_scores, dtype=torch.long),
                fpr_cap=HP.fpr_cap,
                verbose=False,
            )

        # Validation OSR metrics
        known_acc, auroc, unk_recall, far = evaluate_osr(
            model, val_loader_known, val_loader_osr, device,
        )
        thr = float(model.class_thresholds.mean().item())

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  {epoch:<5d} | {avg_loss:<7.4f} | "
                  f"{100*known_acc:<6.2f} | {auroc:<6.4f} | "
                  f"{100*unk_recall:<6.2f} | {100*far:<6.2f} | {thr:<6.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        training_log.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_known_acc": known_acc,
            "val_auroc": auroc,
            "val_unknown_recall": unk_recall,
            "val_false_alarm_rate": far,
            "mean_threshold": thr,
            "lr": sched_calibrator.get_last_lr()[0],
        })

    # ── Restore best, evaluate on held-out test ──────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final threshold calibration with best calibrator state
    val_scores, val_preds = collect_validation_scores(model, val_loader_known, device)
    unk_scores, _ = collect_validation_scores(model, val_loader_osr, device)
    if val_scores.numel() > 0 and unk_scores.numel() > 0:
        model.calibrate_class_thresholds_youden(
            scores_known=val_scores, pred_known=val_preds,
            scores_unknown=unk_scores,
            pred_unknown=torch.zeros_like(unk_scores, dtype=torch.long),
            fpr_cap=HP.fpr_cap, verbose=True,
        )

    test_known_acc, test_auroc, test_recall, test_far = evaluate_osr(
        model, test_loader_known, test_loader_osr, device,
    )

    print(f"\n  Best Val AUROC : {best_auroc:.4f}")
    print(f"  Test Known Acc : {100*test_known_acc:.2f}%")
    print(f"  Test AUROC     : {test_auroc:.4f}")
    print(f"  Test Recall    : {100*test_recall:.2f}%")
    print(f"  Test FAR       : {100*test_far:.2f}%")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint     : {ckpt_path}")

    # ── Save log ─────────────────────────────────────────────────────────────
    log = {
        "created_utc":         datetime.now(timezone.utc).isoformat(),
        "ablation_tag":        tag,
        "ablation_label":      label,
        "disabled_codebooks":  sorted(disabled_codebooks),
        "seed":                SEED,
        "n_per_class":         N_PER_CLASS,
        "epochs":              EPOCHS,
        "codebook_fill_epochs": CODEBOOK_FILL_EPOCHS,
        "best_val_auroc":      best_auroc,
        "test_known_accuracy": test_known_acc,
        "test_auroc":          test_auroc,
        "test_unknown_recall": test_recall,
        "test_false_alarm_rate": test_far,
        "history":             training_log,
    }

    log_dir = project_root / "artifacts" / "logs" / "codebook_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"codebook_ablation_{tag}_seed{SEED}_n{N_PER_CLASS}.json"
    with log_path.open("w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log            : {log_path}")

    return log


def main():
    device = resolve_device("auto")
    print(f"Device: {device}")
    print(f"Seed: {SEED} | n_per_class: {N_PER_CLASS} | spec: {SPEC_VER}")

    print("\nLoading OSR datasets...")
    datasets = load_osr_datasets(PROJECT_ROOT, SEED, N_PER_CLASS, SPEC_VER)

    for tag, disabled, label in ABLATION_VARIANTS:
        train_variant(
            tag=tag,
            disabled_codebooks=disabled,
            label=label,
            datasets=datasets,
            device=device,
            project_root=PROJECT_ROOT,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n\nAll codebook ablation variants trained.")


if __name__ == "__main__":
    main()
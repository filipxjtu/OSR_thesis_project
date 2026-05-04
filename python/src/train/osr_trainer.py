from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..utils import (
    create_train_loader,
    create_eval_loader,
    resolve_device,
    load_osr_datasets,
    prepare_unique_file,
)
from ..models import OsrSAF_TriNet
from ..analysis import generate_osr_confusion_outputs, plot_osr_eval_feature_embedding

from .osr_engine import (
    populate_codebook_epoch,
    train_phase2_epoch,
    evaluate_osr,
    collect_validation_scores,
 )
from .osr_hparams import OSRHParams


CODEBOOK_FILL_EPOCHS = 15

def train_osr_model(
    *,
    seed: int,
    n_per_class: int,
    spec_version: str,
    project_root: Path,
    epochs: int = 50,
    hparams: OSRHParams | None = None,
):
    if hparams is None:
        hparams = OSRHParams()

    report_dir = project_root / "reports" / "validations"
    validation_report = report_dir / f"validation_seed{seed}_n{n_per_class}_{spec_version}.json"

    if not validation_report.exists():
        raise RuntimeError(
            "Validation reports missing. Run run_validation.py first."
        )

    pretrained_path = (
        project_root
        / "artifacts"
        / "checkpoints"
        / f"asymmetric_trinet_seed{seed}_n{n_per_class}.pt"
    )
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Closed-set checkpoint not found: {pretrained_path}\n"
            f"Train the closed-set asymmetric_trinet first via train_model_runner."
        )

    torch.manual_seed(seed)
    device = resolve_device("auto")

    print(f"\n{'=' * 60}")
    print(f"OsrSAF_TriNet | seed={seed} | n={n_per_class}")
    print(f"Device              : {device}")
    print(f"Closed-set ckpt     : {pretrained_path.name}")
    print(f"Codebook fill epochs: {CODEBOOK_FILL_EPOCHS}")
    print(f"Phase 2 (calibrator): until epoch {epochs}")
    print(f"FPR cap (Youden)    : {hparams.fpr_cap:.2f}")
    print(f"Recal interval      : {hparams.threshold_recal_interval}")
    print(f"{'=' * 60}\n")

    datasets = load_osr_datasets(project_root, seed, n_per_class, spec_version)

    train_loader      = create_train_loader(datasets["train"],        hparams.batch_size, device)
    val_loader_known  = create_eval_loader(datasets["val_known"],     hparams.batch_size, device)
    val_loader_osr    = create_eval_loader(datasets["val_unknown"],   hparams.batch_size, device)
    test_loader_known = create_eval_loader(datasets["test_known"],    hparams.batch_size, device)
    test_loader_osr   = create_eval_loader(datasets["test_unknown"],  hparams.batch_size, device)

    model = OsrSAF_TriNet(
        num_classes=10,
        k_centroids=hparams.k_centroids,
        ema_momentum=hparams.ema_momentum,
        warmup_epochs=hparams.warmup_epochs,
        codebook_beta=hparams.codebook_beta,
        threshold_recal_interval=hparams.threshold_recal_interval,
        use_pretrained=True,
        pretrained_path=str(pretrained_path),
    ).to(device)

    for p in model.base.parameters():
        p.requires_grad = False
    model.base.eval()

    print(f"\n[Stage 2.A] Populating codebook over {CODEBOOK_FILL_EPOCHS} epochs (frozen backbone)\n")
    for fill_epoch in range(1, CODEBOOK_FILL_EPOCHS + 1):
        populate_codebook_epoch(model, train_loader, device, epoch=fill_epoch)
        cb_stats = model.get_codebook_stats()
        pct = float(cb_stats["pct_initialised"]) * 100
        spread = float(cb_stats["spread_per_class"].mean())
        updates = float(cb_stats["mean_updates_per_centroid"])
        print(f"  Fill epoch {fill_epoch}/{CODEBOOK_FILL_EPOCHS} | init={pct:.0f}% | spread={spread:.4f} | updates/centroid={updates:.1f}")

    model.phase2_active = True

    opt_calibrator = torch.optim.Adam(
        model.score_calibrator.parameters(),
        lr=hparams.lr_calibrator,
        weight_decay=1e-5,
    )
    sched_calibrator = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_calibrator, T_max=epochs
    )

    training_log: list[dict] = []
    best_auroc  = 0.0
    best_state  = None
    phase2_epoch_count = 0

    print(f"\n[Stage 2.B] Training calibrator on proxy unknowns\n")
    print(
        f"{'Ep':<6} | {'Loss':<7} | {'KnAcc':<6} | "
        f"{'AUROC':<6} | {'Recall':<6} | {'FPR':<6} | {'J':<7} | {'Thr':<6} | {'Codebook'}"
    )
    print("-" * 92)

    for epoch in range(1, epochs + 1):
        avg_loss = train_phase2_epoch(
            model, train_loader, opt_calibrator,
            hparams.lambda_osr, device,
        )
        sched_calibrator.step()
        phase2_epoch_count += 1

        # ---- Threshold recal every epoch (interval default = 1) -----------
        # By recal'ing every epoch, the saved thresholds always reflect the
        # CURRENT calibrator state, so epoch-to-epoch metric comparisons are
        # apples-to-apples. Old behavior (interval=5) caused threshold to
        # change discontinuously, making val_J across epochs noncomparable.
        if phase2_epoch_count % hparams.threshold_recal_interval == 0:
            sk, pk = collect_validation_scores(model, val_loader_known, device)
            su, pu = collect_validation_scores(model, val_loader_osr,   device)
            if sk.numel() > 0 and su.numel() > 0:
                model.calibrate_class_thresholds_youden(
                    sk, pk, su, pu,
                    fpr_cap=hparams.fpr_cap,
                    verbose=False,
                )

        # ---- Validation metrics --------------------------------------------
        val_known_acc = _eval_known_acc(model, val_loader_known, device)
        _, val_auroc, val_recall, val_fpr = evaluate_osr(
            model, val_loader_known, val_loader_osr, device
        )
        val_j = val_recall - val_fpr
        cur_thr = float(model.class_thresholds[0].item())   # all equal under global

        cb_stats = model.get_codebook_stats()
        cb_str = f"{float(cb_stats['pct_initialised']) * 100:.0f}%"

        print(
            f"{epoch:02d}/{epochs} | {avg_loss:<7.4f} | "
            f"{val_known_acc:<6.3f} | {val_auroc:<6.4f} | "
            f"{val_recall:<6.3f} | {val_fpr:<6.3f} | {val_j:<+7.4f} | "
            f"{cur_thr:<6.3f} | {cb_str}"
        )

        training_log.append({
            "epoch":                epoch,
            "train_loss":           avg_loss,
            "val_known_acc":        val_known_acc,
            "val_auroc":            val_auroc,
            "val_unknown_recall":   val_recall,
            "val_fpr":              val_fpr,
            "val_recall_minus_fpr": val_j,
            "threshold":            cur_thr,
        })

        # Model selection: best AUROC. With threshold recal'd every epoch,
        # AUROC is a clean, threshold-invariant criterion — it asks "which
        # calibrator state gives the best score separation?" and the
        # subsequent Youden cap takes care of the operating point.
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best calibrator checkpoint (val AUROC = {best_auroc:.4f})")

    # ---- Final Youden recal on the loaded best state ----------------------
    sk, pk = collect_validation_scores(model, val_loader_known, device)
    su, pu = collect_validation_scores(model, val_loader_osr,   device)
    if sk.numel() > 0 and su.numel() > 0:
        info = model.calibrate_class_thresholds_youden(
            sk, pk, su, pu, fpr_cap=hparams.fpr_cap, verbose=True,
        )

    test_known_acc = _eval_known_acc(model, test_loader_known, device)
    _, test_auroc, test_recall, test_fpr = evaluate_osr(
        model, test_loader_known, test_loader_osr, device
    )

    print(f"\n{'=' * 52}")
    print(f"FINAL TEST RESULTS")
    print(f"Known accuracy  : {test_known_acc:.4f}")
    print(f"AUROC           : {test_auroc:.4f}")
    print(f"Unknown recall  : {test_recall:.4f}")
    print(f"False alarm rate: {test_fpr:.4f}")
    print(f"Youden's J      : {test_recall - test_fpr:+.4f}")
    print(f"Final threshold : {float(model.class_thresholds[0]):.4f}")
    print(f"{'=' * 52}\n")

    ckpt_name = f"osr_saf_trinet_seed{seed}_n{n_per_class}.pt"
    ckpt_path = prepare_unique_file(
        project_root / "artifacts" / "checkpoints", ckpt_name
    )
    torch.save(model.state_dict(), ckpt_path)

    log_name = f"osr_saf_trinet_seed{seed}_n{n_per_class}.json"
    log_path = prepare_unique_file(
        project_root / "artifacts" / "logs" / "osr_training", log_name
    )
    with open(log_path, "w") as f:
        json.dump(
            {
                "created_utc":      datetime.now(timezone.utc).isoformat(),
                "seed":             seed,
                "n_per_class":      n_per_class,
                "total_epochs":     epochs,
                "fill_epochs":      CODEBOOK_FILL_EPOCHS,
                "pretrained_ckpt":  pretrained_path.name,
                "hparams":          asdict(hparams),
                "test_metrics": {
                    "test_acc":    test_known_acc,
                    "test_auroc":  test_auroc,
                    "test_recall": test_recall,
                    "test_fpr":    test_fpr,
                    "test_j":      test_recall - test_fpr,
                },
                "history": training_log,
            },
            f,
            indent=4,
        )

    fig_name = f"osr_saf_trinet_seed{seed}_n{n_per_class}"
    fig_dir  = prepare_unique_file(
        project_root / "reports" / "figures", fig_name
    )
    print(f"Generating OSR diagnostics in: {fig_dir}")
    generate_osr_confusion_outputs(
        model, test_loader_known, test_loader_osr, device, fig_dir
    )
    plot_osr_eval_feature_embedding(
        model, test_loader_known, test_loader_osr, device, fig_dir
    )

    return model


@torch.no_grad()
def _eval_known_acc(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x_stft, x_iq, x_if, y in loader:
        x_stft = x_stft.to(device)
        x_iq   = x_iq.to(device)
        x_if   = x_if.to(device)
        y      = y.to(device)
        logits = model(x_stft, x_iq, x_if)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _eval_osr(model, loader_known, loader_osr, device) -> tuple[float, float]:
    _, auroc, recall, _ = evaluate_osr(model, loader_known, loader_osr, device)
    return auroc, recall
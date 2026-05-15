"""
scripts/run_codebook_ablation_eval.py
=======================================
Evaluates all 4 codebook-ablation variant checkpoints across the 13
SNR-specific eval datasets and prints + saves a unified ablation table.

For every (variant, SNR) pair the script saves:
  - reports/figures/codebook_ablation/<variant>/snr_<snr>dB/osr_confusion_matrix.png
  - reports/figures/codebook_ablation/<variant>/snr_<snr>dB/osr_tsne_embedding.png
  - reports/figures/codebook_ablation/<variant>/snr_<snr>dB/osr_per_class_accuracy.json

And one unified results JSON at:
  - artifacts/logs/codebook_ablation/codebook_ablation_eval_results_seed{SEED}_n{N}.json

SNR seed mapping:
    seed → SNR dB
    410  →  +10
    118  →   +8
    276  →   +6
    314  →   +4
    152  →   +2
    340  →    0
    142  →   -2
    264  →   -4
    336  →   -6
    608  →   -8
    530  →  -10
    472  →  -12
    214  →  -14

Run from project root:
    python scripts/run_codebook_ablation_eval.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


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
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset

from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor
from python.src.utils import create_eval_loader, resolve_device

from python.src.models.ablation_osr_saf_trinet import AblationOsrSAF_TriNet
from python.src.analysis import (
    generate_osr_confusion_outputs,
    plot_osr_eval_feature_embedding,
)


# ── Config ────────────────────────────────────────────────────────────────────
SEED            = 42
N_PER_CLASS     = 2500       # used in checkpoint name
EVAL_N          = 500        # per-class size of SNR-specific eval datasets
SPEC_VER        = "v2"
NUM_CLASSES     = 10
BATCH_SIZE      = 32

# SNR seed → dB level (high SNR first, then descending in 2 dB steps)
SNR_MAP: Dict[int, int] = {
    410: +10,
    118:  +8,
    276:  +6,
    314:  +4,
    152:  +2,
    340:   0,
    142:  -2,
    264:  -4,
    336:  -6,
    608:  -8,
    530: -10,
    472: -12,
    214: -14,
}

# Variant definitions — must match training script
ABLATION_VARIANTS = [
    ("neither", {"cosine", "hamming"}, "Neither codebook"),
    ("cosine",  {"hamming"},           "Cosine only"),
    ("hamming", {"cosine"},            "Hamming only"),
    ("full",    set(),                 "Full (both)"),
]


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_known_loader(eval_seed: int, project_root: Path, device: torch.device):
    """SNR-specific impaired_eval dataset → known loader."""
    path = (
        project_root / "artifacts" / "datasets" / "impaired"
        / f"impaired_dataset_{SPEC_VER}_seed{eval_seed}_n{EVAL_N}_eval.mat"
    )
    if not path.exists():
        raise FileNotFoundError(f"Eval (known) dataset not found: {path}")

    art = load_artifact(str(path), load_params=False)
    x_stft, x_iq, x_if, y = build_feature_tensor(art)
    ds = TensorDataset(x_stft, x_iq, x_if, y)
    return create_eval_loader(ds, batch_size=BATCH_SIZE, device=device)


def load_unknown_loader(eval_seed: int, project_root: Path, device: torch.device) -> Optional[object]:
    """SNR-specific unknown test dataset → unknown loader. Returns None if not present."""
    base_dir = project_root / "artifacts" / "datasets" / "unknown"
    candidates = [
        base_dir / f"unknown_dataset_{SPEC_VER}_seed{eval_seed}_n{EVAL_N}_test.mat",
        base_dir / f"unknown_dataset_{SPEC_VER}_seed{eval_seed}_n{EVAL_N}.mat",
    ]
    for path in candidates:
        if path.exists():
            art = load_artifact(str(path), load_params=False)
            x_stft, x_iq, x_if, _ = build_feature_tensor(art)
            n = x_stft.size(0)
            y_unk = torch.full((n,), -1, dtype=torch.long)
            ds = TensorDataset(x_stft, x_iq, x_if, y_unk)
            return create_eval_loader(ds, batch_size=BATCH_SIZE, device=device)

    print(f"  [WARN] No unknown dataset found at SNR seed {eval_seed} (tried _test.mat and legacy).")
    return None


# ── Variant loader ────────────────────────────────────────────────────────────

def load_variant_model(tag: str, disabled: set, device: torch.device, project_root: Path):
    ckpt_path = (
        project_root / "artifacts" / "checkpoints"
        / f"osr_saf_trinet_codebook_ablation_{tag}_seed{SEED}_n{N_PER_CLASS}.pt"
    )
    if not ckpt_path.exists():
        print(f"  [MISSING] Checkpoint not found: {ckpt_path.name}")
        return None

    model = AblationOsrSAF_TriNet(
        num_classes=NUM_CLASSES,
        disabled_codebooks=disabled,
        use_pretrained=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_at_snr(
    model: torch.nn.Module,
    known_loader,
    unknown_loader,
    device: torch.device,
) -> dict:
    """Compute OSR metrics for one (variant, SNR) point. Mirrors osr_evaluator logic."""
    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_preds:  List[np.ndarray] = []
    all_final:  List[np.ndarray] = []

    for loader in (known_loader, unknown_loader):
        if loader is None:
            continue
        for x_stft, x_iq, x_if, y in loader:
            x_stft = x_stft.to(device)
            x_iq   = x_iq.to(device)
            x_if   = x_if.to(device)
            logits, score = model.forward_with_osr(x_stft, x_iq, x_if)
            preds = logits.argmax(dim=1)
            final = preds.clone()
            final[score > model.class_thresholds[preds]] = -1

            all_labels.append(y.cpu().numpy())
            all_scores.append(score.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_final.append(final.cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    scores_arr = np.concatenate(all_scores)
    preds_arr  = np.concatenate(all_preds)
    final_arr  = np.concatenate(all_final)

    known_mask   = labels_arr != -1
    unknown_mask = labels_arr == -1

    known_acc = float(np.mean(preds_arr[known_mask] == labels_arr[known_mask])) if known_mask.any() else 0.0
    open_set_acc = float(np.mean(labels_arr == final_arr))

    if unknown_mask.any() and known_mask.any():
        binary_labels = unknown_mask.astype(int)
        try:
            auroc = float(roc_auc_score(binary_labels, scores_arr))
        except Exception:
            auroc = 0.5
    else:
        auroc = float("nan")

    unk_recall = float(np.mean(final_arr[unknown_mask] == -1)) if unknown_mask.any() else float("nan")
    far        = float(np.mean(final_arr[known_mask]   == -1)) if known_mask.any()   else float("nan")

    return {
        "known_accuracy":     known_acc,
        "auroc":              auroc,
        "unknown_recall":     unk_recall,
        "false_alarm_rate":   far,
        "open_set_accuracy":  open_set_acc,
    }


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _nanmean(values):
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))])
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _print_table(results: Dict[str, Dict[int, dict]], metric: str, metric_label: str, fmt="{:6.2f}"):
    """Print one metric (e.g. AUROC) variant × SNR table."""
    sorted_snrs = sorted(SNR_MAP.values())  # ascending: -14 → +10

    print("\n" + "=" * 110)
    print(f"CODEBOOK ABLATION — {metric_label}")
    print("=" * 110)

    header = f"{'Variant':<22}" + "".join(f" {s:+4d}dB" for s in sorted_snrs) + " | Mean[-10:-4] | Overall"
    print(header)
    print("-" * len(header))

    for tag, _, label in ABLATION_VARIANTS:
        if tag not in results:
            continue
        per_snr = results[tag]
        # per_snr is keyed by SNR dB, not seed
        row_vals = [per_snr.get(s, {}).get(metric, float("nan")) for s in sorted_snrs]
        mean_low = _nanmean([per_snr.get(s, {}).get(metric) for s in [-10, -8, -6, -4]])
        overall  = _nanmean([per_snr.get(s, {}).get(metric) for s in sorted_snrs])

        # AUROC and recall/FAR tables: print in 2-decimal % for accuracies, 4-decimal for AUROC
        if metric == "auroc":
            row_str = f"{label:<22}" + "".join(f" {v:6.4f}" if not np.isnan(v) else " {:>6}".format("nan") for v in row_vals)
            row_str += f" | {mean_low:11.4f} | {overall:.4f}"
        else:
            row_str = f"{label:<22}" + "".join(f" {100*v:6.2f}" if not np.isnan(v) else " {:>6}".format("nan") for v in row_vals)
            row_str += f" | {100*mean_low:10.2f}% | {100*overall:.2f}%"
        print(row_str)


def _emit_latex(results: Dict[str, Dict[int, dict]], metric: str, metric_label: str) -> str:
    """Build a LaTeX table for one metric across the key SNR columns -8/-6/-4 dB plus mean[-10:-4] and overall."""
    key_snrs    = [-8, -6, -4]
    sorted_snrs = sorted(SNR_MAP.values())

    lines = []
    lines.append(f"% LaTeX — {metric_label}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\hline")
    if metric == "auroc":
        lines.append(r"Variant & -8 dB & -6 dB & -4 dB & Mean[-10:-4] & Overall \\")
    else:
        lines.append(r"Variant & -8 dB & -6 dB & -4 dB & Mean[-10:-4] (\%) & Overall (\%) \\")
    lines.append(r"\hline")

    for tag, _, label in ABLATION_VARIANTS:
        if tag not in results:
            continue
        per_snr = results[tag]
        cols = [per_snr.get(s, {}).get(metric, float("nan")) for s in key_snrs]
        mean_low = _nanmean([per_snr.get(s, {}).get(metric) for s in [-10, -8, -6, -4]])
        overall  = _nanmean([per_snr.get(s, {}).get(metric) for s in sorted_snrs])

        if metric == "auroc":
            row = " & ".join(f"{v:.4f}" if not np.isnan(v) else "n/a" for v in cols)
            lines.append(f"{label} & {row} & {mean_low:.4f} & {overall:.4f} \\\\")
        else:
            row = " & ".join(f"{100*v:.2f}" if not np.isnan(v) else "n/a" for v in cols)
            lines.append(f"{label} & {row} & {100*mean_low:.2f} & {100*overall:.2f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = resolve_device("auto")
    print(f"Device: {device}")
    print(f"Seed: {SEED} | n_per_class (ckpt): {N_PER_CLASS} | n_per_class (eval): {EVAL_N}\n")

    # Pre-build all SNR-specific loaders
    print("Loading eval datasets per SNR point...")
    sorted_seeds = sorted(SNR_MAP.items(), key=lambda kv: kv[1])  # ascending dB

    known_loaders   = {}
    unknown_loaders = {}
    for eval_seed, snr_db in sorted_seeds:
        try:
            known_loaders[eval_seed] = load_known_loader(eval_seed, PROJECT_ROOT, device)
            unknown_loaders[eval_seed] = load_unknown_loader(eval_seed, PROJECT_ROOT, device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {snr_db:+d} dB (seed {eval_seed}): {e}")
    print(f"  Loaded {len(known_loaders)} SNR points.\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    # results[tag][snr_db] = { metrics_dict }
    results: Dict[str, Dict[int, dict]] = {}

    fig_root = PROJECT_ROOT / "reports" / "figures" / "codebook_ablation"
    log_dir  = PROJECT_ROOT / "artifacts" / "logs" / "codebook_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)

    for tag, disabled, label in ABLATION_VARIANTS:
        print(f"\n{'='*70}")
        print(f"  Evaluating: {label}")
        print(f"  Disabled codebooks: {sorted(disabled) or 'none (full)'}")
        print(f"{'='*70}")

        model = load_variant_model(tag, disabled, device, PROJECT_ROOT)
        if model is None:
            continue

        results[tag] = {}

        for eval_seed, snr_db in sorted_seeds:
            if eval_seed not in known_loaders:
                continue

            kl = known_loaders[eval_seed]
            ul = unknown_loaders.get(eval_seed)

            metrics = evaluate_at_snr(model, kl, ul, device)
            results[tag][snr_db] = metrics

            print(f"  SNR {snr_db:+4d} dB  →  "
                  f"KnAcc {100*metrics['known_accuracy']:5.2f}% | "
                  f"AUROC {metrics['auroc']:.4f} | "
                  f"Recall {100*metrics['unknown_recall']:5.2f}% | "
                  f"FAR {100*metrics['false_alarm_rate']:5.2f}% | "
                  f"OS-Acc {100*metrics['open_set_accuracy']:5.2f}%")

            # ── Diagnostics: confusion matrix + t-SNE + per-class JSON ───────
            snr_label = f"{snr_db:+d}dB".replace("+", "p").replace("-", "m")
            out_dir = fig_root / tag / f"snr_{snr_label}"
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                generate_osr_confusion_outputs(
                    model=model,
                    loader_known=kl, loader_osr=ul,
                    device=device, out_dir=out_dir, n_classes=NUM_CLASSES,
                )
            except Exception as e:
                print(f"    [WARN] confusion matrix failed: {e}")

            try:
                plot_osr_eval_feature_embedding(
                    model=model,
                    loader_known=kl, loader_osr=ul,
                    device=device, out_dir=out_dir, n_classes=NUM_CLASSES,
                    title_suffix=f" — {label} @ SNR {snr_db:+d} dB",
                )
            except Exception as e:
                print(f"    [WARN] t-SNE failed: {e}")

        # Free model GPU memory between variants
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print all four metric tables ─────────────────────────────────────────
    _print_table(results, "auroc",            "AUROC by SNR")
    _print_table(results, "known_accuracy",   "Known Accuracy (%) by SNR")
    _print_table(results, "unknown_recall",   "Unknown Recall (%) by SNR")
    _print_table(results, "open_set_accuracy", "Open-Set Accuracy (%) by SNR")
    _print_table(results, "false_alarm_rate", "False Alarm Rate (%) by SNR")

    # ── Save unified results JSON ────────────────────────────────────────────
    sorted_snrs = sorted(SNR_MAP.values())
    serialisable = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_per_class_ckpt": N_PER_CLASS,
        "n_per_class_eval": EVAL_N,
        "spec_version": SPEC_VER,
        "snr_map": {str(seed): db for seed, db in SNR_MAP.items()},
        "variants": {
            tag: {
                "label": label,
                "disabled_codebooks": sorted(disabled),
                "per_snr": {
                    str(snr_db): results.get(tag, {}).get(snr_db, {})
                    for snr_db in sorted_snrs
                },
                "summary": {
                    "mean_auroc_overall":           _nanmean([results.get(tag, {}).get(s, {}).get("auroc")           for s in sorted_snrs]),
                    "mean_known_acc_overall":       _nanmean([results.get(tag, {}).get(s, {}).get("known_accuracy")  for s in sorted_snrs]),
                    "mean_unknown_recall_overall":  _nanmean([results.get(tag, {}).get(s, {}).get("unknown_recall")  for s in sorted_snrs]),
                    "mean_open_set_acc_overall":    _nanmean([results.get(tag, {}).get(s, {}).get("open_set_accuracy") for s in sorted_snrs]),
                    "mean_far_overall":             _nanmean([results.get(tag, {}).get(s, {}).get("false_alarm_rate") for s in sorted_snrs]),
                    "mean_auroc_low_snr":           _nanmean([results.get(tag, {}).get(s, {}).get("auroc")           for s in [-10, -8, -6, -4]]),
                    "mean_known_acc_low_snr":       _nanmean([results.get(tag, {}).get(s, {}).get("known_accuracy")  for s in [-10, -8, -6, -4]]),
                    "mean_unknown_recall_low_snr":  _nanmean([results.get(tag, {}).get(s, {}).get("unknown_recall")  for s in [-10, -8, -6, -4]]),
                    "mean_open_set_acc_low_snr":    _nanmean([results.get(tag, {}).get(s, {}).get("open_set_accuracy") for s in [-10, -8, -6, -4]]),
                },
            }
            for tag, disabled, label in ABLATION_VARIANTS
            if tag in results
        },
    }

    out_json = log_dir / f"codebook_ablation_eval_results_seed{SEED}_n{N_PER_CLASS}.json"
    with out_json.open("w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # ── LaTeX export ─────────────────────────────────────────────────────────
    latex_blocks = [
        _emit_latex(results, "auroc",            "AUROC"),
        _emit_latex(results, "known_accuracy",   "Known Accuracy"),
        _emit_latex(results, "unknown_recall",   "Unknown Recall"),
        _emit_latex(results, "open_set_accuracy", "Open-Set Accuracy"),
    ]
    latex_path = log_dir / f"codebook_ablation_tables_seed{SEED}_n{N_PER_CLASS}.tex"
    with latex_path.open("w") as f:
        f.write("\n\n".join(latex_blocks))
    print(f"LaTeX tables saved: {latex_path}\n")
    for block in latex_blocks:
        print(block)
        print()


if __name__ == "__main__":
    main()
"""
scripts/openmax_eval.py
========================
Evaluates OpenMaxTriNet and OsrSAF_TriNet across the 13 SNR-specific eval
datasets and prints + saves the comparison table.

SNR seed mapping (matches the ablation study and run_osr_evaluation.py):
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

For each method × SNR we report:
    AUROC        — rejection-quality summary, threshold-invariant
    OS-Acc       — open-set accuracy (combined known correct + unknown rejected)
    Known-Acc    — closed-set accuracy on knowns at this SNR
    Unk-Recall   — fraction of unknowns rejected
    FAR          — fraction of knowns wrongly rejected

Run from project root:
    python scripts/openmax_eval.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root (no 'artifacts' directory found).")

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.src.dataio import load_artifact
from python.src.models.openmax_trinet import OpenMaxTriNet
from python.src.models import OsrSAF_TriNet
from python.src.preprocessing import build_feature_tensor
from python.src.utils import create_eval_loader, resolve_device


# ── Config ────────────────────────────────────────────────────────────────────
CKPT_SEED          = 216
CKPT_N_PER_CLASS   = 2500
EVAL_N_PER_CLASS   = 500
SPEC_VER           = "v2"
NUM_CLASSES        = 10
BATCH_SIZE         = 32

# SNR seed → dB level (s1 = +10 dB, ..., s13 = -14 dB; step 2 dB)
SNR_MAP = {
    410: +10, 118:  +8, 276:  +6, 314:  +4,
    152:  +2, 340:   0, 142:  -2, 264:  -4,
    336:  -6, 608:  -8, 530: -10, 472: -12, 214: -14,
}

# Methods to compare. (tag, label, factory)
def _build_openmax(device):
    ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / f"openmax_trinet_seed{CKPT_SEED}_n{CKPT_N_PER_CLASS}.pt"
    if not ckpt.exists():
        return None, ckpt
    model = OpenMaxTriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


def _build_osr_saf(device):
    ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / f"osr_saf_trinet_seed{CKPT_SEED}_n{CKPT_N_PER_CLASS}.pt"
    if not ckpt.exists():
        return None, ckpt
    model = OsrSAF_TriNet(num_classes=NUM_CLASSES, use_pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


METHODS = [
    ("openmax",  "OpenMax",  _build_openmax),
    ("osr_saf",  "OSR-SAF",  _build_osr_saf),
]


# ── Loaders ──────────────────────────────────────────────────────────────────

def _resolve_unknown_path(eval_seed: int) -> Path:
    new_path = (
        PROJECT_ROOT / "artifacts" / "datasets" / "unknown"
        / f"unknown_dataset_{SPEC_VER}_seed{eval_seed}_n{EVAL_N_PER_CLASS}_test.mat"
    )
    if new_path.exists():
        return new_path
    legacy = (
        PROJECT_ROOT / "artifacts" / "datasets" / "unknown"
        / f"unknown_dataset_{SPEC_VER}_seed{eval_seed}_n{EVAL_N_PER_CLASS}.mat"
    )
    return legacy if legacy.exists() else new_path


def load_eval_loaders(eval_seed: int, device):
    """Returns (known_loader, unknown_loader) for one SNR-specific dataset pair."""
    known_path = (
        PROJECT_ROOT / "artifacts" / "datasets" / "impaired"
        / f"impaired_dataset_{SPEC_VER}_seed{eval_seed}_n{EVAL_N_PER_CLASS}_eval.mat"
    )
    unknown_path = _resolve_unknown_path(eval_seed)

    if not known_path.exists():
        raise FileNotFoundError(f"Known eval dataset missing: {known_path}")
    if not unknown_path.exists():
        raise FileNotFoundError(f"Unknown eval dataset missing: {unknown_path}")

    k = load_artifact(str(known_path), load_params=False)
    xk_stft, xk_iq, xk_if, yk = build_feature_tensor(k)
    known_loader = create_eval_loader(
        torch.utils.data.TensorDataset(xk_stft, xk_iq, xk_if, yk),
        batch_size=BATCH_SIZE, device=device,
    )

    u = load_artifact(str(unknown_path), load_params=False)
    xu_stft, xu_iq, xu_if, _ = build_feature_tensor(u)
    unknown_loader = create_eval_loader(
        torch.utils.data.TensorDataset(
            xu_stft, xu_iq, xu_if, torch.full((xu_stft.size(0),), -1),
        ),
        batch_size=BATCH_SIZE, device=device,
    )

    return known_loader, unknown_loader


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_one(model, known_loader, unknown_loader, device):
    """Returns dict with 5 metrics. Works for any model that exposes forward_with_osr."""
    all_labels, all_scores, all_preds, all_final = [], [], [], []

    for loader in (known_loader, unknown_loader):
        for x_stft, x_iq, x_if, y in loader:
            logits, score = model.forward_with_osr(
                x_stft.to(device), x_iq.to(device), x_if.to(device),
            )
            preds = logits.argmax(dim=1)
            final = preds.clone()

            # OSR-SAF uses per-class thresholds; OpenMax uses a single scalar threshold.
            # Detect which one is in play and apply correctly.
            if hasattr(model, "class_thresholds") and model.class_thresholds.numel() > 1:
                final[score > model.class_thresholds[preds]] = -1
            else:
                thr = model.threshold if hasattr(model, "threshold") else model.class_thresholds[0]
                final[score > thr] = -1

            all_labels.append(y.numpy())
            all_scores.append(score.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_final.append(final.cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    scores_arr = np.concatenate(all_scores)
    preds_arr  = np.concatenate(all_preds)
    final_arr  = np.concatenate(all_final)

    known_mask   = labels_arr != -1
    unknown_mask = labels_arr == -1

    known_acc        = float(accuracy_score(labels_arr[known_mask], preds_arr[known_mask])) if known_mask.any() else 0.0
    open_set_acc     = float(np.mean(labels_arr == final_arr))
    binary_labels    = (labels_arr == -1).astype(int)
    try:
        auroc = float(roc_auc_score(binary_labels, scores_arr))
    except ValueError:
        auroc = 0.5
    unknown_recall   = float(np.mean(final_arr[unknown_mask] == -1)) if unknown_mask.any() else 0.0
    false_alarm_rate = float(np.mean(final_arr[known_mask]   == -1)) if known_mask.any() else 0.0

    return {
        "known_accuracy":   known_acc,
        "auroc":            auroc,
        "unknown_recall":   unknown_recall,
        "false_alarm_rate": false_alarm_rate,
        "open_set_accuracy": open_set_acc,
    }


# ── Pretty printing ──────────────────────────────────────────────────────────

def _nanmean(values: List) -> float:
    valid = [v for v in values if v is not None and v == v]
    return sum(valid) / len(valid) if valid else float("nan")


def print_table(results: Dict[str, Dict[int, dict]], metric_key: str, metric_label: str,
                as_pct: bool = True):
    print("\n" + "=" * 110)
    print(f"{metric_label} by SNR")
    print("=" * 110)

    sorted_snrs = sorted({s for m in results.values() for s in m.keys()})
    header = f"{'Method':<12}" + "".join(f" {s:+4d}dB" for s in sorted_snrs) + " | Mean[-10:-4] | Overall"
    print(header)
    print("-" * len(header))

    table_rows = []
    for tag, label, _ in METHODS:
        if tag not in results:
            continue
        per_snr = results[tag]
        row_vals = []
        for s in sorted_snrs:
            v = per_snr.get(s, {}).get(metric_key)
            if v is None:
                row_vals.append(float("nan"))
            else:
                row_vals.append(100.0 * v if as_pct else v)
        mean_low = _nanmean([per_snr.get(s, {}).get(metric_key) for s in [-10, -8, -6, -4]])
        overall  = _nanmean([per_snr.get(s, {}).get(metric_key) for s in sorted_snrs])
        if as_pct:
            mean_low *= 100.0; overall *= 100.0

        row_str = f"{label:<12}" + "".join(f" {v:6.2f}" for v in row_vals)
        suffix = "%" if as_pct else " "
        row_str += f" | {mean_low:9.2f}{suffix}  | {overall:6.2f}{suffix}"
        print(row_str)
        table_rows.append({
            "tag":         tag,
            "label":       label,
            "metric":      metric_key,
            "values":      {int(s): per_snr.get(s, {}).get(metric_key) for s in sorted_snrs},
            "mean_low":    _nanmean([per_snr.get(s, {}).get(metric_key) for s in [-10, -8, -6, -4]]),
            "overall":     _nanmean([per_snr.get(s, {}).get(metric_key) for s in sorted_snrs]),
        })

    print("=" * 110)
    print("Mean[-10:-4] = mean of SNR -10, -8, -6, -4 dB (transition regime)")
    print("Overall      = mean across all SNR points")
    return table_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = resolve_device("auto")
    print(f"Device: {device}")
    print(f"Checkpoint seed/n: {CKPT_SEED} / {CKPT_N_PER_CLASS}")
    print(f"Eval n_per_class : {EVAL_N_PER_CLASS}\n")

    sorted_seed_snr = sorted(SNR_MAP.items(), key=lambda x: x[1])
    eval_seeds      = [s for s, _ in sorted_seed_snr]

    print("Loading eval datasets (one pair per SNR)...")
    loader_pairs = {}
    for eval_seed in eval_seeds:
        try:
            loader_pairs[eval_seed] = load_eval_loaders(eval_seed, device)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    print(f"  Loaded {len(loader_pairs)} dataset pairs.\n")

    results: Dict[str, Dict[int, dict]] = {}

    for tag, label, factory in METHODS:
        print(f"Evaluating: {label}")
        model, ckpt_path = factory(device)
        if model is None:
            print(f"  [MISSING] {ckpt_path.name} not found — skipping {label}.\n")
            continue

        per_snr = {}
        for eval_seed, snr_db in sorted_seed_snr:
            if eval_seed not in loader_pairs:
                continue
            kl, ul = loader_pairs[eval_seed]
            metrics = evaluate_one(model, kl, ul, device)
            per_snr[snr_db] = metrics
            print(f"  SNR {snr_db:+4d} dB → AUROC {metrics['auroc']:.4f} | "
                  f"OS-Acc {100*metrics['open_set_accuracy']:.2f}% | "
                  f"KnAcc {100*metrics['known_accuracy']:.2f}% | "
                  f"Recall {100*metrics['unknown_recall']:.2f}% | "
                  f"FAR {100*metrics['false_alarm_rate']:.2f}%")

        results[tag] = per_snr

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    # ── Tables ───────────────────────────────────────────────────────────────
    auroc_rows  = print_table(results, "auroc",             "AUROC",                 as_pct=False)
    osacc_rows  = print_table(results, "open_set_accuracy", "Open-Set Accuracy (%)", as_pct=True)
    kacc_rows   = print_table(results, "known_accuracy",    "Known Accuracy (%)",    as_pct=True)
    recall_rows = print_table(results, "unknown_recall",    "Unknown Recall (%)",    as_pct=True)
    far_rows    = print_table(results, "false_alarm_rate",  "False Alarm Rate (%)",  as_pct=True)

    # ── LaTeX summary (AUROC + OS-Acc) ──────────────────────────────────────
    print("\n\nLaTeX (AUROC at key SNR columns):")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Method & -8 dB & -6 dB & -4 dB & Mean[-10:-4] & Overall \\\\")
    print("\\hline")
    for tag, label, _ in METHODS:
        if tag not in results:
            continue
        accs = results[tag]
        cols = [accs.get(s, {}).get("auroc", float("nan")) for s in [-8, -6, -4]]
        ml   = _nanmean([accs.get(s, {}).get("auroc") for s in [-10, -8, -6, -4]])
        ov   = _nanmean([accs.get(s, {}).get("auroc") for s in accs.keys()])
        line = f"{label} & " + " & ".join(f"{v:.4f}" for v in cols)
        line += f" & {ml:.4f} & {ov:.4f} \\\\"
        print(line)
    print("\\hline")
    print("\\end{tabular}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    out_dir  = PROJECT_ROOT / "artifacts" / "logs" / "openmax"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openmax_vs_osr_saf_seed{CKPT_SEED}_n{CKPT_N_PER_CLASS}.json"

    payload = {
        "created_utc":     datetime.now(timezone.utc).isoformat(),
        "ckpt_seed":       CKPT_SEED,
        "ckpt_n_per_class": CKPT_N_PER_CLASS,
        "eval_n_per_class": EVAL_N_PER_CLASS,
        "spec_version":    SPEC_VER,
        "snr_map":         {str(k): v for k, v in SNR_MAP.items()},
        "results":         {tag: {str(snr): met for snr, met in per.items()} for tag, per in results.items()},
        "tables": {
            "auroc":             auroc_rows,
            "open_set_accuracy": osacc_rows,
            "known_accuracy":    kacc_rows,
            "unknown_recall":    recall_rows,
            "false_alarm_rate":  far_rows,
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
"""
scripts/run_ablation_eval.py
==============================
Evaluates all 7 ablation variant checkpoints across the 13 SNR-specific
eval datasets and prints + saves the ablation table.

SNR seed mapping (from your own mapping):
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

Output table columns:
    Variant | -8 dB | -6 dB | -4 dB | Mean[-10:-4] | Overall Mean

Run from project root:
    python scripts/run_ablation_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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
from sklearn.metrics import accuracy_score

from python.src.models.ablation_trinet import AblationTriNet
from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor
from python.src.utils import create_eval_loader, resolve_device, FeatureTensorDataset


# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 55
N_PER_CLASS = 2500
SPEC_VER    = "v2"
NUM_CLASSES = 10
BATCH_SIZE  = 32

# SNR seed → dB level
SNR_MAP = {
    410: +10, 118:  +8, 276:  +6, 314:  +4,
    152:  +2, 340:   0, 142:  -2, 264:  -4,
    336:  -6, 608:  -8, 530: -10, 472: -12, 214: -14,
}

# Ablation variants — same order as training script
ABLATION_VARIANTS = [
    ("stft",    {1, 2}, "STFT only"),
    ("iq",      {0, 2}, "IQ only"),
    ("if",      {0, 1}, "IF only"),
    ("stft_iq", {2},    "STFT + IQ"),
    ("stft_if", {1},    "STFT + IF"),
    ("iq_if",   {0},    "IQ + IF"),
    ("full",    set(),  "STFT + IQ + IF (Full)"),
]


def load_eval_dataset(eval_seed: int, project_root: Path, device: torch.device):
    """Load one SNR-specific eval dataset and return a DataLoader."""
    eval_path = (
        project_root / "artifacts" / "datasets" / "impaired"
        / f"impaired_dataset_{SPEC_VER}_seed{eval_seed}_n500_eval.mat"
    )
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_path}")

    artifact = load_artifact(str(eval_path), load_params=False)
    x_stft, x_iq, x_if, y = build_feature_tensor(artifact)
    ds = FeatureTensorDataset(x_stft, x_iq, x_if, y)
    return create_eval_loader(ds, batch_size=BATCH_SIZE)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device) -> float:
    """Returns accuracy as a percentage."""
    model.eval()
    all_preds, all_labels = [], []
    for x_stft, x_iq, x_if, y in loader:
        x_stft, x_iq, x_if = x_stft.to(device), x_iq.to(device), x_if.to(device)
        logits = model(x_stft, x_iq, x_if)
        preds  = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.tolist())
    return 100.0 * accuracy_score(all_labels, all_preds)


def load_variant_model(tag: str, disabled_branches: set, device: torch.device, project_root: Path):
    ckpt_path = (
        project_root / "artifacts" / "checkpoints"
        / f"asymmetric_trinet_ablation_{tag}_seed{SEED}_n{N_PER_CLASS}.pt"
    )
    if not ckpt_path.exists():
        print(f"  [MISSING] Checkpoint not found: {ckpt_path.name}")
        return None

    model = AblationTriNet(num_classes=NUM_CLASSES, disabled_branches=disabled_branches).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    device = resolve_device("auto")
    print(f"Device: {device}\n")

    # Sort eval seeds by SNR ascending for display
    sorted_seeds = sorted(SNR_MAP.items(), key=lambda x: x[1])  # (seed, snr_db)
    snr_values   = [snr for _, snr in sorted_seeds]
    eval_seeds   = [seed for seed, _ in sorted_seeds]

    # Pre-load all eval datasets
    print("Loading eval datasets...")
    loaders = {}
    for eval_seed in eval_seeds:
        try:
            loaders[eval_seed] = load_eval_dataset(eval_seed, PROJECT_ROOT, device)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    print(f"  Loaded {len(loaders)} datasets.\n")

    # ── Main evaluation loop ──────────────────────────────────────────────────
    results: Dict[str, Dict[int, float]] = {}   # tag → {snr_db: acc}

    for tag, disabled, label in ABLATION_VARIANTS:
        print(f"Evaluating: {label}")
        model = load_variant_model(tag, disabled, device, PROJECT_ROOT)
        if model is None:
            continue

        accs_by_snr = {}
        for eval_seed, snr_db in sorted_seeds:
            if eval_seed not in loaders:
                continue
            acc = evaluate_loader(model, loaders[eval_seed], device)
            accs_by_snr[snr_db] = acc
            print(f"  SNR {snr_db:+4d} dB → {acc:.2f}%")

        results[tag] = accs_by_snr

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    # ── Build and print the summary table ────────────────────────────────────
    print("\n" + "="*90)
    print("ABLATION STUDY — Accuracy (%) by SNR")
    print("="*90)

    # Header
    key_snrs    = [ -10, -8, -6, -4, -2, 0]   # all low-to-mid SNR
    report_snrs = [-8, -6, -4]                           # highlight columns

    header = f"{'Variant':<28}" + "".join(f" {s:+4d}dB" for s in key_snrs) + " | Mean[-10:-4] | Overall"
    print(header)
    print("-" * len(header))

    table_rows = []
    for tag, disabled, label in ABLATION_VARIANTS:
        if tag not in results:
            continue
        accs = results[tag]

        row_values = [accs.get(s, float("nan")) for s in key_snrs]
        mean_low   = _nanmean([accs.get(s) for s in [-10, -8, -6, -4]])
        overall    = _nanmean(list(accs.values()))

        row_str = f"{label:<28}" + "".join(f" {v:6.2f}" for v in row_values)
        row_str += f" | {mean_low:11.2f}% | {overall:.2f}%"
        print(row_str)

        table_rows.append({
            "tag":        tag,
            "label":      label,
            "accs_by_snr": accs,
            "mean_low_snr": mean_low,
            "overall_mean": overall,
        })

    print("="*90)
    print("Mean[-10:-4] = mean of SNR -10, -8, -6, -4 dB (transition regime)")
    print("Overall      = mean across all 13 SNR evaluation points\n")

    # ── Also print a clean LaTeX-ready table ─────────────────────────────────
    print("\nLaTeX table (key SNR columns only: -8, -6, -4 dB + mean[-10:-4] + overall):")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Variant & -8 dB & -6 dB & -4 dB & Mean[-10:-4] & Overall Mean \\\\")
    print("\\hline")
    for row in table_rows:
        accs = row["accs_by_snr"]
        vals = [accs.get(s, float("nan")) for s in [-8, -6, -4]]
        line = f"{row['label']} & " + " & ".join(f"{v:.2f}" for v in vals)
        line += f" & {row['mean_low_snr']:.2f} & {row['overall_mean']:.2f} \\\\"
        print(line)
    print("\\hline")
    print("\\end{tabular}")

    # ── Save JSON results ─────────────────────────────────────────────────────
    out_dir  = PROJECT_ROOT / "artifacts" / "logs" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_eval_results_seed{SEED}_n{N_PER_CLASS}.json"
    with out_path.open("w") as f:
        json.dump(table_rows, f, indent=2)
    print(f"\nResults saved: {out_path}")


def _nanmean(values: List) -> float:
    valid = [v for v in values if v is not None and v == v]  # filter None and NaN
    return sum(valid) / len(valid) if valid else float("nan")


if __name__ == "__main__":
    main()
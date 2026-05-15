from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

BLUE_II = {
    "dark": "#081d58",
    "navy": "#253494",
    "ocean": "#1d91c0",
    "sky": "#41b6c4",
    "gray": "#2c2c2c",
}


def generate_osr_confusion_outputs(
        model: torch.nn.Module,
        loader_known: torch.utils.data.DataLoader | None,
        loader_osr: torch.utils.data.DataLoader | None,
        device: torch.device,
        out_dir: Path,
        n_classes: int = 10,
        snr_label: str = None,
):
    """Generates normalized confusion matrix and saves per-class accuracy JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    y_true, y_predicts = [], []

    with torch.no_grad():
        for loader in (loader_known, loader_osr):
            if loader is None:
                continue
            for x_stft, x_iq, x_if, y in loader:
                # Core logic preserved from your original
                preds, _ = model.predict_with_rejection(
                    x_stft.to(device), x_iq.to(device), x_if.to(device)
                )
                y_true.append(y.cpu().numpy())
                y_predicts.append(preds.cpu().numpy())

    if not y_true:
        return

    y_true = np.concatenate(y_true)
    y_predicts = np.concatenate(y_predicts)

    y_true_mapped = np.where(y_true == -1, n_classes, y_true)
    y_predicts_mapped = np.where(y_predicts == -1, n_classes, y_predicts)

    matrix_size = n_classes + 1
    cm = np.zeros((matrix_size, matrix_size), dtype=int)
    for t, p in zip(y_true_mapped, y_predicts_mapped):
        cm[t, p] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    sns.set_theme(style="white")
    plt.figure(figsize=(9, 8))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", rotation=270, labelpad=15)

    plt.xlabel("Predicted Class", fontweight='bold', color=BLUE_II["dark"])
    plt.ylabel("True Class", fontweight='bold', color=BLUE_II["dark"])
    plt.title("OSR Confusion Matrix", color=BLUE_II["dark"], fontweight='bold')

    tick_marks = list(range(n_classes)) + ["Unknown"]
    plt.xticks(range(matrix_size), tick_marks, rotation=45, ha='right')
    plt.yticks(range(matrix_size), tick_marks)

    for i in range(matrix_size):
        for j in range(matrix_size):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if cm_norm[i, j] > 0.5 else BLUE_II["dark"])

    plt.tight_layout()
    if snr_label:
        cm_filename = f"osr_confusion_matrix_{snr_label.replace(' ', '')}.png"
    else:
        cm_filename = f"osr_confusion_matrix.png"
    plt.savefig(out_dir / cm_filename, dpi=300)
    plt.close()

    per_class_accuracy = {
        f"class_{c}" if c < n_classes else "unknown":
            float(np.mean(y_predicts_mapped[y_true_mapped == c] == c))
        for c in range(matrix_size) if np.sum(y_true_mapped == c) > 0
    }
    with open(out_dir / "osr_per_class_accuracy.json", "w") as f:
        json.dump(per_class_accuracy, f, indent=4)


def plot_osr_eval_feature_embedding(
        model: torch.nn.Module,
        loader_known: torch.utils.data.DataLoader | None,
        loader_osr: torch.utils.data.DataLoader | None,
        device: torch.device,
        out_dir: Path,
        n_classes: int = 10,
        title_suffix: str = "",  # Preserved from your PATCH logic
        snr_label: str = None,
):
    """Standardized t-SNE diagnostic plot for both known and unknown features."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    embeddings, labels = [], []

    with torch.no_grad():
        for loader in (loader_known, loader_osr):
            if loader is None:
                continue
            for x_stft, x_iq, x_if, y in loader:
                feat = model.extract_embedding(x_stft.to(device), x_iq.to(device), x_if.to(device))
                embeddings.append(feat.reshape(feat.size(0), -1).cpu().numpy())
                labels.append(y.cpu().numpy())

    if not embeddings:
        return

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("tab20", n_colors=n_classes)

    for c in range(n_classes):
        idx = labels == c
        if np.any(idx):
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], s=15, alpha=0.7,
                        color=palette[c], label=f"Class {c}", edgecolors='none')

    idx_unk = labels == -1
    if np.any(idx_unk):
        plt.scatter(emb_2d[idx_unk, 0], emb_2d[idx_unk, 1], s=35,
                    color=BLUE_II["dark"], marker='X', alpha=0.9, label="Unknown (Anomalies)")

    plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.title(f"OSR Feature Embedding (t-SNE){title_suffix}", color=BLUE_II["dark"], fontweight='bold')
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    sns.despine()

    # Filename cleanup from your original original logic
    safe_suffix = title_suffix.replace(" ", "_").replace("+", "p").replace("-", "m").replace("—", "").strip("_")
    fname = f"osr_feature_embedding{'_' + safe_suffix if safe_suffix else ''}.png"
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_snr_vs_accuracy(
        results: list[dict],
        seed_to_snr: dict[int, float],
        out_dir: Path,
        ckpt_tag: str = "",
) -> None:
    """
    Plots AUROC, Recall, and FAR vs SNR.
    The 'Known Accuracy' subplot has been removed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = []
    for r in results:
        seed = r["eval_dataset"]["seed"]
        if seed in seed_to_snr:
            points.append((seed_to_snr[seed], r["metrics"]))

    if not points:
        return

    points.sort(key=lambda t: t[0])
    snr_vals = [p[0] for p in points]
    auroc = [p[1]["auroc"] for p in points]
    unk_recall = [100.0 * p[1]["unknown_recall"] for p in points]
    far = [100.0 * p[1]["false_alarm_rate"] for p in points]

    sns.set_theme(style="whitegrid")
    # Figure is now a single ax instead of (1, 2)
    fig, ax = plt.subplots(figsize=(8, 5))

    # AUROC on primary Y-axis
    ax.plot(snr_vals, auroc, marker="o", linewidth=2, color=BLUE_II["ocean"], label="AUROC")
    ax.set_ylabel("AUROC Score", fontweight="bold", color=BLUE_II["dark"])
    ax.set_ylim(0, 1.05)

    # Percentage metrics on secondary Y-axis (ax2_pct from your original)
    ax_pct = ax.twinx()
    ax_pct.plot(snr_vals, unk_recall, marker="s", linewidth=2, linestyle="--", color=BLUE_II["navy"],
                label="Unknown Recall (%)")
    ax_pct.plot(snr_vals, far, marker="^", linewidth=2, linestyle=":", color=BLUE_II["sky"],
                label="False Alarm Rate (%)")
    ax_pct.set_ylabel("(%)", fontweight="bold", color=BLUE_II["dark"])
    ax_pct.set_ylim(0, 105)

    ax.set_xlabel("SNR (dB)", fontweight="bold", color=BLUE_II["dark"])
    ax.set_title(f"OSR Metrics vs SNR — ckpt {ckpt_tag}", color=BLUE_II["dark"], fontweight='bold')
    ax.set_xticks(snr_vals)

    # Unified legend logic
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_pct.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right", frameon=False)

    sns.despine(right=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"osr_snr_accuracy_{ckpt_tag}.png", dpi=300)
    plt.close()
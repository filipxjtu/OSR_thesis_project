from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE


BLUE_II = {
    "dark":  "#081d58",
    "navy":  "#253494",
    "ocean": "#1d91c0",
    "sky":   "#41b6c4",
    "gray":  "#2c2c2c",
}


def generate_osr_confusion_outputs(
        model: torch.nn.Module,
        loader_known: torch.utils.data.DataLoader | None,
        loader_osr: torch.utils.data.DataLoader | None,
        device: torch.device,
        out_dir: Path,
        n_classes: int = 10,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    y_true, y_predicts = [], []

    with torch.no_grad():
        for loader in (loader_known, loader_osr):
            if loader is None:
                continue
            for x_stft, x_iq, x_if, y in loader:
                x_stft = x_stft.to(device)
                x_iq   = x_iq.to(device)
                x_if   = x_if.to(device)

                preds, _ = model.predict_with_rejection(x_stft, x_iq, x_if)

                y_true.append(y.cpu().numpy())
                y_predicts.append(preds.cpu().numpy())

    if not y_true:
        return

    y_true = np.concatenate(y_true)
    y_predicts = np.concatenate(y_predicts)

    y_true_mapped     = np.where(y_true == -1,     n_classes, y_true)
    y_predicts_mapped = np.where(y_predicts == -1, n_classes, y_predicts)

    matrix_size = n_classes + 1
    cm = np.zeros((matrix_size, matrix_size), dtype=int)
    for t, p in zip(y_true_mapped, y_predicts_mapped):
        cm[t, p] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    sns.set_theme(style="white")
    plt.figure(figsize=(9, 8))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", rotation=270, labelpad=15)

    plt.xlabel("Predicted Class", fontweight='bold', color=BLUE_II["dark"], labelpad=10)
    plt.ylabel("True Class",      fontweight='bold', color=BLUE_II["dark"], labelpad=10)
    plt.title("OSR Confusion Matrix (Dynamic Per-Class Thresholds)",
              color=BLUE_II["dark"], fontweight='bold', pad=15)

    tick_marks = list(range(n_classes)) + ["Unknown"]
    plt.xticks(range(matrix_size), tick_marks, rotation=45, ha='right')
    plt.yticks(range(matrix_size), tick_marks)

    for i in range(matrix_size):
        for j in range(matrix_size):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if cm_norm[i, j] > 0.5 else BLUE_II["dark"])

    plt.tight_layout()
    plt.savefig(out_dir / "osr_confusion_matrix.png", dpi=300)
    plt.close()

    per_class_accuracy = {}
    for c in range(matrix_size):
        label_name = f"class_{c}" if c < n_classes else "unknown"
        idx = (y_true_mapped == c)
        acc = float(np.mean(y_predicts_mapped[idx] == y_true_mapped[idx])) if np.sum(idx) > 0 else 0.0
        per_class_accuracy[label_name] = acc

    with open(out_dir / "osr_per_class_accuracy.json", "w") as f:
        json.dump(per_class_accuracy, f, indent=4)


def plot_osr_feature_embedding(
        model: torch.nn.Module,
        loader_known: torch.utils.data.DataLoader | None,
        loader_osr: torch.utils.data.DataLoader | None,
        device: torch.device,
        out_dir: Path,
        n_classes: int = 10,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    embeddings, labels = [], []

    with torch.no_grad():
        for loader in (loader_known, loader_osr):
            if loader is None:
                continue
            for x_stft, x_iq, x_if, y in loader:
                x_stft = x_stft.to(device)
                x_iq   = x_iq.to(device)
                x_if   = x_if.to(device)

                feat = model.extract_embedding(x_stft, x_iq, x_if)
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
            plt.scatter(
                emb_2d[idx, 0], emb_2d[idx, 1],
                s=15, alpha=0.7, color=palette[c],
                label=f"Class {c}", edgecolors='none',
            )

    idx_unk = labels == -1
    if np.any(idx_unk):
        plt.scatter(
            emb_2d[idx_unk, 0], emb_2d[idx_unk, 1],
            s=35, color=BLUE_II["dark"], marker='X', alpha=0.9,
            label="Unknown (Anomalies)",
        )

    plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.title("OSR Feature Embedding (t-SNE)", color=BLUE_II["dark"], fontweight='bold')
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    sns.despine()

    plt.tight_layout()
    plt.savefig(out_dir / "osr_feature_embedding.png", dpi=300)
    plt.close()
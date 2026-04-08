from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def generate_osr_confusion_outputs(model, dataloader, device, out_dir: Path, unknown_threshold: float, n_classes=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    y_true = []
    y_predicts = []

    with torch.no_grad():
        for x_stft, x_iq, y in dataloader:
            x_stft, x_iq, y = x_stft.to(device), x_iq.to(device), y.to(device)

            logits, unknown_score, _ = model.forward_with_osr(x_stft, x_iq)

            # 1. Base prediction from logits
            preds = torch.argmax(logits, dim=1)

            # 2. OSR Override: If unknown score > threshold, classify as Unknown (-1)
            is_unknown = unknown_score > unknown_threshold
            preds[is_unknown] = -1

            y_true.append(y.cpu().numpy())
            y_predicts.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_predicts = np.concatenate(y_predicts)

    # Map -1 (Unknown) to index `n_classes` (e.g., 10) for the confusion matrix
    y_true_mapped = np.where(y_true == -1, n_classes, y_true)
    y_predicts_mapped = np.where(y_predicts == -1, n_classes, y_predicts)

    # We now have n_classes + 1 categories (0 to 10)
    matrix_size = n_classes + 1
    cm = np.zeros((matrix_size, matrix_size), dtype=int)

    for t, p in zip(y_true_mapped, y_predicts_mapped):
        cm[t, p] += 1

    # Normalize rows
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # handle division by zero if a class is missing

    # Plotting confusion matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Proportion")

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"OSR Confusion Matrix (Unknown Threshold = {unknown_threshold:.2f})")

    # Set labels: 0-9 for knowns, "Unk" for the 11th class
    tick_marks = list(range(n_classes)) + ["Unk"]
    plt.xticks(range(matrix_size), tick_marks)
    plt.yticks(range(matrix_size), tick_marks)

    for i in range(matrix_size):
        for j in range(matrix_size):
            plt.text(
                j, i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig(out_dir / "osr_confusion_matrix.png", dpi=300)
    plt.close()

    # Per-class accuracy (including unknown)
    per_class_accuracy = {}
    for c in range(matrix_size):
        label_name = f"class_{c}" if c < n_classes else "unknown"
        idx = (y_true_mapped == c)
        if np.sum(idx) == 0:
            acc = 0.0
        else:
            acc = np.mean(y_predicts_mapped[idx] == y_true_mapped[idx])
        per_class_accuracy[label_name] = float(acc)

    with open(out_dir / "osr_per_class_accuracy.json", "w") as f:
        json.dump(per_class_accuracy, f, indent=2)


def plot_osr_feature_embedding(model, dataloader, device, out_dir: Path, n_classes=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for x_stft, x_iq, y in dataloader:
            x_stft, x_iq = x_stft.to(device), x_iq.to(device)

            # Safely extract features directly (handles both model architectures)
            if hasattr(model, 'extract_features'):
                feat = model.extract_features(x_stft, x_iq)
            elif hasattr(model, 'extract_embedding'):
                feat = model.extract_embedding(x_stft, x_iq)
            else:
                raise AttributeError("Model is missing feature extraction method.")

            # Flatten if necessary
            if feat.dim() > 2:
                feat = feat.reshape(feat.size(0), -1)

            embeddings.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))

    # Plot known classes
    for c in range(n_classes):
        idx = labels == c
        if np.any(idx):
            plt.scatter(
                emb_2d[idx, 0], emb_2d[idx, 1],
                s=10, alpha=0.7, label=f"Class {c}"
            )

    # Plot Unknown class (-1) distinctly
    idx_unk = labels == -1
    if np.any(idx_unk):
        plt.scatter(
            emb_2d[idx_unk, 0], emb_2d[idx_unk, 1],
            s=30, color='black', marker='x', alpha=0.8, label="Unknown"
        )

    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("OSR Feature Embedding (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.tight_layout()
    plt.savefig(out_dir / "osr_feature_embedding.png", dpi=300)
    plt.close()
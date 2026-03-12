from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE


def generate_confusion_outputs(model, dataloader, device, out_dir: Path, n_classes=7):

    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    y_true = []
    y_predicts = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predicts = torch.argmax(logits, dim=1)

            y_true.append(y.cpu().numpy())
            y_predicts.append(predicts.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_predicts = np.concatenate(y_predicts)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_predicts):
        cm[t, p] += 1

    # normalize rows
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # plotting confusion matrix
    plt.figure(figsize=(6,5))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="proportion")

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized Confusion matrix")

    plt.xticks(range(n_classes))
    plt.yticks(range(n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(
                j, i,
                f"{cm_norm[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curves
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    model.eval()
    probs_list = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())

    y_scores = np.concatenate(probs_list)

    plt.figure(figsize=(6, 5))

    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_scores[:, c])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"class {c} (AUC={roc_auc:.2f})"
        )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-class ROC curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=300)
    plt.close()


    # per-class accuracy
    per_class_accuracy = {}

    for c in range(n_classes):

        idx = y_true == c

        if np.sum(idx) == 0:
            acc = 0.0
        else:
            acc = np.mean(y_predicts[idx] == y_true[idx])

        per_class_accuracy[f"class_{c}"] = float(acc)

    with open(out_dir / "per_class_accuracy.json", "w") as f:
        json.dump(per_class_accuracy, f, indent=2)


def plot_cnn_feature_embedding(model, dataloader, device, out_dir, n_classes=7):

    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:

            x = x.to(device)

            feat = model.extract_embedding(x)

            embeddings.append(feat.cpu().numpy())
            labels.append(y.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42
    )

    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6,6))

    for c in range(n_classes):

        idx = labels == c

        plt.scatter(
            emb_2d[idx,0],
            emb_2d[idx,1],
            s=6,
            label=f"class {c}"
        )

    plt.legend(markerscale=3)
    plt.title("CNN Feature Embedding (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.savefig(out_dir / "cnn_feature_embedding.png", dpi=300)
    plt.close()
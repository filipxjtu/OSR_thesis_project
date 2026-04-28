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


# ============================================================================
# ADD THIS FUNCTION TO: python/src/analysis/osr_diagnostics.py
# (and add "plot_snr_vs_accuracy" to the __all__ list in __init__.py)
# ============================================================================

def plot_snr_vs_accuracy(
        results: list[dict],
        seed_to_snr: dict[int, float],
        out_dir: Path,
        ckpt_tag: str = "",
) -> None:
    """
    Plot accuracy / OSR metrics vs SNR (dB) from a list of OSR evaluation
    result dicts (as returned by evaluate_osr_model).

    Parameters
    ----------
    results      : list of result dicts returned by evaluate_osr_model.
    seed_to_snr  : mapping  {eval_seed: snr_db}  that describes which
                   fixed-SNR dataset each seed represents.
    out_dir      : directory where the PNG is saved.
    ckpt_tag     : short string appended to the figure title / filename
                   (e.g. "s38_n2500").
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Collect points, sorted by SNR
    # ------------------------------------------------------------------ #
    points: list[tuple[float, dict]] = []
    for r in results:
        seed = r["eval_dataset"]["seed"]
        if seed not in seed_to_snr:
            continue                          # skip seeds with no SNR label
        points.append((seed_to_snr[seed], r["metrics"]))

    if not points:
        print("  [plot_snr_vs_accuracy] No results matched seed_to_snr — skipping.")
        return

    points.sort(key=lambda t: t[0])
    snr_vals   = [p[0] for p in points]
    known_acc  = [100.0 * p[1]["known_accuracy"]   for p in points]
    open_acc   = [100.0 * p[1]["open_set_accuracy"] for p in points]
    auroc      = [p[1]["auroc"]                     for p in points]
    unk_recall = [100.0 * p[1]["unknown_recall"]    for p in points]
    far        = [100.0 * p[1]["false_alarm_rate"]  for p in points]

    # ------------------------------------------------------------------ #
    # Figure — two subplots side by side
    # ------------------------------------------------------------------ #
    import numpy as np

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    title_suffix = f" — ckpt {ckpt_tag}" if ckpt_tag else ""

    # --- Left: closed-set known accuracy & open-set accuracy ----------- #
    ax1.plot(snr_vals, known_acc, marker="o", linewidth=2,
             color=BLUE_II["ocean"],  label="Closed-set accuracy (known)")
    ax1.plot(snr_vals, open_acc,  marker="s", linewidth=2, linestyle="--",
             color=BLUE_II["navy"],  label="Open-set accuracy")
    ax1.set_xlabel("SNR (dB)", fontweight="bold", color=BLUE_II["dark"])
    ax1.set_ylabel("Accuracy (%)", fontweight="bold", color=BLUE_II["dark"])
    ax1.set_title(f"Accuracy vs SNR{title_suffix}",
                  color=BLUE_II["dark"], fontweight="bold")
    ax1.set_xticks(snr_vals)
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=False)
    sns.despine(ax=ax1)

    # --- Right: AUROC, Unknown Recall, FAR ----------------------------- #
    ax2.plot(snr_vals, auroc,      marker="o", linewidth=2,
             color=BLUE_II["ocean"],  label="AUROC")
    ax2.plot(snr_vals, unk_recall, marker="s", linewidth=2, linestyle="--",
             color=BLUE_II["navy"],  label="Unknown recall (%)")
    ax2.plot(snr_vals, far,        marker="^", linewidth=2, linestyle=":",
             color=BLUE_II["sky"],   label="False alarm rate (%)")
    ax2.set_xlabel("SNR (dB)", fontweight="bold", color=BLUE_II["dark"])
    ax2.set_ylabel("Score", fontweight="bold", color=BLUE_II["dark"])
    ax2.set_title(f"OSR Metrics vs SNR{title_suffix}",
                  color=BLUE_II["dark"], fontweight="bold")
    ax2.set_xticks(snr_vals)
    ax2.set_ylim(0, 1.05)
    # convert % axes to 0-1 scale for AUROC comparability
    ax2_pct = ax2.twinx()
    ax2_pct.set_ylim(0, 105)
    ax2_pct.set_ylabel("(%)", color=BLUE_II["gray"])
    ax2_pct.tick_params(axis="y", labelcolor=BLUE_II["gray"])
    ax2.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax2)

    plt.suptitle("OSR Performance vs SNR", fontsize=13,
                 fontweight="bold", color=BLUE_II["dark"], y=1.01)
    plt.tight_layout()

    fname = f"osr_snr_accuracy{'_' + ckpt_tag if ckpt_tag else ''}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot_snr_vs_accuracy] Saved → {out_dir / fname}")

    # ============================================================================
    # PATCH FOR: python/src/analysis/osr_diagnostics.py
    #
    # The existing plot_osr_feature_embedding signature is:
    #   def plot_osr_feature_embedding(model, loader_known, loader_osr,
    #                                  device, out_dir, n_classes=10):
    #
    # Replace it with the version below, which adds an optional `title_suffix`
    # parameter so the SNR label appears in the plot title and filename.
    # Everything else is identical to what's already in the file.
    # ============================================================================

    def plot_osr_feature_embedding(
            model: torch.nn.Module,
            loader_known: torch.utils.data.DataLoader | None,
            loader_osr: torch.utils.data.DataLoader | None,
            device: torch.device,
            out_dir: Path,
            n_classes: int = 10,
            title_suffix: str = "",  # NEW — e.g. " — SNR +4 dB"
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.eval()

        embeddings, labels = [], []

        with torch.no_grad():
            for loader in (loader_known, loader_osr):
                if loader is None:
                    continue
                for x_stft, x_iq, x_if, y in loader:
                    x_stft = x_stft.to(device)
                    x_iq = x_iq.to(device)
                    x_if = x_if.to(device)

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
                    label=f"Class {c}", edgecolors="none",
                )

        idx_unk = labels == -1
        if np.any(idx_unk):
            plt.scatter(
                emb_2d[idx_unk, 0], emb_2d[idx_unk, 1],
                s=35, color=BLUE_II["dark"], marker="X", alpha=0.9,
                label="Unknown (Anomalies)",
            )

        plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1),
                   loc="upper left", frameon=False)
        plt.title(
            f"OSR Feature Embedding (t-SNE){title_suffix}",
            color=BLUE_II["dark"], fontweight="bold",
        )
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        sns.despine()
        plt.tight_layout()

        # Use suffix in filename so per-SNR plots don't overwrite each other
        safe_suffix = title_suffix.replace(" ", "_").replace("+", "p").replace("-", "m") \
            .replace("—", "").strip("_")
        fname = f"osr_feature_embedding{'_' + safe_suffix if safe_suffix else ''}.png"
        plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [plot_osr_feature_embedding] Saved → {out_dir / fname}")



# ============================================================================
# PATCH FOR: python/src/analysis/osr_diagnostics.py
#
# The existing plot_osr_feature_embedding signature is:
#   def plot_osr_feature_embedding(model, loader_known, loader_osr,
#                                  device, out_dir, n_classes=10):
#
# Replace it with the version below, which adds an optional `title_suffix`
# parameter so the SNR label appears in the plot title and filename.
# Everything else is identical to what's already in the file.
# ============================================================================

def plot_osr_eval_feature_embedding(
        model: torch.nn.Module,
        loader_known: torch.utils.data.DataLoader | None,
        loader_osr: torch.utils.data.DataLoader | None,
        device: torch.device,
        out_dir: Path,
        n_classes: int = 10,
        title_suffix: str = "",          # NEW — e.g. " — SNR +4 dB"
):
    out_dir = Path(out_dir)
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
    labels     = np.concatenate(labels)

    tsne   = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
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
                label=f"Class {c}", edgecolors="none",
            )

    idx_unk = labels == -1
    if np.any(idx_unk):
        plt.scatter(
            emb_2d[idx_unk, 0], emb_2d[idx_unk, 1],
            s=35, color=BLUE_II["dark"], marker="X", alpha=0.9,
            label="Unknown (Anomalies)",
        )

    plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1),
               loc="upper left", frameon=False)
    plt.title(
        f"OSR Feature Embedding (t-SNE){title_suffix}",
        color=BLUE_II["dark"], fontweight="bold",
    )
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    sns.despine()
    plt.tight_layout()

    # Use suffix in filename so per-SNR plots don't overwrite each other
    safe_suffix = title_suffix.replace(" ", "_").replace("+", "p").replace("-", "m") \
                               .replace("—", "").strip("_")
    fname = f"osr_feature_embedding{'_' + safe_suffix if safe_suffix else ''}.png"
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [plot_osr_feature_embedding] Saved → {out_dir / fname}")
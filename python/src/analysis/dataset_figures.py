from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ..dataio.loader import load_artifact
from ..preprocessing.dataset_builder import build_feature_tensor


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_time_domain_features(x, y, out_dir: Path, n_classes=7):
    fig, axes = plt.subplots(n_classes, 1, figsize=(8, 10))

    for c in range(n_classes):
        idx = np.where(y == c)[0][0]
        axes[c].plot(x[:, idx])
        axes[c].set_title(f"Class {c}")
        axes[c].set_ylabel("Amplitude")

    axes[-1].set_xlabel("Sample index")

    plt.tight_layout()
    plt.savefig(out_dir / "time_domain_examples.png", dpi=300)
    plt.close()


def plot_stft_features(x_feat, y, out_dir: Path, n_classes=7):
    fig, axes = plt.subplots(1, n_classes, figsize=(14, 3))

    for c in range(n_classes):
        idx = np.where(y == c)[0][0]
        spec = x_feat[idx,0]

        axes[c].imshow(spec, aspect="auto", origin="lower")
        axes[c].set_title(f"Class {c}")
        axes[c].set_xticks([])
        axes[c].set_yticks([])

    plt.tight_layout()
    plt.savefig(out_dir / "stft_examples.png", dpi=300)
    plt.close()


def plot_class_distribution(y, out_dir: Path):
    classes, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)

    plt.xlabel("Class ID")
    plt.ylabel("Number of samples")
    plt.title("Class distribution")

    plt.savefig(out_dir / "class_distribution.png", dpi=300)
    plt.close()


def plot_feature_energy(x_feat, y, out_dir: Path):
    x_flat = x_feat.reshape(x_feat.shape[0], -1)
    energy = np.sum(x_flat * x_flat, axis=1)

    plt.figure(figsize=(6, 4))

    for c in np.unique(y):
        idx = y == c
        plt.hist(energy[idx], bins=40, alpha=0.5, label=f"class {c}")

    plt.xlabel("Feature energy")
    plt.ylabel("Count")
    plt.legend()

    plt.title("Feature energy distribution")

    plt.savefig(out_dir / "feature_energy_distribution.png", dpi=300)
    plt.close()


def plot_feature_mean_spectrum(x_feat, y, out_dir: Path):
    plt.figure(figsize=(8, 4))

    for c in np.unique(y):
        idx = y == c
        data = x_feat[idx, 0]
        mean_spec = data.mean(axis=(0,2))

        plt.plot(mean_spec, label=f"class {c}")

    plt.xlabel("Frequency bin")
    plt.ylabel("Mean magnitude")
    plt.legend()

    plt.title("Mean spectral signature")

    plt.savefig(out_dir / "mean_spectrum.png", dpi=300)
    plt.close()

def plot_tsne_embedding(x_feat, y, out_dir: Path):

    x_flat = x_feat.reshape(x_feat.shape[0], -1)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    x_emb = tsne.fit_transform(x_flat)

    plt.figure(figsize=(6,6))

    for c in np.unique(y):
        idx = y == c
        plt.scatter(
            x_emb[idx,0],
            x_emb[idx,1],
            s=6,
            label=f"class {c}",
        )

    plt.legend(markerscale=3)
    plt.title("t-SNE embedding of STFT features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.savefig(out_dir / "tsne_embedding.png", dpi=300)
    plt.close()


def generate_dataset_figures(seed: int, n_per_class: int, project_root: Path, spec_version: str = "v1"):

    dataset_dir = project_root / "artifacts" / "datasets"
    fig_dir = project_root / "reports" / "figures" / f"dataset_seed{seed}_n{n_per_class}_{spec_version}"

    _ensure_dir(fig_dir)

    train_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"

    train_artifact = load_artifact(str(train_file))

    x_raw = train_artifact.X
    y = train_artifact.y

    x_feat, y_feat = build_feature_tensor(train_artifact)

    x_feat = x_feat.detach().cpu().numpy()
    y_feat = y_feat.detach().cpu().numpy()

    plot_time_domain_features(x_raw, y, fig_dir)
    plot_stft_features(x_feat, y_feat, fig_dir)
    plot_class_distribution(y, fig_dir)
    plot_feature_energy(x_feat, y_feat, fig_dir)
    plot_feature_mean_spectrum(x_feat, y_feat, fig_dir)
    plot_tsne_embedding(x_feat, y_feat, fig_dir)


    print(f"\nDataset figures saved to: {fig_dir}")
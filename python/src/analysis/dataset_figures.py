from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from python.src.dataio.loader import load_artifact
from python.src.preprocessing.dataset_builder import build_feature_tensor


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_time_domain_examples(X, y, out_dir: Path, n_classes=7):
    fig, axes = plt.subplots(n_classes, 1, figsize=(8, 10))

    for c in range(n_classes):
        idx = np.where(y == c)[0][0]
        axes[c].plot(X[:, idx])
        axes[c].set_title(f"Class {c}")
        axes[c].set_ylabel("Amplitude")

    axes[-1].set_xlabel("Sample index")

    plt.tight_layout()
    plt.savefig(out_dir / "time_domain_examples.png", dpi=300)
    plt.close()


def plot_stft_examples(X_feat, y, out_dir: Path, n_classes=7):
    fig, axes = plt.subplots(1, n_classes, figsize=(14, 3))

    for c in range(n_classes):
        idx = np.where(y == c)[0][0]
        spec = X_feat[idx,0]

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


def plot_feature_energy(X_feat, y, out_dir: Path):
    x_flat = X_feat.reshape(X_feat.shape[0], -1)
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


def plot_feature_mean_spectrum(X_feat, y, out_dir: Path):
    plt.figure(figsize=(8, 4))

    for c in np.unique(y):
        idx = y == c
        data = X_feat[idx, 0]
        mean_spec = data.mean(axis=(0,2))

        plt.plot(mean_spec, label=f"class {c}")

    plt.xlabel("Frequency bin")
    plt.ylabel("Mean magnitude")
    plt.legend()

    plt.title("Mean spectral signature")

    plt.savefig(out_dir / "mean_spectrum.png", dpi=300)
    plt.close()

def plot_tsne_embedding(X_feat, y, out_dir: Path):

    from sklearn.manifold import TSNE

    X_flat = X_feat.reshape(X_feat.shape[0], -1)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    X_emb = tsne.fit_transform(X_flat)

    plt.figure(figsize=(6,6))

    for c in np.unique(y):
        idx = y == c
        plt.scatter(
            X_emb[idx,0],
            X_emb[idx,1],
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
    fig_dir = project_root / "reports" / "figures" / f"dataset_seed{seed}_n{n_per_class}"

    _ensure_dir(fig_dir)

    #clean_file = dataset_dir / "clean" / f"clean_dataset_v1_seed{seed}.mat"
    train_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"

    #clean_artifact = load_artifact(str(clean_file))
    train_artifact = load_artifact(str(train_file))

    X_raw = train_artifact.X
    y = train_artifact.y

    X_feat, y_feat = build_feature_tensor(train_artifact)

    X_feat = X_feat.detach().cpu().numpy()
    y_feat = y_feat.detach().cpu().numpy()

    plot_time_domain_examples(X_raw, y, fig_dir)
    plot_stft_examples(X_feat, y_feat, fig_dir)
    plot_class_distribution(y, fig_dir)
    plot_feature_energy(X_feat, y_feat, fig_dir)
    plot_feature_mean_spectrum(X_feat, y_feat, fig_dir)
    plot_tsne_embedding(X_feat, y_feat, fig_dir)


    print(f"\nDataset figures saved to: {fig_dir}")
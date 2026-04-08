from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..dataio.loader import load_artifact
from ..preprocessing.dataset_builder import build_feature_tensor


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_time_domain_features(x, y, out_dir: Path, n_classes=10):
    time_dir = out_dir / "time_domain_plots"
    _ensure_dir(time_dir)

    y = np.asarray(y).reshape(-1)

    for c in range(n_classes):
        idx_candidates = np.where(y == c)[0]
        if len(idx_candidates) == 0:
            continue

        idx = idx_candidates[0]
        signal = x[:, idx]

        if np.iscomplexobj(signal):
            signal = np.abs(signal)

        plt.figure(figsize=(12, 4))
        plt.plot(signal, linewidth=0.5, color="#1f77b4")
        plt.title(f"Class {c} - Time Domain")
        plt.grid(True, alpha=0.3)
        plt.xlim(0, len(signal))
        plt.ylabel("Amplitude")
        plt.xlabel("Sample index")

        plt.tight_layout()
        plt.savefig(time_dir / f"class_{c}_time_domain.png", dpi=300)
        plt.close()


def plot_stft_features(x_feat, y, out_dir: Path, n_classes=10):
    stft_dir = out_dir / "stft_plots"
    _ensure_dir(stft_dir)

    y = np.asarray(y).reshape(-1)

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
    })

    for c in range(n_classes):
        idx_candidates = np.where(y == c)[0]
        if len(idx_candidates) == 0:
            continue

        idx = idx_candidates[0]

        # channel 0 is already log1p(|STFT|)
        spec_vis = x_feat[idx, 0]

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(
            spec_vis,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="bicubic",
        )

        ax.set_xlabel("Time Bin")
        ax.set_ylabel("Frequency Bin")
        ax.set_title(f"Class {c} - STFT Log-Magnitude", pad=20)

        divider = make_axes_locatable(ax)
        ax_time = divider.append_axes("top", 1.0, pad=0.1, sharex=ax)
        ax_freq = divider.append_axes("right", 1.0, pad=0.1, sharey=ax)

        ax_time.plot(np.mean(spec_vis, axis=0), color="#2c2c2c", linewidth=1.5)
        ax_freq.plot(
            np.mean(spec_vis, axis=1),
            np.arange(spec_vis.shape[0]),
            color="#2c2c2c",
            linewidth=1.5,
        )

        ax_time.tick_params(labelbottom=False)
        ax_freq.tick_params(labelleft=False)
        ax_time.spines["top"].set_visible(False)
        ax_time.spines["right"].set_visible(False)
        ax_freq.spines["top"].set_visible(False)
        ax_freq.spines["right"].set_visible(False)

        cbar = fig.colorbar(im, ax=ax_freq, fraction=0.05, pad=0.1)
        cbar.set_label("log1p(|STFT|)", rotation=270, labelpad=15)

        plt.savefig(stft_dir / f"class_{c}_stft.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    plt.rcdefaults()


def plot_class_distribution(y, out_dir: Path):
    y = np.asarray(y).reshape(-1)
    classes, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)

    plt.xlabel("Class ID")
    plt.ylabel("Number of samples")
    plt.title("Class distribution")

    plt.savefig(out_dir / "class_distribution.png", dpi=300)
    plt.close()


def plot_feature_energy(x_feat, y, out_dir: Path):
    y = np.asarray(y).reshape(-1)

    # use magnitude channel only
    mag = x_feat[:, 0]
    x_flat = mag.reshape(mag.shape[0], -1)
    energy = np.sum(x_flat * x_flat, axis=1)

    plt.figure(figsize=(6, 4))

    for c in np.unique(y):
        idx = y == c
        plt.hist(energy[idx], bins=40, alpha=0.5, label=f"class {c}")

    plt.xlabel("Magnitude-channel feature energy")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Feature energy distribution")

    plt.savefig(out_dir / "feature_energy_distribution.png", dpi=300)
    plt.close()


def plot_feature_mean_spectrum(x_feat, y, out_dir: Path):
    y = np.asarray(y).reshape(-1)

    plt.figure(figsize=(8, 4))

    for c in np.unique(y):
        idx = y == c
        data = x_feat[idx, 0]  # log-magnitude channel
        mean_spec = data.mean(axis=(0, 2))

        plt.plot(mean_spec, label=f"class {c}")

    plt.xlabel("Frequency bin")
    plt.ylabel("Mean log-magnitude")
    plt.legend()
    plt.title("Mean spectral signature")

    plt.savefig(out_dir / "mean_spectrum.png", dpi=300)
    plt.close()


def plot_tsne_embedding(x_feat, y, out_dir: Path):
    y = np.asarray(y).reshape(-1)

    x_flat = x_feat.reshape(x_feat.shape[0], -1)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    x_emb = tsne.fit_transform(x_flat)

    plt.figure(figsize=(6, 6))

    for c in np.unique(y):
        idx = y == c
        plt.scatter(
            x_emb[idx, 0],
            x_emb[idx, 1],
            s=6,
            label=f"class {c}",
        )

    plt.legend(markerscale=3)
    plt.title("t-SNE embedding of STFT features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.savefig(out_dir / "tsne_embedding.png", dpi=300)
    plt.close()


def generate_dataset_figures(
    seed: int,
    n_per_class: int,
    project_root: Path,
    spec_version: str = "v2",
):
    dataset_dir = project_root / "artifacts" / "datasets"
    fig_dir = project_root / "reports" / "figures" / f"dataset_seed{seed}_n{n_per_class}_{spec_version}"

    _ensure_dir(fig_dir)

    train_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"

    train_artifact = load_artifact(train_file, load_params=True)

    x_raw = train_artifact.X
    y = np.asarray(train_artifact.y).reshape(-1)

    x_feat,_, y_feat = build_feature_tensor(train_artifact)

    x_feat = x_feat.detach().cpu().numpy()
    y_feat = y_feat.detach().cpu().numpy().reshape(-1)

    plot_time_domain_features(x_raw, y, fig_dir)
    plot_stft_features(x_feat, y_feat, fig_dir)
    plot_class_distribution(y, fig_dir)
    plot_feature_energy(x_feat, y_feat, fig_dir)
    plot_feature_mean_spectrum(x_feat, y_feat, fig_dir)
    plot_tsne_embedding(x_feat, y_feat, fig_dir)

    print(f"\nDataset figures saved to: {fig_dir}")
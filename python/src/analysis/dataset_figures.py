from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ..dataio import load_artifact
from ..preprocessing import build_feature_tensor

# --- The Blue II Academic Palette ---
BLUE_II = {
    "dark": "#081d58",
    "navy": "#253494",
    "ocean": "#1d91c0",
    "sky": "#41b6c4",
    "electric": "#ffffd9",
    "gray": "#2c2c2c"
}


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_time_domain_features(x, y, out_dir: Path):
    time_dir = out_dir / "time_domain_plots"
    _ensure_dir(time_dir)

    y = np.asarray(y).reshape(-1)
    n_classes = sorted(np.unique(y))
    sns.set_theme(style="whitegrid")

    for c in n_classes:
        idx_candidates = np.where(y == c)[0]
        if len(idx_candidates) == 0:
            continue

        idx = idx_candidates[0]
        signal = x[:, idx]

        if np.iscomplexobj(signal):
            signal = np.abs(signal)

        plt.figure(figsize=(12, 4))
        plt.plot(signal, linewidth=0.8, color=BLUE_II["navy"])

        plt.title(f"Class {c} - Time Domain", color=BLUE_II["dark"], fontweight='bold')
        plt.xlim(0, len(signal))
        plt.ylabel("Amplitude")
        plt.xlabel("Sample Index")

        plt.tight_layout()
        plt.savefig(time_dir / f"class_{c}_time_domain.png", dpi=300)
        plt.close()


def plot_stft_features(
    x_feat: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
) -> None:
    """Thesis‑level STFT plots: smooth, clean, no clutter, high contrast."""
    stft_dir = out_dir / "stft_plots"
    _ensure_dir(stft_dir)

    y = np.asarray(y).reshape(-1)
    n_classes = sorted(np.unique(y))

    # Use a crisp white style without grid lines
    with plt.style.context("seaborn-v0_8-white"):
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        })

        for c in n_classes:
            idx_candidates = np.where(y == c)[0]
            if len(idx_candidates) == 0:
                continue

            idx = idx_candidates[0]
            spec_vis = x_feat[idx, 0]          # shape (F, T)

            # Robust contrast: clip extreme percentiles
            vmin, vmax = np.percentile(spec_vis, [5, 99.9])

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                spec_vis,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",      # smooth, no visible pixels
            )

            ax.set_xlabel("Time Bin")
            ax.set_ylabel("Frequency Bin")
            ax.set_title(
                f"Class {c} – STFT (log1p magnitude)",
                color=BLUE_II["dark"],
                fontweight="bold",
                pad=15,
            )

            # Colour bar directly on the right (mimics your original look)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label(
                "log1p(|STFT|)",
                rotation=270,
                labelpad=18,
                fontsize=10,
            )

            sns.despine(ax=ax, trim=True)
            plt.tight_layout()
            plt.savefig(
                stft_dir / f"class_{c}_stft.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)



def plot_feature_energy(x_feat, y, out_dir: Path):
    """Upgraded to a Ridge Plot (Joyplot) for overlapping distributions"""
    y = np.asarray(y).reshape(-1)

    mag = x_feat[:, 0]
    x_flat = mag.reshape(mag.shape[0], -1)
    energy = np.sum(x_flat * x_flat, axis=1)

    # Create a DataFrame for Seaborn
    df = pd.DataFrame({
        'Energy': energy,
        'Class': [f"Class {int(lbl)}" for lbl in y]
    })

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.color_palette("Blues_r", n_colors=14)[:10]
    g = sns.FacetGrid(df, row="Class", hue="Class", aspect=10, height=0.8, palette=pal)

    # Draw the densities
    g.map(sns.kdeplot, "Energy", bw_adjust=.5, clip_on=False, fill=True, alpha=0.8, linewidth=1.5)
    g.map(sns.kdeplot, "Energy", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # Add Class labels to the left of the plots
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=BLUE_II["dark"],
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "Energy")

    # Set overlap
    g.figure.subplots_adjust(hspace=-0.4)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    plt.suptitle("Feature Energy Distribution by Class", y=0.98, color=BLUE_II["dark"], fontweight='bold')

    plt.savefig(out_dir / "feature_energy_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_mean_spectrum(x_feat, y, out_dir: Path):
    y = np.asarray(y).reshape(-1)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("mako", n_colors=len(np.unique(y)))

    for i, c in enumerate(np.unique(y)):
        idx = y == c
        data = x_feat[idx, 0]
        mean_spec = data.mean(axis=(0, 2))

        plt.plot(mean_spec, label=f"Class {c}", color=colors[i], linewidth=1.5)

    plt.xlabel("Frequency Bin")
    plt.ylabel("Mean Log-Magnitude")

    # Put legend outside the plot so it doesn't block the lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.title("Mean Spectral Signature", color=BLUE_II["dark"], fontweight='bold')

    plt.tight_layout()
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

    sns.set_theme(style="white")
    plt.figure(figsize=(8, 8))

    # Use a highly distinct palette since we need to tell 10 classes apart in a scatter plot
    palette = sns.color_palette("tab10", n_colors=len(np.unique(y)))

    for i, c in enumerate(np.unique(y)):
        idx = y == c
        plt.scatter(
            x_emb[idx, 0],
            x_emb[idx, 1],
            s=15,
            alpha=0.7,
            color=palette[i],
            label=f"Class {c}",
            edgecolors='none'
        )

    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.title("t-SNE Embedding of STFT Features", color=BLUE_II["dark"], fontweight='bold')
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    sns.despine()
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_embedding.png", dpi=300)
    plt.close()


def plot_class_distribution(y, out_dir: Path):
    y = np.asarray(y).reshape(-1)
    classes, counts = np.unique(y, return_counts=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))

    sns.barplot(x=classes, y=counts, hue=classes, palette="Blues_d", legend=False)

    plt.xlabel("Class ID")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution", color=BLUE_II["dark"], fontweight='bold')
    sns.despine()

    plt.savefig(out_dir / "class_distribution.png", dpi=300)
    plt.close()


def generate_dataset_figures(
        seed: int,
        n_per_class: int,
        project_root: Path,
        spec_version: str = "v2",
):
    dataset_dir = project_root / "artifacts" / "datasets"
    fig_dir = project_root / "reports" / "figures" / f"dataset_seed{seed}_n{n_per_class}_{spec_version}"
    fig1_dir = project_root / "reports" / "figures" / f"unknown_seed{seed}_n{n_per_class}_{spec_version}"

    _ensure_dir(fig_dir)

    train_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
    unknown_file = dataset_dir / "unknown" / f"unknown_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"

    train_artifact = load_artifact(train_file, load_params=False)
    unknown_artifact = load_artifact(unknown_file, load_params=False)

    x_raw = train_artifact.X
    x_unk = unknown_artifact.X

    y = np.asarray(train_artifact.y).reshape(-1)
    y_unk = np.asarray(unknown_artifact.y).reshape(-1)

    x_feat, _, _, y_feat = build_feature_tensor(train_artifact)
    x_feat = x_feat[:, :1, :, :]
    x_unk_feat, _, _, y_unk_feat = build_feature_tensor(unknown_artifact)
    x_unk_feat = x_unk_feat[:, :1, :, :]

    x_feat = x_feat.detach().cpu().numpy()
    y_feat = y_feat.detach().cpu().numpy().reshape(-1)
    x_unk_feat = x_unk_feat.detach().cpu().numpy()
    y_unk_feat = y_unk_feat.detach().cpu().numpy().reshape(-1)


    print("Generating Time Domain plots...")
    plot_time_domain_features(x_raw, y, fig_dir)
    plot_time_domain_features(x_unk, y_unk, fig1_dir)
    print("Generating STFT plots...")
    plot_stft_features(x_feat, y_feat, fig_dir)
    plot_stft_features(x_unk_feat, y_unk_feat, fig1_dir)
    print("Generating Distribution plots...")
    plot_class_distribution(y, fig_dir)
    plot_class_distribution(y_unk, fig1_dir)
    print("Generating Ridge plots...")
    plot_feature_energy(x_feat, y_feat, fig_dir)
    plot_feature_energy(x_unk_feat, y_unk_feat, fig1_dir)
    print("Generating Mean Spectrum plots...")
    plot_feature_mean_spectrum(x_feat, y_feat, fig_dir)
    plot_feature_mean_spectrum(x_unk_feat, y_unk_feat, fig1_dir)
    print("Generating t-SNE plot...")
    plot_tsne_embedding(x_feat, y_feat, fig_dir)
    plot_tsne_embedding(x_unk_feat, y_unk_feat, fig1_dir)

    print(f"\nDataset figures saved to: {fig_dir}")
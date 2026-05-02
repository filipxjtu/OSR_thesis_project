import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison():
    # Input Data
    snr_values = np.arange(-14, 12, 2)

    # Replace these with your actual extracted values
    results = {
        "AsymmetricTriNet": [0.3828, 0.6444, 0.8678, 0.9440, 0.9708, 0.9846, 0.9912, 0.9950, 0.9960, 0.9974, 0.9980, 0.9974, 0.9980],
        #"ResNet-18": [0.21, 0.30, 0.48, 0.61, 0.72, 0.81, 0.88, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93],
        #"VGG-16": [0.18, 0.25, 0.40, 0.55, 0.68, 0.75, 0.82, 0.85, 0.86, 0.86, 0.87, 0.87, 0.87],
        #"DenseNet-121": [0.24, 0.35, 0.52, 0.66, 0.78, 0.86, 0.91, 0.94, 0.95, 0.95, 0.95, 0.96, 0.96]
    }

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']

    for (label, accs), marker in zip(results.items(), markers):
        # Convert to percentage
        plt.plot(snr_values, [a * 100 for a in accs], label=label,
                 marker=marker, linewidth=2, markersize=6)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Closed-Set Accuracy (%)", fontsize=12)
    plt.title("Model Performance Comparison across SNR Levels", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.xticks(snr_values)
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig("closed_set_snr_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_model_comparison()
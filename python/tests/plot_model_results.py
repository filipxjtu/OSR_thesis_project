import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison():
    # Input Data
    snr_values = np.arange(-14, 12, 2)

    # Replace these with your actual extracted values
    results = {
            "Unkown Recall": [0.3828, 0.4844, 0.5778, 0.6540, 0.7508, 0.7846, 0.7912, 0.8250, 0.8460, 0.8574, 0.8880, 0.8974, 0.8980],
            "AUROC": [0.6538, 0.6854, 0.75008, 0.8012, 0.8758, 0.8814, 0.8900, 0.9120, 0.9180, 0.9230, 0.925, 0.928, 0.9299],
            "FAR": [0.4400, 0.3516, 0.2334, 0.2186, 0.1916, 0.1734, 0.15586, 0.1382, 0.1046, 0.0834, 0.0602, 0.05192, 0.04104],

    }




    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'x', '*', 'p' ]

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
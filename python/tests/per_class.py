import matplotlib.pyplot as plt
import numpy as np

def plot_per_class_accuracy():
    # SNR values from -14 dB to +10 dB (2 dB steps)
    snr_values = np.arange(-14, 12, 2)

    # Per-class accuracy extracted from evaluation logs
    # Class order: 0→STJ, 1→MTJ, 2→LFMJ, 3→SFMJ, 4→PBNJ,
    #              5→FHJ, 6→OFDM, 7→PGPJ, 8→ISRJ, 9→DFTJ
    results = {
        "STJ":   [0.600, 0.886, 0.954, 0.984, 0.996, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        "MTJ":   [0.534, 0.736, 0.910, 0.988, 0.990, 0.996, 1.000, 0.998, 0.998, 1.000, 1.000, 1.000, 1.000],
        "LFMJ":  [0.340, 0.624, 0.856, 0.950, 0.986, 0.998, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        "SFMJ":  [0.244, 0.480, 0.830, 0.976, 0.996, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        "PBNJ":  [0.290, 0.492, 0.642, 0.768, 0.854, 0.922, 0.970, 0.982, 0.990, 0.996, 1.000, 1.000, 0.996],
        "FHJ":   [0.270, 0.478, 0.818, 0.964, 0.994, 0.998, 1.000, 1.000, 1.000, 0.998, 1.000, 1.000, 1.000],
        "OFDM":  [0.306, 0.448, 0.662, 0.812, 0.870, 0.946, 0.978, 0.976, 0.994, 0.998, 0.998, 0.998, 1.000],
        "PGPJ":  [0.360, 0.704, 0.932, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        "ISRJ":  [0.328, 0.594, 0.848, 0.956, 0.988, 0.996, 0.996, 0.996, 0.998, 1.000, 1.000, 1.000, 1.000],
        "DFTJ":  [0.262, 0.422, 0.708, 0.868, 0.952, 0.978, 0.988, 0.994, 0.994, 0.994, 0.994, 0.996, 1.000],
    }

    plt.figure(figsize=(12, 7))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for (label, accs), marker in zip(results.items(), markers):
        plt.plot(snr_values, [a * 100 for a in accs], label=label,
                 marker=marker, linewidth=2, markersize=7)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel("Per-Class Accuracy (%)", fontsize=11)
    #plt.title("Jamming Class Recognition Accuracy vs SNR", fontsize=15)
    plt.legend(loc="lower right", ncol=1, fontsize=10)
    plt.xticks(snr_values)
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig("per_class_snr_accuracy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_per_class_accuracy()
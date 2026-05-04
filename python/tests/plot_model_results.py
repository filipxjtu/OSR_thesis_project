import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison():
    # Input Data
    snr_values = np.arange(-14, 12, 2)

    # Replace these with your actual extracted values
    results = {
            "Full Model": [0.3828, 0.6444, 0.8678, 0.9440, 0.9708, 0.9846, 0.9912, 0.9950, 0.9960, 0.9974, 0.9980, 0.9974,
                       0.9980],
            "STFT only": [0.4238, 0.6154, 0.8008, 0.8812, 0.9158, 0.9414, 0.9600, 0.9690, 0.9830, 0.9900, 0.9926, 0.9928,
                      0.9940],
            "IQ only": [0.2400, 0.3516, 0.5334, 0.7186, 0.8516, 0.9234, 0.9586, 0.9782, 0.9846, 0.9834, 0.9902, 0.9892,
                    0.9904],
            "IF only": [0.1528, 0.1994, 0.2980, 0.4316, 0.5864, 0.7374, 0.8480, 0.9108, 0.9456, 0.9482, 0.9580, 0.9550,
                    0.9508],
            "STFT_IQ": [0.3914, 0.6418, 0.8516, 0.9364, 0.9716, 0.9882, 0.9922, 0.9962, 0.9982, 0.9982, 0.9980, 0.9982,
                    0.9990],
            "STFT_IF": [0.4108, 0.6456, 0.8320, 0.9174, 0.9516, 0.9736, 0.9824, 0.9852, 0.9910, 0.9926, 0.9942, 0.9922,
                    0.9934],
            "IQ_IF": [0.2268, 0.3410, 0.4960, 0.6848, 0.8302, 0.9258, 0.9616, 0.9814, 0.9908, 0.9942, 0.9960, 0.9930,
                  0.9972],
            "AsymmetricTriNet": [0.3828, 0.6444, 0.8678, 0.9440, 0.9708, 0.9846, 0.9912, 0.9950, 0.9960, 0.9974, 0.9980,
                             0.9974, 0.9980],
            "ResNet-18": [0.2919, 0.4307, 0.6438, 0.7961, 0.8782, 0.9281, 0.9588, 0.9661, 0.9762, 0.9832, 0.9863, 0.9883,
                      0.9930],
            "VGG-16": [0.213, 0.3525, 0.5540, 0.7155, 0.8268, 0.8975, 0.9382, 0.9585, 0.9686, 0.9786, 0.9827, 0.9857,
                   0.9873],
            "DenseNet-121": [0.2757, 0.4935, 0.7152, 0.8466, 0.9178, 0.9516, 0.9681, 0.9774, 0.9845, 0.9892, 0.9915, 0.9926,
                         0.9936],
            "Ablated STFT-branch": [0.4238, 0.6154, 0.8008, 0.8812, 0.9158, 0.9414, 0.9600, 0.9690, 0.9830, 0.9900, 0.9926,
                                0.9928, 0.9940],

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
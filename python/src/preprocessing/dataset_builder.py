from __future__ import annotations

import numpy as np
import torch

from .stft import compute_stft
from ..dataio.dataset_artifact import DatasetArtifact


def build_feature_tensor(
    artifact: DatasetArtifact,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts validated artifact into model-ready tensor.
    Output:      X: (Ns, 1, F, T)
                 y: (Ns,).
    """

    x_raw = artifact.X
    y = artifact.y.reshape(-1)
    Ns = x_raw.shape[1]

    features = []
    for i in range(Ns):
        signal = x_raw[:, i]
        s = compute_stft(signal)
        features.append(s)

    x_feat = np.stack(features, axis=0)  # (Ns, F, T)
    x_feat = x_feat[:, None, :, :]       # (Ns, 1, F, T)

    x_tensor = torch.tensor(x_feat, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return x_tensor, y_tensor
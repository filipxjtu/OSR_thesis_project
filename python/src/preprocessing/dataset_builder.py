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
    Output:
        X: (Ns, 1, F, T)
        y: (Ns,)
    """

    X_raw = artifact.X
    y = artifact.y.reshape(-1)

    Ns = X_raw.shape[1]

    features = []

    for i in range(Ns):
        signal = X_raw[:, i]
        S = compute_stft(signal, fs=int(artifact.meta["fs"]))
        features.append(S)

    X_feat = np.stack(features, axis=0)  # (Ns, F, T)
    X_feat = X_feat[:, None, :, :]       # add channel dim

    X_tensor = torch.tensor(X_feat, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor
from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import TensorDataset

from ..dataio import load_artifact
from ..preprocessing import build_feature_tensor, split_dataset


def load_osr_datasets(project_root: Path, seed: int, n_per_class: int, spec_version: str):

    dataset_dir = project_root / "artifacts" / "datasets"

    train_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
    eval_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_eval.mat"
    unknown_file = dataset_dir / "unknown" / f"unknown_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"

    train_artifact = load_artifact(str(train_file), load_params=False)
    eval_artifact = load_artifact(str(eval_file), load_params=False)
    unknown_artifact = load_artifact(str(unknown_file), load_params=False)

    x_stft_known, x_iq_known, y_known = build_feature_tensor(train_artifact)
    x_stft_eval, x_iq_eval, y_eval = build_feature_tensor(eval_artifact)
    x_stft_unknown, x_iq_unknown, _ = build_feature_tensor(unknown_artifact)

    y_unknown = torch.full((x_stft_unknown.shape[0],), -1, dtype=torch.long)
    assert y_unknown.shape[0] == x_iq_unknown.shape[0], "OSR-Dataloader alignment error."

    # 1. Split Known Train into Train/Val (80/20)
    train_set, val_set = split_dataset(x_stft_known, x_iq_known, y_known, train_ratio=0.8, seed=seed)

    # Extract directly from Subset indices for performance (avoids slow list comprehension)
    train_stft = train_set.dataset.tensors[0][train_set.indices]
    train_iq   = train_set.dataset.tensors[1][train_set.indices]
    train_y    = train_set.dataset.tensors[2][train_set.indices]

    val_stft = val_set.dataset.tensors[0][val_set.indices]
    val_iq   = val_set.dataset.tensors[1][val_set.indices]
    val_y    = val_set.dataset.tensors[2][val_set.indices]

    # 2. Split Unknowns (Total 4000 -> Train: 2000 | Val: 500 | Test: 1500)
    unk_train_stft, unk_train_iq, unk_train_y = x_stft_unknown[:2000], x_iq_unknown[:2000], y_unknown[:2000]
    unk_val_stft, unk_val_iq, unk_val_y = x_stft_unknown[2000:2500], x_iq_unknown[2000:2500], y_unknown[2000:2500]
    unk_test_stft, unk_test_iq, unk_test_y = x_stft_unknown[2500:], x_iq_unknown[2500:], y_unknown[2500:]

    # 3. Mixed Training Set (For Phase 2 Calibrator)
    stft_train_mixed = torch.cat([train_stft, unk_train_stft], dim=0)
    iq_train_mixed = torch.cat([train_iq, unk_train_iq], dim=0)
    y_train_mixed = torch.cat([train_y, unk_train_y], dim=0)

    return {
        "train": TensorDataset(stft_train_mixed, iq_train_mixed, y_train_mixed),
        "val_known": TensorDataset(val_stft, val_iq, val_y),
        "val_unknown": TensorDataset(unk_val_stft, unk_val_iq, unk_val_y),
        "test_known": TensorDataset(x_stft_eval, x_iq_eval, y_eval),
        "test_unknown": TensorDataset(unk_test_stft, unk_test_iq, unk_test_y),
    }
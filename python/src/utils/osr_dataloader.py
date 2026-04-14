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

    # Extract unknown features while preserving the original subclass labels
    x_stft_unknown, x_iq_unknown, y_unk_orig = build_feature_tensor(unknown_artifact)

    # Splitting Knowns (shuffled: 80% train, 20% val)
    train_set, val_set = split_dataset(x_stft_known, x_iq_known, y_known, train_ratio=0.8, seed=seed)

    train_stft = train_set.dataset.tensors[0][train_set.indices]
    train_iq = train_set.dataset.tensors[1][train_set.indices]
    train_y = train_set.dataset.tensors[2][train_set.indices]

    val_stft = val_set.dataset.tensors[0][val_set.indices]
    val_iq = val_set.dataset.tensors[1][val_set.indices]
    val_y = val_set.dataset.tensors[2][val_set.indices]

    # Splitting Unknowns by CLASS (Proxy for Train/Val, True for Test)
    unk_classes = torch.unique(y_unk_orig)
    num_unk_classes = len(unk_classes)

    # Dynamically split the jammer types in two
    proxy_classes = unk_classes[:num_unk_classes // 2]
    test_classes = unk_classes[num_unk_classes // 2:]

    # Create boolean masks to physically isolate the jammers
    proxy_mask = torch.isin(y_unk_orig, proxy_classes)
    test_mask = torch.isin(y_unk_orig, test_classes)

    # Isolate Proxy Unknowns (Jammer 1 & 2) and force labels to -1
    proxy_stft = x_stft_unknown[proxy_mask]
    proxy_iq = x_iq_unknown[proxy_mask]
    proxy_y = torch.full((proxy_stft.size(0),), -1, dtype=torch.long)

    # Split Proxy Unknowns into Train (80%) and Val (20%)
    unk_train_set, unk_val_set = split_dataset(proxy_stft, proxy_iq, proxy_y, train_ratio=0.8, seed=seed)

    unk_train_stft = unk_train_set.dataset.tensors[0][unk_train_set.indices]
    unk_train_iq = unk_train_set.dataset.tensors[1][unk_train_set.indices]
    unk_train_y = unk_train_set.dataset.tensors[2][unk_train_set.indices]

    unk_val_stft = unk_val_set.dataset.tensors[0][unk_val_set.indices]
    unk_val_iq = unk_val_set.dataset.tensors[1][unk_val_set.indices]
    unk_val_y = unk_val_set.dataset.tensors[2][unk_val_set.indices]

    # Isolate True Unknowns (Jammer 3 & 4) and force labels to -1
    unk_test_stft = x_stft_unknown[test_mask]
    unk_test_iq = x_iq_unknown[test_mask]
    unk_test_y = torch.full((unk_test_stft.size(0),), -1, dtype=torch.long)

    # 3. Mixed Training Set (For Phase 2 Calibrator)
    # The calibrator will ONLY learn from the Proxy jammers
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
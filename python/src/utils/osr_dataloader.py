from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from ..dataio import load_artifact
from ..preprocessing import build_feature_tensor


def load_osr_datasets(project_root: Path, seed: int, n_per_class: int, spec_version: str):
    dataset_dir = project_root / "artifacts" / "datasets"

    train_file   = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
    eval_file    = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_eval.mat"
    unknown_file = dataset_dir / "unknown"  / f"unknown_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"

    train_artifact   = load_artifact(str(train_file),   load_params=False)
    eval_artifact    = load_artifact(str(eval_file),    load_params=False)
    unknown_artifact = load_artifact(str(unknown_file), load_params=False)

    x_stft_known, x_iq_known, x_if_known, y_known = build_feature_tensor(train_artifact)
    x_stft_eval,  x_iq_eval,  x_if_eval,  y_eval  = build_feature_tensor(eval_artifact)
    x_stft_unk,   x_iq_unk,   x_if_unk,   y_unk_orig = build_feature_tensor(unknown_artifact)

    train_idx, val_idx = _stratified_split_indices(y_known, train_ratio=0.8, seed=seed)

    train_stft, train_iq, train_if, train_y = _gather(x_stft_known, x_iq_known, x_if_known, y_known, train_idx)
    val_stft,   val_iq,   val_if,   val_y   = _gather(x_stft_known, x_iq_known, x_if_known, y_known, val_idx)

    unk_classes = torch.unique(y_unk_orig)
    num_unk_classes = len(unk_classes)
    proxy_classes = unk_classes[: num_unk_classes // 2]
    test_classes  = unk_classes[num_unk_classes // 2 :]

    proxy_mask = torch.isin(y_unk_orig, proxy_classes)
    test_mask  = torch.isin(y_unk_orig, test_classes)

    proxy_stft, proxy_iq, proxy_if = x_stft_unk[proxy_mask], x_iq_unk[proxy_mask], x_if_unk[proxy_mask]
    proxy_y = torch.full((proxy_stft.size(0),), -1, dtype=torch.long)

    proxy_train_idx, proxy_val_idx = _flat_random_split_indices(
        n=proxy_stft.size(0), train_ratio=0.8, seed=seed
    )

    unk_train_stft, unk_train_iq, unk_train_if, unk_train_y = _gather(
        proxy_stft, proxy_iq, proxy_if, proxy_y, proxy_train_idx
    )
    unk_val_stft, unk_val_iq, unk_val_if, unk_val_y = _gather(
        proxy_stft, proxy_iq, proxy_if, proxy_y, proxy_val_idx
    )

    unk_test_stft = x_stft_unk[test_mask]
    unk_test_iq   = x_iq_unk[test_mask]
    unk_test_if   = x_if_unk[test_mask]
    unk_test_y    = torch.full((unk_test_stft.size(0),), -1, dtype=torch.long)

    stft_train_mixed = torch.cat([train_stft, unk_train_stft], dim=0)
    iq_train_mixed   = torch.cat([train_iq,   unk_train_iq],   dim=0)
    if_train_mixed   = torch.cat([train_if,   unk_train_if],   dim=0)
    y_train_mixed    = torch.cat([train_y,    unk_train_y],    dim=0)

    return {
        "train":         TensorDataset(stft_train_mixed, iq_train_mixed, if_train_mixed, y_train_mixed),
        "val_known":     TensorDataset(val_stft,         val_iq,         val_if,         val_y),
        "val_unknown":   TensorDataset(unk_val_stft,     unk_val_iq,     unk_val_if,     unk_val_y),
        "test_known":    TensorDataset(x_stft_eval,      x_iq_eval,      x_if_eval,      y_eval),
        "test_unknown":  TensorDataset(unk_test_stft,    unk_test_iq,    unk_test_if,    unk_test_y),
    }


def _stratified_split_indices(y: torch.Tensor, train_ratio: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = [], []
    for c in torch.unique(y):
        class_idx = torch.where(y == c)[0]
        perm = class_idx[torch.randperm(len(class_idx), generator=generator)]
        cut = int(train_ratio * len(class_idx))
        train_indices.append(perm[:cut])
        val_indices.append(perm[cut:])
    return torch.cat(train_indices), torch.cat(val_indices)


def _flat_random_split_indices(n: int, train_ratio: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    cut = int(train_ratio * n)
    return perm[:cut], perm[cut:]


def _gather(x_stft, x_iq, x_if, y, idx):
    return x_stft[idx], x_iq[idx], x_if[idx], y[idx]
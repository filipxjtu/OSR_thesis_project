from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import TensorDataset

from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor, split_dataset


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
    assert y_unknown.shape[0] == x_iq_unknown.shape[0], "OSR-Dataloader: error in batchsize of inputs."

    # Split known
    train_set, val_set = split_dataset(x_stft_known, x_iq_known, y_known, train_ratio=0.8, seed=seed)

    train_stft = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_iq = torch.stack([train_set[i][1] for i in range(len(train_set))])
    train_y = torch.tensor([train_set[i][2] for i in range(len(train_set))])


    val_stft = torch.stack([val_set[i][0] for i in range(len(val_set))])
    val_iq = torch.stack([val_set[i][1] for i in range(len(val_set))])
    val_y = torch.tensor([val_set[i][2] for i in range(len(val_set))])

    # OSR mix
    stft_train = torch.cat([train_stft, x_stft_unknown], dim=0)
    iq_train = torch.cat([train_iq, x_iq_unknown], dim=0)
    y_train = torch.cat([train_y, y_unknown], dim=0)

    stft_val = torch.cat([val_stft, x_stft_unknown[:500]], dim=0)
    iq_val = torch.cat([val_iq, x_iq_unknown[:500]], dim=0)
    y_val = torch.cat([val_y, y_unknown[:500]], dim=0)

    return {
        "train": TensorDataset(stft_train, iq_train, y_train),
        "val_known": TensorDataset(val_stft, val_iq, val_y),
        "val_osr": TensorDataset(stft_val, iq_val, y_val),
        "eval": TensorDataset(x_stft_eval, x_iq_eval, y_eval),
        "unknown": TensorDataset(x_stft_unknown, x_iq_unknown, y_unknown),
    }
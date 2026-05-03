from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor


def load_osr_datasets(project_root: Path, seed: int, n_per_class: int, spec_version: str):
    """
    Load OSR datasets with **separate proxy and test unknown files**.

    Files expected on disk:
      artifacts/datasets/impaired/impaired_dataset_{ver}_seed{s}_n{n}_train.mat
      artifacts/datasets/impaired/impaired_dataset_{ver}_seed{s}_n{n}_eval.mat
      artifacts/datasets/unknown/unknown_dataset_{ver}_seed{s}_n{n}_proxy.mat
      artifacts/datasets/unknown/unknown_dataset_{ver}_seed{s}_n{n}_test.mat

    Returned dict keys are unchanged from the previous single-file version, so
    callers (osr_trainer.py) keep working without modification:
        train, val_known, val_unknown, test_known, test_unknown

    Differences vs the old loader:
      - There is no class-id slicing of the unknowns. The proxy file is used
        in its entirety for proxy-training (with an internal 80/20 split into
        proxy-train and proxy-val for the calibrator). The test file is used
        in its entirety as held-out test_unknown.
      - This gives the user direct, MATLAB-side control over which signal
        families act as proxy unknowns vs which act as held-out test
        unknowns, and decouples the two.
    """
    dataset_dir = project_root / "artifacts" / "datasets"

    train_file     = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
    eval_file      = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_eval.mat"
    unk_proxy_file = dataset_dir / "unknown"  / f"unknown_dataset_{spec_version}_seed{seed}_n2000_proxy.mat"
    unk_test_file  = dataset_dir / "unknown"  / f"unknown_dataset_{spec_version}_seed{seed}_n2000_test.mat"

    for path, role in (
        (train_file,     "impaired train"),
        (eval_file,      "impaired eval"),
        (unk_proxy_file, "unknown proxy"),
        (unk_test_file,  "unknown test"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"OSR loader: missing {role} dataset at {path}"
            )

    # ---- Knowns ----
    train_artifact = load_artifact(str(train_file), load_params=False)
    eval_artifact  = load_artifact(str(eval_file),  load_params=False)

    x_stft_known, x_iq_known, x_if_known, y_known = build_feature_tensor(train_artifact)
    x_stft_eval,  x_iq_eval,  x_if_eval,  y_eval  = build_feature_tensor(eval_artifact)

    train_idx, val_idx = _stratified_split_indices(y_known, train_ratio=0.8, seed=seed)

    train_stft, train_iq, train_if, train_y = _gather(
        x_stft_known, x_iq_known, x_if_known, y_known, train_idx
    )
    val_stft, val_iq, val_if, val_y = _gather(
        x_stft_known, x_iq_known, x_if_known, y_known, val_idx
    )

    # ---- Unknowns: proxy file → 80/20 proxy-train / proxy-val ----
    proxy_artifact = load_artifact(str(unk_proxy_file), load_params=False)
    x_stft_pxy, x_iq_pxy, x_if_pxy, _y_pxy_orig = build_feature_tensor(proxy_artifact)
    proxy_y = torch.full((x_stft_pxy.size(0),), -1, dtype=torch.long)

    proxy_train_idx, proxy_val_idx = _flat_random_split_indices(
        n=x_stft_pxy.size(0), train_ratio=0.8, seed=seed
    )
    unk_train_stft, unk_train_iq, unk_train_if, unk_train_y = _gather(
        x_stft_pxy, x_iq_pxy, x_if_pxy, proxy_y, proxy_train_idx
    )
    unk_val_stft, unk_val_iq, unk_val_if, unk_val_y = _gather(
        x_stft_pxy, x_iq_pxy, x_if_pxy, proxy_y, proxy_val_idx
    )

    # ---- Unknowns: test file → 100% held-out test_unknown ----
    test_artifact = load_artifact(str(unk_test_file), load_params=False)
    x_stft_tst, x_iq_tst, x_if_tst, _y_tst_orig = build_feature_tensor(test_artifact)
    unk_test_y = torch.full((x_stft_tst.size(0),), -1, dtype=torch.long)

    # ---- Mixed train: known(train portion) + proxy(train portion) ----
    stft_train_mixed = torch.cat([train_stft, unk_train_stft], dim=0)
    iq_train_mixed   = torch.cat([train_iq,   unk_train_iq],   dim=0)
    if_train_mixed   = torch.cat([train_if,   unk_train_if],   dim=0)
    y_train_mixed    = torch.cat([train_y,    unk_train_y],    dim=0)

    print(
        f"[load_osr_datasets] proxy unknowns: "
        f"{x_stft_pxy.size(0)} samples (train {unk_train_stft.size(0)} / val {unk_val_stft.size(0)})"
    )
    print(
        f"[load_osr_datasets] test  unknowns: "
        f"{x_stft_tst.size(0)} samples (held out)"
    )

    return {
        "train":         TensorDataset(stft_train_mixed, iq_train_mixed, if_train_mixed, y_train_mixed),
        "val_known":     TensorDataset(val_stft,         val_iq,         val_if,         val_y),
        "val_unknown":   TensorDataset(unk_val_stft,     unk_val_iq,     unk_val_if,     unk_val_y),
        "test_known":    TensorDataset(x_stft_eval,      x_iq_eval,      x_if_eval,      y_eval),
        "test_unknown":  TensorDataset(x_stft_tst,       x_iq_tst,       x_if_tst,       unk_test_y),
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


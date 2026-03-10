from python.src.dataio import load_artifact
from python.src.preprocessing.dataset_builder import build_feature_tensor
from python.src.preprocessing.splitting import split_dataset

import torch

def debug_preprocessing(name, artifact):
    print(f"\n=== DEBUG: {name} ===")

    X, y = build_feature_tensor(artifact)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)

    # ----- Numeric sanity (Torch version) -----
    print("X min/max:", float(X.min()), float(X.max()))
    print("X mean/std:", float(X.mean()), float(X.std()))
    print("NaN in X:", torch.isnan(X).any().item())
    print("Inf in X:", torch.isinf(X).any().item())

    # ----- Class distribution -----
    unique, counts = torch.unique(y, return_counts=True)
    print("Class distribution:",
          dict(zip(unique.tolist(), counts.tolist())))

    # ----- Split test -----
    # ----- Split test -----
    train_set, val_set = split_dataset(X, y, train_ratio=0.8)

    print("Train size:", len(train_set))
    print("Val size:", len(val_set))
    print("Train + Val equals total:",
          len(train_set) + len(val_set) == X.shape[0])

    # Inspect one batch manually
    x_sample, y_sample = train_set[0]
    print("Sample shape:", x_sample.shape)
    print("Sample label:", y_sample)

    # Class distribution from subsets

    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    val_labels = torch.tensor([val_set[i][1] for i in range(len(val_set))])

    u_train, c_train = torch.unique(train_labels, return_counts=True)
    u_val, c_val = torch.unique(val_labels, return_counts=True)

    print("Train class dist:", dict(zip(u_train.tolist(), c_train.tolist())))
    print("Val class dist:", dict(zip(u_val.tolist(), c_val.tolist())))

###################################################################################################################

artifact  = load_artifact("clean_dataset_v1_seed10_n400.mat")
artifact1 = load_artifact("impaired_dataset_v1_seed10_n400_train.mat")
artifact2 = load_artifact("impaired_dataset_v1_seed10_n400_eval.mat")

# Run checks
debug_preprocessing("Impaired Train", artifact1)
debug_preprocessing("Impaired Eval", artifact2)
debug_preprocessing("Clean", artifact)



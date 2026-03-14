from __future__ import annotations

import sys
from pathlib import Path

import torch

from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor, split_dataset
from python.src.validation import run_validation_gate, ValidationError


def find_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")

def compute_snr_debug(name, clean, impaired):

    x_clean = torch.tensor(clean.X)
    x_imp = torch.tensor(impaired.X)

    # ensure same layout (N × Ns)
    assert x_clean.shape == x_imp.shape

    signal_power = torch.mean(x_clean ** 2, dim=0)
    noise_power = torch.mean((x_imp - x_clean) ** 2, dim=0)

    snr = 10 * torch.log10(signal_power / (noise_power + 1e-12))

    print(f"\nEstimated SNR statistics of {name}")
    print("-" * 25)
    print(f"Mean SNR: {snr.mean():.2f} dB")
    print(f"Min  SNR: {snr.min():.2f} dB")
    print(f"Max  SNR: {snr.max():.2f} dB")


def debug_preprocessing(name, artifact):
    print(f"\n DEBUG: {name}")
    print(f"-" * 35)

    X, y = build_feature_tensor(artifact)

    print(f"\nX shape: {X.shape}, \t X dtype: {X.dtype}")
    print(f"y shape: {y.shape}, \t y dtype: {y.dtype}")

    # ----- Numeric sanity (Torch version) -----
    print(f"\nX min/max: {X.min():.5f}, {float(X.max()):.5f}")
    print(f"X mean/std: {float(X.mean()):.5f}, {float(X.std()):.5}")
    print("NaN in X:", torch.isnan(X).any().item())
    print("Inf in X:z", torch.isinf(X).any().item())

    # ----- Class distribution -----
    unique, counts = torch.unique(y, return_counts=True)
    print("\nClass distribution:",
          dict(zip(unique.tolist(), counts.tolist())))

    # ----- Split test -----
    train_set, val_set = split_dataset(X, y, train_ratio=0.8)

    print(f"\nTrain size: {len(train_set)},\t Val size: {len(val_set)}")
    print("Train + Val equals total:", len(train_set) + len(val_set) == X.shape[0])

    # Inspect one batch manually
    x_sample, y_sample = train_set[0]
    print(f"\nSample shape: {x_sample.shape}, \t Sample label: {y_sample}")

    # Class distribution from subsets
    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    val_labels = torch.tensor([val_set[i][1] for i in range(len(val_set))])

    u_train, c_train = torch.unique(train_labels, return_counts=True)
    u_val, c_val = torch.unique(val_labels, return_counts=True)

    print("\nTrain class dist:", dict(zip(u_train.tolist(), c_train.tolist())))
    print("Val class dist:", dict(zip(u_val.tolist(), c_val.tolist())))


def run_debug_checks(clean_file, train_file, eval_file):

    print("\nrunning pre-validation debug check")
    print("-" * 35)

    try:
        # Load and debug each artifact
        artifact_clean = load_artifact(clean_file)
        artifact_train = load_artifact(train_file)
        artifact_eval = load_artifact(eval_file)


        # Run checks
        debug_preprocessing("Clean Dataset", artifact_clean)
        debug_preprocessing("Impaired Train Dataset", artifact_train)
        debug_preprocessing("Impaired Eval Dataset", artifact_eval)

        # Estimated SNR statistics
        compute_snr_debug("Train Data", artifact_clean, artifact_train)
        compute_snr_debug("Evaluation Data", artifact_clean, artifact_eval)



        print("\ndebug checks completed successfully")
        print("-" * 35 + "\n")
        return True

    except Exception as e:
        print(f"\nDEBUG CHECKS FAILED: {e}")
        return False


def main():

    #parameters user can change
    seeds = [27]
    n_per_class = 600
    spec_version = "v2"
    enable_feature_checks = True
    partial_features_check = False
    enable_repro_check = True
    repro_trials = 2

    for s in seeds:
        clean_file = f"clean_dataset_{spec_version}_seed{s}_n{n_per_class}.mat"
        train_file = f"impaired_dataset_{spec_version}_seed{s}_n{n_per_class}_train.mat"
        eval_file = f"impaired_dataset_{spec_version}_seed{s}_n{n_per_class}_eval.mat"

        project_root = find_project_root()

        report_name = f"validation_seed{s}_n{n_per_class}_{spec_version}.json"

        report_dir = project_root / "reports" / "validations"
        report_path = report_dir / report_name

        if not run_debug_checks(clean_file, train_file, eval_file):
            print("\nDebug checks FAILED - Abort Validation.")
            sys.exit(1)

        if report_path.exists():
            print(f"\nExisting validation report found at: {report_path}\n")
            print("DATASET VALIDATION ALREADY PASSED.\n")
        else:
            try:
                run_validation_gate(
                    clean_file=str(clean_file),
                    train_file=str(train_file),
                    eval_file=str(eval_file),
                    spec_version=spec_version,
                    n_classes=7,
                    report_name=report_name,
                    enable_feature_checks=enable_feature_checks,
                    partial_features_check=partial_features_check,
                    enable_repro_check=enable_repro_check,
                    repro_trial=repro_trials,
                )
                print("\nDATASET VALIDATION PASSED.\n")

            except ValidationError as e:
                print("\nDATASET VALIDATION FAILED")
                raise e


if __name__ == "__main__":
    main()
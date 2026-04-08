from __future__ import annotations

from pathlib import Path

import cProfile
import pstats

from python.src.dataio.loader import load_artifact
from python.src.validation.runner import validate_all, ValidationConfig
from python.src.validation.exceptions import ValidationError
from python.src.validation.types import Dataset, DatasetBundle


def find_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")


def main():
    # parameters
    seed = 32
    n_per_class = 1000
    spec_version = "v2"

    # file names
    clean_file = f"clean_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"
    train_file = f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
    eval_file = f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_eval.mat"

    unknown_file = f"unknown_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"
    clean_unk_file = f"clean_unk_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"

    # path
    project_root = find_project_root()
    base_path = project_root / "artifacts" / "datasets"

    clean_path = base_path / "clean" / clean_file
    train_path = base_path / "impaired" / train_file
    eval_path = base_path / "impaired" / eval_file
    unknown_path = base_path / "unknown" / unknown_file
    clean_unk_path = base_path / "unknown" / clean_unk_file

    # load artifacts
    artifact_clean = load_artifact(clean_path, load_params=True)
    artifact_train = load_artifact(train_path, load_params=True)
    artifact_eval = load_artifact(eval_path, load_params=True)
    artifact_unknown = load_artifact(unknown_path, load_params=True)
    artifact_clean_unk = load_artifact(clean_unk_path, load_params=True)

    # build bundle
    bundle = DatasetBundle(
        clean=Dataset(artifact_clean, "clean"),
        impaired_train=Dataset(artifact_train, "impaired_train"),
        impaired_eval=Dataset(artifact_eval, "impaired_eval"),
        unknown=Dataset(artifact_unknown, "unknown"),
        clean_unk=Dataset(artifact_clean_unk, "clean_unk"),
    )

    # config
    config = ValidationConfig(
        spec_version_expected=spec_version,
        n_classes_expected=10,
        enable_repro_check=True,
        repro_trials=2,
        partial_features_check=False,
    )

    # run validation
    try:
        _ = validate_all(bundle, config)

        print("\nDATASET VALIDATION PASSED.\n\n")

    except ValidationError as e:
        print("\nDATASET VALIDATION FAILED\n\n")

        if hasattr(e, "args") and len(e.args) > 0:
            failures = e.args[0]
            for f in failures:
                print(f"- {f.check_id}: {f.message}")

        raise e


if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # save profile
    profiler.dump_stats("validation_profile.prof")

    stats = pstats.Stats(profiler)

    print("\n" + "=" * 80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)
    stats.sort_stats("cumulative").print_stats(30)

    print("\n" + "=" * 80)
    print("TOP 20 FUNCTIONS BY TOTAL TIME")
    print("=" * 80)
    stats.sort_stats("time").print_stats(20)

    print("\n" + "=" * 80)
    print("TOP 20 FUNCTIONS BY NUMBER OF CALLS")
    print("=" * 80)
    stats.sort_stats("calls").print_stats(20)
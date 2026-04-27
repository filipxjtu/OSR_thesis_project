from __future__ import annotations

from pathlib import Path

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


    # User parameters
    seeds = [38, 55]
    n_per_class = 2500
    spec_version = "v2"

    project_root = find_project_root()
    file_path = project_root / "artifacts" / "datasets"

    for seed in seeds:

        clean_file = f"clean_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"
        train_file = f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
        eval_file = f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_eval.mat"

        unknown_file = f"unknown_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"
        clean_unk_file = f"clean_unk_dataset_{spec_version}_seed{seed}_n{n_per_class}.mat"

        clean_path = file_path / "clean" / clean_file
        train_path = file_path / "impaired" / train_file
        eval_path = file_path / "impaired" / eval_file

        unknown_path = file_path / "unknown" / unknown_file
        clean_unk_path = file_path / "unknown" / clean_unk_file


        # Report path
        report_name = f"validation_seed{seed}_n{n_per_class}_{spec_version}.json"
        report_dir = project_root / "reports" / "validations"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / report_name

        if report_path.exists():
            print(f"\nValidation file already exists.")
            print(f"file found at: {report_path}\n")
            continue

        print(f"\nValidating dataset of seed {seed}...")
        print("-"*40)

        # Load artifacts (explicit)
        artifact_clean = load_artifact(clean_path, load_params=True)
        artifact_train = load_artifact(train_path, load_params=True)
        artifact_eval = load_artifact(eval_path, load_params=True)

        artifact_unknown = load_artifact(unknown_path, load_params=True)
        artifact_clean_unk = load_artifact(clean_unk_path, load_params=True)


        # Build bundle (correct)
        bundle = DatasetBundle(
            clean=Dataset(artifact_clean, "clean"),
            impaired_train=Dataset(artifact_train, "impaired_train"),
            impaired_eval=Dataset(artifact_eval, "impaired_eval"),
            unknown=Dataset(artifact_unknown, "unknown") ,
            clean_unk=Dataset(artifact_clean_unk, "clean_unk"),
        )

        # Validation config
        config = ValidationConfig(
            spec_version_expected=spec_version,
            n_classes_expected=10,
            enable_repro_check=True,
            repro_trials=2,
            partial_features_check=False,
        )

        # Run validation
        try:
            summary = validate_all(bundle, config)

            # PASS - save to file
            summary.save_json(report_path)

            print("\nDATASET VALIDATION PASSED.")
            print(f"Report saved to: {report_path}\n")

        except ValidationError as e:

            print("\nDATASET VALIDATION FAILED\n")

            # Print full collected failures
            if hasattr(e, "args") and len(e.args) > 0:
                failures = e.args[0]
                for f in failures:
                    print(f"- {f.check_id}: {f.message}")

            raise e

if __name__ == "__main__":
    main()
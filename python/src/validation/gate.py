from __future__ import annotations

from pathlib import Path

from ..dataio import load_artifact
from ..preprocessing import build_feature_tensor

from .runner import validate_all, ValidationConfig
from .types import DatasetBundle, DatasetView


# adapter
class ArtifactAdapter(DatasetView):
    def __init__(self, name: str, artifact):
        self._name = name
        self._artifact = artifact
        self._X_feat, self._y_feat = build_feature_tensor(artifact)

    @property
    def name(self) -> str:
        return self._name

    def x_time(self):
        return self._artifact.X

    def y(self):
        return self._artifact.y.reshape(-1).astype("int64")

    def x_feat(self):
        return self._X_feat

    def meta(self):
        return self._artifact.meta

# validation gate
def run_validation_gate(
    *,
    clean_file: str,
    train_file: str,
    eval_file: str,
    spec_version: str,
    n_classes: int,
    report_name: str,
    enable_feature_checks: bool,
    partial_features_check: bool,
    enable_repro_check: bool,
    repro_trial,
) -> None:
    """
    Executes full dataset validation and raise ValidationError if failed.
    Saves JSON summary if validation PASS.
    """

    clean_artifact = load_artifact(clean_file)
    train_artifact = load_artifact(train_file)
    eval_artifact  = load_artifact(eval_file)

    bundle = DatasetBundle(
        clean=ArtifactAdapter("clean", clean_artifact),
        impaired_train=ArtifactAdapter("impaired_train", train_artifact),
        impaired_eval=ArtifactAdapter("impaired_eval", eval_artifact),
    )

    config = ValidationConfig(
        spec_version_expected=spec_version,
        n_classes_expected=n_classes,
        enable_feature_checks=enable_feature_checks,
        partial_features_check=partial_features_check,
        enable_repro_check=enable_repro_check,
        repro_trials=repro_trial,
    )

    summary = validate_all(
        bundle=bundle,
        config=config,
    )

    # Save report
    project_root = Path(__file__).resolve().parents[3]
    report_dir = project_root / "reports" / "validations"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / report_name
    summary.save_json(report_path)
    print(f"\nValidation report saved to: {report_path}")
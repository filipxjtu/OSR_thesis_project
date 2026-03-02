from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np

from python.src.dataio import load_artifact
from python.src.preprocessing.dataset_builder import build_feature_tensor
from python.src.preprocessing.splitting import split_dataset

from python.src.validation.runner import validate_all, ValidationConfig
from python.src.validation.types import DatasetBundle, DatasetView
from python.src.validation.exceptions import ValidationError


# ==========================================================
# Adapter: Wrap DatasetArtifact into DatasetView interface
# ==========================================================

class ArtifactAdapter(DatasetView):
    def __init__(self, name: str, artifact):
        self._name = name
        self._artifact = artifact

        # Build feature tensor once (used in feature validation)
        self._X_feat, self._y_feat = build_feature_tensor(artifact)

    @property
    def name(self) -> str:
        return self._name

    def x_time(self):
        # Raw time-domain matrix
        return self._artifact.X

    def y(self):
        return self._artifact.y.reshape(-1).astype("int64")

    def x_feat(self):
        return self._X_feat

    def meta(self):
        return self._artifact.meta


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    # ------------------------------------------------------
    # Load artifacts (uses your real loader)
    # ------------------------------------------------------
    clean_artifact = load_artifact("clean_dataset_v1_seed49.mat")
    train_artifact = load_artifact("impaired_dataset_v1_seed49_train.mat")
    eval_artifact = load_artifact("impaired_dataset_v1_seed49_eval.mat")

    # ------------------------------------------------------
    # Wrap into DatasetBundle
    # ------------------------------------------------------
    bundle = DatasetBundle(
        clean=ArtifactAdapter("clean", clean_artifact),
        impaired_train=ArtifactAdapter("impaired_train", train_artifact),
        impaired_eval=ArtifactAdapter("impaired_eval", eval_artifact),
    )

    # ------------------------------------------------------
    # Configure validation
    # ------------------------------------------------------
    config = ValidationConfig(
        spec_version_expected="v1",
        n_classes_expected=7,
        enable_feature_checks=True,
        enable_repro_check=True,
        repro_trials=2,
        min_effect_size_time_train=0.05,
        min_effect_size_time_eval=0.05,
        min_effect_size_freq_train=0.05,
        min_effect_size_freq_eval=0.05,
        min_effect_size_train_vs_eval_time=0.02,
        min_effect_size_train_vs_eval_freq=0.02,
    )

    # ------------------------------------------------------
    # Run validation
    # ------------------------------------------------------
    try:
        summary = validate_all(
            bundle=bundle,
            config=config,
            loader_for_repro=lambda: DatasetBundle(
                clean=ArtifactAdapter("clean", load_artifact("clean_dataset_v1_seed8.mat")),
                impaired_train=ArtifactAdapter("impaired_train", load_artifact("impaired_dataset_v1_seed8_train.mat")),
                impaired_eval=ArtifactAdapter("impaired_eval", load_artifact("impaired_dataset_v1_seed8_eval.mat")),
            ),
        )
    except ValidationError as e:
        print("\n❌ DATASET VALIDATION FAILED")
        print(e)
        sys.exit(1)

    # ------------------------------------------------------
    # Save validation report
    # ------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports" / "statistical"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "validation_seed49.json"
    summary.save_json(report_path)

    print("\n✅ DATASET VALIDATION PASSED")
    print(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()


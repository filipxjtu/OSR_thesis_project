from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .checks import Thresholds
from .exceptions import ValidationError, FailedCheck
from .gate import run_validation_gate
from .repro import ReproConfig
from .summary import ValidationSummary
from .types import DatasetBundle

VALIDATOR_VERSION = "validation_v2"


@dataclass(frozen=True)
class ValidationConfig:
    spec_version_expected: str
    n_classes_expected: int

    fs_hz: float | None = None

    # Time-domain
    mean_abs_max: float = 1.0
    std_min: float = 1e-3
    peak_to_rms_max: float = 50.0

    # Frequency-domain
    dc_ratio_max: float = 0.15
    flatness_min: float = 1e-4
    centroid_min_hz: float = 1.0
    centroid_max_hz_ratio: float = 0.49

    # Known-domain separation
    min_effect_size_freq_train: float = 0.20
    min_effect_size_freq_eval: float = 0.20
    min_effect_size_train_vs_eval_freq: float = 0.03

    # Phase-domain
    phase_variance_min: float = 1e-4
    cos_sin_unit_error_max: float = 1e-6
    phase_uniformity_min: float = 1e-3

    # Unknown-domain separation
    min_effect_size_unknown_vs_clean: float = 0.05
    min_effect_size_unknown_vs_train: float = 0.05
    min_effect_size_unknown_vs_eval: float = 0.03
    min_effect_size_unknown_vs_clean_unk: float = 0.02

    # Reproducibility
    enable_repro_check: bool = True
    repro_trials: int = 2

    # Other runtime options
    partial_features_check: bool = False


def _thresholds_from_config(c: ValidationConfig) -> Thresholds:
    return Thresholds(
        mean_abs_max=c.mean_abs_max,
        std_min=c.std_min,
        peak_to_rms_max=c.peak_to_rms_max,
        dc_ratio_max=c.dc_ratio_max,
        flatness_min=c.flatness_min,
        centroid_min_hz=c.centroid_min_hz,
        centroid_max_hz_ratio=c.centroid_max_hz_ratio,
        min_effect_size_freq_train=c.min_effect_size_freq_train,
        min_effect_size_freq_eval=c.min_effect_size_freq_eval,
        min_effect_size_train_vs_eval_freq=c.min_effect_size_train_vs_eval_freq,
        phase_variance_min=c.phase_variance_min,
        cos_sin_unit_error_max=c.cos_sin_unit_error_max,
        phase_uniformity_min=c.phase_uniformity_min,
        min_effect_size_unknown_vs_clean=c.min_effect_size_unknown_vs_clean,
        min_effect_size_unknown_vs_train=c.min_effect_size_unknown_vs_train,
        min_effect_size_unknown_vs_eval=c.min_effect_size_unknown_vs_eval,
        min_effect_size_unknown_vs_clean_unk=c.min_effect_size_unknown_vs_clean_unk,
    )


def validate_all(
    bundle: DatasetBundle,
    config: ValidationConfig,
    *,
    loader_for_repro: Callable[[], DatasetBundle] | None = None,
) -> ValidationSummary:
    """
    Executes the full validation pipeline and returns ValidationSummary.

    Runner owns the final PASS/FAIL decision.
    Gate only aggregates.
    """

    meta = bundle.clean.meta
    spec_v = str(meta.get("spec_version", ""))

    if spec_v != config.spec_version_expected:
        raise ValidationError([
            FailedCheck(
                check_id="C000.spec_version_expected",
                message="Spec version mismatch",
                details={
                    "expected": config.spec_version_expected,
                    "got": spec_v,
                },
            )
        ])

    fs_hz = float(config.fs_hz if config.fs_hz is not None else meta.get("fs"))

    if not (fs_hz > 0):
        raise ValidationError([
            FailedCheck(
                check_id="C000.fs_hz_present",
                message="fs_hz missing/invalid",
                details={"fs_hz": meta.get("fs", None)},
            )
        ])

    th = _thresholds_from_config(config)

    repro_config = None
    repro_loader = None

    if config.enable_repro_check and loader_for_repro is not None:
        repro_config = ReproConfig(
            trials=config.repro_trials,
            require_identical_digest=True,
        )
        repro_loader = loader_for_repro

    summary = run_validation_gate(
        bundle=bundle,
        fs_hz=fs_hz,
        n_classes=config.n_classes_expected,
        thresholds=th,
        repro_loader=repro_loader,
        repro_config=repro_config,
        partial_features_check=config.partial_features_check,
    )

    # keep validator version centralized here
    summary.validator_version = VALIDATOR_VERSION

    if summary.status == "FAIL":
        raise ValidationError([
            FailedCheck(
                check_id=cid,
                message="Validation check failed",
                details={"summary_ref": cid},
            )
            for cid in summary.checks_failed
        ])

    return summary
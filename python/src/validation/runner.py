from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .checks import (
    Thresholds,
    check_no_nan_inf,
    check_time_domain_stats,
    check_freq_domain_stats,
    check_class_balance,
    check_cross_mode_separation,
)
from .exceptions import ValidationError, FailedCheck
from .features import FeatureThresholds, feature_checks
from .repro import ReproConfig, check_reproducibility
from .summary import ValidationSummary
from .types import DatasetBundle

VALIDATOR_VERSION = "validation_v2"


@dataclass(frozen=True)
class ValidationConfig:
    spec_version_expected: str
    n_classes_expected: int

    fs_hz: float | None = None

    # Time
    mean_abs_max: float = 0.05
    std_min: float = 1e-3
    peak_to_rms_max: float = 50.0

    # Freq
    dc_ratio_max: float = 0.10
    flatness_min: float = 1e-4
    centroid_min_hz: float = 1.0
    centroid_max_hz_ratio: float = 0.49

    # Balance
    class_balance_tol: float = 0.00

    # Separation thresholds (mode-specific)
    min_effect_size_time_train: float = 0.20
    min_effect_size_time_eval: float = 0.20
    min_effect_size_freq_train: float = 0.20
    min_effect_size_freq_eval: float = 0.20

    # Train vs eval must differ
    min_effect_size_train_vs_eval_time: float = 0.10
    min_effect_size_train_vs_eval_freq: float = 0.10

    # Feature thresholds
    enable_feature_checks: bool = True
    feat_std_min: float = 1e-6
    feat_energy_min: float = 1e-8
    min_effect_size_feat_train: float = 0.20
    min_effect_size_feat_eval: float = 0.20
    min_effect_size_feat_train_vs_eval: float = 0.10

    # Reproducibility
    enable_repro_check: bool = True
    repro_trials: int = 2


def _thresholds_from_config(c: ValidationConfig) -> Thresholds:
    return Thresholds(
        mean_abs_max=c.mean_abs_max,
        std_min=c.std_min,
        peak_to_rms_max=c.peak_to_rms_max,
        dc_ratio_max=c.dc_ratio_max,
        flatness_min=c.flatness_min,
        centroid_min_hz=c.centroid_min_hz,
        centroid_max_hz_ratio=c.centroid_max_hz_ratio,
        class_balance_tol=c.class_balance_tol,
        min_effect_size_freq_train=c.min_effect_size_freq_train,
        min_effect_size_freq_eval=c.min_effect_size_freq_eval,
        min_effect_size_train_vs_eval_freq=c.min_effect_size_train_vs_eval_freq,
    )


def validate_all(
    bundle: DatasetBundle,
    config: ValidationConfig,
    *,
    loader_for_repro: Callable[[], DatasetBundle] | None = None,
) -> ValidationSummary:
    th = _thresholds_from_config(config)

    meta = bundle.clean.meta()
    spec_v = str(meta.get("spec_version", ""))
    if spec_v != config.spec_version_expected:
        raise ValidationError(
            [FailedCheck(
                check_id="C000.spec_version_expected",
                message="Spec version mismatch",
                details={"expected": config.spec_version_expected, "got": spec_v},
            )]
        )

    fs_hz = float(config.fs_hz if config.fs_hz is not None else meta.get("fs"))
    if not (fs_hz > 0):
        raise ValidationError(
            [FailedCheck(
                check_id="C000.fs_hz_present",
                message="fs_hz missing/invalid in metadata and not provided in config",
                details={"fs_hz": meta.get("fs_hz", None)},
            )]
        )

    summary = ValidationSummary(validator_version=VALIDATOR_VERSION)
    summary.thresholds = {
        "time.mean_abs_max": th.mean_abs_max,
        "time.std_min": th.std_min,
        "time.peak_to_rms_max": th.peak_to_rms_max,
        "freq.dc_ratio_max": th.dc_ratio_max,
        "freq.flatness_min": th.flatness_min,
        "freq.centroid_min_hz": th.centroid_min_hz,
        "freq.centroid_max_hz_ratio": th.centroid_max_hz_ratio,
        "class.balance_tol": th.class_balance_tol,
        "sep.min_effect_size_freq_train": th.min_effect_size_freq_train,
        "sep.min_effect_size_freq_eval": th.min_effect_size_freq_eval,
        "sep.min_effect_size_train_vs_eval_freq": th.min_effect_size_train_vs_eval_freq,
    }

    failures: list[FailedCheck] = []

    failures += check_no_nan_inf(bundle)

    f, m = check_time_domain_stats(bundle, th)
    failures += f
    summary.metrics["time_domain"] = m

    f, m = check_freq_domain_stats(bundle, fs_hz=fs_hz, th=th)
    failures += f
    summary.metrics["freq_domain"] = m

    f, m = check_class_balance(bundle, n_classes=config.n_classes_expected, th=th)
    failures += f
    summary.metrics["class_balance"] = m

    f, m = check_cross_mode_separation(bundle, fs_hz=fs_hz, th=th)
    failures += f
    summary.metrics["cross_mode"] = m

    # Feature-domain checks (NEW)
    if config.enable_feature_checks:
        fth = FeatureThresholds(
            feat_std_min=config.feat_std_min,
            feat_energy_min=config.feat_energy_min,
            min_effect_size_feat_train=config.min_effect_size_feat_train,
            min_effect_size_feat_eval=config.min_effect_size_feat_eval,
            min_effect_size_feat_train_vs_eval=config.min_effect_size_feat_train_vs_eval,
        )
        f, m = feature_checks(bundle, fth)
        failures += f
        summary.metrics["feature_domain"] = m
    else:
        summary.metrics["feature_domain"] = {"skipped": True, "reason": "disabled by config"}

    # Reproducibility (NEW per-mode + bundle digest)
    if config.enable_repro_check:
        if loader_for_repro is None:
            summary.notes.append("Reproducibility check skipped: loader_for_repro not provided.")
        else:
            rc = ReproConfig(trials=config.repro_trials, require_identical_digest=True)
            f, m = check_reproducibility(
                loader=loader_for_repro,
                fs_hz=fs_hz,
                n_classes=config.n_classes_expected,
                th=th,
                rc=rc,
            )
            failures += f
            summary.metrics["reproducibility"] = m

    if failures:
        summary.status = "FAIL"
        summary.checks_failed = [x.check_id for x in failures]
        raise ValidationError(failures)

    summary.status = "PASS"
    summary.checks_passed = [
        "C001.no_nan_inf",
        "C010.time_mean_near_zero",
        "C011.time_std_not_collapsed",
        "C012.time_peak_to_rms_reasonable",
        "C020.freq_dc_ratio_small",
        "C021.freq_flatness_not_degenerate",
        "C022.freq_centroid_sane_low",
        "C023.freq_centroid_sane_high",
        "C030.class_balance_exact" if th.class_balance_tol == 0.0 else "C031.class_balance_tolerance",
        "C040.crossmode_time_clean_vs_train",
        "C041.crossmode_time_clean_vs_eval",
        "C042.crossmode_freq_clean_vs_train",
        "C043.crossmode_freq_clean_vs_eval",
        "C044.crossmode_time_train_vs_eval",
        "C045.crossmode_freq_train_vs_eval",
        # Feature checks might be skipped (so don't claim pass unconditionally)
    ]
    return summary
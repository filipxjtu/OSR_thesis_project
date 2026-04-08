from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import FailedCheck
from .stats import (
    time_domain_stats,
    freq_domain_stats,
    phase_domain_stats,
    effect_size_delta,
)
from .types import DatasetBundle


@dataclass(frozen=True)
class Thresholds:
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
    min_effect_size_unknown_vs_eval: float = 0.05
    min_effect_size_unknown_vs_clean_unk: float = 0.02


def _require(cond: bool, check_id: str, message: str, details: dict[str, Any]) -> list[FailedCheck]:
    return [] if cond else [FailedCheck(check_id=check_id, message=message, details=details)]


def _samples_first(x: np.ndarray) -> np.ndarray:
    """
    Validation layer uses samples-first shape: (Ns, N).
    Loader returns (N, Ns), so validator datasets should be transposed before reaching stats.
    This helper only enforces rank and returns input unchanged.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array with shape (Ns, N), got shape {x.shape}")
    return x


def check_no_nan_inf(bundle: DatasetBundle) -> list[FailedCheck]:
    fails: list[FailedCheck] = []

    for ds in bundle.all_datasets():
        x = np.asarray(ds.X)
        bad = ~np.isfinite(x)

        fails += _require(
            not bool(np.any(bad)),
            "C001.no_nan_inf_time",
            f"{ds.name}: found NaN/Inf in time-domain samples",
            {
                "dataset": ds.name,
                "shape": list(x.shape),
                "bad_count": int(np.sum(bad)),
            },
        )

    return fails


def check_time_domain_stats(bundle: DatasetBundle, th: Thresholds) -> tuple[list[FailedCheck], dict[str, Any]]:
    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    for ds in bundle.all_datasets():
        x = _samples_first(np.asarray(ds.X).T)
        s = time_domain_stats(x)

        metrics[ds.name] = {
            "mean": s.mean,
            "std": s.std,
            "min": s.min,
            "max": s.max,
            "rms": s.rms,
            "peak_to_rms": s.peak_to_rms,
            "skewness": s.skewness,
            "kurtosis_excess": s.kurtosis_excess,
        }

        fails += _require(
            abs(s.mean) <= th.mean_abs_max,
            "C010.time_mean_near_zero",
            f"{ds.name}: |mean| too large",
            {
                "dataset": ds.name,
                "mean": s.mean,
                "threshold": th.mean_abs_max,
            },
        )

        fails += _require(
            s.std >= th.std_min,
            "C011.time_std_not_collapsed",
            f"{ds.name}: std too small (collapse suspected)",
            {
                "dataset": ds.name,
                "std": s.std,
                "threshold": th.std_min,
            },
        )

        fails += _require(
            s.peak_to_rms <= th.peak_to_rms_max,
            "C012.time_peak_to_rms_reasonable",
            f"{ds.name}: peak_to_rms too large (spikes / clipping suspected)",
            {
                "dataset": ds.name,
                "peak_to_rms": s.peak_to_rms,
                "threshold": th.peak_to_rms_max,
            },
        )

    return fails, metrics


def check_freq_domain_stats(bundle: DatasetBundle, fs_hz: float, th: Thresholds) -> tuple[list[FailedCheck], dict[str, Any]]:
    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    for ds in bundle.all_datasets():
        x = _samples_first(np.asarray(ds.X).T)
        s = freq_domain_stats(x, fs_hz=fs_hz)

        metrics[ds.name] = {
            "dc_ratio": s.dc_ratio,
            "spectral_centroid_hz": s.spectral_centroid,
            "spectral_bandwidth_hz": s.spectral_bandwidth,
            "spectral_flatness": s.spectral_flatness,
            "rolloff_95_hz": s.rolloff_95,
        }

        fails += _require(
            s.dc_ratio <= th.dc_ratio_max,
            "C020.freq_dc_ratio_small",
            f"{ds.name}: DC ratio too large",
            {
                "dataset": ds.name,
                "dc_ratio": s.dc_ratio,
                "threshold": th.dc_ratio_max,
            },
        )

        fails += _require(
            s.spectral_flatness >= th.flatness_min,
            "C021.freq_flatness_not_degenerate",
            f"{ds.name}: spectral flatness too small (degenerate spectrum suspected)",
            {
                "dataset": ds.name,
                "flatness": s.spectral_flatness,
                "threshold": th.flatness_min,
            },
        )

        fails += _require(
            s.spectral_centroid >= th.centroid_min_hz,
            "C022.freq_centroid_sane_low",
            f"{ds.name}: spectral centroid too low",
            {
                "dataset": ds.name,
                "centroid_hz": s.spectral_centroid,
                "threshold": th.centroid_min_hz,
            },
        )

        fails += _require(
            s.spectral_centroid <= th.centroid_max_hz_ratio * fs_hz,
            "C023.freq_centroid_sane_high",
            f"{ds.name}: spectral centroid too high relative to fs",
            {
                "dataset": ds.name,
                "centroid_hz": s.spectral_centroid,
                "fs_hz": fs_hz,
                "threshold_ratio": th.centroid_max_hz_ratio,
            },
        )

    return fails, metrics


def check_phase_domain_stats(bundle: DatasetBundle, th: Thresholds) -> tuple[list[FailedCheck], dict[str, Any]]:
    """
    Phase checks are meaningful only for complex IQ.
    For real-valued signals, stats.phase_domain_stats returns safe defaults.
    """
    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    for ds in bundle.all_datasets():
        x = _samples_first(np.asarray(ds.X).T)
        s = phase_domain_stats(x)

        metrics[ds.name] = {
            "phase_mean": s.phase_mean,
            "phase_std": s.phase_std,
            "phase_variance": s.phase_variance,
            "cos_sin_unit_error": s.cos_sin_unit_error,
            "phase_uniformity": s.phase_uniformity,
        }

        fails += _require(
            s.phase_variance >= th.phase_variance_min,
            "C100.phase_variance_not_collapsed",
            f"{ds.name}: phase variance too small (phase collapse suspected)",
            {
                "dataset": ds.name,
                "phase_variance": s.phase_variance,
                "threshold": th.phase_variance_min,
            },
        )

        fails += _require(
            s.cos_sin_unit_error <= th.cos_sin_unit_error_max,
            "C101.phase_unit_circle_consistent",
            f"{ds.name}: cos/sin unit-circle consistency violated",
            {
                "dataset": ds.name,
                "cos_sin_unit_error": s.cos_sin_unit_error,
                "threshold": th.cos_sin_unit_error_max,
            },
        )

        fails += _require(
            s.phase_uniformity >= th.phase_uniformity_min,
            "C102.phase_uniformity_non_degenerate",
            f"{ds.name}: phase distribution too degenerate",
            {
                "dataset": ds.name,
                "phase_uniformity": s.phase_uniformity,
                "threshold": th.phase_uniformity_min,
            },
        )

    return fails, metrics


def check_class_balance(bundle: DatasetBundle, n_classes: int) -> tuple[list[FailedCheck], dict[str, Any]]:
    """
    Applies only to KNOWN datasets:
        clean, impaired_train, impaired_eval

    Unknown datasets are intentionally excluded.
    """
    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    for ds in bundle.known_datasets():
        y = np.asarray(ds.y).reshape(-1).astype(np.int64)
        counts = np.bincount(y, minlength=n_classes)

        metrics[ds.name] = {"counts": counts.tolist()}

        ok = bool(np.all(counts == counts[0]))

        fails += _require(
            ok,
            "C030.class_balance_exact",
            f"{ds.name}: class counts not exactly equal",
            {
                "dataset": ds.name,
                "counts": counts.tolist(),
            },
        )

    return fails, metrics


def check_known_cross_mode_separation(
    bundle: DatasetBundle,
    fs_hz: float,
    th: Thresholds,
    partial_features_check: bool,
) -> tuple[list[FailedCheck], dict[str, Any]]:
    """
    Known-domain separation:
    - clean vs impaired_train
    - clean vs impaired_eval
    - impaired_train vs impaired_eval
    """
    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    x_clean = _samples_first(np.asarray(bundle.clean.X).T)
    x_tr = _samples_first(np.asarray(bundle.impaired_train.X).T)
    x_ev = _samples_first(np.asarray(bundle.impaired_eval.X).T)

    def avg_spec(x: np.ndarray) -> np.ndarray:
        if partial_features_check:
            max_samples = 256
            if x.shape[0] > max_samples:
                x = x[:max_samples]

        X = np.fft.fft(x, axis=1)
        return np.mean(np.abs(X), axis=0)

    spec_clean = avg_spec(x_clean)
    spec_tr = avg_spec(x_tr)
    spec_ev = avg_spec(x_ev)

    d_freq_tr = effect_size_delta(spec_clean, spec_tr)
    d_freq_ev = effect_size_delta(spec_clean, spec_ev)
    d_freq_te = effect_size_delta(spec_tr, spec_ev)

    metrics["known_effect_sizes"] = {
        "freq_clean_vs_imp_train": float(d_freq_tr),
        "freq_clean_vs_imp_eval": float(d_freq_ev),
        "freq_imp_train_vs_imp_eval": float(d_freq_te),
        "fs_hz": float(fs_hz),
    }

    fails += _require(
        d_freq_tr >= th.min_effect_size_freq_train,
        "C040.crossmode_freq_clean_vs_train",
        "Impaired(train) not sufficiently different from clean in frequency-domain",
        {
            "effect_size": float(d_freq_tr),
            "threshold": th.min_effect_size_freq_train,
        },
    )

    fails += _require(
        d_freq_ev >= th.min_effect_size_freq_eval,
        "C041.crossmode_freq_clean_vs_eval",
        "Impaired(eval) not sufficiently different from clean in frequency-domain",
        {
            "effect_size": float(d_freq_ev),
            "threshold": th.min_effect_size_freq_eval,
        },
    )

    fails += _require(
        d_freq_te >= th.min_effect_size_train_vs_eval_freq,
        "C042.crossmode_freq_train_vs_eval",
        "Impaired(train) and impaired(eval) are too similar in frequency-domain (mode contamination suspected)",
        {
            "effect_size": float(d_freq_te),
            "threshold": th.min_effect_size_train_vs_eval_freq,
        },
    )

    return fails, metrics


def check_unknown_separation(bundle: DatasetBundle, th: Thresholds) -> tuple[list[FailedCheck], dict[str, Any]]:
    """
    Unknown-domain checks:
    - unknown vs clean
    - unknown vs impaired_train
    - unknown vs impaired_eval
    - optional unknown vs clean_unk
    """
    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    if not bundle.has_unknown:
        return fails, {"skipped": True, "reason": "unknown dataset not provided"}

    x_unk = _samples_first(np.asarray(bundle.unknown.X).T)
    x_clean = _samples_first(np.asarray(bundle.clean.X).T)
    x_tr = _samples_first(np.asarray(bundle.impaired_train.X).T)
    x_ev = _samples_first(np.asarray(bundle.impaired_eval.X).T)

    def avg_spec(x: np.ndarray) -> np.ndarray:
        X = np.fft.fft(x, axis=1)
        return np.mean(np.abs(X), axis=0)

    spec_unk = avg_spec(x_unk)
    spec_clean = avg_spec(x_clean)
    spec_tr = avg_spec(x_tr)
    spec_ev = avg_spec(x_ev)

    d_unk_clean = effect_size_delta(spec_unk, spec_clean)
    d_unk_tr = effect_size_delta(spec_unk, spec_tr)
    d_unk_ev = effect_size_delta(spec_unk, spec_ev)

    metrics["unknown_effect_sizes"] = {
        "unknown_vs_clean": float(d_unk_clean),
        "unknown_vs_impaired_train": float(d_unk_tr),
        "unknown_vs_impaired_eval": float(d_unk_ev),
    }

    fails += _require(
        d_unk_clean >= th.min_effect_size_unknown_vs_clean,
        "C200.unknown_vs_clean",
        "Unknown dataset is too similar to clean known dataset",
        {
            "effect_size": float(d_unk_clean),
            "threshold": th.min_effect_size_unknown_vs_clean,
        },
    )

    fails += _require(
        d_unk_tr >= th.min_effect_size_unknown_vs_train,
        "C201.unknown_vs_impaired_train",
        "Unknown dataset is too similar to impaired_train dataset",
        {
            "effect_size": float(d_unk_tr),
            "threshold": th.min_effect_size_unknown_vs_train,
        },
    )

    fails += _require(
        d_unk_ev >= th.min_effect_size_unknown_vs_eval,
        "C202.unknown_vs_impaired_eval",
        "Unknown dataset is too similar to impaired_eval dataset",
        {
            "effect_size": float(d_unk_ev),
            "threshold": th.min_effect_size_unknown_vs_eval,
        },
    )

    if bundle.has_unknown_clean:
        x_clean_unk = _samples_first(np.asarray(bundle.clean_unk.X).T)
        spec_clean_unk = avg_spec(x_clean_unk)
        d_unk_clean_unk = effect_size_delta(spec_unk, spec_clean_unk)

        metrics["unknown_effect_sizes"]["unknown_vs_clean_unk"] = float(d_unk_clean_unk)

        fails += _require(
            d_unk_clean_unk >= th.min_effect_size_unknown_vs_clean_unk,
            "C203.unknown_vs_clean_unk",
            "Unknown impaired dataset is too similar to unknown clean dataset",
            {
                "effect_size": float(d_unk_clean_unk),
                "threshold": th.min_effect_size_unknown_vs_clean_unk,
            },
        )

    return fails, metrics
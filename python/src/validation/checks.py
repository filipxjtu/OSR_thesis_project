from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import FailedCheck
from .stats import (
    time_domain_stats,
    freq_domain_stats,
    effect_size_delta,
)
from .types import DatasetBundle


@dataclass(frozen=True)
class Thresholds:
    # Time-domain
    mean_abs_max: float = 0.05
    std_min: float = 1e-3
    peak_to_rms_max: float = 50.0

    # Frequency-domain
    dc_ratio_max: float = 0.10
    flatness_min: float = 1e-4
    centroid_min_hz: float = 1.0
    centroid_max_hz_ratio: float = 0.49

    # Cross-mode differences (clean vs impaired)
    min_effect_size_freq_train: float = 0.20
    min_effect_size_freq_eval: float = 0.20

    # Train vs Eval must differ (mode contamination gate)
    min_effect_size_train_vs_eval_freq: float = 0.05


def _require(cond: bool, check_id: str, message: str, details: dict[str, Any]) -> list[FailedCheck]:
    return [] if cond else [FailedCheck(check_id=check_id, message=message, details=details)]


def check_no_nan_inf(bundle: DatasetBundle) -> list[FailedCheck]:
    fails: list[FailedCheck] = []
    for ds in bundle.datasets():
        x = np.asarray(ds.x_time(), dtype=np.float64)
        bad = ~np.isfinite(x)
        fails += _require(
            not bool(np.any(bad)),
            "C001.no_nan_inf",
            f"{ds.name}: found NaN/Inf in time-domain samples",
            {"dataset": ds.name, "bad_count": int(np.sum(bad))},
        )
    return fails


def check_time_domain_stats(bundle: DatasetBundle, th: Thresholds) -> tuple[list[FailedCheck], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    fails: list[FailedCheck] = []

    for ds in bundle.datasets():
        s = time_domain_stats(ds.x_time())
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
            {"dataset": ds.name, "mean": s.mean, "threshold": th.mean_abs_max},
        )
        fails += _require(
            s.std >= th.std_min,
            "C011.time_std_not_collapsed",
            f"{ds.name}: std too small (collapse)",
            {"dataset": ds.name, "std": s.std, "threshold": th.std_min},
        )
        fails += _require(
            s.peak_to_rms <= th.peak_to_rms_max,
            "C012.time_peak_to_rms_reasonable",
            f"{ds.name}: peak_to_rms too large (spikes/clipping suspected)",
            {"dataset": ds.name, "peak_to_rms": s.peak_to_rms, "threshold": th.peak_to_rms_max},
        )

    return fails, metrics


def check_freq_domain_stats(bundle: DatasetBundle, fs_hz: float, th: Thresholds) -> tuple[list[FailedCheck], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    fails: list[FailedCheck] = []

    for ds in bundle.datasets():
        n_time = ds.meta().get("N")

        x_time = ensure_samples_first(np.asarray(ds.x_time(), dtype=np.float64), n_time)
        s = freq_domain_stats(x_time, fs_hz=fs_hz)
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
            {"dataset": ds.name, "dc_ratio": s.dc_ratio, "threshold": th.dc_ratio_max},
        )

        fails += _require(
            s.spectral_flatness >= th.flatness_min,
            "C021.freq_flatness_not_degenerate",
            f"{ds.name}: spectral flatness too small (degenerate spectrum suspected)",
            {"dataset": ds.name, "flatness": s.spectral_flatness, "threshold": th.flatness_min},
        )
        fails += _require(
            s.spectral_centroid >= th.centroid_min_hz,
            "C022.freq_centroid_sane_low",
            f"{ds.name}: centroid too low",
            {"dataset": ds.name, "centroid_hz": s.spectral_centroid, "threshold": th.centroid_min_hz},
        )
        fails += _require(
            s.spectral_centroid <= th.centroid_max_hz_ratio * fs_hz,
            "C023.freq_centroid_sane_high",
            f"{ds.name}: centroid too high relative to fs",
            {
                "dataset": ds.name,
                "centroid_hz": s.spectral_centroid,
                "fs_hz": fs_hz,
                "threshold_ratio": th.centroid_max_hz_ratio,
            },
        )

    return fails, metrics


def check_class_balance(bundle: DatasetBundle, n_classes: int) -> tuple[list[FailedCheck], dict[str, Any]]:
    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    def counts(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.int64)
        return np.bincount(y, minlength=n_classes)

    for ds in bundle.datasets():
        c = counts(ds.y())
        metrics[ds.name] = {"counts": c.tolist()}

        ok = bool(np.all(c == c[0]))

        fails += _require(
            ok,
            "C030.class_balance_exact",
            f"{ds.name}: class counts not exactly equal",
            {"dataset": ds.name, "counts": c.tolist()},
        )

    return fails, metrics


def ensure_samples_first(x: np.ndarray, n_time_expected: int) -> np.ndarray:
    # Expect shape (N_samples, N_time)
    if x.ndim != 2:
        raise ValueError("Expected 2D time-domain matrix")

    if x.shape[1] == n_time_expected:
        return x
    elif x.shape[0] == n_time_expected:
        return x.T
    else:
        raise ValueError("Expected 2D time-domain matrix")


def check_cross_mode_separation(
        bundle: DatasetBundle,
        fs_hz: float,
        th: Thresholds,
        partial_features_check: bool = False
) -> tuple[list[FailedCheck], dict[str, Any]]:

    """
    - clean vs impaired_train must differ (freq)
    - clean vs impaired_eval must differ (freq)
    - impaired_train vs impaired_eval must differ (freq)
    """

    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {}

    x_clean = ensure_samples_first(np.asarray(bundle.clean.x_time(), dtype=np.float64), bundle.clean.meta().get("N"))
    x_tr = ensure_samples_first(np.asarray(bundle.impaired_train.x_time(), dtype=np.float64), bundle.impaired_train.meta().get("N"))
    x_ev = ensure_samples_first(np.asarray(bundle.impaired_eval.x_time(), dtype=np.float64), bundle.impaired_eval.meta().get("N"))

   # Frequency-domain effect sizes using avg FFT magnitude vectors
    def avg_spec(x: np.ndarray) -> np.ndarray:
        if partial_features_check:
            max_samples = 256
            if x.shape[0] > max_samples:
                x = x[:max_samples]
        X = np.fft.rfft(x, axis=1)
        return np.mean(np.abs(X), axis=0)

    spec_clean = avg_spec(x_clean)
    spec_tr = avg_spec(x_tr)
    spec_ev = avg_spec(x_ev)

    d_freq_tr = effect_size_delta(spec_clean, spec_tr)
    d_freq_ev = effect_size_delta(spec_clean, spec_ev)
    d_freq_te = float(np.linalg.norm(spec_tr - spec_ev) / (np.linalg.norm(spec_tr) + 1e-12))

    metrics["effect_sizes"] = {
        "freq_clean_vs_imp_train": float(d_freq_tr),
        "freq_clean_vs_imp_eval": float(d_freq_ev),
        "freq_imp_train_vs_imp_eval": float(d_freq_te),
        "fs_hz": float(fs_hz),
    }

    fails += _require(
        d_freq_tr >= th.min_effect_size_freq_train,
        "C040.crossmode_freq_clean_vs_train",
        "Impaired(train) not sufficiently different from clean in frequency-domain",
        {"effect_size": float(d_freq_tr), "threshold": th.min_effect_size_freq_train},
    )

    fails += _require(
        d_freq_ev >= th.min_effect_size_freq_eval,
        "C041.crossmode_freq_clean_vs_eval",
        "Impaired(eval) not sufficiently different from clean in frequency-domain",
        {"effect_size": float(d_freq_ev), "threshold": th.min_effect_size_freq_eval},
    )

    fails += _require(
        d_freq_te >= th.min_effect_size_train_vs_eval_freq,
        "C042.crossmode_freq_train_vs_eval",
        "Impaired(train) and impaired(eval) are too similar in frequency-domain (mode contamination suspected)",
        {"effect_size": float(d_freq_te), "threshold": th.min_effect_size_train_vs_eval_freq},
    )

    return fails, metrics
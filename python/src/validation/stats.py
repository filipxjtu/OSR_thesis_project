from __future__ import annotations

from dataclasses import dataclass
import numpy as np



@dataclass(frozen=True)
class TimeStats:
    mean: float
    std: float
    min: float
    max: float
    rms: float
    peak_to_rms: float
    skewness: float
    kurtosis_excess: float


@dataclass(frozen=True)
class FreqStats:
    dc_ratio: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_flatness: float
    rolloff_95: float


@dataclass(frozen=True)
class PhaseStats:
    phase_mean: float
    phase_std: float
    phase_variance: float

    cos_sin_unit_error: float   # deviation from cos²+sin²=1
    phase_uniformity: float     # entropy-like proxy


def time_domain_stats(x: np.ndarray) -> TimeStats:
    """
    x: (Ns, N) complex or real
    operates on magnitude
    """

    x = np.asarray(x)
    x_mag = np.abs(x)

    mean = float(np.mean(x_mag))
    std = float(np.std(x_mag))
    xmin = float(np.min(x_mag))
    xmax = float(np.max(x_mag))

    rms = float(np.sqrt(np.mean(x_mag**2)))
    peak_to_rms = float(xmax / (rms + 1e-12))

    centered = x_mag - mean
    m2 = np.mean(centered**2)
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)

    skewness = float(m3 / (m2**1.5 + 1e-12))
    kurtosis_excess = float(m4 / (m2**2 + 1e-12) - 3.0)

    return TimeStats(
        mean=mean,
        std=std,
        min=xmin,
        max=xmax,
        rms=rms,
        peak_to_rms=peak_to_rms,
        skewness=skewness,
        kurtosis_excess=kurtosis_excess,
    )


def freq_domain_stats(x: np.ndarray, fs_hz: float) -> FreqStats:
    """   x: (Ns, N)  """

    x = np.asarray(x)

    X = np.fft.fft(x, axis=1)
    mag = np.abs(X)

    mag_mean = np.mean(mag, axis=0)

    N = mag_mean.shape[0]
    freqs = np.fft.fftfreq(N, d=1 / fs_hz)

    mag_sum = np.sum(mag_mean) + 1e-12

    centroid = float(np.sum(freqs * mag_mean) / mag_sum)

    bandwidth = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * mag_mean) / mag_sum)
    )

    geo_mean = np.exp(np.mean(np.log(mag_mean + 1e-12)))
    arith_mean = np.mean(mag_mean)
    flatness = float(geo_mean / (arith_mean + 1e-12))

    dc = mag_mean[0]
    dc_ratio = float(dc / mag_sum)

    cum_sum = np.cumsum(mag_mean)
    idx = np.searchsorted(cum_sum, 0.95 * mag_sum)
    rolloff_95 = float(freqs[min(idx, N - 1)])

    return FreqStats(
        dc_ratio=dc_ratio,
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        spectral_flatness=flatness,
        rolloff_95=rolloff_95,
    )


def phase_domain_stats(x: np.ndarray) -> PhaseStats:
    """   x: (Ns, N) complex   """

    x = np.asarray(x)

    if not np.iscomplexobj(x):
        # real signal
        return PhaseStats(
            phase_mean=0.0,
            phase_std=0.0,
            phase_variance=0.0,
            cos_sin_unit_error=0.0,
            phase_uniformity=0.0,
        )

    phase = np.angle(x)

    phase_mean = float(np.mean(phase))
    phase_std = float(np.std(phase))
    phase_variance = float(np.var(phase))

    # unit circle consistency
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    unit_error = float(np.mean(np.abs(cos_p**2 + sin_p**2 - 1.0)))

    # uniformity proxy (entropy-like)
    hist, _ = np.histogram(phase, bins=32, range=(-np.pi, np.pi), density=True)
    hist = hist + 1e-12
    entropy = -1 * np.sum(hist * np.log(hist))
    phase_uniformity = float(entropy)

    return PhaseStats(
        phase_mean=phase_mean,
        phase_std=phase_std,
        phase_variance=phase_variance,
        cos_sin_unit_error=unit_error,
        phase_uniformity=phase_uniformity,
    )


def effect_size_delta(a: np.ndarray, b: np.ndarray) -> float:
    """   Normalized difference """

    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    return float(
        np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)
    )


def stable_digest(values: dict[str, float]) -> str:
    """     Stable numeric digest for reproducibility checks.   """

    import hashlib

    items = sorted(values.items(), key=lambda kv: kv[0])
    s = "|".join(f"{k}={v:.10e}" for k, v in items).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
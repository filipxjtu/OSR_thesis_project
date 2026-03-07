from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


ArrayF = npt.NDArray[np.floating]


def _safe_mean(x: ArrayF) -> float:
    return float(np.mean(x))


def _safe_std(x: ArrayF) -> float:
    return float(np.std(x))


def _safe_min(x: ArrayF) -> float:
    return float(np.min(x))


def _safe_max(x: ArrayF) -> float:
    return float(np.max(x))


def _rms(x: ArrayF) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _kurtosis_excess(x: ArrayF) -> float:
    # excess kurtosis = E[(x-mu)^4]/sigma^4 - 3
    mu = np.mean(x)
    s2 = np.var(x)
    if s2 == 0:
        return float("nan")
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (s2 ** 2) - 3.0)


def _skewness(x: ArrayF) -> float:
    mu = np.mean(x)
    s = np.std(x)
    if s == 0:
        return float("nan")
    m3 = np.mean((x - mu) ** 3)
    return float(m3 / (s ** 3))


@dataclass(frozen=True)
class TimeDomainStats:
    mean: float
    std: float
    min: float
    max: float
    rms: float
    peak_to_rms: float
    skewness: float
    kurtosis_excess: float


def time_domain_stats(x_time: ArrayF) -> TimeDomainStats:
    """
    x_time: (N_samples, N_time)
    Computes dataset-level stats (flattened).
    """
    x = np.asarray(x_time, dtype=np.float64).reshape(-1)
    rms = _rms(x)
    peak = max(abs(_safe_min(x)), abs(_safe_max(x)))
    ptr = float(peak / rms) if rms > 0 else float("inf")
    return TimeDomainStats(
        mean=_safe_mean(x),
        std=_safe_std(x),
        min=_safe_min(x),
        max=_safe_max(x),
        rms=rms,
        peak_to_rms=ptr,
        skewness=_skewness(x),
        kurtosis_excess=_kurtosis_excess(x),
    )


@dataclass(frozen=True)
class FreqDomainStats:
    # FFT-derived summary (dataset average)
    dc_ratio: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_flatness: float
    rolloff_95: float


def _spectral_flatness(p: ArrayF, eps: float = 1e-12) -> float:
    p = np.maximum(p, eps)
    gm = float(np.exp(np.mean(np.log(p))))
    am = float(np.mean(p))
    return float(gm / am) if am > 0 else float("nan")


def freq_domain_stats(x_time: ArrayF, fs_hz: float, n_time_expected: int=4800) -> FreqDomainStats:
    """
    Uses magnitude spectrum averaged over samples.
    x_time: (N_samples, N_time)
    """
    x = np.asarray(x_time, dtype=np.float64)
    if x.shape[1] == n_time_expected:
        pass
    elif x.shape[0] == n_time_expected:
        x = x.T
    else:
        raise ValueError("x_time must have shape (N_samples, N_time)")
    n = x.shape[1]

    X = np.fft.rfft(x, axis=1)
    mag = np.abs(X)
    p = np.mean(mag ** 2, axis=0)  # avg power vs freq bin, shape (n_rfft,)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    total = float(np.sum(p)) + 1e-12

    dc_ratio = float(p[0] / total)

    centroid = float(np.sum(freqs * p) / total)
    bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p) / total))

    flat = _spectral_flatness(p)

    cumsum = np.cumsum(p)
    idx = int(np.searchsorted(cumsum, 0.95 * cumsum[-1]))
    rolloff = float(freqs[min(idx, len(freqs) - 1)])

    return FreqDomainStats(
        dc_ratio=dc_ratio,
        spectral_centroid=centroid,
        spectral_bandwidth=bw,
        spectral_flatness=flat,
        rolloff_95=rolloff,
    )


def effect_size_delta(a: ArrayF, b: ArrayF, eps: float = 1e-12) -> float:
    """
    Cohen-like delta on flattened arrays: |mu_a - mu_b| / pooled_std
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a)), float(np.std(b))
    pooled = float(np.sqrt(0.5 * (sa * sa + sb * sb)))
    return float(abs(ma - mb) / (pooled + eps))


def stable_digest(values: dict[str, float]) -> str:
    """
    Stable numeric digest for reproducibility checks.
    Quantizes floats to fixed precision and hashes the string.
    """
    import hashlib

    items = sorted(values.items(), key=lambda kv: kv[0])
    s = "|".join(f"{k}={v:.10e}" for k, v in items).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
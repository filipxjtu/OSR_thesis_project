from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .types import Dataset, DatasetBundle



# Internal helpers
def _samples_first(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x

# Core feature extractors
def compute_magnitude(x: np.ndarray) -> np.ndarray:
    return np.abs(x)

def compute_spectrum(x: np.ndarray) -> np.ndarray:
    """
    Mean magnitude spectrum over samples.
    Input: x: (Ns, N), Output: spectrum: (N,)
    """
    X = np.fft.fft(x, axis=1)
    mag = np.abs(X)
    return np.mean(mag, axis=0)

def compute_phase(x: np.ndarray) -> np.ndarray:
    """ Phase of complex signal. eturns: phase: (Ns, N)  """
    if not np.iscomplexobj(x):
        return np.zeros_like(x, dtype=np.float64)
    return np.angle(x)

def compute_phase_components(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Returns cos(phase), sin(phase). Used for consistency checks. """
    phase = compute_phase(x)
    return np.cos(phase), np.sin(phase)

def compute_class_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    """     Class histogram (fixed length) """
    y = np.asarray(y).reshape(-1).astype(np.int64)
    return np.bincount(y, minlength=n_classes)


# Dataset-level feature pack
def extract_dataset_features(ds: Dataset) -> Dict[str, Any]:
    """
    Extract reusable features for a single dataset.
    This function centralizes all repeated computations used across checks.
    """

    x = _samples_first(np.asarray(ds.X).T)

    features: Dict[str, Any] = {}

    # time-domain
    features["magnitude"] = compute_magnitude(x)

    # frequency-domain
    features["spectrum"] = compute_spectrum(x)

    # phase
    phase = compute_phase(x)
    cos_p, sin_p = compute_phase_components(x)

    features["phase"] = phase
    features["cos_phase"] = cos_p
    features["sin_phase"] = sin_p

    # labels
    features["labels"] = np.asarray(ds.y).reshape(-1)

    return features


# Bundle-level feature pack
def extract_bundle_features(bundle: DatasetBundle) -> Dict[str, Dict[str, Any]]:
    """     Extract features for all datasets. Output: { dataset_name: {feature_dict} }    """
    out: Dict[str, Dict[str, Any]] = {}
    for ds in bundle.all_datasets():
        out[ds.name] = extract_dataset_features(ds)
    return out


# Separation helpers
def compute_spectrum_distance(a: np.ndarray, b: np.ndarray) -> float:
    """     Wrapper over effect-size-like distance for spectra.     """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(
        np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)
    )

def compute_phase_variability(phase: np.ndarray) -> float:
    """     Scalar measure of phase spread.     """
    return float(np.var(phase))

def compute_unit_circle_error(cos_p: np.ndarray, sin_p: np.ndarray) -> float:
    """     Measures violation of cos² + sin² = 1     """
    return float(np.mean(np.abs(cos_p**2 + sin_p**2 - 1.0)))
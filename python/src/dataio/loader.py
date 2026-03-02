from __future__ import annotations

from pathlib import Path

import h5py
import os

from .contract import validate_and_normalize
from .exceptions import ArtifactLoadError
from .dataset_artifact import DatasetArtifact


def load_artifact(path: str | Path) -> DatasetArtifact:
    """
    loader for MATLAB v7.3 (.mat HDF5) artifacts.
    """

    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / 'artifacts' / 'datasets'

    folders_to_check = ['clean', 'impaired']

    p = None
    for folder in folders_to_check:
        candidate = data_root / folder / path
        if candidate.exists():
            p = candidate
            break

    if p is None:
        raise FileNotFoundError(str(p))
    if p.suffix.lower() != ".mat":
        raise ValueError(f"Expected .mat file, got: {p.suffix}")

    try:
        with h5py.File(p, "r") as f:
            artifact = validate_and_normalize(f, path=str(p))
            return artifact
    except ArtifactLoadError:
        raise
    except Exception as e:
        raise ArtifactLoadError(f"Unhandled loader error for {p}: {e}") from e

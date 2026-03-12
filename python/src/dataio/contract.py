from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from .exceptions import (
    DtypeMismatchError,
    HashMismatchError,
    MetadataError,
    MissingFieldError,
    MultipleRootError,
    NumericDomainError,
    RootNotFoundError,
    ShapeMismatchError,
    AlignmentError,
)

from .dataset_artifact import DatasetArtifact


def _is_hdf5_ref(ds: h5py.Dataset) -> bool:
    return ds.dtype == h5py.ref_dtype

def _follow_ref(f: h5py.File, ref: h5py.Reference) -> Any:
    if not ref:
        return None
    return f[ref]

def _read_matlab_struct(f: h5py.File, grp: h5py.Group) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in grp.keys():
        out[k] = _read_matlab_value(f, grp[k])
    return out

def _read_matlab_cell(f: h5py.File, ds: h5py.Dataset) -> list[Any]:
    data = ds[()]
    flat = data.reshape(-1)
    out: list[Any] = []
    for ref in flat:
        target = _follow_ref(f, ref)
        out.append(_read_matlab_value(f, target))
    return out

def _read_dataset_value(ds: h5py.Dataset) -> Any:
    arr = ds[()]

    if isinstance(arr, np.ndarray):
        if arr.dtype == np.uint16:
            return "".join(chr(int(c)) for c in arr.flatten()).rstrip("\x00")
        if arr.size == 1:
            return arr.item()

    if isinstance(arr, (bytes, bytearray)):
        return arr.decode("utf-8", errors="replace")

    return arr


def _read_matlab_value(f: h5py.File, obj: Any) -> Any:
    if isinstance(obj, h5py.Group):
        return _read_matlab_struct(f, obj)

    if isinstance(obj, h5py.Dataset):
        if _is_hdf5_ref(obj):
            refs = obj[()]
            if refs.shape == ():
                target = _follow_ref(f, refs)
                return _read_matlab_value(f, target)

            flat_refs = refs.reshape(-1)
            values = []
            for ref in flat_refs:
                target = _follow_ref(f, ref)
                values.append(_read_matlab_value(f, target))

            if all(isinstance(v, str) for v in values):
                return "".join(values)
            return values
        return _read_dataset_value(obj)
    return obj


def _read_root_group(f: h5py.File, root: str) -> h5py.Group:
    if root not in f:
        raise RootNotFoundError(f"Root '{root}' not found.")
    obj = f[root]
    if not isinstance(obj, h5py.Group):
        raise RootNotFoundError(f"Root '{root}' exists but is not a struct/group.")
    return obj


def _detect_root(f: h5py.File) -> str:
    candidates = [k for k in f.keys() if k in {"dataset", "impaired_data"}]
    if len(candidates) == 0:
        raise RootNotFoundError("Neither 'dataset' nor 'impaired_data' found.")
    if len(candidates) > 1:
        raise MultipleRootError(f"Multiple root structs found: {candidates}")
    return candidates[0]


def compute_simple64_checksum(x_raw: np.ndarray, y_raw: np.ndarray, N: int, Ns: int) -> int:
    x_raw = np.asarray(x_raw)
    y_raw = np.asarray(y_raw)

    # match matlab X(:)
    if x_raw.shape == (N, Ns):
        x_flat = x_raw.ravel(order="F")
    elif x_raw.shape == (Ns, N):
        x_flat = x_raw.ravel(order="C")
    else:
        raise AlignmentError(f"Unexpected X shape for hashing: {x_raw.shape}")

    # match matlab double(y(:))
    if y_raw.shape == (Ns, 1):
        y_flat = y_raw.ravel(order="F")
    elif y_raw.shape == (1, Ns):
        y_flat = y_raw.ravel(order="C")
    elif y_raw.shape == (Ns,):
        y_flat = y_raw.ravel(order="C")
    else:
        raise AlignmentError(f"Unexpected y shape for hashing: {y_raw.shape}")

    y_flat = y_flat.astype(np.float64, copy=False)

    if x_flat.dtype != np.float64:
        x_flat = x_flat.astype(np.float64, copy=False)

    # concatenate like in matlab
    data = np.concatenate([x_flat, y_flat])

    # view as raw bytes
    bytes_ = data.view(np.uint8)

    # vectorized accumulation
    h = np.uint64(14695981039346656037)
    h = np.uint64(h + np.sum(bytes_, dtype=np.uint64))

    return int(h)


# Validate
def validate_and_normalize(f: h5py.File, path: str) -> DatasetArtifact:

    root = _detect_root(f)
    root_grp = _read_root_group(f, root)
    root_dict = _read_matlab_struct(f, root_grp)

    # required fields
    if root == "dataset":
        required = ["X_clean", "y", "params", "meta"]
        x_name = "X_clean"
    else:
        required = ["X_imp", "y", "params", "imp_params", "meta"]
        x_name = "X_imp"

    for r in required:
        if r not in root_dict:
            raise MissingFieldError(f"{path}: missing field '{r}' in root '{root}'")

    # extract arrays (keep row for checksum)
    x_raw = np.asarray(root_dict[x_name])
    y_raw = np.asarray(root_dict["y"])

    x = x_raw
    y = y_raw

    params = root_dict["params"]
    imp_params = root_dict["imp_params"] if root == "impaired_data" else None

    meta_raw = root_dict["meta"]
    if not isinstance(meta_raw, dict):
        raise MetadataError(f"{path}: meta must be a struct/dict.")
    meta: dict[str, Any] = meta_raw

    # normalize orientation
    N_meta = int(meta["N"])
    Ns_meta = int(meta["Ns"])

    if x.shape == (Ns_meta, N_meta):
        x = x.T

    y = y.reshape(-1, 1)
    if y.shape == (1, Ns_meta):
        y = y.T

    # shape validation
    if x.shape != (N_meta, Ns_meta):
        raise ShapeMismatchError(
            f"{path}: X shape mismatch. expected {(N_meta, Ns_meta)}, got {x.shape}"
        )

    if y.shape != (Ns_meta, 1):
        raise ShapeMismatchError(
            f"{path}: y shape mismatch. expected {(Ns_meta, 1)}, got {y.shape}"
        )

    # dtype validation
    if x.dtype != np.float64:
        raise DtypeMismatchError(f"{path}: X must be float64, got {x.dtype}")

    if y.dtype != np.int32:
        raise DtypeMismatchError(f"{path}: y must be int32, got {y.dtype}")

    # numeric validation
    if not np.isfinite(x).all():
        raise NumericDomainError(f"{path}: X contains NaN/Inf")

    # meta validation
    required_meta = [
        "spec_version",
        "dataset_seed",
        "artifact_hash_fn",
        "artifact_hash",
        "layout",
        "dtype_policy",
        "N",
        "Ns",
        "fs",
        "n_per_class",
        "class_set",
        "dataset_version",
        "created_utc",
    ]

    for k in required_meta:
        if k not in meta:
            raise MetadataError(f"{path}: missing meta key '{k}'")

    meta_hash_fn = str(meta["artifact_hash_fn"])
    if meta_hash_fn != "simple64_checksum":
        raise MetadataError(f"{path}: unsupported hash function: {meta_hash_fn}")

    layout = str(meta["layout"])
    if layout != "N_by_Ns_columns_are_samples":
        raise MetadataError(f"{path}: layout mismatch: {layout}")

    if root == "impaired_data":
        mode = str(meta["mode"])
        if mode not in {"train", "eval"}:
            raise MetadataError(f"{path}: invalid mode: {mode}")

       # hash verification
    computed_hash = compute_simple64_checksum(x_raw, y_raw, N_meta, Ns_meta)
    stored_hash = int(meta["artifact_hash"])

    if computed_hash != stored_hash:
        raise HashMismatchError(
            f"{path}: artifact_hash mismatch. expected {stored_hash}, got {computed_hash}"
        )

    return DatasetArtifact(
        X=x,
        y=y,
        params=params,
        imp_params=imp_params,
        meta=meta,
        root=root,
    )
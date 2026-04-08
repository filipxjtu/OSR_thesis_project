from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from .exceptions import FailedCheck
from .stats import stable_digest
from .types import DatasetBundle
from .checks import (
    Thresholds,
    check_time_domain_stats,
    check_freq_domain_stats,
    check_phase_domain_stats,
    check_known_cross_mode_separation,
    check_unknown_separation,
)


@dataclass(frozen=True)
class ReproConfig:
    trials: int = 2
    require_identical_digest: bool = True


def _flatten_selected_scalars(obj: dict[str, Any], prefix: str, out: dict[str, float]) -> None:
    for k, v in obj.items():
        key = f"{prefix}.{k}"
        if isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            _flatten_selected_scalars(v, key, out)


def _feature_digest(bundle: DatasetBundle) -> str:
    """
    lightweight and deterministic.
    mean/std magnitude and mean/std phase on every dataset present in the bundle
    """
    flat: dict[str, float] = {}

    for ds in bundle.all_datasets():
        x = np.asarray(ds.X).T   # (Ns, N)

        mag = np.abs(x)
        flat[f"{ds.name}.mag_mean"] = float(np.mean(mag))
        flat[f"{ds.name}.mag_std"] = float(np.std(mag))

        if np.iscomplexobj(x):
            phase = np.angle(x)
            flat[f"{ds.name}.phase_mean"] = float(np.mean(phase))
            flat[f"{ds.name}.phase_std"] = float(np.std(phase))
        else:
            flat[f"{ds.name}.phase_mean"] = 0.0
            flat[f"{ds.name}.phase_std"] = 0.0

    return stable_digest(flat)


def _mode_digest(
    bundle: DatasetBundle,
    mode_name: str,
    fs_hz: float,
    n_classes: int,
    th: Thresholds,
) -> str:
    """
    Digest one dataset role from its scalar validation metrics.
    """
    ds = getattr(bundle, mode_name)

    single_bundle = DatasetBundle(
        clean=ds if mode_name == "clean" else bundle.clean,
        impaired_train=ds if mode_name == "impaired_train" else bundle.impaired_train,
        impaired_eval=ds if mode_name == "impaired_eval" else bundle.impaired_eval,
        unknown=bundle.unknown if mode_name != "unknown" else bundle.unknown,
        clean_unk=bundle.clean_unk if mode_name != "clean_unk" else bundle.clean_unk,
    )

    flat: dict[str, float] = {}

    _, t_met = check_time_domain_stats(single_bundle, th)
    _, f_met = check_freq_domain_stats(single_bundle, fs_hz, th)
    _, p_met = check_phase_domain_stats(single_bundle, th)

    _flatten_selected_scalars(t_met.get(ds.name, {}), f"time.{ds.name}", flat)
    _flatten_selected_scalars(f_met.get(ds.name, {}), f"freq.{ds.name}", flat)
    _flatten_selected_scalars(p_met.get(ds.name, {}), f"phase.{ds.name}", flat)

    y = np.asarray(ds.y).reshape(-1)
    if ds.name in {"clean", "impaired_train", "impaired_eval"}:
        counts = np.bincount(y.astype(np.int64), minlength=n_classes)
        flat[f"class.{ds.name}.total"] = float(np.sum(counts))
        flat[f"class.{ds.name}.min"] = float(np.min(counts))
        flat[f"class.{ds.name}.max"] = float(np.max(counts))
    else:
        unique, counts = np.unique(y.astype(np.int64), return_counts=True)
        flat[f"class.{ds.name}.total"] = float(np.sum(counts))
        flat[f"class.{ds.name}.num_unique"] = float(len(unique))
        if len(counts) > 0:
            flat[f"class.{ds.name}.min"] = float(np.min(counts))
            flat[f"class.{ds.name}.max"] = float(np.max(counts))

    return stable_digest(flat)


def _bundle_digest(
    bundle: DatasetBundle,
    fs_hz: float,
    n_classes: int,
    th: Thresholds,
) -> str:
    """
    Digest full bundle from separation metrics + per-mode digests + feature digest.
    """
    flat: dict[str, float] = {}

    _, known_sep = check_known_cross_mode_separation(
        bundle=bundle,
        fs_hz=fs_hz,
        th=th,
        partial_features_check=False,
    )
    _flatten_selected_scalars(known_sep, "known_sep", flat)

    _, unk_sep = check_unknown_separation(bundle, th)
    _flatten_selected_scalars(unk_sep, "unk_sep", flat)

    known_sep_digest = stable_digest(flat)
    feat_digest = _feature_digest(bundle)

    mode_parts = [
        f"clean={_mode_digest(bundle, 'clean', fs_hz, n_classes, th)}",
        f"imp_train={_mode_digest(bundle, 'impaired_train', fs_hz, n_classes, th)}",
        f"imp_eval={_mode_digest(bundle, 'impaired_eval', fs_hz, n_classes, th)}",
    ]

    if bundle.unknown is not None:
        mode_parts.append(
            f"unknown={_mode_digest(bundle, 'unknown', fs_hz, n_classes, th)}"
        )

    if bundle.clean_unk is not None:
        mode_parts.append(
            f"clean_unk={_mode_digest(bundle, 'clean_unk', fs_hz, n_classes, th)}"
        )

    mode_parts.append(f"known_sep={known_sep_digest}")
    mode_parts.append(f"feat={feat_digest}")

    import hashlib
    blob = "|".join(mode_parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def check_reproducibility(
    loader: Callable[[], DatasetBundle],
    fs_hz: float,
    n_classes: int,
    th: Thresholds,
    rc: ReproConfig,
) -> tuple[list[FailedCheck], dict[str, Any]]:
    """
    Loads the same bundle multiple times and verifies digest identity.
    """
    records: list[dict[str, str]] = []

    for _ in range(rc.trials):
        bundle = loader()

        rec: dict[str, str] = {
            "clean_digest": _mode_digest(bundle, "clean", fs_hz, n_classes, th),
            "imp_train_digest": _mode_digest(bundle, "impaired_train", fs_hz, n_classes, th),
            "imp_eval_digest": _mode_digest(bundle, "impaired_eval", fs_hz, n_classes, th),
            "feature_digest": _feature_digest(bundle),
            "bundle_digest": _bundle_digest(bundle, fs_hz, n_classes, th),
        }

        if bundle.unknown is not None:
            rec["unknown_digest"] = _mode_digest(bundle, "unknown", fs_hz, n_classes, th)

        if bundle.clean_unk is not None:
            rec["clean_unk_digest"] = _mode_digest(bundle, "clean_unk", fs_hz, n_classes, th)

        records.append(rec)

    def all_equal(key: str) -> bool:
        return all(r[key] == records[0][key] for r in records[1:]) if records else True

    identical = {k: all_equal(k) for k in records[0].keys()} if records else {}
    ok = all(identical.values()) if rc.require_identical_digest else True

    fails: list[FailedCheck] = []
    if rc.require_identical_digest and not ok:
        fails.append(
            FailedCheck(
                check_id="C300.reproducibility_digest_identical",
                message="Reproducibility failure: one or more digests differ across trials",
                details={
                    "records": records,
                    "identical": identical,
                    "trials": rc.trials,
                },
            )
        )

    metrics = {
        "records": records,
        "identical": identical,
        "trials": rc.trials,
    }

    return fails, metrics
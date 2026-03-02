from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from .exceptions import FailedCheck
from .stats import stable_digest
from .types import DatasetBundle
from .checks import Thresholds, check_time_domain_stats, check_freq_domain_stats, check_class_balance, check_cross_mode_separation


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


def _mode_digest(
    bundle: DatasetBundle,
    mode_name: str,
    fs_hz: float,
    n_classes: int,
    th: Thresholds,
) -> str:
    """
    Digest each mode from its own time+freq stats and class totals/min/max.
    """
    # Build a temporary "single-dataset bundle view" by selecting one ds at a time
    ds = getattr(bundle, mode_name)

    # time/freq stats dicts are nested by ds.name; extract only this ds
    _, tmet = check_time_domain_stats(bundle, th)
    _, fmet = check_freq_domain_stats(bundle, fs_hz, th)
    _, cmet = check_class_balance(bundle, n_classes, th)

    flat: dict[str, float] = {}

    _flatten_selected_scalars(tmet.get(ds.name, {}), f"time.{ds.name}", flat)
    _flatten_selected_scalars(fmet.get(ds.name, {}), f"freq.{ds.name}", flat)

    counts = cmet.get(ds.name, {}).get("counts", [])
    if counts:
        flat[f"class.{ds.name}.total"] = float(sum(counts))
        flat[f"class.{ds.name}.min"] = float(min(counts))
        flat[f"class.{ds.name}.max"] = float(max(counts))

    return stable_digest(flat)


def _bundle_digest(
    bundle: DatasetBundle,
    fs_hz: float,
    n_classes: int,
    th: Thresholds,
) -> str:
    """
    Digest overall separation metrics + all per-mode digests.
    """
    _, smet = check_cross_mode_separation(bundle, fs_hz, th)
    flat: dict[str, float] = {}
    _flatten_selected_scalars(smet, "sep", flat)

    # Incorporate per-mode digests as numeric values by hashing them into floats? No.
    # Instead incorporate the digest strings directly by hashing into one final string.
    # stable_digest only accepts floats, so do final hashing on concatenated strings here.
    import hashlib

    d_clean = _mode_digest(bundle, "clean", fs_hz, n_classes, th)
    d_tr = _mode_digest(bundle, "impaired_train", fs_hz, n_classes, th)
    d_ev = _mode_digest(bundle, "impaired_eval", fs_hz, n_classes, th)

    sep_digest = stable_digest(flat)
    blob = f"clean={d_clean}|train={d_tr}|eval={d_ev}|sep={sep_digest}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def check_reproducibility(
    loader: Callable[[], DatasetBundle],
    fs_hz: float,
    n_classes: int,
    th: Thresholds,
    rc: ReproConfig,
) -> tuple[list[FailedCheck], dict[str, Any]]:
    """
    Loads the same artifacts multiple times (or regenerates via deterministic pipeline),
    computes per-mode digests + bundle digest, and enforces identity across trials.
    """
    records: list[dict[str, str]] = []

    for _ in range(rc.trials):
        bundle = loader()
        rec = {
            "clean_digest": _mode_digest(bundle, "clean", fs_hz, n_classes, th),
            "imp_train_digest": _mode_digest(bundle, "impaired_train", fs_hz, n_classes, th),
            "imp_eval_digest": _mode_digest(bundle, "impaired_eval", fs_hz, n_classes, th),
        }
        rec["bundle_digest"] = _bundle_digest(bundle, fs_hz, n_classes, th)
        records.append(rec)

    def all_equal(key: str) -> bool:
        return all(r[key] == records[0][key] for r in records[1:]) if records else True

    identical = {k: all_equal(k) for k in records[0].keys()} if records else {}
    ok = all(identical.values()) if rc.require_identical_digest else True

    fails: list[FailedCheck] = []
    if rc.require_identical_digest and not ok:
        fails.append(
            FailedCheck(
                check_id="C050.reproducibility_digest_identical",
                message="Reproducibility failure: one or more digests differ across trials",
                details={"records": records, "identical": identical},
            )
        )

    metrics = {"records": records, "identical": identical, "trials": rc.trials}
    return fails, metrics
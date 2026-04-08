from __future__ import annotations

from typing import Callable, Any

from .types import DatasetBundle
from .checks import (
    Thresholds,
    check_no_nan_inf,
    check_time_domain_stats,
    check_freq_domain_stats,
    check_phase_domain_stats,
    check_class_balance,
    check_known_cross_mode_separation,
    check_unknown_separation,
)
from .repro import check_reproducibility, ReproConfig
from .exceptions import FailedCheck
from .summary import ValidationSummary


VALIDATOR_VERSION = "v2.0"

# Helper
def _collect_check_ids(fails: list[FailedCheck]) -> list[str]:
    return [f.check_id for f in fails]


def _unique(seq: list[str]) -> list[str]:
    return sorted(list(set(seq)))


def run_validation_gate(
    bundle: DatasetBundle,
    fs_hz: float,
    n_classes: int,
    thresholds: Thresholds | None = None,
    repro_loader: Callable[[], DatasetBundle] | None = None,
    repro_config: ReproConfig | None = None,
    partial_features_check: bool = False,
) -> ValidationSummary:

    th = thresholds or Thresholds()

    checks_failed: list[str] = []
    checks_passed: list[str] = []
    metrics: dict[str, Any] = {}
    notes: list[str] = []


    # Numeric domain
    fails = check_no_nan_inf(bundle)
    if fails:
        checks_failed.extend(_collect_check_ids(fails))
    else:
        checks_passed.append("C001.no_nan_inf_time")

    metrics["numeric"] = {"passed": len(fails) == 0}


    # Time-domain
    fails, m = check_time_domain_stats(bundle, th)
    metrics["time"] = m

    if fails:
        checks_failed.extend(_collect_check_ids(fails))
    else:
        checks_passed.append("time_domain_stats")


    # Frequency-domain
    fails, m = check_freq_domain_stats(bundle, fs_hz, th)
    metrics["frequency"] = m

    if fails:
        checks_failed.extend(_collect_check_ids(fails))
    else:
        checks_passed.append("freq_domain_stats")


    # Phase-domain
    fails, m = check_phase_domain_stats(bundle, th)
    metrics["phase"] = m

    if fails:
        checks_failed.extend(_collect_check_ids(fails))
    else:
        checks_passed.append("phase_domain_stats")


    # Class balance
    fails, m = check_class_balance(bundle, n_classes)
    metrics["class_balance"] = m

    if fails:
        checks_failed.extend(_collect_check_ids(fails))
    else:
        checks_passed.append("class_balance")


    # Known separation
    fails, m = check_known_cross_mode_separation(
        bundle,
        fs_hz,
        th,
        partial_features_check=partial_features_check,
    )
    metrics["known_separation"] = m

    if fails:
        checks_failed.extend(_collect_check_ids(fails))
    else:
        checks_passed.append("known_separation")


    # Unknown separation
    fails, m = check_unknown_separation(bundle, th)
    metrics["unknown_separation"] = m

    if fails:
        checks_failed.extend(_collect_check_ids(fails))
    else:
        if not bundle.has_unknown:
            notes.append("Unknown dataset not provided → unknown checks skipped")
        else:
            checks_passed.append("unknown_separation")


    # Reproducibility
    if repro_loader is not None:
        rc = repro_config or ReproConfig()

        fails, m = check_reproducibility(
            loader=repro_loader,
            fs_hz=fs_hz,
            n_classes=n_classes,
            th=th,
            rc=rc,
        )

        metrics["reproducibility"] = m

        if fails:
            checks_failed.extend(_collect_check_ids(fails))
        else:
            checks_passed.append("reproducibility")

    else:
        metrics["reproducibility"] = {
            "skipped": True,
            "reason": "no loader provided",
        }
        notes.append("Reproducibility check skipped")

    # Finalize
    checks_failed = _unique(checks_failed)
    checks_passed = _unique(checks_passed)

    status = "PASS" if len(checks_failed) == 0 else "FAIL"

    summary = ValidationSummary(
        validator_version=VALIDATOR_VERSION,
        status=status,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        metrics=metrics,
        thresholds=th.__dict__,
        notes=notes,
    )

    return summary
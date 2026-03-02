from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .exceptions import FailedCheck, ValidationError
from .summary import ValidationSummary


@dataclass(frozen=True)
class BaselineTolerances:
    # absolute tolerance on selected scalar metrics
    abs_tol: float = 1e-3
    # relative tolerance (fraction of baseline value)
    rel_tol: float = 0.05


def _get_scalar(metrics: dict[str, Any], path: str) -> float | None:
    """
    Path format: "time_domain.clean.std" etc.
    """
    cur: Any = metrics
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def compare_to_baseline(
    summary: ValidationSummary,
    baseline: ValidationSummary,
    keys: list[str],
    tol: BaselineTolerances,
) -> None:
    """
    Fails loudly if key scalars drift beyond tolerances.
    """
    failures: list[FailedCheck] = []

    for k in keys:
        a = _get_scalar(summary.metrics, k)
        b = _get_scalar(baseline.metrics, k)
        if a is None or b is None:
            failures.append(
                FailedCheck(
                    check_id="B000.baseline_key_missing",
                    message="Baseline comparison key missing",
                    details={"key": k, "current": a, "baseline": b},
                )
            )
            continue

        abs_err = abs(a - b)
        rel_err = abs_err / (abs(b) + 1e-12)

        ok = (abs_err <= tol.abs_tol) or (rel_err <= tol.rel_tol)
        if not ok:
            failures.append(
                FailedCheck(
                    check_id="B010.baseline_metric_drift",
                    message="Metric drift beyond tolerance",
                    details={
                        "key": k,
                        "current": a,
                        "baseline": b,
                        "abs_err": abs_err,
                        "rel_err": rel_err,
                        "abs_tol": tol.abs_tol,
                        "rel_tol": tol.rel_tol,
                    },
                )
            )

    if failures:
        raise ValidationError(failures)
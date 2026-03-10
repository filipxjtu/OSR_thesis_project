from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailedCheck:
    check_id: str
    message: str
    details: dict[str, Any]

class ValidationError(RuntimeError):
    """
    Raised when one or more mandatory sanity checks fail.
    """

    def __init__(self, failures: list[FailedCheck]) -> None:
        self.failures = failures
        msg = self._format_message(failures)
        super().__init__(msg)

    @staticmethod
    def _format_message(failures: list[FailedCheck]) -> str:
        lines = [f"Dataset validation failed: {len(failures)} check(s) failed."]
        for f in failures[:20]:
            lines.append(f"- {f.check_id}: {f.message}")
        if len(failures) > 20:
            lines.append(f"... and {len(failures) - 20} more.")
        return "\n".join(lines)
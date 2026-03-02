from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json
from pathlib import Path


@dataclass
class ValidationSummary:
    validator_version: str
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "PASS"  # PASS / FAIL
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_version": self.validator_version,
            "created_utc": self.created_utc,
            "status": self.status,
            "checks_passed": list(self.checks_passed),
            "checks_failed": list(self.checks_failed),
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "notes": list(self.notes),
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @staticmethod
    def load_json(path: str | Path) -> "ValidationSummary":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        s = ValidationSummary(validator_version=str(d["validator_version"]))
        s.created_utc = str(d.get("created_utc", ""))
        s.status = str(d.get("status", ""))
        s.checks_passed = list(d.get("checks_passed", []))
        s.checks_failed = list(d.get("checks_failed", []))
        s.metrics = dict(d.get("metrics", {}))
        s.thresholds = dict(d.get("thresholds", {}))
        s.notes = list(d.get("notes", []))
        return s
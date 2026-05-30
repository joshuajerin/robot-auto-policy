"""Safety checks that are intentionally outside the editable agent surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SafetyResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def run_locomotion_safety_checks(metrics: dict[str, Any]) -> SafetyResult:
    reasons: list[str] = []

    if float(metrics.get("max_torque_limit_fraction", 0.0)) > 1.0:
        reasons.append("torque limit exceeded")
    if float(metrics.get("joint_limit_violation_rate", 0.0)) > 0.02:
        reasons.append("joint limit violation rate too high")
    if float(metrics.get("fall_rate", 0.0)) > 0.35:
        reasons.append("fall rate too high")
    if float(metrics.get("base_height_min", 1.0)) < 0.35:
        reasons.append("base height collapsed below safety threshold")
    if bool(metrics.get("nan_actions", False)):
        reasons.append("policy produced NaN actions")

    return SafetyResult(passed=not reasons, reasons=reasons)


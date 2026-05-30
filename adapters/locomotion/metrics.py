"""Locomotion metric conversion."""

from __future__ import annotations

from typing import Any

from core.schemas import ScoreBreakdown
from core.scoring import score_from_metrics


def score_locomotion(raw_metrics: dict[str, Any]) -> ScoreBreakdown:
    policy_id = str(raw_metrics.get("policy_id", "unknown_policy"))
    return score_from_metrics(policy_id, raw_metrics)


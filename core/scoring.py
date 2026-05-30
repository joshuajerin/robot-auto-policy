"""Locked scoring and accept/reject rules."""

from __future__ import annotations

from typing import Any

from core.schemas import ScoreBreakdown


MIN_SCORE_DELTA = 0.03
MAX_BASE_REGRESSION = 0.05
MIN_EVAL_SEEDS = 8


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def score_from_metrics(policy_id: str, raw_metrics: dict[str, Any]) -> ScoreBreakdown:
    """Convert raw evaluator metrics into the fixed RoboGenesis score."""

    command_tracking = _clamp01(raw_metrics.get("command_tracking"))
    survival_no_fall = _clamp01(raw_metrics.get("survival_no_fall"))
    stability = _clamp01(raw_metrics.get("stability"))
    generated_scenario_success = _clamp01(raw_metrics.get("generated_scenario_success"))
    gait_quality = _clamp01(raw_metrics.get("gait_quality"))
    energy_efficiency = _clamp01(raw_metrics.get("energy_efficiency"))
    smoothness = _clamp01(raw_metrics.get("smoothness"))
    recovery_from_disturbance = _clamp01(raw_metrics.get("recovery_from_disturbance"))
    safety_penalty = _clamp01(raw_metrics.get("safety_penalty"))
    regression_penalty = _clamp01(raw_metrics.get("regression_penalty"))

    total_score = (
        0.20 * command_tracking
        + 0.20 * survival_no_fall
        + 0.15 * stability
        + 0.15 * generated_scenario_success
        + 0.10 * gait_quality
        + 0.10 * energy_efficiency
        + 0.05 * smoothness
        + 0.05 * recovery_from_disturbance
        - safety_penalty
        - regression_penalty
    )

    return ScoreBreakdown(
        policy_id=policy_id,
        total_score=max(0.0, min(1.0, total_score)),
        command_tracking=command_tracking,
        survival_no_fall=survival_no_fall,
        stability=stability,
        generated_scenario_success=generated_scenario_success,
        gait_quality=gait_quality,
        energy_efficiency=energy_efficiency,
        smoothness=smoothness,
        recovery_from_disturbance=recovery_from_disturbance,
        safety_penalty=safety_penalty,
        regression_penalty=regression_penalty,
        safety_passed=bool(raw_metrics.get("safety_passed", True)),
        base_success=_clamp01(raw_metrics.get("base_success", survival_no_fall)),
        eval_seed_count=int(raw_metrics.get("eval_seed_count", 0)),
        reward_hacking_detected=bool(raw_metrics.get("reward_hacking_detected", False)),
    )


def should_accept(old: ScoreBreakdown, new: ScoreBreakdown) -> bool:
    """Locked policy promotion rule."""

    return (
        new.total_score >= old.total_score + MIN_SCORE_DELTA
        and new.safety_passed
        and new.base_success >= old.base_success - MAX_BASE_REGRESSION
        and new.generated_scenario_success >= old.generated_scenario_success
        and new.eval_seed_count >= MIN_EVAL_SEEDS
        and not new.reward_hacking_detected
    )


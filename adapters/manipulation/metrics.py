"""Manipulation metric conversion."""

from __future__ import annotations

from typing import Any

from core.schemas import ScoreBreakdown


def score_manipulation(raw_metrics: dict[str, Any]) -> ScoreBreakdown:
    policy_id = str(raw_metrics.get("policy_id", "unknown_manipulation_policy"))
    task_success = _clamp01(raw_metrics.get("task_success_rate", raw_metrics.get("success_rate", 0.0)))
    task_progress = _clamp01(raw_metrics.get("task_progress", raw_metrics.get("completion_rate", task_success)))
    grasp_stability = _clamp01(raw_metrics.get("grasp_stability", raw_metrics.get("grasp_success_rate", 0.0)))
    placement_accuracy = _clamp01(raw_metrics.get("placement_accuracy", 1.0 - float(raw_metrics.get("placement_error_m", 1.0))))
    generated_success = _clamp01(raw_metrics.get("generated_scenario_success", 0.0))
    energy_efficiency = _clamp01(raw_metrics.get("energy_efficiency", 1.0))
    smoothness = _clamp01(raw_metrics.get("smoothness", 1.0))
    recovery = _clamp01(raw_metrics.get("recovery_from_disturbance", raw_metrics.get("slip_recovery_success", 0.0)))
    collision_rate = _clamp01(raw_metrics.get("collision_rate", 0.0))
    force_violation_rate = _clamp01(raw_metrics.get("force_violation_rate", 0.0))
    safety_penalty = _clamp01(raw_metrics.get("safety_penalty", max(collision_rate, force_violation_rate) * 0.25))
    regression_penalty = _clamp01(raw_metrics.get("regression_penalty", 0.0))

    total_score = (
        0.25 * task_success
        + 0.15 * task_progress
        + 0.15 * grasp_stability
        + 0.15 * placement_accuracy
        + 0.10 * generated_success
        + 0.08 * energy_efficiency
        + 0.07 * smoothness
        + 0.05 * recovery
        - safety_penalty
        - regression_penalty
    )

    return ScoreBreakdown(
        policy_id=policy_id,
        total_score=max(0.0, min(1.0, total_score)),
        command_tracking=task_progress,
        survival_no_fall=task_success,
        stability=grasp_stability,
        generated_scenario_success=generated_success,
        gait_quality=placement_accuracy,
        energy_efficiency=energy_efficiency,
        smoothness=smoothness,
        recovery_from_disturbance=recovery,
        safety_penalty=safety_penalty,
        regression_penalty=regression_penalty,
        safety_passed=bool(raw_metrics.get("safety_passed", collision_rate < 0.2 and force_violation_rate < 0.1)),
        base_success=_clamp01(raw_metrics.get("base_success", task_success)),
        eval_seed_count=int(raw_metrics.get("eval_seed_count", 0)),
        reward_hacking_detected=bool(raw_metrics.get("reward_hacking_detected", False)),
    )


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))

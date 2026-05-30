"""Deterministic locomotion scenario generation and frontier classification."""

from __future__ import annotations

from adapters.base import ExperimentHistory
from core.schemas import ScenarioSpec


def generate_locomotion_scenarios(history: ExperimentHistory) -> list[ScenarioSpec]:
    """Generate a compact scenario batch aimed at the current learning frontier."""

    failure_text = " ".join(str(report) for report in history.failure_reports).lower()
    seen_ids = {str(row.get("scenario_id")) for row in history.scenario_matrix if row.get("scenario_id")}

    candidates = [
        _low_friction(),
        _side_push(),
        _slope_up(),
        _rough_heightfield(),
        _payload(),
        _motor_delay(),
    ]

    prioritized: list[ScenarioSpec] = []
    if "push" in failure_text:
        prioritized.append(_side_push())
    if "rough" in failure_text or "toe_drag" in failure_text:
        prioritized.append(_rough_heightfield())
    if "slip" in failure_text:
        prioritized.append(_low_friction())

    merged = prioritized + candidates
    unique: list[ScenarioSpec] = []
    used: set[str] = set()
    for scenario in merged:
        if scenario.scenario_id in used or scenario.scenario_id in seen_ids:
            continue
        used.add(scenario.scenario_id)
        unique.append(scenario)
    return unique[:4]


def classify_frontier(success_rate: float) -> str:
    if success_rate > 0.90:
        return "too_easy"
    if 0.30 <= success_rate <= 0.80:
        return "learning_frontier"
    if 0.10 <= success_rate < 0.30:
        return "too_hard"
    if 0.0 <= success_rate < 0.10:
        return "too_hard"
    return "invalid"


def _low_friction() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="low_friction_walk_v001",
        task_family="locomotion",
        difficulty=0.25,
        terrain={"type": "flat", "friction_range": [0.35, 0.7]},
        disturbances={},
        robot_variation={},
        evaluation={"num_episodes": 64, "success_condition": "walk_10m_without_falling"},
    )


def _side_push() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="side_push_recovery_v001",
        task_family="locomotion",
        difficulty=0.35,
        terrain={"type": "flat", "friction_range": [0.65, 1.2]},
        disturbances={"push_impulse_probability": 0.05, "push_force_range_n": [20, 80]},
        robot_variation={},
        evaluation={"num_episodes": 64, "success_condition": "recover_after_side_push"},
    )


def _slope_up() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="uphill_walk_v001",
        task_family="locomotion",
        difficulty=0.40,
        terrain={"type": "slope", "slope_range_deg": [3, 8], "friction_range": [0.65, 1.2]},
        disturbances={},
        robot_variation={},
        evaluation={"num_episodes": 64, "success_condition": "walk_10m_without_falling"},
    )


def _rough_heightfield() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="rough_heightfield_walk_v001",
        task_family="locomotion",
        difficulty=0.50,
        terrain={"type": "rough", "height_noise_m": 0.04, "slope_range_deg": [-6, 6]},
        disturbances={},
        robot_variation={},
        evaluation={"num_episodes": 64, "success_condition": "walk_10m_without_falling"},
    )


def _payload() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="payload_walk_v001",
        task_family="locomotion",
        difficulty=0.55,
        terrain={"type": "flat", "friction_range": [0.65, 1.2]},
        disturbances={},
        robot_variation={"payload_mass_kg": [1.0, 3.0]},
        evaluation={"num_episodes": 64, "success_condition": "walk_10m_without_falling"},
    )


def _motor_delay() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="motor_delay_walk_v001",
        task_family="locomotion",
        difficulty=0.60,
        terrain={"type": "flat", "friction_range": [0.65, 1.2]},
        disturbances={},
        robot_variation={"action_delay_steps": [1, 3], "motor_strength_scale": [0.85, 1.0]},
        evaluation={"num_episodes": 64, "success_condition": "walk_10m_without_falling"},
    )


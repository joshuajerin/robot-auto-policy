"""Deterministic locomotion scenario generation and frontier classification."""

from __future__ import annotations

import json
import re
from typing import Any

from adapters.base import ExperimentHistory
from core.schemas import ScenarioSpec


SCENARIO_SEQUENCE = [
    "flat_velocity_randomization",
    "low_friction_walk",
    "side_push_recovery",
    "uphill_walk",
    "downhill_walk",
    "rough_heightfield_walk",
    "payload_walk",
    "motor_delay_walk",
    "motor_strength_variation",
    "mixed_terrain_pushes",
]


def generate_locomotion_scenarios(history: ExperimentHistory, batch_size: int = 4) -> list[ScenarioSpec]:
    """Generate a compact scenario batch aimed at the current learning frontier.

    The generator is deliberately deterministic for hackathon repeatability:
    it seeds the scenario bank, then mutates previous scenarios based on held-out
    success rates. That gives the AutoResearch loop a real environment frontier
    without needing an LLM to invent arbitrary physics parameters.
    """

    seen_ids = _seen_scenario_ids(history)
    latest = _latest_scenario_results(history)
    proposals: list[ScenarioSpec] = []

    if not latest:
        proposals.extend(_failure_conditioned_scenarios(history, seen_ids))
        proposals.extend(_initial_frontier_batch(seen_ids | {scenario.scenario_id for scenario in proposals}))
    else:
        proposals.extend(_mutate_from_frontier(latest, seen_ids))
        proposals.extend(_failure_conditioned_scenarios(history, seen_ids | {scenario.scenario_id for scenario in proposals}))
    proposals.extend(_coverage_backfill(seen_ids | {scenario.scenario_id for scenario in proposals}))

    unique: list[ScenarioSpec] = []
    used: set[str] = set()
    for scenario in proposals:
        if scenario.scenario_id in used or scenario.scenario_id in seen_ids:
            continue
        used.add(scenario.scenario_id)
        unique.append(scenario)
    return unique[:batch_size]


def classify_frontier(success_rate: float) -> str:
    if success_rate > 0.90:
        return "too_easy"
    if 0.30 <= success_rate <= 0.80:
        return "learning_frontier"
    if 0.0 <= success_rate < 0.30:
        return "too_hard"
    return "invalid"


def _initial_frontier_batch(seen_ids: set[str]) -> list[ScenarioSpec]:
    return [
        _make_scenario("flat_velocity_randomization", 0.15, seen_ids=seen_ids),
        _make_scenario("low_friction_walk", 0.25, seen_ids=seen_ids),
        _make_scenario("side_push_recovery", 0.35, seen_ids=seen_ids),
        _make_scenario("uphill_walk", 0.40, seen_ids=seen_ids),
    ]


def _mutate_from_frontier(latest: dict[str, dict[str, Any]], seen_ids: set[str]) -> list[ScenarioSpec]:
    proposals: list[ScenarioSpec] = []
    expanded_parent_ids = {
        str(row.get("parent_scenario_id"))
        for row in latest.values()
        if row.get("parent_scenario_id")
    }
    for scenario_id, row in sorted(latest.items(), key=lambda item: float(item[1].get("difficulty") or 0.0)):
        if scenario_id in expanded_parent_ids:
            continue
        success = _coerce_float(row.get("success_rate"))
        difficulty = _coerce_float(row.get("difficulty"))
        kind = _scenario_kind(scenario_id)
        if kind not in SCENARIO_SEQUENCE:
            continue

        classification = classify_frontier(success)
        if classification == "too_easy":
            proposals.append(
                _make_scenario(
                    kind,
                    min(1.0, difficulty + 0.12),
                    parent_scenario_id=scenario_id,
                    seen_ids=seen_ids,
                )
            )
        elif classification == "learning_frontier":
            proposals.append(
                _make_scenario(
                    _next_kind(kind),
                    min(1.0, difficulty + 0.08),
                    parent_scenario_id=scenario_id,
                    seen_ids=seen_ids,
                )
            )
        elif classification == "too_hard":
            proposals.append(
                _make_scenario(
                    kind,
                    max(0.05, difficulty - 0.10),
                    parent_scenario_id=scenario_id,
                    seen_ids=seen_ids,
                )
            )
    return proposals


def _failure_conditioned_scenarios(history: ExperimentHistory, seen_ids: set[str]) -> list[ScenarioSpec]:
    failure_text = " ".join(_jsonish(report) for report in history.failure_reports).lower()
    proposals: list[ScenarioSpec] = []
    if "push" in failure_text:
        proposals.append(_make_scenario("side_push_recovery", 0.38, seen_ids=seen_ids))
    if "rough" in failure_text or "toe_drag" in failure_text:
        proposals.append(_make_scenario("rough_heightfield_walk", 0.48, seen_ids=seen_ids))
    if "slip" in failure_text:
        proposals.append(_make_scenario("low_friction_walk", 0.30, seen_ids=seen_ids))
    if "energy" in failure_text or "oscillatory" in failure_text:
        proposals.append(_make_scenario("motor_delay_walk", 0.45, seen_ids=seen_ids))
    return proposals


def _coverage_backfill(seen_ids: set[str]) -> list[ScenarioSpec]:
    proposals: list[ScenarioSpec] = []
    for index, kind in enumerate(SCENARIO_SEQUENCE):
        proposals.append(_make_scenario(kind, min(0.95, 0.15 + index * 0.08), seen_ids=seen_ids))
    return proposals


def _make_scenario(
    kind: str,
    difficulty: float,
    *,
    parent_scenario_id: str | None = None,
    seen_ids: set[str] | None = None,
) -> ScenarioSpec:
    seen_ids = seen_ids or set()
    difficulty = round(max(0.0, min(1.0, difficulty)), 3)
    scenario_id = _next_scenario_id(kind, seen_ids)
    terrain: dict[str, Any] = {"type": "flat", "friction_range": [0.65, 1.2]}
    disturbances: dict[str, Any] = {}
    robot_variation: dict[str, Any] = {}
    success_condition = "walk_10m_without_falling"

    if kind == "flat_velocity_randomization":
        terrain = {"type": "flat", "friction_range": [0.65, 1.2]}
    elif kind == "low_friction_walk":
        low = round(max(0.2, 0.65 - difficulty * 0.75), 3)
        terrain = {"type": "flat", "friction_range": [low, 0.95]}
    elif kind == "side_push_recovery":
        force_hi = int(45 + difficulty * 140)
        terrain = {"type": "flat", "friction_range": [0.6, 1.2]}
        disturbances = {
            "push_impulse_probability": round(0.02 + difficulty * 0.12, 3),
            "push_force_range_n": [20, force_hi],
        }
        success_condition = "recover_after_side_push"
    elif kind == "uphill_walk":
        slope = round(3 + difficulty * 10, 2)
        terrain = {"type": "slope", "slope_range_deg": [3, slope], "friction_range": [0.65, 1.2]}
    elif kind == "downhill_walk":
        slope = round(3 + difficulty * 10, 2)
        terrain = {"type": "slope", "slope_range_deg": [-slope, -3], "friction_range": [0.65, 1.2]}
    elif kind == "rough_heightfield_walk":
        terrain = {
            "type": "rough",
            "height_noise_m": round(0.015 + difficulty * 0.10, 3),
            "slope_range_deg": [round(-2 - difficulty * 9, 2), round(2 + difficulty * 9, 2)],
            "friction_range": [0.55, 1.2],
        }
    elif kind == "payload_walk":
        terrain = {"type": "flat", "friction_range": [0.65, 1.2]}
        robot_variation = {"payload_mass_kg": [0.5, round(0.75 + difficulty * 5.0, 2)]}
    elif kind == "motor_delay_walk":
        robot_variation = {
            "action_delay_steps": [1, max(1, int(1 + difficulty * 5))],
            "motor_strength_scale": [round(max(0.65, 1.0 - difficulty * 0.25), 3), 1.0],
        }
    elif kind == "motor_strength_variation":
        robot_variation = {
            "motor_strength_scale": [round(max(0.55, 1.0 - difficulty * 0.35), 3), round(1.0 + difficulty * 0.15, 3)]
        }
    elif kind == "mixed_terrain_pushes":
        terrain = {
            "type": "rough",
            "height_noise_m": round(0.02 + difficulty * 0.08, 3),
            "slope_range_deg": [round(-2 - difficulty * 8, 2), round(2 + difficulty * 8, 2)],
            "friction_range": [round(max(0.35, 0.65 - difficulty * 0.35), 3), 1.2],
        }
        disturbances = {
            "push_impulse_probability": round(0.02 + difficulty * 0.08, 3),
            "push_force_range_n": [20, int(50 + difficulty * 120)],
        }
        robot_variation = {"action_delay_steps": [0, max(1, int(difficulty * 3))]}

    return ScenarioSpec(
        scenario_id=scenario_id,
        parent_scenario_id=parent_scenario_id,
        task_family="locomotion",
        difficulty=difficulty,
        terrain=terrain,
        disturbances=disturbances,
        robot_variation=robot_variation,
        evaluation={"num_episodes": 64, "success_condition": success_condition},
    )


def _latest_scenario_results(history: ExperimentHistory) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in history.scenario_matrix:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            continue
        if row.get("success_rate") is None:
            latest.setdefault(scenario_id, row)
        else:
            latest[scenario_id] = row
    return latest


def _seen_scenario_ids(history: ExperimentHistory) -> set[str]:
    return {str(row.get("scenario_id")) for row in history.scenario_matrix if row.get("scenario_id")}


def _next_scenario_id(kind: str, seen_ids: set[str]) -> str:
    prefix = f"{kind}_v"
    highest = 0
    for scenario_id in seen_ids:
        if not scenario_id.startswith(prefix):
            continue
        match = re.search(r"_v(\d+)$", scenario_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{kind}_v{highest + 1:03d}"


def _scenario_kind(scenario_id: str) -> str:
    return re.sub(r"_v\d+$", "", scenario_id)


def _next_kind(kind: str) -> str:
    try:
        index = SCENARIO_SEQUENCE.index(kind)
    except ValueError:
        return SCENARIO_SEQUENCE[0]
    return SCENARIO_SEQUENCE[min(len(SCENARIO_SEQUENCE) - 1, index + 1)]


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _jsonish(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)

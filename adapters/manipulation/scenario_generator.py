"""Deterministic manipulation scenario generation."""

from __future__ import annotations

import json
import re
from typing import Any

from adapters.base import ExperimentHistory
from core.schemas import ScenarioSpec


SCENARIO_SEQUENCE = [
    "single_cube_move",
    "target_pose_place",
    "cluttered_target_interaction",
    "distractor_occlusion",
    "low_friction_object",
    "heavy_object_move",
    "narrow_bin_place",
    "mixed_clutter_generalization",
]


def generate_manipulation_scenarios(history: ExperimentHistory, batch_size: int = 4) -> list[ScenarioSpec]:
    seen_ids = _seen_scenario_ids(history)
    latest = _latest_scenario_results(history)
    proposals: list[ScenarioSpec] = []

    if not latest:
        proposals.extend(_failure_conditioned_scenarios(history, seen_ids))
        proposals.extend(_initial_batch(seen_ids | {scenario.scenario_id for scenario in proposals}))
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


def classify_manipulation_frontier(success_rate: float) -> str:
    if success_rate > 0.90:
        return "too_easy"
    if 0.30 <= success_rate <= 0.80:
        return "learning_frontier"
    if 0.0 <= success_rate < 0.30:
        return "too_hard"
    return "invalid"


def _initial_batch(seen_ids: set[str]) -> list[ScenarioSpec]:
    return [
        _make_scenario("single_cube_move", 0.10, seen_ids=seen_ids),
        _make_scenario("target_pose_place", 0.22, seen_ids=seen_ids),
        _make_scenario("cluttered_target_interaction", 0.34, seen_ids=seen_ids),
        _make_scenario("distractor_occlusion", 0.42, seen_ids=seen_ids),
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

        classification = classify_manipulation_frontier(success)
        if classification == "too_easy":
            proposals.append(
                _make_scenario(kind, min(1.0, difficulty + 0.14), parent_scenario_id=scenario_id, seen_ids=seen_ids)
            )
        elif classification == "learning_frontier":
            proposals.append(
                _make_scenario(_next_kind(kind), min(1.0, difficulty + 0.10), parent_scenario_id=scenario_id, seen_ids=seen_ids)
            )
        elif classification == "too_hard":
            proposals.append(
                _make_scenario(kind, max(0.05, difficulty - 0.12), parent_scenario_id=scenario_id, seen_ids=seen_ids)
            )
    return proposals


def _failure_conditioned_scenarios(history: ExperimentHistory, seen_ids: set[str]) -> list[ScenarioSpec]:
    failure_text = " ".join(_jsonish(report) for report in history.failure_reports).lower()
    proposals: list[ScenarioSpec] = []
    if "contact" in failure_text or "slip" in failure_text:
        proposals.append(_make_scenario("low_friction_object", 0.38, seen_ids=seen_ids))
    if "occlusion" in failure_text or "clutter" in failure_text:
        proposals.append(_make_scenario("distractor_occlusion", 0.46, seen_ids=seen_ids))
    if "placement" in failure_text or "miss" in failure_text:
        proposals.append(_make_scenario("target_pose_place", 0.36, seen_ids=seen_ids))
    if "mass" in failure_text or "force" in failure_text:
        proposals.append(_make_scenario("heavy_object_move", 0.44, seen_ids=seen_ids))
    return proposals


def _coverage_backfill(seen_ids: set[str]) -> list[ScenarioSpec]:
    return [
        _make_scenario(kind, min(0.95, 0.10 + index * 0.10), seen_ids=seen_ids)
        for index, kind in enumerate(SCENARIO_SEQUENCE)
    ]


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
    workspace = _workspace(difficulty)
    objects = _objects_for_kind(kind, difficulty)
    task_graph = ["balance_before_reach", "approach_target", "establish_end_effector_contact", "secure_target"]
    evaluation: dict[str, Any] = {
        "num_episodes": 64,
        "success_condition": "h1_secure_and_move_target_10cm",
        "max_steps": 240,
    }
    robot_variation: dict[str, Any] = {}
    disturbances: dict[str, Any] = {}

    if kind == "target_pose_place":
        task_graph.append("place_at_goal")
        evaluation["success_condition"] = "h1_place_target_within_5cm"
        evaluation["placement_tolerance_m"] = round(max(0.025, 0.07 - difficulty * 0.04), 3)
    elif kind == "cluttered_target_interaction":
        task_graph = [
            "balance_before_reach",
            "segment_target",
            "approach_target",
            "avoid_clutter",
            "secure_target",
            "move_target",
        ]
        evaluation["success_condition"] = "h1_move_target_without_clutter_collision"
    elif kind == "distractor_occlusion":
        task_graph = ["balance_before_reach", "move_occluder_if_needed", "approach_target", "secure_target", "move_target"]
        evaluation["success_condition"] = "h1_recover_visible_target_and_move"
    elif kind == "low_friction_object":
        disturbances = {"object_surface_friction": [round(max(0.15, 0.55 - difficulty * 0.35), 3), 0.9]}
        evaluation["success_condition"] = "h1_move_without_object_slip"
    elif kind == "heavy_object_move":
        robot_variation = {"arm_contact_force_scale": [0.8, 1.0], "balance_margin_scale": [0.9, 1.0]}
        disturbances = {"object_mass_scale": [1.0, round(1.2 + difficulty * 3.0, 2)]}
        evaluation["success_condition"] = "h1_move_heavy_object_without_force_violation"
    elif kind == "narrow_bin_place":
        task_graph = [
            "balance_before_reach",
            "approach_target",
            "secure_target",
            "move_target",
            "place_inside_bin",
        ]
        evaluation["success_condition"] = "h1_place_target_inside_bin"
        evaluation["bin_clearance_m"] = round(max(0.015, 0.08 - difficulty * 0.06), 3)
    elif kind == "mixed_clutter_generalization":
        task_graph = [
            "balance_before_reach",
            "select_target",
            "avoid_clutter",
            "secure_target",
            "move_target",
            "place_at_goal",
        ]
        evaluation["success_condition"] = "h1_move_and_place_target_in_mixed_clutter"
        disturbances = {
            "object_mass_scale": [0.8, round(1.0 + difficulty * 2.2, 2)],
            "object_surface_friction": [round(max(0.2, 0.65 - difficulty * 0.35), 3), 1.2],
        }

    return ScenarioSpec(
        scenario_id=scenario_id,
        parent_scenario_id=parent_scenario_id,
        task_family="manipulation",
        robot_id="unitree_h1",
        difficulty=difficulty,
        workspace=workspace,
        objects=objects,
        task_graph=task_graph,
        dataset={
            "dataset_id": "robogenesis_h1_manipulation_primitives_v1",
            "asset_manifest": "assets/manipulation_objects/manifest.json",
            "robot_spec": "assets/h1_robot_spec.json",
        },
        disturbances=disturbances,
        robot_variation=robot_variation,
        evaluation=evaluation,
    )


def _workspace(difficulty: float) -> dict[str, Any]:
    return {
        "type": "tabletop",
        "robot_id": "unitree_h1",
        "table_asset": "assets/manipulation_objects/usd/lab_table.usda",
        "bounds_m": [[0.25, -0.45, 0.0], [0.85, 0.45, 0.45]],
        "goal_region_m": {
            "center": [round(0.55 + difficulty * 0.10, 3), 0.22, 0.04],
            "radius": round(max(0.035, 0.09 - difficulty * 0.04), 3),
        },
    }


def _objects_for_kind(kind: str, difficulty: float) -> list[dict[str, Any]]:
    target = {
        "object_id": "target_cube",
        "role": "target",
        "asset_path": "assets/manipulation_objects/usd/target_cube.usda",
        "size_m": [0.055, 0.055, 0.055],
        "mass_kg": round(0.08 + difficulty * 0.18, 3),
        "initial_pose": "sample_on_table",
    }
    objects = [target]
    distractor_count = 0
    if kind in {"cluttered_target_interaction", "mixed_clutter_generalization"}:
        distractor_count = max(2, int(2 + difficulty * 5))
    elif kind == "distractor_occlusion":
        distractor_count = max(1, int(1 + difficulty * 3))
        objects.append(
            {
                "object_id": "occluding_block",
                "role": "occluder",
                "asset_path": "assets/manipulation_objects/usd/distractor_block.usda",
                "initial_relation": "partially_occluding_target",
            }
        )
    elif kind == "narrow_bin_place":
        objects.append(
            {
                "object_id": "goal_bin",
                "role": "receptacle",
                "asset_path": "assets/manipulation_objects/usd/narrow_bin.usda",
                "initial_pose": "fixed_goal_region",
            }
        )

    for index in range(distractor_count):
        objects.append(
            {
                "object_id": f"distractor_{index + 1:02d}",
                "role": "distractor",
                "asset_path": "assets/manipulation_objects/usd/cylinder_can.usda"
                if index % 2
                else "assets/manipulation_objects/usd/distractor_block.usda",
                "initial_pose": "sample_nonoverlap_on_table",
            }
        )
    return objects


def _latest_scenario_results(history: ExperimentHistory) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in history.scenario_matrix:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            continue
        if row.get("task_family") not in (None, "manipulation"):
            continue
        if row.get("success_rate") is None:
            latest.setdefault(scenario_id, row)
        else:
            latest[scenario_id] = row
    return latest


def _seen_scenario_ids(history: ExperimentHistory) -> set[str]:
    return {
        str(row.get("scenario_id"))
        for row in history.scenario_matrix
        if row.get("scenario_id") and row.get("task_family") in (None, "manipulation")
    }


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

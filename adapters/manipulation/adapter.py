"""Manipulation task adapter.

This adapter makes tabletop pick/place a first-class task family without
claiming the Phase-1 H1 locomotion runner can train it yet. It defines the
bounded patch surface, scenario frontier, scoring, and failure taxonomy that a
future Isaac Lab manipulation runner will consume.
"""

from __future__ import annotations

from typing import Any

from adapters.base import ExperimentHistory, Rollout
from core.schemas import FailureReport, ScenarioSpec, ScoreBreakdown, TaskSpec


class ManipulationAdapter:
    task_family = "manipulation"

    _allowed_patch_paths = [
        "configs/manipulation/rewards.yaml",
        "configs/manipulation/curriculum.yaml",
        "configs/manipulation/domain_randomization.yaml",
        "configs/manipulation/objects.yaml",
        "configs/manipulation/ppo.yaml",
    ]

    failure_taxonomy = [
        "missed_grasp",
        "unstable_grasp",
        "object_slip",
        "placement_miss",
        "collision_with_clutter",
        "workspace_limit",
        "timeout_no_progress",
        "excessive_force",
        "excessive_energy",
        "fails_under_occlusion",
        "fails_with_mass_variation",
        "reward_hacking_contact",
    ]

    def default_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="tabletop_pick_place_v1",
            task_family="manipulation",
            objective="pick a target object from a table and place it at a requested goal pose without collisions",
            base_env="Isaac-Lift-Cube-Franka-v0",
            commands={
                "target_object": "sampled_from_scenario",
                "goal_pose": "sampled_from_scenario",
                "success_condition": "lift_and_place_target",
            },
            style_targets={
                "low_collision": True,
                "stable_grasp": True,
                "smooth_end_effector_motion": True,
                "minimal_object_slip": True,
            },
        )

    def allowed_patch_paths(self) -> list[str]:
        return list(self._allowed_patch_paths)

    def generate_scenarios(self, history: ExperimentHistory) -> list[ScenarioSpec]:
        from adapters.manipulation.scenario_generator import generate_manipulation_scenarios

        return generate_manipulation_scenarios(history)

    def score(self, raw_metrics: dict[str, Any]) -> ScoreBreakdown:
        from adapters.manipulation.metrics import score_manipulation

        return score_manipulation(raw_metrics)

    def diagnose(self, rollouts: list[Rollout], metrics: dict[str, Any]) -> FailureReport:
        from adapters.manipulation.failure_diagnosis import diagnose_manipulation_failure

        return diagnose_manipulation_failure(rollouts, metrics)

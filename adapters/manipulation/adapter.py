"""H1-only manipulation task adapter.

Isaac Lab 2.0.x ships stock manipulation tasks for fixed arms such as Franka
and UR10, but not for Unitree H1. This adapter keeps the AutoResearch surface
H1-first by targeting a custom H1 tabletop manipulation environment instead of
silently falling back to a non-H1 robot.
"""

from __future__ import annotations

from typing import Any

from adapters.base import ExperimentHistory, Rollout
from core.schemas import FailureReport, ScenarioSpec, ScoreBreakdown, TaskSpec


class ManipulationAdapter:
    task_family = "manipulation"
    robot_id = "unitree_h1"
    robot_spec = "assets/h1_robot_spec.json"
    base_env = "RoboGenesis-H1-Tabletop-Manipulation-v0"

    _allowed_patch_paths = [
        "configs/manipulation/rewards.yaml",
        "configs/manipulation/curriculum.yaml",
        "configs/manipulation/domain_randomization.yaml",
        "configs/manipulation/objects.yaml",
        "configs/manipulation/ppo.yaml",
    ]

    failure_taxonomy = [
        "missed_contact",
        "unstable_contact",
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
            task_id="h1_tabletop_manipulation_v1",
            task_family="manipulation",
            objective=(
                "Use Unitree H1 whole-body control and arm contact to move a tabletop object "
                "from the robot's left side of the table to a right-side goal region without falling."
            ),
            base_env=self.base_env,
            robot_id=self.robot_id,
            robot_spec=self.robot_spec,
            requires_custom_env=True,
            commands={
                "target_object": "sampled_from_scenario",
                "start_region": "left_side_of_table",
                "goal_pose": "right_side_of_table",
                "success_condition": "h1_transfer_target_cube_left_to_right_on_table",
                "allowed_stock_fallbacks": [],
            },
            style_targets={
                "whole_body_balance": True,
                "upright_torso": True,
                "low_collision": True,
                "stable_end_effector_contact": True,
                "smooth_arm_motion": True,
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

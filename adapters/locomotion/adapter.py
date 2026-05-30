"""Locomotion task adapter.

The adapter owns locomotion defaults and the allowed training surface. Isaac
Lab-specific execution lives in the Modal runner; this class defines what the
research loop is allowed to ask for.
"""

from __future__ import annotations

from typing import Any

from adapters.base import ExperimentHistory, Rollout
from core.schemas import FailureReport, ScenarioSpec, ScoreBreakdown, TaskSpec


class LocomotionAdapter:
    task_family = "locomotion"

    _allowed_patch_paths = [
        "configs/locomotion/rewards.yaml",
        "configs/locomotion/curriculum.yaml",
        "configs/locomotion/domain_randomization.yaml",
        "configs/locomotion/actuators.yaml",
        "configs/locomotion/ppo.yaml",
        "configs/locomotion/terrain.yaml",
    ]

    failure_taxonomy = [
        "fall_forward",
        "fall_backward",
        "fall_sideways",
        "foot_slip",
        "toe_drag",
        "knee_collapse",
        "torso_pitch_instability",
        "command_tracking_failure",
        "excessive_energy",
        "joint_limit_abuse",
        "fails_on_push",
        "fails_on_slope",
        "fails_on_rough_terrain",
        "gait_asymmetry",
        "stuck_or_no_progress",
        "oscillatory_actions",
    ]

    def default_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="human_like_locomotion_v1",
            task_family="locomotion",
            objective="walk forward while maintaining stable humanoid gait",
            base_env="Isaac-Velocity-Flat-H1-v0",
            commands={
                "linear_velocity_x": [0.4, 1.2],
                "linear_velocity_y": [0.0, 0.0],
                "yaw_velocity": [-0.2, 0.2],
            },
            style_targets={
                "cadence_hz": 1.8,
                "stride_symmetry": True,
                "upright_torso": True,
                "low_foot_slip": True,
            },
        )

    def allowed_patch_paths(self) -> list[str]:
        return list(self._allowed_patch_paths)

    def generate_scenarios(self, history: ExperimentHistory) -> list[ScenarioSpec]:
        from adapters.locomotion.scenario_generator import generate_locomotion_scenarios

        return generate_locomotion_scenarios(history)

    def score(self, raw_metrics: dict[str, Any]) -> ScoreBreakdown:
        from adapters.locomotion.metrics import score_locomotion

        return score_locomotion(raw_metrics)

    def diagnose(self, rollouts: list[Rollout], metrics: dict[str, Any]) -> FailureReport:
        from adapters.locomotion.failure_diagnosis import diagnose_locomotion_failure

        return diagnose_locomotion_failure(rollouts, metrics)


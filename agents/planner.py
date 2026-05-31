"""Research planner with a deterministic fallback policy.

The production path can replace this module's body with an OpenAI structured
output call that returns the same `PatchSpec`. The local fallback is useful for
tests, demos, and offline development.
"""

from __future__ import annotations

import json
from typing import Any

from adapters.locomotion.adapter import LocomotionAdapter
from adapters.manipulation.adapter import ManipulationAdapter
from core.schemas import PatchSpec


def propose_patch(context: dict[str, Any]) -> PatchSpec:
    task_family = str((context.get("task_spec") or {}).get("task_family") or "locomotion")
    if task_family == "manipulation":
        return propose_manipulation_patch(context)
    return propose_locomotion_patch(context)


def propose_locomotion_patch(context: dict[str, Any]) -> PatchSpec:
    adapter = LocomotionAdapter()
    failure = _primary_failure(context)

    if failure in {"torso_pitch_instability", "fall_forward", "fall_backward", "fall_sideways"}:
        return PatchSpec(
            experiment_name="increase_torso_stability",
            hypothesis="The policy is losing balance because torso stabilization is underweighted.",
            allowed_files=[
                "configs/locomotion/rewards.yaml",
                "configs/locomotion/curriculum.yaml",
            ],
            patch={
                "reward_weights.torso_upright": 0.55,
                "reward_weights.stability": 0.75,
                "curriculum.roughness_start": 0.0,
                "curriculum.roughness_end": 0.035,
            },
            expected_effect="Higher survival and lower pitch/roll instability on fixed seeds.",
            risk="May make the gait conservative or slower.",
            rollback="Restore previous torso and stability weights plus roughness ramp.",
        )

    if failure in {"toe_drag", "fails_on_rough_terrain"}:
        return PatchSpec(
            experiment_name="increase_foot_clearance_for_roughness",
            hypothesis="The policy is failing rough terrain because swing feet clip the terrain.",
            allowed_files=[
                "configs/locomotion/rewards.yaml",
                "configs/locomotion/curriculum.yaml",
            ],
            patch={
                "reward_weights.foot_clearance": 0.35,
                "reward_weights.gait_symmetry": 0.25,
                "curriculum.roughness_start": 0.01,
                "curriculum.roughness_end": 0.08,
            },
            expected_effect="Better rough-terrain survival and fewer toe-drag contacts.",
            risk="Can create exaggerated high-stepping if pushed too far.",
            rollback="Revert foot clearance, gait symmetry, and roughness curriculum values.",
        )

    if failure == "foot_slip":
        return PatchSpec(
            experiment_name="improve_low_friction_contact",
            hypothesis="The policy slips because friction variation and contact penalties are too weak.",
            allowed_files=[
                "configs/locomotion/rewards.yaml",
                "configs/locomotion/domain_randomization.yaml",
            ],
            patch={
                "reward_weights.foot_slip_penalty": 0.4,
                "domain_randomization.friction_range": [0.45, 1.25],
            },
            expected_effect="Better low-friction success and lower foot slip rate.",
            risk="May reduce speed if contact becomes too conservative.",
            rollback="Restore foot-slip reward and friction randomization range.",
        )

    if failure == "fails_on_push":
        return PatchSpec(
            experiment_name="add_push_recovery_curriculum",
            hypothesis="The policy falls after side pushes because recovery is underrepresented.",
            allowed_files=[
                "configs/locomotion/rewards.yaml",
                "configs/locomotion/domain_randomization.yaml",
                "configs/locomotion/curriculum.yaml",
            ],
            patch={
                "reward_weights.recovery": 0.35,
                "domain_randomization.push_impulse_probability": 0.05,
                "domain_randomization.push_force_range_n": [20, 90],
                "curriculum.push_probability_end": 0.08,
            },
            expected_effect="Better recovery from lateral disturbances without base-task regression.",
            risk="May overfit to pushes and sacrifice smooth command tracking.",
            rollback="Restore recovery reward, push probability, and push force range.",
        )

    if failure == "excessive_energy":
        return PatchSpec(
            experiment_name="reduce_energy_spikes",
            hypothesis="The policy uses high-torque corrections that hurt energy and smoothness metrics.",
            allowed_files=["configs/locomotion/rewards.yaml"],
            patch={
                "reward_weights.energy_penalty": 0.08,
                "reward_weights.smoothness": 0.18,
            },
            expected_effect="Lower torque spikes and smoother action sequences.",
            risk="May weaken recovery from disturbances.",
            rollback="Restore energy and smoothness weights.",
        )

    return PatchSpec(
        experiment_name="increase_command_tracking_baseline",
        hypothesis="No dominant failure is isolated, so improve the base walking objective first.",
        allowed_files=adapter.allowed_patch_paths()[:1],
        patch={"reward_weights.command_tracking": 1.15},
        expected_effect="Higher fixed-scenario command tracking without touching evaluator logic.",
        risk="Could reduce robustness if tracking dominates balance.",
        rollback="Restore command tracking weight.",
    )


def propose_manipulation_patch(context: dict[str, Any]) -> PatchSpec:
    adapter = ManipulationAdapter()
    failure = _primary_failure(context)

    if failure in {"missed_grasp", "unstable_grasp"}:
        return PatchSpec(
            experiment_name="improve_grasp_acquisition",
            hypothesis="The policy is failing because target approach and stable closure are underweighted.",
            allowed_files=[
                "configs/manipulation/rewards.yaml",
                "configs/manipulation/curriculum.yaml",
            ],
            patch={
                "reward_weights.grasp_success": 1.0,
                "reward_weights.contact_quality": 0.5,
                "curriculum.target_pose_randomization_end_m": 0.12,
            },
            expected_effect="Higher grasp success with less early pose-randomization pressure.",
            risk="May overfit to easy target poses and under-train placement.",
            rollback="Restore grasp/contact weights and target pose randomization end value.",
        )

    if failure == "object_slip":
        return PatchSpec(
            experiment_name="stabilize_object_contact",
            hypothesis="Objects slip because contact stability and friction randomization are too weak.",
            allowed_files=[
                "configs/manipulation/rewards.yaml",
                "configs/manipulation/domain_randomization.yaml",
            ],
            patch={
                "reward_weights.object_stability": 0.65,
                "reward_weights.stable_lift": 0.85,
                "domain_randomization.object_friction_range": [0.35, 1.25],
            },
            expected_effect="Lower object slip under friction variation.",
            risk="May create overly slow or conservative lifting.",
            rollback="Restore object stability, stable lift, and object friction range.",
        )

    if failure == "placement_miss":
        return PatchSpec(
            experiment_name="tighten_goal_placement",
            hypothesis="Placement is failing because lift reward dominates goal-pose precision.",
            allowed_files=[
                "configs/manipulation/rewards.yaml",
                "configs/manipulation/curriculum.yaml",
            ],
            patch={
                "reward_weights.placement_accuracy": 1.0,
                "curriculum.bin_clearance_end_m": 0.035,
                "curriculum.target_pose_randomization_end_m": 0.16,
            },
            expected_effect="Higher placement accuracy without changing evaluator logic.",
            risk="May slow task completion while the policy learns precise placement.",
            rollback="Restore placement reward and placement curriculum values.",
        )

    if failure in {"collision_with_clutter", "fails_under_occlusion"}:
        return PatchSpec(
            experiment_name="add_clutter_and_occlusion_curriculum",
            hypothesis="The policy collides or loses targets because clutter/occlusion are introduced too abruptly.",
            allowed_files=[
                "configs/manipulation/rewards.yaml",
                "configs/manipulation/curriculum.yaml",
            ],
            patch={
                "reward_weights.collision_penalty": 0.8,
                "curriculum.clutter_density_end": 0.45,
                "curriculum.occlusion_probability_end": 0.35,
            },
            expected_effect="Better target recovery and fewer clutter collisions.",
            risk="May make early exploration too conservative.",
            rollback="Restore collision penalty and clutter/occlusion curriculum.",
        )

    if failure in {"fails_with_mass_variation", "excessive_force"}:
        return PatchSpec(
            experiment_name="smooth_mass_and_force_curriculum",
            hypothesis="The policy uses excessive force because mass variation is too abrupt.",
            allowed_files=[
                "configs/manipulation/rewards.yaml",
                "configs/manipulation/domain_randomization.yaml",
                "configs/manipulation/curriculum.yaml",
            ],
            patch={
                "reward_weights.force_penalty": 0.45,
                "domain_randomization.object_mass_scale": [0.8, 2.0],
                "curriculum.object_mass_scale_end": 2.0,
            },
            expected_effect="Lower force violations with more stable heavy-object lifting.",
            risk="May reduce success on the heaviest generated objects.",
            rollback="Restore force penalty and object mass ranges.",
        )

    return PatchSpec(
        experiment_name="increase_manipulation_task_completion",
        hypothesis="No dominant manipulation failure is isolated, so improve base task completion first.",
        allowed_files=adapter.allowed_patch_paths()[:1],
        patch={"reward_weights.task_completion": 1.15},
        expected_effect="Higher pick/place completion under fixed scenarios.",
        risk="Could over-prioritize completion over contact quality.",
        rollback="Restore task completion reward.",
    )


def _primary_failure(context: dict[str, Any]) -> str:
    reports = context.get("failure_reports") or []
    if not reports:
        return "command_tracking_failure"

    latest = reports[0]
    if isinstance(latest, dict):
        report_json = latest.get("failure_report_json")
        if isinstance(report_json, str):
            try:
                decoded = json.loads(report_json)
            except json.JSONDecodeError:
                decoded = {}
            if isinstance(decoded, dict) and isinstance(decoded.get("primary_failure"), str):
                return decoded["primary_failure"]
            for token in [
                "toe_drag",
                "foot_slip",
                "fails_on_push",
                "fails_on_rough_terrain",
                "torso_pitch_instability",
                "excessive_energy",
                "missed_grasp",
                "unstable_grasp",
                "object_slip",
                "placement_miss",
                "collision_with_clutter",
                "fails_under_occlusion",
                "fails_with_mass_variation",
                "excessive_force",
            ]:
                if token in report_json:
                    return token
        value = latest.get("primary_failure")
        if isinstance(value, str):
            return value
    return "command_tracking_failure"

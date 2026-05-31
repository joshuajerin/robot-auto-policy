"""Metric-first manipulation failure diagnosis."""

from __future__ import annotations

from typing import Any

from adapters.base import Rollout
from core.schemas import FailureReport


def diagnose_manipulation_failure(rollouts: list[Rollout], metrics: dict[str, Any]) -> FailureReport:
    secondary: list[str] = []
    causes: list[str] = []
    directions: list[str] = []

    grasp = float(metrics.get("grasp_success_rate", metrics.get("grasp_stability", 1.0)))
    slip = float(metrics.get("object_slip_rate", 0.0))
    placement_error = float(metrics.get("placement_error_m", 0.0))
    collision = float(metrics.get("collision_rate", 0.0))
    timeout = float(metrics.get("timeout_rate", 0.0))
    force = float(metrics.get("force_violation_rate", 0.0))
    occlusion_success = float(metrics.get("occlusion_success_rate", 1.0))
    mass_success = float(metrics.get("mass_variation_success", 1.0))
    progress = float(metrics.get("task_progress", 1.0))

    if grasp < 0.45:
        primary = "missed_grasp"
        causes.append("target approach or gripper closure reward is under-specified")
        directions.extend(["increase grasp success reward", "slow early curriculum around target pose randomization"])
    elif slip > 0.35:
        primary = "object_slip"
        causes.append("contact is not robust to friction or object geometry")
        directions.extend(["increase object stability reward", "widen object friction randomization"])
    elif placement_error > 0.07:
        primary = "placement_miss"
        causes.append("goal pose precision is weak relative to lift reward")
        directions.extend(["increase placement accuracy reward", "add staged place curriculum"])
    elif collision > 0.25:
        primary = "collision_with_clutter"
        causes.append("clutter avoidance is underrepresented")
        directions.extend(["increase collision penalty", "add clutter-density curriculum"])
    elif force > 0.15:
        primary = "excessive_force"
        causes.append("policy is using high-force contact to solve manipulation")
        directions.extend(["increase force penalty", "constrain gripper force scale"])
    elif occlusion_success < 0.5:
        primary = "fails_under_occlusion"
        causes.append("target visibility and occluder handling are not represented enough")
        directions.extend(["add occlusion curriculum", "increase target selection reward"])
    elif mass_success < 0.5:
        primary = "fails_with_mass_variation"
        causes.append("object mass randomization is outside the learned contact regime")
        directions.extend(["smooth mass randomization curriculum", "increase stable lift reward"])
    elif timeout > 0.35 or progress < 0.45:
        primary = "timeout_no_progress"
        causes.append("policy is not making reliable task progress")
        directions.extend(["increase task progress reward", "simplify early object placement"])
    else:
        primary = "unstable_grasp"
        causes.append("metrics do not isolate a single severe manipulation failure")
        directions.append("generate frontier manipulation scenarios and rerun diagnosis")

    if slip > 0.2 and primary != "object_slip":
        secondary.append("object_slip")
    if collision > 0.15 and primary != "collision_with_clutter":
        secondary.append("collision_with_clutter")
    if placement_error > 0.05 and primary != "placement_miss":
        secondary.append("placement_miss")
    if force > 0.1 and primary != "excessive_force":
        secondary.append("excessive_force")

    return FailureReport(
        primary_failure=primary,
        secondary_failures=secondary,
        evidence={
            "rollout_count": len(rollouts),
            "grasp_success_rate": grasp,
            "object_slip_rate": slip,
            "placement_error_m": placement_error,
            "collision_rate": collision,
            "timeout_rate": timeout,
            "force_violation_rate": force,
            "occlusion_success_rate": occlusion_success,
            "mass_variation_success": mass_success,
        },
        likely_causes=causes,
        suggested_research_directions=directions,
    )

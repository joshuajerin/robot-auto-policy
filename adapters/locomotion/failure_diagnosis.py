"""Metric-first locomotion failure diagnosis."""

from __future__ import annotations

from typing import Any

from adapters.base import Rollout
from core.schemas import FailureReport


def diagnose_locomotion_failure(rollouts: list[Rollout], metrics: dict[str, Any]) -> FailureReport:
    secondary: list[str] = []
    causes: list[str] = []
    directions: list[str] = []

    survival = float(metrics.get("survival_no_fall", 1.0))
    stability = float(metrics.get("stability", 1.0))
    foot_slip = float(metrics.get("foot_slip_events_per_meter", 0.0))
    foot_clearance = float(metrics.get("foot_clearance_mean", 0.08))
    push_success = float(metrics.get("push_recovery_success", 1.0))
    rough_success = float(metrics.get("rough_terrain_success", 1.0))
    energy = float(metrics.get("energy_efficiency", 1.0))
    command = float(metrics.get("command_tracking", 1.0))

    if survival < 0.6 and stability < 0.55:
        primary = "torso_pitch_instability"
        causes.append("torso stability reward or curriculum may be too weak")
        directions.extend(["increase torso upright reward", "reduce early terrain difficulty"])
    elif foot_clearance < 0.03:
        primary = "toe_drag"
        causes.append("swing foot clearance appears insufficient")
        directions.extend(["increase foot clearance reward", "add swing-phase clearance term"])
    elif foot_slip > 1.5:
        primary = "foot_slip"
        causes.append("contact behavior is not robust to friction or terrain changes")
        directions.extend(["increase foot slip penalty", "widen friction randomization"])
    elif push_success < 0.5:
        primary = "fails_on_push"
        causes.append("push recovery is underrepresented")
        directions.extend(["add push curriculum", "increase recovery reward"])
    elif rough_success < 0.5:
        primary = "fails_on_rough_terrain"
        causes.append("roughness curriculum may be too abrupt")
        directions.extend(["smooth roughness ramp", "increase foot clearance reward"])
    elif command < 0.55:
        primary = "command_tracking_failure"
        causes.append("velocity command reward may be too weak")
        directions.append("increase command tracking reward")
    elif energy < 0.45:
        primary = "excessive_energy"
        causes.append("policy uses high-torque or jerky actions")
        directions.extend(["increase energy penalty", "increase smoothness reward"])
    else:
        primary = "stuck_or_no_progress"
        causes.append("metrics do not isolate a single severe failure")
        directions.append("generate frontier scenarios and rerun diagnosis")

    if foot_slip > 1.0 and primary != "foot_slip":
        secondary.append("foot_slip")
    if push_success < 0.7 and primary != "fails_on_push":
        secondary.append("fails_on_push")
    if rough_success < 0.7 and primary != "fails_on_rough_terrain":
        secondary.append("fails_on_rough_terrain")
    if energy < 0.5 and primary != "excessive_energy":
        secondary.append("excessive_energy")

    return FailureReport(
        primary_failure=primary,
        secondary_failures=secondary,
        evidence={
            "rollout_count": len(rollouts),
            "survival_no_fall": survival,
            "stability": stability,
            "foot_slip_events_per_meter": foot_slip,
            "foot_clearance_mean": foot_clearance,
            "push_recovery_success": push_success,
            "rough_terrain_success": rough_success,
        },
        likely_causes=causes,
        suggested_research_directions=directions,
    )


"""Parallel specialist subagents for correction planning.

These are deterministic local subagents today, with the same `PatchSpec`
contract that the OpenAI planner uses. They can run in parallel after the
system ingests metrics/failure reports and needs candidate corrective actions.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from adapters.base import ExperimentHistory
from adapters.locomotion import LocomotionAdapter
from agents.planner import propose_locomotion_patch
from core.patch_validator import validate_patch_spec
from core.schemas import FailureReport, PatchSpec, ScenarioSpec


@dataclass(frozen=True)
class SubagentCandidate:
    agent_name: str
    patch: PatchSpec | None = None
    scenarios: list[ScenarioSpec] = field(default_factory=list)
    rationale: str = ""
    score: float = 0.0
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["patch"] = self.patch.to_dict() if self.patch else None
        data["scenarios"] = [scenario.to_dict() for scenario in self.scenarios]
        return data


@dataclass(frozen=True)
class CorrectionPlan:
    primary_patch: PatchSpec | None
    patch_candidates: list[SubagentCandidate]
    scenario_candidates: list[ScenarioSpec]
    failure_report: FailureReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_patch": self.primary_patch.to_dict() if self.primary_patch else None,
            "patch_candidates": [candidate.to_dict() for candidate in self.patch_candidates],
            "scenario_candidates": [scenario.to_dict() for scenario in self.scenario_candidates],
            "failure_report": self.failure_report.to_dict(),
        }


def run_parallel_correction_subagents(
    *,
    failure_report: FailureReport,
    metrics: dict[str, Any],
    history: ExperimentHistory | None = None,
) -> CorrectionPlan:
    history = history or ExperimentHistory()
    context = {
        "failure_reports": [failure_report.to_dict()],
        "metrics": metrics,
        "recent_experiments": history.recent_experiments,
        "scenario_matrix": history.scenario_matrix,
    }
    subagents: list[Callable[[], SubagentCandidate]] = [
        lambda: _planner_subagent(context),
        lambda: _stability_subagent(failure_report),
        lambda: _contact_subagent(failure_report),
        lambda: _terrain_curriculum_subagent(failure_report),
        lambda: _efficiency_subagent(failure_report),
        lambda: _scenario_subagent(history),
    ]

    candidates: list[SubagentCandidate] = []
    with ThreadPoolExecutor(max_workers=len(subagents)) as executor:
        futures = [executor.submit(agent) for agent in subagents]
        for future in as_completed(futures):
            candidates.append(_validate_candidate(future.result(), failure_report))

    patch_candidates = sorted(
        [candidate for candidate in candidates if candidate.patch is not None],
        key=lambda candidate: (not candidate.validation_errors, candidate.score),
        reverse=True,
    )
    scenario_candidates: list[ScenarioSpec] = []
    seen_scenarios: set[str] = set()
    for candidate in candidates:
        for scenario in candidate.scenarios:
            if scenario.scenario_id not in seen_scenarios:
                scenario_candidates.append(scenario)
                seen_scenarios.add(scenario.scenario_id)

    primary_patch = next((candidate.patch for candidate in patch_candidates if not candidate.validation_errors), None)
    return CorrectionPlan(
        primary_patch=primary_patch,
        patch_candidates=patch_candidates,
        scenario_candidates=scenario_candidates,
        failure_report=failure_report,
    )


def _validate_candidate(candidate: SubagentCandidate, failure_report: FailureReport) -> SubagentCandidate:
    if candidate.patch is None:
        return candidate
    result = validate_patch_spec(candidate.patch)
    score = candidate.score + _failure_match_bonus(candidate.patch, failure_report)
    return SubagentCandidate(
        agent_name=candidate.agent_name,
        patch=candidate.patch,
        scenarios=candidate.scenarios,
        rationale=candidate.rationale,
        score=score,
        validation_errors=result.errors,
    )


def _failure_match_bonus(patch: PatchSpec, failure_report: FailureReport) -> float:
    keys = set(patch.patch)
    primary = failure_report.primary_failure
    if primary == "fails_on_push" and {"reward_weights.recovery", "domain_randomization.push_impulse_probability"} & keys:
        return 2.0
    if primary in {"toe_drag", "fails_on_rough_terrain"} and "reward_weights.foot_clearance" in keys:
        return 2.0
    if primary == "foot_slip" and "reward_weights.foot_slip_penalty" in keys:
        return 2.0
    if primary == "excessive_energy" and "reward_weights.energy_penalty" in keys:
        return 2.0
    if primary in {"torso_pitch_instability", "fall_forward", "fall_backward", "fall_sideways"} and "reward_weights.torso_upright" in keys:
        return 2.0
    return 0.0


def _planner_subagent(context: dict[str, Any]) -> SubagentCandidate:
    patch = propose_locomotion_patch(context)
    return SubagentCandidate(
        agent_name="planner_generalist",
        patch=patch,
        rationale="General planner proposes one bounded patch from failure context.",
        score=1.0,
    )


def _stability_subagent(failure_report: FailureReport) -> SubagentCandidate:
    patch = PatchSpec(
        experiment_name="stability_agent_torso_recovery",
        hypothesis="The policy needs stronger torso and survival shaping before harder scenarios.",
        allowed_files=["configs/locomotion/rewards.yaml", "configs/locomotion/curriculum.yaml"],
        patch={
            "reward_weights.torso_upright": 0.6,
            "reward_weights.stability": 0.8,
            "curriculum.roughness_start": 0.0,
            "curriculum.roughness_end": 0.04,
        },
        expected_effect="Reduce falls and improve base survival without changing evaluator logic.",
        risk="May become overly conservative and reduce speed.",
        rollback="Revert torso, stability, and roughness curriculum values.",
    )
    return SubagentCandidate(
        agent_name="stability_specialist",
        patch=patch,
        rationale=f"Targets primary failure {failure_report.primary_failure} with stability shaping.",
        score=0.8,
    )


def _contact_subagent(failure_report: FailureReport) -> SubagentCandidate:
    patch = PatchSpec(
        experiment_name="contact_agent_slip_clearance",
        hypothesis="Contact instability can be reduced by improving foot clearance and slip penalties.",
        allowed_files=["configs/locomotion/rewards.yaml", "configs/locomotion/domain_randomization.yaml"],
        patch={
            "reward_weights.foot_clearance": 0.35,
            "reward_weights.foot_slip_penalty": 0.4,
            "domain_randomization.friction_range": [0.45, 1.25],
        },
        expected_effect="Lower foot slip and fewer terrain contacts during swing.",
        risk="May create high stepping or slow the gait.",
        rollback="Revert foot clearance, slip penalty, and friction randomization.",
    )
    return SubagentCandidate(
        agent_name="contact_specialist",
        patch=patch,
        rationale=f"Targets primary failure {failure_report.primary_failure} with contact robustness.",
        score=0.7,
    )


def _terrain_curriculum_subagent(failure_report: FailureReport) -> SubagentCandidate:
    patch = PatchSpec(
        experiment_name="terrain_agent_frontier_curriculum",
        hypothesis="The policy should train on a smoother roughness and slope frontier.",
        allowed_files=["configs/locomotion/curriculum.yaml", "configs/locomotion/terrain.yaml"],
        patch={
            "curriculum.roughness_start": 0.01,
            "curriculum.roughness_end": 0.08,
            "curriculum.slope_end_deg": 6.0,
            "terrain.type": "rough",
            "terrain.height_noise_m": 0.04,
        },
        expected_effect="Improve generated rough/slope scenario survival.",
        risk="May regress flat task if introduced too aggressively.",
        rollback="Restore flat terrain and previous curriculum bounds.",
    )
    return SubagentCandidate(
        agent_name="terrain_curriculum_specialist",
        patch=patch,
        rationale=f"Targets primary failure {failure_report.primary_failure} with frontier terrain.",
        score=0.6,
    )


def _efficiency_subagent(failure_report: FailureReport) -> SubagentCandidate:
    patch = PatchSpec(
        experiment_name="efficiency_agent_smooth_energy",
        hypothesis="Energy spikes and action jerk should be reduced while preserving base tracking.",
        allowed_files=["configs/locomotion/rewards.yaml"],
        patch={
            "reward_weights.energy_penalty": 0.08,
            "reward_weights.smoothness": 0.18,
        },
        expected_effect="Improve energy efficiency and smoothness metrics.",
        risk="May weaken recovery from disturbances.",
        rollback="Restore energy and smoothness reward weights.",
    )
    return SubagentCandidate(
        agent_name="efficiency_specialist",
        patch=patch,
        rationale=f"Targets primary failure {failure_report.primary_failure} with smoothness shaping.",
        score=0.5,
    )


def _scenario_subagent(history: ExperimentHistory) -> SubagentCandidate:
    scenarios = LocomotionAdapter().generate_scenarios(history)
    return SubagentCandidate(
        agent_name="scenario_specialist",
        scenarios=scenarios,
        rationale="Generates frontier locomotion scenarios for the next eval/training curriculum.",
        score=0.4,
    )


def failure_report_from_json(path: Path) -> FailureReport:
    payload = json.loads(path.read_text())
    return FailureReport(
        primary_failure=payload["primary_failure"],
        secondary_failures=list(payload.get("secondary_failures", [])),
        evidence=dict(payload.get("evidence", {})),
        likely_causes=list(payload.get("likely_causes", [])),
        suggested_research_directions=list(payload.get("suggested_research_directions", [])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-json", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text()) if args.metrics_json else {}
    plan = run_parallel_correction_subagents(
        failure_report=failure_report_from_json(Path(args.failure_json)),
        metrics=metrics,
    )
    output = json.dumps(plan.to_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()

"""RoboGenesis AutoResearch controller.

The dry-run path exercises the full decision loop without requiring Isaac Lab.
Modal execution can be plugged in by replacing `_simulate_candidate_metrics`
with a call to `modal_runner.modal_app.train_and_eval_job`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from adapters.base import ExperimentHistory
from adapters.locomotion import LocomotionAdapter
from agents.planner import propose_locomotion_patch
from agents.reviewer import review_policy_candidate
from agents.scenario_agent import generate_scenarios
from adapters.locomotion.scenario_generator import classify_frontier
from core.experiment_db import ExperimentDB
from core.patch_validator import apply_yaml_patch
from core.schemas import FailureReport, PatchSpec, ScoreBreakdown


BASELINE_RAW_METRICS: dict[str, Any] = {
    "policy_id": "baseline_0000",
    "command_tracking": 0.61,
    "survival_no_fall": 0.72,
    "stability": 0.55,
    "generated_scenario_success": 0.20,
    "gait_quality": 0.39,
    "energy_efficiency": 0.48,
    "smoothness": 0.44,
    "recovery_from_disturbance": 0.28,
    "base_success": 0.72,
    "eval_seed_count": 8,
    "safety_passed": True,
    "push_recovery_success": 0.35,
    "rough_terrain_success": 0.25,
    "foot_slip_events_per_meter": 1.2,
    "foot_clearance_mean": 0.04,
}

RAINDROP_EVENT_NAME = "robogenesis-autoresearch-dry-run"
_RAINDROP_INITIALIZED = False


def run_dry_research_loop(repo_root: Path, db_path: Path, experiments: int) -> list[dict[str, Any]]:
    adapter = LocomotionAdapter()
    db = ExperimentDB(db_path)
    best = adapter.score(BASELINE_RAW_METRICS)
    best_policy_id = best.policy_id

    db.insert_policy(
        policy_id=best.policy_id,
        parent_policy_id=None,
        checkpoint_path="artifacts/baseline_0000/checkpoint.pt",
        metrics=best,
        accepted=True,
    )
    baseline_failure = adapter.diagnose([], BASELINE_RAW_METRICS)
    db.insert_failure_report("baseline", best.policy_id, baseline_failure)

    summaries: list[dict[str, Any]] = []
    for index in range(experiments):
        experiment_id = f"exp_{index + 1:04d}"
        history = ExperimentHistory(
            recent_experiments=db.recent_experiments(10),
            scenario_matrix=db.scenario_matrix(),
            failure_reports=db.recent_failures(5),
        )
        context = {
            "task_spec": adapter.default_task_spec().to_dict(),
            "best_policy": best.to_dict(),
            "recent_experiments": history.recent_experiments,
            "scenario_matrix": history.scenario_matrix,
            "failure_reports": history.failure_reports,
        }
        patch = propose_locomotion_patch(context)
        changes = apply_yaml_patch(patch, repo_root=repo_root, dry_run=True)

        scenarios = generate_scenarios(adapter, history)
        db.insert_scenarios(scenarios)

        candidate_policy_id = f"policy_{index + 1:04d}"
        raw_metrics = _simulate_candidate_metrics(candidate_policy_id, best, patch, index)
        candidate = adapter.score(raw_metrics)
        failure_report = adapter.diagnose([], raw_metrics)
        review = review_policy_candidate(best, candidate)

        db.insert_experiment(
            experiment_id=experiment_id,
            parent_policy_id=best_policy_id,
            patch=patch,
            status="accepted" if review.accepted else "rejected",
            score_before=best.total_score,
            score_after=candidate.total_score,
            accepted=review.accepted,
            modal_job_id=None,
        )
        db.insert_policy(
            policy_id=candidate_policy_id,
            parent_policy_id=best_policy_id,
            checkpoint_path=f"artifacts/{experiment_id}/checkpoint.pt",
            metrics=candidate,
            accepted=review.accepted,
        )
        db.insert_failure_report(experiment_id, candidate_policy_id, failure_report)

        scenario_results = []
        for scenario in scenarios:
            success = _scenario_success_for_candidate(candidate, scenario.difficulty)
            frontier_status = classify_frontier(success)
            failure_modes = [] if success >= 0.5 else [failure_report.primary_failure]
            db.insert_scenario_eval(
                scenario_id=scenario.scenario_id,
                policy_id=candidate_policy_id,
                success_rate=success,
                score=success,
                failure_modes=failure_modes,
                rollout_video_path=f"artifacts/{experiment_id}/{scenario.scenario_id}.mp4",
            )
            scenario_results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "parent_scenario_id": scenario.parent_scenario_id,
                    "difficulty": scenario.difficulty,
                    "success_rate": success,
                    "frontier_status": frontier_status,
                    "terrain": scenario.terrain,
                    "disturbances": scenario.disturbances,
                    "robot_variation": scenario.robot_variation,
                    "failure_modes": failure_modes,
                }
            )

        summaries.append(
            {
                "experiment_id": experiment_id,
                "patch": patch.to_dict(),
                "config_changes": changes,
                "score_before": round(best.total_score, 4),
                "score_after": round(candidate.total_score, 4),
                "accepted": review.accepted,
                "review_reasons": review.reasons,
                "primary_failure": failure_report.primary_failure,
                "generated_scenarios": scenario_results,
            }
        )

        if review.accepted:
            best = candidate
            best_policy_id = candidate_policy_id

    db.close()
    return summaries


def run_traced_dry_research_loop(
    repo_root: Path,
    db_path: Path,
    experiments: int,
    *,
    event_id: str | None = None,
    user_id: str = "local-user",
    convo_id: str | None = None,
    source: str = "cli",
    model: str | None = None,
    trace_input: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the local AutoResearch loop and mirror one interaction to Workshop."""

    raindrop = _init_raindrop()
    input_payload = trace_input or {
        "experiments": experiments,
        "dbPath": str(db_path),
        "source": source,
    }
    properties: dict[str, Any] = {
        "experiments": experiments,
        "dbPath": str(db_path),
        "source": source,
        "repoRoot": str(repo_root),
    }
    if model:
        properties["model"] = model
    if event_id:
        properties["replayRunId"] = event_id

    interaction = None
    if raindrop is not None:
        interaction = raindrop.begin(
            user_id=user_id,
            event=RAINDROP_EVENT_NAME,
            event_id=event_id,
            input=json.dumps(input_payload, sort_keys=True, default=str),
            convo_id=convo_id,
            properties=properties,
        )

    try:
        summaries = run_dry_research_loop(repo_root, db_path, experiments)
    except Exception as exc:
        if interaction is not None:
            interaction.finish(
                output=json.dumps(
                    {"status": "error", "message": str(exc)},
                    sort_keys=True,
                    default=str,
                ),
                properties={**properties, "status": "error"},
            )
        if raindrop is not None:
            raindrop.flush()
        raise

    if interaction is not None:
        interaction.finish(
            output=json.dumps(
                {"status": "done", "summaries": summaries},
                sort_keys=True,
                default=str,
            ),
            properties={**properties, "status": "done"},
        )
        raindrop.flush()

    return summaries


def _init_raindrop() -> Any | None:
    global _RAINDROP_INITIALIZED

    try:
        import raindrop.analytics as raindrop
    except ImportError:
        return None

    if not _RAINDROP_INITIALIZED:
        api_key = os.environ.get("RAINDROP_WRITE_KEY") or None
        local_workshop_url = os.environ.get("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/")
        tracing_enabled = bool(api_key)
        raindrop.init(
            api_key=api_key,
            local_workshop_url=local_workshop_url,
            tracing_enabled=tracing_enabled,
            auto_instrument=tracing_enabled,
        )
        _RAINDROP_INITIALIZED = True

    return raindrop


def _simulate_candidate_metrics(
    policy_id: str,
    best: ScoreBreakdown,
    patch: PatchSpec,
    index: int,
) -> dict[str, Any]:
    metrics = best.to_dict()
    metrics["policy_id"] = policy_id
    metrics["eval_seed_count"] = 8
    metrics["safety_passed"] = True
    metrics["reward_hacking_detected"] = False

    keys = set(patch.patch)
    if "reward_weights.recovery" in keys:
        metrics["survival_no_fall"] = _bump(metrics["survival_no_fall"], 0.04)
        metrics["generated_scenario_success"] = _bump(metrics["generated_scenario_success"], 0.12)
        metrics["recovery_from_disturbance"] = _bump(metrics["recovery_from_disturbance"], 0.25)
        metrics["push_recovery_success"] = 0.65
    if "reward_weights.foot_clearance" in keys:
        metrics["generated_scenario_success"] = _bump(metrics["generated_scenario_success"], 0.10)
        metrics["gait_quality"] = _bump(metrics["gait_quality"], 0.08)
        metrics["stability"] = _bump(metrics["stability"], 0.04)
        metrics["rough_terrain_success"] = 0.62
        metrics["foot_clearance_mean"] = 0.055
    if "reward_weights.foot_slip_penalty" in keys:
        metrics["generated_scenario_success"] = _bump(metrics["generated_scenario_success"], 0.08)
        metrics["gait_quality"] = _bump(metrics["gait_quality"], 0.05)
        metrics["foot_slip_events_per_meter"] = 0.6
    if "reward_weights.energy_penalty" in keys:
        metrics["energy_efficiency"] = _bump(metrics["energy_efficiency"], 0.12)
        metrics["smoothness"] = _bump(metrics["smoothness"], 0.08)
    if "reward_weights.torso_upright" in keys:
        metrics["stability"] = _bump(metrics["stability"], 0.10)
        metrics["survival_no_fall"] = _bump(metrics["survival_no_fall"], 0.05)
    if "reward_weights.command_tracking" in keys:
        metrics["command_tracking"] = _bump(metrics["command_tracking"], 0.04 + 0.01 * index)

    return metrics


def _scenario_success_for_candidate(candidate: ScoreBreakdown, difficulty: float) -> float:
    raw = candidate.generated_scenario_success + 0.35 - difficulty * 0.45
    return round(max(0.0, min(1.0, raw)), 3)


def _bump(value: Any, amount: float) -> float:
    return max(0.0, min(1.0, float(value) + amount))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run deterministic local loop without Isaac Lab.")
    parser.add_argument("--experiments", type=int, default=3)
    parser.add_argument("--db", default="artifacts/research.db")
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit("Only --dry-run is implemented locally; use modal_runner for Isaac execution.")

    repo_root = Path(__file__).resolve().parents[1]
    summaries = run_traced_dry_research_loop(
        repo_root,
        Path(args.db),
        args.experiments,
        source="cli",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

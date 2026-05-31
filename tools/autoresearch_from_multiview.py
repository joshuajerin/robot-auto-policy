"""Create a Modal AutoResearch iteration from Isaac multiview rollout videos.

The multiview renderer writes a compact ``multiview_autoresearch_input.json``
next to the front/side/diagonal videos. This tool treats that artifact as a
research observation: it summarizes the videos, diagnoses rollout metrics, builds
a bounded PatchSpec, generates frontier scenarios, and optionally submits a new
Phase-1 Modal training job.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ExperimentHistory
from adapters.locomotion.adapter import LocomotionAdapter
from core.experiment_db import ExperimentDB
from core.patch_validator import apply_yaml_patch, validate_patch_spec
from core.schemas import FailureReport, PatchSpec, ScenarioSpec
from eval.locomotion_score import score_metrics
from modal_runner.deployed import DEFAULT_APP_NAME
from modal_runner.phase1 import build_phase1_spec


ACTION_JERK_THRESHOLD = 0.24
TORSO_TILT_THRESHOLD = 0.30
COMMAND_ERROR_THRESHOLD = 0.30


@dataclass(frozen=True)
class MultiviewAutoResearchResult:
    experiment_ids: list[str]
    parent_policy_id: str
    patch: dict[str, Any]
    failure_report: dict[str, Any]
    scenario_ids: list[str]
    score_before: float | None
    spec_paths: list[str]
    context_path: str
    patch_path: str
    modal_call_ids: list[str]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiview-input", required=True)
    parser.add_argument("--raw-metrics", default="")
    parser.add_argument("--db", default="artifacts/research.db")
    parser.add_argument("--phase1-config", default="configs/locomotion/phase1_h1.yaml")
    parser.add_argument("--output-dir", default="artifacts/autoresearch_multiview")
    parser.add_argument("--experiment-prefix", default="autoresearch_h1_multiview_smooth")
    parser.add_argument("--parent-policy-id", default="")
    parser.add_argument("--seed-start", type=int, default=2300)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--video-length", type=int, default=240)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--env", default="")
    args = parser.parse_args()

    result = run_from_multiview(args)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def run_from_multiview(args: argparse.Namespace) -> MultiviewAutoResearchResult:
    multiview_path = (REPO_ROOT / args.multiview_input).resolve() if not Path(args.multiview_input).is_absolute() else Path(args.multiview_input)
    multiview_data = json.loads(multiview_path.read_text())
    multiview_summary = _summarize_multiview(multiview_data, multiview_path)
    parent_policy_id = args.parent_policy_id or _infer_parent_policy_id(multiview_summary, multiview_path)

    raw_metrics_path = _resolve_raw_metrics(args.raw_metrics, parent_policy_id, multiview_path)
    score_before = _score_before(raw_metrics_path)
    failure_report = _diagnose_multiview(multiview_summary)
    patch = _propose_multiview_patch(
        failure_report=failure_report,
        max_iterations=args.max_iterations,
        num_envs=args.num_envs,
    )
    validate_patch_spec(patch).raise_for_errors()
    config_changes = apply_yaml_patch(patch, repo_root=REPO_ROOT, dry_run=True)

    scenarios = _generate_scenarios(failure_report)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": "tools.autoresearch_from_multiview",
        "parent_policy_id": parent_policy_id,
        "multiview_input_path": str(multiview_path),
        "raw_metrics_path": str(raw_metrics_path) if raw_metrics_path else None,
        "score_before": score_before,
        "multiview": multiview_summary,
        "failure_report": failure_report.to_dict(),
        "patch": patch.to_dict(),
        "config_changes": config_changes,
        "generated_scenarios": [scenario.to_dict() for scenario in scenarios],
        "note": (
            "Videos are passed as research inputs for diagnosis and lineage. "
            "The locked Phase-1 Modal runner directly applies PPO controls; "
            "reward/curriculum patches are recorded until the Isaac adapter maps "
            "them into task-specific environment configs."
        ),
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_prefix = _safe_name(args.experiment_prefix)
    context_path = output_dir / f"{safe_prefix}_{timestamp}_context.json"
    patch_path = output_dir / f"{safe_prefix}_{timestamp}_patch.json"
    _write_json(context_path, context)
    _write_json(patch_path, patch.to_dict())

    specs: list[dict[str, Any]] = []
    spec_paths: list[str] = []
    experiment_ids: list[str] = []
    for index in range(max(1, args.num_runs)):
        seed = args.seed_start + index
        experiment_id = f"{safe_prefix}_{timestamp}_seed-{seed}"
        spec = _build_modal_spec(
            phase1_config=Path(args.phase1_config),
            experiment_id=experiment_id,
            seed=seed,
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            video_length=args.video_length,
            parent_policy_id=parent_policy_id,
            score_before=score_before,
            patch=patch,
            config_changes=config_changes,
            scenarios=scenarios,
            multiview_summary=multiview_summary,
            context_path=context_path,
        )
        spec_path = output_dir / f"{experiment_id}.json"
        _write_json(spec_path, spec)
        specs.append(spec)
        spec_paths.append(str(spec_path))
        experiment_ids.append(experiment_id)

    modal_call_ids: list[str] = []
    status = "proposed"
    if args.submit:
        modal_call_ids = _submit_specs(specs, args.app_name, args.env or None)
        status = "running"

    _record_research_memory(
        db_path=Path(args.db),
        experiment_ids=experiment_ids,
        parent_policy_id=parent_policy_id,
        patch=patch,
        status=status,
        score_before=score_before,
        call_ids=modal_call_ids,
        scenarios=scenarios,
        failure_report=failure_report,
    )

    return MultiviewAutoResearchResult(
        experiment_ids=experiment_ids,
        parent_policy_id=parent_policy_id,
        patch=patch.to_dict(),
        failure_report=failure_report.to_dict(),
        scenario_ids=[scenario.scenario_id for scenario in scenarios],
        score_before=score_before,
        spec_paths=spec_paths,
        context_path=str(context_path),
        patch_path=str(patch_path),
        modal_call_ids=modal_call_ids,
        status=status,
    )


def _summarize_multiview(data: dict[str, Any], input_path: Path) -> dict[str, Any]:
    view_summaries: list[dict[str, Any]] = []
    for view in data.get("views", []):
        video_path = _local_video_path(input_path, view.get("primary_video_path"))
        view_summaries.append(
            {
                "view": view.get("view"),
                "policy_id": view.get("policy_id"),
                "seed": view.get("seed"),
                "camera": view.get("camera"),
                "frame_count": view.get("frame_count"),
                "done_step": view.get("done_step"),
                "mean_action_jerk": view.get("mean_action_jerk"),
                "mean_action_l2": view.get("mean_action_l2"),
                "mean_command_error_xy": view.get("mean_command_error_xy"),
                "mean_torso_tilt_xy": view.get("mean_torso_tilt_xy"),
                "max_torso_tilt_xy": view.get("max_torso_tilt_xy"),
                "mean_reward": view.get("mean_reward"),
                "primary_video_path": view.get("primary_video_path"),
                "local_video_path": str(video_path) if video_path else None,
                "video_exists": bool(video_path and video_path.exists()),
            }
        )
    return {
        "aggregate": data.get("aggregate", {}),
        "primary_failures": data.get("primary_failures", []),
        "view_count": data.get("view_count", len(view_summaries)),
        "views": view_summaries,
        "diagnoses": _compact_diagnoses(data.get("diagnoses", [])),
    }


def _compact_diagnoses(diagnoses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for diagnosis in diagnoses:
        compact.append(
            {
                "primary_failure": diagnosis.get("primary_failure"),
                "secondary_failures": diagnosis.get("secondary_failures", []),
                "evidence": diagnosis.get("evidence", {}),
                "suggested_research_directions": diagnosis.get("suggested_research_directions", []),
            }
        )
    return compact


def _local_video_path(input_path: Path, container_path: str | None) -> Path | None:
    if not container_path:
        return None
    rel = container_path.lstrip("/")
    if rel.startswith("artifacts/"):
        return input_path.parent / rel
    return input_path.parent / rel


def _infer_parent_policy_id(summary: dict[str, Any], input_path: Path) -> str:
    for view in summary.get("views", []):
        if view.get("policy_id"):
            return str(view["policy_id"])
    return input_path.parent.parent.name


def _resolve_raw_metrics(raw_metrics_arg: str, parent_policy_id: str, multiview_path: Path) -> Path | None:
    if raw_metrics_arg:
        path = Path(raw_metrics_arg)
        return (REPO_ROOT / path).resolve() if not path.is_absolute() else path
    candidate = multiview_path.parent.parent / "modal_artifacts" / parent_policy_id / "raw_eval_metrics.json"
    return candidate if candidate.exists() else None


def _score_before(raw_metrics_path: Path | None) -> float | None:
    if not raw_metrics_path or not raw_metrics_path.exists():
        return None
    return float(score_metrics(json.loads(raw_metrics_path.read_text()))["total_score"])


def _diagnose_multiview(summary: dict[str, Any]) -> FailureReport:
    aggregate = summary.get("aggregate", {})
    any_done = bool(aggregate.get("any_done", False))
    max_torso_tilt = _float(aggregate.get("max_torso_tilt_xy"))
    mean_action_jerk = _float(aggregate.get("mean_action_jerk"))
    mean_command_error = _float(aggregate.get("mean_command_error_xy"))
    secondary: list[str] = []

    if any_done or max_torso_tilt > TORSO_TILT_THRESHOLD:
        primary = "torso_pitch_instability"
        secondary.extend(["fall_risk", "base_stability_margin"])
        likely_causes = [
            "torso stabilization or balance terms are too weak",
            "training budget has not converged enough on stable posture",
        ]
        directions = ["increase torso/stability emphasis", "keep base flat walking regression checks locked"]
    elif mean_command_error > COMMAND_ERROR_THRESHOLD:
        primary = "command_tracking_failure"
        secondary.extend(["velocity_tracking_error"])
        likely_causes = [
            "command tracking is still undertrained relative to balance",
            "short training budget leaves speed control inconsistent",
        ]
        directions = ["increase command tracking pressure", "extend PPO budget before harder scenarios"]
    elif mean_action_jerk > ACTION_JERK_THRESHOLD or _has_secondary(summary, "oscillatory_actions"):
        primary = "oscillatory_actions"
        secondary.extend(["action_jerk", "energy_spikes"])
        likely_causes = [
            "the policy is viable but still produces abrupt action changes",
            "smoothness and energy terms should be strengthened before scaling terrain difficulty",
        ]
        directions = ["increase smoothness and energy penalties", "evaluate motor-delay and push scenarios next"]
    else:
        primary = "no_major_failure_detected"
        secondary.extend(["needs_harder_scenarios"])
        likely_causes = ["flat-ground rollout is too easy to expose the next weakness"]
        directions = ["generate harder push, slope, roughness, and low-friction scenarios"]

    return FailureReport(
        primary_failure=primary,
        secondary_failures=secondary,
        evidence={
            "aggregate": aggregate,
            "view_count": summary.get("view_count"),
            "video_inputs": [
                {
                    "view": view.get("view"),
                    "local_video_path": view.get("local_video_path"),
                    "video_exists": view.get("video_exists"),
                }
                for view in summary.get("views", [])
            ],
        },
        likely_causes=likely_causes,
        suggested_research_directions=directions,
    )


def _has_secondary(summary: dict[str, Any], label: str) -> bool:
    for diagnosis in summary.get("diagnoses", []):
        if label in diagnosis.get("secondary_failures", []):
            return True
    return False


def _propose_multiview_patch(
    *,
    failure_report: FailureReport,
    max_iterations: int,
    num_envs: int,
) -> PatchSpec:
    if failure_report.primary_failure == "torso_pitch_instability":
        return PatchSpec(
            experiment_name="multiview_stabilize_torso_and_continue_training",
            hypothesis="The multiview rollout shows elevated torso tilt or fall risk, so the next run should emphasize posture and continue PPO convergence.",
            allowed_files=["configs/locomotion/rewards.yaml", "configs/locomotion/ppo.yaml"],
            patch={
                "reward_weights.torso_upright": 0.55,
                "reward_weights.stability": 0.75,
                "ppo.max_iterations": max_iterations,
                "ppo.num_envs": num_envs,
            },
            expected_effect="Higher survival, lower torso tilt, and fewer near-fall recovery spikes in the next multiview render.",
            risk="May produce a conservative gait if posture dominates velocity tracking.",
            rollback="Restore torso/stability weights and the previous PPO budget.",
        )
    if failure_report.primary_failure == "command_tracking_failure":
        return PatchSpec(
            experiment_name="multiview_improve_command_tracking",
            hypothesis="The multiview rollout survives but has command tracking error, so continue PPO while slightly emphasizing velocity tracking.",
            allowed_files=["configs/locomotion/rewards.yaml", "configs/locomotion/ppo.yaml"],
            patch={
                "reward_weights.command_tracking": 1.15,
                "reward_weights.stability": 0.70,
                "ppo.max_iterations": max_iterations,
                "ppo.num_envs": num_envs,
            },
            expected_effect="Lower velocity error without regressing survival on flat walking.",
            risk="Could make gait less robust if tracking dominates balance.",
            rollback="Restore command tracking/stability weights and previous PPO budget.",
        )
    if failure_report.primary_failure == "oscillatory_actions":
        return PatchSpec(
            experiment_name="multiview_reduce_action_jerk",
            hypothesis="The multiview rollout is stable but has elevated action jerk, so strengthen smoothness/energy pressure and continue PPO.",
            allowed_files=["configs/locomotion/rewards.yaml", "configs/locomotion/ppo.yaml"],
            patch={
                "reward_weights.smoothness": 0.18,
                "reward_weights.energy_penalty": 0.07,
                "reward_weights.torso_upright": 0.48,
                "ppo.max_iterations": max_iterations,
                "ppo.num_envs": num_envs,
            },
            expected_effect="Smoother gait in side/front/diagonal views with lower action jerk and no fall regression.",
            risk="May weaken fast recovery if smoothness is over-weighted.",
            rollback="Restore smoothness, energy, torso-upright weights, and previous PPO budget.",
        )
    return PatchSpec(
        experiment_name="multiview_frontier_scenario_probe",
        hypothesis="The flat multiview rollout has no major failure, so keep the policy budget fixed and probe harder generated scenarios next.",
        allowed_files=["configs/locomotion/ppo.yaml"],
        patch={
            "ppo.max_iterations": max_iterations,
            "ppo.num_envs": num_envs,
        },
        expected_effect="More converged baseline for push, slope, roughness, and low-friction scenario evaluation.",
        risk="Longer training can still overfit flat terrain if scenario evaluation is not enabled.",
        rollback="Restore the previous PPO budget.",
    )


def _generate_scenarios(failure_report: FailureReport) -> list[ScenarioSpec]:
    adapter = LocomotionAdapter()
    history = ExperimentHistory(
        failure_reports=[
            {
                "policy_id": "multiview_parent",
                "failure_report_json": json.dumps(failure_report.to_dict(), sort_keys=True),
            }
        ]
    )
    return adapter.generate_scenarios(history)


def _build_modal_spec(
    *,
    phase1_config: Path,
    experiment_id: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
    video_length: int,
    parent_policy_id: str,
    score_before: float | None,
    patch: PatchSpec,
    config_changes: dict[str, dict[str, Any]],
    scenarios: list[ScenarioSpec],
    multiview_summary: dict[str, Any],
    context_path: Path,
) -> dict[str, Any]:
    config_path = (REPO_ROOT / phase1_config).resolve() if not phase1_config.is_absolute() else phase1_config
    overrides = SimpleNamespace(
        task="",
        runner="",
        device="",
        num_envs=num_envs,
        max_iterations=max_iterations,
        seed=seed,
        video_length=video_length,
        style_context="",
        motion_context="",
    )
    spec = build_phase1_spec(config_path, experiment_id, overrides)
    _project_patch_to_modal_spec(spec, patch)
    spec["autoresearch"] = {
        "controller": "tools.autoresearch_from_multiview",
        "created_at": datetime.now(UTC).isoformat(),
        "parent_policy_id": parent_policy_id,
        "score_before": score_before,
        "patch": patch.to_dict(),
        "config_changes": config_changes,
        "generated_scenarios": [scenario.to_dict() for scenario in scenarios],
        "multiview_context": multiview_summary,
        "multiview_context_path": str(context_path),
        "quick_iteration": {
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "seed": seed,
            "video_length": video_length,
        },
    }
    return spec


def _project_patch_to_modal_spec(spec: dict[str, Any], patch: PatchSpec) -> None:
    train = spec.setdefault("train", {})
    ppo_overrides: dict[str, Any] = {}
    for key, value in patch.patch.items():
        if key == "ppo.max_iterations":
            train["max_iterations"] = int(value)
        elif key == "ppo.num_envs":
            train["num_envs"] = int(value)
        elif key.startswith("ppo."):
            ppo_overrides[key.removeprefix("ppo.")] = value
    if ppo_overrides:
        train["ppo_overrides"] = ppo_overrides


def _record_research_memory(
    *,
    db_path: Path,
    experiment_ids: list[str],
    parent_policy_id: str,
    patch: PatchSpec,
    status: str,
    score_before: float | None,
    call_ids: list[str],
    scenarios: list[ScenarioSpec],
    failure_report: FailureReport,
) -> None:
    db = ExperimentDB(REPO_ROOT / db_path if not db_path.is_absolute() else db_path)
    try:
        db.insert_scenarios(scenarios)
        for index, experiment_id in enumerate(experiment_ids):
            db.insert_experiment(
                experiment_id=experiment_id,
                parent_policy_id=parent_policy_id,
                patch=patch,
                status=status,
                score_before=score_before,
                score_after=None,
                accepted=False,
                modal_job_id=call_ids[index] if index < len(call_ids) else None,
            )
            db.insert_failure_report(experiment_id, parent_policy_id, failure_report)
    finally:
        db.close()


def _submit_specs(specs: list[dict[str, Any]], app_name: str, environment_name: str | None) -> list[str]:
    from modal_runner.deployed import submit_phase1_specs_to_deployed

    return submit_phase1_specs_to_deployed(specs, app_name=app_name, environment_name=environment_name)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")
    return safe or "autoresearch_multiview"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()

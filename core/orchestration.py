"""AutoResearch orchestration for quick Modal-backed locomotion iterations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from adapters.base import ExperimentHistory, TaskAdapter
from adapters.registry import get_task_adapter
from agents.openai_planner import propose_patch_with_openai
from agents.planner import propose_patch
from agents.scenario_agent import generate_scenarios
from core.autoresearch_loop import BASELINE_RAW_METRICS
from core.experiment_db import ExperimentDB
from core.patch_validator import apply_yaml_patch, validate_patch_spec
from core.raindrop_trace import RaindropRun, now_ms
from core.schemas import PatchSpec, ScenarioSpec, ScoreBreakdown
from modal_runner.deployed import DEFAULT_APP_NAME, submit_phase1_specs_to_deployed
from modal_runner.phase1 import build_phase1_spec


@dataclass(frozen=True)
class OrchestrationConfig:
    repo_root: Path
    db_path: Path
    task_family: str = "locomotion"
    phase1_config: Path = Path("configs/locomotion/phase1_h1.yaml")
    output_dir: Path = Path("artifacts/autoresearch_specs")
    experiment_prefix: str = "autoresearch_h1"
    experiments: int = 1
    seed_start: int = 900
    num_envs: int = 512
    max_iterations: int = 10
    video_length: int = 60
    use_openai: bool = False
    submit: bool = False
    app_name: str = DEFAULT_APP_NAME
    environment_name: str | None = None


@dataclass(frozen=True)
class OrchestrationStep:
    experiment_id: str
    parent_policy_id: str
    patch: dict[str, Any]
    config_changes: dict[str, dict[str, Any]]
    scenario_ids: list[str]
    modal_spec_path: str
    modal_call_id: str | None
    status: str
    score_before: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModalReconcileResult:
    experiment_id: str
    db_status: str
    ready_for_review: bool
    review_blockers: list[str]
    checkpoint_count: int
    video_count: int
    primary_video_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_orchestration(config: OrchestrationConfig) -> list[OrchestrationStep]:
    """Prepare or submit bounded AutoResearch experiments.

    The controller deliberately keeps the evaluator and Modal runner locked. It
    only produces structured experiment specs and records lineage. In quick mode
    the Modal spec is capped by ``max_iterations`` and ``num_envs`` so we can
    iterate on runner failures before scaling training.
    """

    repo_root = config.repo_root.resolve()
    output_dir = (repo_root / config.output_dir).resolve() if not config.output_dir.is_absolute() else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_task_adapter(config.task_family)
    trace = RaindropRun.start(
        event_name="robogenesis-modal-orchestration",
        input_payload=_orchestration_trace_input(config),
        properties={
            "task_family": config.task_family,
            "experiments": config.experiments,
            "submit": config.submit,
            "source": "core.orchestration",
        },
    )
    db = ExperimentDB(config.db_path)
    try:
        started = now_ms()
        best = _ensure_best_policy(db, adapter)
        trace.record_task(
            "load_best_policy",
            input_payload={"task_family": adapter.task_family},
            output_payload=best.to_dict(),
            properties={"task_family": adapter.task_family, "policy_id": best.policy_id},
            start_ms=started,
        )
        steps: list[OrchestrationStep] = []

        for index in range(max(1, config.experiments)):
            seed = config.seed_start + index
            experiment_id = _experiment_id(config.experiment_prefix, index=index, seed=seed)
            history = ExperimentHistory(
                recent_experiments=db.recent_experiments(10),
                scenario_matrix=db.scenario_matrix(),
                failure_reports=db.recent_failures(5),
            )
            context = _planner_context(adapter, best, history)
            started = now_ms()
            patch = _propose_patch(context, use_openai=config.use_openai)
            validate_patch_spec(patch).raise_for_errors()
            config_changes = apply_yaml_patch(patch, repo_root=repo_root, dry_run=True)
            trace.record_task(
                "propose_and_validate_patch",
                input_payload={"experiment_id": experiment_id, "context": context},
                output_payload={"patch": patch.to_dict(), "config_changes": config_changes},
                properties={"experiment_id": experiment_id, "task_family": adapter.task_family},
                start_ms=started,
            )

            started = now_ms()
            scenarios = generate_scenarios(adapter, history)
            db.insert_scenarios(scenarios)
            trace.record_task(
                "generate_scenarios",
                input_payload={"experiment_id": experiment_id, "history": history.__dict__},
                output_payload=[scenario.to_dict() for scenario in scenarios],
                properties={
                    "experiment_id": experiment_id,
                    "task_family": adapter.task_family,
                    "scenario_count": len(scenarios),
                },
                start_ms=started,
            )

            started = now_ms()
            modal_spec = _build_experiment_spec(
                config,
                adapter=adapter,
                experiment_id=experiment_id,
                seed=seed,
                best=best,
                patch=patch,
                config_changes=config_changes,
                scenarios=scenarios,
            )
            spec_path = output_dir / f"{experiment_id}.json"
            spec_path.write_text(json.dumps(modal_spec, indent=2, sort_keys=True) + "\n")
            trace.record_task(
                "write_modal_experiment_spec",
                input_payload={"experiment_id": experiment_id, "output_dir": str(output_dir)},
                output_payload={"modal_spec_path": str(spec_path), "modal_spec": modal_spec},
                properties={"experiment_id": experiment_id, "task_family": adapter.task_family},
                start_ms=started,
            )

            call_id = None
            status = "proposed"
            if config.submit:
                started = now_ms()
                call_id = submit_phase1_specs_to_deployed(
                    [modal_spec],
                    app_name=config.app_name,
                    environment_name=config.environment_name,
                )[0]
                status = "running"
                trace.record_task(
                    "submit_modal_phase1_job",
                    input_payload={"experiment_id": experiment_id, "app_name": config.app_name},
                    output_payload={"function_call_id": call_id, "status": status},
                    properties={"experiment_id": experiment_id, "modal_call_id": call_id},
                    start_ms=started,
                )

            db.insert_experiment(
                experiment_id=experiment_id,
                parent_policy_id=best.policy_id,
                patch=patch,
                status=status,
                score_before=best.total_score,
                score_after=None,
                accepted=False,
                modal_job_id=call_id,
            )

            steps.append(
                OrchestrationStep(
                    experiment_id=experiment_id,
                    parent_policy_id=best.policy_id,
                    patch=patch.to_dict(),
                    config_changes=config_changes,
                    scenario_ids=[scenario.scenario_id for scenario in scenarios],
                    modal_spec_path=str(spec_path),
                    modal_call_id=call_id,
                    status=status,
                    score_before=best.total_score,
                )
            )

        trace.finish(
            output_payload={"status": "done", "steps": [step.to_dict() for step in steps]},
            status="done",
            properties={"step_count": len(steps)},
        )
        return steps
    except Exception as exc:
        trace.finish(
            output_payload={"status": "error", "message": str(exc)},
            status="error",
            error=exc,
        )
        raise
    finally:
        db.close()


def reconcile_modal_experiments(
    db_path: Path,
    *,
    experiment_ids: list[str] | None = None,
    volume: str = "robogenesis-runs",
) -> list[ModalReconcileResult]:
    """Poll Modal Volume artifacts and update experiment status rows.

    This does not promote policies. Promotion still requires artifact ingestion,
    scoring, and the locked reviewer. The reconcile step only marks whether a
    running job has produced enough artifacts for review.
    """

    from tools.modal_artifact_status import summarize_experiments

    db = ExperimentDB(db_path)
    try:
        if experiment_ids is None:
            experiment_ids = [str(row["id"]) for row in db.experiments_by_status("running")]
        if not experiment_ids:
            return []

        statuses = summarize_experiments(volume, experiment_ids)
        results: list[ModalReconcileResult] = []
        for status in statuses:
            db_status = "completed" if status.ready_for_review else "running"
            if status.has_render_error and not status.has_eval_metrics:
                db_status = "failed"
            db.update_experiment_status(status.experiment_id, status=db_status)
            results.append(
                ModalReconcileResult(
                    experiment_id=status.experiment_id,
                    db_status=db_status,
                    ready_for_review=status.ready_for_review,
                    review_blockers=status.review_blockers,
                    checkpoint_count=status.checkpoint_count,
                    video_count=status.video_count,
                    primary_video_path=status.primary_video_path,
                )
            )
        return results
    finally:
        db.close()


BASELINE_MANIPULATION_RAW_METRICS: dict[str, Any] = {
    "policy_id": "baseline_manipulation_0000",
    "task_success_rate": 0.32,
    "task_progress": 0.42,
    "contact_success_rate": 0.45,
    "contact_stability": 0.38,
    "placement_accuracy": 0.28,
    "generated_scenario_success": 0.10,
    "energy_efficiency": 0.70,
    "smoothness": 0.62,
    "slip_recovery_success": 0.20,
    "collision_rate": 0.18,
    "force_violation_rate": 0.06,
    "base_success": 0.32,
    "eval_seed_count": 8,
    "safety_passed": True,
    "object_slip_rate": 0.30,
    "placement_error_m": 0.09,
}


def _ensure_best_policy(db: ExperimentDB, adapter: TaskAdapter) -> ScoreBreakdown:
    prefixes = _policy_prefixes(adapter)
    row = db.conn.execute(
        f"""
        SELECT metrics_json
        FROM policies
        WHERE accepted = 1
          AND ({' OR '.join('policy_id LIKE ?' for _ in prefixes)})
        ORDER BY score DESC, created_at DESC
        LIMIT 1
        """,
        tuple(prefixes),
    ).fetchone()
    if row is not None:
        return ScoreBreakdown(**json.loads(row["metrics_json"]))

    raw_metrics = BASELINE_RAW_METRICS if adapter.task_family == "locomotion" else BASELINE_MANIPULATION_RAW_METRICS
    baseline = adapter.score(raw_metrics)
    db.insert_policy(
        policy_id=baseline.policy_id,
        parent_policy_id=None,
        checkpoint_path=f"artifacts/{baseline.policy_id}/checkpoint.pt",
        metrics=baseline,
        accepted=True,
    )
    db.insert_failure_report(f"{adapter.task_family}_baseline", baseline.policy_id, adapter.diagnose([], raw_metrics))
    return baseline


def _policy_prefixes(adapter: TaskAdapter) -> list[str]:
    if adapter.task_family == "locomotion":
        return ["baseline_0000", "autoresearch_h1%", "policy_%"]
    return [f"baseline_{adapter.task_family}%", f"{adapter.task_family}_%"]


def _planner_context(adapter: TaskAdapter, best: ScoreBreakdown, history: ExperimentHistory) -> dict[str, Any]:
    return {
        "task_spec": adapter.default_task_spec().to_dict(),
        "best_policy": best.to_dict(),
        "recent_experiments": history.recent_experiments,
        "scenario_matrix": history.scenario_matrix,
        "failure_reports": history.failure_reports,
        "mode": "quick_modal_iteration" if adapter.task_family == "locomotion" else "adapter_research_planning",
        "constraints": {
            "one_patch_only": True,
            "locked_evaluator": True,
            "locked_modal_runner": adapter.task_family == "locomotion",
            "max_iterations": 10,
            "primary_goal": (
                "surface runner/training failures quickly before scaling compute"
                if adapter.task_family == "locomotion"
                else "prepare bounded manipulation scenarios and training configs before runner integration"
            ),
        },
    }


def _propose_patch(context: dict[str, Any], *, use_openai: bool) -> PatchSpec:
    if use_openai:
        return propose_patch_with_openai(context, use_fallback=True)
    return propose_patch(context)


def _build_experiment_spec(
    config: OrchestrationConfig,
    *,
    adapter: TaskAdapter,
    experiment_id: str,
    seed: int,
    best: ScoreBreakdown,
    patch: PatchSpec,
    config_changes: dict[str, dict[str, Any]],
    scenarios: list[ScenarioSpec],
) -> dict[str, Any]:
    if adapter.task_family != "locomotion":
        return _build_adapter_research_spec(
            config,
            adapter=adapter,
            experiment_id=experiment_id,
            seed=seed,
            best=best,
            patch=patch,
            config_changes=config_changes,
            scenarios=scenarios,
        )

    return _build_locomotion_modal_spec(
        config,
        experiment_id=experiment_id,
        seed=seed,
        best=best,
        patch=patch,
        config_changes=config_changes,
        scenarios=scenarios,
    )


def _build_locomotion_modal_spec(
    config: OrchestrationConfig,
    *,
    experiment_id: str,
    seed: int,
    best: ScoreBreakdown,
    patch: PatchSpec,
    config_changes: dict[str, dict[str, Any]],
    scenarios: list[ScenarioSpec],
) -> dict[str, Any]:
    overrides = SimpleNamespace(
        task="",
        runner="",
        device="",
        num_envs=config.num_envs,
        max_iterations=config.max_iterations,
        seed=seed,
        video_length=config.video_length,
        style_context="",
        motion_context="",
    )
    phase1_config = config.phase1_config
    if not phase1_config.is_absolute():
        phase1_config = config.repo_root / phase1_config
    spec = build_phase1_spec(phase1_config, experiment_id, overrides)
    _apply_patch_to_modal_spec(spec, patch)
    spec["autoresearch"] = {
        "parent_policy_id": best.policy_id,
        "score_before": best.total_score,
        "patch": patch.to_dict(),
        "config_changes": config_changes,
        "generated_scenarios": [scenario.to_dict() for scenario in scenarios],
        "created_at": datetime.now(UTC).isoformat(),
        "controller": "core.orchestration",
        "quick_iteration": {
            "num_envs": config.num_envs,
            "max_iterations": config.max_iterations,
            "seed": seed,
            "video_length": config.video_length,
        },
    }
    return spec


def _build_adapter_research_spec(
    config: OrchestrationConfig,
    *,
    adapter: TaskAdapter,
    experiment_id: str,
    seed: int,
    best: ScoreBreakdown,
    patch: PatchSpec,
    config_changes: dict[str, dict[str, Any]],
    scenarios: list[ScenarioSpec],
) -> dict[str, Any]:
    if adapter.task_family == "manipulation":
        phase1_config = config.phase1_config
        if phase1_config == Path("configs/locomotion/phase1_h1.yaml"):
            phase1_config = Path("configs/manipulation/phase1_h1_tabletop.yaml")
        if not phase1_config.is_absolute():
            phase1_config = config.repo_root / phase1_config
        overrides = SimpleNamespace(
            task=adapter.default_task_spec().base_env,
            runner="rsl_rl",
            device="",
            num_envs=config.num_envs,
            max_iterations=config.max_iterations,
            seed=seed,
            video_length=config.video_length,
            style_context="",
            motion_context="",
        )
        spec = build_phase1_spec(phase1_config, experiment_id, overrides)
        _apply_patch_to_modal_spec(spec, patch)
        spec.setdefault("train", {})["use_patched_runner"] = True
        spec["task_family"] = adapter.task_family
        spec["task_spec"] = adapter.default_task_spec().to_dict()
        spec["autoresearch"] = {
            "parent_policy_id": best.policy_id,
            "score_before": best.total_score,
            "patch": patch.to_dict(),
            "config_changes": config_changes,
            "generated_scenarios": [scenario.to_dict() for scenario in scenarios],
            "created_at": datetime.now(UTC).isoformat(),
            "controller": "core.orchestration",
            "quick_iteration": {
                "num_envs": config.num_envs,
                "max_iterations": config.max_iterations,
                "seed": seed,
                "video_length": config.video_length,
            },
            "training_surface": {
                "robot": "unitree_h1",
                "scene": "tabletop_transfer",
                "object_task": "move target cube from left side of table to right-side goal region",
                "runner": "modal.phase1_baseline_job",
                "custom_env": "modal_runner/isaac_scripts/robogenesis_tasks/h1_tabletop_transfer/h1_tabletop_transfer_env.py",
            },
        }
        return spec

    return {
        "experiment_id": experiment_id,
        "task_family": adapter.task_family,
        "task_spec": adapter.default_task_spec().to_dict(),
        "runner": {
            "status": "not_configured",
            "reason": "No deployed training runner is wired for this adapter yet.",
        },
        "autoresearch": {
            "parent_policy_id": best.policy_id,
            "score_before": best.total_score,
            "patch": patch.to_dict(),
            "config_changes": config_changes,
            "generated_scenarios": [scenario.to_dict() for scenario in scenarios],
            "created_at": datetime.now(UTC).isoformat(),
            "controller": "core.orchestration",
            "seed": seed,
            "next_runner_work": [
                "map ScenarioSpec objects into an Isaac Lab scene builder",
                "connect manipulation task config to Modal train/evaluate functions",
                "lock manipulation evaluator before accepting policies",
            ],
        },
    }


def _apply_patch_to_modal_spec(spec: dict[str, Any], patch: PatchSpec) -> None:
    """Apply only runner-relevant patch values to the Modal spec.

    Reward, curriculum, terrain, actuator, and domain-randomization changes are
    preserved in ``spec["autoresearch"]``. The phase-1 runner currently exposes
    quick training controls directly, so PPO iteration/env-count patches are the
    only values safely projected into the locked Modal command surface.
    """

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


def _experiment_id(prefix: str, *, index: int, seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix).strip("_")
    return f"{safe_prefix}_{stamp}_iter-{index + 1:02d}_seed-{seed}"


def _orchestration_trace_input(config: OrchestrationConfig) -> dict[str, Any]:
    return {
        "repo_root": str(config.repo_root),
        "db_path": str(config.db_path),
        "task_family": config.task_family,
        "phase1_config": str(config.phase1_config),
        "output_dir": str(config.output_dir),
        "experiment_prefix": config.experiment_prefix,
        "experiments": config.experiments,
        "seed_start": config.seed_start,
        "num_envs": config.num_envs,
        "max_iterations": config.max_iterations,
        "video_length": config.video_length,
        "use_openai": config.use_openai,
        "submit": config.submit,
        "app_name": config.app_name,
        "environment_name": config.environment_name,
    }

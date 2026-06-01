"""AutoResearch orchestration for quick Modal-backed H1 locomotion iterations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from adapters.base import ExperimentHistory, TaskAdapter
from adapters.locomotion import LocomotionAdapter
from agents.openai_planner import propose_patch_with_openai
from agents.planner import propose_patch
from agents.scenario_agent import generate_scenarios
from core.autoresearch_loop import BASELINE_RAW_METRICS
from core.experiment_db import ExperimentDB
from core.patch_validator import apply_yaml_patch, validate_patch_spec
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


def run_orchestration(config: OrchestrationConfig) -> list[OrchestrationStep]:
    """Prepare or submit bounded H1 locomotion experiments.

    The controller produces a single structured config patch per experiment,
    writes a Modal phase-1 spec, and records the proposed or running job in
    SQLite. Artifact sync later scores the completed Modal output and applies
    the locked accept/reject rule.
    """

    if config.task_family != "locomotion":
        raise ValueError("Only the locomotion task family is enabled in this cleanup branch.")

    repo_root = config.repo_root.resolve()
    output_dir = (repo_root / config.output_dir).resolve() if not config.output_dir.is_absolute() else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = LocomotionAdapter()
    db = ExperimentDB(config.db_path)
    try:
        best = _ensure_best_policy(db, adapter)
        steps: list[OrchestrationStep] = []

        for index in range(max(1, config.experiments)):
            seed = config.seed_start + index
            experiment_id = _experiment_id(config.experiment_prefix, index=index, seed=seed)
            history = ExperimentHistory(
                recent_experiments=db.recent_experiments(10),
                scenario_matrix=db.scenario_matrix(),
                failure_reports=db.recent_failures(5),
            )
            context = _planner_context(adapter, best, history, config)
            patch = _propose_patch(context, use_openai=config.use_openai)
            validate_patch_spec(patch).raise_for_errors()
            config_changes = apply_yaml_patch(patch, repo_root=repo_root, dry_run=True)

            scenarios = generate_scenarios(adapter, history)
            db.insert_scenarios(scenarios)

            modal_spec = _build_locomotion_modal_spec(
                config,
                experiment_id=experiment_id,
                seed=seed,
                best=best,
                patch=patch,
                config_changes=config_changes,
                scenarios=scenarios,
            )
            spec_path = output_dir / f"{experiment_id}.json"
            spec_path.write_text(json.dumps(modal_spec, indent=2, sort_keys=True) + "\n")

            call_id = None
            status = "proposed"
            if config.submit:
                call_id = submit_phase1_specs_to_deployed(
                    [modal_spec],
                    app_name=config.app_name,
                    environment_name=config.environment_name,
                )[0]
                status = "running"

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

        return steps
    finally:
        db.close()


def _ensure_best_policy(db: ExperimentDB, adapter: TaskAdapter) -> ScoreBreakdown:
    prefixes = ["baseline_0000", "autoresearch_h1%", "policy_%"]
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

    baseline = adapter.score(BASELINE_RAW_METRICS)
    db.insert_policy(
        policy_id=baseline.policy_id,
        parent_policy_id=None,
        checkpoint_path=f"artifacts/{baseline.policy_id}/checkpoint.pt",
        metrics=baseline,
        accepted=True,
    )
    db.insert_failure_report("locomotion_baseline", baseline.policy_id, adapter.diagnose([], BASELINE_RAW_METRICS))
    return baseline


def _planner_context(
    adapter: TaskAdapter,
    best: ScoreBreakdown,
    history: ExperimentHistory,
    config: OrchestrationConfig,
) -> dict[str, Any]:
    return {
        "task_spec": adapter.default_task_spec().to_dict(),
        "best_policy": best.to_dict(),
        "recent_experiments": history.recent_experiments,
        "scenario_matrix": history.scenario_matrix,
        "failure_reports": history.failure_reports,
        "mode": "quick_modal_iteration",
        "constraints": {
            "one_patch_only": True,
            "locked_evaluator": True,
            "locked_modal_runner": True,
            "num_envs": config.num_envs,
            "max_iterations": config.max_iterations,
            "primary_goal": "surface runner/training failures quickly before scaling compute",
        },
    }


def _propose_patch(context: dict[str, Any], *, use_openai: bool) -> PatchSpec:
    if use_openai:
        return propose_patch_with_openai(context, use_fallback=True)
    return propose_patch(context)


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


def _apply_patch_to_modal_spec(spec: dict[str, Any], patch: PatchSpec) -> None:
    """Apply only runner-relevant patch values to the Modal command surface."""

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

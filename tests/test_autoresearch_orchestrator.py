import json
import sqlite3
from pathlib import Path

from core.experiment_db import ExperimentDB
from core.orchestration import OrchestrationConfig, reconcile_modal_experiments, run_orchestration
from core.schemas import PatchSpec


def test_orchestrator_prepares_quick_modal_spec(tmp_path) -> None:
    output_dir = tmp_path / "specs"
    db_path = tmp_path / "research.db"

    steps = run_orchestration(
        OrchestrationConfig(
            repo_root=Path.cwd(),
            db_path=db_path,
            output_dir=output_dir,
            experiment_prefix="quick_loop",
            seed_start=123,
            num_envs=64,
            max_iterations=10,
            video_length=24,
            submit=False,
        )
    )

    assert len(steps) == 1
    step = steps[0]
    assert step.status == "proposed"
    assert step.modal_call_id is None
    assert step.parent_policy_id == "baseline_0000"
    assert step.scenario_ids

    spec = json.loads(Path(step.modal_spec_path).read_text())
    assert spec["experiment_id"] == step.experiment_id
    assert spec["train"]["num_envs"] == 64
    assert spec["train"]["max_iterations"] == 10
    assert spec["train"]["seed"] == 123
    assert spec["render"]["video_length"] == 24
    assert spec["autoresearch"]["patch"]["experiment_name"] == step.patch["experiment_name"]
    assert spec["autoresearch"]["quick_iteration"]["max_iterations"] == 10

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, modal_job_id FROM experiments WHERE id = ?", (step.experiment_id,)).fetchone()
    assert row == ("proposed", None)


def test_orchestrator_records_deployed_submission(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_submit(specs, *, app_name, environment_name=None):
        calls.append({"specs": specs, "app_name": app_name, "environment_name": environment_name})
        return ["fc-test-123"]

    monkeypatch.setattr("core.orchestration.submit_phase1_specs_to_deployed", fake_submit)
    steps = run_orchestration(
        OrchestrationConfig(
            repo_root=Path.cwd(),
            db_path=tmp_path / "research.db",
            output_dir=tmp_path / "specs",
            experiment_prefix="submit_loop",
            submit=True,
            app_name="robogenesis-test",
            environment_name="main",
        )
    )

    assert steps[0].status == "running"
    assert steps[0].modal_call_id == "fc-test-123"
    assert calls[0]["app_name"] == "robogenesis-test"
    assert calls[0]["environment_name"] == "main"

    with sqlite3.connect(tmp_path / "research.db") as conn:
        row = conn.execute("SELECT status, modal_job_id FROM experiments WHERE id = ?", (steps[0].experiment_id,)).fetchone()
    assert row == ("running", "fc-test-123")


def test_orchestrator_prepares_manipulation_adapter_spec(tmp_path) -> None:
    steps = run_orchestration(
        OrchestrationConfig(
            repo_root=Path.cwd(),
            db_path=tmp_path / "manipulation.db",
            output_dir=tmp_path / "specs",
            task_family="manipulation",
            experiment_prefix="manipulation_loop",
            submit=False,
        )
    )

    spec = json.loads(Path(steps[0].modal_spec_path).read_text())

    assert steps[0].parent_policy_id == "baseline_manipulation_0000"
    assert spec["task_family"] == "manipulation"
    assert spec["task"] == "RoboGenesis-H1-Tabletop-Manipulation-v0"
    assert spec["runner"] == "rsl_rl"
    assert spec["train"]["max_iterations"] == 10
    assert spec["train"]["use_patched_runner"] is True
    assert spec["autoresearch"]["training_surface"]["scene"] == "tabletop_transfer"
    assert spec["autoresearch"]["generated_scenarios"][0]["objects"]


def test_reconcile_updates_completed_modal_experiment(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "research.db"
    db = ExperimentDB(db_path)
    patch = PatchSpec(
        experiment_name="quick_patch",
        hypothesis="test",
        allowed_files=["configs/locomotion/rewards.yaml"],
        patch={"reward_weights.command_tracking": 1.1},
        expected_effect="test",
        risk="test",
        rollback="test",
    )
    db.insert_experiment("exp_ready", "baseline_0000", patch, status="running", modal_job_id="fc-test")
    db.close()

    class ReadyStatus:
        experiment_id = "exp_ready"
        ready_for_review = True
        review_blockers = []
        checkpoint_count = 1
        video_count = 1
        primary_video_path = "rollout_telemetry.mp4"
        has_render_error = False
        has_eval_metrics = True

    monkeypatch.setattr("tools.modal_artifact_status.summarize_experiments", lambda volume, experiment_ids: [ReadyStatus()])

    results = reconcile_modal_experiments(db_path, experiment_ids=["exp_ready"])

    assert results[0].db_status == "completed"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status FROM experiments WHERE id = 'exp_ready'").fetchone()
    assert row == ("completed",)

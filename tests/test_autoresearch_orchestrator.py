import json
import sqlite3
from pathlib import Path

from core.orchestration import OrchestrationConfig, run_orchestration


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


def test_orchestrator_rejects_non_locomotion_task_family(tmp_path) -> None:
    try:
        run_orchestration(
            OrchestrationConfig(
                repo_root=Path.cwd(),
                db_path=tmp_path / "research.db",
                output_dir=tmp_path / "specs",
                task_family="navigation",
            )
        )
    except ValueError as exc:
        assert "Only the locomotion task family" in str(exc)
    else:
        raise AssertionError("expected non-locomotion orchestration to fail")

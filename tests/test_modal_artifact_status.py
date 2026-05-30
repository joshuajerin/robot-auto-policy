from types import SimpleNamespace

from tools.modal_artifact_status import ExperimentArtifactStatus, list_experiment_files


def test_experiment_artifact_status_flags_outputs() -> None:
    status = ExperimentArtifactStatus(
        experiment_id="quick",
        files=[
            "raw_eval_metrics.json",
            "eval_metrics.json",
            "artifact_manifest.json",
            "rollout_trace.json",
            "logs/rsl_rl/run/model_9.pt",
            "logs/rsl_rl/run/videos/rollout.mp4",
        ],
    )

    assert status.has_raw_metrics
    assert status.has_eval_metrics
    assert status.has_manifest
    assert status.has_rollout_trace
    assert not status.has_render_error
    assert status.video_count == 1
    assert status.video_paths == ["logs/rsl_rl/run/videos/rollout.mp4"]
    assert status.primary_video_path == "logs/rsl_rl/run/videos/rollout.mp4"
    assert status.checkpoint_count == 1
    assert status.ready_for_review
    assert status.review_blockers == []


def test_experiment_artifact_status_reports_missing_video() -> None:
    status = ExperimentArtifactStatus(
        experiment_id="quick",
        files=[
            "raw_eval_metrics.json",
            "eval_metrics.json",
            "artifact_manifest.json",
            "rollout_trace.json",
            "logs/rsl_rl/run/model_9.pt",
        ],
    )

    assert status.video_count == 0
    assert not status.ready_for_review
    assert "missing rollout video" in status.review_blockers


def test_list_experiment_files_treats_missing_modal_directory_as_empty(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="No such file or directory")

    monkeypatch.setattr("tools.modal_artifact_status.subprocess.run", fake_run)

    assert list_experiment_files("robogenesis-runs", "exp_not_started") == []

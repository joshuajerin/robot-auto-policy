from pathlib import Path

from core.artifact_sync import (
    build_modal_get_command,
    modal_experiment_local_path,
    modal_experiment_remote_path,
    sync_and_ingest_modal_experiment,
)


def test_modal_artifact_paths_are_sanitized() -> None:
    experiment_id = "quick loop/seed 1"

    assert modal_experiment_remote_path(experiment_id) == "/experiments/quick-loop-seed-1"
    assert modal_experiment_local_path("artifacts/modal_downloads", experiment_id) == Path(
        "artifacts/modal_downloads/quick-loop-seed-1"
    )


def test_build_modal_get_command_includes_volume_remote_and_destination() -> None:
    command = build_modal_get_command(
        "robogenesis-runs",
        "exp_001",
        "artifacts/modal_downloads",
        environment_name="main",
    )

    assert command == [
        "modal",
        "volume",
        "get",
        "--force",
        "--env",
        "main",
        "robogenesis-runs",
        "/experiments/exp_001",
        "artifacts/modal_downloads",
    ]


def test_sync_and_ingest_downloads_reviews_then_ingests(monkeypatch, tmp_path) -> None:
    calls: list[object] = []

    def fake_run(command, check):
        calls.append(("run", command, check))

    def fake_ingest(artifact_dir, *, db_path, parent_policy_id, accepted):
        calls.append(("ingest", artifact_dir, db_path, parent_policy_id, accepted))
        return {"experiment_id": "exp_001", "score": 0.42}

    monkeypatch.setattr("core.artifact_sync.subprocess.run", fake_run)
    monkeypatch.setattr("core.artifact_sync.ingest_artifact_dir", fake_ingest)
    monkeypatch.setattr(
        "core.artifact_sync._review_artifact_acceptance",
        lambda artifact_dir, db_path, parent_policy_id, parent_score, accepted_override: (
            False,
            ["score_delta_too_small:0.10<0.53"],
        ),
    )

    result = sync_and_ingest_modal_experiment(
        "exp_001",
        db_path=tmp_path / "research.db",
        destination_root=tmp_path / "downloads",
        parent_policy_id="baseline_0000",
    )

    assert result.downloaded is True
    assert result.ingested is True
    assert result.accepted is False
    assert result.review_reasons == ["score_delta_too_small:0.10<0.53"]
    assert result.artifact_dir == str(tmp_path / "downloads" / "exp_001")
    assert calls[0] == (
        "run",
        [
            "modal",
            "volume",
            "get",
            "--force",
            "robogenesis-runs",
            "/experiments/exp_001",
            str(tmp_path / "downloads"),
        ],
        True,
    )
    assert calls[1] == (
        "ingest",
        tmp_path / "downloads" / "exp_001",
        tmp_path / "research.db",
        "baseline_0000",
        False,
    )
    assert result.ingest_summary == {"experiment_id": "exp_001", "score": 0.42}


def test_sync_accept_override_bypasses_locked_review(monkeypatch, tmp_path) -> None:
    review_calls = 0

    def fake_review(artifact_dir, *, db_path, parent_policy_id, parent_score, accepted_override):
        nonlocal review_calls
        review_calls += 1
        assert accepted_override is True
        return True, ["manual_accept_override"]

    monkeypatch.setattr("core.artifact_sync.subprocess.run", lambda command, check: None)
    monkeypatch.setattr("core.artifact_sync.ingest_artifact_dir", lambda *args, **kwargs: {"score": 0.9})
    monkeypatch.setattr("core.artifact_sync._review_artifact_acceptance", fake_review)

    result = sync_and_ingest_modal_experiment(
        "exp_001",
        db_path=tmp_path / "research.db",
        destination_root=tmp_path / "downloads",
        accepted=True,
    )

    assert result.accepted is True
    assert result.review_reasons == ["manual_accept_override"]
    assert review_calls == 1

import json

import core.raindrop_trace as raindrop_trace


def test_artifact_video_page_url_is_repo_relative(tmp_path) -> None:
    video = tmp_path / "artifacts" / "modal_downloads" / "exp_001" / "rollout.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    url = raindrop_trace.artifact_video_page_url(video, repo_root=tmp_path)

    assert url == "http://127.0.0.1:61020/artifact-video?path=artifacts/modal_downloads/exp_001/rollout.mp4"


def test_publish_artifact_run_replays_modal_task_manifest(monkeypatch, tmp_path) -> None:
    posts: list[tuple[str, dict]] = []

    monkeypatch.setattr(raindrop_trace, "workshop_available", lambda local_workshop_url=None: True)
    monkeypatch.setattr(raindrop_trace, "init_raindrop_sdk", lambda local_workshop_url=None: None)
    monkeypatch.setattr(
        raindrop_trace,
        "post_workshop_json",
        lambda local_workshop_url, path, payload: posts.append((path, payload)) or True,
    )

    artifact_dir = tmp_path / "artifacts" / "modal_downloads" / "exp_001"
    artifact_dir.mkdir(parents=True)
    video = artifact_dir / "rollout_telemetry.mp4"
    video.write_bytes(b"video")
    (artifact_dir / "raindrop_trace.json").write_text(
        json.dumps(
            {
                "event_name": "robogenesis-sim-run",
                "event_id": "exp_001",
                "experiment_id": "exp_001",
                "video_paths": [str(video)],
                "tasks": [
                    {
                        "name": "train_policy",
                        "input": {"max_iterations": 2},
                        "output": {"return_code": 0},
                        "start_ms": 1000,
                        "end_ms": 1500,
                        "status": "ok",
                    },
                    {
                        "name": "render_telemetry_video",
                        "input": {"fps": 20},
                        "output": {"return_code": 0},
                        "start_ms": 1600,
                        "end_ms": 1900,
                        "status": "ok",
                    },
                ],
            }
        )
    )

    summary = raindrop_trace.publish_artifact_run(
        artifact_dir,
        ingest_summary={"experiment_id": "exp_001", "score": 0.42, "rollout_video_paths": [str(video)]},
        accepted=False,
        review_reasons=["score_delta_too_small"],
        db_path=tmp_path / "research.db",
        repo_root=tmp_path,
    )

    assert summary["enabled"] is True
    assert summary["local_enabled"] is True
    assert summary["event_id"] == "exp_001"
    assert summary["task_count"] == 3
    assert summary["video_pages"] == [
        "http://127.0.0.1:61020/artifact-video?path=artifacts/modal_downloads/exp_001/rollout_telemetry.mp4"
    ]

    span_names = [
        span["name"]
        for _, payload in posts
        for resource in payload["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]
    assert "train_policy" in span_names
    assert "render_telemetry_video" in span_names
    assert "sync_and_ingest_artifacts" in span_names
    assert "robogenesis-sim-run" in span_names

    render_spans = [
        span
        for _, payload in posts
        for resource in payload["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
        if span["name"] == "render_telemetry_video"
    ]
    render_attrs = {
        attr["key"]: attr["value"]
        for span in render_spans
        for attr in span["attributes"]
    }
    render_output = json.loads(render_attrs["traceloop.entity.output"]["stringValue"])
    assert render_output["render_videos"][0]["video_page_url"] == (
        "http://127.0.0.1:61020/artifact-video?path=artifacts/modal_downloads/exp_001/rollout_telemetry.mp4"
    )


def test_publish_artifact_run_localizes_nested_modal_video_paths(monkeypatch, tmp_path) -> None:
    posts: list[tuple[str, dict]] = []

    monkeypatch.setattr(raindrop_trace, "workshop_available", lambda local_workshop_url=None: True)
    monkeypatch.setattr(raindrop_trace, "init_raindrop_sdk", lambda local_workshop_url=None: None)
    monkeypatch.setattr(
        raindrop_trace,
        "post_workshop_json",
        lambda local_workshop_url, path, payload: posts.append((path, payload)) or True,
    )

    artifact_dir = tmp_path / "artifacts" / "modal_downloads" / "exp_nested"
    video = artifact_dir / "logs" / "rsl_rl" / "run" / "rollout.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    remote_video = "/runs/experiments/exp_nested/logs/rsl_rl/run/rollout.mp4"
    (artifact_dir / "raindrop_trace.json").write_text(
        json.dumps(
            {
                "event_name": "robogenesis-sim-run",
                "event_id": "exp_nested",
                "experiment_id": "exp_nested",
                "video_paths": [remote_video],
            }
        )
    )

    summary = raindrop_trace.publish_artifact_run(
        artifact_dir,
        ingest_summary={"experiment_id": "exp_nested"},
        repo_root=tmp_path,
    )

    assert summary["video_pages"] == [
        "http://127.0.0.1:61020/artifact-video?path=artifacts/modal_downloads/exp_nested/logs/rsl_rl/run/rollout.mp4"
    ]

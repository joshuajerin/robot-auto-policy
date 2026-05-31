import json

from tools.publish_raindrop_renders import discover_render_runs, publish_render_runs


def test_discover_render_runs_groups_nested_modal_and_lambda_videos(tmp_path) -> None:
    experiment_id = "autoresearch_h1_video_seed-1"
    artifact_root = tmp_path / "artifacts"
    outer = artifact_root / "multiview_lambda" / experiment_id
    marker_dir = outer / "modal_artifacts" / experiment_id
    marker_dir.mkdir(parents=True)
    (marker_dir / "raindrop_trace.json").write_text(
        json.dumps({"experiment_id": experiment_id, "video_paths": [f"/runs/experiments/{experiment_id}/rollout.mp4"]})
    )
    (marker_dir / "rollout.mp4").write_bytes(b"rollout")
    lambda_video = outer / "lambda_render" / "artifacts" / "multiview" / "front" / "front-policy-step-0.mp4"
    lambda_video.parent.mkdir(parents=True)
    lambda_video.write_bytes(b"front")

    runs = discover_render_runs(artifact_root)

    assert len(runs) == 1
    assert runs[0].experiment_id == experiment_id
    assert runs[0].artifact_dir == marker_dir
    assert runs[0].video_root == outer
    assert runs[0].video_paths == tuple(sorted([lambda_video.resolve(), (marker_dir / "rollout.mp4").resolve()]))


def test_discover_render_runs_includes_loose_video_folders(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    video = artifact_root / "lambda_renders" / "manip_lift_cube" / "rl-video-step-0.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    runs = discover_render_runs(artifact_root)

    assert len(runs) == 1
    assert runs[0].experiment_id == "manip_lift_cube"
    assert runs[0].artifact_dir == video.parent
    assert runs[0].video_paths == (video.resolve(),)


def test_publish_render_runs_suffixes_duplicate_experiment_ids(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    for root_name in ("modal_downloads", "lambda_renders"):
        video = artifact_root / root_name / "shared_exp" / "rollout.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")

    summaries = publish_render_runs(artifact_root=artifact_root, db_path=tmp_path / "research.db", dry_run=True)

    event_ids = {summary["event_id"] for summary in summaries}
    assert len(event_ids) == 2
    assert all(event_id.startswith("shared_exp--") for event_id in event_ids)

from pathlib import Path

from modal_runner.modal_app import _build_telemetry_render_cmd, _fallback_eval_metrics, _find_videos, _score_metrics


def test_fallback_eval_metrics_fail_safely() -> None:
    raw = _fallback_eval_metrics("phase1_quick", "eval did not write metrics")
    score = _score_metrics(raw)

    assert raw["evaluation_errors"] == ["eval did not write metrics"]
    assert score["safety_passed"] is False
    assert "evaluation emitted errors" in score["safety_reasons"]
    assert score["total_score"] == 0.0


def test_telemetry_render_command_does_not_use_isaac_cameras() -> None:
    cmd = _build_telemetry_render_cmd(
        trace_path=Path("/runs/experiments/quick/rollout_trace.json"),
        score_path=Path("/runs/experiments/quick/eval_metrics.json"),
        output_path=Path("/runs/experiments/quick/rollout_telemetry.mp4"),
        render_spec={"video_length": 60, "seed": 907, "fps": 20},
    )

    assert "--seed" not in cmd
    assert "--enable_cameras" not in cmd
    assert "play.py" not in " ".join(cmd)
    assert "render_telemetry_video.py" in " ".join(cmd)


def test_find_videos_lists_rendered_rollouts(tmp_path) -> None:
    first = tmp_path / "logs" / "rsl_rl" / "run" / "videos" / "rollout_a.mp4"
    second = tmp_path / "logs" / "rsl_rl" / "run" / "videos" / "rollout_b.mp4"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"video-a")
    second.write_bytes(b"video-b")

    assert _find_videos(tmp_path) == [str(first), str(second)]

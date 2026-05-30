import json

from modal_runner.isaac_scripts.render_telemetry_video import render_telemetry_video


def test_telemetry_render_writes_mp4(tmp_path) -> None:
    trace = {
        "policy_id": "quick",
        "frames": [
            {
                "step": 1,
                "reward": -0.1,
                "done": False,
                "velocity_command": [0.5, 0.0, 0.0],
                "base_lin_vel": [0.1, 0.0, 0.0],
                "joint_pos": [0.0] * 19,
                "actions": [0.0] * 19,
            }
        ],
    }
    output = tmp_path / "rollout_telemetry.mp4"

    render_telemetry_video(
        trace=trace,
        metrics={"policy_id": "quick", "total_score": 0.1, "survival_no_fall": 0.0},
        output=output,
        title="test",
        fps=1,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_telemetry_render_writes_fallback_mp4_for_empty_trace(tmp_path) -> None:
    output = tmp_path / "fallback.mp4"

    render_telemetry_video(
        trace={"policy_id": "quick", "frames": [], "render_note": "empty trace"},
        metrics={"policy_id": "quick", "total_score": 0.0, "survival_no_fall": 0.0},
        output=output,
        title="fallback",
        fps=10,
    )

    assert output.exists()
    assert output.stat().st_size > 0

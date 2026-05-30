from pathlib import Path

from modal_runner.modal_app import _build_render_cmd, _fallback_eval_metrics, _score_metrics


def test_fallback_eval_metrics_fail_safely() -> None:
    raw = _fallback_eval_metrics("phase1_quick", "eval did not write metrics")
    score = _score_metrics(raw)

    assert raw["evaluation_errors"] == ["eval did not write metrics"]
    assert score["safety_passed"] is False
    assert "evaluation emitted errors" in score["safety_reasons"]
    assert score["total_score"] == 0.0


def test_render_command_does_not_pass_unsupported_seed() -> None:
    cmd = _build_render_cmd(
        runner="rsl_rl",
        task="Isaac-Velocity-Flat-H1-v0",
        checkpoint_path=Path("/logs/rsl_rl/h1_flat/run/model_9.pt"),
        render_spec={"num_envs": 1, "video_length": 60, "seed": 907},
    )

    assert "--seed" not in cmd
    assert "--load_run" in cmd
    assert "run" in cmd

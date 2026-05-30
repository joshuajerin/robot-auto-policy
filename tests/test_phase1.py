from pathlib import Path

from modal_runner.phase1 import build_phase1_spec


class Args:
    task = ""
    runner = ""
    device = ""
    num_envs = 64
    max_iterations = 2
    seed = 123
    video_length = 60


def test_phase1_spec_uses_h1_task_and_overrides() -> None:
    spec = build_phase1_spec(Path("configs/locomotion/phase1_h1.yaml"), "baseline_h1_test", Args())

    assert spec["experiment_id"] == "baseline_h1_test"
    assert spec["task"] == "Isaac-Velocity-Flat-H1-v0"
    assert spec["robot_spec"] == "assets/h1_robot_spec.json"
    assert spec["train"]["num_envs"] == 64
    assert spec["train"]["max_iterations"] == 2
    assert spec["train"]["seed"] == 123
    assert spec["render"]["video_length"] == 60


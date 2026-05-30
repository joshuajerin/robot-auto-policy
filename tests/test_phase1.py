from pathlib import Path
import json

from modal_runner.phase1 import build_phase1_spec


class Args:
    task = ""
    runner = ""
    device = ""
    num_envs = 64
    max_iterations = 2
    seed = 123
    video_length = 60
    style_context = ""


def test_phase1_spec_uses_h1_task_and_overrides() -> None:
    spec = build_phase1_spec(Path("configs/locomotion/phase1_h1.yaml"), "baseline_h1_test", Args())

    assert spec["experiment_id"] == "baseline_h1_test"
    assert spec["task"] == "Isaac-Velocity-Flat-H1-v0"
    assert spec["robot_spec"] == "assets/h1_robot_spec.json"
    assert spec["train"]["num_envs"] == 64
    assert spec["train"]["max_iterations"] == 2
    assert spec["train"]["seed"] == 123
    assert spec["render"]["video_length"] == 60


def test_phase1_spec_embeds_style_context(tmp_path) -> None:
    context_path = tmp_path / "style_context.json"
    context_path.write_text(json.dumps({"style": "upright human walk", "cadence_hz": 1.75}))

    args = Args()
    args.style_context = str(context_path)
    spec = build_phase1_spec(Path("configs/locomotion/phase1_h1.yaml"), "baseline_h1_video", args)

    assert spec["style_context"]["style"] == "upright human walk"
    assert spec["style_context_path"] == str(context_path)

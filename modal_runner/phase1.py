"""Build and optionally launch a phase-1 H1 baseline experiment spec."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def build_phase1_spec(config_path: Path, experiment: str, overrides: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    spec = {
        "experiment_id": experiment,
        "task": overrides.task or config["task"],
        "runner": overrides.runner or config.get("runner", "rsl_rl"),
        "device": overrides.device or config.get("device", "cuda:0"),
        "robot_spec": config.get("robot_spec", "assets/h1_robot_spec.json"),
        "train": dict(config.get("train", {})),
        "eval": dict(config.get("eval", {})),
        "render": dict(config.get("render", {})),
    }
    if overrides.num_envs is not None:
        spec["train"]["num_envs"] = overrides.num_envs
    if overrides.max_iterations is not None:
        spec["train"]["max_iterations"] = overrides.max_iterations
    if overrides.seed is not None:
        spec["train"]["seed"] = overrides.seed
    if overrides.video_length is not None:
        spec["render"]["video_length"] = overrides.video_length
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/locomotion/phase1_h1.yaml")
    parser.add_argument("--experiment", default="baseline_h1_001")
    parser.add_argument("--task", default="")
    parser.add_argument("--runner", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--video-length", type=int)
    parser.add_argument("--launch-modal", action="store_true")
    args = parser.parse_args()

    spec = build_phase1_spec(Path(args.config), args.experiment, args)
    spec_json = json.dumps(spec, sort_keys=True)

    if args.launch_modal:
        subprocess.run(
            [
                "modal",
                "run",
                "modal_runner/modal_app.py",
                "--action",
                "phase1",
                "--experiment-spec-json",
                spec_json,
            ],
            check=True,
        )
        return

    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


"""Build and optionally launch a phase-1 H1 baseline experiment spec."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modal_runner.deployed import DEFAULT_APP_NAME, submit_phase1_specs_to_deployed


def build_phase1_spec(config_path: Path, experiment: str, overrides: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    spec = {
        "experiment_id": experiment,
        "task": getattr(overrides, "task", "") or config["task"],
        "runner": getattr(overrides, "runner", "") or config.get("runner", "rsl_rl"),
        "device": getattr(overrides, "device", "") or config.get("device", "cuda:0"),
        "robot_spec": config.get("robot_spec", "assets/h1_robot_spec.json"),
        "train": dict(config.get("train", {})),
        "eval": dict(config.get("eval", {})),
        "render": dict(config.get("render", {})),
    }
    style_context = getattr(overrides, "style_context", "")
    if style_context:
        spec["style_context"] = json.loads(Path(style_context).read_text())
        spec["style_context_path"] = style_context
    if getattr(overrides, "num_envs", None) is not None:
        spec["train"]["num_envs"] = overrides.num_envs
    if getattr(overrides, "max_iterations", None) is not None:
        spec["train"]["max_iterations"] = overrides.max_iterations
    if getattr(overrides, "seed", None) is not None:
        spec["train"]["seed"] = overrides.seed
    if getattr(overrides, "video_length", None) is not None:
        spec["render"]["video_length"] = overrides.video_length
    return spec


def build_batch_specs(config_path: Path, experiment: str, overrides: argparse.Namespace) -> list[dict[str, Any]]:
    count = max(1, int(getattr(overrides, "num_runs", 1)))
    seed_start = int(getattr(overrides, "seed_start", 42))
    specs: list[dict[str, Any]] = []
    for index in range(count):
        run_args = argparse.Namespace(**vars(overrides))
        run_args.seed = seed_start + index
        run_experiment = f"{experiment}-seed-{run_args.seed}" if count > 1 else experiment
        specs.append(build_phase1_spec(config_path, run_experiment, run_args))
    return specs


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
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--video-length", type=int)
    parser.add_argument("--style-context", default="")
    parser.add_argument("--launch-modal", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--use-deployed", action="store_true")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--write-spec", default="")
    args = parser.parse_args()

    specs = build_batch_specs(Path(args.config), args.experiment, args)
    spec_json = json.dumps(specs if args.num_runs > 1 else specs[0], sort_keys=True)

    if args.write_spec:
        Path(args.write_spec).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_spec).write_text(json.dumps(specs if args.num_runs > 1 else specs[0], indent=2, sort_keys=True) + "\n")

    if args.launch_modal:
        if args.use_deployed:
            call_ids = submit_phase1_specs_to_deployed(specs, app_name=args.app_name)
            print(json.dumps({"app_name": args.app_name, "function_call_ids": call_ids}, indent=2, sort_keys=True))
            return
        if args.detach:
            for spec in specs:
                subprocess.run(
                    [
                        "modal",
                        "run",
                        "--detach",
                        "modal_runner/modal_app.py::phase1_baseline_job",
                        "--experiment-spec-json",
                        json.dumps(spec, sort_keys=True),
                    ],
                    check=True,
                )
        else:
            subprocess.run(
                [
                    "modal",
                    "run",
                    "modal_runner/modal_app.py",
                    "--action",
                    "phase1-batch-detach" if args.num_runs > 1 else "phase1",
                    "--experiment-spec-json",
                    spec_json,
                ],
                check=True,
            )
        return

    print(json.dumps(specs if args.num_runs > 1 else specs[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

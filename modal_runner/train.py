"""Build an experiment spec for Modal Isaac Lab training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--config", default="configs/locomotion/phase1_h1.yaml")
    parser.add_argument("--runner", default="rsl_rl")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    train_config = dict(config.get("train", {}))
    spec = {
        "experiment_id": args.experiment,
        "task": args.task or config.get("task", "Isaac-Velocity-Flat-H1-v0"),
        "runner": args.runner or config.get("runner", "rsl_rl"),
        "num_envs": args.num_envs or train_config.get("num_envs", 4096),
        "max_iterations": args.max_iterations or train_config.get("max_iterations", 1000),
        "seed": args.seed if args.seed is not None else train_config.get("seed"),
    }
    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

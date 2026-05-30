"""Build an experiment spec for Modal Isaac Lab training."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--runner", default="rsl_rl")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    spec = {
        "experiment_id": args.experiment,
        "task": args.task,
        "runner": args.runner,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
    }
    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


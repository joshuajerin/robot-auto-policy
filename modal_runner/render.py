"""Render command spec helper for policy rollout videos."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", default="model_999.pt")
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--video-length", type=int, default=240)
    args = parser.parse_args()

    print(
        json.dumps(
            {
                "experiment_id": args.experiment,
                "checkpoint": args.checkpoint,
                "task": args.task,
                "video_length": args.video_length,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


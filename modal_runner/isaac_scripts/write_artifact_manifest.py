"""Write a phase-1 artifact manifest after Modal sync."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--metrics", default="")
    parser.add_argument("--score", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--h1-asset-report", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifact_root)
    manifest = {
        "experiment_id": args.experiment_id,
        "task": args.task,
        "artifact_root": str(root),
        "checkpoint_path": args.checkpoint or None,
        "raw_metrics_path": args.metrics or None,
        "score_path": args.score or None,
        "rollout_video_path": args.video or None,
        "h1_asset_report_path": args.h1_asset_report or None,
        "files": [str(path.relative_to(root)) for path in sorted(root.rglob("*")) if path.is_file()],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


"""Prepare a user MP4 as a humanoid motion-imitation skill context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.video_motion_imitation import prepare_user_video_motion_skill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="artifacts/user_motion_skill")
    parser.add_argument("--robot-spec", default="assets/h1_robot_spec.json")
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--extract-frames", action="store_true")
    args = parser.parse_args()

    result = prepare_user_video_motion_skill(
        video_path=args.video,
        output_dir=args.output_dir,
        robot_spec_path=args.robot_spec,
        sample_fps=args.sample_fps,
        extract_frames=args.extract_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

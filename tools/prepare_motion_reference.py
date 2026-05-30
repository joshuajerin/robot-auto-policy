"""Prepare an open research mocap reference for locomotion training context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.motion_reference import DEFAULT_REFERENCE_CONFIG, prepare_motion_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_REFERENCE_CONFIG))
    parser.add_argument("--motion-id", default="07_01")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result = prepare_motion_reference(
        config_path=args.config,
        motion_id=args.motion_id,
        output_dir=args.output_dir,
        refresh=args.refresh,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "context"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

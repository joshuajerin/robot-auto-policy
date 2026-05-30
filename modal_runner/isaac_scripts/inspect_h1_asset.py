"""Inspect the Unitree H1 asset inside an Isaac Lab container.

This script is executed by Modal inside the Isaac Lab image. It avoids importing
project code so it can run even before the RoboGenesis package is installed in
the container.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


H1_PATTERNS = (
    "*H1*.usd",
    "*h1*.usd",
    "*Unitree*H1*.usd",
    "*unitree*h1*.usd",
)


def find_h1_usd(search_roots: list[Path]) -> list[str]:
    matches: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in H1_PATTERNS:
            for path in root.rglob(pattern):
                lower = str(path).lower()
                if "h1" in lower and path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                    matches.add(str(path))
    return sorted(matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    args = parser.parse_args()

    roots = [
        Path("/workspace"),
        Path("/root"),
        Path("/isaac-lab"),
        Path("/opt"),
        Path(os.environ.get("ISAACLAB_PATH", "/missing")),
    ]
    matches = find_h1_usd(roots)
    report = {
        "robot_id": "unitree_h1",
        "task": args.task,
        "usd_candidates": matches,
        "resolved_asset_path": matches[0] if matches else None,
        "asset_resolution_note": (
            "If no local USD is found, Isaac Lab may resolve H1 through its packaged "
            "asset registry or Omniverse/Nucleus path when the task is constructed."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


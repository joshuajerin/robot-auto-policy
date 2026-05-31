"""Register RoboGenesis H1 manipulation tasks, then run Isaac Lab rsl_rl play."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import robogenesis_tasks  # noqa: F401


def main() -> None:
    play_path = Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py")
    if not play_path.exists():
        play_path = Path("/workspace/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py")
    sys.path.insert(0, str(play_path.parent))
    runpy.run_path(str(play_path), run_name="__main__")


if __name__ == "__main__":
    main()


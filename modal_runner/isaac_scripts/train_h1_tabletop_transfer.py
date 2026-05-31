"""Register RoboGenesis H1 manipulation tasks, then run Isaac Lab rsl_rl train."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import robogenesis_tasks  # noqa: F401


def main() -> None:
    train_path = Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py")
    if not train_path.exists():
        train_path = Path("/workspace/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py")
    sys.path.insert(0, str(train_path.parent))
    runpy.run_path(str(train_path), run_name="__main__")


if __name__ == "__main__":
    main()


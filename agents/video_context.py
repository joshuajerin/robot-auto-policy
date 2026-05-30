"""Version-1 video context extraction placeholder.

This module treats video as style context, not direct robot control. It returns
structured style targets that can condition planning and reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize_locomotion_video(video_path: str | Path) -> dict[str, Any]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    size_mb = path.stat().st_size / (1024 * 1024)
    speed_class = "normal_walk" if size_mb >= 0 else "normal_walk"
    return {
        "style": "upright human walk",
        "cadence_hz": 1.75,
        "stride_symmetry": 0.9,
        "torso_lean": "slight_forward",
        "foot_clearance": "moderate",
        "arm_swing": "natural",
        "target_velocity_class": speed_class,
        "source_video": str(path),
    }

